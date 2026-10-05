"""The swarm harness: fan out, judge, converge.

:class:`Swarm` is deliberately dumb. It spawns workers, fans tasks out,
collects results, hands them to a reviewer (``swarm.review``), charges the
treasury, benches stalled slots, and loops until tasks converge, the budget
runs out, or the watchdog calls a flatline. All the smarts live in the
reviewer; the harness just runs the loop.
"""

from __future__ import annotations

import concurrent.futures as futures
import queue
import threading
from collections import deque
from itertools import islice

from .backends import _execute_one, _worker_name, _require_positive_timeout, run_batch
from .identity import WorkerIdentity
from .ledger import Treasury
from .memory import Memory, WorkerMemory
from .report import SwarmReport
from .review import review as _default_review_fn
from .watchdog import StallWatcher, SwarmWatchdog
from .worker import Context, Result, Task, WorkerFn


class Swarm:
    """Fan-out / review / converge loop over a pool of worker slots."""

    def __init__(
        self,
        worker_fn: WorkerFn,
        n: int = 50,
        budget: float = 10.0,
        task_cost: float = 0.02,
        rubric: str = "",
        max_rounds: int = 5,
        timeout: float = 30.0,
        pass_threshold: float = 0.85,
        backend: str = "thread",
        strategy: str = "shard",
        stall_ticks: int = 20,
        max_tasks: int = 10000,
        memory_path: str = "swarm_memory.md",
        reviewer_backend: str = "heuristic",
        _memory=None,
        _treasury=None,
        _review_fn=None,
        _swarm_watchdog=None,
    ):
        self.worker_fn = worker_fn
        self.n = n
        self.budget = budget
        self.task_cost = task_cost
        self.rubric = rubric
        self.max_rounds = max_rounds
        # Audit L3: a non-positive/None timeout would hang the swarm forever
        # on a hung worker. Fail fast instead.
        self.timeout = _require_positive_timeout(timeout)
        self.pass_threshold = pass_threshold
        self.backend = backend
        self.strategy = strategy
        self.stall_ticks = stall_ticks
        self.max_tasks = max_tasks
        self.memory_path = memory_path
        self.reviewer_backend = reviewer_backend

        # Injection points for tests; default path builds the real collaborators.
        self.memory = _memory if _memory is not None else Memory(path=memory_path)
        self.treasury = _treasury if _treasury is not None else Treasury(budget=budget)
        self._review_fn = _review_fn if _review_fn is not None else _default_review_fn
        self._swarm_watchdog = (
            _swarm_watchdog if _swarm_watchdog is not None else SwarmWatchdog()
        )

        self._slot_watchers = {i: StallWatcher(stall_ticks) for i in range(n)}
        self._benched: set[int] = set()
        # Consecutive non-pass verdicts per slot: bench after 3 (spec).
        # A pass resets the counter. Benched slots are excluded from
        # future fan-outs; their tasks redistribute over the rest.
        self._consec_fails: dict[int, int] = {}
        # Rolling pass/fail window per slot (audit M4): catches workers that
        # evade the consecutive rule with intermittent passes but still fail
        # most of the time. Bench when a full 10-verdict window drops under
        # a 40% pass rate. (Any threshold below ~1/3 would be redundant with
        # the consecutive-3 rule by pigeonhole; 40% catches the
        # fail,fail,pass cycler.)
        self._recent: dict[int, deque[bool]] = {}
        self.report = SwarmReport()

    # ------------------------------------------------------------------ ctx

    def _ctx(self, idx: int, round_no: int) -> Context:
        identity = WorkerIdentity(idx)
        # Workers see a WorkerMemory proxy: their writes are stamped with
        # their own handle (no forgeable writer field) and they cannot vote
        # on reputations (audit H3).
        return Context(
            memory=WorkerMemory(self.memory, identity.handle),
            treasury=self.treasury,
            identity=identity,
            round=round_no,
            worker_index=idx,
        )

    # ---------------------------------------------------------------- bench

    def bench_slot(self, idx: int) -> None:
        """Bench a worker slot: it is excluded from future fan-outs."""
        self._benched.add(idx)

    @property
    def benched(self) -> frozenset[int]:
        """Indexes of benched worker slots."""
        return frozenset(self._benched)

    # --------------------------------------------------------------- fan_out

    def fan_out(
        self, tasks: list[Task], round_no: int = 0, strategy: str = "shard"
    ) -> list[Result]:
        """Run ``tasks`` across the un-benched slots. Results in a defined order.

        Strategies:

        - ``"shard"`` — round-robin tasks over slots. ``len(tasks)``
          executions; ``results[i]`` corresponds to ``tasks[i]``.
        - ``"scatter"`` — every task to every slot. **Combinatorial
          warning:** this is ``len(tasks) * len(slots)`` executions, so it
          explodes fast. Results are grouped task-major:
          ``[(t0,s0), (t0,s1), ..., (t1,s0), ...]``.
        - ``"worksteal"`` — thread backend only. Tasks go on a shared
          :class:`queue.Queue`; one puller thread per slot steals work until
          the queue is empty. Each puller builds its own ctx and runs
          :func:`~swarm.backends._execute_one` with a per-task timeout via a
          nested single-worker executor. Results are restored to task order.
          On the process backend, worksteal falls back to ``"shard"`` (a
          shared in-process queue cannot feed child processes).

        Unknown strategies raise :class:`ValueError`. If every slot is
        benched, raises :class:`RuntimeError`.
        """
        slots = [i for i in range(self.n) if i not in self._benched]
        if not slots:
            raise RuntimeError("all worker slots benched")

        if strategy == "shard":
            assignments = [
                (task, self._ctx(slots[i % len(slots)], round_no))
                for i, task in enumerate(tasks)
            ]
            return run_batch(
                self.worker_fn,
                assignments,
                backend=self.backend,
                timeout=self.timeout,
                max_workers=min(len(assignments), max(1, 2 * self.n)),
                task_cost=self.task_cost,
            )
        if strategy == "scatter":
            assignments = [
                (task, self._ctx(slot, round_no))
                for task in tasks
                for slot in slots
            ]
            return run_batch(
                self.worker_fn,
                assignments,
                backend=self.backend,
                timeout=self.timeout,
                max_workers=min(len(assignments), max(1, 2 * self.n)),
                task_cost=self.task_cost,
            )
        if strategy == "worksteal":
            if self.backend != "thread":
                # Documented fallback: a shared in-process queue cannot feed
                # child processes, so worksteal degrades to shard.
                return self.fan_out(tasks, round_no=round_no, strategy="shard")
            return self._fan_out_worksteal(tasks, slots, round_no)
        raise ValueError(
            f"unknown strategy: {strategy!r} (expected 'shard', 'scatter' or 'worksteal')"
        )

    def _fan_out_worksteal(
        self, tasks: list[Task], slots: list[int], round_no: int
    ) -> list[Result]:
        """Work-stealing fan-out. Thread backend only; results in task order."""
        tasks = list(tasks)
        if not tasks:
            return []
        work: queue.Queue = queue.Queue()
        for pos, task in enumerate(tasks):
            work.put((pos, task))
        results: list[Result | None] = [None] * len(tasks)

        puller_slots = slots[: max(1, min(len(slots), len(tasks)))]
        pullers = [
            threading.Thread(
                target=self._worksteal_puller,
                args=(slot, round_no, work, results),
                daemon=True,
                name=f"swarm-steal-{slot:04d}",
            )
            for slot in puller_slots
        ]
        for t in pullers:
            t.start()
        for t in pullers:
            t.join()
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            # Should be unreachable: _execute_one never raises and the timeout
            # path always fills the slot. Fail loudly rather than silently
            # corrupting task order.
            raise RuntimeError(f"worksteal lost results for task positions {missing}")
        return results  # type: ignore[return-value]

    def _worksteal_puller(
        self,
        slot: int,
        round_no: int,
        work: queue.Queue,
        results: list,
    ) -> None:
        # One nested single-worker executor per puller gives every stolen
        # task its own timeout; on timeout the inner future is abandoned
        # exactly like the thread backend (never blocks the swarm).
        inner = futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"swarm-ws-{slot:04d}"
        )
        try:
            while True:
                try:
                    pos, task = work.get_nowait()
                except queue.Empty:
                    return
                ctx = self._ctx(slot, round_no)
                fut = inner.submit(_execute_one, self.worker_fn, task, ctx, self.task_cost)
                try:
                    results[pos] = fut.result(timeout=self.timeout)
                except futures.TimeoutError:
                    results[pos] = Result(
                        task_id=task.id,
                        worker=_worker_name(ctx),
                        worker_index=slot,
                        output="",
                        confidence=None,
                        error=f"timeout after {self.timeout}s",
                        cost=self.task_cost,
                        duration_s=self.timeout,
                    )
        finally:
            inner.shutdown(wait=False)

    # ------------------------------------------------------------------- run

    def run(self, tasks) -> SwarmReport:
        """Run the converge loop: fan out, review, charge, bench stalls, repeat.

        ``tasks`` may be any iterable of payload dicts; it is capped at
        ``max_tasks``. Empty input returns a report with zero rounds (no crash).
        """
        pending = [
            Task(id=f"task-{i:05d}", payload=p)
            for i, p in enumerate(islice(tasks, self.max_tasks))
        ]
        round_no = 0
        while pending and round_no < self.max_rounds and self.treasury.can_spend(self.task_cost):
            try:
                results = self.fan_out(pending, round_no=round_no, strategy=self.strategy)
            except RuntimeError as e:
                # Every slot benched: the swarm cannot make progress.
                # Record and stop gracefully rather than crashing.
                self.report.note(f"stopping: {e}")
                break
            verdicts = self._review_fn(results, self.rubric, backend=self.reviewer_backend)
            # Audit L2: a backend returning fewer (or more) verdicts than
            # results would silently drop tasks via zip() truncation.
            if len(verdicts) != len(results):
                raise ValueError(
                    f"review backend returned {len(verdicts)} verdicts for "
                    f"{len(results)} results; must be one verdict per result"
                )
            self.memory.learn_from_round(verdicts, round_no)
            # Treasury clamps at zero; the REPORT must show what was actually
            # charged, not the unclamped sum (audit L1).
            charged = self.treasury.charge(len(results) * self.task_cost)
            for r, v in zip(results, verdicts):
                prog = 1.0 if v.verdict == "pass" else (0.5 if v.verdict == "retry" else 0.0)
                if self._slot_watchers[r.worker_index].observe(f"task:{r.task_id}", prog) == "escalate":
                    self.bench_slot(r.worker_index)
                    self.report.note(f"benched {r.worker} (repeated stalls)")
                # Spec: bench a slot after 3 consecutive non-pass verdicts.
                if v.verdict == "pass":
                    self._consec_fails[r.worker_index] = 0
                else:
                    fails = self._consec_fails.get(r.worker_index, 0) + 1
                    self._consec_fails[r.worker_index] = fails
                    if fails == 3:
                        self.bench_slot(r.worker_index)
                        self.report.note(f"benched {r.worker} (3 consecutive failures)")
                # Audit M4: rolling window catches high-failure workers that
                # dodge the consecutive rule with intermittent passes.
                recent = self._recent.setdefault(r.worker_index, deque(maxlen=10))
                recent.append(v.verdict == "pass")
                if (
                    len(recent) == 10
                    and sum(recent) < 4
                    and r.worker_index not in self._benched
                ):
                    self.bench_slot(r.worker_index)
                    self.report.note(
                        f"benched {r.worker} (pass rate {sum(recent)}/10 over last 10)"
                    )
            self.report.record_round(round_no, verdicts, results, cost=charged)
            if self._swarm_watchdog.observe_swarm(verdicts) == "escalate":
                self.report.note("swarm-level flatline - stopping early")
                break
            pending = [t for t, v in zip(pending, verdicts) if v.verdict != "pass" and v.confidence < self.pass_threshold]
            round_no += 1
        self.report.pending = pending
        return self.report
