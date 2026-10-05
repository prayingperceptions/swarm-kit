"""Execution backends for swarm-kit: run worker functions concurrently.

Two backends, one contract: :func:`run_batch` always returns results in
assignment order, one :class:`~swarm.worker.Result` per assignment.

Backends
--------
- ``"thread"`` — :class:`concurrent.futures.ThreadPoolExecutor`. Shares the
  process, so workers see the real ``ctx`` (memory, treasury, identity).
  On timeout the future is **abandoned**: we record a timeout Result and
  move on. The hung thread keeps running in the background until the worker
  function returns — it never blocks the swarm. (Threads that never return
  linger until interpreter exit; keep worker functions bounded.)

- ``"process"`` — :class:`concurrent.futures.ProcessPoolExecutor`. True
  parallelism and isolation, with two hard constraints:

  1. ``worker_fn`` MUST be picklable: a module-level function, not a
     closure, lambda, or bound method.
  2. ``ctx`` cannot carry live objects across the process boundary, so each
     child gets a **slim ctx**: ``identity``, ``round`` and ``worker_index``
     are preserved; ``memory`` and ``treasury`` are ``None``.

  LIMITATION (stated plainly): workers on the process backend cannot use
  ``ctx.memory`` or ``ctx.treasury``. If your worker needs shared memory or
  live budget accounting, use the thread backend.

Any other backend name raises :class:`ValueError`.
"""

from __future__ import annotations

import concurrent.futures as futures
import time

from .worker import Context, Result, Task, WorkerFn


def _worker_name(ctx: Context) -> str:
    """Best-effort worker handle from a ctx (identity may be a slim fake)."""
    ident = getattr(ctx, "identity", None)
    handle = getattr(ident, "handle", None) if ident is not None else None
    if handle:
        return str(handle)
    return f"worker-{ctx.worker_index:04d}"


def _execute_one(
    worker_fn: WorkerFn, task: Task, ctx: Context, task_cost: float = 0.0
) -> Result:
    """Run one task on one worker. Never raises.

    Times the call, catches every :class:`Exception` from the worker into
    ``Result.error``, and enforces the worker contract
    (``{"output": str, "confidence"?: float}``). Contract violations are
    recorded as error Results, not raised. (:class:`BaseException` such as
    ``KeyboardInterrupt``/``SystemExit`` is deliberately *not* swallowed —
    those belong to the operator, not the worker.)
    """
    start = time.perf_counter()
    name = _worker_name(ctx)
    try:
        # Canonical task id wins over a payload "id" key (audit L4): the
        # worker-facing dict always carries the harness-assigned id.
        task_dict = {**task.payload, "id": task.id}
        returned = worker_fn(task_dict, ctx)
        duration_s = time.perf_counter() - start

        if not isinstance(returned, dict):
            return Result(
                task_id=task.id,
                worker=name,
                worker_index=ctx.worker_index,
                output=str(returned),
                confidence=None,
                error=(
                    "contract violation: worker returned "
                    f"{type(returned).__name__}, expected dict"
                ),
                cost=task_cost,
                duration_s=duration_s,
            )
        if "output" not in returned:
            return Result(
                task_id=task.id,
                worker=name,
                worker_index=ctx.worker_index,
                output="",
                confidence=None,
                error="contract violation: worker result missing 'output' key",
                cost=task_cost,
                duration_s=duration_s,
            )
        output = returned["output"]
        output = "" if output is None else str(output)
        confidence = returned.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = None
        return Result(
            task_id=task.id,
            worker=name,
            worker_index=ctx.worker_index,
            output=output,
            confidence=confidence,
            error=None,
            cost=task_cost,
            duration_s=duration_s,
        )
    except Exception as exc:  # noqa: BLE001 — the whole point is to catch all
        duration_s = time.perf_counter() - start
        return Result(
            task_id=task.id,
            worker=name,
            worker_index=ctx.worker_index,
            output="",
            confidence=None,
            error=f"{type(exc).__name__}: {exc}",
            cost=task_cost,
            duration_s=duration_s,
        )


def _timeout_result(task: Task, ctx: Context, timeout: float, task_cost: float) -> Result:
    return Result(
        task_id=task.id,
        worker=_worker_name(ctx),
        worker_index=ctx.worker_index,
        output="",
        confidence=None,
        error=f"timeout after {timeout}s",
        cost=task_cost,
        duration_s=timeout,
    )


def _collect(
    executor: futures.Executor,
    worker_fn: WorkerFn,
    jobs: list[tuple[Task, Context]],
    timeout: float,
    task_cost: float,
) -> list[Result]:
    """Submit all jobs, collect in order, abandon (never await) hung work.

    ``timeout`` is a SHARED batch deadline, not a per-task allowance: every
    result is awaited with only the time remaining until
    ``start + timeout``. This bounds worst-case wall clock to ~``timeout``
    per batch no matter how many tasks hang (audit H2) — a per-task timeout
    would let N hung tasks stall the swarm for N * timeout.

    The executor is shut down with ``wait=False``: a hung worker must never
    block the swarm, so we do not wait for abandoned futures. (Their threads
    or processes keep running until the worker function returns.)
    """
    futs = [executor.submit(_execute_one, worker_fn, task, ctx, task_cost)
            for task, ctx in jobs]
    deadline = time.monotonic() + timeout
    results: list[Result] = []
    try:
        for (task, ctx), fut in zip(jobs, futs):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                results.append(fut.result(timeout=remaining))
            except futures.TimeoutError:
                results.append(_timeout_result(task, ctx, timeout, task_cost))
            except Exception as exc:  # _execute_one should never raise; belt and braces
                results.append(
                    Result(
                        task_id=task.id,
                        worker=_worker_name(ctx),
                        worker_index=ctx.worker_index,
                        output="",
                        confidence=None,
                        error=f"harness error: {type(exc).__name__}: {exc}",
                        cost=task_cost,
                        duration_s=0.0,
                    )
                )
        return results
    finally:
        executor.shutdown(wait=False)


def _run_batch_thread(
    worker_fn: WorkerFn,
    assignments: list[tuple[Task, Context]],
    timeout: float,
    max_workers: int | None,
    task_cost: float,
) -> list[Result]:
    # NOTE: we deliberately do NOT use `with ThreadPoolExecutor(...)` here:
    # __exit__ waits for pending futures, which would block the swarm on a
    # hung worker. _collect shuts down with wait=False instead; hung threads
    # are abandoned and never block the swarm.
    executor = futures.ThreadPoolExecutor(max_workers=max_workers)
    return _collect(executor, worker_fn, assignments, timeout, task_cost)


def _run_batch_process(
    worker_fn: WorkerFn,
    assignments: list[tuple[Task, Context]],
    timeout: float,
    max_workers: int | None,
    task_cost: float,
) -> list[Result]:
    # worker_fn must be picklable (module-level function). The ctx is rebuilt
    # slim for the child: identity / round / worker_index survive, but live
    # collaborators cannot cross the process boundary, so memory=None and
    # treasury=None. This is a documented limitation, not a bug: use the
    # thread backend if workers need shared memory or budget accounting.
    slim = [
        (
            task,
            Context(
                memory=None,
                treasury=None,
                identity=ctx.identity,
                round=ctx.round,
                worker_index=ctx.worker_index,
            ),
        )
        for task, ctx in assignments
    ]
    executor = futures.ProcessPoolExecutor(max_workers=max_workers)
    return _collect(executor, worker_fn, slim, timeout, task_cost)


def run_batch(
    worker_fn: WorkerFn,
    assignments: list[tuple[Task, Context]],
    backend: str = "thread",
    timeout: float = 30.0,
    max_workers: int | None = None,
    task_cost: float = 0.0,
) -> list[Result]:
    """Run ``assignments`` = [(Task, Context), ...] concurrently.

    Returns one :class:`~swarm.worker.Result` per assignment, **in
    assignment order**. Unknown backend names raise :class:`ValueError`.

    ``timeout`` is a shared batch deadline (see :func:`_collect`); it must
    be positive. On the process backend, ``worker_fn`` is validated as
    picklable up front so a bad worker fails fast with a clear error
    instead of burning rounds on doomed retries.
    """
    assignments = list(assignments)
    if not assignments:
        return []
    timeout = _require_positive_timeout(timeout)
    if backend == "thread":
        return _run_batch_thread(worker_fn, assignments, timeout, max_workers, task_cost)
    if backend == "process":
        _require_picklable(worker_fn)
        return _run_batch_process(worker_fn, assignments, timeout, max_workers, task_cost)
    raise ValueError(f"unknown backend: {backend!r} (expected 'thread' or 'process')")


def _require_positive_timeout(timeout: float) -> float:
    """Coerce/validate a timeout; audit L3: timeout=None must never hang the swarm."""
    try:
        timeout_f = float(timeout)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be a positive number, got {timeout!r}") from None
    if not timeout_f > 0:
        raise ValueError(f"timeout must be positive, got {timeout!r}")
    return timeout_f


def _require_picklable(worker_fn: WorkerFn) -> None:
    """Fail fast with a clear error if the process backend can't ship worker_fn."""
    import pickle

    try:
        pickle.dumps(worker_fn)
    except Exception as exc:
        raise ValueError(
            "process backend requires a picklable worker_fn "
            "(module-level function, not a lambda/closure/bound method); "
            f"got {type(worker_fn).__name__}: {exc}"
        ) from exc
