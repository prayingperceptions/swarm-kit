"""End-to-end tests for the Swarm round loop, written against the public API.

These tests target the real ``swarm`` package in this repo. Reviewer
semantics they rely on (see ``swarm/review.py``):

- Verdict confidence is computed by the REVIEWER, not the worker. A clean
  output passes at 1.0; each failed rubric check costs 0.25.
- Empty output -> "fail" at confidence 1.0, which is TERMINAL
  (1.0 >= pass_threshold, so it is never retried).
- A worker crash or timeout -> "retry" at confidence 0.5: the reviewer has
  no output to judge, so it cannot be confident in failure. The task is
  retried (possibly on a different worker); max_rounds bounds poison tasks.
- A "fail" scored BELOW pass_threshold (e.g. a rubric violation at 0.75)
  IS retried next round.
- Rubric MUST:/MUST NOT: lines are matched literally (word boundaries)
  against the output text, so outputs must contain the MUST terms.
- ``report.rounds`` entries are count dicts: {"round": N, "verdicts":
  {"pass": 8, ...}, "results": ..., "cost": ..., "per_worker": {...}}.
- ``report.pending`` is the list of ``Task`` objects still not passing when
  the loop exited (empty on convergence).

Contract under test::

    swarm = Swarm(worker_fn=..., n=..., budget=..., task_cost=..., rubric=...,
                  max_rounds=..., timeout=..., pass_threshold=...,
                  backend=..., strategy=..., stall_ticks=..., max_tasks=...,
                  memory_path=..., reviewer_backend=...)
    report = swarm.run(tasks)      # any iterable of dicts, capped at max_tasks
    report.rounds                  # one record per completed round
    report.total_passed            # pass verdicts across all rounds
    report.summary()               # human-readable string

    # inside worker_fn(task, ctx): task is {"id": ..., **payload};
    # ctx.memory / ctx.treasury / ctx.identity.handle / ctx.round /
    # ctx.worker_index. Worker returns {"output": str, "confidence"?: float}.
"""
import itertools

from swarm import Swarm

# A rubric the heuristic reviewer can actually satisfy: the MUST term has to
# appear literally (word-boundary match) in the output text.
RUBRIC = "MUST: forty-two"


def make_swarm(worker_fn, tmp_path, **overrides):
    kwargs = dict(
        worker_fn=worker_fn,
        n=4,
        budget=100.0,
        task_cost=0.02,
        rubric=RUBRIC,
        max_rounds=5,
        timeout=30,
        pass_threshold=0.85,
        backend="thread",
        strategy="shard",
        stall_ticks=20,
        max_tasks=10000,
        memory_path=str(tmp_path / "swarm_memory.md"),
        reviewer_backend="heuristic",
    )
    kwargs.update(overrides)
    return Swarm(**kwargs)


def test_converges_in_two_rounds_when_second_attempt_passes(tmp_path):
    calls = {}

    def worker(task, ctx):
        tid = task["id"]
        calls[tid] = calls.get(tid, 0) + 1
        if calls[tid] == 1:
            # Violates MUST -> fail at 0.75 < pass_threshold -> retried.
            return {"output": "idk", "confidence": 0.1}
        return {"output": f"The answer for {tid} is forty-two.", "confidence": 0.97}

    report = make_swarm(worker, tmp_path).run([{"q": f"question {i}"} for i in range(8)])

    assert len(report.rounds) == 2, "round 1 fails everything, round 2 passes everything"
    assert report.rounds[0]["verdicts"] == {"fail": 8}
    assert report.rounds[-1]["verdicts"] == {"pass": 8}
    assert report.pending == [], "pending must drain to zero on convergence"
    assert len(calls) == 8 and all(v == 2 for v in calls.values())
    assert report.total_passed == 8


def test_stops_at_max_rounds_when_nothing_passes(tmp_path):
    def worker(task, ctx):
        # Rubric violation -> fail at 0.75 < pass_threshold -> retried every
        # round. (Empty output would be fail at 1.0: a confident fail is
        # terminal, so it exercises convergence, not the max_rounds path.)
        return {"output": "idk", "confidence": 0.0}

    report = make_swarm(worker, tmp_path, max_rounds=3).run([{"q": i} for i in range(6)])

    assert len(report.rounds) == 3
    assert all(r["verdicts"] == {"fail": 6} for r in report.rounds)
    assert len(report.pending) == 6, "nothing ever passed"
    assert report.total_passed == 0


def test_budget_stops_loop_early_and_treasury_never_negative(tmp_path):
    observed = []
    treasuries = []

    def worker(task, ctx):
        treasuries.append(ctx.treasury)
        observed.append(ctx.treasury.remaining)
        return {"output": "idk", "confidence": 0.0}  # fail@0.75, always retryable

    # budget 0.05 / task_cost 0.02: round 0 runs, the charge is clamped to the
    # remaining budget, and the loop must stop before round 1 -- long before
    # max_rounds=10 -- because the treasury can't spend another task.
    report = make_swarm(worker, tmp_path, n=4, budget=0.05, task_cost=0.02,
                        max_rounds=10).run([{"q": i} for i in range(10)])

    assert observed, "the loop must attempt work before the budget binds"
    assert all(r >= 0 for r in observed), "treasury.remaining went negative mid-run"
    assert len(report.rounds) < 10, "budget, not max_rounds, must stop the loop"
    assert len(report.pending) == 10, "budget stopped the loop with work unfinished"
    final = treasuries[-1].remaining
    assert 0 <= final < 0.02, "treasury must end unable to afford another task"


def test_infinite_generator_capped_at_max_tasks(tmp_path):
    seen_ids = []

    def worker(task, ctx):
        seen_ids.append(task["id"])
        return {"output": f"processed {task['id']}: forty-two", "confidence": 0.99}

    tasks = ({"n": i} for i in itertools.count())  # infinite iterable
    report = make_swarm(worker, tmp_path, n=8, max_tasks=40).run(tasks)

    # If the harness didn't cap the iterable, this test would hang forever.
    assert len(set(seen_ids)) == 40, "harness must cap the iterable at max_tasks"
    assert report.total_passed == 40
    assert report.pending == []
    assert len(report.rounds) == 1


def test_crash_on_one_task_retries_until_max_rounds(tmp_path):
    # Task dicts carry no "id" here: the harness assigns zero-padded ids
    # (task-00000, task-00001, ...) before calling the worker.
    crash_id = "task-00003"
    attempts = []

    def worker(task, ctx):
        # This task ALWAYS crashes: the reviewer must ask for a retry
        # (low confidence, no output to judge), never a terminal fail.
        if task["id"] == crash_id:
            attempts.append(task["id"])
            raise RuntimeError("worker crashed")
        return {"output": f"done {task['id']}: forty-two", "confidence": 0.96}

    report = make_swarm(worker, tmp_path, max_rounds=3).run([{"q": i} for i in range(10)])

    # 9 tasks pass round 1; the crashed task is retried every round until
    # max_rounds stops the loop (poison tasks are bounded, not infinite).
    assert len(report.rounds) == 3
    assert report.rounds[0]["verdicts"] == {"pass": 9, "retry": 1}
    assert len(attempts) == 3, "crashed task retried once per round"
    assert [t.id for t in report.pending] == [crash_id]
    assert isinstance(report.summary(), str) and report.summary()
