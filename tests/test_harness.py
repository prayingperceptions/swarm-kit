"""Unit tests for swarm.harness: fan-out strategies, benching, and run()."""

import threading

import pytest

from swarm.harness import Swarm

from conftest import (
    FakeMemory,
    FakeSwarmWatchdog,
    FakeTreasury,
    FakeVerdict,
    fail_all_review,
    make_swarm,
    make_tasks,
    pass_all_review,
)


def _module_worker(task_dict, ctx):
    """Module-level (picklable) worker for process-backend tests."""
    return {"output": f"done:{task_dict['id']}", "confidence": 0.9}


# ------------------------------------------------------------------ fan_out


def test_shard_distributes_round_robin():
    swarm = make_swarm(n=4)
    results = swarm.fan_out(make_tasks(10))
    assert len(results) == 10
    # results[i] corresponds to tasks[i]; slots cycle 0,1,2,3,0,1,...
    assert [r.worker_index for r in results] == [i % 4 for i in range(10)]
    assert [r.task_id for r in results] == [f"task-{i:05d}" for i in range(10)]
    assert results[0].worker == "worker-0000"
    assert results[5].worker == "worker-0001"


def test_shard_passes_task_cost_and_round_into_ctx():
    seen = []

    def worker(task_dict, ctx):
        seen.append((ctx.round, ctx.worker_index))
        return {"output": "ok"}

    swarm = make_swarm(n=2, worker_fn=worker, task_cost=0.07)
    results = swarm.fan_out(make_tasks(3), round_no=4)
    assert all(r.cost == pytest.approx(0.07) for r in results)
    assert sorted(s for _, s in seen) == [0, 0, 1]
    assert all(rnd == 4 for rnd, _ in seen)


def test_scatter_multiplies_tasks_by_slots():
    swarm = make_swarm(n=3)
    results = swarm.fan_out(make_tasks(2), strategy="scatter")
    # every task to every slot: 2 tasks x 3 slots = 6 executions
    assert len(results) == 6
    assert sorted(r.task_id for r in results) == [
        "task-00000", "task-00000", "task-00000",
        "task-00001", "task-00001", "task-00001",
    ]
    # task-major grouping: first three results are task-00000 on slots 0,1,2
    assert [r.worker_index for r in results[:3]] == [0, 1, 2]
    assert [r.worker_index for r in results[3:]] == [0, 1, 2]


def test_unknown_strategy_raises_value_error():
    swarm = make_swarm()
    with pytest.raises(ValueError, match="unknown strategy"):
        swarm.fan_out(make_tasks(2), strategy="teleport")


def test_fan_out_empty_tasks_returns_empty():
    swarm = make_swarm()
    assert swarm.fan_out([]) == []


def test_crashed_worker_yields_error_result_others_fine():
    def flaky(task_dict, ctx):
        if task_dict["id"] == "task-00001":
            raise RuntimeError("boom")
        return {"output": "fine", "confidence": 1.0}

    swarm = make_swarm(n=2, worker_fn=flaky)
    results = swarm.fan_out(make_tasks(3))
    assert len(results) == 3
    assert results[0].error is None and results[0].output == "fine"
    assert results[1].error is not None and "boom" in results[1].error
    assert results[1].output == ""
    assert results[2].error is None


def test_fan_out_timeout_yields_error_result_mentioning_timeout():
    release = threading.Event()

    def slow(task_dict, ctx):
        release.wait(10)
        return {"output": "late"}

    swarm = make_swarm(n=2, worker_fn=slow, timeout=0.2)
    try:
        results = swarm.fan_out(make_tasks(2))
    finally:
        release.set()
    assert len(results) == 2
    assert all(r.error and "timeout" in r.error for r in results)


def test_worksteal_returns_results_in_task_order():
    swarm = make_swarm(n=4)
    results = swarm.fan_out(make_tasks(9), strategy="worksteal")
    assert len(results) == 9
    assert [r.task_id for r in results] == [f"task-{i:05d}" for i in range(9)]
    assert all(r.error is None for r in results)
    assert all(0 <= r.worker_index < 4 for r in results)


def test_worksteal_falls_back_to_shard_on_process_backend():
    # Needs a module-level (picklable) worker: the process backend now
    # validates picklability eagerly (audit M3).
    swarm = make_swarm(n=3, backend="process", worker_fn=_module_worker)
    tasks = make_tasks(4)
    results = swarm.fan_out(tasks, strategy="worksteal")
    # shard fallback: one result per task, in task order, round-robin slots
    assert [r.task_id for r in results] == [t.id for t in tasks]
    assert [r.worker_index for r in results] == [0, 1, 2, 0]


# ------------------------------------------------------------------ benching


def test_bench_slot_excludes_slot_from_next_fan_out():
    seen = []

    def worker(task_dict, ctx):
        seen.append(ctx.worker_index)
        return {"output": "ok"}

    swarm = make_swarm(n=4, worker_fn=worker)
    swarm.bench_slot(1)
    swarm.bench_slot(3)
    assert swarm.benched == {1, 3}
    swarm.fan_out(make_tasks(8))
    assert 1 not in seen and 3 not in seen
    assert set(seen) == {0, 2}


def test_all_slots_benched_raises_runtime_error():
    swarm = make_swarm(n=2)
    swarm.bench_slot(0)
    swarm.bench_slot(1)
    with pytest.raises(RuntimeError, match="all worker slots benched"):
        swarm.fan_out(make_tasks(2))


def test_three_consecutive_failures_bench_the_slot():
    # Slot 0 always fails review; slot 1 always passes. After 3 consecutive
    # non-pass verdicts, slot 0 must be benched and its tasks redistributed
    # to slot 1 -- no task is lost.
    def review_fn(results, rubric, backend="heuristic"):
        out = []
        for r in results:
            if r.worker_index == 0:
                out.append(FakeVerdict(id=r.task_id, verdict="fail",
                                       confidence=0.5, reason="bad",
                                       worker=r.worker))
            else:
                out.append(FakeVerdict(id=r.task_id, verdict="pass",
                                       confidence=1.0, reason="good",
                                       worker=r.worker))
        return out

    swarm = make_swarm(n=2, _review_fn=review_fn, max_rounds=5)
    report = swarm.run([{"i": i} for i in range(4)])

    assert 0 in swarm.benched, "slot 0 benched after 3 consecutive failures"
    assert 1 not in swarm.benched, "healthy slot 1 never benched"
    # Every task eventually passed via slot 1: benching redistributes work,
    # it does not drop it.
    assert report.total_passed == 4
    assert report.pending == []
    assert any("3 consecutive failures" in n for n in report.notes)


def test_pass_resets_consecutive_failure_count():
    # A pass between failures resets the consecutive counter, and a 50%
    # worker stays above the rolling window's 40% floor: a flaky-but-working
    # slot is not benched.
    calls = {"n": 0}

    def review_fn(results, rubric, backend="heuristic"):
        out = []
        for r in results:
            calls["n"] += 1
            # pass, fail, pass, fail, ... never 2 in a row, 50% pass rate
            verdict = "fail" if calls["n"] % 2 == 0 else "pass"
            conf = 1.0 if verdict == "pass" else 0.5
            out.append(FakeVerdict(id=r.task_id, verdict=verdict,
                                   confidence=conf, reason="x",
                                   worker=r.worker))
        return out

    swarm = make_swarm(n=1, _review_fn=review_fn, max_rounds=10)
    swarm.run([{"i": i} for i in range(6)])
    assert swarm.benched == frozenset(), "flaky 50%-pass slot must not be benched"


def test_run_benches_slot_on_stall_escalation():
    swarm = make_swarm(n=2, max_rounds=3, _review_fn=fail_all_review)
    # Force slot 0's watcher to escalate on every observation.
    swarm._slot_watchers[0].observe = lambda action, progress: "escalate"
    report = swarm.run([{"q": 1}, {"q": 2}])
    assert 0 in swarm.benched
    assert any("benched worker-0000" in note for note in report.notes)


def test_swarm_watchdog_escalation_stops_run_early():
    swarm = make_swarm(
        max_rounds=10,
        _review_fn=fail_all_review,
        _swarm_watchdog=FakeSwarmWatchdog(escalate_after=2),
    )
    report = swarm.run([{"q": i} for i in range(3)])
    assert len(report.rounds) == 2
    assert any("flatline" in note for note in report.notes)


# ---------------------------------------------------------------------- run


def test_run_converges_when_reviewer_passes_everything_on_round_two():
    calls = {"n": 0}

    def reviewer(results, rubric, backend="heuristic"):
        calls["n"] += 1
        verdict = "pass" if calls["n"] >= 2 else "retry"
        conf = 1.0 if verdict == "pass" else 0.1
        return [
            FakeVerdict(id=r.task_id, verdict=verdict, confidence=conf,
                        reason="fake", worker=r.worker)
            for r in results
        ]

    treasury = FakeTreasury(100.0)
    memory = FakeMemory()
    swarm = make_swarm(
        n=4, max_rounds=5, task_cost=0.02,
        _review_fn=reviewer, _treasury=treasury, _memory=memory,
    )
    report = swarm.run([{"q": i} for i in range(6)])

    # Round 0: all retry (low confidence -> stay pending). Round 1: all pass.
    assert calls["n"] == 2
    assert len(report.rounds) == 2
    assert report.rounds[0]["verdicts"] == {"retry": 6}
    assert report.rounds[1]["verdicts"] == {"pass": 6}
    assert report.pending == [], "everything passed: nothing pending"
    # Treasury charged once per result per round.
    assert treasury.spent == pytest.approx(12 * 0.02)
    # Memory learned from both rounds.
    assert [rn for rn, _ in memory.rounds_learned] == [0, 1]
    # Summary renders without crashing.
    summary = report.summary()
    assert "Rounds run : 2" in summary
    assert "pass=6" in summary


def test_run_respects_max_rounds():
    swarm = make_swarm(max_rounds=3, _review_fn=fail_all_review)
    report = swarm.run([{"q": i} for i in range(4)])
    assert len(report.rounds) == 3
    assert [r["round"] for r in report.rounds] == [0, 1, 2]
    # Low-confidence fails stay pending across all rounds.
    assert [t.id for t in report.pending] == [f"task-{i:05d}" for i in range(4)]


def test_run_pending_holds_confident_fails_out():
    # Confident fails (confidence >= pass_threshold) are NOT retried.
    def confident_fail(results, rubric, backend="heuristic"):
        return [
            FakeVerdict(id=r.task_id, verdict="fail", confidence=1.0,
                        reason="fake", worker=r.worker)
            for r in results
        ]

    swarm = make_swarm(max_rounds=5, _review_fn=confident_fail)
    report = swarm.run([{"q": 1}])
    assert len(report.rounds) == 1
    assert report.pending == []


def test_run_stops_when_budget_exhausted():
    # task_cost 0.02, 4 tasks/round -> 0.08/round; budget 0.05 allows 1 round
    # (can_spend(0.02) is true once, then the clamped charge drains it).
    swarm = make_swarm(
        max_rounds=10, task_cost=0.02,
        _treasury=FakeTreasury(0.05),
        _review_fn=fail_all_review,
    )
    report = swarm.run([{"q": i} for i in range(4)])
    assert len(report.rounds) == 1


def test_run_empty_tasks_returns_clean_report():
    swarm = make_swarm()
    report = swarm.run([])
    assert report.rounds == []
    assert report.notes == []
    summary = report.summary()
    assert "Rounds run : 0" in summary


def test_run_accepts_any_iterable_and_caps_at_max_tasks():
    swarm = make_swarm(n=2, max_rounds=1, max_tasks=5, _review_fn=pass_all_review)
    report = swarm.run(({"q": i} for i in range(100)))  # generator, not a list
    assert report.total_results == 5


def test_run_records_per_worker_counts_and_notes_in_summary():
    swarm = make_swarm(n=2, max_rounds=1, _review_fn=pass_all_review)
    report = swarm.run([{"q": 1}, {"q": 2}])
    report.note("custom note")
    summary = report.summary()
    assert "worker-0000: 1" in summary
    assert "worker-0001: 1" in summary
    assert "custom note" in summary
    assert "Total cost" in summary
    assert "Overall pass rate" in summary


def test_swarm_uses_default_collaborators_when_not_injected(tmp_path):
    # Default path constructs real Memory/Treasury/Watchdog/review.
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)  # keep swarm_memory.md out of the repo
    try:
        swarm = Swarm(worker_fn=lambda td, ctx: {"output": "fine", "confidence": 1.0},
                      n=2, max_rounds=1, rubric="")
        report = swarm.run([{"q": 1}])
        assert len(report.rounds) == 1
        assert (tmp_path / "swarm_memory.md").exists()
    finally:
        os.chdir(cwd)


# ------------------------------------------------------- audit fixes (M4/L2/L1)


def test_rolling_window_benches_intermittent_failer():
    # Audit M4: fail, fail, pass repeating forever evades the consecutive-3
    # rule but must trip the rolling window (pass rate < 40% over last 10).
    calls = {"n": 0}

    def review_fn(results, rubric, backend="heuristic"):
        out = []
        for r in results:
            calls["n"] += 1
            verdict = "pass" if calls["n"] % 3 == 0 else "fail"
            out.append(FakeVerdict(id=r.task_id, verdict=verdict,
                                   confidence=0.5 if verdict == "fail" else 1.0,
                                   reason="x", worker=r.worker))
        return out

    swarm = make_swarm(n=1, _review_fn=review_fn, max_rounds=20)
    swarm.run([{"i": i} for i in range(12)])
    assert 0 in swarm.benched, "intermittent 66%-failer must be benched by the window"
    assert any("pass rate" in n for n in swarm.report.notes)


def test_verdict_count_mismatch_raises():
    # Audit L2: a backend returning fewer verdicts than results must fail
    # loudly, not silently drop tasks via zip() truncation.
    def review_fn(results, rubric, backend="heuristic"):
        return [FakeVerdict(id="t", verdict="pass", confidence=1.0,
                            reason="x", worker="w")]

    swarm = make_swarm(n=2, _review_fn=review_fn)
    with pytest.raises(ValueError, match="one verdict per result"):
        swarm.run([{"i": i} for i in range(4)])


def test_report_cost_reflects_actual_charge_not_unclamped_sum():
    # Audit L1: with budget $0.05 and 10 tasks at $0.02, the report must say
    # $0.05 (what the treasury actually charged), not $0.20.
    from swarm.harness import Swarm

    real_swarm = Swarm(
        worker_fn=lambda t, c: {"output": "ok"},
        n=2, budget=0.05, task_cost=0.02, max_rounds=1, timeout=5,
        memory_path="/tmp/l1_test_memory.md",
        _memory=FakeMemory(), _swarm_watchdog=FakeSwarmWatchdog(),
    )
    report = real_swarm.run([{"i": i} for i in range(10)])
    assert report.total_cost == pytest.approx(0.05)
