"""Unit tests for swarm.backends: _execute_one and run_batch."""

import threading

import pytest

from swarm.backends import _execute_one, run_batch
from swarm.worker import Context, Result, Task

from conftest import FakeIdentity


def make_ctx(idx=0, round_no=0):
    return Context(
        memory="mem",
        treasury="treas",
        identity=FakeIdentity(idx),
        round=round_no,
        worker_index=idx,
    )


def ok_worker(task_dict, ctx):
    return {"output": f"hello {task_dict['id']}", "confidence": 0.75}


# ---------------------------------------------------------------- _execute_one


def test_execute_one_success_parses_contract():
    task = Task(id="t-1", payload={"x": 1})
    r = _execute_one(ok_worker, task, make_ctx(idx=3, round_no=2), task_cost=0.05)
    assert isinstance(r, Result)
    assert r.task_id == "t-1"
    assert r.worker == "worker-0003"
    assert r.worker_index == 3
    assert r.output == "hello t-1"
    assert r.confidence == 0.75
    assert r.error is None
    assert r.cost == 0.05
    assert r.duration_s >= 0.0


def test_execute_one_builds_task_dict_from_id_and_payload():
    seen = {}

    def worker(task_dict, ctx):
        seen.update(task_dict)
        return {"output": "ok"}

    _execute_one(worker, Task(id="abc", payload={"k": "v"}), make_ctx())
    assert seen == {"id": "abc", "k": "v"}


def test_execute_one_catches_worker_exception():
    def boom(task_dict, ctx):
        raise RuntimeError("kaput")

    r = _execute_one(boom, Task(id="t-9", payload={}), make_ctx(idx=1))
    assert r.task_id == "t-9"
    assert r.worker == "worker-0001"
    assert r.output == ""
    assert r.confidence is None
    assert r.error is not None and "kaput" in r.error
    assert "RuntimeError" in r.error
    assert r.duration_s >= 0.0


def test_execute_one_non_dict_return_is_contract_violation():
    r = _execute_one(lambda td, ctx: "not a dict", Task(id="t-2", payload={}), make_ctx())
    assert r.error is not None and "dict" in r.error
    assert r.output == "not a dict"


def test_execute_one_missing_output_key_is_contract_violation():
    r = _execute_one(lambda td, ctx: {"confidence": 1.0}, Task(id="t-3", payload={}), make_ctx())
    assert r.error is not None and "output" in r.error


def test_execute_one_bad_confidence_coerces_to_none():
    r = _execute_one(
        lambda td, ctx: {"output": "ok", "confidence": "high"},
        Task(id="t-4", payload={}),
        make_ctx(),
    )
    assert r.error is None
    assert r.confidence is None


def test_execute_one_falls_back_when_identity_missing():
    ctx = Context(worker_index=7)  # identity=None
    r = _execute_one(ok_worker, Task(id="t-5", payload={}), ctx)
    assert r.worker == "worker-0007"


# ---------------------------------------------------------------- run_batch


def assignments(n, start_idx=0):
    return [
        (Task(id=f"task-{i:05d}", payload={"i": i}), make_ctx(idx=start_idx + i))
        for i in range(n)
    ]


def test_run_batch_thread_preserves_assignment_order():
    results = run_batch(ok_worker, assignments(8), backend="thread", max_workers=4)
    assert len(results) == 8
    assert [r.task_id for r in results] == [f"task-{i:05d}" for i in range(8)]
    assert all(r.error is None for r in results)
    assert [r.worker_index for r in results] == list(range(8))


def test_run_batch_thread_task_cost_flows_through():
    results = run_batch(ok_worker, assignments(3), backend="thread", task_cost=0.11)
    assert all(r.cost == 0.11 for r in results)


def test_run_batch_empty_assignments_returns_empty():
    assert run_batch(ok_worker, [], backend="thread") == []


def test_run_batch_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="unknown backend"):
        run_batch(ok_worker, assignments(1), backend="carrier-pigeon")


def test_run_batch_thread_timeout_abandons_hung_worker():
    release = threading.Event()

    def slow_worker(task_dict, ctx):
        release.wait(10)  # hangs until the test releases it
        return {"output": "too slow"}

    try:
        results = run_batch(
            slow_worker, assignments(2), backend="thread", timeout=0.2, max_workers=2
        )
    finally:
        release.set()  # let the abandoned thread finish promptly
    assert len(results) == 2
    for r in results:
        assert r.error is not None and "timeout" in r.error.lower()
        assert "0.2s" in r.error


def test_run_batch_thread_crashed_worker_does_not_affect_others():
    def flaky(task_dict, ctx):
        if task_dict["id"] == "task-00001":
            raise ValueError("boom")
        return {"output": "fine"}

    results = run_batch(flaky, assignments(3), backend="thread", max_workers=3)
    assert results[0].error is None and results[0].output == "fine"
    assert results[1].error is not None and "boom" in results[1].error
    assert results[2].error is None and results[2].output == "fine"


# ---------------------------------------------------------------- process backend


def proc_worker(task_dict, ctx):
    # Module-level => picklable. Reports what the slim ctx looks like.
    return {
        "output": (
            f"{ctx.identity.handle}|mem={ctx.memory}|treas={ctx.treasury}"
            f"|round={ctx.round}|idx={ctx.worker_index}"
        )
    }


def slow_proc_worker(task_dict, ctx):
    # Module-level => picklable. Sleeps past the timeout on purpose.
    import time as _time

    _time.sleep(2)
    return {"output": "late"}


def test_run_batch_process_uses_slim_ctx():
    results = run_batch(proc_worker, assignments(3), backend="process", max_workers=2)
    assert len(results) == 3
    assert [r.task_id for r in results] == [f"task-{i:05d}" for i in range(3)]
    for i, r in enumerate(results):
        assert r.error is None, r.error
        # Slim ctx: identity/round/worker_index survive; memory/treasury are None.
        assert f"worker-{i:04d}" in r.output
        assert "mem=None" in r.output
        assert "treas=None" in r.output
        assert f"idx={i}" in r.output


def test_run_batch_process_timeout_mentions_timeout():
    results = run_batch(
        slow_proc_worker, assignments(1), backend="process", timeout=0.2
    )
    assert len(results) == 1
    assert results[0].error is not None and "timeout" in results[0].error.lower()


# ------------------------------------------------------- audit fixes (H2/M3/L3/L4)


def test_timeout_is_shared_batch_deadline_not_per_task():
    # Audit H2: N hung tasks must not cost N * timeout of wall clock.
    import time as _time

    def hang(task_dict, ctx):
        import time as _t
        _t.sleep(30)

    t0 = _time.perf_counter()
    results = run_batch(hang, assignments(4), backend="thread", timeout=0.5)
    dt = _time.perf_counter() - t0
    assert len(results) == 4
    assert all(r.error and "timeout" in r.error for r in results)
    assert dt < 2.0, f"batch deadline violated: 4 hung tasks took {dt:.2f}s"


def test_process_backend_rejects_unpicklable_worker_fn_eagerly():
    # Audit M3: a lambda must fail fast with a clear error, not burn rounds
    # on doomed "harness error" retries.
    with pytest.raises(ValueError, match="picklable"):
        run_batch(lambda t, c: {"output": "x"}, assignments(2),
                  backend="process", timeout=5)


def test_nonpositive_timeout_raises_value_error():
    # Audit L3: timeout=None/0/negative must never silently hang the swarm.
    for bad in (None, 0, -1, "soon"):
        with pytest.raises(ValueError, match="timeout"):
            run_batch(ok_worker, assignments(1), backend="thread", timeout=bad)


def test_payload_id_does_not_shadow_canonical_task_id():
    # Audit L4: the harness-assigned Task.id always wins in the worker dict.
    seen = {}

    def worker(task_dict, ctx):
        seen.update(task_dict)
        return {"output": "ok"}

    _execute_one(worker, Task(id="task-00007", payload={"id": "t-7", "k": "v"}),
                 make_ctx())
    assert seen["id"] == "task-00007"
    assert seen["k"] == "v"
