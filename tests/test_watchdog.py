"""Tests for swarm.watchdog: StallWatcher + SwarmWatchdog."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.watchdog import StallWatcher, SwarmWatchdog  # noqa: E402


@dataclass
class Verdict:
    id: str
    verdict: str  # "pass" | "fail" | "retry"
    confidence: float
    reason: str
    worker: str


def make_verdicts(passes: int, total: int, tag: str = "r") -> list[Verdict]:
    return [
        Verdict(
            id=f"{tag}-{i}",
            verdict="pass" if i < passes else "fail",
            confidence=0.9,
            reason="",
            worker="worker-0001",
        )
        for i in range(total)
    ]


# --- StallWatcher ---


def test_climbing_progress_never_fires():
    w = StallWatcher(stall_ticks=5)
    for i in range(50):
        assert w.observe("work", float(i)) == "ok"
        assert w.flat_streak == 0


def test_nudge_fires_exactly_once_at_stall_ticks():
    w = StallWatcher(stall_ticks=5)
    w.observe("work", 3.0)  # sets best
    results = [w.observe("work", 3.0) for _ in range(20)]
    assert results[4] == "nudge"  # flat_streak hits 5 exactly on this tick
    assert results.count("nudge") == 1


def test_escalate_at_two_times_stall_ticks():
    w = StallWatcher(stall_ticks=5)
    w.observe("work", 3.0)
    results = [w.observe("work", 3.0) for _ in range(30)]
    assert results[9] == "escalate"  # flat_streak hits 10 exactly
    assert results.count("escalate") == 1
    # nudge fired once, everything else ok
    assert results.count("nudge") == 1
    assert all(r == "ok" for r in results[:4])


def test_improvement_resets_and_rearms():
    w = StallWatcher(stall_ticks=5)
    w.observe("work", 3.0)
    for _ in range(5):
        w.observe("work", 3.0)  # -> nudge on 5th
    assert w.flat_streak == 5
    # Strictly greater progress re-arms
    assert w.observe("work", 3.5) == "ok"
    assert w.flat_streak == 0
    # Nudge fires again after another full window
    results = [w.observe("work", 3.5) for _ in range(5)]
    assert results[-1] == "nudge"
    assert results.count("nudge") == 1


def test_equal_progress_counts_as_flat():
    w = StallWatcher(stall_ticks=3)
    w.observe("work", 2.0)
    assert w.observe("work", 2.0) == "ok"  # streak 1
    assert w.observe("work", 2.0) == "ok"  # streak 2
    assert w.observe("work", 2.0) == "nudge"  # streak 3 -> nudge


def test_regressed_progress_counts_as_flat():
    w = StallWatcher(stall_ticks=2)
    w.observe("work", 2.0)
    assert w.observe("work", 1.0) == "ok"  # streak 1 (regression, not progress)
    assert w.observe("work", 1.0) == "nudge"  # streak 2


def test_reset_clears_state():
    w = StallWatcher(stall_ticks=3)
    w.observe("work", 1.0)
    for _ in range(3):
        w.observe("work", 1.0)
    assert w.flat_streak == 3
    w.reset()
    assert w.flat_streak == 0
    assert w.observe("work", 0.5) == "ok"  # first observation after reset


def test_invalid_stall_ticks_rejected():
    with pytest.raises(ValueError):
        StallWatcher(stall_ticks=0)
    with pytest.raises(ValueError):
        StallWatcher(stall_ticks=-4)


# --- SwarmWatchdog ---


def test_improving_pass_rates_ok():
    wd = SwarmWatchdog(flat_rounds=3)
    assert wd.observe_swarm(make_verdicts(2, 10)) == "ok"  # 0.2
    assert wd.observe_swarm(make_verdicts(4, 10)) == "ok"  # 0.4
    assert wd.observe_swarm(make_verdicts(6, 10)) == "ok"  # 0.6
    assert wd.observe_swarm(make_verdicts(8, 10)) == "ok"  # 0.8


def test_flat_for_flat_rounds_escalates():
    wd = SwarmWatchdog(flat_rounds=3)
    assert wd.observe_swarm(make_verdicts(5, 10)) == "ok"  # 0.5 baseline
    assert wd.observe_swarm(make_verdicts(5, 10)) == "ok"  # 0.5
    assert wd.observe_swarm(make_verdicts(5, 10)) == "ok"  # 0.5
    # Last 3 rounds max (0.5) <= prior window max (0.5) -> escalate
    assert wd.observe_swarm(make_verdicts(5, 10)) == "escalate"


def test_declining_pass_rates_escalate():
    wd = SwarmWatchdog(flat_rounds=3)
    wd.observe_swarm(make_verdicts(8, 10))  # 0.8
    wd.observe_swarm(make_verdicts(4, 10))  # 0.4
    wd.observe_swarm(make_verdicts(2, 10))  # 0.2
    assert wd.observe_swarm(make_verdicts(1, 10)) == "escalate"  # 0.1


def test_empty_verdicts_ok():
    wd = SwarmWatchdog(flat_rounds=3)
    assert wd.observe_swarm([]) == "ok"
    assert wd.observe_swarm(make_verdicts(5, 10)) == "ok"


def test_first_call_always_ok():
    wd = SwarmWatchdog(flat_rounds=3)
    assert wd.observe_swarm(make_verdicts(0, 10)) == "ok"


def test_improvement_after_flat_avoids_escalate():
    wd = SwarmWatchdog(flat_rounds=3)
    wd.observe_swarm(make_verdicts(5, 10))  # 0.5
    wd.observe_swarm(make_verdicts(5, 10))  # 0.5
    assert wd.observe_swarm(make_verdicts(9, 10)) == "ok"  # 0.9 improves


def test_reset():
    wd = SwarmWatchdog(flat_rounds=2)
    wd.observe_swarm(make_verdicts(5, 10))
    wd.observe_swarm(make_verdicts(5, 10))
    wd.observe_swarm(make_verdicts(5, 10))  # escalates
    wd.reset()
    assert wd.observe_swarm(make_verdicts(0, 10)) == "ok"
