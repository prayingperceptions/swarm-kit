"""Tests for swarm.ledger: Treasury."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.ledger import Treasury  # noqa: E402


def test_charge_clamps_at_budget_never_negative():
    t = Treasury(100.0)
    assert t.charge(30.0) == pytest.approx(30.0)
    assert t.charge(90.0) == pytest.approx(70.0)  # clamped to remaining
    assert t.remaining == pytest.approx(0.0)
    assert t.spent == pytest.approx(100.0)
    # Charging on empty budget yields 0 and never goes negative
    assert t.charge(10.0) == pytest.approx(0.0)
    assert t.remaining == pytest.approx(0.0)
    assert t.spent == pytest.approx(100.0)


def test_zero_budget():
    t = Treasury(0.0)
    assert t.charge(5.0) == pytest.approx(0.0)
    assert t.remaining == pytest.approx(0.0)
    assert not t.can_spend(0.1)


def test_negative_budget_rejected():
    with pytest.raises(ValueError):
        Treasury(-1.0)


def test_can_spend_boundary():
    t = Treasury(100.0)
    assert t.can_spend(100.0)
    assert t.can_spend(0.0)
    assert t.can_spend(-5.0)  # non-positive always fits
    assert not t.can_spend(100.01)
    t.charge(40.0)
    assert t.can_spend(60.0)
    assert not t.can_spend(60.01)


def test_negative_charge_rejected():
    t = Treasury(100.0)
    with pytest.raises(ValueError):
        t.charge(-1.0)
    assert t.spent == pytest.approx(0.0)
    assert t.remaining == pytest.approx(100.0)


def _run_race() -> tuple[float, float, float]:
    """300 threads charging 1.0 against a 100.0 budget."""
    t = Treasury(100.0)
    total = 0.0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal total
        got = t.charge(1.0)
        with lock:
            total += got

    threads = [threading.Thread(target=worker) for _ in range(300)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return total, t.remaining, t.spent


def test_concurrency_race_three_runs():
    for _ in range(3):
        total, remaining, spent = _run_race()
        assert total == pytest.approx(100.0), f"total={total}"
        assert remaining == pytest.approx(0.0), f"remaining={remaining}"
        assert spent == pytest.approx(100.0), f"spent={spent}"
        assert remaining >= 0.0
