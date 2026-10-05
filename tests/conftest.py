"""Shared test setup for swarm-kit unit tests.

The underscore injection points on Swarm (_memory, _treasury, _review_fn,
_swarm_watchdog) take these fakes so the harness/backends tests run
standalone and deterministically, without touching disk (real Memory writes
a markdown file) or depending on sibling-module internals.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@dataclass
class FakeVerdict:
    id: str
    verdict: str
    confidence: float
    reason: str
    worker: str = ""


@dataclass
class FakeIdentity:
    index: int

    @property
    def handle(self) -> str:
        return f"worker-{self.index:04d}"


class FakeMemory:
    def __init__(self, path="swarm_memory.md", half_life_days=7.0):
        self.path = path
        self.rounds_learned: list = []

    def remember(self, text, salience=0.5, tags=(), writer=""):
        pass

    def recall(self, query, k=5, min_pass_rate=0.0):
        return []

    def note_outcome(self, writer, verdict):
        pass

    def learn_from_round(self, verdicts, round_no):
        self.rounds_learned.append((round_no, list(verdicts)))


class FakeTreasury:
    """Mirrors the real Treasury contract: clamp at zero, never negative."""

    def __init__(self, budget: float):
        self._budget = float(budget)
        self._spent = 0.0

    def charge(self, amount: float) -> float:
        actual = max(0.0, min(float(amount), self._budget - self._spent))
        self._spent += actual
        return actual

    def can_spend(self, amount: float) -> bool:
        return (self._budget - self._spent) >= float(amount)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self._budget - self._spent


class FakeSwarmWatchdog:
    def __init__(self, flat_rounds: int = 3, escalate_after: int | None = None):
        self.flat_rounds = flat_rounds
        self.escalate_after = escalate_after
        self.calls = 0

    def observe_swarm(self, verdicts) -> str:
        self.calls += 1
        if self.escalate_after is not None and self.calls >= self.escalate_after:
            return "escalate"
        return "ok"


def pass_all_review(results, rubric, backend="heuristic"):
    return [
        FakeVerdict(id=r.task_id, verdict="pass", confidence=1.0,
                    reason="fake", worker=r.worker)
        for r in results
    ]


def fail_all_review(results, rubric, backend="heuristic"):
    return [
        FakeVerdict(id=r.task_id, verdict="fail", confidence=0.0,
                    reason="fake", worker=r.worker)
        for r in results
    ]


def make_swarm(**overrides):
    """Build a Swarm with faked collaborators and a trivial worker."""
    from swarm.harness import Swarm

    def worker_fn(task_dict, ctx):
        return {"output": f"done:{task_dict['id']}", "confidence": 0.9}

    kwargs = dict(
        worker_fn=worker_fn,
        n=4,
        budget=100.0,
        task_cost=0.02,
        max_rounds=5,
        timeout=5.0,
        backend="thread",
        _memory=FakeMemory(),
        _treasury=FakeTreasury(100.0),
        _review_fn=pass_all_review,
        _swarm_watchdog=FakeSwarmWatchdog(),
    )
    kwargs.update(overrides)
    return Swarm(**kwargs)


def make_tasks(n, prefix="task"):
    from swarm.worker import Task

    return [Task(id=f"{prefix}-{i:05d}", payload={"i": i}) for i in range(n)]
