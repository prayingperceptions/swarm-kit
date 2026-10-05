# Adapted from loopbreaker (https://github.com/prayingperceptions/loopbreaker) — MIT
"""Stall detection: single-agent progress stalls + swarm-level verdict stalls."""

from __future__ import annotations


class StallWatcher:
    """Watch an agent loop and fire when progress stalls.

    Feed every loop tick to :meth:`observe` with a short ``action`` label and
    a ``progress`` score (higher = closer to done). The watcher tracks the
    best progress ever seen and counts consecutive observations with *no
    improvement* (``progress <= best``).

    - When the flat-streak reaches exactly ``stall_ticks`` -> return
      ``"nudge"`` **once**: the agent should propose a new tactic.
    - If the streak reaches exactly ``2 * stall_ticks`` with still no
      improvement -> return ``"escalate"`` **once**: ping a human or stop
      gracefully instead of spinning forever.
    - Otherwise return ``"ok"``.
    - Any observation with ``progress`` strictly greater than the best seen
      resets the streak (re-arming nudge and escalate) and records the new
      best.

    Equal scores don't count as progress: only *strictly* better scores move
    the bar, because "trying the same thing again" is exactly the failure
    mode this exists to catch.
    """

    def __init__(self, stall_ticks: int = 20) -> None:
        if stall_ticks < 1:
            raise ValueError("stall_ticks must be >= 1")
        self.stall_ticks = stall_ticks
        self.reset()

    def reset(self) -> None:
        """Clear all state: forget the best progress and restart the streak."""
        self._best: float | None = None  # best progress score seen so far
        self.flat_streak: int = 0  # consecutive observations with no improvement
        self._nudged: bool = False  # True once "nudge" has been returned
        self._escalated: bool = False  # True once "escalate" has been returned

    @property
    def best(self) -> float | None:
        """Best progress score seen since the last reset (None if none)."""
        return self._best

    def observe(self, action: str, progress: float) -> str:
        """Record one loop tick. Returns "ok" | "nudge" | "escalate".

        ``action`` is a free-form label (e.g. "fetch_invoice_pdf") used for
        debugging, not for scoring. ``progress`` is any numeric score where
        higher means closer to done.
        """
        # Improvement: strictly above the best we've ever seen. Reset the
        # flat streak and re-arm the nudge/escalate cycle.
        if self._best is None or progress > self._best:
            self._best = progress
            self.flat_streak = 0
            self._nudged = False
            self._escalated = False
            return "ok"

        # No improvement: progress is flat (or regressed). Count the streak.
        self.flat_streak += 1

        # First full window of flatness -> tell the agent to try a new tactic.
        # Fire exactly once per arming cycle.
        if not self._nudged and self.flat_streak >= self.stall_ticks:
            self._nudged = True
            return "nudge"

        # A second full window with still no improvement -> stop and escalate.
        # Fire exactly once per arming cycle.
        if self._nudged and not self._escalated and self.flat_streak >= 2 * self.stall_ticks:
            self._escalated = True
            return "escalate"

        return "ok"


class SwarmWatchdog:
    """Watch a swarm's per-round verdict pass rates and fire on stagnation.

    Call :meth:`observe_swarm` once per convergence round with the round's
    verdicts. A round's pass rate is the fraction of verdicts whose
    ``verdict`` field equals ``"pass"``. If the best pass rate seen over the
    last ``flat_rounds`` rounds is no better than the best pass rate seen
    before that window — i.e. no improvement for ``flat_rounds`` consecutive
    rounds — the swarm is stalled and we return ``"escalate"``. Otherwise
    return ``"ok"``. The first call always returns ``"ok"``.
    """

    def __init__(self, flat_rounds: int = 3) -> None:
        if flat_rounds < 1:
            raise ValueError("flat_rounds must be >= 1")
        self.flat_rounds = flat_rounds
        self._history: list[float] = []

    def reset(self) -> None:
        """Clear the pass-rate history."""
        self._history = []

    @property
    def history(self) -> list[float]:
        """Copy of the recorded per-round pass rates."""
        return list(self._history)

    def observe_swarm(self, verdicts) -> str:
        """Record one swarm round. Returns "ok" | "escalate".

        ``verdicts`` is an iterable of objects with a ``verdict`` attribute
        (a ``Verdict`` has ``id``, ``verdict`` in {"pass","fail","retry"},
        ``confidence``, ``reason``, ``worker``). An empty list yields a
        pass rate of 0.0 and returns "ok".
        """
        if not verdicts:
            self._history.append(0.0)
            return "ok"

        passes = sum(1 for v in verdicts if getattr(v, "verdict", None) == "pass")
        pass_rate = passes / len(verdicts)
        self._history.append(pass_rate)

        if len(self._history) == 1:
            return "ok"

        # Recent window: last flat_rounds rounds. Baseline: everything before.
        recent = self._history[-self.flat_rounds :]
        baseline = self._history[: -self.flat_rounds]

        if not baseline:
            return "ok"

        if max(recent) <= max(baseline):
            return "escalate"
        return "ok"
