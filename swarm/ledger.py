"""Treasury: a thread-safe spend budget with hard clamping at zero."""

from __future__ import annotations

import threading


class Treasury:
    """Track spending against a fixed budget. Thread-safe.

    All state mutations are guarded by a single lock that is never held
    across callbacks (there are none — charge/can_spend do pure arithmetic
    under the lock). ``remaining`` can never go negative: charges are
    clamped to the remaining budget.
    """

    def __init__(self, budget: float) -> None:
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self._lock = threading.Lock()
        self._budget = float(budget)
        self._spent = 0.0

    @property
    def budget(self) -> float:
        """The original budget."""
        return self._budget

    @property
    def spent(self) -> float:
        """Total charged so far."""
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        """Budget left. Never negative."""
        with self._lock:
            return self._budget - self._spent

    def charge(self, amount: float) -> float:
        """Charge ``amount`` against the budget; returns the actual charged.

        Charges are clamped at the remaining budget: if ``amount`` exceeds
        what is left, only the remainder is charged and returned. Never
        returns a negative value.
        """
        if amount < 0:
            raise ValueError("charge amount must be >= 0")
        with self._lock:
            actual = min(amount, self._budget - self._spent)
            # Guard the invariant even if float rounding misbehaves.
            if actual < 0:
                actual = 0.0
            self._spent += actual
            return actual

    def can_spend(self, amount: float) -> bool:
        """True if ``amount`` fits within the remaining budget.

        Non-positive amounts always fit.
        """
        if amount <= 0:
            return True
        with self._lock:
            return self._budget - self._spent >= amount
