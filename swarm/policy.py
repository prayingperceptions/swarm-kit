"""Policy: an allow/ask/deny gate on action classes."""

from __future__ import annotations

_DECISIONS = frozenset({"allow", "ask", "deny"})


class NeedsApproval(Exception):
    """Raised when an action class requires approval in autonomous mode."""

    def __init__(self, action_class: str) -> None:
        super().__init__(f"action class {action_class!r} requires approval")
        self.action_class = action_class


class Policy:
    """Gate actions by class: allow, ask, or deny. Fails closed.

    Unknown action classes (not present in ``rules``) are denied. In
    autonomous mode, a rule of ``"ask"`` raises :class:`NeedsApproval`;
    in non-autonomous mode the string ``"ask"`` is returned so the caller
    can surface the approval step itself.
    """

    def __init__(self, rules: dict[str, str], autonomous: bool = True) -> None:
        for action_class, decision in rules.items():
            if decision not in _DECISIONS:
                raise ValueError(
                    f"invalid rule for {action_class!r}: {decision!r} "
                    f"(must be one of {_DECISIONS})"
                )
        self.rules = dict(rules)
        self.autonomous = autonomous

    def check(self, action_class: str) -> str:
        """Return "allow" | "ask" | "deny" for ``action_class``.

        Raises :class:`NeedsApproval` when the rule is "ask" and this policy
        is in autonomous mode.
        """
        decision = self.rules.get(action_class, "deny")  # fail closed
        if decision == "ask" and self.autonomous:
            raise NeedsApproval(action_class)
        return decision
