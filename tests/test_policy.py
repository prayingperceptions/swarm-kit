"""Tests for swarm.policy: Policy + NeedsApproval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.policy import NeedsApproval, Policy  # noqa: E402


def test_allow_deny_per_rules():
    p = Policy({"read_state": "allow", "delete_everything": "deny"})
    assert p.check("read_state") == "allow"
    assert p.check("delete_everything") == "deny"


def test_unknown_action_fails_closed():
    p = Policy({"read_state": "allow"})
    assert p.check("launch_nukes") == "deny"
    assert Policy({}).check("anything") == "deny"


def test_ask_autonomous_raises():
    p = Policy({"spend_money": "ask"}, autonomous=True)
    with pytest.raises(NeedsApproval) as exc_info:
        p.check("spend_money")
    assert exc_info.value.action_class == "spend_money"


def test_ask_non_autonomous_returns_string():
    p = Policy({"spend_money": "ask"}, autonomous=False)
    assert p.check("spend_money") == "ask"


def test_bad_rule_value_rejected_at_construction():
    with pytest.raises(ValueError):
        Policy({"read_state": "maybe"})
    with pytest.raises(ValueError):
        Policy({"read_state": "ALLOW"})
    with pytest.raises(ValueError):
        Policy({"read_state": ""})


def test_rules_are_copied():
    rules = {"a": "allow"}
    p = Policy(rules)
    rules["a"] = "deny"
    assert p.check("a") == "allow"
