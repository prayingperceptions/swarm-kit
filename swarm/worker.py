"""Worker-side data contracts for swarm-kit.

A worker is just a function::

    def work(task_dict: dict, ctx: Context) -> dict:
        return {"output": "...", "confidence": 0.9}

The harness calls ``worker_fn({"id": task.id, **task.payload}, ctx)`` and
expects a dict with an ``"output"`` string and an optional ``"confidence"``
float. Anything else is recorded as a contract violation in the Result —
never a crash.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Task:
    """One unit of work: a stable id plus an opaque payload dict."""

    id: str
    payload: dict


@dataclass
class Context:
    """What a worker receives alongside the task dict.

    ``memory`` is a per-worker :class:`~swarm.memory.WorkerMemory` proxy:
    ``remember()`` stamps the worker's own handle (it takes no ``writer``
    argument) and reputation voting is not exposed. ``treasury`` is the
    shared budget. (Both are ``None`` on the process backend, which ships a
    slim ctx — see ``swarm.backends``.) ``identity`` exposes ``.handle``
    (e.g. ``"worker-0007"``).
    """

    memory: Any = None
    treasury: Any = None
    identity: Any = None
    round: int = 0
    worker_index: int = 0


@dataclass
class Result:
    """Outcome of a single task execution on a single worker slot."""

    task_id: str
    worker: str
    worker_index: int
    output: str
    confidence: float | None
    error: str | None
    cost: float
    duration_s: float


#: A worker is any callable taking (task_dict, ctx) and returning a dict.
#:
#: Contract: return {"output": str, "confidence"?: float}. Raise on failure
#: (the harness converts it to an error Result and the task is retried).
#:
#: ``task_dict`` is ``{**payload, "id": canonical_id}``: the harness assigns
#: every task a unique canonical id (``"task-00000"``, ...) which is also
#: what results, verdicts, and the report use. Do NOT put an ``"id"`` key in
#: your payload — the canonical id overrides it (audit L4). Name your own
#: keys anything else (``"qid"``, ``"question_id"``, ...).
#:
#: HARD REQUIREMENT (audit M5): keep worker functions BOUNDED. Timeouts
#: abandon hung workers but cannot kill them — a worker that sleeps forever
#: leaks its thread (or process) until it returns. No unbounded sleeps,
#: no infinite loops, no blocking on external resources without your own
#: internal timeout.
WorkerFn = Callable[[dict, "Context"], dict]
