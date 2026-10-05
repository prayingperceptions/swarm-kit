#!/usr/bin/env python3
"""fanout_300.py -- THE swarm-kit demo.

300 tasks. 50 workers. One dumb fan-out per round, one smart review pass,
and a convergence loop that retries only the stragglers.

The workers are deliberately flaky (seeded): ~70% nail it, ~20% shrug with
"idk", ~10% crash outright. Watch the pending queue drain as the review
pass separates signal from noise, round after round. Crashed tasks get
retry@0.5 -- a crash is no signal about quality, so the swarm tries again
(possibly on a different worker) instead of giving up.

Run from the repo root (``pip install -e .`` also works):

    python examples/fanout_300.py

Must finish in under 60 seconds.
"""
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm import Swarm

SEED = 20261005
N_TASKS = 300
N_WORKERS = 50

# The heuristic reviewer matches MUST:/MUST NOT: terms literally
# (word boundaries) against the output, so good outputs name the term.
RUBRIC = """MUST: deploy
MUST NOT: idk"""

TOPICS = [
    "supply-chain delays",
    "battery recycling",
    "urban beekeeping",
    "tidal energy",
    "open-source LLMs",
    "micro-mobility",
    "vertical farming",
    "quantum networking",
    "e-bike subsidies",
    "rainwater harvesting",
]

random.seed(SEED)
_state = {"crashes": 0}
_state_lock = threading.Lock()


def worker_fn(task, ctx):
    """Seeded flaky worker: ~70% good / ~20% sloppy / ~10% crash."""
    roll = random.random()
    if roll < 0.10:
        with _state_lock:
            _state["crashes"] += 1
        raise RuntimeError("worker crashed")
    if roll < 0.30:
        return {"output": "idk", "confidence": 0.15}
    return {
        # Unique per task (duplicates are flagged as stuck workers),
        # names the MUST term so the reviewer can pass it.
        "output": f"{task['id']} | {task['topic']}: deploy early, measure everything, iterate weekly.",
        "confidence": 0.95,
    }


def main():
    tasks = [
        {"id": f"task-{i:05d}", "topic": random.choice(TOPICS)}
        for i in range(N_TASKS)
    ]
    budget = 10.0
    mem_path = Path("swarm_memory.md")
    if mem_path.exists():
        mem_path.unlink()  # fresh collective brain for the demo
    swarm = Swarm(
        worker_fn=worker_fn,
        n=N_WORKERS,
        budget=budget,
        task_cost=0.02,
        rubric=RUBRIC,
        max_rounds=5,
        timeout=30,
        pass_threshold=0.85,
        backend="thread",
        strategy="shard",
        stall_ticks=20,
        max_tasks=10000,
        memory_path=str(mem_path),
        reviewer_backend="heuristic",
    )

    print(f"swarm-kit demo: {N_TASKS} tasks x {N_WORKERS} workers (seed {SEED})")
    t0 = time.perf_counter()
    report = swarm.run(tasks)
    dt = time.perf_counter() - t0

    print()
    print(report.summary())
    # The summary above already includes the per-round verdict distribution;
    # note how the fail counts shrink round over round as stragglers retry.
    print()
    passed = report.total_passed
    print(
        f"convergence: {N_TASKS} tasks -> {passed} passed, "
        f"{N_TASKS - passed} still pending after {len(report.rounds)} rounds "
        f"({_state['crashes']} worker crashes absorbed by retries)"
    )
    print(f"wall clock: {dt:.1f}s   total cost: ${report.total_cost:.2f} "
          f"(budget ${budget:.2f})")
    mem_entries = sum(1 for line in mem_path.read_text().splitlines()
                      if line.startswith("## "))
    print(f"shared memory: {mem_path} ({mem_entries} round summaries, writers attributed)")
    print("300 tasks. 50 workers. 0 status meetings.")
    print("Don't orchestrate. Fan out, judge, converge.")


if __name__ == "__main__":
    main()
