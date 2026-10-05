#!/usr/bin/env python3
"""research_swarm.py -- a realistic research swarm.

12 sub-questions about opening a second office in Austin, 6 researchers.
Each researcher "researches" from a canned KNOWLEDGE base (plus seeded
noise), writes findings to the shared memory -- attributed to their worker
handle -- and later rounds *recall* what earlier researchers already found
instead of re-deriving it. Two of the questions are near-duplicates on
purpose: the review pass flags the repeat as a possible stuck worker and
dedups it before convergence.

Run from the repo root (``pip install -e .`` also works):

    python examples/research_swarm.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm import Swarm

random.seed(7)

KNOWLEDGE = {
    "commercial-rent": "Prime Austin office rent averages $48/sq ft/yr, ~35% below San Francisco.",
    "talent-pool": "Austin adds ~15k tech workers a year; UT Austin graduates 2,400 CS majors annually.",
    "tax-incentives": "Texas has no corporate or personal income tax; Austin offers 5-year Chapter 380 abatements.",
    "flight-connectivity": "Austin (AUS) has nonstops to 90+ destinations, but no direct transatlantic beyond London.",
    "competitor-presence": "Oracle, Tesla, Samsung and 6,500 startups already operate in metro Austin.",
    "cost-of-living": "Austin housing costs 2.1x the national median, up 40% since 2020.",
    "timezone-overlap": "An Austin office on Central Time gets full overlap with both coasts' business hours.",
    "regulatory-climate": "Texas ranks #1 for business climate; Austin permitting averages 6-9 months.",
    "office-vacancy": "Downtown Austin office vacancy sits at 24%, the highest in a decade.",
    "university-pipeline": "UT Austin, Texas State and ACC feed a 60k-student hiring pipeline.",
    "quality-of-life": "Austin ranks top-10 for live music, trails and food scene; summer heat is the main complaint.",
}

QUESTIONS = [
    ("q01", "commercial-rent", "What does prime office space cost in Austin?"),
    ("q02", "talent-pool", "How deep is the Austin tech talent pool?"),
    ("q03", "tax-incentives", "What tax incentives exist for a new Austin office?"),
    ("q04", "flight-connectivity", "How well connected is Austin airport?"),
    ("q05", "competitor-presence", "Which competitors already operate in Austin?"),
    ("q06", "cost-of-living", "What is the cost of living for relocating staff?"),
    ("q07", "timezone-overlap", "How does Central Time affect coast-to-coast collaboration?"),
    ("q08", "regulatory-climate", "What is the regulatory climate for new businesses?"),
    ("q09", "office-vacancy", "What is the downtown office vacancy rate?"),
    ("q10", "university-pipeline", "What university hiring pipeline exists?"),
    ("q11", "quality-of-life", "What is quality of life like for employees?"),
    # q12 is a near-duplicate of q11 on purpose: the review pass dedups it.
    ("q12", "quality-of-life", "Would employees actually enjoy living in Austin?"),
]

RUBRIC = """MUST: Austin
MUST NOT: preliminary
MUST NOT: maybe"""

ANSWERS = {}  # task id -> latest good answer (last write wins)


def researcher(task, ctx):
    topic = task["topic"]
    qid = task["qid"]
    # Recall first: if a teammate already researched this topic, reuse it
    # instead of re-deriving it. Shared memory compounds across rounds.
    hits = [h for h in ctx.memory.recall(topic, k=5) if topic in h.get("tags", [])]
    if hits and random.random() < 0.90:
        finding, confidence = hits[0]["text"], 0.93
    elif random.random() < 0.80:
        finding, confidence = KNOWLEDGE[topic], 0.95
    else:
        # Noisy take: hedged and low-value -> fails review -> retried, and
        # the retry will likely just recall the finding from memory.
        return {
            "output": f"preliminary: still digging into {topic}, maybe check back",
            "confidence": 0.4,
        }
    # Every memory entry carries its writer's handle, stamped by the
    # harness: attribution is free and unforgable.
    ctx.memory.remember(finding, tags=(topic,))
    ANSWERS[qid] = finding
    return {"output": finding, "confidence": confidence}


def main():
    tasks = [
        {"qid": qid, "topic": topic, "question": question}
        for qid, topic, question in QUESTIONS
    ]
    mem_path = Path("swarm_memory.md")
    if mem_path.exists():
        mem_path.unlink()  # fresh collective brain for the demo
    swarm = Swarm(
        worker_fn=researcher,
        n=6,
        budget=5.0,
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

    print("research swarm: 12 sub-questions x 6 researchers")
    print("brief: should we open a second office in Austin?")
    report = swarm.run(tasks)
    print()
    print(report.summary())

    # Final converged answers, deduped by topic (q11/q12 merge into one).
    seen = {}
    for qid, topic, _question in QUESTIONS:
        if topic not in seen and qid in ANSWERS:
            seen[topic] = ANSWERS[qid]
    dupes = len(QUESTIONS) - len(seen)
    missing = [qid for qid, _, _ in QUESTIONS if qid not in ANSWERS]
    print()
    print(
        f"CONVERGED ANSWERS "
        f"({len(QUESTIONS)} questions -> {len(seen)} unique findings, "
        f"{dupes} duplicate merged)"
    )
    for topic, finding in seen.items():
        print(f"  [{topic}] {finding}")
    if missing:
        print(f"  (no converged answer for: {', '.join(missing)})")
    mem_entries = sum(1 for line in mem_path.read_text().splitlines()
                      if line.startswith("## "))
    print(f"shared memory: {mem_path} ({mem_entries} entries, writers attributed)")


if __name__ == "__main__":
    main()
