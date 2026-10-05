"""SwarmReport: per-round bookkeeping and a human-readable summary."""

from __future__ import annotations

import time


class SwarmReport:
    """Accumulates round results and renders a readable summary."""

    def __init__(self) -> None:
        self.rounds: list[dict] = []
        self.notes: list[str] = []
        # Tasks still not passing when the run loop exited (set by Swarm.run;
        # empty for manually-built reports). List of Task objects.
        self.pending: list = []
        self._t0 = time.time()

    def record_round(self, round_no: int, verdicts, results, cost: float | None = None) -> None:
        """Record one convergence round's verdicts and results.

        ``cost`` is the amount actually charged to the treasury for the
        round; when omitted it falls back to summing per-result costs.
        The harness passes the clamped charge (audit L1) so the report
        never claims more spend than the budget allowed.
        """
        verdicts = list(verdicts)
        results = list(results)
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        per_worker: dict[str, int] = {}
        for r in results:
            per_worker[r.worker] = per_worker.get(r.worker, 0) + 1
        self.rounds.append(
            {
                "round": round_no,
                "verdicts": counts,
                "results": len(results),
                "cost": float(cost) if cost is not None else sum(r.cost for r in results),
                "per_worker": per_worker,
            }
        )

    def note(self, msg: str) -> None:
        """Attach a free-text note (benching, escalation, early stop)."""
        self.notes.append(str(msg))

    @property
    def total_cost(self) -> float:
        """Total task cost charged across all recorded rounds."""
        return sum(r["cost"] for r in self.rounds)

    @property
    def total_results(self) -> int:
        """Total results recorded across all rounds."""
        return sum(r["results"] for r in self.rounds)

    @property
    def total_passed(self) -> int:
        """Total 'pass' verdicts across all rounds."""
        return sum(r["verdicts"].get("pass", 0) for r in self.rounds)

    @property
    def wall_time(self) -> float:
        """Seconds since this report was created."""
        return time.time() - self._t0

    def summary(self) -> str:
        """One readable block: rounds, verdicts, pass rate, cost, workers, notes."""
        lines = ["Swarm run summary", "================="]
        lines.append(f"Rounds run : {len(self.rounds)}")
        lines.append(f"Wall time  : {self.wall_time:.2f}s")

        if not self.rounds:
            lines.append("No rounds executed (no pending tasks, budget exhausted, or max_rounds=0).")
        else:
            lines.append(f"Total tasks executed : {self.total_results}")
            if self.total_results:
                rate = 100.0 * self.total_passed / self.total_results
                lines.append(
                    f"Overall pass rate    : {self.total_passed}/{self.total_results} ({rate:.1f}%)"
                )
            else:
                lines.append("Overall pass rate    : n/a (no results)")
            lines.append(f"Total cost           : ${self.total_cost:.4f}")
            lines.append(f"Tasks still pending  : {len(self.pending)}")
            lines.append("")
            lines.append("Per-round verdicts:")
            for r in self.rounds:
                dist = ", ".join(
                    f"{k}={v}" for k, v in sorted(r["verdicts"].items())
                ) or "no verdicts"
                lines.append(
                    f"  round {r['round']}: {dist} "
                    f"({r['results']} results, ${r['cost']:.4f})"
                )
            lines.append("")
            lines.append("Tasks per worker (all rounds):")
            per_worker: dict[str, int] = {}
            for r in self.rounds:
                for worker, count in r["per_worker"].items():
                    per_worker[worker] = per_worker.get(worker, 0) + count
            if per_worker:
                for worker in sorted(per_worker):
                    lines.append(f"  {worker}: {per_worker[worker]}")
            else:
                lines.append("  (none)")

        if self.notes:
            lines.append("")
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        return "\n".join(lines)
