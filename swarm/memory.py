# Adapted from recall (https://github.com/prayingperceptions/recall) — MIT
"""The swarm-kit COLLECTIVE BRAIN: append-only Markdown memory with writer reputation.

On-disk format (hand-edit freely — the parser is lenient and never crashes):

    ## 2026-10-05 09:14:02 | salience=0.8 | tags: x402, pricing | writer: worker-3
    <the text, possibly multi-line>

    ## 2026-10-05 09:20:11 | salience=0.3 | writer: worker-1
    <another entry; tags omitted, writer still recorded>

Blank line between entries. Every entry records its writer (audit trail).
Any line that doesn't parse as an entry is silently skipped on read.

Recall scoring:
    score = (0.5 * keyword_overlap + 0.3 * salience + 0.2 * recency)
            * (0.25 + 0.75 * pass_rate(writer))
  - keyword_overlap: fraction of query words (len >= 3, lowercased) found in
    the entry text + tags.
  - recency: 0.5 ** (age_days / half_life_days), default half-life 7 days.
  - pass_rate(writer): tracked via _note_outcome() through learn_from_round();
    unseen writers get 1.0 (benefit of the doubt), so a writer with 0/10
    passes is down-weighted to 0.25x relative to a fresh writer.
  - Writer attribution is harness-stamped: workers see a WorkerMemory proxy
    whose remember() stamps their own handle (no forgeable writer field),
    and reputation voting is not exposed to workers at all.
  - writer_stats persist in a <name>.reputation.json sidecar, so restarts
    don't forgive burned writers.

Thread-safe: every public method holds a re-entrant lock, so the swarm can
remember/recall from many threads.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# A header line starts with "## " followed by a timestamp. The segments after
# the timestamp (salience=, tags:, writer:) are parsed order-independently;
# anything unrecognized is ignored.
_HEADER_RE = re.compile(r"^##\s*(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def _parse_ts(ts_str: str) -> datetime | None:
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(ts_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


class Memory:
    """Append-only collective memory backed by a single Markdown file.

    Writer attribution is HARNESS-STAMPED, not self-reported: workers never
    touch this class directly. Each worker gets a :class:`WorkerMemory`
    proxy via its :class:`~swarm.worker.Context`; the proxy stamps the
    worker's own identity handle on every entry it writes, and it does not
    expose reputation voting at all. Reputation is earned through review
    verdicts (``learn_from_round``), never voted.
    """

    def __init__(self, path: str | Path = "swarm_memory.md", half_life_days: float = 7.0):
        self.path = Path(path)
        self.half_life_days = half_life_days
        # writer -> [passes, total]; tracked via _note_outcome(), persisted
        # to a sidecar so a restart doesn't forgive burned writers (audit M1).
        self.writer_stats: dict[str, list[int]] = {}
        self._lock = threading.RLock()
        self._load_reputation()

    # ------------------------------------------------- reputation persistence

    def _reputation_path(self) -> Path:
        return self.path.with_suffix(".reputation.json")

    def _load_reputation(self) -> None:
        try:
            import json

            raw = self._reputation_path().read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                self.writer_stats = {
                    str(w): [int(s[0]), int(s[1])]
                    for w, s in data.items()
                    if isinstance(s, (list, tuple)) and len(s) == 2
                }
        except (OSError, ValueError):
            self.writer_stats = {}

    def _save_reputation(self) -> None:
        try:
            import json

            self._reputation_path().write_text(
                json.dumps(self.writer_stats, indent=1), encoding="utf-8"
            )
        except OSError:
            pass

    # ------------------------------------------------------------------ write

    def remember(
        self,
        text: str,
        salience: float = 0.5,
        tags: tuple[str, ...] | list[str] = (),
        writer: str = "",
    ) -> None:
        """Append an entry. Salience is clamped to [0, 1]. Writer is always recorded."""
        with self._lock:
            salience = max(0.0, min(1.0, float(salience)))
            tags = [t.strip() for t in (tags or []) if t and str(t).strip()]
            writer = str(writer or "")
            ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            header = f"## {ts} | salience={salience:g}"
            if tags:
                header += f" | tags: {', '.join(tags)}"
            header += f" | writer: {writer}"
            block = header + "\n" + str(text).rstrip("\n") + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            exists = self.path.exists()
            with self.path.open("a", encoding="utf-8") as f:
                if exists and self.path.stat().st_size > 0:
                    f.write("\n")  # blank line between entries
                f.write(block)

    # ------------------------------------------------------------------ read

    def _load(self) -> list[dict]:
        """Parse the file leniently: skip anything that isn't an entry, never crash."""
        if not self.path.exists():
            return []
        try:
            raw = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        entries: list[dict] = []
        lines = raw.split("\n")
        i = 0
        while i < len(lines):
            m = _HEADER_RE.match(lines[i])
            if not m:
                i += 1
                continue
            ts = _parse_ts(m.group("ts"))
            if ts is None:
                i += 1
                continue
            salience = 0.5
            tags: list[str] = []
            writer = ""
            # Segments after the timestamp, split on "|", order-independent.
            for seg in lines[i][m.end():].split("|"):
                seg = seg.strip()
                low = seg.lower()
                if low.startswith("salience="):
                    try:
                        salience = max(0.0, min(1.0, float(seg.split("=", 1)[1])))
                    except (ValueError, IndexError):
                        salience = 0.5
                elif low.startswith("tags:"):
                    tags = [t.strip() for t in seg.split(":", 1)[1].split(",") if t.strip()]
                elif low.startswith("writer:"):
                    writer = seg.split(":", 1)[1].strip()
            i += 1
            body: list[str] = []
            while i < len(lines) and not _HEADER_RE.match(lines[i]):
                body.append(lines[i])
                i += 1
            entries.append(
                {
                    "text": "\n".join(body).strip(),
                    "salience": salience,
                    "tags": tags,
                    "writer": writer,
                    "ts": ts.isoformat(sep=" ", timespec="seconds"),
                }
            )
        return entries

    # ------------------------------------------------------------------ score

    def _recency(self, ts_iso: str, now: datetime | None = None) -> float:
        ts = _parse_ts(ts_iso)
        if ts is None:
            return 0.0
        now = now or datetime.now()
        age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
        half_life = max(self.half_life_days, 1e-9)
        return 0.5 ** (age_days / half_life)

    def _keyword_overlap(self, query: str, entry: dict) -> float:
        qwords = _words(query)
        if not qwords:
            return 0.0
        hay = _words(entry["text"] + " " + " ".join(entry["tags"]))
        return len(qwords & hay) / len(qwords)

    def pass_rate(self, writer: str) -> float:
        """Fraction of recorded passes for a writer; 1.0 for unseen writers."""
        stats = self.writer_stats.get(writer or "")
        if not stats or stats[1] == 0:
            return 1.0
        return stats[0] / stats[1]

    def score(self, query: str, entry: dict) -> float:
        """score = (0.5*overlap + 0.3*salience + 0.2*recency) * reputation weight."""
        base = (
            0.5 * self._keyword_overlap(query, entry)
            + 0.3 * entry["salience"]
            + 0.2 * self._recency(entry["ts"])
        )
        weight = 0.25 + 0.75 * self.pass_rate(entry.get("writer", ""))
        return base * weight

    def recall(self, query: str, k: int = 5, min_pass_rate: float = 0.0) -> list[dict]:
        """Top-k entries for a query.

        Each dict has text/salience/tags/writer/ts/score/pass_rate.
        ``min_pass_rate`` quarantines writers below a reputation floor
        (audit M2): down-weighting reorders, but a floor excludes.

        Note (audit I3): the file is re-read and re-parsed on every call —
        O(file) per recall. Fine at demo scale; for large memories, front
        with a cache or shard by topic.
        """
        with self._lock:
            entries = self._load()  # read the file on every call: always fresh
            kept = []
            for e in entries:
                pr = self.pass_rate(e.get("writer", ""))
                if pr < min_pass_rate:
                    continue
                e["score"] = self.score(query, e)
                e["pass_rate"] = pr
                kept.append(e)
            kept.sort(key=lambda e: e["score"], reverse=True)
            return kept[: max(0, k)]

    # ------------------------------------------------------- reputation

    def _note_outcome(self, writer: str, verdict: str) -> None:
        """Record one judged outcome. writer_stats[writer] = [passes, total].

        Private on purpose (audit H3): reputation is earned through review
        verdicts via learn_from_round(), never voted by workers. Workers
        reach memory through WorkerMemory, which does not expose this.
        """
        with self._lock:
            writer = str(writer or "")
            if not writer:
                return
            stats = self.writer_stats.setdefault(writer, [0, 0])
            if str(verdict).lower() == "pass":
                stats[0] += 1
            stats[1] += 1

    def learn_from_round(self, verdicts: list, round_no: int) -> None:
        """Fold a judged round into the collective brain.

        (a) _note_outcome() for every verdict that names a worker;
        (b) remember() one round summary entry (writer "swarm").
        Reputation is persisted to the sidecar file afterwards.
        """
        with self._lock:
            verdicts = list(verdicts)
            np_ = sum(1 for v in verdicts if str(getattr(v, "verdict", "")) == "pass")
            nf = sum(1 for v in verdicts if str(getattr(v, "verdict", "")) == "fail")
            nr = sum(1 for v in verdicts if str(getattr(v, "verdict", "")) == "retry")
            for v in verdicts:
                worker = str(getattr(v, "worker", "") or "")
                if worker:
                    self._note_outcome(worker, getattr(v, "verdict", ""))
            fail_reasons = Counter(
                str(getattr(v, "reason", "") or "")
                for v in verdicts
                if str(getattr(v, "verdict", "")) == "fail"
                and str(getattr(v, "reason", "") or "").strip()
            )
            text = f"round {round_no}: {np_} pass / {nf} fail / {nr} retry"
            top = [reason for reason, _ in fail_reasons.most_common(3)]
            if top:
                text += " | top failures: " + " // ".join(top)
            self.remember(
                text,
                salience=0.7,
                tags=("round-summary",),
                writer="swarm",
            )
            self._save_reputation()


class WorkerMemory:
    """The per-worker view of the shared :class:`Memory`.

    This is what workers see as ``ctx.memory``. The writer identity is
    STAMPED from the worker's own handle — there is no ``writer`` parameter
    to forge (audit H3a) — and reputation voting is not exposed at all
    (audit H3b). Reads (``recall``/``pass_rate``/``score``) delegate to the
    shared memory; writes go through with the stamped writer.
    """

    def __init__(self, memory: Memory, handle: str):
        self._memory = memory
        self._handle = str(handle)

    @property
    def handle(self) -> str:
        """The identity stamped on every entry this worker writes."""
        return self._handle

    def remember(
        self,
        text: str,
        salience: float = 0.5,
        tags: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Append an entry, attributed to this worker. No writer parameter:
        the handle is stamped by the harness, so poison cannot be signed as
        someone else."""
        self._memory.remember(text, salience=salience, tags=tags, writer=self._handle)

    def recall(self, query: str, k: int = 5, min_pass_rate: float = 0.0) -> list[dict]:
        """Top-k entries for a query (shared view; see :meth:`Memory.recall`)."""
        return self._memory.recall(query, k=k, min_pass_rate=min_pass_rate)

    def pass_rate(self, writer: str) -> float:
        """Reputation of a writer (see :meth:`Memory.pass_rate`)."""
        return self._memory.pass_rate(writer)

    def score(self, query: str, entry: dict) -> float:
        """Reputation-weighted score (see :meth:`Memory.score`)."""
        return self._memory.score(query, entry)
