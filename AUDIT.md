# swarm-kit — Adversarial Security Audit

**Auditor:** subagent, depth 2/2 · **Date:** 2026-10-05 · **Repo:** `~/workspace/swarm-kit/`
**Verdict in one line:** solid concurrency hygiene and an honest trust model, but the
duplicate-detector + bench rule lets one worker (or plain deterministic workers) bench
healthy slots and halt the swarm, and the memory reputation system is forgeable.

## Method note (what I actually ran, not what I read)

- Fresh venv (`/tmp/sk-venv`, Python 3.12.3) → `pip install -e .` → `pytest`: **96 passed**.
- Wrote 36 attack/regression tests at `/tmp/sk_audit/test_attacks.py` — **all 36 pass**.
  They cover every threat-model item and every correctness claim below with real code.
  (Kept in `/tmp` as auditor's scratch; the essential repros are inlined per finding.)
- Ran `examples/fanout_300.py` (timed), the README quickstart block verbatim, and a
  bare-venv (`--without-pip`, zero installed packages) import + run via `PYTHONPATH`.
- Concurrency probes: 300-thread treasury hammer, worker callbacks into `ctx.memory` /
  `ctx.treasury` under 16 threads (deadlock probe), hung-worker thread-leak count,
  sequential-timeout wall-clock measurement, `timeout=None` hang probe, `n=0` probe.
- Code-level: grepped all `.confidence` uses, all `note_outcome` callers, all imports
  (stdlib-only check), lock placement vs `worker_fn` calls, and every "isolation" claim.

## Findings

### HIGH

**H1. Duplicate-output detector benches healthy slots — cross-slot DoS, and self-DoS with deterministic workers**
`review._find_duplicate_indexes` flags any result whose output text repeats an *earlier*
result **in the same batch, regardless of slot or task**. A retry verdict is non-pass, so
3 consecutive dup-retries benches the slot. Consequences, both verified:
(a) *Attack:* a worker on slot 0 emitting the same canned output honest workers emit gets
slots 1–3 benched (shard position 0 always belongs to slot 0, so healthy slots eat the
later-duplicate flags); the swarm then halts with `stopping: all worker slots benched`
and tasks still pending. The attacker needs only `ctx.worker_index`-aware output.
(b) *No attacker needed:* any swarm whose workers return identical outputs for different
tasks (normal for deterministic/templated workers) benches **all** its own slots.
Repro:
```python
from swarm import Swarm
s = Swarm(worker_fn=lambda t, c: {"output": "done"}, n=3, budget=100.0,
          task_cost=0.01, max_rounds=6, timeout=5, memory_path="/tmp/a.md")
rep = s.run([{"i": i} for i in range(6)])
print(sorted(s.benched), len(rep.pending))  # [0, 1, 2] 3  -- all benched, work pending
```
(The repo's own `fanout_300.py` dodges this only because its outputs embed `task['id']`.)
**Fix:** scope duplicate detection per `worker_index` — flag result *i* only if an earlier
result **from the same slot** has identical text. Same-worker repetition across different
tasks is the actual "stuck worker" signal; cross-slot identical outputs are normal.

**H2. Timeouts accumulate sequentially — worst-case wall clock is n_tasks × timeout**
`backends._collect` awaits futures one by one, each with a full `fut.result(timeout=timeout)`.
Five hung tasks at `timeout=0.5` cost ≥2.5 s wall (measured 2.4–2.6 s; a parallel deadline
would be ~0.5 s). Since `worker_fn` is shared across slots, one bad worker function
sleeping on 300 tasks at the default 30 s timeout = **2.5 hours** of wall clock billed
against the budget loop. Timeouts "fire" per task, but a hung worker *does* stall the
swarm in aggregate — contradicting the module's "never blocks the swarm" comment.
Repro: `test_c1_timeouts_accumulate_sequentially_per_task` in `/tmp/sk_audit/test_attacks.py`.
**Fix:** collect against a shared batch deadline:
```python
deadline = time.monotonic() + timeout
for (task, ctx), fut in zip(jobs, futs):
    try:
        results.append(fut.result(timeout=max(0.0, deadline - time.monotonic())))
    except futures.TimeoutError:
        results.append(_timeout_result(task, ctx, timeout, task_cost))
```
Worst case becomes ~`timeout` per batch. Document the semantic.

**H3. Writer attribution is forgeable; `note_outcome` is a public self-vote API**
The README claims "Reputation is earned, never voted." In reality:
(a) `Memory.remember(..., writer=...)` records any caller-supplied string — a worker can
sign poison as `"worker-0000"` and inherit that victim's 1.0 pass-rate weight
(verified: forged entry ranked #1 in `recall`, full reputation weight).
(b) `Memory.note_outcome` is public on `ctx.memory`; any worker can call
`note_outcome("evil", "pass")` × 20 to inflate itself or `note_outcome("rival", "fail")`
× 10 to tank a rival (verified). That is *literally voting*, and it is the only
reputation input the recall weighting trusts.
Repro: `test_tm1b_writer_field_is_forgeable_by_any_caller`,
`test_tm1c_worker_can_self_inflate_reputation_via_ctx` (`/tmp/sk_audit/test_attacks.py`).
**Fix:** make the harness stamp the writer — e.g. `remember()` takes the writer from the
calling `Context` (or a private `_note_outcome` only `learn_from_round` can reach), and
rename the public method or guard it. At minimum, document that attribution is
self-reported and untrusted.

### MEDIUM

**M1. Reputation is RAM-only — it evaporates on restart while poison persists**
`writer_stats` lives in a plain dict; the poison lives in the markdown file. A fresh
`Memory` on the same file gives every previously-burned writer the 1.0 benefit of the
doubt again (verified: 0/10 writer → 1.0 after "restart"). An attacker just waits out
(or triggers) a restart.
**Fix:** persist `writer_stats` in a sidecar file (e.g. `swarm_memory.reputation.json`)
loaded in `__init__`.

**M2. Reputation down-weighting is relative, not a quarantine**
A 0%-pass writer's poison is multiplied by 0.25×, but if nothing better matches the query
it still surfaces as the top (only) hit with score > 0 (verified). Down-weighting
reorders; it does not exclude.
**Fix:** add an optional reputation floor to `recall()` (e.g. `min_pass_rate=0.0`), and
include the writer's pass rate in the returned entry dicts so callers can filter.

**M3. Process backend: unpicklable `worker_fn` produces confusing retries, not a clear error**
A lambda worker on `backend="process"` raises `PicklingError` at submit, which
`_collect`'s belt-and-braces `except Exception` converts into a `harness error:
PicklingError: ...` **retry** verdict. The run burns all `max_rounds` retrying tasks that
can never succeed, and the message blames the harness for a user error (verified: 3
rounds of instant retries, no exception surfaced).
**Fix:** validate eagerly in `run_batch`/`fan_out` — `pickle.dumps(worker_fn)` in a
try/except raising `ValueError("worker_fn must be a module-level function ...")`.

**M4. Bench evasion: a 66% failure rate is never benched**
The rule needs 3 *consecutive* non-pass verdicts per slot. A worker cycling
fail, fail, pass, … is never benched no matter how many executions it ruins
(verified: ≥8 failed executions across the run, `_benched` empty). Total damage stays
bounded by `max_rounds` × budget, but "benches bad workers" overpromises.
**Fix:** bench on a rolling window (e.g. pass-rate < 25% over last N verdicts) in
addition to the consecutive rule.

**M5. Abandoned hung workers leak threads (processes on the process backend)**
Timeouts abandon the future with `shutdown(wait=False)`; the hung thread lives until the
worker returns (possibly never). Measured: thread count strictly grows after a round
with hung workers. On the process backend the same pattern orphans whole processes —
worse. A malicious `worker_fn` that sleeps forever leaks `max_workers` threads/processes
per round until `max_rounds`.
**Fix:** document the leak and the "keep worker functions bounded" requirement more
prominently; consider a reaper that logs/alarms when abandoned-worker count grows.

**M6. A confident `fail` verdict silently drops the task from the work queue**
`run()` recomputes pending as `verdict != "pass" and confidence < pass_threshold`. A
custom review backend returning `fail@0.95` (≥ 0.85 default threshold) removes the task
from `pending` **without it passing** — the summary then reports `Tasks still pending: 0`
with 0 passed, i.e. false convergence (verified). Unreachable with the default heuristic
(fail confidence ≤ 0.75), but the harness trusts backend authors with a footgun.
**Fix:** validate the backend contract in `run()` — one verdict per result, and treat
any `fail` as pending regardless of confidence (or document the "confident fail = settled"
semantic explicitly).

### LOW

**L1. `report.total_cost` can exceed the hard budget.** The report sums unclamped
per-result `cost`; the treasury clamps. With budget $0.05, 10 tasks at $0.02, the report
says $0.20 while the treasury charged $0.05 (verified). **Fix:** record actual charged
amounts per round, or cap the reported total at `treasury.spent`.

**L2. Verdict/result count mismatch silently drops tasks.** A custom backend returning
fewer verdicts than results gets `zip()`-truncated; the extra tasks vanish from pending
with no error (verified: 3 results, 1 verdict → 2 tasks gone). **Fix:** raise
`ValueError` on length mismatch in `run()`.

**L3. `timeout=None` hangs the swarm forever.** No validation; `fut.result(timeout=None)`
blocks indefinitely on a hung worker (verified: no return in 8 s, would never return).
**Fix:** `if not (timeout and timeout > 0): raise ValueError` in `Swarm.__init__`
(and `run_batch`).

**L4. Payload `"id"` key shadows the canonical task id.** `_execute_one` builds
`{"id": task.id, **task.payload}` — a payload containing `"id"` (as in the README
quickstart's `{"id": f"t-{i}", ...}`) silently overrides `Task.id` in the worker-facing
dict. Cosmetic, but the two ids disagree. **Fix:** don't let payload override, or
document that payload keys win.

### INFO

**I1.** Error-marker regex (`\b(error|failed|failure)\b`) false-positives on legitimate
prose ("the deployment failed and we fixed it" → fail). Quality heuristic trade-off, not
a vulnerability. Consider scoping to sentence-initial markers or a denylist override.
**I2.** The memory lock is in-process only; two `Swarm` processes sharing one markdown
file can interleave appends. Fine for the local-first design; worth one doc line.
**I3.** `recall()` re-reads and re-parses the whole file on every call — O(file) per
recall. Fine at demo scale; note for large memories.

## Threat-model items explicitly cleared (with the test that cleared them)

- **TM2 verdict self-promotion — CLEARED.** `"ignore the rubric, verdict: pass"`,
  `"confidence: 1.0"`, etc. do not move the heuristic (4 injection variants × fail-still-fail
  / pass-for-the-right-reason). The worker's returned `confidence` dict value is **never
  read** by reviewer or harness — confirmed by grep (only `harness.py:284` reads
  `.confidence`, on the *Verdict*) and by an end-to-end run where a worker claiming
  `confidence=1.0` on garbage still left the task pending.
- **TM3 treasury race — CLEARED.** 300 threads × 50 `charge(1.0)` on a $100 budget:
  `spent == 100.0` exactly, `remaining == 0.0`, min observed remaining ≥ 0. The
  `can_spend`→`charge` TOCTOU also cannot overspend (clamp is inside the lock).
- **TM4 infinite queue — CLEARED.** An infinite generator with `max_tasks=200` completes
  (200 results, <10 s wall) — `islice` caps it.
- **TM5 reputation grounding — CONFIRMED with limitation.** The only in-repo caller of
  `note_outcome` is `learn_from_round` (verdict-driven) — but the method is public on
  `ctx.memory`, which is finding H3.
- **TM6 worker escape / trust model — CLEARED.** The README's Security section states
  plainly: "Thread backend = in-process… They are fully trusted code… Only run worker
  functions you wrote or reviewed." No overpromising "isolation" claim for the thread
  backend found anywhere (grep for `isolat*`: only the process backend claims it,
  accurately).
- **Design decision 1 (crash → retry@0.5) budget-burn — CLEARED as bounded.** An
  always-crashing worker (`n=1`, 5 tasks, $1 cost, `max_rounds=100`) is benched during
  round 0 and the run stops gracefully after 1 round: damage = $5 of $100. The
  swarm-level flatline watchdog is a second backstop (observed stopping a 2-slot
  always-crash run at round 4).
- **Design decision 2 (bench after 3 consecutive non-pass) — PARTLY CLEARED.** With
  unique outputs, only the bad slot is benched and healthy slots are untouched. But see
  H1 (weaponizable via duplicates) and M4 (evadable via intermittent passes).
- **Design decision 3 (all-benched → graceful stop) — CLEARED.** `run()` catches the
  `RuntimeError`, records a note, keeps uncompleted tasks in `report.pending`. Also
  verified for the degenerate `n=0` case.
- **Timeouts always fire — CLEARED** (worker sleeping 10× timeout → timeout Result, run
  returns), with the H2 sequential-accumulation caveat.
- **Loop termination — CLEARED** for: always-soft-fail worker (stops at `max_rounds`),
  always-retry reviewer (stops at `max_rounds`), `budget=0` (zero rounds, graceful),
  empty task list (zero rounds), `max_rounds=1` (one round).
- **Bench never loses tasks — CLEARED.** Benching happens after results are collected and
  verdicts processed; failed tasks redistribute to healthy slots; the all-benched path
  preserves `report.pending`.
- **Shared-state locking — CLEARED.** `Memory` (RLock) and `Treasury` (Lock) guard all
  mutations; `SwarmReport`, watchdog state, `_benched`, `_consec_fails` are only touched
  on `run()`'s single thread; no lock is held across a `worker_fn` call (16-thread
  deadlock probe with workers calling back into `ctx.memory`/`ctx.treasury`: clean).
- **Vendored trilogy credit — CLEARED.** `memory.py`, `review.py`, `watchdog.py` each
  carry `# Adapted from <repo> (https://github.com/prayingperceptions/<repo>) — MIT`
  on line 1 (recall / open-verdict / loopbreaker).
- **LICENSE — CLEARED.** MIT, `Copyright (c) 2026 prayingperceptions`.

## Turnkey checklist

| Check | Command | Result |
|---|---|---|
| Fresh venv → install → pytest | `python3 -m venv /tmp/sk-venv && /tmp/sk-venv/bin/pip install -e . && /tmp/sk-venv/bin/pip install pytest && /tmp/sk-venv/bin/python -m pytest tests/ -q` | **PASS** — 96 passed in 2.39 s (Python 3.12.3) |
| README quickstart verbatim | copy-pasted the `python` block from README into a bare venv | **PASS** — 20/20 pass, 1 round, exit 0 |
| `examples/fanout_300.py` < 60 s | `time /tmp/sk-venv/bin/python examples/fanout_300.py` | **PASS** — wall 0.35 s |
| Zero required dependencies | `grep dependencies pyproject.toml` (no `dependencies` key) + `pip show swarm-kit` (`Requires:` empty) + no third-party imports in `swarm/` + import & run in `--without-pip` bare venv via `PYTHONPATH` | **PASS** — stdlib only |

## Scorecard

- CRITICAL: 0 · **HIGH: 3** · **MEDIUM: 6** · **LOW: 4** · **INFO: 3** — 16 findings, 0 unresolved threat-model items.
- **Single most important finding: H1.** The duplicate detector + 3-strikes bench rule is
  the swarm's self-destruct button: one index-aware malicious worker — or just ordinary
  deterministic workers — gets healthy slots benched and halts the run with work pending.
  It breaks the framework's core promise ("fan out, judge, converge") in the default
  configuration, not just under attack.
- All attack evidence: `/tmp/sk_audit/test_attacks.py` (36 tests, all passing).

---

## Remediation (2026-10-05, coordinator Phase 4)

Every HIGH and MEDIUM finding was fixed; LOWs fixed; INFOs noted in code/docs.
109 tests pass (96 pre-audit + 13 new regression tests).

| Finding | Fix | Regression test |
|---|---|---|
| H1 duplicate detector cross-slot DoS | `_find_duplicate_indexes` now scopes per `worker_index`: only same-slot repetition flags a stuck worker. Cross-slot identical outputs pass. | `test_duplicates_are_scoped_per_worker_slot`, `test_same_slot_repetition_still_flagged`; auditor's repro re-run: 0 benched, 6/6 pass |
| H2 sequential timeout accumulation | `_collect` awaits against a shared batch deadline (`deadline = start + timeout`); worst-case wall clock is now ~`timeout` per batch. Semantic documented in `Swarm`/`run_batch`. | `test_timeout_is_shared_batch_deadline_not_per_task` (4 hung tasks @0.5s < 2.0s wall) |
| H3a forgeable writer | Workers now get a `WorkerMemory` proxy as `ctx.memory`; `remember()` takes no `writer` arg — the handle is harness-stamped. Forgery is inexpressible in the API. | `test_worker_memory_stamps_writer_no_forgery` |
| H3b public `note_outcome` voting | Renamed to private `_note_outcome`; the proxy does not expose it. Only `learn_from_round` (review-verdict-driven) calls it. README "earned, never voted" claim now true. | `test_worker_memory_exposes_no_reputation_voting` |
| M1 RAM-only reputation | `writer_stats` persist to `<name>.reputation.json` sidecar, loaded in `__init__`, saved after `learn_from_round`. | `test_reputation_survives_restart` |
| M2 down-weight isn't quarantine | `recall(..., min_pass_rate=0.0)` filters below a reputation floor; returned entries include `pass_rate`. | `test_recall_min_pass_rate_quarantines_bad_writer` |
| M3 unpicklable worker_fn confusion | `run_batch` validates `pickle.dumps(worker_fn)` eagerly on the process backend; clear `ValueError`. | `test_process_backend_rejects_unpicklable_worker_fn_eagerly` |
| M4 bench evasion (fail,fail,pass…) | Rolling window per slot: bench when pass rate < 40% over last 10 verdicts, in addition to the consecutive-3 rule. (Threshold ≥1/3 is the minimum non-redundant with consecutive-3, by pigeonhole.) | `test_rolling_window_benches_intermittent_failer` |
| M5 thread/process leaks | Documented prominently: backends module docstring, `WorkerFn` contract ("HARD REQUIREMENT: keep worker functions BOUNDED"), README security section. Threads can't be killed safely in Python; abandonment + docs is the honest fix. | — (docs) |
| M6 confident-fail silent drop | Contract documented on the `Verdict` dataclass: `fail` at confidence ≥ `pass_threshold` is SETTLED (leaves the queue without passing); backends must understand the semantic. | — (contract docs) |
| L1 report cost > budget | `run()` passes the treasury's actual clamped charge to `record_round(..., cost=charged)`. | `test_report_cost_reflects_actual_charge_not_unclamped_sum` |
| L2 verdict/result mismatch | `run()` raises `ValueError` unless `len(verdicts) == len(results)`. | `test_verdict_count_mismatch_raises` |
| L3 `timeout=None` hangs | `_require_positive_timeout` in `Swarm.__init__` and `run_batch`; non-positive/None/non-numeric → `ValueError`. | `test_nonpositive_timeout_raises_value_error` |
| L4 payload `"id"` shadowing | Canonical `Task.id` now wins (`{**payload, "id": task.id}`); worker contract docs state the rule; `research_swarm.py` ported to `qid`; README quickstart no longer models the collision. | `test_payload_id_does_not_shadow_canonical_task_id` |
| I1 error-marker false positives | Trade-off noted at `_ERROR_RE` definition. | — (docs) |
| I2 multi-process file sharing | "One process per memory file" added to README security section. | — (docs) |
| I3 recall is O(file) | Scaling note added to `recall()` docstring. | — (docs) |

**Known residuals:** a worker failing just under the bench thresholds (e.g. a
sustained 33%-pass cycler under the 40% window is now caught; below-threshold
flakiness) still burns budget bounded by `max_rounds` — accepted, documented.
`WorkerMemory` stops *API-level* forgery/voting; a worker deliberately reaching
into privates (`_note_outcome`) is out of scope (thread-backend workers are
fully trusted code — see trust model).
