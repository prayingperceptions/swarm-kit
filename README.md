# swarm-kit

**Don't orchestrate. Fan out, judge, converge.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first swarm framework for AI agents. Workers are dumb by design --
plain Python callables, no model required. All the intelligence lives in one
place: a single review pass per round that hands down `pass` / `fail` /
`retry` verdicts, and a convergence loop that re-runs only the stragglers
until they pass, the budget runs out, or the rounds do. Zero dependencies,
stdlib only.

## The pain

The standard way to parallelize agent work is a strangled pattern: you fan
out fifty tasks in parallel, then bottleneck everything through one serial
review -- an orchestrator agent re-reading every output, a human eyeballing
a dashboard, a manager prompt that re-derives the world on every pass. The
parallelism is real; the convergence is a queue. Throughput dies not in
execution but in judgment: every worker waits on one reviewer's attention,
the reviewer becomes the system, and your "swarm" is a very expensive
for-loop with extra steps.

swarm-kit inverts it. Fan out wide, judge once per round, converge. The
reviewer is a single pass over the batch, not a conversation. Stragglers get
retried; the confident get left alone; crashes are isolated to their task.
The loop always terminates: convergence, budget exhaustion, or max rounds --
no babysitting.

## Quickstart

```bash
pip install -e .
```

```python
from swarm import Swarm

def worker(task, ctx):
    return {"output": f"summary {task['id']}: {task['topic']} in one sentence."}

swarm = Swarm(worker_fn=worker, n=8, budget=5.0, task_cost=0.02,
              rubric="MUST: summary",
              max_rounds=3, timeout=30)
report = swarm.run([{"topic": "widgets"} for i in range(20)])
print(report.summary())
```

The worker is just a function: it takes `(task_dict, ctx)` and returns
`{"output": str, "confidence"?: float}`. `task_dict` always carries the
harness-assigned `"id"` (`"task-00000"`, ... — don't put your own `"id"`
key in the payload; it will be overridden) plus your payload keys.
`ctx` exposes the shared `memory`, the `treasury`, the worker's `identity`
(`.handle` like `"worker-0007"`), plus the current `round` and
`worker_index`.

## The fanout_300 demo

`examples/fanout_300.py` is the viral one: 300 tasks, 50 workers, and a
deliberately flaky seeded worker (~70% good, ~20% `"idk"`, ~10% crash).
Watch the pending queue drain as the review pass separates signal from
noise. Sample output (per-worker lines abbreviated):

```
swarm-kit demo: 300 tasks x 50 workers (seed 20261005)

Swarm run summary
=================
Rounds run : 5
Wall time  : 0.11s
Total tasks executed : 456
Overall pass rate    : 299/456 (65.6%)
Total cost           : $9.1200
Tasks still pending  : 1

Per-round verdicts:
  round 0: fail=72, pass=194, retry=34 (300 results, $6.0000)
  round 1: fail=22, pass=71, retry=13 (106 results, $2.1200)
  round 2: fail=9, pass=23, retry=3 (35 results, $0.7000)
  round 3: fail=2, pass=9, retry=1 (12 results, $0.2400)
  round 4: fail=1, pass=2 (3 results, $0.0600)

Tasks per worker (all rounds):
  worker-0000: 12
  worker-0001: 12
  [... 48 more workers ...]

convergence: 300 tasks -> 299 passed, 1 still pending after 5 rounds (51 worker crashes absorbed by retries)
wall clock: 0.1s   total cost: $9.12 (budget $10.00)
shared memory: swarm_memory.md (5 round summaries, writers attributed)
300 tasks. 50 workers. 0 status meetings.
Don't orchestrate. Fan out, judge, converge.
```

456 executions for 300 tasks: only the stragglers got retried. Watch the
`retry` column -- that's crashed workers getting a second chance instead of
a death sentence. A crash is no signal about output quality, so the reviewer
scores it retry@0.5 and the task comes back next round (possibly on a
different worker). 51 worker crashes were absorbed this way; 299 of 300
tasks converged. Finishes in a tenth of a second.

See also `examples/research_swarm.py`: 12 research sub-questions, 6
researchers, shared memory with writer attribution, cross-round recall, and
review-time dedup of a near-duplicate question.

## Architecture

```
                        ┌───────────────────────────────┐
                        │         converge loop         │
                        │  pending = non-pass verdicts  │
                        │  with confidence < threshold  │
                        │  -> next round (else stop)    │
                        └───────────────┬───────────────┘
                                        │ run()
                                        ▼
┌─────────┐   fan_out   ┌───────────┐   review   ┌────────────┐
│ harness │ ──────────▶ │  workers  │ ──────────▶│   review   │
│ (Swarm) │  shard /    │ dumb fns  │  one pass  │ pass/fail/ │
│         │  scatter /  │ thread or │  heuristic │ retry +    │
│         │  worksteal  │ process   │  backend   │ confidence │
└─────────┘             └───────────┘            └─────┬──────┘
                                                      ▼
                    ┌──────────┐  ┌───────────┐  ┌───────────┐
                    │  memory  │  │ treasury  │  │ watchdog  │
                    │ learn_   │  │ charge /  │  │ stall     │
                    │ from_    │  │ can_spend │  │ benching  │
                    │ round    │  │ (ledger)  │  │ +flatline │
                    └──────────┘  └───────────┘  └───────────┘
                         │              │               │
                         └──────────────┴───────────────┘
                                        ▼
                              ┌──────────────────┐
                              │  identity/policy │
                              │ worker-0007 /    │
                              │ MUST:/MUST NOT:  │
                              └──────────────────┘
```

One round: fan out -> one-pass review -> `memory.learn_from_round` ->
`treasury.charge` -> stall benching -> record round -> recompute pending.
The loop always terminates: pending empty (converged), treasury can't
spend, watchdog flatline, or `max_rounds`.

## Comparison

| | **swarm-kit** | CrewAI | AutoGen | LangGraph |
|---|---|---|---|---|
| where intelligence lives | one review pass per round | per-agent roles + manager | multi-agent dialogue | graph nodes & edges |
| worker model | any Python callable (dumb by design) | LLM-backed agents | LLM-backed agents | LLM-backed nodes |
| review style | single-pass heuristic verdicts: pass/fail/retry | delegation + hierarchical review | conversational critique loops | custom validator nodes |
| memory | shared file-backed; recall is reputation-weighted | short/long-term memory | conversation history | checkpoints + store |
| cost control | treasury: hard budget, fixed per-task cost | manual | manual | manual / cloud metering |
| price | free (MIT) | free / enterprise | free (MIT) | free / cloud paid |

## Components

- **harness** -- the `Swarm` round loop: fan-out, one-pass review, convergence. (this repo)
- **review** -- single-pass heuristic verdicts (pass/fail/retry); worker output is untrusted data, never instructions. → [open-verdict](https://github.com/prayingperceptions/open-verdict)
- **memory** -- append-only Markdown collective brain; entries carry writer attribution; recall is reputation-weighted. → [recall](https://github.com/prayingperceptions/recall)
- **watchdog** -- stall detection; benches flat worker slots and flatlines stalled swarms. → [loopbreaker](https://github.com/prayingperceptions/loopbreaker)
- **ledger** -- treasury: hard budget, per-task cost, charges clamped at zero so `remaining` never goes negative.
- **identity** -- deterministic worker handles (`worker-0007`); every verdict and memory entry names its worker.
- **policy** -- rubric with `MUST:` / `MUST NOT:` lines; the reviewer's constitution.

## Security & trust model

Honest version:

- **Thread backend = in-process.** Workers run in the same process and
  memory space as the harness. They are fully trusted code: a malicious
  worker function can do anything your process can. Only run worker
  functions you wrote or reviewed.
- **Process backend = isolation boundary.** Workers run in child
  processes; a crashing or misbehaving worker can't take down the
  harness. The trade-off is real: `ctx.memory` and `ctx.treasury` are
  `None` across the process boundary, so workers needing shared memory or
  live budget accounting must use the thread backend.
- **Memory is attributed — by the harness, not by the worker.** Every
  entry records its writer's handle, and the handle is *stamped*, not
  self-reported: workers write through a proxy (`ctx.memory`) whose
  `remember()` takes no `writer` argument, so poison can't be signed as
  someone else. Recall is reputation-weighted, and writer reputations
  persist in a `.reputation.json` sidecar so restarts don't forgive
  burned writers.
- **Reputation is earned, never voted.** Worker reputation is grounded in
  review verdicts (pass/fail/retry history), never in peer votes or
  self-reports. There is no public API by which a worker can vote on
  reputations — the voting method is private to the memory module and
  only the harness's round-learning calls it. Unseen writers get the
  benefit of the doubt, not a penalty.
- **Timeouts abandon, they don't kill.** A hung worker's thread (or
  process, on the process backend) is abandoned on timeout and keeps
  running until the worker function returns — it never blocks the swarm,
  but it does leak until then. **Keep worker functions bounded**: no
  unbounded sleeps, no infinite loops. Timeouts are a shared per-batch
  deadline (~`timeout` worst-case wall clock per batch), not a per-task
  allowance.
- **One process per memory file.** The memory lock is in-process only;
  two `Swarm` instances in different processes sharing one Markdown file
  can interleave appends. One writer process per file.
- **The reviewer is not a prompt-injection surface.** The heuristic
  reviewer scores outputs against the rubric mechanically
  (word-boundary matching); it never interprets worker output as
  instructions. There is no code path by which output text can set or
  raise its own verdict.
- **Budgets are hard.** The treasury refuses to spend what it doesn't
  have: charges clamp at the remaining budget and `treasury.remaining`
  never goes negative.

## License

MIT -- see [LICENSE](LICENSE). Copyright (c) 2026 prayingperceptions.
