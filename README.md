# Irys Harness

Irys Harness is a local-first benchmark and document-reasoning harness for running agent backends, saving reproducible traces, and inspecting why a run succeeded or failed.

The repository currently focuses on two surfaces:

- benchmark execution for Harvey LAB and long-context/document QA suites;
- a local matter runner for testing document-heavy product workflows over user-provided folders.

## Current Status

Implemented:

- trace-first run state with events, documents, chunks, retrieval iterations, evidence packets, artifacts, metrics, and diagnosis fields;
- Harvey LAB task loading, output packaging, scoring integration, batch smoke runs, resumable tracking, and score-diagnosis utilities;
- Agent Bench bridge for long-context/document benchmarks;
- model-call metrics with token and cost accounting;
- local product matter runner with recursive folder ingestion, multi-chat matter history, live workstream events, source review, and per-message/matter cost totals;
- editable product plan preview that ranks likely first-read documents from the objective and corpus structure before expensive document loading starts;
- product UI workstream, source-review, held-back-document, rerun-comparison, and steering surfaces for testing matter workflows over local corpora.

## Benchmark Performance

The strongest current signal is Harvey LAB rubric performance. Recent development samples are in the 80-85% overall rubric-pass range, with several practice-area and task-family slices around 90% or better. The harness is also being checked against adjacent long-context, grounding, citation, extraction, and document-QA suites through the Agent Bench bridge so Harvey progress does not become a one-benchmark artifact.

These are local development measurements from the sibling `harvey-labs` evaluator and Agent Bench bridge. They are not official benchmark submissions; refresh them locally before publishing new external claims.

| Run | Scope | Task pass rate | Rubric pass rate |
|---|---:|---:|---:|
| `irys-sample-250-reconciliation-patch-v1` | 250 tasks | 16.0% | 85.10% |
| `irys-reconciliation-patch-30-v1` | 30 tasks | 26.67% | 80.36% |
| `irys-harvey-all-action-source-inventory-v1` | 498 tasks | 3.21% | 73.38% |
| `irys-harvey-all-bankruptcy-sale-motion-v1` | 1,247 tasks | 3.37% | 70.57% |

Representative high-signal Harvey slices from the 250-task sample:

| Slice | Rubric pass rate |
|---|---:|
| White-collar defense and investigations | 91.35% |
| Corporate governance | 90.72% |
| Employment and labor | 89.80% |
| Structured finance and securitization | 89.60% |
| Funds and asset management | 88.34% |

Representative Agent Bench bridge smokes:

| Suite | Scope | Result |
|---|---:|---:|
| NoLiMa | 10 examples | 10/10 passed |
| FACTS Grounding | 10 examples | 9/10 passed |
| CUAD | 10 examples | 9/10 passed, 94.71% avg score |
| L-CiteEval | 2 examples | 2/2 passed |
| LongBench v2 | 10 examples | 6/10 passed |

Interpretation:

- Harvey LAB task-level pass is strict: every criterion on a task must pass.
- Rubric pass rate is the more granular signal during development because Harvey tasks often contain dozens of criteria.
- Some non-passing Harvey tasks are still useful product signals rather than simple source-understanding failures. Trace review distinguishes missing information from drafting-shape misses, exact-rubric formatting misses, and cases where a product answer intentionally avoids redundant repetition that a benchmark rubric expects.
- The latest 250-task run kept most token volume in the cheap-worker tier while reserving the strongest model for final synthesis; that run averaged about 82% cheap-worker token share and 18% strong-synthesis token share.
- Raw benchmark outputs, traces, and evaluator artifacts are intentionally ignored by git. Reproduce or refresh numbers locally before publishing new claims.
- Secondary Agent Bench suites are wired through the bridge and are used as regression pressure for general long-context behavior. No single aggregate score is claimed yet because the suite mix is intentionally broad and still being calibrated.
- Active product work is focused on the UI and steering layer: local folder selection, plan inspection, live workstream visibility, source review, held-back document audit, cost display, rerun comparison, and user nudges over matter-specific corpora.

## Supported Benchmark Surfaces

Native harness:

- `fixture`
- `harvey_lab_sample`
- `harvey_lab`

Agent Bench bridge defaults:

- `longbench_v2`
- `facts_grounding`
- `docfinqa`
- `hotpotqa`
- `musique`
- `cuad`
- `nolima`
- `mrcr`
- `counting_stars`
- `loong`
- `l_citeeval`
- `fanoutqa`
- `multihop_rag`
- `nocha`
- `locomo`
- `qasa`
- `qmsum`
- `longhealth`
- `repoqa`
- `long_code_arena`
- `financebench`

## Setup

Create a local environment and install the package in editable mode:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Create `.env` from `.env.example` and add local API keys as needed. Do not commit `.env`.

## CLI

Run the built-in fixture benchmark and write a trace:

```bash
python -m irys_harness run --benchmark fixture --task-id smoke --trace-dir traces/dev
```

Inspect and diagnose a trace:

```bash
python -m irys_harness inspect --trace traces/dev/fixture/smoke.json
python -m irys_harness diagnose --trace traces/dev/fixture/smoke.json
python -m irys_harness diagnose-scores --scores ../harvey-labs/results/irys/run_id/scores.json
```

Check the local environment:

```bash
python -m irys_harness doctor
```

Open and close a local experiment record:

```bash
python -m irys_harness experiment open --baseline traces/baseline --hypothesis "Improve routing" --target routing
python -m irys_harness experiment close --experiment experiments/exp_id.json --run traces/experiment --accept --decision-reason "Score held and token use decreased."
```

## Harvey LAB

Harvey LAB support expects the Harvey repository as a local sibling by default:

```text
../harvey-labs
```

List and run tasks:

```bash
python -m irys_harness list-tasks --benchmark harvey_lab_sample --limit 5
python -m irys_harness run --benchmark harvey_lab_sample --task-id first --trace-dir traces/dev --output-dir outputs/dev
python -m irys_harness run --benchmark harvey_lab_sample --task-id immigration/compare-draft-eb --trace-dir traces/dev --output-dir outputs/dev --live-synthesis
```

Prepare and score an evaluator package:

```bash
python -m irys_harness prepare-harvey-eval --trace traces/dev/harvey_lab_sample/antitrust-competition/analyze-antitrust-hsr-strategy.json
python -m irys_harness score-harvey --run-id irys/run_id --task-id antitrust-competition/analyze-antitrust-hsr-strategy
```

Run a resumable Harvey smoke:

```bash
python -m irys_harness harvey-smoke --split sample250 --workers 24 --score-parallel 24 --resume --execute-score
```

## Agent Bench Bridge

Run the configured bridge against the sibling Agent Bench checkout:

```bash
python -m irys_harness agent-bench-smoke --agent-bench-root ../agent-bench --benchmark-workers 4
```

Use `--benchmark` repeatedly to target specific suites:

```bash
python -m irys_harness agent-bench-smoke --agent-bench-root ../agent-bench --benchmark docfinqa:train --benchmark l_citeeval:test
```

## Product Matter Runner

Run over a local user corpus:

```bash
python -m irys_harness product-run --objective "Summarize the key obligations." --path ./matter-docs --matter-id acme-review --chat-id main --worker-source-planning
```

Serve the local UI:

```bash
python -m irys_harness product-ui --host 127.0.0.1 --port 8765
```

The product UI supports:

- native local file/folder pickers;
- recursive local folder paths;
- optional worker source planning that reviews the file inventory before first read and falls back to deterministic path scoring if model planning is unavailable;
- editable first-read plan preview before a run;
- steering-note plan preview before a rerun, so users can inspect changed source focus before applying it;
- first-read audit showing documents that were held back and why;
- live workstream events while the run is active;
- source review and open-question summaries;
- multiple chats per matter;
- conversation history limited to user questions and final answers;
- per-message and matter-level cost display.

## Tests

Run the test suite:

```bash
python -m pytest
```

For a narrower product/UI check:

```bash
python -m pytest tests/test_product.py
```

## Public Repo Hygiene

The repository ignores local secrets, private planning notes, benchmark data, outputs, traces, and experiment artifacts. Keep public docs focused on implemented behavior, commands, and reproducible evaluation surfaces.
