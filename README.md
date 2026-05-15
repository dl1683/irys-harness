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
- editable product plan preview that ranks likely first-read documents from the objective and corpus structure before expensive document loading starts.

## Benchmark Performance

The table below summarizes recent local Harvey LAB development snapshots. These are development measurements from the local sibling `harvey-labs` evaluator, not a committed benchmark artifact or an official release claim.

| Run | Scope | Task pass rate | Rubric pass rate |
|---|---:|---:|---:|
| `irys-sample-250-reconciliation-patch-v1` | 250 tasks | 16.0% | 84.43% |
| `irys-reconciliation-patch-30-v1` | 30 tasks | 26.67% | 80.36% |
| `irys-harvey-all-action-source-inventory-v1` | 498 tasks | 3.21% | 73.38% |
| `irys-harvey-all-bankruptcy-sale-motion-v1` | 1,247 tasks | 3.37% | 70.57% |

Interpretation:

- Harvey LAB task-level pass is strict: every criterion on a task must pass.
- Rubric pass rate is the more granular signal during development because Harvey tasks often contain dozens of criteria.
- Raw benchmark outputs, traces, and evaluator artifacts are intentionally ignored by git. Reproduce or refresh numbers locally before publishing new claims.
- Secondary Agent Bench suites are wired through the bridge, but no public aggregate score is claimed yet in this repository.

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
python -m irys_harness agent-bench --agent-bench-root ../agent-bench --benchmark-workers 4
```

Use `--benchmark` repeatedly to target specific suites:

```bash
python -m irys_harness agent-bench --agent-bench-root ../agent-bench --benchmark docfinqa:train --benchmark l_citeeval:test
```

## Product Matter Runner

Run over a local user corpus:

```bash
python -m irys_harness product-run --objective "Summarize the key obligations." --path ./matter-docs --matter-id acme-review --chat-id main
```

Serve the local UI:

```bash
python -m irys_harness product-ui --host 127.0.0.1 --port 8765
```

The product UI supports:

- native local file/folder pickers;
- recursive local folder paths;
- editable first-read plan preview before a run;
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
