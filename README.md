# Irys Harness

Irys Harness is an evaluation harness for running agent backends against benchmark tasks and recording reproducible results.

The public repository will document implemented benchmark support, runner behavior, adapters, scoring, and reproducibility workflows as they are built.

## Current CLI

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

Run the product matter flow over user-provided documents:

```bash
python -m irys_harness product-run --objective "Summarize the key obligations." --path ./matter-docs --matter-id acme-review --chat-id main
```

Product traces can carry conversation history for a matter chat. The history is limited to prior user questions and final answers; retrieval remains scoped to the current objective and active document corpus.

Serve the local product UI:

```bash
python -m irys_harness product-ui --host 127.0.0.1 --port 8765
```

The product UI supports recursive local folder paths, matter/chat trace listing, per-message cost, and matter-level cost totals.

Open and close a local experiment record:

```bash
python -m irys_harness experiment open --baseline traces/baseline --hypothesis "Improve routing" --target routing
python -m irys_harness experiment close --experiment experiments/exp_id.json --run traces/experiment --accept --decision-reason "Score held and token use decreased."
```

List and load Harvey LAB metadata when the sibling repository is available:

```bash
python -m irys_harness list-tasks --benchmark harvey_lab_sample --limit 5
python -m irys_harness run --benchmark harvey_lab_sample --task-id first --trace-dir traces/dev --output-dir outputs/dev
python -m irys_harness run --benchmark harvey_lab_sample --task-id immigration/compare-draft-eb --trace-dir traces/dev --output-dir outputs/dev --live-synthesis
python -m irys_harness prepare-harvey-eval --trace traces/dev/harvey_lab_sample/antitrust-competition/analyze-antitrust-hsr-strategy.json
python -m irys_harness score-harvey --run-id irys/run_id --task-id antitrust-competition/analyze-antitrust-hsr-strategy
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
