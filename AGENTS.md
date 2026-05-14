# Agent Instructions

This repository may contain local private guidance that is intentionally ignored by git.

Before making architecture, benchmark, tracing, evaluation, or documentation changes:

1. Read `.irys-private/MISSION.md` if it exists.
2. Read `.irys-private/OPERATING_RULES.md` if it exists.
3. Read `.irys-private/GOAL_STATE.md` if it exists.
4. Read any focused private specs in `.irys-private/` that are relevant to the task.

Do not copy private guidance into public docs, commit messages, pull request text, or public examples.

Public-facing files should describe implemented behavior and how to use the harness. They should not reveal private strategy, internal planning state, agent workflow experiments, benchmark playbooks, secrets, or unpublished evaluation notes.

Keep benchmark work reproducible:

- prefer explicit config over hidden defaults;
- keep run state inspectable;
- save structured traces for every meaningful run;
- preserve benchmark isolation unless a benchmark explicitly allows external state or tools;
- verify private artifacts remain ignored before finalizing.

When diagnosing poor benchmark performance, inspect traces and source code before changing prompts or configs. Prefer targeted, measured experiments over broad rewrites.

Default improvement bias: improve the intelligence substrate before adding prompt pressure. Prefer structured intermediate state, deliverable contracts, workbook/artifact schemas, deterministic row-level extraction, task-family digests, calculators, verifiers, routing gates, and diagnosis loops when they address the failure mode.

Harvey-facing changes need regression pressure. For every meaningful change that could affect Harvey LAB behavior, run at least a 10-task Harvey smoke before accepting it: mix targeted tasks for the changed capability with random sample tasks from the 250-task sample. Keep the 120-task split only as a legacy quick comparison, and use the 500-task split as a larger checkpoint band. Compare against the prior baseline, check regressions, and write the design reflection before treating the change as a system improvement.
