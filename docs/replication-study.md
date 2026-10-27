# Replicating the injury result across independent worlds

## Research question

Do immediate computational-injury effects recur across independent evolutionary histories, and how much of later population-level improvement coincides with replacement of the original injured cohort?

This is the next validation step for Lesion Lab. It does **not** add individual repair or lifetime learning. The current simulator evolves inherited weights through births; it does not update an injured individual's weights during its lifetime. Consequently, a follow-up population score cannot establish individual recovery.

## What the runner adds

`replication.py` starts each world from a different, explicitly declared random seed. Within each world, it forks the same healthy population into sham, designated-region, and scattered-unit conditions. The two injury placements affect the same original identities. Every world also receives cue-erasure and memory-reset controls.

The analysis gives each evolutionary world equal weight. A world with more sampled agents or more task trials does not become more independent evidence. Percentile-bootstrap intervals resample worlds, and every world's raw effect remains visible. Intervals based on only a few seeds can be unstable; they are not a substitute for additional replication.

The runner refuses an existing output path, records all planned seeds before execution, preserves failures, and withholds the aggregate if any planned world is missing. It also refuses mixed source versions, incorrect assay seeds or counts, unmatched immediate identities or task streams, incorrect step budgets, and inconsistent cohort-retention counts. A world with no sampled injured individuals remains visible as a missing metric; it is not silently dropped to improve a summary.

## Run a small software check

```sh
python3 replication.py --preset smoke --seeds 101 202 303 \
  --out runs/replication-smoke --max-batch-seconds 120
```

The smoke preset uses 64 agents, eight recurrent units, 8,192 scheduled warmup steps, and 2,048 follow-up steps per branch. It requires at least 512 complete scored warmup episodes. It is deliberately small and is labeled `smoke-software-validation`. This run checks execution, identity pairing, units, and output generation. It is **not** evidence that the published 100-million-step result has replicated or that competent memory or recovery has emerged.

Outputs remain inside the new run directory:

| File | Purpose |
| --- | --- |
| `replication.json` | Declared configuration, seeds, source hashes, completion or failure state, runtime and native binary hash. |
| `summary.json` | World-level effects, equal-weight aggregate estimates, bootstrap intervals and interpretation limits. |
| `worlds.csv` | One row per world; scheduled steps, actual neural updates, complete scored episodes, behavioral effects and original-cohort retention. |
| `seed-*/` | Original native checkpoints, matched branches, frozen assays, cue controls and source snapshots. |
| `source/` | Exact orchestration and analysis source used for the batch. |
| `seed-*.jsonl`, `seed-*-*.log` | Native-run and control logs, including errors. |

## Plan a substantive replication before running it

The existing `experiment.json` can be used as the numeric recipe. The seed in that file is deliberately overridden by the declared batch seeds; all other numeric experiment settings are retained. For example:

```sh
python3 replication.py --config experiment.json \
  --seeds 101 202 303 404 505 606 707 808 909 1010 \
  --out runs/replication-100m --max-batch-seconds 19000
```

This is a large, explicitly requested run, not the default. It schedules ten independent 100-million-step warmups and three 10-million-step follow-ups per world, plus assays. The batch retains every checkpoint. Before running, estimate total storage and elapsed time from a smaller pilot; `max_checkpoint_gib` is the per-world forecast used by the existing runner, not a limit on the whole batch. Each new world checks the remaining disk reserve. Native stages retain the existing CPU and allocation-budget controls. The new runner applies the remaining batch wall-clock budget to subprocess timeouts and checks it before analysis and before and after aggregation; analysis may overrun that budget before the next check. The memory estimate is not an operating-system resident-memory cap.

Use fresh seeds that were not selected because they gave favorable results. Fix the seed list, model size, task definition, intervention dose, exposure, compute budget, exclusion rules, and primary contrasts before viewing the outcome. If a run fails its episode minimum, report it and revise a future complete batch prospectively; do not keep only successful worlds or quietly increase one seed's training budget.

## Proposed analysis plan

1. **Check task competence.** Report healthy overall, go, and no-go accuracy, plus cue-erasure and memory-reset effects. The expected cue-blind overall ceiling for the designed task is 62.5%, so weak chance baselines are insufficient to establish cue memory.
2. **Measure immediate causal effects.** For each world, compare the same sampled identities on the same frozen task streams before and immediately after injury. Report population-wide and injured-cohort effects separately. Compare scattered and designated placements within the same original injured cohort.
3. **Use independent worlds for the aggregate.** Report every world, the equal-weight mean, standard deviation, and descriptive world-bootstrap interval. Decide the number of seeds using pilot variability and the effect size worth resolving; ten is a starting design choice, not a guarantee of adequate power.
4. **Describe follow-up populations honestly.** Report each injury branch's follow-up score relative to its immediate score and to the matched sham, alongside the number and fraction of original injured IDs still alive. These are changing-population comparisons. They are not paired tests of individual repair.
5. **Keep null and adverse results.** No source, seed, dose, or task-family changes after looking at the held-out result. Separate any exploratory analyses from the planned contrasts.

## What must change to test repair itself

A later study must introduce an explicit within-individual learning mechanism while keeping each original injury mask fixed. It must compare that mechanism against frozen injured controllers, preserve identity, and distinguish survival from behavioral improvement. Test on held-out task sequences and task families. Include a no-birth/no-replacement condition so improved population scores cannot be explained solely by healthy descendants replacing injured individuals. The current runner supplies replication infrastructure for that future experiment; it does not implement or validate the repair mechanism.

The designated region remains an arbitrary controller partition. Nothing in this replication establishes a frontal-lobe analogue, subjective experience, psychological trauma, or biological tissue repair.
