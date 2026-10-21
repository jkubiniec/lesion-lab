# Measured results: the 100M-step population experiment

Completed October 21, 2026. One evolutionary seed (42), one lesion seed (9043), and three matched branches. The full evolved populations and exact run source are available in the [v0.1.0 release](https://github.com/jkubiniec/lesion-lab/releases/tag/v0.1.0). [Machine-readable summary](results/latest/summary.json), [raw assays and logs](results/latest/), and [protocol](PROTOCOL.md).

## What the longer run achieved

The healthy population completed **99,966,597 actual neural updates**, **12,112,649 complete scored task episodes**, and **771,746 simulator-supplied births** before injury. This exceeds the required minimum of 10 million complete trials. It has 16,384 independent 64-unit controllers, with 4,867 weights each and 79,740,928 weights across the population.

The recorded continuation ran for 14.32 minutes including assays and saves, after a previously computed 20-million-step healthy prefix. Its terminal wrapper finished in 14 minutes 23 seconds. The shared healthy trajectory is counted once below; each branch adds only its own new work. Across the final experiment there are **129,966,509 actual neural updates** and **15,731,484 complete scored episodes** over 130 million scheduled steps, excluding frozen evaluation probes.

| Stage | Additional scheduled steps | Actual neural updates | Scored complete episodes | Births |
| --- | --- | --- | --- | --- |
| Shared healthy trajectory | 100,000,000 | 99,966,597 | 12,112,649 | 771,746 |
| Sham follow-up | 10,000,000 | 9,999,972 | 1,205,918 | 95,526 |
| Designated region follow-up | 10,000,000 | 9,999,980 | 1,206,498 | 94,056 |
| Scattered units follow-up | 10,000,000 | 9,999,960 | 1,206,419 | 94,297 |


A scheduled step visits one resident slot, including skips of dead occupants. An actual update evaluates a living controller. A scored task episode requires that the same individual received the initial cue before its probe. These are agent–environment interactions, not the pair interactions of research BFF. Births are explicit copying/mutation rules; this does not demonstrate spontaneous invention of reproduction.

## Measured effects on the whole sampled population

![Frozen assay performance](docs/figures/assay-performance.svg)

Each assay uses 512 living sampled controllers × 64 held-out episodes = 32,768 probes. Weights are frozen; transient state resets at the beginning of each episode. The held-out seed is 11045. Immediate conditions share IDs, slots and target sequences; final populations can contain different occupants.

| Condition | Time | Overall | Go choice | No-go withholding |
| --- | --- | --- | --- | --- |
| Shared healthy baseline | Before injury | 70.83% | 61.31% | 99.08% |
| Sham | Immediate | 70.83% | 61.31% | 99.08% |
| Sham | After 10M more steps | 71.88% | 62.69% | 99.15% |
| Designated region | Immediate | 68.09% | 58.46% | 96.66% |
| Designated region | After 10M more steps | 71.96% | 62.77% | 99.21% |
| Scattered units | Immediate | 67.67% | 57.83% | 96.84% |
| Scattered units | After 10M more steps | 71.75% | 62.55% | 99.03% |


The designated lesion reduces the whole sampled population from 70.83% to 68.09%; scattered injury reduces it to 67.67%. Only a quarter of individuals are injured, so these whole-population drops dilute the direct within-injured effect.

## What injury changes within affected individuals

![Paired effects among injured sampled identities](docs/figures/paired-injury-effects.svg)

Both interventions select the same 4,096 living individuals and disable eight of 64 recurrent units per affected controller. The designated intervention chooses units within the last 16; the scattered control chooses across all 64. Severity 0.5 therefore means half of the 16-unit region, or 12.5% of an affected controller. Both interventions leave the inherited genome hash unchanged and keep masks separate from weights.

The following changes are percentage points, with descriptive 95% percentile bootstrap intervals from 2,000 resamples of **individual mean changes**. There are 140 injured controllers in the held-out sample. Intervals characterize this one evolved world and mask seed; they do not measure reproducibility across evolutionary runs. No-go and go rows average controller-specific rates, which can differ slightly from differences of pooled rates.

| Injury | Paired injured agents | Overall Δ [95% interval] | Go Δ [95% interval] | No-go Δ [95% interval] |
| --- | --- | --- | --- | --- |
| Designated region | 140 | -10.01 [-11.59, -8.60] | -10.42 [-12.36, -8.65] | -8.11 [-11.15, -5.44] |
| Scattered units | 140 | -11.57 [-13.24, -9.99] | -12.91 [-15.09, -11.04] | -7.54 [-10.97, -4.50] |


The intervention forces disabled activations to zero, removing their recurrent-state and readout contributions. Across injured probes, the designated intervention changes 20.12% of actions and the scattered intervention changes 23.60%. Overall accuracy worsens in 121 of 140 designated-region agents and 119 of 140 scattered-injury agents. Some agents improve or remain unchanged: a mask can disturb a previously incorrect policy as well as a correct one. These are behavioral effects of computational capacity loss, not a measure of felt trauma.

Direct placement comparison, scattered minus designated:

| Metric | Difference in percentage points [95% interval] | Paired agents |
| --- | --- | --- |
| accuracy | -1.56 [-3.55, +0.42] | 140 |
| go_accuracy | -2.49 [-4.70, -0.20] | 140 |
| nogo_accuracy | +0.57 [-3.86, +4.99] | 140 |


The overall placement interval includes zero. Go choice is more disrupted by the sampled scattered placement in this world; this is exploratory, with several outcomes and no independent replication. It does not establish executive localization. The final quarter was designated by index, not discovered as a frontal or executive module.

## Evidence for cue and recurrent-state dependence

![Memory diagnostic controls](docs/figures/cue-controls.svg)

With the go/no-go instruction supplied explicitly, a cue-blind policy can achieve **62.5% expected overall** by withholding correctly on 25% of trials and guessing direction on the 75% go trials. The weaker static left/right baseline is 37.5%, always withholding 25%, and uniform random action 33.3%.

| Diagnostic | Overall | Go choice | No-go withholding | Paired overall Δ [95% interval], pp |
| --- | --- | --- | --- | --- |
| Normal | 70.83% | 61.31% | 99.08% | Reference |
| Cue erased | 61.87% | 49.27% | 99.26% | -8.96 [-9.37, -8.55] |
| Memory reset | 59.89% | 50.73% | 87.07% | -10.94 [-11.77, -10.19] |


Normal go-choice accuracy is above 50% binary chance and falls to approximately 50% with the cue removed or recurrent state reset. Overall normal performance also exceeds the expected cue-blind bound. These comparisons support dependence on the initial cue and retained state under these tasks. Cue erasure and abrupt state reset change the input/state distribution; they do not fully localize the computation, establish broad working-memory capacity, or demonstrate general intelligence. The memory-reset control also disrupts no-go behavior, so its full effect is broader than cue erasure alone.

## Why the follow-up rebound is not repair

![Original injured cohort retention](docs/figures/injured-cohort.svg)

| Injury | Original injured | Original IDs still alive | Retained | Injury-bearing slots replaced at birth |
| --- | --- | --- | --- | --- |
| Designated region | 4096 | 13 | 0.32% | 4083 |
| Scattered units | 4096 | 11 | 0.27% | 4085 |


About 99.7% of the original injured cohort has disappeared by follow-up. Stable-ID checks verify that survivors retain their masks, and cohort losses match recorded injury-bearing-slot replacements. Newborns receive new IDs and healthy masks. A population can return to around 72% accuracy while almost all original injured agents have been replaced. **No individual lesion repair or lifetime weight learning is implemented.** Follow-up differences combine continued evolution, replacement, deaths and changes in population composition.

![Interval performance and evolutionary births](docs/figures/evolution.svg)

The evolution figure uses counter differences for interval accuracy, rather than confusing lifetime cumulative scores with current performance. It stitches the healthy 0→20M prefix to the 20M→100M continuation without counting the shared 20M row twice.

## Relation to BFF and scientific scope

![BFF and neural-model relationship](docs/figures/bff-to-neural.svg)

The supplied apankrat/bff is an ordinary Brainfuck interpreter; research BFF/CuBFF uses interacting self-modifying code/data tapes. This project extends the **experimental question** into a separate, richer neural ecology with explicit reproduction, task behavior, acquired impairment, matched controls and ancestry. These neural controllers did not emerge from BFF programs. The source review and deliberate departures are documented in [RESEARCH.md](RESEARCH.md). [BFF_COMPARISON.md](BFF_COMPARISON.md) compares the whitepaper mechanisms, measurement units, relevance, falsifiable hypotheses, and experiments needed to connect the systems more directly.

This is a reproducible single-world computational lesion pilot. It is not a numerical replication of published CuBFF dynamics, a validated human traumatic brain injury model, evidence of subjective trauma/consciousness, or proof of anatomical frontal-lobe specialization. The next useful experiments are independent evolution and lesion seeds, functional unit mapping, richer held-out tasks, and same-individual plasticity with persistent masks and replacement controlled.

## Resources, files and provenance

The native engine uses one CPU thread. The run records a 512 MiB model-allocation estimate budget, 3 GiB saved-data forecast, 5 GiB free-disk reserve, 1,200 CPU seconds per native process and 1,800 seconds overall. The estimate is not a hard cap on total process RSS. The sandbox prevented lowering process priority; its actual recorded nice increment is zero. All 62 current validation tests pass, including exact native continuation, acquired-mask persistence, genome preservation, frozen-assay identity matching, portable/native inference parity, resource cleanup and release checksums.

The earlier generated run folders and the superseded 20M pilot were deleted at the owner's request, freeing approximately 3.3 GiB; [cleanup-record.json](docs/cleanup-record.json) records removals. The final run retains its healthy prefix metrics, source and parent checkpoint hash. [experiment.json](experiment.json) and [reproduce.py](reproduce.py) regenerate the 100M healthy trajectory from random initialization, its branches, memory controls, tables and figures. The README also gives the 20M+80M continuation recipe.

Full checkpoints contain every resident's weights, state, identity, ancestry, energy and mask; fixed-slot JSON exports are examples chosen without fitness selection. The source/results release includes exact frozen run source. Original analysis provenance hashes identify the unredacted local input records. Public metadata removes workstation paths and compacts repeated resident records; `SHA256SUMS` identifies the actual published downloads. Full resident records remain in the native checkpoints. Native little-endian files and floating-point dynamics do not promise cross-platform bit identity.
