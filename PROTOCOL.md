# Experimental protocol

## Research question and scope

How does a persistent loss of computational capacity change acquired task behavior, and does the location of that loss matter when its size is matched? Which follow-up changes reflect the same injured individuals, and which reflect selection or replacement?

The primary experiment is the recurrent neural population described below. Two separate legacy implementations remain available: a BFF-inspired bytecode soup and a small supervised controller. The neural population did not emerge from either of them. These experiments measure computational function, not subjective trauma, consciousness, anatomical frontal-lobe function or clinical traumatic brain injury.

## Primary neural experiment

Initialize 16,384 resident agents with 64 fully recurrent hidden units, eight inputs and three action outputs. Each has 4,867 inherited weights. Keep genotype, working state, stable individual identity and acquired injury mask distinct. Task performance changes energy; the simulator copies and mutates weights at a reproductive energy threshold. Offspring receive a new identity, empty working state and a healthy mask. Population capacity stays fixed, and a birth can replace another living individual when no dead slot is available.

Reproduction is an explicit simulator rule. There is no gradient training, lifetime weight learning or lesion repair mechanism. Agents interact with an environment and compete through replacement; the scheduled steps are not BFF pair interactions.

### Workload and stopping rule

1. Use seed 42 for this exploratory world. Record source hashes, settings, environment and the exact random state in checkpoints.
2. Run **100 million cumulative healthy scheduled steps**. The executed continuation starts from a healthy 20-million-step checkpoint and requests another 80 million. Record the parent checkpoint SHA-256 and retain prefix metrics/source provenance. A from-scratch 100-million-step command is given in [NEURAL_MODEL.md](NEURAL_MODEL.md).
3. Require **at least 10 million cumulative complete scored episodes** at the healthy checkpoint. If unmet, preserve that checkpoint and stop before the intervention branches. This is a workload gate, not a task-competence criterion.
4. Assay the healthy population with frozen weights. Each assay samples up to 512 living individuals, with 64 complete episodes per individual, using the same assay seed across conditions.
5. Clone the healthy checkpoint into the three conditions below. Measure immediate effects before any additional ecological evolution.
6. Run **10 million additional scheduled steps per branch**, then repeat the frozen assay and count surviving original injured identities.
7. Report actual counters and runtime. Do not treat the requested budgets as evidence that a stage completed. The runner saves checkpoints at completed stages; an interrupted active stage can have partial logs without a usable final snapshot.

| Counter | Operational definition |
| --- | --- |
| Scheduled step / `decisions` | Visit one resident slot; skip it if its occupant is dead |
| `neural_evaluations` | Actual recurrent updates in living individuals |
| `episodes` | Scored probes for an individual that received the episode's initial cue |
| `births` | Supplied reproduction events that copy and mutate inherited weights |
| Original injured survivor | An original injured stable ID still alive, with its persistent mask |

One task has eight phases. Dead slots and individuals born after the cue can reduce completed episodes relative to scheduled work. Assay episodes are additional frozen measurements, not ecological evolution steps. These quantities do not count gradient updates or neuronal firings.

### Task and competence controls

Each episode supplies a binary cue, six delay steps with distractors, and a probe. The probe supplies a normal/reversed rule and a go/no-go signal, but no cue. On go trials, choose left/right from the remembered cue and current rule; on no-go trials, withhold. Go probability is 75%; no-go probability is 25%. Recurrent state resets at the initial cue. An individual born after the cue cannot score that episode's probe.

Expected overall scores are 33.3% for uniform three-action choice, 25% for always withholding and 37.5% for a static direction. A controller that reads the go signal, always withholds on no-go and guesses a direction on go can score **62.5% without cue memory**. Go-only direction chance is 50%. Finite assays have realized task counts, which are retained with the results.

Evaluate the same frozen baseline and task seed with the initial cue erased and with recurrent state reset immediately before the probe. These interventions test dependence on cue information and recurrent dynamics. Report go and no-go accuracy separately. Similar scores under cue erasure weaken evidence for successful cue use; a reset effect alone can also reflect removal of other recurrent activity. None of these controls establishes anatomical localization.

The current schedule does not require memory competence before injury. Interpret the lesion effects relative to the measured baseline and controls. A confirmatory localization study should predeclare a competence criterion and map unit functions before naming a region.

### Matched acquired injury

| Condition | Intervention |
| --- | --- |
| Sham | No additional disabled units |
| Designated region | Disable eight units sampled within the final quarter of hidden units |
| Scattered units | Disable eight units sampled across all hidden units |

The designated region contains 16 of 64 units. Severity is 0.5 of that region; exposure is 0.25 of the living population, with rounded realized counts recorded. Both injury conditions select the same individual cohort with a shared lesion seed and disable equal unit counts. The native option calls the designated condition `executive`; this is an implementation name, not established executive specialization. Equal unit counts do not guarantee equal functional importance or dynamical effects.

An acquired mask clamps selected hidden states to zero, including across episode resets. It prevents those units from transmitting recurrent or output activity without editing inherited weights. A new offspring does not inherit the mask. Immediate genome hashes must agree with the healthy checkpoint.

Environmental tasks are counter keyed by seed, episode and slot, independently of evolutionary randomness. Immediate frozen assays match identities, slots and task trials. Ecological branches can diverge through births and deaths, so follow-up samples can contain different individuals even with the same sampling seed.

### Outcomes and uncertainty

Primary descriptive outcomes are immediate changes in overall, go and no-go accuracy in the directly injured sampled individuals, paired against their own baseline on identical task trials. Also report the population sample before injury and after follow-up, action changes, correct trials gained/lost, births, deaths, and retention of original injured IDs. Keep population composition changes separate from within-individual effects.

Bootstrap intervals resample sampled individual mean differences within this one evolved world. Repeated trials within an individual are not independent evolutionary runs. These intervals do not establish reproducibility across seeds, biological effects or clinical significance. For go/no-go subgroup metrics, an average of individual differences can differ from the pooled-trial accuracy change.

An injured survivor retains its mask and inherited weights. No implemented process repairs the mask or retrains those weights within its lifetime. A shrinking injured cohort and rising population score can occur together through replacement. Use “population follow-up performance” and “injured-cohort retention” rather than an unqualified recovery claim.

### Resource and reproducibility controls

The larger run uses a single native compute thread, a **512 MiB model-allocation estimate budget**, a **3 GiB saved-data forecast budget**, a **5 GiB free-disk reserve**, a **1,200-second CPU limit per native process**, and a **1,800-second overall wall deadline**. File-size limits and disabled core dumps also apply. An allocation estimate is not a guaranteed maximum for total process RSS. The app sandbox rejects lowering process priority; the manifest records the actual nice increment.

Record source snapshots, source/checkpoint hashes, parameters, seeds, realized masks, counter definitions, prefix metrics and final status. Native checkpoints contain the full population, state, masks, ancestry, energy, counters and random state. Same-engine continuation is tested against uninterrupted execution. Native little-endian storage and floating-point dynamics do not imply cross-platform bit identity. Portable JSON examples support inference but are not substitutes for the full population checkpoint.

Publish completed measurements with their code, model assets and checksums. Clearly distinguish original local provenance hashes from checksums of sanitized public metadata. Earlier generated artifacts were deleted at the user's request; [docs/cleanup-record.json](docs/cleanup-record.json) preserves a compact deletion record. Future experiments should use fresh output folders and a recorded retention policy.

## Next neural experiments: proposed, not implemented

1. Repeat evolution with independent seeds and repeat interventions with several lesion-mask seeds. Estimate variation across evolved worlds before selecting a confirmatory sample size.
2. Establish cue-memory competence, inhibition and rule use separately. Map single-unit and distributed ablations before choosing functionally named regions.
3. Measure injury dose and exposure curves, with predeclared task outcomes and matched controls. Preserve negative and incomplete runs.
4. Add a same-individual plasticity mechanism. Keep original injured IDs and masks fixed, then compare injured learning-enabled, injured learning-disabled and intact learning-enabled agents on matched tasks. Control or account for replacement so it cannot masquerade as relearning.
5. Test whether improvements persist on held-out task distributions. Distinguish inherited evolutionary change, lifetime plasticity and replacement in every analysis.

## Separate legacy experiment: BFF-inspired program soup

The supplied [apankrat/bff](https://github.com/apankrat/bff) repository is a conventional Brainfuck interpreter. The self-reproducing computational-life reference is [Computational Life](https://arxiv.org/html/2406.19108v1) and [CuBFF](https://github.com/paradigms-of-intelligence/cubff). [RESEARCH.md](RESEARCH.md) records reviewed commits and semantic departures in the local CPU soup. No code from those repositories was reused.

A soup pair interaction joins two ordered 64-byte tapes, executes the mutable instruction/data tape, separates it and returns the halves to resident slots. Mutation changes bytes before execution. An acquired soup lesion instead masks instruction execution at selected substrate locations without changing bytes.

The legacy random-origin feasibility design uses three million pairs per seed, followed by an independent-food copying assay. Its gate requires an unchanged source to produce exact copies against at least three independent foods and at least one functional copying descendant. Failure is recorded as `copying_not_verified`; it does not prove that shorter, shifted or cooperative replicators are absent. An engineered known replicator is a positive control, not spontaneous emergence.

Once copying is verified, fork into sham, contiguous focal and scattered execution masks with matched disabled-site counts and exposure. Immediate assays use the same sampled slots and fresh-food seed. Measure exact-copy success, repeated successful sources and functional descendants. Activity, changed bytes and entropy are secondary measurements, not reproduction criteria. A successful copy into healthy food does not demonstrate repair of the injured source substrate.

The published GPU research uses a different scale and scheduler. A few million local pairs do not guarantee emergence. Independent origin seeds, multiple later checkpoints, descendant generations and motif/cooperative assays are needed before a stable-emergence claim.

## Separate legacy experiment: supervised controller

`executive.py` implements a 67-weight model with four sensory units, two hand-designed cue-memory units, eight executive hidden units and three motor outputs. Gradient descent trains executive and motor weights; a lossless cue buffer supplies delay retention. Its controller and task distribution are distinct from the evolved recurrent population above.

Freeze a fitted model and compare acquired unit loss on identical held-out trials. The dose sweep removes 0, 2, 4, 6 or 8 units, with 16 sampled masks per condition and dose. Nonexecutive masks span sensory, memory and motor pools. Complete motor loss is explicitly `no_output` and scored as failure, rather than successful withholding. Repeated masks are variations on one fitted model, not independent trained models or biological subjects.

This legacy task balances go/no-go trials, so always withholding scores 50% overall; that baseline must not be transferred to the neural task's 25% no-go distribution. Go-choice chance remains 50%. Its built-in cue buffer does not demonstrate emergent memory, discovery of an unannounced rule reversal or origin from the BFF soup.
