# Model card: Lesion Lab evolutionary recurrent population

## Identity and intended use

An experimental population of 16,384 independent recurrent controllers, evolved from random weights on a small cue/rule/go task. Each agent has eight inputs, 64 recurrent units and three action readouts, with 4,867 inherited float32 parameters. The population contains 79,740,928 weights in total. It is not one jointly trained 79.7-million-parameter network or a language model.

Use it to inspect digital evolution, matched computational lesions, ancestry and population replacement. It has no validated clinical, psychological or anatomical interpretation. It does not simulate tissue impact, inflammation, subjective suffering or psychological trauma.

## Training and environment

Training means evolutionary selection and mutation across simulator-supplied births. There are no pretrained weights, gradient updates or lifetime weight learning. A single seed (42) supplies the published evolutionary trajectory. The healthy trajectory runs for 100 million scheduled resident visits, followed by three independently forked 10-million-step continuations. `experiment.json` records parameters; `results/latest/summary.json` records measured workloads and outcomes.

Each task lasts eight phases: one binary cue, six delay phases with distractors, and a probe. On the probe, inputs explicitly supply the response rule and go/no-go instruction. Go probability is 75%; no-go probability is 25%. A correct probe gives +8 energy and an incorrect probe gives −4; every update costs 0.03, and responding before the probe costs another 0.05. Agents start at 48 energy. At energy >=72, a parent pays 36 and produces a child with 32, replacing an empty or another occupied slot. Each inherited parameter mutates with probability 0.015 through a bounded triangular perturbation with scale 0.05; weights are clamped to [−4,4]. These are designed ecology rules, not an emerged copying algorithm.

## Intervention

At the common preinjury checkpoint, select 25% of living individuals. Disable eight of each affected controller's 64 hidden units (12.5% of that controller), either within the final-quarter partition or across all hidden units. Severity 0.5 refers to half of the 16-unit designated region, not half of the entire network. The same original individuals receive the two injury conditions. A sham branch receives no masks. Acquired masks force corresponding recurrent activations to zero; they leave inherited weights intact and persist for that individual's life. The native name `executive` means only the designated final-quarter partition; functional specialization has not been demonstrated.

New offspring receive new IDs and healthy masks. There is no lesion repair. A later aggregate score can change because healthy offspring replace injured individuals. See the stable-ID cohort figure before interpreting follow-up performance.

## Evaluation and limits

Frozen assays use 512 living sampled controllers and 64 episodes per controller, with a separate held-out seed (11045). Immediate comparisons verify the same IDs, slots and task targets. Follow-up assays describe the then-living population; they do not automatically compare the same individuals. Agent-resampling intervals describe variation within this one world, not uncertainty across independent evolutionary replications.

Expected overall baselines: uniform random action 33.3%, always withhold 25%, static left/right 37.5%. A cue-blind policy can reach 62.5% by using the supplied go/no-go instruction and guessing go direction. Therefore overall accuracy alone cannot establish memory. Cue-erased and preprobe memory-reset controls are released alongside ordinary assays; interpret their paired changes and go accuracy.

No evidence of consciousness, general intelligence, clinical TBI fidelity, executive localization or same-individual recovery is established. Stronger claims require independent seeds, lesion seeds, task distributions and functional mapping.

## Files and loading

The GitHub Release contains the full healthy baseline and three final populations as `*-population-neural02.bin.gz`. Each includes all weights, identities/ancestry, energy, recurrent state, acquired masks, counters and RNG state. These are the full evolved model artifacts, not merely summary statistics. `SHA256SUMS` and `release_manifest.json` identify their compressed and uncompressed hashes.

NEURAL02 uses the native little-endian scalar layout produced by this C++ implementation. It is tested on the recorded host; other endian/ABI layouts are unsupported. Cross-platform transcendental math and float accumulation may yield divergent long evolutionary trajectories. Frozen run source is included in the source/results archive.

Portable `models/*-agent-slot-0.json` exports contain one predetermined resident from each snapshot, with topology, float32 weights, state, mask and provenance. Slot 0 was chosen before inspecting fitness; it is neither the best agent nor a representative statistical sample, and its lesion status is recorded in the file. `model_io.py` implements dependency-free inference. All-population native checkpoints remain the authoritative evolved models.

## Resource use and provenance

Native computation uses one CPU thread, a conservative model-memory estimate budget of 512 MiB (not a hard RSS guarantee), a 3 GiB saved-data forecast, 5 GiB free-disk reserve, 1,200 CPU seconds per process and a 1,800-second experiment deadline. A minimum 10-million scored-episode check must pass before injury. Checkpoints are written at stage completion; stopping during a stage loses unsaved work.

The source review distinguishes the supplied ordinary Brainfuck interpreter from the research CuBFF system. This neural model builds on the experimental question of reproduction and persistent impairment; it did not evolve out of BFF programs. See [RESEARCH.md](RESEARCH.md), [BFF_COMPARISON.md](BFF_COMPARISON.md), and [README.md](README.md).
