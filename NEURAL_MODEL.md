# Recurrent neural population

The larger experiment uses **16,384 resident slots**, each initially occupied by a randomly initialized recurrent controller with **64 hidden units**, eight inputs and three action outputs. It extends the project's computational-life experiments with a separate neural substrate, explicit ecological selection, stable identities and acquired injury. Its results are reported in [RESULTS.md](RESULTS.md); [PROTOCOL.md](PROTOCOL.md) defines the comparisons.

Each controller has **4,867 inherited float weights**: input, recurrent and output connections plus biases. The population contains **79,740,928 weights**, occupying 318,963,712 bytes, approximately **304.19 MiB**. These are separate small controllers; the combined parameter count does not describe one jointly trained network. The implementation uses a CPU, a C++17 compiler and Python's standard library, with no pretrained model or external model API.

## State, inheritance and selection

| Component | Meaning | What happens at birth |
| --- | --- | --- |
| Genotype | Inherited neural weights | Copied from the parent with random mutation |
| Working state | Recurrent activity within an eight-step task | Reset to zero |
| Identity | Stable individual ID, parent ID and birth counter | New ID and recorded parent |
| Energy | Task rewards minus activity and reproductive costs | Offspring starts with its own energy |
| Acquired mask | Hidden units persistently forced to zero | Healthy mask; parental injury is not inherited |

The simulator supplies reproduction when energy reaches a threshold. Successful responses earn energy; errors and ongoing activity cost energy. Birth copies weights into another resident slot, preferentially using a dead slot; otherwise it replaces a living individual. Population capacity stays fixed. There is **no gradient training, lifetime weight learning or injury repair rule**. “Evolved model” means selection and mutation across births.

Agents receive tasks from an environment. They affect one another through competition and slot replacement; communication and pairwise social interactions are not implemented. This differs from BFF's direct interaction between two executable tapes.

## Task and useful baselines

An episode consists of an initial binary cue, six delay steps with distractors, and a probe. The probe supplies the normal/reversed response rule and a go/no-go signal. On go trials the target is left or right according to the earlier cue and current rule; on no-go trials the target is withholding. Cue information is absent from the probe input. Memory is reset at the start of each episode.

Go trials occur with probability 75%; no-go trials with probability 25%. Expected overall scores are:

| Policy | Expected overall accuracy |
| --- | ---: |
| Uniform choice among all three actions | 33.3% |
| Always withhold | 25% |
| Always choose one fixed direction | 37.5% |
| Use the supplied go signal, withhold on no-go and guess a direction on go | **62.5%** |

The last policy needs no cue memory. A score above 37.5% alone therefore does not establish successful memory. Report go and no-go accuracy separately, and compare the frozen baseline against controls that erase the cue or reset recurrent state immediately before the probe. Finite assays have realized condition counts that can differ from these expectations.

## Acquired injury

The designated region is the final quarter of hidden units: 16 of 64. Severity 0.5 disables **eight units per affected individual**. Exposure 0.25 selects a quarter of the **living** individuals, with rounded counts recorded. A scattered-unit control disables the same number of hidden units across the whole controller in the same selected cohort. Sham adds no mask. The native option is named `executive`; the interpretation is **designated region**, because executive specialization has not been established.

An injured unit's state is clamped to zero. It sends no recurrent or output activity, while its inherited weights remain stored and unchanged by the intervention. The mask persists for the same individual, including across episode resets. This is a functional computational perturbation, not a validated anatomical frontal-lobe lesion or human traumatic brain injury model. Felt trauma and consciousness are not measured.

Every branch starts from the same healthy checkpoint. Immediate assays match sampled identities, slots and task trials. Follow-up assays sample currently living slots and may include different individuals. Original injured IDs are tracked separately: their disappearance can reflect death or replacement, and a better population score is not evidence that an injured individual healed.

## Larger experiment and counts

The larger design has **100 million cumulative healthy scheduled steps**, followed by **10 million additional scheduled steps in each of three branches**: sham, designated region and scattered units. Before applying these interventions, the runner requires **at least 10 million cumulative complete scored episodes**. If this workload gate fails, it preserves the healthy checkpoint and stops before injury comparisons. The gate establishes a minimum amount of task exposure; it does not establish memory competence or regional specialization.

The run continues a 20-million-step healthy checkpoint for another 80 million steps. A fresh 100-million-step run with the same engine and seed supplies the corresponding from-scratch recipe:

```sh
python3 -u neural_terminal.py --population 16384 --hidden 64 --seed 42 \
  --decisions 100000000 --minimum-episodes 10000000 --followup 10000000 \
  --assay-samples 512 --assay-trials 64 --progress-every 1000000 \
  --max-memory-mib 512 --max-checkpoint-gib 3 --minimum-free-gib 5 \
  --cpu-seconds 1200 --max-run-seconds 1800 \
  --out runs/large-from-scratch-42
```

To continue an existing healthy checkpoint, use `--resume PATH_TO_WARMUP.bin --decisions 80000000` with the remaining settings above and a fresh output folder. The decision argument is **additional** work; the episode gate checks cumulative episodes. The manifest records the parent checkpoint's SHA-256, inherited seed and source hashes. The published result records the actual executed workload and any continuation separately.

| Counter | What it counts |
| --- | --- |
| `decisions` | Scheduled visits to resident slots, including dead slots that are skipped |
| `neural_evaluations` | Actual recurrent-controller updates in living agents |
| `episodes` | Complete-cue probes scored in living individuals |
| `births` | Simulator-supplied copying and mutation events |

An episode has eight scheduled phases; births and deaths can prevent completion. These counters are not interchangeable with one another, gradient updates, neuron firings or BFF pair interactions. Frozen assays are additional measurements and do not advance ecological counters or alter the population.

## Resource bounds and saved models

This larger configuration uses one native compute thread, a **512 MiB model-allocation estimate budget**, a **3 GiB saved-data forecast budget**, a **5 GiB minimum free-disk reserve**, a **1,200-second CPU limit per native process**, and a **1,800-second overall wall-time deadline**. These are explicit larger-run settings; the smaller launcher defaults remain lower. The preallocation estimate is not a guaranteed cap on total process RSS. Output-file size limits and disabled core dumps also apply.

The runner attempts to lower CPU priority. The app sandbox rejects that operation on the reference machine, so the manifest records a nice increment of zero. No machine-wide resource settings are changed. Longer wall time allows more work without increasing the number of simultaneous compute threads.

Stages save checkpoints on completion. An interruption can lose the active stage's unsaved work; completed checkpoints remain usable. Ctrl-C stops the runner's process group. Each new run uses a fresh folder.

The full `NEURAL02` checkpoint stores weights, recurrent state, energy, identities, masks, counters and random-generator state. Native snapshots use the reference machine's little-endian binary representation. Same-engine continuation is tested against uninterrupted execution; cross-platform bit identity is not promised. Portable per-agent JSON supports inspection and Python inference, with small numerical differences possible because of arithmetic precision and `tanh` implementations. A predetermined slot-zero export is an example, not the best performer or a population average.

## Relation to BFF and next experiments

The supplied [apankrat/bff repository](https://github.com/apankrat/bff) is a conventional Brainfuck interpreter. The interview's reproduction reference is [Computational Life](https://arxiv.org/html/2406.19108v1) and [CuBFF](https://github.com/paradigms-of-intelligence/cubff), where mutable executable tapes can produce functional copies. This neural population builds on evolvable computation and controlled perturbation; its supplied birth rule does not demonstrate spontaneous emergence of a copying mechanism. [RESEARCH.md](RESEARCH.md) documents the source distinction and the separate local bytecode implementation.

Next steps are independent evolutionary seeds, multiple lesion seeds, competence checks with cue controls, and systematic unit-ablation mapping before functional labels. A later plasticity experiment could preserve the same injured identities and fixed masks while comparing learning enabled versus disabled. That same-individual learning mechanism is **proposed, not implemented** in this version.
