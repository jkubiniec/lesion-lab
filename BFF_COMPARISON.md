# How Lesion Lab relates to the original BFF research

The original research asks how executable programs can acquire self-reproduction through interaction. This experiment asks how evolved task behavior changes when computational capacity is persistently disabled. The connection is the study of inherited computation under perturbation. The neural experiment changes the substrate, selection process and measured function; it is an extension of that research direction, not a replication of the paper's origin-of-life result.

![Experimental relevance and what each model tests](docs/figures/experimental-relevance.svg)

## Which BFF reference?

The primary paper is Agüera y Arcas and colleagues' [*Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction*, arXiv:2406.19108v1](https://arxiv.org/html/2406.19108v1). Its BFF family puts executable instructions and writable data on the same tape. The central result concerns copying arising from program interactions without an explicit task-fitness objective. [Paper, Section 2](https://arxiv.org/html/2406.19108v1#S2).

The supplied [apankrat/bff repository](https://github.com/apankrat/bff/tree/f9e03ef793f46695b6ad81ed8cfc745a28c66b44) is a conventional Brainfuck interpreter. Its [instruction implementation](https://github.com/apankrat/bff/blob/f9e03ef793f46695b6ad81ed8cfc745a28c66b44/bff.c) separates compiled instructions from its integer data tape; it does not contain the paper's population experiment. The relevant research implementation is [CuBFF, reviewed commit f212e849027c98fcf4b242eccfb5fed435223e23](https://github.com/paradigms-of-intelligence/cubff/tree/f212e849027c98fcf4b242eccfb5fed435223e23). [RESEARCH.md](RESEARCH.md) records the source review. Lesion Lab is new code and does not incorporate source from either repository.

## What changed in the mechanism?

The reference defaults and byte representation are defined in [CuBFF common.h](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common.h); its [scheduler and mutation code](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common_language.h) define shuffled pairing, concatenation and writeback. The [BFF evaluator](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/bff.inc.h) defines the shared-tape instructions. These links pin the reviewed implementation revision; they do not identify an exact paper-era release. The local neural mechanism is in [neural_engine.cpp](neural_engine.cpp).

| Dimension | Reference BFF primordial soup | Lesion Lab neural experiment |
| --- | --- | --- |
| Resident substrate | 131,072 tapes, 64 bytes each | 16,384 separate recurrent controllers |
| Representation | Mutable executable byte strings | Eight inputs, 64 hidden units, three outputs; 4,867 inherited weights per controller |
| Elementary event | Execute a concatenated ordered pair, then split it | Visit one resident slot and, if alive, update its recurrent controller |
| Schedule | Shuffle tapes into disjoint pairs each epoch | Cycle through slots; eight phases form a task episode |
| Copying | Instructions can overwrite another tape with functional code | Simulator copies parental weights at an energy threshold |
| Variation | Self-modification and background byte mutation | Random weight mutation at birth; temporary recurrent dynamics within a lifetime |
| Selection | Persistence and propagation through execution | Explicit task rewards, energy costs, births and replacement |
| Population accounting | Fixed number of tape slots; content is overwritten | Fixed slot capacity; new stable IDs distinguish offspring from former occupants |
| Primary function tested | Functional copying and its emergence | Cue-dependent choice and go/no-go behavior |
| Acquired intervention here | Separate local soup experiment masks execution sites | Persistent zero-state masks disable selected hidden units without editing weights |

The neural model begins with a reproduction mechanism already supplied. Its evolutionary success therefore cannot answer whether a copying mechanism emerged spontaneously. Recurrent weights do not encode BFF instructions, and there is no implemented developmental path from the soup to these controllers.

```mermaid
flowchart TB
  subgraph Reference["Reference BFF research"]
    A[Random executable tapes] --> B[Pair, execute and modify]
    B --> C[Functional copying may arise]
    C --> D[Copying code spreads by overwrite]
  end
  subgraph Neural["Lesion Lab neural experiment"]
    E[Random recurrent weights] --> F[Explicit tasks change energy]
    F --> G[Supplied birth copies and mutates weights]
    G --> H[Freeze common population checkpoint]
    H --> I[Sham / designated / scattered masks]
    I --> J[Paired behavior and identity tracking]
  end
```

## The numerical budgets are different units

The paper's [Figure 5](https://arxiv.org/html/2406.19108v1#S2.F5) reports a transition within 16,000 epochs in 40% of 1,000 runs under its default mutation setting. At 131,072 tapes and one disjoint pairing per epoch, the corresponding budget is a derived **1,048,576,000 pair interactions**:

`16,000 × (131,072 / 2) = 1,048,576,000`.

That is a reported frequency under a specific design, not a guarantee of emergence after a fixed number of interactions. The paper also considers other languages, spatial arrangements and mutation settings; this comparison concerns its default well-mixed BFF experiment. [Paper, Section 2.1 and Figure 5](https://arxiv.org/html/2406.19108v1#S2.SS1).

Our healthy run completed **100,000,000 scheduled slot steps**, **99,966,597 actual RNN updates** and **12,112,649 complete scored tasks**. It then ran another 10,000,000 scheduled steps in each of three branches. A scheduled slot visit, an eight-phase task and a BFF pair execution are different operations with different computational work. These totals do not support a “faster emergence” comparison. See [results/latest/summary.json](results/latest/summary.json) and [PROTOCOL.md](PROTOCOL.md).

## Why the new experiment is useful

It adds a measurable distinction between **what is inherited** and **what an acquired impairment disables**. We can preserve exactly the same weights and identity, change a persistent execution-capacity mask, and test the same tasks immediately. That supports a causal statement about this implemented intervention in this saved system, with a sham to check that the evaluation machinery itself introduces no change.

The task also makes the function more specific than general activity. A controller must retain cue information, choose under a supplied rule and withhold when instructed. Removing the cue or resetting recurrent state tests what its successful responses depend on. The cue-blind baseline matters: using the supplied go signal and guessing a direction can achieve 62.5% expected overall accuracy without remembering the cue.

In this run, the frozen baseline scored **70.83%** overall; cue erasure reduced it to **61.87%**, and resetting recurrent state before the probe reduced it to **59.89%**. Together these results support cue-dependent performance in the tested population. They do not establish general intelligence or a biologically realistic working-memory system. The direct injury comparisons and exact counters are reported in [RESULTS.md](RESULTS.md) and [cue_control_effects.csv](results/latest/cue_control_effects.csv).

Stable IDs resolve a further ambiguity: a population may improve while the original injured individuals disappear. Here, only **13 of 4,096** original designated-region individuals and **11 of 4,096** original scattered-injury individuals remained after follow-up. Their persistent masks were not repaired. The population can change through healthy offspring and replacement, which is a different phenomenon from an injured individual relearning. [Cohort measurements](results/latest/cohort_retention.csv).

## Hypotheses that can fail

| Hypothesis | Supporting observation | Observation that would weaken or disconfirm it |
| --- | --- | --- |
| Successful choices depend on the earlier cue | Paired cue erasure reduces go-choice performance | Performance stays the same under cue erasure across independent runs |
| Acquired masking disrupts the tested behavior | Immediate paired deficits with weights unchanged and sham unchanged | No reliable deficit, or an equal change in sham |
| The designated quarter has special functional importance | A predeclared direct comparison shows a larger effect than size-matched scattered masks across worlds | Similar or larger scattered-mask effects; unit mapping finds no specialization |
| Later improvement occurs within injured individuals | The same IDs improve while masks stay fixed, with replacement controlled | Improvement is explained by new IDs while original injured individuals remain impaired or disappear |
| A reproduction mechanism emerges | Executable programs independently construct functional descendants without a supplied copying operation | Copying occurs only because the simulator's birth rule performs it |

The current neural model can test the first two hypotheses in one world. It has not established designated-region specialization. Its lack of lifetime plasticity and rapid replacement prevent a claim of same-individual repair. Its supplied birth rule does not test spontaneous origins. Bootstrap intervals over sampled individuals quantify within-world assay variation; they are not independent evolutionary replication.

## What would bridge the experiments more directly?

1. **Reproduce the source substrate first.** Match the research instruction semantics, shuffled pairing, population, mutation, execution budget and emergence measurements. Use independent seeds, keep unsuccessful runs and confirm functional descendants. Compare against a documented local departure rather than treating the two schedulers as equivalent.
2. **Add controlled acquired impairment to verified replicators.** Fork a common world, separate inherited code changes from persistent substrate masks, and use fresh-food copying assays. Include partial, shifted and cooperative copying tests; exact whole-window copying alone has limited scope. The paper itself discusses those detection limits in [Section 2.1](https://arxiv.org/html/2406.19108v1#S2.SS1).
3. **Define an explicit genotype-to-controller bridge.** Let an executable genome construct a controller or its weights, and specify how descendants are constructed. Compare simulator-supplied birth against an independently verified program-executed copying mechanism. A coupled system is not evidence of origins merely because both components are present.
4. **Separate robustness, evolution and plasticity.** Compare the same fixed lesion across independent worlds, then add a separately tested lifetime learning mechanism. Preserve injured identities and masks while controlling reproduction so replacement cannot masquerade as recovery.

This makes the relationship testable: the present contribution is a reproducible experiment on functional vulnerability, task dependence and population replacement. A claim about spontaneous reproduction, frontal-lobe injury or consciousness would require additional mechanisms, measurements and validation.
