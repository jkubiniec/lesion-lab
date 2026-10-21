# Source review

Reviewed October 21, 2026. External text and source code were treated as research evidence. The implementation here is new code, not a modified copy of either linked repository.

## The interview

Read the full unique transcript of [The theory that connects the origin of life to the rise of intelligence, with Blaise Agüera y Arcas](https://bigthink.com/series/full-interview/agi-life-consciousness/). The page repeats the transcript; repeated text is not a second interview.

The relevant mechanism is repeated interaction between short programs with shared code/data. A program that writes functional copies can persist through subsequent interactions. The interview explicitly describes damage that breaks copying. Its later discussion of functionalism and consciousness supplies philosophical context, not an experimental demonstration of consciousness. It distinguishes basic copying from later extensions with additional behaviors.

The interview's small-tape narrative differs from the fixed 64-byte setup documented in the paper. This project uses the latter as its concrete starting point.

## The supplied GitHub repository

Reviewed [apankrat/bff commit f9e03ef793f46695b6ad81ed8cfc745a28c66b44](https://github.com/apankrat/bff/tree/f9e03ef793f46695b6ad81ed8cfc745a28c66b44).

Its [README](https://github.com/apankrat/bff/blob/f9e03ef793f46695b6ad81ed8cfc745a28c66b44/README.md) describes an optimizing Brainfuck interpreter. Its [C source](https://github.com/apankrat/bff/blob/f9e03ef793f46695b6ad81ed8cfc745a28c66b44/bff.c) compiles instructions into an array separate from the growing integer data tape. It implements ordinary input/output. There is no self-modifying instruction tape, organism population, stochastic interaction schedule, copying assay, or mutation process. GitHub identifies it as an original repository, rather than a fork.

Consequently, extending it directly would require adding nearly the whole experiment framework and changing its execution model. It is useful context for a minimal interpreter, but is not the BFF soup described in the interview. Its source header mentions BSD licensing; no code from it was reused here.

## The research substrate

Primary source: [Agüera y Arcas et al., Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction, arXiv:2406.19108v1](https://arxiv.org/html/2406.19108v1).

Reviewed [CuBFF commit f212e849027c98fcf4b242eccfb5fed435223e23](https://github.com/paradigms-of-intelligence/cubff/tree/f212e849027c98fcf4b242eccfb5fed435223e23), including [bff.inc.h](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/bff.inc.h), [common_language.h](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common_language.h), and [common.h](https://github.com/paradigms-of-intelligence/cubff/blob/f212e849027c98fcf4b242eccfb5fed435223e23/common.h). The repository supplies an Apache 2.0 license. The paper also links experiments in [benlaurie/bff-ben](https://github.com/benlaurie/bff-ben/tree/paper1).

The research BFF instruction set has ten byte-valued instructions. Data and code occupy the same tape:

| Instruction | Operation |
| --- | --- |
| `<`, `>` | Move head 0 left/right |
| `{`, `}` | Move head 1 left/right |
| `+`, `-` | Increment/decrement byte under head 0, modulo 256 |
| `.` | Copy byte under head 0 to head 1 |
| `,` | Copy byte under head 1 to head 0 |
| `[`, `]` | Conditional nested loops based on the byte under head 0 |

All remaining byte values are no-ops. Heads wrap; the instruction pointer does not. Matching brackets are found from the current, potentially modified tape. An unmatched required bracket ends execution. In the `noheads` variant both heads and instruction pointer start at zero. Another research variant uses the first two bytes to initialize the heads.

Research defaults use 131,072 programs of 64 bytes. Each epoch shuffles and pairs them; an ordered pair is concatenated, mutated, executed once, split, and returned to its original slots. Programs overwrite other programs: there is no explicit birth operation or growing population. The default mutation probability is 1/4,096 per byte.

The paper's [Figure 5](https://arxiv.org/html/2406.19108v1#S2.F5) reports a transition in 40% of 1,000 default runs within 16,000 epochs. At that population, 16,000 epochs contain approximately 1.05 billion pair interactions. A few million interactions in a smaller soup is a pilot, not an assurance of emergence.

## Positive control and assay scope

The sanitized palindrome from [Figure 4](https://arxiv.org/html/2406.19108v1#S2.F4) is reconstructed as:

```python
"[[{.>]-]" + " " * 48 + "]-]>.{[["
```

It is 64 bytes. Tests check that it preserves itself and copies exactly into fresh random food, and that the copied descendant can do the same. Inserting it is an engineered positive control and must never be described as spontaneous emergence.

The paper explicitly notes that exact whole-window tests can miss shorter, shifted, and cooperative replicators. Our gate prioritizes an unambiguous copying criterion; failure to pass this criterion is not a universal negative result about replication.

## Intentional departures

- CPU implementation and a smaller default population, so these are not numerical replications of the published GPU experiments.
- Independent ordered pairs sampled with replacement, rather than one random permutation per epoch.
- Bracket scanning consumes execution budget. This imposes a clear bound on all VM work, but can alter selection compared with counting only fetched instructions.
- Persistent slot-local execution masks model acquired substrate impairment, separately from inherited code mutation.
- An independently trained modular controller supplies named functional tasks; it is not a phenotype that emerged from the soup.

Keep these differences in every interpretation. Comparable source semantics alone do not make the dynamics interchangeable.
