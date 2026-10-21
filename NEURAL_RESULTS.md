# Neural population results

See [RESULTS.md](RESULTS.md) for the larger experiment's measured outcomes and interpretation, [results/latest](results/latest) for the generated tables and summary, and [docs/figures](docs/figures) for the diagrams. [NEURAL_MODEL.md](NEURAL_MODEL.md) describes the model; [PROTOCOL.md](PROTOCOL.md) defines its comparisons and limitations.

The larger design uses 16,384 recurrent agents with 64 hidden units each: 79,740,928 inherited weights across separate controllers. It schedules 100 million cumulative healthy steps and 10 million additional steps per sham, designated-region and scattered-unit branch. A minimum of 10 million complete scored healthy episodes is required before applying injury. Scheduled steps, actual RNN updates and scored episodes are reported separately.

Earlier generated runs were deleted at the user's request to recover disk space. Their compact deletion record is [docs/cleanup-record.json](docs/cleanup-record.json); this page no longer links to their removed checkpoints or reports. The final results document and generated data are the authority for completed measurements; the design budget above is not itself a measured result.

Interpret accuracy against the 62.5% expected cue-blind policy, paired cue controls and go/no-go scores. Injury masks persist in original individuals, but offspring have healthy masks. Population improvement must not be described as individual repair. The neural birth mechanism is supplied explicitly and did not emerge from the BFF soup.
