#!/usr/bin/env python3
"""Run the published experiment and its memory controls with recorded budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import neural_lab

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--figures", type=Path, default=Path("docs/reproduced-figures"))
    args = parser.parse_args()
    configuration = json.loads((ROOT / "experiment.json").read_text())
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error("Choose a new output directory; existing runs are preserved.")
    options = ("population", "hidden", "seed", "decisions", "followup", "minimum_episodes",
               "severity", "exposure", "assay_samples", "assay_trials", "progress_every",
               "max_memory_mib", "max_checkpoint_gib", "minimum_free_gib", "cpu_seconds", "max_run_seconds")
    command = [sys.executable, "-u", str(ROOT / "neural_terminal.py"), "--out", str(out)]
    for name in options:
        command += ["--" + name.replace("_", "-"), str(configuration[name])]
    subprocess.run(command, cwd=ROOT, check=True)
    neural_lab.LIMITS.update(cpu_seconds=configuration["cpu_seconds"],
                            max_memory_mib=configuration["max_memory_mib"],
                            max_checkpoint_gib=configuration["max_checkpoint_gib"])
    for mode in ("erased", "memory-reset"):
        target = out / ("baseline_cue_erased_assay.json" if mode == "erased" else "baseline_memory_reset_assay.json")
        subprocess.run([str(neural_lab.ENGINE), "assay", "--load", str(out / "warmup.bin"),
                        "--samples", str(configuration["assay_samples"]),
                        "--trials", str(configuration["assay_trials"]),
                        "--seed", str(configuration["seed"] + 11003),
                        "--cue-mode", mode, "--max-memory-mib", str(configuration["max_memory_mib"]),
                        "--output", str(target)], check=True, preexec_fn=neural_lab.native_limits)
    subprocess.run([sys.executable, str(ROOT / "analyze_large_run.py"), str(out),
                    "--output", str(args.analysis.resolve()), "--figures", str(args.figures.resolve())],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
