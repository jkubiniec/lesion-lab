#!/usr/bin/env python3
"""Replicate matched lesion experiments across independent evolutionary worlds.

Standard library only. Smoke runs verify the pipeline; they do not demonstrate
memory competence, recovery, or a research finding. See docs/replication-study.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
from statistics import mean, stdev

import analyze_large_run as analysis
import neural_lab
from neural_terminal import stop_process_group

ROOT = Path(__file__).resolve().parent
OPTIONS = ("population", "hidden", "decisions", "followup", "minimum_episodes",
           "severity", "exposure", "assay_samples", "assay_trials", "progress_every",
           "max_memory_mib", "max_checkpoint_gib", "minimum_free_gib", "cpu_seconds",
           "max_run_seconds")
SMOKE = dict(population=64, hidden=8, decisions=8192, followup=2048,
             minimum_episodes=512, severity=0.5, exposure=0.25,
             assay_samples=64, assay_trials=16, progress_every=8192,
             max_memory_mib=64, max_checkpoint_gib=0.5, minimum_free_gib=1,
             cpu_seconds=20, max_run_seconds=45)
SOURCE_FILES = ("neural_engine.cpp", "neural_lab.py", "neural_terminal.py",
                "analyze_large_run.py", "replication.py")


def validate_configuration(configuration: dict, seeds: list[int]) -> dict:
    if len(seeds) < 3 or len(seeds) > 50:
        raise ValueError("Use 3..50 independently specified evolution seeds.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not independent evolutionary replications.")
    if any(type(seed) is not int or not 0 <= seed <= 2**63 - 11004 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers below 2**63 - 11003.")
    missing = set(OPTIONS) - configuration.keys()
    if missing:
        raise ValueError("Missing configuration fields: " + ", ".join(sorted(missing)))
    config = {key: configuration[key] for key in OPTIONS}
    floats = {"severity", "exposure", "max_checkpoint_gib", "minimum_free_gib"}
    for key, value in config.items():
        if key in floats:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
        elif type(value) is not int:
            raise ValueError(f"{key} must be an integer.")
    bounds = {"population": (2, 262144), "hidden": (4, 256),
              "decisions": (0, 2**63-1), "followup": (0, 2**63-1),
              "minimum_episodes": (0, 2**63-1), "severity": (0, 1), "exposure": (0, 1),
              "assay_samples": (1, 100000), "assay_trials": (1, 100000),
              "progress_every": (1, 2**63-1), "max_memory_mib": (32, 1024),
              "max_checkpoint_gib": (0.000001, 8), "minimum_free_gib": (1, 100),
              "cpu_seconds": (1, 1800), "max_run_seconds": (1, 1800)}
    for key, (low, high) in bounds.items():
        if not low <= config[key] <= high:
            raise ValueError(f"{key} must be in {low}..{high}.")
    if config["assay_samples"] * config["assay_trials"] > 10000000:
        raise ValueError("assay_samples * assay_trials must not exceed 10,000,000.")
    return config


def source_hashes() -> dict:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def world_interval(values: list[float], repeats: int = 4000, seed: int = 7919) -> dict:
    """Every input value is one world's effect; never pool agents across worlds."""
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("World effects must be a nonempty list of finite values.")
    if not 100 <= repeats <= 10000:
        raise ValueError("Use 100..10,000 bootstrap replicates.")
    n = len(values)
    result = {"n_worlds": n, "mean": mean(values),
              "standard_deviation": stdev(values) if n > 1 else None,
              "minimum": min(values), "maximum": max(values),
              "ci95_low": None, "ci95_high": None,
              "resampling_unit": "independent_evolutionary_world",
              "world_weighting": "equal", "bootstrap_repeats": repeats,
              "interval_warning": "Small-seed percentile intervals are descriptive and can be unstable."}
    if n >= 3:
        rng = random.Random(seed)
        draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n
                 for _ in range(repeats)]
        result["ci95_low"] = analysis.percentile(draws, 0.025)
        result["ci95_high"] = analysis.percentile(draws, 0.975)
    return result


def paired_mean(before: dict, after: dict, injured_only: bool) -> tuple:
    """Point estimate only: uncertainty is computed across worlds, not here."""
    if any(before.get(key) != after.get(key) for key in ("seed", "trials")):
        raise ValueError("Cannot pair assays with unmatched task seeds or trial counts.")
    previous = {row["id"]: row for row in before["per_sample"]}
    deltas = []
    for row in after["per_sample"]:
        if injured_only and not (row.get("injured") or row.get("disabled_count", 0)):
            continue
        old = previous.get(row["id"])
        if old is None:
            raise ValueError("Cannot pair an identity absent from the preceding assay.")
        for key in ("slot", "episodes", "go_trials", "nogo_trials", "target_per_trial", "go_per_trial"):
            if old.get(key) != row.get(key):
                raise ValueError(f"Cannot pair unmatched task streams or slots ({key}).")
        before_rate = analysis.safe_rate(old, "accuracy")
        after_rate = analysis.safe_rate(row, "accuracy")
        if before_rate is not None and after_rate is not None:
            deltas.append(after_rate - before_rate)
    return (mean(deltas) if deltas else None), len(deltas)


def rate_difference(after: dict, before: dict):
    rates = [analysis.safe_rate(record, "accuracy") for record in (after, before)]
    return rates[0] - rates[1] if all(value is not None for value in rates) else None


def validate_assay(assay: dict, expected_seed: int, config: dict, cue_mode: str = "normal") -> None:
    if (assay.get("seed") != expected_seed + 11003 or
            assay.get("trials") != config["assay_trials"] or
            assay.get("frozen_weights") is not True or assay.get("cue_mode") != cue_mode):
        raise ValueError("Assay does not match its declared task seed, trial count, frozen state, or cue mode.")
    expected_samples = min(config["assay_samples"], assay["alive_pool_count"])
    if assay.get("samples") != expected_samples or len(assay.get("per_sample", [])) != expected_samples:
        raise ValueError("Assay sample count differs from the declared alive sampling frame.")
    if len({row["id"] for row in assay["per_sample"]}) != expected_samples:
        raise ValueError("An assay contains duplicate identities.")


def read_world(folder: Path, expected_seed: int, config: dict, expected_hashes: dict) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"World {expected_seed} is incomplete; it cannot enter the aggregate.")
    if manifest.get("seed") != expected_seed or manifest.get("resume_from") is not None:
        raise ValueError("Each replication must start independently from its specified seed.")
    for key in ("population", "hidden", "severity", "exposure"):
        if manifest.get(key) != config[key]:
            raise ValueError(f"World {expected_seed} has mismatched {key}.")
    for key in ("neural_engine.cpp", "neural_lab.py", "neural_terminal.py"):
        if manifest.get("source_sha256", {}).get(key) != expected_hashes[key]:
            raise ValueError(f"World {expected_seed} used mismatched source: {key}.")
    warmup = manifest["warmup_final"]
    if warmup["decisions"] != config["decisions"] or warmup["episodes"] < config["minimum_episodes"]:
        raise ValueError("World did not meet its predeclared scheduled-step/episode budgets.")
    baseline = manifest["baseline_assay"]
    validate_assay(baseline, expected_seed, config)
    arms = manifest["arms"]
    if set(arms) != {"sham", "designated_region", "random"}:
        raise ValueError("All three matched branches are required.")
    baseline_ids = [(row["id"], row["slot"]) for row in baseline["per_sample"]]
    measures = {"baseline_" + metric: analysis.safe_rate(baseline, metric)
                for metric in analysis.METRICS}
    counts = {"warmup_scheduled_steps": warmup["decisions"],
              "warmup_neural_updates": warmup["neural_evaluations"],
              "warmup_complete_scored_episodes": warmup["episodes"],
              "baseline_alive": baseline["alive_pool_count"]}
    if arms["designated_region"]["immediate"]["injured_ids"] != arms["random"]["immediate"]["injured_ids"]:
        raise ValueError("Lesion placements must affect the same original identities.")
    for name, arm in arms.items():
        validate_assay(arm["immediate_assay"], expected_seed, config)
        validate_assay(arm["final_assay"], expected_seed, config)
        if [(row["id"], row["slot"]) for row in arm["immediate_assay"]["per_sample"]] != baseline_ids:
            raise ValueError("Immediate assays must sample the same identities and task slots.")
        delta, paired = paired_mean(baseline, arm["immediate_assay"], False)
        if paired != len(baseline_ids):
            raise ValueError("Every immediate sample must be paired with its healthy counterpart.")
        measures[name + "_immediate_population_delta"] = delta
        measures[name + "_followup_population_delta_from_immediate"] = rate_difference(
            arm["final_assay"], arm["immediate_assay"])
        counts[name + "_followup_alive"] = arm["final_assay"]["alive_pool_count"]
        counts[name + "_sampled_population"] = paired
        counts[name + "_followup_scheduled_steps"] = arm["final"]["decisions"] - arm["immediate"]["decisions"]
        counts[name + "_followup_neural_updates"] = arm["final"]["neural_evaluations"] - arm["immediate"]["neural_evaluations"]
        counts[name + "_followup_complete_scored_episodes"] = arm["final"]["episodes"] - arm["immediate"]["episodes"]
        if counts[name + "_followup_scheduled_steps"] != config["followup"]:
            raise ValueError("Follow-up branch has the wrong scheduled-step budget.")
        if name == "sham":
            if delta not in (None, 0) or arm["immediate"]["injured_ids"]:
                raise ValueError("Sham must preserve the healthy immediate assay and have no injured IDs.")
            continue
        injured_delta, injured_samples = paired_mean(baseline, arm["immediate_assay"], True)
        measures[name + "_immediate_injured_delta"] = injured_delta
        counts[name + "_sampled_injured"] = injured_samples
        original = set(arm["immediate"]["injured_ids"])
        survivors = sum(row["id"] in original and bool(row["alive"])
                        for row in arm["final"]["individuals"])
        if any(row["id"] in original and row["alive"] and not row["injured"]
               for row in arm["final"]["individuals"]):
            raise ValueError("An original injured survivor has lost its persistent mask.")
        if survivors != arm["final"]["injured_alive"]:
            raise ValueError("Original-cohort identity count disagrees with the recorded retention count.")
        counts[name + "_original_injured"] = len(original)
        counts[name + "_original_injured_alive"] = survivors
        measures[name + "_original_injured_retention"] = survivors / len(original) if original else None
        measures[name + "_followup_population_minus_sham"] = rate_difference(
            arm["final_assay"], arms["sham"]["final_assay"])
    measures["scattered_minus_designated_injured_delta"], counts["placement_paired_injured"] = paired_mean(
        arms["designated_region"]["immediate_assay"], arms["random"]["immediate_assay"], True)
    for mode, filename in (("cue_erased", "baseline_cue_erased_assay.json"),
                           ("memory_reset", "baseline_memory_reset_assay.json")):
        control = json.loads((folder / filename).read_text())
        validate_assay(control, expected_seed, config, "erased" if mode == "cue_erased" else "memory-reset")
        if [(row["id"], row["slot"]) for row in control["per_sample"]] != baseline_ids:
            raise ValueError("Cue controls must sample every baseline identity in the same task slot.")
        measures[mode + "_minus_baseline"], counts[mode + "_paired_individuals"] = paired_mean(baseline, control, False)
        if counts[mode + "_paired_individuals"] != len(baseline_ids):
            raise ValueError("Every cue-control sample must pair with its healthy counterpart.")
        for metric in analysis.METRICS:
            measures[mode + "_" + metric] = analysis.safe_rate(control, metric)
    return {"seed": expected_seed, "measures": measures, "counts": counts,
            "manifest_sha256": neural_lab.sha256_file(folder / "manifest.json")}


def aggregate(worlds: list[dict], planned_seeds: list[int], repeats: int = 4000) -> dict:
    actual = [world["seed"] for world in worlds]
    if len(set(actual)) != len(actual) or set(actual) != set(planned_seeds):
        raise ValueError("Aggregate requires exactly one completed world for every planned seed.")
    metrics = set(worlds[0]["measures"])
    if any(set(world["measures"]) != metrics for world in worlds):
        raise ValueError("All worlds must report the same metrics.")
    summaries = {}
    for name in sorted(metrics):
        values = [world["measures"][name] for world in worlds]
        missing = [world["seed"] for world in worlds if world["measures"][name] is None]
        if missing:
            summaries[name] = {"n_worlds": len(values) - len(missing), "missing_seeds": missing,
                               "mean": None, "ci95_low": None, "ci95_high": None,
                               "status": "not_aggregated_missing_world_metric"}
        else:
            summaries[name] = {"status": "complete", **world_interval(values, repeats)}
    return {"schema_version": 1, "planned_seeds": planned_seeds, "n_worlds": len(worlds),
            "inference_unit": "independent_evolutionary_world",
            "within_world_task_trials_are_not_independent_worlds": True,
            "repair_or_lifetime_learning_implemented": False,
            "followup_interpretation": "Changing populations with healthy offspring; not individual repair.",
            "metrics": summaries, "worlds": worlds}


def run_checked(command: list[str], log: Path, timeout: float, *, native: bool = False) -> None:
    if timeout <= 0:
        raise TimeoutError("Batch wall-clock budget exhausted.")
    with log.open("x") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True,
                                   preexec_fn=neural_lab.native_limits if native else None)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Command exited with {code}; see {log.name}.")
        except BaseException:
            stop_process_group(process)
            raise


def execute(out: Path, seeds: list[int], configuration: dict, label: str,
            max_batch_seconds: int, repeats: int) -> dict:
    config = validate_configuration(configuration, seeds)
    if max_batch_seconds < 1 or not 100 <= repeats <= 10000:
        raise ValueError("Batch seconds must be positive; use 100..10,000 bootstrap repeats.")
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hashes = source_hashes()
    record = {"schema_version": 1, "status": "running", "label": label,
              "configuration": config, "planned_seeds": seeds, "completed_seeds": [],
              "source_sha256": hashes, "max_batch_seconds": max_batch_seconds,
              "bootstrap_repeats": repeats, "python": sys.version, "platform": platform.platform(),
              "statistical_unit": "independent_evolutionary_world",
              "no_lifetime_repair": True, "small_run_warning": "Smoke output verifies software only."}
    neural_lab.save_json(out / "replication.json", record)
    (out / "source").mkdir()
    for name in SOURCE_FILES:
        (out / "source" / name).write_bytes((ROOT / name).read_bytes())
    remaining = lambda: max_batch_seconds - (time.monotonic() - started)
    worlds = []
    active_seed = None
    try:
        for active_seed in seeds:
            if source_hashes() != hashes:
                raise ValueError("Source files changed during the batch; refusing mixed-code results.")
            folder = out / f"seed-{active_seed}"
            print(f"World {active_seed}: {config['decisions']:,} scheduled warmup steps; "
                  f"{config['followup']:,} follow-up steps per branch.", flush=True)
            command = [sys.executable, "-u", str(ROOT / "neural_lab.py"), "run", "--out", str(folder), "--seed", str(active_seed)]
            for key in OPTIONS:
                command += ["--" + key.replace("_", "-"), str(config[key])]
            run_checked(command, out / f"seed-{active_seed}.jsonl",
                        min(config["max_run_seconds"] + 10, remaining()))
            neural_lab.LIMITS.update(cpu_seconds=config["cpu_seconds"],
                                    max_memory_mib=config["max_memory_mib"],
                                    max_checkpoint_gib=config["max_checkpoint_gib"])
            for mode, filename in (("erased", "baseline_cue_erased_assay.json"),
                                   ("memory-reset", "baseline_memory_reset_assay.json")):
                control = [str(neural_lab.ENGINE), "assay", "--load", str(folder / "warmup.bin"),
                           "--samples", str(config["assay_samples"]), "--trials", str(config["assay_trials"]),
                           "--seed", str(active_seed + 11003), "--cue-mode", mode,
                           "--max-memory-mib", str(config["max_memory_mib"]), "--output", str(folder / filename)]
                run_checked(control, out / f"seed-{active_seed}-{mode}.log",
                            min(config["cpu_seconds"] + 10, remaining()), native=True)
            if remaining() <= 0:
                raise TimeoutError("Batch wall-clock budget exhausted before world analysis.")
            worlds.append(read_world(folder, active_seed, config, hashes))
            record["completed_seeds"].append(active_seed)
            record["elapsed_seconds"] = time.monotonic() - started
            neural_lab.save_json(out / "replication.json", record)
            print(f"World {active_seed} complete: {worlds[-1]['counts']['warmup_complete_scored_episodes']:,} "
                  "complete scored episodes. No repair mechanism is implemented.", flush=True)
        if remaining() <= 0:
            raise TimeoutError("Batch wall-clock budget exhausted before aggregation.")
        summary = aggregate(worlds, seeds, repeats)
        if remaining() <= 0:
            raise TimeoutError("Batch wall-clock budget exhausted during aggregation.")
        summary["label"] = label
        summary["source_sha256"] = hashes
        neural_lab.save_json(out / "summary.json", summary)
        with (out / "worlds.csv").open("w", newline="") as stream:
            rows = [{"seed": world["seed"], **world["counts"], **world["measures"]} for world in worlds]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        record["status"] = "complete"
        record["native_binary_sha256"] = neural_lab.sha256_file(neural_lab.ENGINE)
        record["elapsed_seconds"] = time.monotonic() - started
        neural_lab.save_json(out / "replication.json", record)
        return summary
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      failed_seed=active_seed, error=str(error), elapsed_seconds=time.monotonic() - started)
        neural_lab.save_json(out / "replication.json", record)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=("smoke",))
    source.add_argument("--config", type=Path, help="A JSON recipe with the experiment.json numeric fields.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-batch-seconds", type=int, default=300)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    try:
        config = SMOKE if args.preset else json.loads(args.config.read_text())
        label = "smoke-software-validation" if args.preset else "configured-replication"
        summary = execute(args.out, args.seeds, config, label, args.max_batch_seconds, args.bootstrap_repeats)
        print(f"Complete: {summary['n_worlds']} independent worlds. {args.out / 'summary.json'}", flush=True)
        if args.preset:
            print("Smoke result only; do not present this run as evidence of learned memory or recovery.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed stages and the batch failure record are preserved.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
