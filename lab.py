#!/usr/bin/env python3
"""Local, reproducible digital-evolution and acquired-lesion experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "build" / "bff_engine"
ARMS = ("sham", "focal", "random")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build() -> None:
    source = ROOT / "engine.cpp"
    if ENGINE.exists() and ENGINE.stat().st_mtime >= source.stat().st_mtime:
        return
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("Install the macOS command line developer tools or a C++17 compiler.")
    ENGINE.parent.mkdir(exist_ok=True)
    subprocess.run([compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic", str(source), "-o", str(ENGINE)], check=True)


def native(*args: object) -> None:
    subprocess.run([str(ENGINE), *(str(a) for a in args)], check=True, cwd=ROOT)


def checkpoint_metadata(path: Path) -> dict:
    result = subprocess.run([str(ENGINE), "inspect", "--load", str(path)], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def checkpoint_origin(path: Path) -> tuple[str, dict | None]:
    """Only inherit an origin label when a prior manifest matches these bytes."""
    prior_path = path.parent / "manifest.json"
    if prior_path.exists():
        prior = read_json(prior_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if prior.get("warmup_sha256") == digest:
            return prior.get("mode", "resumed_unknown_origin"), {
                "manifest": str(prior_path), "manifest_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                "checkpoint_sha256": digest,
            }
    return "resumed_unknown_origin", None


def copy_gate(assay: dict) -> dict:
    """An assay gate, not a claim of consciousness or open-ended evolution."""
    candidates = [
        p["slot"] for p in assay.get("per_sample", [])
        if p.get("exact_copy_successes", 0) >= 3
        and p.get("second_generation_successes", 0) >= 1
    ]
    return {
        "passed": bool(candidates),
        "candidate_slots": candidates,
        "rule": "At least one sampled tape copies exactly into >=3 independent foods, retains the source, and produces >=1 copying descendant.",
        "interpretation": "Operational copying evidence at this checkpoint; stable emergence requires repeated checkpoints and independent origin seeds.",
    }


def assay(checkpoint: Path, output: Path, samples: int, trials: int, seed: int) -> dict:
    native("assay", "--load", checkpoint, "--samples", samples, "--trials", trials, "--seed", seed, "--output", output)
    return read_json(output)


def execute_experiment(args: argparse.Namespace, seeded: bool) -> Path:
    build()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("This output folder already contains an experiment. Choose a new --out folder to preserve it.")
    started = time.time()
    initial_interaction = 0
    origin_provenance = None
    mode = "engineered_positive_control" if seeded else "random_origin"
    resumed = Path(args.resume).expanduser().resolve() if args.resume else None
    if resumed:
        inherited = checkpoint_metadata(resumed)
        if inherited["lesion"]["disabled_sites"]:
            raise ValueError("Fresh matched comparisons require an unlesioned starting checkpoint; use a saved warmup.bin.")
        if inherited["length"] != 64:
            raise ValueError("The report and protocol require 64-byte tapes.")
        args.population, args.seed = inherited["population"], inherited["seed"]
        args.steps, args.mutation = inherited["steps"], inherited["mutation"]
        initial_interaction = inherited["interaction"]
        mode, origin_provenance = checkpoint_origin(resumed)
    manifest = {
        "schema_version": 1, "name": "Lesion Lab", "mode": mode,
        "seed": args.seed, "population": args.population, "length": 64, "step_budget": args.steps,
        "warmup_interactions": args.interactions, "followup_interactions": args.after,
        "initial_interaction": initial_interaction,
        "checkpoint_interaction": initial_interaction + args.interactions,
        "mutation_per_byte": args.mutation, "lesion_severity": args.severity,
        "lesion_exposure": args.exposure, "lesion_seed": args.seed + 9001,
        "assay_seed": args.seed + 11003, "samples": args.samples, "trials": args.trials,
        "scheduler": "ordered pairs sampled with replacement; common deterministic random draws across arms",
        "substrate": "BFF-inspired noheads; fixed64-byte tapes; scan work counts against execution budget",
        "lesion_model": "persistent execution-disabled positions attached to resident slots; code bytes stay intact; masks are not copied",
        "resume_from": str(resumed) if resumed else None, "origin_provenance": origin_provenance,
        "status": "warming_up", "arms": {},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "engine.cpp", ROOT / "lab.py", ROOT / "executive.py", ROOT / "report.html"]},
    }
    write_json(out / "manifest.json", manifest)
    manifest["environment"] = {"python": sys.version, "platform": platform.platform()}
    frozen_sources = out / "source"
    frozen_sources.mkdir(exist_ok=True)
    for filename in manifest["source_sha256"]:
        shutil.copy2(ROOT / filename, frozen_sources / filename)
    write_json(out / "manifest.json", manifest)
    print(f"Starting {mode}: {args.interactions:,} additional pair interactions.", flush=True)
    command = ["run", "--interactions", args.interactions,
               "--checkpoint", out / "warmup.bin", "--metrics", out / "warmup.csv",
               "--metric-every", max(1, args.interactions // 60)]
    if resumed:
        command.extend(["--load", resumed])
    else:
        command.extend(["--population", args.population, "--length", 64, "--steps", args.steps,
                        "--seed", args.seed, "--mutation", args.mutation])
        if seeded:
            command.append("--seeded")
    native(*command)
    manifest["warmup_sha256"] = hashlib.sha256((out / "warmup.bin").read_bytes()).hexdigest()
    baseline = assay(out / "warmup.bin", out / "baseline_assay.json", args.samples, args.trials, manifest["assay_seed"])
    manifest["requested_samples"] = args.samples
    manifest["samples"] = baseline["samples"]
    manifest["baseline_assay"] = "baseline_assay.json"
    manifest["copy_gate"] = copy_gate(baseline)
    write_json(out / "manifest.json", manifest)
    if manifest["copy_gate"]["passed"]:
        print("Copying assay passed. Forking the same checkpoint into three matched conditions.", flush=True)
        for arm in ARMS:
            print(f"Running {arm} condition…", flush=True)
            arm_path = out / arm
            arm_path.mkdir(exist_ok=True)
            lesion = "none" if arm == "sham" else arm
            # Zero-interaction checkpoint applies the lesion and captures the immediate effect.
            native("run", "--load", out / "warmup.bin", "--interactions", 0,
                   "--checkpoint", arm_path / "injury.bin", "--metrics", arm_path / "injury.csv",
                   "--lesion", lesion, "--severity", args.severity, "--exposure", args.exposure,
                   "--lesion-seed", manifest["lesion_seed"])
            assay(arm_path / "injury.bin", arm_path / "immediate_assay.json", args.samples, args.trials, manifest["assay_seed"])
            native("run", "--load", arm_path / "injury.bin", "--interactions", args.after,
                   "--checkpoint", arm_path / "final.bin", "--metrics", arm_path / "followup.csv",
                   "--metric-every", max(1, args.after // 60))
            assay(arm_path / "final.bin", arm_path / "final_assay.json", args.samples, args.trials, manifest["assay_seed"])
            manifest["arms"][arm] = {"immediate": f"{arm}/immediate_assay.json", "final": f"{arm}/final_assay.json", "metrics": f"{arm}/followup.csv"}
            write_json(out / "manifest.json", manifest)
        manifest["status"] = "complete"
    else:
        print("Copying was not verified in the sampled tapes. Checkpoint saved; damage comparison deferred.", flush=True)
        manifest["status"] = "copying_not_verified"
    if args.with_executive:
        from executive import run_executive_experiment
        print("Training and testing the separate functional-control model…", flush=True)
        run_executive_experiment(seed=args.seed, output=out / "executive.json")
        manifest["executive"] = "executive.json"
    manifest["elapsed_seconds"] = round(time.time() - started, 3)
    write_json(out / "manifest.json", manifest)
    return create_report(out)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def create_report(out: Path) -> Path:
    manifest = read_json(out / "manifest.json")
    data = {"manifest": manifest, "warmup": read_csv(out / "warmup.csv"), "arms": {}}
    if manifest.get("baseline_assay"):
        data["baseline"] = read_json(out / manifest["baseline_assay"])
    for name, files in manifest.get("arms", {}).items():
        data["arms"][name] = {"metrics": read_csv(out / files["metrics"]), "immediate": read_json(out / files["immediate"]), "final": read_json(out / files["final"])}
    if manifest.get("executive"):
        data["executive"] = read_json(out / manifest["executive"])
    payload = json.dumps(data).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ROOT / "report.html").read_text()
    target = out / "report.html"
    target.write_text(template.replace("__LAB_DATA__", payload))
    print(f"Report: {target}", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Compile the bounded native simulator")
    sub.add_parser("test", help="Run the scientific and implementation checks")
    for name, seeded in [("demo", True), ("evolve", False)]:
        p = sub.add_parser(name, help="Engineered copying control" if seeded else "Search for copying from random bytes")
        p.add_argument("--out", default=f"runs/{name}-{int(time.time())}")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--population", type=int, default=128 if seeded else 1024)
        p.add_argument("--steps", type=int, default=8192)
        p.add_argument("--interactions", type=int, default=0 if seeded else 3000000)
        p.add_argument("--after", type=int, default=20000 if seeded else 300000)
        p.add_argument("--mutation", type=float, default=0.0 if seeded else 1 / 4096)
        p.add_argument("--severity", type=float, default=0.25)
        p.add_argument("--exposure", type=float, default=1.0 if seeded else 0.25)
        p.add_argument("--samples", type=int, default=64 if seeded else 128)
        p.add_argument("--trials", type=int, default=32)
        p.add_argument("--resume", help="Resume an existing checkpoint for this many additional interactions")
        p.add_argument("--with-executive", action="store_true", help="Also run the independent learned-control experiment")
    p = sub.add_parser("report", help="Rebuild an existing run's interactive report")
    p.add_argument("folder")
    p = sub.add_parser("executive", help="Run the independent learned-control experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/executive.json")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build()
        elif args.command == "test":
            build()
            native("--self-test")
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True, cwd=ROOT)
        elif args.command in ("demo", "evolve"):
            if args.population < 2 or args.steps <= 0 or args.interactions < 0 or args.after < 0:
                raise ValueError("Population >=2, positive step budget, and nonnegative interaction counts are required.")
            if args.samples < 1 or args.trials < 3:
                raise ValueError("Use >=1 sampled tape and >=3 independent assay foods.")
            if not 0 <= args.severity <= 1 or not 0 <= args.exposure <= 1 or not 0 <= args.mutation <= 1:
                raise ValueError("Mutation, severity and exposure must be between 0 and 1.")
            if args.seed < 0:
                raise ValueError("Use a nonnegative seed.")
            execute_experiment(args, args.command == "demo")
        elif args.command == "report":
            create_report(Path(args.folder).expanduser().resolve())
        elif args.command == "executive":
            from executive import run_executive_experiment
            result = run_executive_experiment(seed=args.seed, output=Path(args.out).expanduser().resolve())
            print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
