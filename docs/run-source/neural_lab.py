#!/usr/bin/env python3
"""Large recurrent-agent population experiments with explicit energy reproduction."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "build" / "neural_engine"
LIMITS = {"cpu_seconds": 120, "max_memory_mib": 256, "max_checkpoint_gib": 1.0}


def sha256_file(path: Path) -> str:
    """Hash population checkpoints without allocating their full contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check_episode_target(manifest: dict, minimum: int) -> dict:
    actual = int(manifest["warmup_final"]["episodes"])
    target = {"minimum": minimum, "actual": actual, "passed": actual >= minimum,
              "unit": "cumulative complete scored episodes at the healthy warmup checkpoint"}
    manifest["warmup_episode_target"] = target
    return target


def native_limits() -> None:
    """Apply limits in a child process; never change machine-wide settings."""
    try:
        os.nice(10)
    except PermissionError:
        # Some app sandboxes prohibit setpriority even when only lowering it.
        pass
    seconds = int(LIMITS["cpu_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    maximum_file = int(LIMITS["max_checkpoint_gib"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_FSIZE, (maximum_file, maximum_file))


def lower_priority_available() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import os; os.nice(10)"], capture_output=True)
    return probe.returncode == 0


def memory_estimate(population: int, hidden: int) -> int:
    return population * (4 * parameter_count(hidden) + 16 * hidden + 512) + 32 * 1024**2


def emit(kind: str, **fields) -> None:
    print(json.dumps({"kind": kind, **fields}), flush=True)


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parameter_count(hidden: int) -> int:
    return hidden * (8 + hidden + 1) + 3 * (hidden + 1)


def storage_estimate(population: int, hidden: int, branches: bool = True) -> int:
    # Genomes plus deliberately conservative allowance for state, metadata and scratch.
    checkpoint = population * (4 * parameter_count(hidden) + 16 * hidden + 512)
    return checkpoint * (7 if branches else 1) + 256 * 1024**2


def build() -> None:
    source = ROOT / "neural_engine.cpp"
    if ENGINE.exists() and ENGINE.stat().st_mtime >= source.stat().st_mtime:
        return
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("A C++17 compiler is required.")
    ENGINE.parent.mkdir(exist_ok=True)
    emit("message", message="Compiling the recurrent neural simulator…")
    subprocess.run([compiler, "-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic", str(source), "-o", str(ENGINE)], check=True, preexec_fn=native_limits)


def inspect(path: Path) -> dict:
    result = subprocess.run([str(ENGINE), "inspect", "--load", str(path), "--max-memory-mib", str(LIMITS["max_memory_mib"])], text=True, capture_output=True, check=True, preexec_fn=native_limits)
    return json.loads(result.stdout)


def stream_native(arguments: list, stage: str, start: int, count: int) -> None:
    emit("stage", stage=stage, stage_start_decisions=start, stage_requested_decisions=count)
    process = subprocess.Popen([str(ENGINE), *(str(v) for v in arguments), "--max-memory-mib", str(LIMITS["max_memory_mib"])], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT, preexec_fn=native_limits)
    try:
        for line in process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                emit("message", stage=stage, message=line.rstrip())
                continue
            event["stage"] = stage
            event["stage_start_decisions"] = start
            event["stage_requested_decisions"] = count
            print(json.dumps(event), flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()


def assay(checkpoint: Path, output: Path, args) -> dict:
    result = subprocess.run([str(ENGINE), "assay", "--load", str(checkpoint), "--samples", str(args.assay_samples),
                             "--trials", str(args.assay_trials), "--seed", str(args.seed + 11003), "--output", str(output), "--max-memory-mib", str(LIMITS["max_memory_mib"])],
                            capture_output=True, text=True, check=True, preexec_fn=native_limits)
    value = json.loads(output.read_text())
    emit("assay", stage=output.parent.name, output=str(output), accuracy=value.get("accuracy"),
         go_accuracy=value.get("go_accuracy"), nogo_accuracy=value.get("nogo_accuracy"),
         episodes=value.get("episodes"), message="Frozen-weight task assay; copying is supplied by the simulator.")
    return value


def percent(value) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def generate_report(out: Path, manifest: dict) -> None:
    baseline = manifest.get("baseline_assay", {})
    rows = []
    for name, result in manifest.get("arms", {}).items():
        for label, key in [("Immediate", "immediate_assay"), ("Follow-up", "final_assay")]:
            a = result.get(key, {})
            rows.append([name, label, percent(a.get("accuracy")), percent(a.get("go_accuracy")), percent(a.get("nogo_accuracy"))])
    header = ["Condition", "Time", "Overall", "Go choice", "No-go withholding"]
    cells = lambda row, tag: "<tr>" + "".join(f"<{tag}>{html.escape(str(x))}</{tag}>" for x in row) + "</tr>"
    table = cells(header, "th") + cells(["Shared baseline", "Before injury", percent(baseline.get("accuracy")), percent(baseline.get("go_accuracy")), percent(baseline.get("nogo_accuracy"))], "td") + "".join(cells(row, "td") for row in rows)
    p = manifest["population"]
    h = manifest["hidden"]
    total = p * parameter_count(h)
    summary = [
        "LESION LAB — RECURRENT NEURAL POPULATION", "",
        f"Population: {p:,} resident agents; {h} recurrent units each",
        f"Evolved parameters: {parameter_count(h):,} per agent; {total:,} across the population",
        f"Warmup decisions: {manifest.get('warmup_final', {}).get('decisions', 0):,}",
        f"Warmup task accuracy: {percent(baseline.get('accuracy'))}",
        "Reproduction: simulator-supplied energy threshold; inherited weights mutate at birth.",
        "No gradient training or lifetime relearning is implemented in this evolutionary model.",
        "The designated lesion region is an engineered partition, not a verified frontal analogue.", "",
    ]
    for name, arm in manifest.get("arms", {}).items():
        final = arm["final"]
        summary.append(f"{name}: assay {percent(arm['final_assay'].get('accuracy'))}; cumulative births {final.get('births', 0):,}; original injured agents alive {final.get('injured_alive', 0):,}")
    summary.extend(["", f"Status: {manifest['status']}", f"Report: {out / 'report.html'}", f"Manifest: {out / 'manifest.json'}"])
    (out / "summary.txt").write_text("\n".join(summary) + "\n")
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Lesion Lab · neural population</title>
<style>body{{background:#101720;color:#e3edf3;font:16px/1.6 system-ui;margin:0}}main{{max-width:1100px;margin:auto;padding:44px 24px}}h1{{font-size:46px;letter-spacing:-2px}}h2{{font-weight:500}}.muted{{color:#9fb1bf}}.stats{{display:flex;gap:32px;flex-wrap:wrap}}.stats strong{{font-size:30px;display:block}}.card{{padding:24px;border:1px solid #344352;border-radius:10px;margin:22px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #344352;padding:12px 8px}}th{{color:#9fb1bf}}a{{color:#83d7cc}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.note{{border-left:3px solid #e6b777;padding-left:14px}}@media(max-width:650px){{table{{font-size:11px}}td,th{{padding:8px 3px}}}}</style>
<main><p class="muted">LESION LAB / LARGE NEURAL POPULATION</p><h1>Recurrent agents under injury</h1>
<div class="stats"><div><strong>{p:,}</strong>resident agents</div><div><strong>{h}</strong>recurrent units / agent</div><div><strong>{total:,}</strong>inherited weights</div></div>
<div class="card"><h2>A richer computational organism</h2><p>Each agent has a fully recurrent neural controller, temporary memory state, finite energy, stable identity and ancestry. Successful task responses alter its energy. Reproduction copies and mutates inherited weights into a new individual.</p><p class="note">Birth is supplied by the simulator. These agents did not emerge from the BFF soup. The controller has no gradient training or lifetime relearning. This is a computational lesion experiment with untested biological validity.</p></div>
<div class="card"><h2>Frozen-weight task assays</h2><p class="muted">Remember an initial cue through distractors, then obey an explicitly supplied normal/reversed rule on go trials or withhold on no-go trials. Assay memory is reset per episode. Uniform three-action chance is 33.3%; always withholding is 25%; static left/right choice is 37.5% in expectation.</p><div style="overflow:auto"><table>{table}</table></div><p class="muted">Same sampling seed across arms. These are task probes nested inside one evolved world, not independent evolutionary replications. See raw assay JSON for realized condition counts.</p></div>
<div class="card"><h2>Intervention and inheritance</h2><p>The designated region is the final quarter of recurrent units. A focal intervention disables units within that region; the control disables the same number across all recurrent units. This partition has no established executive specialization.</p><p>Acquired masks follow the injured individual's stable ID. New offspring have new identities and healthy masks. A falling injured population can reflect replacement; it is not evidence of repair.</p></div>
<div class="card"><h2>Run record</h2><pre>{html.escape(chr(10).join(summary))}</pre><p><a href="manifest.json">Complete manifest</a> · <a href="baseline_assay.json">Baseline assay</a> · <a href="summary.txt">Terminal summary</a></p><details><summary>Parameters and measured outcomes</summary><pre>{html.escape(json.dumps(manifest, indent=2))}</pre></details></div></main></html>'''
    (out / "report.html").write_text(report)


def execute(args) -> Path:
    build()
    out = Path(args.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty output folder; previous experiments are preserved.")
    out.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    inherited = inspect(resume) if resume else None
    if inherited:
        args.population, args.hidden, args.seed = inherited["population"], inherited["hidden"], inherited["seed"]
        if inherited.get("disabled_units", 0):
            raise ValueError("Fresh matched injury comparisons require an uninjured starting checkpoint.")
    expected_memory = memory_estimate(args.population, args.hidden)
    if expected_memory > args.max_memory_mib * 1024**2:
        raise ValueError(f"Estimated model working memory {expected_memory / 1024**2:.1f} MiB exceeds the {args.max_memory_mib} MiB budget. Reduce population or hidden units.")
    needed = storage_estimate(args.population, args.hidden, not args.no_lesions)
    if needed > args.max_checkpoint_gib * 1024**3:
        raise ValueError(f"Estimated saved-data budget {needed / 1024**3:.2f} GiB exceeds the {args.max_checkpoint_gib:.2f} GiB limit. Reduce population or skip injury branches.")
    available = shutil.disk_usage(out).free
    if available < needed + args.minimum_free_gib * 1024**3:
        raise ValueError(f"This run would leave less than {args.minimum_free_gib:.1f} GiB free. Reduce population or free disk space.")
    manifest = {
        "schema_version": 1, "experiment": "evolved_recurrent_population", "status": "running",
        "population": args.population, "hidden": args.hidden, "seed": args.seed,
        "parameters_per_agent": parameter_count(args.hidden),
        "total_parameters": args.population * parameter_count(args.hidden),
        "weight_bytes": args.population * parameter_count(args.hidden) * 4,
        "resource_limits": {"estimated_model_memory_mib": expected_memory / 1024**2, "model_memory_budget_mib": args.max_memory_mib,
                            "estimated_saved_data_gib": needed / 1024**3, "saved_data_budget_gib": args.max_checkpoint_gib,
                            "minimum_free_disk_gib": args.minimum_free_gib, "native_cpu_seconds_per_process": args.cpu_seconds,
                            "max_run_wall_seconds": args.max_run_seconds, "native_nice_increment": 10 if lower_priority_available() else 0, "native_threads": 1,
                            "memory_limit_type": "preallocation estimate enforced by simulator; not a machine-wide or guaranteed RSS limit"},
        "warmup_additional_decisions": args.decisions, "followup_additional_decisions": args.followup,
        "minimum_warmup_scored_episodes": args.minimum_episodes,
        "severity": args.severity, "exposure": args.exposure, "lesion_seed": args.seed + 9001,
        "lesion_dose_basis": "fraction_of_designated_region", "designated_region": "last quarter of hidden units; unverified functional specialization",
        "reproduction": "simulator-supplied energy threshold; copying and mutation of weights at birth",
        "learning": "genetic evolution across births; no gradient training or lifetime relearning",
        "biological_validation": False, "source_bff_emergence": False,
        "resume_from": str(resume) if resume else None,
        "resumed_checkpoint_sha256": sha256_file(resume) if resume else None,
        "environment": {"python": sys.version, "platform": platform.platform()}, "arms": {},
    }
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    manifest["source_sha256"] = {}
    for name in ("neural_engine.cpp", "neural_lab.py", "neural_terminal.py"):
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, source_dir / name)
            manifest["source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    save_json(out / "manifest.json", manifest)
    emit("resources", **manifest["resource_limits"])
    emit("configuration", **{k: v for k, v in manifest.items() if k in ("population", "hidden", "parameters_per_agent", "total_parameters", "weight_bytes", "reproduction")})
    started = time.monotonic()
    start = inherited.get("decisions", 0) if inherited else 0
    command = ["run", "--decisions", args.decisions, "--checkpoint", out / "warmup.bin", "--metrics", out / "warmup.jsonl", "--progress-every", args.progress_every]
    if resume:
        command += ["--load", resume]
    else:
        command += ["--population", args.population, "--hidden", args.hidden, "--seed", args.seed]
    stream_native(command, "warmup", start, args.decisions)
    manifest["warmup_final"] = inspect(out / "warmup.bin")
    target = check_episode_target(manifest, args.minimum_episodes)
    if not target["passed"]:
        manifest["status"] = "episode_target_not_met"
        save_json(out / "manifest.json", manifest)
        raise RuntimeError(f"Warmup produced {target['actual']:,} complete scored episodes; "
                           f"at least {target['minimum']:,} were required. The healthy checkpoint "
                           "is preserved; resume it with a larger decision budget before injury comparisons.")
    manifest["baseline_assay"] = assay(out / "warmup.bin", out / "baseline_assay.json", args)
    save_json(out / "manifest.json", manifest)
    if not args.no_lesions:
        for arm, lesion in (("sham", "none"), ("designated_region", "executive"), ("random", "random")):
            folder = out / arm
            folder.mkdir()
            stream_native(["run", "--load", out / "warmup.bin", "--decisions", 0, "--checkpoint", folder / "injury.bin", "--metrics", folder / "injury.jsonl", "--lesion", lesion,
                           "--severity", args.severity, "--exposure", args.exposure, "--lesion-seed", args.seed + 9001], arm + "_injury", manifest["warmup_final"]["decisions"], 0)
            immediate = inspect(folder / "injury.bin")
            if immediate["genome_hash"] != manifest["warmup_final"]["genome_hash"]:
                raise RuntimeError("An acquired injury changed inherited weights.")
            result = {"immediate": immediate, "immediate_assay": assay(folder / "injury.bin", folder / "immediate_assay.json", args)}
            stream_native(["run", "--load", folder / "injury.bin", "--decisions", args.followup, "--checkpoint", folder / "final.bin", "--metrics", folder / "followup.jsonl", "--progress-every", args.progress_every], arm,
                          manifest["warmup_final"]["decisions"], args.followup)
            result["final"] = inspect(folder / "final.bin")
            result["final_assay"] = assay(folder / "final.bin", folder / "final_assay.json", args)
            manifest["arms"][arm] = result
            save_json(out / "manifest.json", manifest)
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.monotonic() - started
    save_json(out / "manifest.json", manifest)
    generate_report(out, manifest)
    emit("complete", status="complete", report=str(out / "report.html"), summary=str(out / "summary.txt"), elapsed_seconds=manifest["elapsed_seconds"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    p = sub.add_parser("run")
    p.add_argument("--population", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--decisions", type=int, default=1000000)
    p.add_argument("--minimum-episodes", type=int, default=0,
                   help="Require this many cumulative complete scored episodes after healthy warmup; preserve checkpoint and stop if unmet.")
    p.add_argument("--followup", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--resume")
    p.add_argument("--severity", type=float, default=0.5)
    p.add_argument("--exposure", type=float, default=0.25)
    p.add_argument("--progress-every", type=int, default=50000)
    p.add_argument("--assay-samples", type=int, default=128)
    p.add_argument("--assay-trials", type=int, default=32)
    p.add_argument("--no-lesions", action="store_true")
    p.add_argument("--max-memory-mib", type=int, default=256)
    p.add_argument("--cpu-seconds", type=int, default=120)
    p.add_argument("--max-checkpoint-gib", type=float, default=1.0)
    p.add_argument("--minimum-free-gib", type=float, default=5.0)
    p.add_argument("--max-run-seconds", type=int, default=300)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build()
            return
        if args.population < 2 or args.hidden < 4 or args.decisions < 0 or args.followup < 0 or args.seed < 0 or args.minimum_episodes < 0:
            raise ValueError("Use population >=2, hidden >=4, and nonnegative budgets, seed and minimum episodes.")
        if args.progress_every < 1 or args.assay_samples < 1 or args.assay_trials < 1:
            raise ValueError("Progress interval and assay counts must be positive.")
        if not 0 <= args.severity <= 1 or not 0 <= args.exposure <= 1:
            raise ValueError("Severity and exposure must be between 0 and 1.")
        if not 32 <= args.max_memory_mib <= 1024 or not 1 <= args.cpu_seconds <= 1800 or not 1 <= args.max_run_seconds <= 1800:
            raise ValueError("Use memory budget 32..1024 MiB, per-process CPU budget 1..1800 seconds, wall budget 1..1800 seconds.")
        if not 0 < args.max_checkpoint_gib <= 8 or not 1 <= args.minimum_free_gib <= 100:
            raise ValueError("Use saved-data budget >0 and <=8 GiB; reserve at least 1 GiB free disk.")
        LIMITS.update(cpu_seconds=args.cpu_seconds, max_memory_mib=args.max_memory_mib, max_checkpoint_gib=args.max_checkpoint_gib)
        def deadline(_signal, _frame):
            raise TimeoutError(f"Stopped at the {args.max_run_seconds}-second total runtime limit.")
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, args.max_run_seconds)
        try:
            execute(args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except KeyboardInterrupt:
        emit("stopped", status="interrupted", message="Completed stage checkpoints are preserved; the active stage may be incomplete.")
        raise SystemExit(130)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        emit("error", message=str(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
