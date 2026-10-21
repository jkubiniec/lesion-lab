#!/usr/bin/env python3
"""Show real streamed progress from the neural-population experiment."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def stop_process_group(process: subprocess.Popen) -> None:
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            # A child may survive the orchestrator; stop the whole group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        except subprocess.TimeoutExpired:
            continue


def show_record(record: dict) -> None:
    kind = record.get("kind", record.get("type"))
    stage = record.get("stage", "simulation")
    if kind == "stage":
        print(f"\n{str(stage).upper()} — starting", flush=True)
        return
    if kind == "configuration":
        print(f"Actual population: {int(record['population']):,} | hidden units: {int(record['hidden']):,} | "
              f"parameters per agent: {int(record['parameters_per_agent']):,} | "
              f"total parameters: {int(record['total_parameters']):,}", flush=True)
        return
    if kind in ("message", "error", "stopped"):
        print(record.get("message", str(kind)), flush=True)
        return
    if kind == "assay":
        measures = [f"{label} {float(record[key]):.1%}"
                    for key, label in (("accuracy", "accuracy"), ("go_accuracy", "go"), ("nogo_accuracy", "no-go"))
                    if record.get(key) is not None]
        print(f"Frozen-weight assay [{stage}]: " + " | ".join(measures), flush=True)
        return
    if kind == "complete" and "decisions" not in record:
        print(f"Experiment complete. Report: {record.get('report', 'see output folder')}", flush=True)
        return
    if "decisions" not in record:
        print(json.dumps(record, ensure_ascii=False), flush=True)
        return
    total = int(record["decisions"])
    start = int(record.get("stage_start_decisions", record.get("start_decisions", 0)))
    done = max(0, total - start)
    requested = record.get("stage_requested_decisions")
    rate = float(record.get("decisions_per_second", 0))
    elapsed = float(record.get("elapsed_seconds", 0))
    progress = f"{done:,} new scheduled agent steps (total {total:,})"
    if requested is not None and int(requested) > 0:
        requested = int(requested)
        progress = f"{done:,} / {requested:,} new scheduled agent steps ({min(100, done / requested * 100):.1f}%)"
    timing = f"{rate:,.0f}/s | elapsed {duration(elapsed)}"
    if requested is not None and rate > 0:
        timing += f" | ETA {duration(max(0, requested - done) / rate)}"
    print(f"[{stage}] {progress} | {timing}", flush=True)
    metrics = []
    for key, label in (("neural_evaluations", "actual RNN updates"), ("episodes", "scored episodes"),
                       ("alive", "alive"), ("births", "cumulative births"), ("deaths", "cumulative deaths")):
        if key in record:
            metrics.append(f"{label} {int(record[key]):,}")
    for key, label in (("accuracy", "task accuracy"), ("go_accuracy", "go"), ("nogo_accuracy", "no-go")):
        if key in record:
            metrics.append(f"{label} {float(record[key]):.1%}")
    print("  " + " | ".join(metrics), flush=True)
    if record.get("disabled_units", 0):
        print(f"  Disabled units {int(record['disabled_units']):,} | "
              f"original injured agents alive {int(record.get('injured_alive', 0)):,} | "
              f"injured-agent birth replacements {int(record.get('injured_birth_replacements', 0)):,}", flush=True)


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise ValueError(f"Output already exists; choose a fresh folder: {out}")
    runner = ROOT / "neural_lab.py"
    if not runner.is_file():
        raise ValueError(f"Experiment runner is not available: {runner}")
    command = [sys.executable, "-u", str(runner), "run", "--population", str(args.population),
               "--hidden", str(args.hidden), "--decisions", str(args.decisions),
               "--minimum-episodes", str(args.minimum_episodes),
               "--followup", str(args.followup), "--seed", str(args.seed), "--out", str(out),
               "--severity", str(args.severity), "--exposure", str(args.exposure),
               "--progress-every", str(args.progress_every), "--assay-samples", str(args.assay_samples),
               "--assay-trials", str(args.assay_trials), "--max-memory-mib", str(args.max_memory_mib),
               "--cpu-seconds", str(args.cpu_seconds), "--max-checkpoint-gib", str(args.max_checkpoint_gib),
               "--minimum-free-gib", str(args.minimum_free_gib), "--max-run-seconds", str(args.max_run_seconds)]
    if args.no_lesions:
        command.append("--no-lesions")
    if args.resume:
        checkpoint = Path(args.resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Starting checkpoint does not exist: {checkpoint}")
        command.extend(["--resume", str(checkpoint)])
    print("LESION LAB — LIVE NEURAL POPULATION", flush=True)
    print(f"Population requested: {args.population:,} | hidden units per agent: {args.hidden:,}", flush=True)
    print(f"Warmup: {args.decisions:,} scheduled agent steps | followup per condition: {args.followup:,}", flush=True)
    print(f"Resource budgets: estimated model memory {args.max_memory_mib:,} MiB | "
          f"CPU {args.cpu_seconds:,} seconds per stage | checkpoint storage {args.max_checkpoint_gib:g} GiB", flush=True)
    print(f"Whole-run time limit: {args.max_run_seconds:,} seconds | disk space to leave free: {args.minimum_free_gib:g} GiB", flush=True)
    print(f"Output: {out}", flush=True)
    print("Scheduled steps include inactive agents; actual RNN updates and scored episodes are counted separately.", flush=True)
    print("ETA uses measured scheduled-step speed.", flush=True)
    print("Ctrl-C stops the runner and all simulator processes. Checkpoints are saved at completed stages.\n", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    raw_log = None
    pending_records = []
    try:
        for line in process.stdout:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip(), flush=True)
                continue
            if not isinstance(record, dict):
                print(line.rstrip(), flush=True)
                continue
            if raw_log is None and out.is_dir():
                raw_log = (out / "console.jsonl").open("x")
                for pending in pending_records:
                    raw_log.write(pending)
                pending_records.clear()
            if raw_log is None:
                pending_records.append(line)
            else:
                raw_log.write(line)
                raw_log.flush()
            show_record(record)
        returncode = process.wait()
    except KeyboardInterrupt:
        stop_process_group(process)
        print(f"\nStopped after {duration(time.monotonic() - started)}. Saved files: {out}", flush=True)
        return 130
    except BaseException:
        stop_process_group(process)
        raise
    finally:
        if raw_log is not None:
            raw_log.close()
        process.stdout.close()
    print(f"\nRun {'finished' if returncode == 0 else 'failed'} after {duration(time.monotonic() - started)}.", flush=True)
    if out.is_dir():
        print(f"Saved results: {out}", flush=True)
    report = out / "report.html"
    if report.is_file():
        print(f"Report: {report}", flush=True)
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=4_096)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--decisions", type=int, default=1_000_000)
    parser.add_argument("--minimum-episodes", type=int, default=0,
                        help="Minimum cumulative complete scored episodes required before injury comparisons.")
    parser.add_argument("--followup", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--exposure", type=float, default=0.25)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--assay-samples", type=int, default=128)
    parser.add_argument("--assay-trials", type=int, default=32)
    parser.add_argument("--no-lesions", action="store_true")
    parser.add_argument("--max-memory-mib", type=int, default=256)
    parser.add_argument("--cpu-seconds", type=int, default=120)
    parser.add_argument("--max-checkpoint-gib", type=float, default=1.0)
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument("--max-run-seconds", type=int, default=300)
    parser.add_argument("--out", default=str(ROOT / "runs" / datetime.now().strftime("neural-terminal-%Y%m%d-%H%M%S-%f")))
    args = parser.parse_args()
    if args.population < 2 or args.hidden < 4 or args.decisions < 0 or args.followup < 0 or args.seed < 0 or args.minimum_episodes < 0:
        parser.error("Population must be >=2, hidden units >=4, decisions, followup, seed and minimum episodes nonnegative.")
    if args.progress_every < 1 or args.assay_samples < 1 or args.assay_trials < 1:
        parser.error("Progress interval and assay counts must be positive.")
    if not 0 <= args.severity <= 1 or not 0 <= args.exposure <= 1:
        parser.error("Severity and exposure must be between 0 and 1.")
    if args.max_memory_mib <= 0 or args.cpu_seconds <= 0 or args.max_checkpoint_gib <= 0 or args.minimum_free_gib < 0 or args.max_run_seconds <= 0:
        parser.error("Memory, CPU and checkpoint budgets must be positive; minimum free space must be nonnegative.")
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Unable to start: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
