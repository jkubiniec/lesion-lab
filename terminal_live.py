#!/usr/bin/env python3
"""Run a fresh experiment while showing actual, flushed simulation progress."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def latest_metrics(path: Path) -> dict | None:
    """Ignore a partially written final line; counters are never interpolated."""
    try:
        contents = path.read_text()
    except FileNotFoundError:
        return None
    if not contents.endswith("\n"):
        contents = contents[:contents.rfind("\n") + 1]
    latest = None
    for row in csv.DictReader(io.StringIO(contents)):
        try:
            latest = {
                "interaction": int(row["interaction"]),
                "exact_copy_events": int(row["exact_copy_events"]),
            }
        except (ValueError, TypeError, KeyError):
            continue
    return latest


def manifest_at(out: Path) -> dict:
    try:
        return json.loads((out / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def stop_process_group(process: subprocess.Popen) -> None:
    """Stop the wrapper's simulation and any native/compiler descendants."""
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            # The group can outlive its leader, so also clear surviving children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
        except subprocess.TimeoutExpired:
            continue


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    resume = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume is not None and not resume.is_file():
        raise ValueError(f"Starting checkpoint does not exist: {resume}")
    # A fresh folder also protects incomplete runs, whose manifest may be absent.
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(ROOT / "lab.py"), "evolve", "--out", str(out),
               "--interactions", str(args.interactions), "--seed", str(args.seed),
               "--after", str(args.after), "--population", str(args.population),
               "--steps", str(args.steps), "--samples", str(args.samples), "--trials", str(args.trials)]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    print("LESION LAB — LIVE COMPUTATION", flush=True)
    print(f"New pair interactions requested: {args.interactions:,}", flush=True)
    print(f"Starting checkpoint: {resume or 'random initialization'}", flush=True)
    print(f"Output: {out}\nFull engine log: {out / 'run.log'}", flush=True)
    print("Counts below are completed pairs from flushed measurements; ETA is an estimate.", flush=True)
    print("Exact-copy events are candidates. Independent copying assays verify reproduction.", flush=True)
    print("Ctrl-C stops this run and its simulator. Checkpoints are saved at completed stages.\n", flush=True)
    started = time.monotonic()
    with (out / "run.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            next_update = 0.0
            while True:
                elapsed = time.monotonic() - started
                returncode = process.poll()
                if elapsed >= next_update or returncode is not None:
                    manifest = manifest_at(out)
                    metrics = latest_metrics(out / "warmup.csv")
                    if metrics is None:
                        print(f"Elapsed {duration(elapsed)} | starting simulator; no completed measurement yet", flush=True)
                    else:
                        initial = manifest.get("initial_interaction", 0)
                        completed = max(0, metrics["interaction"] - initial)
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = max(0, args.interactions - completed)
                        eta = duration(remaining / rate) if rate else "estimating"
                        percent = min(100.0, completed / args.interactions * 100)
                        print(f"{completed:>12,} / {args.interactions:,} new pairs ({percent:5.1f}%) | "
                              f"{rate:,.0f}/s avg | elapsed {duration(elapsed)} | ETA {eta} | "
                              f"exact-copy events {metrics['exact_copy_events']:,}", flush=True)
                        if completed >= args.interactions and returncode is None:
                            print("Warmup complete; running independent assays and any eligible damage comparisons.", flush=True)
                    next_update = elapsed + 1.0
                if returncode is not None:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            stop_process_group(process)
            print(f"\nStopped by Ctrl-C after {duration(time.monotonic() - started)}. Files: {out}", flush=True)
            return 130
        except BaseException:
            stop_process_group(process)
            raise
    manifest = manifest_at(out)
    print(f"\nRun {'finished' if returncode == 0 else 'failed'} after {duration(time.monotonic() - started)}.", flush=True)
    print(f"Status: {manifest.get('status', 'no final manifest')}", flush=True)
    gate = manifest.get("copy_gate")
    if gate is not None:
        print(f"Independent reproduction assay: {'PASSED' if gate['passed'] else 'NOT VERIFIED'}", flush=True)
    report = out / "report.html"
    if report.is_file():
        print(f"Report: {report}", flush=True)
    if returncode != 0:
        print(f"Details: {out / 'run.log'}", flush=True)
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=int, default=30_000_000)
    parser.add_argument("--resume", help="Continue an existing warmup.bin without changing it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "runs" / datetime.now().strftime("terminal-%Y%m%d-%H%M%S-%f")))
    parser.add_argument("--after", type=int, default=300_000)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--trials", type=int, default=32)
    args = parser.parse_args()
    if args.interactions <= 0 or args.after < 0 or args.seed < 0:
        parser.error("Use positive --interactions and nonnegative --after and --seed.")
    if args.population < 2 or args.steps <= 0 or args.samples < 1 or args.trials < 3:
        parser.error("Use population >=2, steps >0, samples >=1 and trials >=3.")
    try:
        return run(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Unable to start: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
