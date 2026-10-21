#!/usr/bin/env python3
"""Display an existing run's real progress without starting another simulation."""
import argparse
import json
from pathlib import Path
import time

from neural_terminal import show_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run.expanduser().resolve()
    log = folder / "console.jsonl"
    if not log.is_file():
        parser.error("The run has no console log yet.")
    print(f"LESION LAB — WATCHING {folder.name}", flush=True)
    print("This window displays the existing simulation. Closing it stops only this viewer.\n", flush=True)
    with log.open() as stream:
        while True:
            line = stream.readline()
            if line:
                try:
                    show_record(json.loads(line))
                except json.JSONDecodeError:
                    print(line.rstrip(), flush=True)
                continue
            manifest = folder / "manifest.json"
            if manifest.exists():
                try:
                    status = json.loads(manifest.read_text()).get("status")
                except (OSError, json.JSONDecodeError):
                    status = "running"
                if status != "running":
                    print(f"\nSaved run status: {status}", flush=True)
                    break
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nViewer closed. The simulation continues in its original session.")
