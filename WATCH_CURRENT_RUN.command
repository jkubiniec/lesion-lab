#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1
python3 -u watch_run.py runs/large-16384-long-seed42
read -r -p "Press Return to close this viewer. "
