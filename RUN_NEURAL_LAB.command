#!/bin/bash
# Double-click this file in Finder to run the neural population in macOS Terminal.
cd -- "$(dirname -- "$0")" || exit 1
python3 -u neural_terminal.py --population 4096 --hidden 64 --decisions 1000000 --followup 100000 "$@"
lab_status=$?
printf '\nSimulation exited with status %s. Results are saved in the runs folder.\n' "$lab_status"
read -r -p "Press Return to close this terminal. "
exit "$lab_status"
