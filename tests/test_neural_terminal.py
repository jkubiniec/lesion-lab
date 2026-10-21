import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neural_terminal
from neural_terminal import show_record


class NeuralTerminalTests(unittest.TestCase):
    def test_resume_percentage_and_eta_use_additional_stage_decisions(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show_record({"kind": "progress", "stage": "warmup", "decisions": 6_500_000,
                         "stage_start_decisions": 6_000_000, "stage_requested_decisions": 3_000_000,
                         "decisions_per_second": 100_000, "elapsed_seconds": 5,
                         "accuracy": 0.75, "births": 12, "deaths": 4, "neural_evaluations": 6_450_000})
        self.assertIn("500,000 / 3,000,000 new scheduled agent steps (16.7%)", output.getvalue())
        self.assertIn("ETA 00:00:25", output.getvalue())
        self.assertIn("task accuracy 75.0%", output.getvalue())
        self.assertIn("cumulative births 12", output.getvalue())
        self.assertIn("actual RNN updates 6,450,000", output.getvalue())

    def test_zero_decision_injury_stage_has_no_percentage_division(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show_record({"kind": "progress", "stage": "injury", "decisions": 100,
                         "stage_start_decisions": 100, "stage_requested_decisions": 0,
                         "decisions_per_second": 0, "elapsed_seconds": 0})
        self.assertIn("0 new scheduled agent steps (total 100)", output.getvalue())

    def test_existing_output_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / "console.jsonl"
            evidence.write_text("prior evidence")
            result = subprocess.run([sys.executable, str(ROOT / "neural_terminal.py"), "--out", folder],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(evidence.read_text(), "prior evidence")

    def test_resource_limits_reach_runner_without_default_population_inflation(self):
        with tempfile.TemporaryDirectory() as folder:
            fake_root = Path(folder)
            (fake_root / "neural_lab.py").write_text(
                "import json,pathlib,sys\n"
                "out=pathlib.Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir()\n"
                "print(json.dumps({'kind':'arguments','argv':sys.argv[1:]}),flush=True)\n")
            out = fake_root / "run"
            arguments = ["neural_terminal.py", "--out", str(out), "--max-memory-mib", "192",
                         "--cpu-seconds", "17", "--max-checkpoint-gib", "0.75", "--minimum-free-gib", "6.5",
                         "--max-run-seconds", "42", "--minimum-episodes", "17"]
            with mock.patch.object(neural_terminal, "ROOT", fake_root), mock.patch.object(sys, "argv", arguments), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = neural_terminal.main()
            self.assertEqual(status, 0)
            forwarded = json.loads((out / "console.jsonl").read_text())["argv"]
            for flag, value in (("--population", "4096"), ("--max-memory-mib", "192"),
                                ("--cpu-seconds", "17"), ("--max-checkpoint-gib", "0.75"),
                                ("--minimum-free-gib", "6.5"), ("--max-run-seconds", "42"),
                                ("--decisions", "1000000"), ("--followup", "100000"),
                                ("--minimum-episodes", "17")):
                self.assertEqual(forwarded[forwarded.index(flag) + 1], value)

    @unittest.skipUnless((ROOT / "build" / "neural_engine").exists(), "Native neural engine is not built")
    def test_real_small_run_streams_progress_saves_json_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            result = subprocess.run([sys.executable, str(ROOT / "neural_terminal.py"), "--out", str(out),
                                     "--population", "8", "--hidden", "4", "--decisions", "40",
                                     "--followup", "20", "--progress-every", "20",
                                     "--assay-samples", "4", "--assay-trials", "4"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("40 / 40 new scheduled agent steps (100.0%)", result.stdout)
            self.assertIn("Frozen-weight assay", result.stdout)
            self.assertIn("Run finished", result.stdout)
            records = [json.loads(line) for line in (out / "console.jsonl").read_text().splitlines()]
            self.assertTrue(any(record.get("kind") == "complete" for record in records))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["warmup_final"]["decisions"], 40)
            self.assertEqual(set(manifest["arms"]), {"sham", "designated_region", "random"})
            self.assertTrue((out / "report.html").is_file())


if __name__ == "__main__":
    unittest.main()
