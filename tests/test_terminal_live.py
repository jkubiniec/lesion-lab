import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from terminal_live import latest_metrics


class TerminalLiveTests(unittest.TestCase):
    def test_partial_measurement_never_becomes_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "warmup.csv"
            path.write_text("interaction,exact_copy_events\n6000000,2\n6500000,3")
            self.assertEqual(latest_metrics(path), {"interaction": 6000000, "exact_copy_events": 2})
            path.write_text(path.read_text() + "\n")
            self.assertEqual(latest_metrics(path), {"interaction": 6500000, "exact_copy_events": 3})

    def test_existing_folder_is_preserved_even_without_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "run.log"
            sentinel.write_text("prior evidence")
            result = subprocess.run([sys.executable, str(ROOT / "terminal_live.py"), "--out", folder],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "prior evidence")
            self.assertFalse((Path(folder) / "manifest.json").exists())

    def test_small_real_run_produces_counts_assay_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            result = subprocess.run([sys.executable, str(ROOT / "terminal_live.py"), "--out", str(out),
                                     "--interactions", "2", "--after", "0", "--population", "4",
                                     "--steps", "32", "--samples", "1", "--trials", "3"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("2 / 2 new pairs", result.stdout)
            self.assertIn("Independent reproduction assay:", result.stdout)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_interaction"], 2)
            self.assertTrue((out / "report.html").is_file())
            self.assertIn("Starting random_origin", (out / "run.log").read_text())


if __name__ == "__main__":
    unittest.main()
