import unittest
import sys
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import copy_gate, checkpoint_origin, ROOT, ENGINE


class EvidenceGateTests(unittest.TestCase):
    def test_population_similarity_and_activity_cannot_pass(self):
        self.assertFalse(copy_gate({"exact_copy_successes": 1000, "per_sample": []})["passed"])

    def test_one_off_copy_without_copying_descendant_cannot_pass(self):
        self.assertFalse(copy_gate({"per_sample": [{"slot": 3, "exact_copy_successes": 10, "second_generation_successes": 0}]})["passed"])

    def test_independent_foods_and_functioning_descendant_pass(self):
        gate = copy_gate({"per_sample": [{"slot": 9, "exact_copy_successes": 3, "second_generation_successes": 1}]})
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["candidate_slots"], [9])

    def test_origin_requires_manifest_matching_the_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "warmup.bin"
            path.write_bytes(b"checkpoint")
            manifest = path.parent / "manifest.json"
            manifest.write_text(json.dumps({"mode": "random_origin", "warmup_sha256": "wrong"}))
            self.assertEqual(checkpoint_origin(path), ("resumed_unknown_origin", None))
            manifest.write_text(json.dumps({"mode": "random_origin", "warmup_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
            mode, provenance = checkpoint_origin(path)
            self.assertEqual(mode, "random_origin")
            self.assertEqual(provenance["manifest"], str(manifest))

    def test_wrapper_resume_preserves_effective_checkpoint_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "seed.bin"
            subprocess.run([str(ENGINE), "run", "--population", "4", "--seed", "9", "--steps", "8", "--mutation", "0", "--interactions", "2", "--checkpoint", str(checkpoint), "--metrics", str(root / "first.csv")], check=True, capture_output=True)
            out = root / "continued"
            subprocess.run([sys.executable, str(ROOT / "lab.py"), "evolve", "--resume", str(checkpoint), "--interactions", "1", "--after", "0", "--samples", "2", "--trials", "3", "--out", str(out)], check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual((manifest["population"], manifest["seed"], manifest["step_budget"], manifest["mutation_per_byte"]), (4, 9, 8, 0))
            self.assertEqual((manifest["initial_interaction"], manifest["checkpoint_interaction"]), (2, 3))
            self.assertEqual(manifest["mode"], "resumed_unknown_origin")


if __name__ == "__main__":
    unittest.main()
