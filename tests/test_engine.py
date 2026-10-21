"""Integration checks for resumability, causal pairing, and strict reproduction."""
import csv
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_dir = tempfile.TemporaryDirectory(prefix="lesion-engine-tests-")
        cls.binary = Path(os.environ.get("BFF_ENGINE", Path(cls.build_dir.name) / "engine"))
        if "BFF_ENGINE" not in os.environ:
            compiler = shutil.which("clang++") or shutil.which("g++")
            if not compiler:
                raise unittest.SkipTest("a C++17 compiler is required")
            subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                            str(ROOT / "engine.cpp"), "-o", str(cls.binary)], check=True,
                           capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.build_dir.cleanup()

    def setUp(self):
        self.files = tempfile.TemporaryDirectory(prefix="lesion-engine-case-")
        self.path = Path(self.files.name)

    def tearDown(self):
        self.files.cleanup()

    def command(self, *args, ok=True):
        result = subprocess.run([str(self.binary), *map(str, args)], capture_output=True,
                                text=True, timeout=30)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def run_world(self, name, count, *args):
        checkpoint, metrics = self.path / f"{name}.bff", self.path / f"{name}.csv"
        self.command("run", "--interactions", count, "--checkpoint", checkpoint,
                     "--metrics", metrics, "--metric-every", 25, *args)
        return checkpoint, metrics

    def inspect(self, checkpoint):
        return json.loads(self.command("inspect", "--load", checkpoint).stdout)

    def test_native_semantics_and_positive_control(self):
        result = self.command("--self-test")
        self.assertIn("self-test passed", result.stdout)

    def test_checkpoint_resume_is_bit_exact(self):
        continuous, _ = self.run_world("continuous", 250, "--population", 16, "--seed", 42,
                                       "--steps", 512, "--mutation", .01)
        prefix, _ = self.run_world("prefix", 100, "--population", 16, "--seed", 42,
                                   "--steps", 512, "--mutation", .01)
        resumed, metrics = self.run_world("resumed", 150, "--load", prefix)
        self.assertEqual(continuous.read_bytes(), resumed.read_bytes())
        with metrics.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(int(rows[0]["interaction"]), 100)
        self.assertEqual(int(rows[-1]["interaction"]), 250)

    def test_lesions_match_dose_preserve_genome_and_future_randomness(self):
        baseline, _ = self.run_world("baseline", 0, "--population", 32, "--seed", 67,
                                     "--steps", 256)
        sham, _ = self.run_world("sham", 300, "--load", baseline)
        results = {}
        for kind in ("focal", "random"):
            lesioned, _ = self.run_world(kind, 0, "--load", baseline, "--lesion", kind,
                                         "--severity", .25, "--exposure", .5,
                                         "--lesion-seed", 808)
            info = self.inspect(lesioned)
            self.assertEqual(info["lesion"]["disabled_sites"], 256)
            self.assertEqual(info["lesion"]["affected_slots"], 16)
            # Binary format: 152-byte metadata, then soup, then resident masks.
            data = lesioned.read_bytes()
            self.assertEqual(data[152:152 + 32 * 64], baseline.read_bytes()[152:152 + 32 * 64])
            resumed, _ = self.run_world(kind + "-after", 300, "--load", lesioned)
            results[kind] = self.inspect(resumed)
            self.assertEqual(resumed.read_bytes()[-32 * 64:], data[-32 * 64:])
            reapplied, _ = self.run_world(kind + "-reapplied", 0, "--load", lesioned,
                                          "--lesion", kind, "--severity", .25,
                                          "--exposure", .5, "--lesion-seed", 808)
            self.assertEqual(reapplied.read_bytes(), data)
        expected_rng = self.inspect(sham)["rng_state"]
        self.assertEqual(results["focal"]["rng_state"], expected_rng)
        self.assertEqual(results["random"]["rng_state"], expected_rng)

    def test_strict_reproduction_probe_detects_control_and_full_injury(self):
        checkpoint, _ = self.run_world("control", 0, "--population", 16, "--seed", 6,
                                       "--seeded", "--mutation", 0)
        outputs = {}
        for kind, severity in (("none", 0), ("focal", 1)):
            output = self.path / f"assay-{kind}.json"
            self.command("assay", "--load", checkpoint, "--trials", 8, "--samples", 16,
                         "--output", output, "--seed", 123, "--lesion", kind,
                         "--severity", severity, "--exposure", 1)
            outputs[kind] = json.loads(output.read_text())
        self.assertGreaterEqual(outputs["none"]["qualifying_samples"], 1)
        control = next(row for row in outputs["none"]["per_sample"] if row["slot"] == 0)
        self.assertEqual(control["exact_copy_successes"], 8)
        self.assertEqual(control["second_generation_successes"], 8)
        self.assertEqual(outputs["focal"]["exact_copy_successes"], 0)
        self.assertEqual([x["slot"] for x in outputs["none"]["per_sample"]],
                         [x["slot"] for x in outputs["focal"]["per_sample"]])

    def test_corrupted_checkpoint_and_invalid_arguments_fail_cleanly(self):
        checkpoint, _ = self.run_world("valid", 0, "--population", 16)
        original = checkpoint.read_bytes()
        for name, data in (("truncated", original[:-1]),
                           ("zero-rng", original[:88] + bytes(32) + original[120:]),
                           ("nan-lesion", original[:128] + struct.pack("<d", float("nan")) + original[136:])):
            path = self.path / (name + ".bff")
            path.write_bytes(data)
            self.command("inspect", "--load", path, ok=False)
        self.command("run", "--checkpoint", self.path / "bad.bff", "--metrics", self.path / "bad.csv",
                     "--severity", 1.1, ok=False)


if __name__ == "__main__":
    unittest.main()
