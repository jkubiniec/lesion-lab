"""Small integration checks for the recurrent ecology, resource bounds, and lesion causality."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NeuralEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="neural-engine-tests-")
        cls.binary = Path(os.environ.get("NEURAL_ENGINE", Path(cls.build.name) / "neural_engine"))
        if "NEURAL_ENGINE" not in os.environ:
            compiler = shutil.which("clang++") or shutil.which("g++")
            if not compiler:
                raise unittest.SkipTest("C++17 compiler required")
            subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic",
                            str(ROOT / "neural_engine.cpp"), "-o", str(cls.binary)],
                           check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.files = tempfile.TemporaryDirectory(prefix="neural-engine-case-")
        self.path = Path(self.files.name)

    def tearDown(self):
        self.files.cleanup()

    def command(self, *args, ok=True):
        result = subprocess.run([str(self.binary), *map(str, args)], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode == 0, ok, result.stderr)
        return result

    def run_world(self, name, decisions, *args):
        checkpoint = self.path / f"{name}.nrl"
        metrics = self.path / f"{name}.jsonl"
        self.command("run", "--decisions", decisions, "--checkpoint", checkpoint,
                     "--metrics", metrics, "--progress-every", 31, *args)
        return checkpoint, [json.loads(line) for line in metrics.read_text().splitlines()]

    def inspect(self, checkpoint):
        return json.loads(self.command("inspect", "--load", checkpoint).stdout)

    def test_inheritance_masks_and_reset_positive_controls(self):
        self.assertIn("self-test passed", self.command("--self-test").stdout)

    def test_checkpoint_resume_is_bit_exact_across_episode_boundaries(self):
        continuous, _ = self.run_world("continuous", 3000, "--population", 16, "--hidden", 8,
                                       "--seed", 42)
        prefix, _ = self.run_world("prefix", 777, "--population", 16, "--hidden", 8,
                                   "--seed", 42)
        resumed, metrics = self.run_world("resumed", 2223, "--load", prefix)
        self.assertEqual(continuous.read_bytes(), resumed.read_bytes())
        self.assertEqual(metrics[0]["decisions"], 777)
        self.assertEqual(metrics[-1]["decisions"], 3000)
        info = self.inspect(continuous)
        self.assertEqual(info["episodes"], info["go_trials"] + info["nogo_trials"])
        self.assertEqual(info["correct"], info["go_correct"] + info["nogo_correct"])
        self.assertGreater(info["births"], 0)
        self.assertGreater(info["next_id"], 17)

    def test_acquired_lesions_match_dose_and_do_not_modify_genomes_or_rng(self):
        baseline, _ = self.run_world("baseline", 17, "--population", 16, "--hidden", 16,
                                     "--seed", 45)
        initial = self.inspect(baseline)
        results = {}
        for kind in ("executive", "random"):
            cp, _ = self.run_world(kind, 0, "--load", baseline, "--lesion", kind,
                                   "--severity", .5, "--exposure", .5, "--lesion-seed", 808)
            info = self.inspect(cp)
            self.assertEqual(info["genome_hash"], initial["genome_hash"])
            self.assertEqual(info["rng_state"], initial["rng_state"])
            self.assertEqual(info["disabled_units"], 16)  # 8 individuals * 2/4 region units
            self.assertEqual(info["injured_initial"], 8)
            self.assertEqual(info["injured_alive"], 8)
            results[kind] = info
            after, _ = self.run_world(kind + "-after", 100, "--load", cp)
            self.assertEqual(self.inspect(after)["mask_hash"], info["mask_hash"])
            reapplied, _ = self.run_world(kind + "-again", 0, "--load", cp,
                                          "--lesion", kind, "--severity", .5,
                                          "--exposure", .5, "--lesion-seed", 808)
            self.assertEqual(cp.read_bytes(), reapplied.read_bytes())
        self.assertEqual(results["executive"]["injured_ids"], results["random"]["injured_ids"])
        self.assertNotEqual(results["executive"]["mask_hash"], results["random"]["mask_hash"])

    def test_heldout_assay_is_frozen_deterministic_and_uses_same_sample_slots(self):
        cp, _ = self.run_world("baseline", 180, "--population", 16, "--hidden", 8)
        original = cp.read_bytes()
        rows = []
        for _ in range(2):
            rows.append(json.loads(self.command("assay", "--load", cp, "--samples", 12,
                                                "--trials", 32, "--seed", 765).stdout))
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[0]["episodes"], 384)
        self.assertEqual(rows[0]["go_trials"] + rows[0]["nogo_trials"], 384)
        self.assertTrue(rows[0]["frozen_weights"])
        self.assertEqual(rows[0]["cue_mode"], "normal")
        self.assertEqual(rows[0]["alive_pool_count"], 16)
        records = rows[0]["per_sample"]
        self.assertEqual([r["slot"] for r in records], rows[0]["sample_slots"])
        for key in ("episodes", "correct", "go_trials", "go_correct", "nogo_trials", "nogo_correct"):
            self.assertEqual(sum(r[key] for r in records), rows[0][key])
        info = self.inspect(cp)
        for record in records:
            individual = info["individuals"][record["slot"]]
            self.assertEqual(record["id"], individual["id"])
            self.assertEqual(record["parent"], individual["parent"])
            self.assertFalse(record["injured"])
            self.assertEqual(record["disabled_count"], 0)
            self.assertEqual(record["correct"], record["correct_per_trial"].count("1"))
            self.assertEqual(record["go_trials"], record["go_per_trial"].count("1"))
            self.assertEqual(len(record["correct_per_trial"]), 32)
            self.assertEqual(record["correct_per_trial"], "".join(
                "1" if action == target else "0"
                for action, target in zip(record["action_per_trial"], record["target_per_trial"])))
        self.assertEqual(cp.read_bytes(), original)
        lesion, _ = self.run_world("lesion", 0, "--load", cp, "--lesion", "executive",
                                   "--severity", 1, "--exposure", 1)
        injured = json.loads(self.command("assay", "--load", lesion, "--samples", 12,
                                          "--trials", 32, "--seed", 765).stdout)
        self.assertEqual(rows[0]["sample_slots"], injured["sample_slots"])
        self.assertEqual(rows[0]["go_trials"], injured["go_trials"])
        self.assertEqual([r["id"] for r in records], [r["id"] for r in injured["per_sample"]])
        self.assertTrue(all(r["injured"] and r["disabled_count"] == 2 for r in injured["per_sample"]))

    def test_assay_cue_controls_are_deterministic_frozen_and_preserve_targets(self):
        cp, _ = self.run_world("baseline", 180, "--population", 16, "--hidden", 8)
        before = cp.read_bytes()
        info = self.inspect(cp)
        results = {}
        for mode in ("normal", "erased", "memory-reset"):
            args = ("assay", "--load", cp, "--samples", 12, "--trials", 32,
                    "--seed", 765, "--cue-mode", mode)
            first = json.loads(self.command(*args).stdout)
            self.assertEqual(first, json.loads(self.command(*args).stdout))
            self.assertEqual(first["cue_mode"], mode)
            results[mode] = first
        self.assertEqual(cp.read_bytes(), before)
        self.assertEqual(self.inspect(cp), info)
        for mode in ("erased", "memory-reset"):
            self.assertEqual(results[mode]["sample_slots"], results["normal"]["sample_slots"])
            self.assertEqual([r["target_per_trial"] for r in results[mode]["per_sample"]],
                             [r["target_per_trial"] for r in results["normal"]["per_sample"]])
        self.command("assay", "--load", cp, "--cue-mode", "invalid", ok=False)
        self.command("inspect", "--load", cp, "--cue-mode", "erased", ok=False)

    def test_resource_guard_invalid_options_and_checkpoint_integrity(self):
        cp, _ = self.run_world("valid", 0, "--population", 16, "--hidden", 8)
        for name, data in (("truncated", cp.read_bytes()[:-1]),
                           ("corrupt", cp.read_bytes()[:100] + bytes([cp.read_bytes()[100] ^ 1]) + cp.read_bytes()[101:])):
            bad = self.path / (name + ".nrl")
            bad.write_bytes(data)
            self.command("inspect", "--load", bad, ok=False)
        rejected = self.command("run", "--population", 16384, "--hidden", 64,
                                "--decisions", 0, "--checkpoint", self.path / "oversize.nrl", ok=False)
        self.assertIn("before allocation", rejected.stderr)
        self.assertFalse((self.path / "oversize.nrl").exists())
        self.command("run", "--population", 16, "--hidden", 8, "--severity", "nan",
                     "--checkpoint", self.path / "bad.nrl", ok=False)
        self.command("run", "--population", 16, "--hidden", 8, "--unknown", 1, ok=False)


if __name__ == "__main__":
    unittest.main()
