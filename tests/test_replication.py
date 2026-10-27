import copy
import json
from pathlib import Path
import tempfile
import unittest

import replication


def world(seed, value, sampled_agents=1):
    return {"seed": seed, "measures": {"effect": value},
            "counts": {"sampled_agents": sampled_agents}}


class ReplicationStatisticsTests(unittest.TestCase):
    def test_worlds_receive_equal_weight_not_agent_or_trial_weight(self):
        data = [world(1, 0.1, 10000), world(2, 0.2), world(3, 0.6)]
        result = replication.aggregate(data, [1, 2, 3], repeats=100)
        self.assertAlmostEqual(result["metrics"]["effect"]["mean"], 0.3)
        self.assertEqual(result["metrics"]["effect"]["n_worlds"], 3)
        self.assertEqual(result["inference_unit"], "independent_evolutionary_world")
        self.assertFalse(result["repair_or_lifetime_learning_implemented"])

    def test_missing_and_duplicate_worlds_cannot_silently_enter_summary(self):
        with self.assertRaisesRegex(ValueError, "every planned seed"):
            replication.aggregate([world(1, 0.1), world(2, 0.2)], [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "every planned seed"):
            replication.aggregate([world(1, 0.1), world(1, 0.2), world(3, 0.3)], [1, 2, 3])

    def test_missing_injured_sample_is_visible_not_dropped(self):
        summary = replication.aggregate([world(1, 0.1), world(2, None), world(3, 0.3)], [1, 2, 3])
        metric = summary["metrics"]["effect"]
        self.assertEqual(metric["missing_seeds"], [2])
        self.assertIsNone(metric["mean"])
        self.assertIsNone(metric["ci95_low"])

    def test_small_sample_interval_is_reproducible_and_scope_is_explicit(self):
        first = replication.world_interval([-0.2, 0.1, 0.3], repeats=100)
        self.assertEqual(first, replication.world_interval([-0.2, 0.1, 0.3], repeats=100))
        self.assertIsNone(replication.world_interval([0.1], repeats=100)["ci95_low"])
        with self.assertRaisesRegex(ValueError, "finite"):
            replication.world_interval([0.1, float("nan")], repeats=100)

    def test_configuration_rejects_duplicate_seeds_and_nonfinite_limits(self):
        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            replication.validate_configuration(replication.SMOKE, [1, 1, 2])
        with self.assertRaisesRegex(ValueError, "finite"):
            replication.validate_configuration({**replication.SMOKE, "severity": float("nan")}, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "10,000,000"):
            replication.validate_configuration({**replication.SMOKE, "assay_samples": 100000, "assay_trials": 101}, [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "hidden"):
            replication.validate_configuration({**replication.SMOKE, "hidden": 512}, [1, 2, 3])

    def test_assay_mismatch_cannot_be_treated_as_paired_evidence(self):
        config = {**replication.SMOKE, "assay_trials": 4, "assay_samples": 1}
        sample = {"seed": 11004, "trials": 4, "frozen_weights": True, "cue_mode": "normal",
                  "alive_pool_count": 1, "samples": 1, "per_sample": [{"id": 1}]}
        replication.validate_assay(sample, 1, config)
        for key, value in (("seed", 11005), ("trials", 8), ("cue_mode", "erased"), ("frozen_weights", False)):
            with self.assertRaisesRegex(ValueError, "Assay does not match"):
                replication.validate_assay({**sample, key: value}, 1, config)


class ReplicationNativeTests(unittest.TestCase):
    def test_three_native_worlds_preserve_pairing_counters_and_output(self):
        config = {**replication.SMOKE, "population": 32, "decisions": 4096,
                  "followup": 1024, "minimum_episodes": 256, "assay_samples": 32,
                  "assay_trials": 8, "progress_every": 4096}
        seeds = [107, 211, 307]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "replication"
            result = replication.execute(out, seeds, config, "test-only", 90, 100)
            self.assertEqual(result["n_worlds"], 3)
            self.assertEqual(json.loads((out / "replication.json").read_text())["status"], "complete")
            self.assertTrue((out / "worlds.csv").is_file())
            for record in result["worlds"]:
                self.assertEqual(record["counts"]["warmup_scheduled_steps"], 4096)
                self.assertGreaterEqual(record["counts"]["warmup_complete_scored_episodes"], 256)
                self.assertLess(record["counts"]["warmup_complete_scored_episodes"], 4096)
                self.assertEqual(record["measures"]["sham_immediate_population_delta"], 0)
                for arm in ("designated_region", "random"):
                    self.assertEqual(record["counts"][arm + "_followup_scheduled_steps"], 1024)
                    self.assertGreater(record["counts"][arm + "_sampled_injured"], 0)
            with self.assertRaises(FileExistsError):
                replication.execute(out, seeds, config, "test-only", 90, 100)
            hashes = replication.source_hashes()
            folder = out / "seed-107"
            with self.assertRaisesRegex(ValueError, "mismatched source"):
                replication.read_world(folder, 107, config, {**hashes, "neural_engine.cpp": "wrong"})
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            extinct = copy.deepcopy(manifest)
            arm = extinct["arms"]["designated_region"]
            for key in ("samples", "episodes", "correct", "go_trials", "go_correct", "nogo_trials", "nogo_correct", "alive_pool_count"):
                arm["final_assay"][key] = 0
            arm["final_assay"]["per_sample"] = []
            arm["final_assay"]["sample_slots"] = []
            arm["final"]["injured_alive"] = 0
            for individual in arm["final"]["individuals"]:
                individual["alive"] = 0
            manifest_path.write_text(json.dumps(extinct))
            row = replication.read_world(folder, 107, config, hashes)
            self.assertEqual(row["counts"]["designated_region_followup_alive"], 0)
            self.assertIsNone(row["measures"]["designated_region_followup_population_minus_sham"])
            manifest_path.write_text(json.dumps(manifest))
            control_path = folder / "baseline_cue_erased_assay.json"
            control = json.loads(control_path.read_text())
            altered = copy.deepcopy(control)
            altered["per_sample"][0]["slot"] += 100
            control_path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "every baseline identity"):
                replication.read_world(folder, 107, config, hashes)
            control_path.write_text(json.dumps(control))
            manifest["arms"]["designated_region"]["final"]["injured_alive"] += 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identity count"):
                replication.read_world(folder, 107, config, hashes)


if __name__ == "__main__":
    unittest.main()
