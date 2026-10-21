import copy
import random
from pathlib import Path
import tempfile
import unittest

from executive import (
    Controller,
    EXECUTIVE_UNITS,
    Genome,
    MOTOR_UNITS,
    NO_OUTPUT,
    Trial,
    evaluate,
    make_trials,
    run_executive_experiment,
    train_controller,
)


class ExecutiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fitted, cls.training = train_controller(seed=7)
        cls.heldout = make_trials(seed=91, repeats=8)

    def test_training_learns_cue_rule_and_inhibition_on_heldout_delays(self):
        metrics = evaluate(self.fitted.clone(), self.heldout)
        # Separate thresholds prevent the 50% no-go majority from hiding failure.
        self.assertGreaterEqual(metrics["delayed_choice_accuracy"], 0.95)
        self.assertGreaterEqual(metrics["inhibition_accuracy"], 0.95)
        self.assertGreaterEqual(metrics["reversal_accuracy"], 0.95)
        self.assertLess(self.training["last_window_mean_loss"], self.training["first_window_mean_loss"])

    def test_acquired_lesion_survives_episode_reset_and_training(self):
        controller = self.fitted.clone()
        unit = EXECUTIVE_UNITS[0]
        controller.acquire_lesion([unit, "memory.cue_a"])
        controller.predict(Trial(0, 6, True, 0))
        controller.reset_state()
        self.assertEqual(controller.memory, [0.0, 0.0])
        self.assertEqual(controller.lesions, {unit, "memory.cue_a"})
        before = copy.deepcopy(controller.genome.executive_weights[0])
        controller.learn(Trial(0, 5, True, 1))
        self.assertEqual(before, controller.genome.executive_weights[0])
        self.assertEqual(controller.lesions, {unit, "memory.cue_a"})
        features = controller.trial_features(Trial(0, 6, True, 0))
        hidden, _ = controller._forward(features)
        self.assertEqual(hidden[0], 0.0)
        self.assertEqual(controller.memory[0], 0.0)

    def test_acquired_damage_and_episode_state_are_not_inherited(self):
        parent = self.fitted.clone()
        original_genome = copy.deepcopy(parent.genome)
        parent.acquire_lesion([EXECUTIVE_UNITS[1]])
        parent.predict(Trial(1, 3, True, 1))
        child = parent.offspring()
        self.assertEqual(parent.genome, original_genome)
        self.assertEqual(child.genome, parent.genome)
        self.assertIsNot(child.genome, parent.genome)
        self.assertEqual(child.memory, [0.0, 0.0])
        self.assertEqual(child.lesions, set())

    def test_sham_is_exactly_equal_and_arms_have_equal_lesion_counts(self):
        result = run_executive_experiment(seed=7)
        conditions = result["conditions"]
        self.assertEqual(conditions["sham"]["pre"], conditions["sham"]["post"])
        self.assertTrue(all(value == 0 for value in conditions["sham"]["delta"].values()))
        self.assertEqual(len(conditions["executive_lesion"]["lesioned_units"]), 2)
        self.assertEqual(len(conditions["random_nonexecutive_lesion"]["lesioned_units"]), 2)
        for condition in conditions.values():
            self.assertEqual(condition["offspring_lesion_count"], 0)
            self.assertTrue(condition["acquired_lesion_persists_after_state_reset"])

    def test_no_go_target_and_reversed_choice_are_distinct(self):
        self.assertEqual(Trial(0, 3, False, 1).target, 0)
        self.assertEqual(Trial(0, 3, True, 0).target, 1)
        self.assertEqual(Trial(0, 3, True, 1).target, 2)

    def test_invalid_lesions_fail_before_mutating_controller(self):
        controller = Controller(Genome.random(random.Random(0)))
        with self.assertRaises(ValueError):
            controller.acquire_lesion(["not.a.unit"])
        self.assertEqual(controller.lesions, set())

    def test_complete_motor_ablation_is_explicit_failure(self):
        controller = self.fitted.clone()
        controller.acquire_lesion(MOTOR_UNITS)
        self.assertEqual(controller.predict(Trial(0, 5, False, 0)), NO_OUTPUT)
        metrics = evaluate(controller, self.heldout)
        self.assertEqual(metrics["overall_accuracy"], 0.0)
        self.assertEqual(metrics["inhibition_accuracy"], 0.0)
        self.assertEqual(metrics["no_output_trials"], len(self.heldout))
        self.assertEqual(metrics["confusion_labels"][-1], "no_output")

    def test_serialized_roundtrip_preserves_weights_lesions_and_episode_state(self):
        controller = self.fitted.clone()
        controller.acquire_lesion(["executive.2", "memory.cue_a"])
        controller.trial_features(Trial(1, 5, True, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = controller.save(Path(directory) / "model.json")
            restored = Controller.load(path)
        self.assertEqual(restored.to_dict(), controller.to_dict())
        self.assertEqual(evaluate(restored, self.heldout), evaluate(controller, self.heldout))
        self.assertEqual(restored.offspring().lesions, set())

    def test_serialization_rejects_nonfinite_or_wrong_shape_parameters(self):
        data = self.fitted.to_dict()
        data["genotype"]["motor_weights"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            Controller.from_dict(data)
        data = self.fitted.to_dict()
        data["genotype"]["executive_weights"][0].pop()
        with self.assertRaises(ValueError):
            Controller.from_dict(data)

    def test_dose_response_masks_are_count_matched_and_zero_dose_matches_sham(self):
        from executive import lesion_dose_response

        doses = lesion_dose_response(self.fitted, self.heldout, seed=7, masks_per_condition=4)
        self.assertEqual([row["lesion_count"] for row in doses], [0, 2, 4, 6, 8])
        for row in doses:
            for condition in ("executive_lesion", "random_nonexecutive_lesion"):
                arm = row[condition]
                self.assertEqual(arm["n_masks"], 4)
                for sample in arm["samples"]:
                    self.assertEqual(len(sample["lesioned_units"]), row["lesion_count"])
                for metric, summary in arm["summary"].items():
                    self.assertLessEqual(summary["min"], summary["mean"])
                    self.assertLessEqual(summary["mean"], summary["max"])
                    if row["lesion_count"] == 0:
                        self.assertEqual(summary["mean"], row["sham"][metric])

    def test_experiment_saves_reloadable_model_alongside_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "executive.json"
            result = run_executive_experiment(seed=7, output=output)
            self.assertTrue(output.exists())
            model_path = Path(result["model_file"])
            self.assertEqual(model_path, output.with_name("executive_model.json").resolve())
            restored = Controller.load(model_path)
        self.assertEqual(restored.lesions, set())
        self.assertEqual(restored.memory, [0.0, 0.0])
        self.assertEqual(evaluate(restored, make_trials(seed=1007)), result["baseline"])


if __name__ == "__main__":
    unittest.main()
