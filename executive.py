"""Task-directed functional-lesion experiment; this is not a biological brain model.

The recurrent cue buffer and module boundaries are hand designed. Only the
executive and motor weights learn, by supervised gradient descent. This is a
separate experiment from the BFF soup: its controller did not emerge there.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable


SENSORY_UNITS = ("sensory.cue_a", "sensory.cue_b", "sensory.rule", "sensory.go")
MEMORY_UNITS = ("memory.cue_a", "memory.cue_b")
EXECUTIVE_UNITS = tuple(f"executive.{i}" for i in range(8))
MOTOR_UNITS = ("motor.withhold", "motor.left", "motor.right")
NONEXECUTIVE_UNITS = SENSORY_UNITS + MEMORY_UNITS + MOTOR_UNITS
ALL_UNITS = SENSORY_UNITS + MEMORY_UNITS + EXECUTIVE_UNITS + MOTOR_UNITS
ACTION_NAMES = ("withhold", "left", "right")
NO_OUTPUT = 3


def architecture_spec() -> dict:
    return {
        "sensory_units": list(SENSORY_UNITS),
        "memory_units": list(MEMORY_UNITS),
        "executive_units": list(EXECUTIVE_UNITS),
        "motor_units": list(MOTOR_UNITS),
        "executive_activation": "tanh",
        "motor_activation": "softmax",
        "inherited_state": "learned executive and motor weights",
        "noninherited_state": "episode memory and acquired lesion mask",
    }


@dataclass(frozen=True)
class Trial:
    """One cue, a silent delay, then a rule-cued go or no-go probe."""

    cue: int
    delay: int
    go: bool
    rule: int

    @property
    def target(self) -> int:
        return 0 if not self.go else 1 + (self.cue ^ self.rule)


@dataclass
class Genome:
    """Inherited parameters, deliberately without lesions or transient state."""

    executive_weights: list[list[float]]
    motor_weights: list[list[float]]

    @classmethod
    def random(cls, rng: random.Random) -> "Genome":
        # Four features plus a bias feed eight tanh executive units.
        return cls(
            [[rng.uniform(-0.6, 0.6) for _ in range(5)] for _ in EXECUTIVE_UNITS],
            [[rng.uniform(-0.6, 0.6) for _ in range(9)] for _ in MOTOR_UNITS],
        )


class Controller:
    """Small plastic controller with persistent, acquired unit-ablation masks."""

    def __init__(self, genome: Genome):
        self.genome = copy.deepcopy(genome)
        self.lesions: set[str] = set()
        self.memory = [0.0, 0.0]

    def reset_state(self) -> None:
        """Reset episode memory; an acquired lesion survives this operation."""
        self.memory = [0.0, 0.0]

    def acquire_lesion(self, units: Iterable[str]) -> None:
        names = set(units)
        unknown = names.difference(ALL_UNITS)
        if unknown:
            raise ValueError(f"Unknown units: {sorted(unknown)}")
        self.lesions.update(names)
        for i, name in enumerate(MEMORY_UNITS):
            if name in self.lesions:
                self.memory[i] = 0.0

    def clone(self) -> "Controller":
        clone = Controller(self.genome)
        clone.acquire_lesion(self.lesions)
        return clone

    def offspring(self) -> "Controller":
        """Copy inherited parameters, excluding acquired lesions and memory."""
        return Controller(self.genome)

    def to_dict(self) -> dict:
        """Serialize inherited weights separately from acquired/runtime state."""
        return {
            "schema_version": 1,
            "kind": "lesion_lab_executive_controller",
            "architecture": architecture_spec(),
            "genotype": {
                "executive_weights": copy.deepcopy(self.genome.executive_weights),
                "motor_weights": copy.deepcopy(self.genome.motor_weights),
            },
            "acquired_state": {"lesioned_units": sorted(self.lesions)},
            "episode_state": {"memory": self.memory[:]},
            "notes": "Task-trained controller; hand-designed cue memory; not emerged from BFF or a validated brain model",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Controller":
        if data.get("schema_version") != 1 or data.get("kind") != "lesion_lab_executive_controller":
            raise ValueError("Unsupported controller format")
        if data.get("architecture") != architecture_spec():
            raise ValueError("Controller architecture does not match this implementation")

        def finite_vector(values, size):
            if not isinstance(values, list) or len(values) != size:
                raise ValueError("Invalid controller parameter dimensions")
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Controller parameters must be finite numbers")
            return [float(v) for v in values]

        try:
            genotype = data["genotype"]
            executive_rows = genotype["executive_weights"]
            motor_rows = genotype["motor_weights"]
            if len(executive_rows) != len(EXECUTIVE_UNITS) or len(motor_rows) != len(MOTOR_UNITS):
                raise ValueError("Invalid controller parameter dimensions")
            genome = Genome(
                [finite_vector(row, 5) for row in executive_rows],
                [finite_vector(row, 9) for row in motor_rows],
            )
            controller = cls(genome)
            controller.memory = finite_vector(data["episode_state"]["memory"], 2)
            controller.acquire_lesion(data["acquired_state"]["lesioned_units"])
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete or invalid controller format") from error
        return controller

    def save(self, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Controller":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _features(self, cue: int | None, go: bool, rule: int) -> list[float]:
        sensed = [float(cue == 0), float(cue == 1), float(rule), float(go)]
        for i, name in enumerate(SENSORY_UNITS):
            if name in self.lesions:
                sensed[i] = 0.0
        if cue is not None:
            self.memory = sensed[:2]
        for i, name in enumerate(MEMORY_UNITS):
            if name in self.lesions:
                self.memory[i] = 0.0
        return self.memory + sensed[2:] + [1.0]

    def _forward(self, features: list[float]) -> tuple[list[float], list[float]]:
        hidden = [
            0.0 if name in self.lesions else math.tanh(sum(w * x for w, x in zip(row, features)))
            for name, row in zip(EXECUTIVE_UNITS, self.genome.executive_weights)
        ]
        logits = [
            -1e9 if name in self.lesions else sum(w * x for w, x in zip(row, hidden + [1.0]))
            for name, row in zip(MOTOR_UNITS, self.genome.motor_weights)
        ]
        if all(name in self.lesions for name in MOTOR_UNITS):
            return hidden, [0.0, 0.0, 0.0]
        peak = max(logits)
        exp_logits = [math.exp(value - peak) for value in logits]
        total = sum(exp_logits)
        return hidden, [value / total for value in exp_logits]

    def trial_features(self, trial: Trial) -> list[float]:
        self.reset_state()
        self._features(trial.cue, False, 0)
        for _ in range(trial.delay):
            self._features(None, False, 0)
        return self._features(None, trial.go, trial.rule)

    def predict(self, trial: Trial) -> int:
        _, probabilities = self._forward(self.trial_features(trial))
        if not any(probabilities):
            return NO_OUTPUT
        return max(range(3), key=probabilities.__getitem__)

    def learn(self, trial: Trial, learning_rate: float = 0.08) -> float:
        features = self.trial_features(trial)
        hidden, probabilities = self._forward(features)
        errors = [p - float(i == trial.target) for i, p in enumerate(probabilities)]
        errors = [0.0 if name in self.lesions else error for name, error in zip(MOTOR_UNITS, errors)]
        hidden_errors = [
            0.0 if name in self.lesions else (1.0 - value * value)
            * sum(errors[j] * self.genome.motor_weights[j][i] for j in range(3))
            for i, (name, value) in enumerate(zip(EXECUTIVE_UNITS, hidden))
        ]
        for row, error in zip(self.genome.motor_weights, errors):
            for i, value in enumerate(hidden + [1.0]):
                row[i] -= learning_rate * error * value
        for row, error in zip(self.genome.executive_weights, hidden_errors):
            for i, value in enumerate(features):
                row[i] -= learning_rate * error * value
        return -math.log(max(probabilities[trial.target], 1e-15))


def make_trials(seed: int, repeats: int = 64, delays: tuple[int, ...] = (5, 6, 7, 8)) -> list[Trial]:
    """Balance all eight cue/rule/go conditions and shuffle episode order."""
    if repeats < 1 or not delays or min(delays) < 0:
        raise ValueError("repeats must be positive and delays must be nonnegative")
    rng = random.Random(seed)
    trials = [
        Trial(cue, rng.choice(delays), go, rule)
        for _ in range(repeats)
        for cue in (0, 1)
        for rule in (0, 1)
        for go in (False, True)
    ]
    rng.shuffle(trials)
    return trials


def train_controller(seed: int = 7, training_steps: int = 12000) -> tuple[Controller, dict]:
    if training_steps < 1:
        raise ValueError("training_steps must be positive")
    rng = random.Random(seed)
    controller = Controller(Genome.random(rng))
    losses: list[float] = []
    for _ in range(training_steps):
        trial = Trial(rng.randrange(2), rng.randrange(1, 5), bool(rng.randrange(2)), rng.randrange(2))
        losses.append(controller.learn(trial))
    window = min(200, training_steps)
    controller.reset_state()
    return controller, {
        "steps": training_steps,
        "method": "supervised online SGD; cross-entropy",
        "learning_rate": 0.08,
        "first_window_mean_loss": sum(losses[:window]) / window,
        "last_window_mean_loss": sum(losses[-window:]) / window,
        "training_delays": [1, 2, 3, 4],
        "trained_parameters": "executive and motor weights only; cue memory is hand coded",
    }


def evaluate(controller: Controller, trials: list[Trial]) -> dict:
    if not trials:
        raise ValueError("Evaluation requires at least one trial")
    groups = {
        "overall_accuracy": [0, 0],
        "delayed_choice_accuracy": [0, 0],
        "inhibition_accuracy": [0, 0],
        "reversal_accuracy": [0, 0],
        "normal_rule_accuracy": [0, 0],
    }
    confusion = [[0, 0, 0] for _ in range(3)]
    no_output_trials = 0
    for trial in trials:
        prediction = controller.predict(trial)
        if prediction == NO_OUTPUT:
            no_output_trials += 1
            if len(confusion[0]) == 3:
                for row in confusion:
                    row.append(0)
        correct = int(prediction == trial.target)
        confusion[trial.target][prediction] += 1
        names = ["overall_accuracy"]
        if trial.go:
            names += ["delayed_choice_accuracy", "reversal_accuracy" if trial.rule else "normal_rule_accuracy"]
        else:
            names += ["inhibition_accuracy"]
        for name in names:
            groups[name][0] += correct
            groups[name][1] += 1
    metrics = {name: good / count if count else None for name, (good, count) in groups.items()}
    return {
        **metrics,
        "trials": len(trials),
        "counts": {name: count for name, (_, count) in groups.items()},
        "confusion_matrix": confusion,
        "confusion_labels": list(ACTION_NAMES) + (["no_output"] if no_output_trials else []),
        "confusion_target_labels": list(ACTION_NAMES),
        "confusion_orientation": "rows=target, columns=prediction",
        "no_output_trials": no_output_trials,
    }


def lesion_dose_response(
    fitted: Controller,
    trials: list[Trial],
    seed: int,
    masks_per_condition: int = 16,
) -> list[dict]:
    """Measure sensitivity to sampled masks on one fixed, trained controller.

    Draws are uniform without replacement within a mask, with replacement across
    masks. Repeated masks are possible, and unavoidable at dose zero or when all
    eight executive units are removed. These are not independent training runs.
    """
    if masks_per_condition < 1:
        raise ValueError("masks_per_condition must be positive")
    baseline = evaluate(fitted.clone(), trials)
    metric_names = [name for name in baseline if name.endswith("_accuracy")]
    doses = []
    for count in (0, 2, 4, 6, 8):
        row = {"lesion_count": count, "sham": copy.deepcopy(baseline)}
        for index, (condition, pool) in enumerate((
            ("executive_lesion", EXECUTIVE_UNITS),
            ("random_nonexecutive_lesion", NONEXECUTIVE_UNITS),
        )):
            rng = random.Random(seed + count * 100 + index)
            samples = []
            for _ in range(masks_per_condition):
                units = sorted(rng.sample(pool, count))
                arm = fitted.clone()
                arm.acquire_lesion(units)
                metrics = evaluate(arm, trials)
                samples.append({
                    "lesioned_units": units,
                    "metrics": {name: metrics[name] for name in metric_names},
                    "no_output_trials": metrics["no_output_trials"],
                })
            summary = {}
            for name in metric_names:
                values = [sample["metrics"][name] for sample in samples]
                summary[name] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
            row[condition] = {
                "n_masks": masks_per_condition,
                "unique_masks": len({tuple(sample["lesioned_units"]) for sample in samples}),
                "summary": summary,
                "samples": samples,
                "all_motor_units_removed_masks": sum(sample["no_output_trials"] > 0 for sample in samples),
            }
        doses.append(row)
    return doses


def run_executive_experiment(
    seed: int = 7,
    output: str | Path | None = None,
    lesion_count: int = 2,
    training_steps: int = 12000,
) -> dict:
    """Fit once, clone the fitted controller, then apply matched focal lesions.

    Executive units and nonexecutive units are selected uniformly without
    replacement from their named pools. This single-seed demonstration estimates
    neither a population effect nor anatomical specificity.
    """
    if not 1 <= lesion_count <= 2:
        raise ValueError("This demonstration supports lesion_count=1 or 2")
    fitted, training = train_controller(seed, training_steps)
    trials = make_trials(seed + 1000)
    baseline = evaluate(fitted, trials)
    rng = random.Random(seed + 2000)
    lesion_sets = {
        "sham": [],
        "executive_lesion": sorted(rng.sample(EXECUTIVE_UNITS, lesion_count)),
        "random_nonexecutive_lesion": sorted(rng.sample(NONEXECUTIVE_UNITS, lesion_count)),
    }
    conditions = {}
    accuracy_names = [name for name in baseline if name.endswith("_accuracy")]
    for name, units in lesion_sets.items():
        arm = fitted.clone()
        arm.acquire_lesion(units)
        post = evaluate(arm, trials)
        conditions[name] = {
            "lesioned_units": units,
            "pre": copy.deepcopy(baseline),
            "post": post,
            "delta": {metric: post[metric] - baseline[metric] for metric in accuracy_names},
            "acquired_lesion_persists_after_state_reset": set(units) == arm.lesions,
            "offspring_lesion_count": len(arm.offspring().lesions),
        }
    result = {
        "schema_version": 1,
        "experiment": "task_directed_executive_controller",
        "seed": seed,
        "design": {
            "model_type": "hand-designed modular plastic controller with recurrent cue buffer",
            "separate_from_bff": True,
            "biological_validation": False,
            "task": "delayed two-choice cue memory; go/no-go; explicitly rule-cued reversal",
            "learning": "task-directed supervised training before lesion; weights frozen for evaluation",
            "memory": "hand-coded lossless cue buffer; delay generalization is built in",
            "reversal": "rule supplied at probe; no unsignaled contingency-change learning",
            "lesion": "persistent acquired unit ablation; no tissue, pain, or subjective trauma model",
            "comparison": "equal unit count, unequal functional importance and connection count",
            "interpretation": "functional intervention demonstration; no claim of a frontal lobe or consciousness",
        },
        "architecture": architecture_spec(),
        "training": training,
        "evaluation": {
            "trials": len(trials),
            "heldout_delays": [5, 6, 7, 8],
            "balanced_conditions": "all 8 cue/rule/go combinations, 64 repeats each",
            "paired": True,
            "random_action_chance": 1.0 / 3.0,
            "always_withhold_accuracy": 0.5,
            "go_only_random_choice_chance": 0.5,
            "statistical_scope": "one fitted controller and one lesion draw; descriptive only",
        },
        "baseline": baseline,
        "conditions": conditions,
        "dose_response_design": {
            "doses": [0, 2, 4, 6, 8],
            "masks_per_condition_per_dose": 16,
            "trained_controllers": 1,
            "sampling": "uniform without replacement within each mask, with replacement across masks; repeated masks allowed",
            "paired": "same trained weights and heldout trials for every mask; sham repeated at every dose",
            "interpretation": "variation across lesion masks, not independent training replicates or population uncertainty",
            "complete_motor_ablation": "explicit no_output response, scored incorrect for every target, including withhold",
            "complete_motor_ablation_caveat": "failure to produce a decision is not counted as successful inhibition; motor paralysis is not modeled",
            "comparison_caveat": "equal unit counts do not match connection loss or functional importance",
        },
        "dose_response": lesion_dose_response(fitted, trials, seed + 3000),
        "model_file": None,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = path.with_name("executive_model.json")
        fitted.reset_state()
        fitted.save(model_path)
        result["model_file"] = str(model_path.resolve())
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/executive.json"))
    parser.add_argument("--lesion-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--training-steps", type=int, default=12000)
    args = parser.parse_args()
    result = run_executive_experiment(args.seed, args.output, args.lesion_count, args.training_steps)
    print(f"Saved {args.output}")
    for name, arm in result["conditions"].items():
        metrics = arm["post"]
        print(f"{name}: accuracy={metrics['overall_accuracy']:.3f}; "
              f"inhibition={metrics['inhibition_accuracy']:.3f}; "
              f"reversal={metrics['reversal_accuracy']:.3f}; lesions={arm['lesioned_units']}")


if __name__ == "__main__":
    main()
