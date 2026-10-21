#!/usr/bin/env python3
"""Inspect NEURAL02 checkpoints and export one agent without loading its population.

NEURAL02 is the native C++ little-endian checkpoint used on the reference run.
The individual JSON representation is portable and uses Python's standard library.
Exported slot zero is a deterministic example, not a selected best performer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO

STAT_NAMES = (
    "decisions", "neural_evaluations", "episodes", "correct", "go_trials",
    "go_correct", "nogo_trials", "nogo_correct", "nonprobe", "premature",
    "births", "deaths", "replaced_deaths", "injured_birth_replacements",
)
PERSON = struct.Struct("<QQQQQfBB")
MAGIC = b"NEURAL02"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _unpack(stream: BinaryIO, layout: str):
    spec = struct.Struct("<" + layout)
    data = stream.read(spec.size)
    if len(data) != spec.size:
        raise ValueError("Truncated checkpoint header")
    return spec.unpack(data)


def inspect_checkpoint(path: Path) -> dict:
    """Read small metadata and validate exact file length, without reading weights."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("Expected little-endian NEURAL02 checkpoint")
        population, hidden, seed, next_id = _unpack(stream, "QQQQ")
        if not 2 <= population <= 262144 or not 4 <= hidden <= 256:
            raise ValueError("Invalid checkpoint topology or byte order")
        mutation, mutation_scale, rng_state = _unpack(stream, "ddQ")
        stats = dict(zip(STAT_NAMES, _unpack(stream, "Q" * len(STAT_NAMES))))
        kind, severity, exposure, lesion_seed, injured_count = _unpack(stream, "QddQQ")
        if kind > 2 or injured_count > population:
            raise ValueError("Invalid checkpoint lesion metadata")
        for value in (mutation, mutation_scale, severity, exposure):
            if not math.isfinite(value):
                raise ValueError("Nonfinite checkpoint settings")
        if not 0 <= mutation <= 1 or mutation_scale < 0 or not 0 <= severity <= 1 or not 0 <= exposure <= 1:
            raise ValueError("Invalid checkpoint settings")
        people_offset = stream.tell() + injured_count * 8
        parameters = hidden * hidden + 12 * hidden + 3
        genomes_offset = people_offset + population * PERSON.size
        state_offset = genomes_offset + population * parameters * 4
        mask_offset = state_offset + population * hidden * 4
        expected_length = mask_offset + population * hidden + 8
        if path.stat().st_size != expected_length:
            raise ValueError("Checkpoint length does not match its declared topology")
        stream.seek(-8, 2)
        native_checksum, = _unpack(stream, "Q")
    return {
        "checkpoint_format": "NEURAL02", "byte_order": "little",
        "population": population, "hidden": hidden, "inputs": 8, "outputs": 3,
        "parameters_per_agent": parameters, "seed": seed, "next_id": next_id,
        "mutation": mutation, "mutation_scale": mutation_scale, "rng_state": rng_state,
        "stats": stats, "lesion": {"kind": ("none", "executive", "random")[kind],
            "severity": severity, "exposure": exposure, "seed": lesion_seed,
            "original_injured_count": injured_count},
        "offsets": {"people": people_offset, "genomes": genomes_offset,
            "state": state_offset, "mask": mask_offset},
        "checkpoint_bytes": expected_length,
        "native_checksum": f"{native_checksum:016x}",
        "native_checksum_algorithm": {"name": "FNV-1a-style 64-bit, simulator custom offset basis",
                                      "offset_basis": 1469598103934665603,
                                      "prime": 1099511628211},
        "validation": "header and exact length; native checksum not recomputed",
    }


def verify_native_checksum(path: Path) -> None:
    """Verify native FNV-1a-style checksum; linear pure-Python work can be slow."""
    remaining = Path(path).stat().st_size - 8
    value = 1469598103934665603
    with Path(path).open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Truncated checkpoint")
            for byte in block:
                value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
            remaining -= len(block)
        expected, = _unpack(stream, "Q")
    if value != expected:
        raise ValueError("Checkpoint native checksum mismatch")


def extract_agent(path: Path, slot: int = 0, *, verify_checksum: bool = False) -> dict:
    path = Path(path)
    metadata = inspect_checkpoint(path)
    if not 0 <= slot < metadata["population"]:
        raise ValueError("Slot is outside this population")
    if verify_checksum:
        verify_native_checksum(path)
    hidden, count = metadata["hidden"], metadata["parameters_per_agent"]
    with path.open("rb") as stream:
        stream.seek(metadata["offsets"]["people"] + slot * PERSON.size)
        values = _unpack(stream, "QQQQQfBB")
        person = dict(zip(("id", "parent", "born_at_decision", "episodes", "correct",
                           "energy", "alive", "task_ready"), values))
        stream.seek(metadata["offsets"]["genomes"] + slot * count * 4)
        weights = list(_unpack(stream, "f" * count))
        stream.seek(metadata["offsets"]["state"] + slot * hidden * 4)
        state = list(_unpack(stream, "f" * hidden))
        stream.seek(metadata["offsets"]["mask"] + slot * hidden)
        mask = list(stream.read(hidden))
    if any(not math.isfinite(v) for v in weights + state + [person["energy"]]):
        raise ValueError("Nonfinite agent parameters or state")
    if person["alive"] not in (0, 1) or person["task_ready"] not in (0, 1) or not 0 < person["id"] < metadata["next_id"]:
        raise ValueError("Invalid agent identity or status")
    if any(m not in (0, 1) or (m and state[i] != 0) for i, m in enumerate(mask)):
        raise ValueError("Invalid lesion mask or state")
    offset = 0
    blocks = []
    for name, shape in (("recurrent", [hidden, hidden]), ("input", [hidden, 8]),
                         ("hidden_bias", [hidden]), ("readout", [3, hidden]),
                         ("output_bias", [3])):
        size = math.prod(shape)
        blocks.append({"name": name, "shape": shape, "offset": offset, "length": size,
                       "order": "row-major"})
        offset += size
    metadata.pop("offsets")
    if verify_checksum:
        metadata["validation"] = "header, exact length, native custom-offset FNV-1a-style checksum"
    return {"schema": "lesion-lab-agent-v1", "selection": {"rule": "fixed resident slot",
        "slot": slot, "fitness_selected": False}, "checkpoint": metadata,
        "checkpoint_sha256": sha256_file(path), "topology": {"inputs": 8,
        "hidden": hidden, "outputs": 3, "activation": "tanh", "readout": "argmax",
        "actions": ["withhold", "left", "right"]}, "individual": person,
        "parameter_layout": blocks, "weights": weights, "state": state,
        "lesion_mask": mask,
        "numerics": "Stored IEEE-754 float32 values, represented as JSON numbers. Python inference uses double arithmetic and can differ near ties.",
        "learning": "Evolved weights; no gradient training or within-lifetime learning."}


def predict(model: dict, inputs: list[float], state: list[float] | None = None,
            *, reset: bool = False) -> tuple[int, list[float], list[float]]:
    """One recurrent inference step; returns action, next state, and readout scores."""
    if model.get("schema") != "lesion-lab-agent-v1":
        raise ValueError("Unsupported agent schema")
    hidden = int(model["topology"]["hidden"])
    expected = hidden * hidden + 12 * hidden + 3
    if len(inputs) != 8 or len(model["weights"]) != expected or len(model["lesion_mask"]) != hidden:
        raise ValueError("Input, weights or lesion mask shape mismatch")
    previous = [0.0] * hidden if reset else list(model["state"] if state is None else state)
    if len(previous) != hidden or any(not math.isfinite(float(x)) for x in [*inputs, *previous]):
        raise ValueError("Invalid input or recurrent state")
    w, mask = model["weights"], model["lesion_mask"]
    io = hidden * hidden
    bo = io + hidden * 8
    oo, ob = bo + hidden, bo + hidden + 3 * hidden
    next_state = [0.0 if mask[j] else math.tanh(w[bo+j] +
        sum(w[j*hidden+k] * previous[k] for k in range(hidden)) +
        sum(w[io+j*8+k] * inputs[k] for k in range(8))) for j in range(hidden)]
    scores = [w[ob+a] + sum(w[oo+a*hidden+j] * next_state[j] for j in range(hidden))
              for a in range(3)]
    return max(range(3), key=scores.__getitem__), next_state, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "extract"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--verify-native-checksum", action="store_true")
        if name == "extract":
            command.add_argument("--slot", type=int, default=0)
            command.add_argument("--output", required=True, type=Path)
    inference = sub.add_parser("predict")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--input", required=True, help="JSON array of eight input values")
    inference.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.command == "predict":
        model = json.loads(args.model.read_text())
        action, state, scores = predict(model, json.loads(args.input), reset=args.reset)
        print(json.dumps({"action": action, "action_name": model["topology"]["actions"][action],
                          "state": state, "scores": scores}, allow_nan=False))
    elif args.command == "inspect":
        value = inspect_checkpoint(args.checkpoint)
        if args.verify_native_checksum:
            verify_native_checksum(args.checkpoint)
            value["validation"] = "header, exact length, native custom-offset FNV-1a-style checksum"
        print(json.dumps(value, indent=2))
    else:
        value = extract_agent(args.checkpoint, args.slot, verify_checksum=args.verify_native_checksum)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "slot": args.slot,
                          "parameters": len(value["weights"]), "checkpoint_sha256": value["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
