"""Portable model extraction, native inference parity and release integrity checks."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from model_io import (extract_agent, inspect_checkpoint, predict, sha256_file,
                      verify_native_checksum)
from package_release import package, reassemble, refresh, sanitize, split_asset
import package_release

ROOT = Path(__file__).resolve().parents[1]
MASK64 = (1 << 64) - 1


def mix(value):
    value = (value + 0x9e3779b97f4a7c15) & MASK64
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK64
    return value ^ (value >> 31)


def test_checkpoint(path):
    """A small format fixture with known state and distinct resident genomes."""
    hidden, population = 4, 2
    parameters = hidden * hidden + 12 * hidden + 3
    raw = bytearray(b"NEURAL02")
    raw.extend(struct.pack("<QQQQddQ", population, hidden, 42, 3, .015, .05, 123))
    raw.extend(struct.pack("<" + "Q" * 14, *([0] * 14)))
    raw.extend(struct.pack("<QddQQ", 0, 0, 0, 0, 0))
    for slot in range(population):
        raw.extend(struct.pack("<QQQQQfBB", slot + 1, 0, 0, 0, 0, 48, 1, 0))
    for slot in range(population):
        weights = [0.] * parameters
        weights[-3 + slot] = 2.
        raw.extend(struct.pack("<" + "f" * parameters, *weights))
    raw.extend(struct.pack("<" + "f" * population * hidden, *([0.] * population * hidden)))
    raw.extend(bytes(population * hidden))
    checksum = 1469598103934665603
    for byte in raw:
        checksum = ((checksum ^ byte) * 1099511628211) & MASK64
    raw.extend(struct.pack("<Q", checksum))
    path.write_bytes(raw)


class ModelIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-io-test-")
        self.path = Path(self.temp.name)
        self.checkpoint = self.path / "warmup.bin"
        test_checkpoint(self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def test_extract_distinct_slots_and_recurrent_prediction(self):
        info = inspect_checkpoint(self.checkpoint)
        self.assertEqual(info["population"], 2)
        self.assertEqual(info["parameters_per_agent"], 67)
        self.assertEqual(info["checkpoint_bytes"], self.checkpoint.stat().st_size)
        first = extract_agent(self.checkpoint, 0, verify_checksum=True)
        second = extract_agent(self.checkpoint, 1)
        self.assertEqual(first["individual"]["id"], 1)
        self.assertEqual(second["individual"]["id"], 2)
        self.assertFalse(first["selection"]["fitness_selected"])
        self.assertEqual(sum(block["length"] for block in first["parameter_layout"]), 67)
        self.assertEqual(predict(first, [0.] * 8)[0], 0)
        self.assertEqual(predict(second, [0.] * 8)[0], 1)
        self.assertEqual(first["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        first["weights"][48] = 3.  # hidden bias begins after 16 recurrent + 32 input weights
        self.assertGreater(predict(first, [0.] * 8)[1][0], .9)
        first["lesion_mask"][0] = 1
        self.assertEqual(predict(first, [0.] * 8)[1][0], 0.)

    def test_corruption_truncation_slot_and_shape_fail_closed(self):
        with self.assertRaises(ValueError):
            extract_agent(self.checkpoint, 2)
        with self.assertRaises(ValueError):
            predict(extract_agent(self.checkpoint), [0.])
        raw = self.checkpoint.read_bytes()
        self.checkpoint.write_bytes(raw[:-1])
        with self.assertRaises(ValueError):
            inspect_checkpoint(self.checkpoint)
        self.checkpoint.write_bytes(raw[:300] + bytes([raw[300] ^ 1]) + raw[301:])
        with self.assertRaises(ValueError):
            verify_native_checksum(self.checkpoint)

    def test_release_has_full_weights_portable_agent_and_sanitized_metadata(self):
        (self.path / "manifest.json").write_text(json.dumps({"status": "complete", "population": 2,
            "arms": {}, "resume_from": str(ROOT / "runs/private-path/warmup.bin"),
            "environment": {"author": "person@example.com"}}))
        output = self.path / "release"
        result = package(self.path, output, minimum_free_gib=0)
        self.assertEqual(result["population_assets"], 1)
        asset = output / "baseline-population-neural02.bin.gz"
        self.assertEqual(gzip.decompress(asset.read_bytes()), self.checkpoint.read_bytes())
        public_manifest = (output / "results/manifest.json").read_text()
        self.assertNotIn(str(ROOT), public_manifest)
        self.assertNotIn("person@example.com", public_manifest)
        exported = json.loads((output / "models/baseline-agent-slot-0.json").read_text())
        self.assertEqual(len(exported["weights"]), 67)
        with tarfile.open(output / "lesion-lab-source-results.tar.gz") as archive:
            self.assertIn("lesion-lab/neural_engine.cpp", archive.getnames())
            self.assertIn("lesion-lab/release/models/baseline-agent-slot-0.json", archive.getnames())
            self.assertTrue(all(member.mtime == 0 and not member.name.startswith("/") for member in archive))
        for line in (output / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split("  ", 1)
            self.assertEqual(sha256_file(output / name), expected)

    def test_chunk_reassembly_verifies_each_chunk_and_cleans_failure(self):
        source = self.path / "population.bin.gz"
        source.write_bytes(bytes(range(256)) * 9000)
        record = split_asset(source, 1024 * 1024)
        manifest = self.path / "chunks.json"
        manifest.write_text(json.dumps({"assets": [record]}))
        restored = reassemble(manifest, self.path / "restored")[0]
        self.assertEqual(sha256_file(restored), sha256_file(source))
        first = self.path / record["parts"][0]["filename"]
        first.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            reassemble(manifest, self.path / "broken")
        self.assertEqual(list((self.path / "broken").iterdir()), [])
        self.assertEqual(sanitize("/Users/anonymous/Documents/file"), "<home>/Documents/file")

    def test_source_bundle_includes_config_ci_and_analysis_with_matching_public_hashes(self):
        source = self.path / "source"
        source.mkdir()
        (source / "Makefile").write_text("test:\n\tpython3 -m unittest discover\n")
        (source / "experiment.json").write_text('{"population":2}')
        (source / ".github/workflows").mkdir(parents=True)
        (source / ".github/workflows/test.yml").write_text("name: test\n")
        (source / "results/latest").mkdir(parents=True)
        (source / "results/latest/analysis.json").write_text('{"episodes":100}')
        (source / "docs").mkdir()
        (source / "docs/cleanup-record.json").write_text(json.dumps({
            "path": "/Users/privateperson/Documents/runs", "email": "privateperson@example.com"}))
        code = b'print("unchanged implementation")\n'
        (source / "example.py").write_bytes(code)
        (self.path / "manifest.json").write_text('{"status":"complete","arms":{}}')
        output = self.path / "public"
        with patch.object(package_release, "ROOT", source):
            package(self.path, output, minimum_free_gib=0)
        release = json.loads((output / "release_manifest.json").read_text())
        with tarfile.open(output / "lesion-lab-source-results.tar.gz") as archive:
            for name in ("Makefile", "experiment.json", ".github/workflows/test.yml", "results/latest/analysis.json"):
                self.assertIn("lesion-lab/" + name, archive.getnames())
            public = archive.extractfile("lesion-lab/docs/cleanup-record.json").read()
            self.assertNotIn(b"privateperson", public)
            self.assertEqual(archive.extractfile("lesion-lab/example.py").read(), code)
            for name, expected in release["source_sha256"].items():
                self.assertEqual(hashlib.sha256(archive.extractfile("lesion-lab/" + name).read()).hexdigest(), expected)

    def test_refresh_updates_comparison_without_rewriting_models_and_rejects_changed_checkpoint(self):
        source = self.path / "source"
        source.mkdir()
        (source / "README.md").write_text("Initial documentation.\n")
        (self.path / "manifest.json").write_text('{"status":"complete","arms":{}}')
        output = self.path / "public"
        with patch.object(package_release, "ROOT", source):
            package(self.path, output, minimum_free_gib=0)
            old_release = json.loads((output / "release_manifest.json").read_text())
            record = old_release["model_assets"][0]
            population = output / record["filename"]
            example = output / record["individual_json"]
            original_population = (sha256_file(population), population.stat().st_mtime_ns)
            original_example = (sha256_file(example), example.stat().st_mtime_ns)
            (source / "README.md").write_text("Updated documentation linking the comparison.\n")
            (source / "BFF_COMPARISON.md").write_text("A dedicated comparison with the original whitepaper.\n")
            with patch.object(package_release, "gzip_file", side_effect=AssertionError("Population must not be recompressed")), \
                    patch.object(package_release, "extract_agent", side_effect=AssertionError("Portable models must not be regenerated")):
                result = refresh(self.path, output, minimum_free_gib=0)
            self.assertEqual(result["population_assets_unchanged"], 1)
            self.assertEqual((sha256_file(population), population.stat().st_mtime_ns), original_population)
            self.assertEqual((sha256_file(example), example.stat().st_mtime_ns), original_example)
            updated = json.loads((output / "release_manifest.json").read_text())
            self.assertNotEqual(updated["source_version_sha256"], old_release["source_version_sha256"])
            self.assertEqual(updated["model_assets"], old_release["model_assets"])
            with tarfile.open(output / "lesion-lab-source-results.tar.gz") as archive:
                self.assertIn(b"original whitepaper", archive.extractfile("lesion-lab/BFF_COMPARISON.md").read())
                self.assertNotIn("lesion-lab/release/SHA256SUMS", archive.getnames())
                for name, expected in updated["source_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile("lesion-lab/" + name).read()).hexdigest(), expected)
            for line in (output / "SHA256SUMS").read_text().splitlines():
                expected, name = line.split("  ", 1)
                self.assertEqual(sha256_file(output / name), expected)
            preserved_manifest = (output / "release_manifest.json").read_bytes()
            preserved_archive = sha256_file(output / "lesion-lab-source-results.tar.gz")
            raw = bytearray(self.checkpoint.read_bytes())
            raw[300] ^= 1
            self.checkpoint.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Local checkpoint no longer matches"):
                refresh(self.path, output, minimum_free_gib=0)
            self.assertEqual((output / "release_manifest.json").read_bytes(), preserved_manifest)
            self.assertEqual(sha256_file(output / "lesion-lab-source-results.tar.gz"), preserved_archive)


class NativeExportParityTests(unittest.TestCase):
    def test_exported_agents_match_native_frozen_assay(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++17 compiler required for native parity")
        with tempfile.TemporaryDirectory(prefix="native-model-parity-") as temp:
            root = Path(temp)
            binary, checkpoint = root / "engine", root / "model.bin"
            subprocess.run([compiler, "-std=c++17", "-O2", str(ROOT / "neural_engine.cpp"),
                            "-o", str(binary)], check=True, capture_output=True, timeout=60)
            subprocess.run([str(binary), "run", "--population", "4", "--hidden", "8",
                "--decisions", "1200", "--checkpoint", str(checkpoint)],
                check=True, capture_output=True, timeout=30)
            native = json.loads(subprocess.run([str(binary), "assay", "--load", str(checkpoint),
                "--samples", "4", "--trials", "32", "--seed", "765"],
                check=True, capture_output=True, text=True, timeout=30).stdout)
            correct, trials = 0, 0
            for slot in native["sample_slots"]:
                model = extract_agent(checkpoint, slot, verify_checksum=True)
                for episode in range(32):
                    draw = lambda c: mix(765 ^ mix(episode + 0x5000) ^ mix(slot + 0x9000) ^ mix(c + 0xd000))
                    cue, rule, go = bool(draw(0) & 1), bool(draw(1) & 1), bool(draw(2) & 3)
                    state = None
                    for phase in range(8):
                        inputs = [0.] * 8
                        if phase == 0:
                            inputs[int(cue)] = 1.
                        elif phase == 7:
                            inputs[3:6] = [1., 1. if rule else -1., 1. if go else -1.]
                        else:
                            inputs[2] = 1.
                            inputs[6], inputs[7] = float(draw(8 + phase) & 3 == 0), float(draw(24 + phase) & 3 == 0)
                        action, state, _ = predict(model, inputs, state, reset=phase == 0)
                    target = (2 if cue ^ rule else 1) if go else 0
                    correct += action == target
                    trials += 1
            self.assertEqual(trials, native["episodes"])
            self.assertEqual(correct, native["correct"])


if __name__ == "__main__":
    unittest.main()
