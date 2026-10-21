import importlib.util
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("neural_lab", ROOT / "neural_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


class NeuralOrchestrationTests(unittest.TestCase):
    def test_resume_hash_streams_and_episode_target_uses_scored_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "checkpoint.bin"
            contents = bytes(range(256)) * 10000
            checkpoint.write_bytes(contents)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("Whole-checkpoint reads are forbidden")):
                self.assertEqual(lab.sha256_file(checkpoint), hashlib.sha256(contents).hexdigest())
        manifest = {"warmup_final": {"decisions": 100_000_000, "episodes": 9_999_999}}
        self.assertFalse(lab.check_episode_target(manifest, 10_000_000)["passed"])
        manifest["warmup_final"]["episodes"] += 1
        self.assertTrue(lab.check_episode_target(manifest, 10_000_000)["passed"])
        self.assertEqual(manifest["warmup_episode_target"]["actual"], 10_000_000)

    @unittest.skipUnless((ROOT / "build" / "neural_engine").exists(), "Native neural engine is not built")
    def test_unmet_episode_target_preserves_checkpoint_and_prevents_lesions(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "run"
            result = subprocess.run([sys.executable, str(ROOT / "neural_lab.py"), "run",
                "--out", str(out), "--population", "8", "--hidden", "4",
                "--decisions", "64", "--minimum-episodes", "100",
                "--cpu-seconds", "1200"], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("complete scored episodes", result.stdout)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "episode_target_not_met")
            self.assertFalse(manifest["warmup_episode_target"]["passed"])
            self.assertTrue((out / "warmup.bin").is_file())
            self.assertFalse((out / "sham").exists())
            self.assertFalse((out / "baseline_assay.json").exists())

    def test_population_weight_count_and_storage_budget(self):
        self.assertEqual(lab.parameter_count(64), 4867)
        self.assertEqual(16384 * lab.parameter_count(64), 79_740_928)
        self.assertGreater(lab.storage_estimate(16384, 64), 7 * 16384 * 4867 * 4)

    def test_cautious_default_fits_memory_and_disk_caps_but_large_run_is_gated(self):
        self.assertLess(lab.memory_estimate(4096, 64), 256 * 1024**2)
        self.assertGreater(lab.memory_estimate(16384, 64), 256 * 1024**2)
        self.assertLess(lab.storage_estimate(4096, 64), 1024**3)
        self.assertGreater(lab.storage_estimate(16384, 64), 1024**3)

    def test_report_states_reproduction_and_evolution_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = {"population": 16, "hidden": 8, "status": "complete", "baseline_assay": {"accuracy": .375, "go_accuracy": .5, "nogo_accuracy": 0}, "arms": {}}
            lab.generate_report(Path(folder), manifest)
            text = (Path(folder) / "report.html").read_text()
            self.assertIn("Birth is supplied by the simulator", text)
            self.assertIn("no gradient training or lifetime relearning", text)
            self.assertIn("no established executive specialization", text)
            self.assertIn("37.50%", text)
            self.assertTrue((Path(folder) / "summary.txt").exists())

    def test_limits_apply_to_child_without_changing_parent(self):
        before = resource.getrlimit(resource.RLIMIT_CPU)
        code = "import json,resource; print(json.dumps({k:resource.getrlimit(getattr(resource,k)) for k in ['RLIMIT_CPU','RLIMIT_CORE','RLIMIT_FSIZE']}))"
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, preexec_fn=lab.native_limits)
        limits = json.loads(child.stdout)
        self.assertEqual(limits["RLIMIT_CPU"], [120, 125])
        self.assertEqual(limits["RLIMIT_CORE"], [0, 0])
        self.assertEqual(limits["RLIMIT_FSIZE"], [1024**3, 1024**3])
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CPU), before)

    def test_interrupted_stage_stops_its_child(self):
        with tempfile.TemporaryDirectory() as folder:
            stub = Path(folder) / "slow_engine"
            pid_path = Path(folder) / "pid"
            stub.write_text(f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\nPath({str(pid_path)!r}).write_text(str(os.getpid()))\nprint('{{\"decisions\":0}}',flush=True)\ntime.sleep(30)\n")
            stub.chmod(0o755)
            decoder = json.loads
            def stop_after_child_is_ready(line):
                value = decoder(line)
                if "decisions" in value:
                    raise TimeoutError("test deadline after the child reports readiness")
                return value
            with patch.object(lab, "ENGINE", stub), patch.object(lab.json, "loads", side_effect=stop_after_child_is_ready), self.assertRaises(TimeoutError):
                lab.stream_native([], "test", 0, 10)
            pid = int(pid_path.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
