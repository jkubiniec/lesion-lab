#!/usr/bin/env python3
"""Create public results, portable agents, and full population release assets.

All checkpoint hashing and compression streams in bounded memory. Full population
assets belong in a GitHub Release, not ordinary Git history. Optional chunks support
small-file distribution and verified reassembly with this same script.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

from model_io import extract_agent, inspect_checkpoint, sha256_file

ROOT = Path(__file__).resolve().parent
BLOCK = 1024 * 1024


def sanitize(value):
    """Remove workstation paths from metadata/logs without changing numeric results."""
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = value.replace(str(ROOT), ".")
        value = re.sub(r"/(?:Users|home)/[^/\s\"'<>]+", "<home>", value)
        value = re.sub(r"/private/(?:tmp|var/folders)/[^\s\"'<>]+", "<temporary-path>", value)
        value = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "<redacted-email>", value, flags=re.I)
        return value
    return value


def compact_population_records(value):
    """Keep aggregate results small; all resident records remain in model assets."""
    if isinstance(value, dict):
        result = {key: compact_population_records(item) for key, item in value.items()
                  if key != "individuals"}
        if isinstance(value.get("individuals"), list):
            result["individual_records"] = {"count": len(value["individuals"]),
                "storage": "Full resident identity, ancestry, energy and task records are in the corresponding population checkpoint."}
        return result
    if isinstance(value, list):
        return [compact_population_records(item) for item in value]
    return value


def public_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".json":
        return (json.dumps(sanitize(compact_population_records(json.loads(raw))), indent=2, allow_nan=False) + "\n").encode()
    if path.suffix == ".jsonl":
        return ("\n".join(json.dumps(sanitize(json.loads(line)), allow_nan=False)
                          for line in raw.decode().splitlines() if line.strip()) + "\n").encode()
    text = raw.decode()
    if path.suffix == ".html":
        # The generated simulator report embeds every resident in its expandable
        # audit block. The public checkpoint retains those records; link to the
        # compact aggregate manifest so the report stays small and easy to open.
        text = re.sub(r"<details><summary>Parameters and measured outcomes</summary><pre>.*?</pre></details>",
                      '<details><summary>Parameters and measured outcomes</summary><p><a href="manifest.json">Download aggregate manifest</a>. Full individual records are in the population checkpoints.</p></details>',
                      text, flags=re.S)
    return sanitize(text).encode()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gzip_file(source: Path, target: Path, level: int = 1) -> None:
    with source.open("rb") as src, target.open("wb") as dst:
        with gzip.GzipFile(filename="", fileobj=dst, mode="wb", compresslevel=level, mtime=0) as zipped:
            shutil.copyfileobj(src, zipped, length=BLOCK)


def _source_files() -> list[Path]:
    files = [path for path in ROOT.iterdir() if path.is_file() and
             (path.suffix in (".py", ".cpp", ".h", ".md", ".html", ".command", ".sh", ".json", ".toml", ".yml", ".yaml", ".txt")
              or path.name in ("LICENSE", ".gitignore", "Makefile"))]
    for folder in ("tests", "docs", "figures", ".github", "results"):
        parent = ROOT / folder
        if parent.exists():
            files.extend(path for path in parent.rglob("*") if path.is_file() and
                         "__pycache__" not in path.parts and path.suffix not in
                         (".pyc", ".bin", ".nrl", ".gz", ".npz", ".pt", ".pth", ".safetensors")
                         and not re.search(r"\.part\d+$", path.name))
    return sorted(files)


def public_source_bytes(path: Path) -> bytes:
    """Sanitize public prose/metadata; keep implementation files byte exact."""
    if path.suffix in (".md", ".html", ".json", ".jsonl", ".txt", ".csv", ".tsv", ".svg", ".yml", ".yaml"):
        return public_bytes(path)
    return path.read_bytes()


def _tar_entry(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(payload), 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def source_catalog() -> tuple[list[Path], dict, str]:
    sources = _source_files()
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(public_source_bytes(path)).hexdigest()
              for path in sources}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return sources, hashes, version


def write_source_archive(run: Path, output: Path, sources: list[Path], compresslevel: int) -> None:
    """Replace only the small archive; never rewrite existing population assets."""
    archive_path = output / "lesion-lab-source-results.tar.gz"
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    try:
        with temporary.open("wb") as destination:
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb", mtime=0,
                               compresslevel=compresslevel) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|") as archive:
                    for path in sources:
                        _tar_entry(archive, "lesion-lab/" + path.relative_to(ROOT).as_posix(), public_source_bytes(path))
                    for path in sorted(output.rglob("*")):
                        if path.is_file() and path.suffix not in (".gz", ".tmp") and path.name != "SHA256SUMS" and not re.search(r"\.part\d+$", path.name):
                            _tar_entry(archive, "lesion-lab/release/" + path.relative_to(output).as_posix(), path.read_bytes())
                    # Preserve exact implementation bytes used for the experiment.
                    if (run / "source").exists():
                        for path in sorted((run / "source").rglob("*")):
                            if path.is_file():
                                _tar_entry(archive, "lesion-lab/release/run-source/" + path.relative_to(run / "source").as_posix(), path.read_bytes())
        temporary.replace(archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_checksums(output: Path) -> list[Path]:
    paths = sorted(path for path in output.rglob("*") if path.is_file() and
                   path.name != "SHA256SUMS" and not path.name.endswith(".tmp"))
    temporary = output / "SHA256SUMS.tmp"
    temporary.write_text("".join(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n" for path in paths))
    temporary.replace(output / "SHA256SUMS")
    return paths


def write_model_card(output: Path, release: dict, slot: int) -> None:
    model_card = ROOT / "MODEL_CARD.md"
    if model_card.exists():
        (output / "MODEL_CARD.md").write_bytes(public_bytes(model_card))
    else:
        (output / "MODEL_CARD.md").write_text("# Evolved recurrent population\n\n" + release["training"] +
            "\n\n" + "\n".join("- " + value for value in release["limitations"]) +
            f"\n\nThe portable example is predetermined slot {slot}; it was not selected by fitness.\n")
    with (output / "MODEL_CARD.md").open("a") as card:
        card.write(f"\nPortable examples use fixed resident slot {slot} without selecting for fitness or injury status. "
                   "The example can therefore be uninjured in an injury-condition snapshot. Population lesion comparisons use the full population, not this single example. "
                   "JSON weights store float32 values; portable Python inference uses double arithmetic and may differ near decision ties. Native checkpoints do not promise cross-platform bit-identical replay.\n")


def split_asset(path: Path, size_bytes: int) -> dict:
    if size_bytes < BLOCK:
        raise ValueError("Chunk size must be at least 1 MiB")
    record = {"filename": path.name, "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "parts": []}
    with path.open("rb") as source:
        index = 1
        while True:
            block = source.read(min(BLOCK, size_bytes))
            if not block:
                break
            part = path.with_name(path.name + f".part{index:04d}")
            digest = hashlib.sha256()
            written = 0
            with part.open("wb") as target:
                while block:
                    target.write(block)
                    digest.update(block)
                    written += len(block)
                    if written == size_bytes:
                        break
                    block = source.read(min(BLOCK, size_bytes - written))
            record["parts"].append({"filename": part.name, "bytes": written,
                                     "sha256": digest.hexdigest()})
            index += 1
    return record


def reassemble(manifest_path: Path, output: Path) -> list[Path]:
    records = json.loads(manifest_path.read_text())["assets"]
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for record in records:
        name = record["filename"]
        if Path(name).name != name:
            raise ValueError("Unsafe asset filename")
        target = output / name
        if target.exists():
            raise ValueError(f"Output already exists: {name}")
        temporary = target.with_suffix(target.suffix + ".tmp")
        digest, total = hashlib.sha256(), 0
        try:
            with temporary.open("xb") as result:
                for part in record["parts"]:
                    partname = part["filename"]
                    if Path(partname).name != partname:
                        raise ValueError("Unsafe chunk filename")
                    path = manifest_path.parent / partname
                    part_digest, length = hashlib.sha256(), 0
                    with path.open("rb") as stream:
                        while block := stream.read(BLOCK):
                            part_digest.update(block)
                            digest.update(block)
                            length += len(block)
                            total += len(block)
                            result.write(block)
                    if length != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
                        raise ValueError(f"Chunk integrity check failed: {partname}")
            if total != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Reassembled asset integrity check failed: {name}")
            temporary.replace(target)
            generated.append(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return generated


def package(run: Path, output: Path, *, include_branches: bool = True,
            slot: int = 0, chunk_size_mib: int = 0, compresslevel: int = 1,
            minimum_free_gib: float = 5.0) -> dict:
    run, output = run.resolve(), output.resolve()
    manifest = compact_population_records(json.loads((run / "manifest.json").read_text()))
    if manifest.get("status") != "complete":
        raise ValueError("Publish only a completed experiment")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty release output directory")
    snapshots = [("baseline", run / "warmup.bin")]
    if include_branches:
        snapshots.extend((name + "-final", run / name / "final.bin")
                         for name in ("sham", "designated_region", "random")
                         if name in manifest.get("arms", {}))
    for _, checkpoint in snapshots:
        inspect_checkpoint(checkpoint)
    # Compression can slightly enlarge incompressible inputs. Splitting temporarily
    # requires one extra compressed asset, but never one extra whole population in RAM.
    checkpoint_bytes = sum(path.stat().st_size for _, path in snapshots)
    scratch_bytes = max(path.stat().st_size for _, path in snapshots) if chunk_size_mib else 0
    required = int((checkpoint_bytes + scratch_bytes) * 1.02) + 64 * BLOCK
    free = shutil.disk_usage(output.parent if output.parent.exists() else run).free
    if free < required + minimum_free_gib * 1024**3:
        raise ValueError("Packaging would violate the minimum-free-disk reserve")
    output.mkdir(parents=True, exist_ok=True)
    public_run = output / "results"
    public_run.mkdir()
    for path in sorted(run.rglob("*")):
        if not path.is_file() or "source" in path.relative_to(run).parts:
            continue
        if path.suffix not in (".json", ".jsonl", ".txt", ".html"):
            continue
        if path.stat().st_size > 128 * BLOCK:
            raise ValueError(f"Unexpectedly large metadata file: {path.name}")
        target = public_run / path.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(public_bytes(path))
    model_dir = output / "models"
    model_dir.mkdir()
    artifacts, chunks = [], []
    for label, checkpoint in snapshots:
        print(json.dumps({"stage": "model_export", "snapshot": label}), flush=True)
        exported = extract_agent(checkpoint, slot)
        agent_path = model_dir / f"{label}-agent-slot-{slot}.json"
        write_json(agent_path, exported)
        name = f"{label}-population-neural02.bin.gz"
        target = output / name
        gzip_file(checkpoint, target, compresslevel)
        record = {"snapshot": label, "filename": name, "bytes": target.stat().st_size,
                  "sha256": sha256_file(target), "uncompressed_bytes": checkpoint.stat().st_size,
                  "uncompressed_sha256": exported["checkpoint_sha256"],
                  "individual_json": agent_path.relative_to(output).as_posix(),
                  "selection_rule": "fixed resident slot; no fitness or injury-status selection",
                  "portable_example_injured": any(exported["lesion_mask"]),
                  "portable_example_alive": bool(exported["individual"]["alive"])}
        artifacts.append(record)
        if chunk_size_mib:
            chunks.append(split_asset(target, chunk_size_mib * BLOCK))
            target.unlink()
    if chunks:
        write_json(output / "chunks.json", {"schema": "lesion-lab-chunks-v1", "assets": chunks,
                   "reassembly": "python3 package_release.py reassemble --manifest chunks.json --output restored"})
    sources, source_hashes, source_version = source_catalog()
    release = {"schema": "lesion-lab-release-v1", "experiment": sanitize(manifest),
        "source_version_sha256": source_version, "source_sha256": source_hashes,
        "source_hash_scope": "Exact published source-archive member bytes. C++/Python are unchanged; prose and metadata remove workstation paths and email addresses. SHA256SUMS verifies published assets; run-source hashes may refer to unsanitized original inputs.",
        "model_assets": artifacts, "checkpoint_format": "NEURAL02, native little-endian",
        "portable_agent_format": "lesion-lab-agent-v1, JSON with IEEE-754 float32 values",
        "representative_agent": {"selection": "predetermined resident slot", "slot": slot,
            "fitness_selected": False, "caution": "One resident is not a population aggregate or a best model. Slot zero can be uninjured even in an injury-condition snapshot; selection does not depend on injury status."},
        "training": "Evolutionary selection and mutation across births; no gradient training or lifetime relearning.",
        "limitations": ["Single seed; no independent evolutionary replicates.",
            "Injury is computational unit masking; no validated biological TBI mapping.",
            "The final-quarter region has no established executive specialization.",
            "Population improvement can reflect replacement; no individual lesion repair is implemented."]}
    write_json(output / "release_manifest.json", release)
    write_model_card(output, release, slot)
    write_source_archive(run, output, sources, compresslevel)
    checksummed = write_checksums(output)
    return {"status": "complete", "population_assets": len(artifacts),
            "source_version_sha256": source_version, "release": str(output),
            "disk_bytes": sum(path.stat().st_size for path in checksummed)}


def refresh(run: Path, output: Path, *, compresslevel: int = 1,
            minimum_free_gib: float = 5.0) -> dict:
    """Refresh source/prose without rewriting already published population assets.

    Existing gzip assets, checkpoint contents, and portable JSON examples are
    verified against the original release's pinned hashes before any mutation.
    Checkpoint verification streams SHA256 and reads only their small headers.
    """
    run, output = run.resolve(), output.resolve()
    release_path = output / "release_manifest.json"
    release = json.loads(release_path.read_text())
    if release.get("schema") != "lesion-lab-release-v1" or release.get("experiment", {}).get("status") != "complete":
        raise ValueError("Refresh requires a completed, previously packaged release")
    manifest = compact_population_records(json.loads((run / "manifest.json").read_text()))
    if manifest.get("status") != "complete":
        raise ValueError("Refresh requires a completed source experiment")
    original = release["experiment"]
    for key in ("population", "hidden", "seed", "resumed_checkpoint_sha256", "source_sha256"):
        if key in original and manifest.get(key) != original[key]:
            raise ValueError(f"Source experiment provenance changed: {key}")
    checksum_path = output / "SHA256SUMS"
    prior_checksums = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        prior_checksums[name] = digest
    artifacts = release.get("model_assets", [])
    if not artifacts:
        raise ValueError("No population assets are pinned by this release")
    validated = []
    for record in artifacts:
        name = record["filename"]
        if Path(name).name != name:
            raise ValueError("Unsafe population asset filename")
        asset = output / name
        if not asset.is_file():
            raise ValueError("Refresh expects unsplit population gzip assets; reassemble chunks first")
        if asset.stat().st_size != record["bytes"] or sha256_file(asset) != record["sha256"] or prior_checksums.get(name) != record["sha256"]:
            raise ValueError(f"Existing population asset integrity check failed: {name}")
        label = record["snapshot"]
        if label == "baseline":
            checkpoint, expected = run / "warmup.bin", original.get("warmup_final", {})
        elif label in ("sham-final", "designated_region-final", "random-final"):
            arm = label.removesuffix("-final")
            checkpoint = run / arm / "final.bin"
            expected = original.get("arms", {}).get(arm, {}).get("final", {})
        else:
            raise ValueError(f"Unsupported snapshot label: {label}")
        info = inspect_checkpoint(checkpoint)
        if checkpoint.stat().st_size != record["uncompressed_bytes"] or sha256_file(checkpoint) != record["uncompressed_sha256"]:
            raise ValueError(f"Local checkpoint no longer matches the packaged model: {label}")
        for key, actual in info["stats"].items():
            if key in expected and actual != expected[key]:
                raise ValueError(f"Checkpoint counters do not match the original experiment: {label}/{key}")
        agent_name = record["individual_json"]
        agent_path = (output / agent_name).resolve()
        if not agent_path.is_relative_to(output) or not agent_path.is_file():
            raise ValueError("Unsafe or missing portable example path")
        if sha256_file(agent_path) != prior_checksums.get(agent_name):
            raise ValueError(f"Portable example integrity check failed: {agent_name}")
        example = json.loads(agent_path.read_text())
        if example.get("checkpoint_sha256") != record["uncompressed_sha256"]:
            raise ValueError(f"Portable example checkpoint provenance mismatch: {agent_name}")
        validated.append({"snapshot": label, "checkpoint_sha256": record["uncompressed_sha256"],
                          "asset_sha256": record["sha256"]})
        print(json.dumps({"stage": "verified_existing_model", "snapshot": label}), flush=True)
    sources, source_hashes, source_version = source_catalog()
    # Reserve enough scratch for an uncompressed upper bound of the new small
    # source/results archive. Existing model gzip assets are never duplicated.
    archive_inputs = [path for path in output.rglob("*") if path.is_file() and
                      path.suffix not in (".gz", ".tmp") and path.name != "SHA256SUMS"]
    required = sum(path.stat().st_size for path in [*sources, *archive_inputs]) + 32 * BLOCK
    if shutil.disk_usage(output).free < required + minimum_free_gib * 1024**3:
        raise ValueError("Refreshing the source archive would violate the free-disk reserve")
    release["source_sha256"] = source_hashes
    release["source_version_sha256"] = source_version
    release["source_refresh"] = {
        "model_assets": "Verified unchanged against previously pinned compressed and uncompressed SHA256 values; not recompressed.",
        "portable_examples": "Verified against prior SHA256SUMS and checkpoint provenance; not regenerated.",
        "experiment_results": "Previously published compact numeric results retained unchanged.",
        "validated_snapshots": validated}
    write_json(release_path, release)
    write_model_card(output, release, int(release["representative_agent"]["slot"]))
    write_source_archive(run, output, sources, compresslevel)
    checksummed = write_checksums(output)
    return {"status": "complete", "mode": "source_refresh", "population_assets_unchanged": len(validated),
            "source_version_sha256": source_version, "release": str(output),
            "disk_bytes": sum(path.stat().st_size for path in checksummed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--slot", type=int, default=0)
    build.add_argument("--baseline-only", action="store_true")
    build.add_argument("--chunk-size-mib", type=int, default=0)
    build.add_argument("--compresslevel", type=int, choices=range(1, 10), default=1)
    build.add_argument("--minimum-free-gib", type=float, default=5.0)
    update = commands.add_parser("refresh", help="Refresh source/prose and checksums while retaining exact population asset bytes")
    update.add_argument("--run", type=Path, required=True)
    update.add_argument("--output", type=Path, required=True)
    update.add_argument("--compresslevel", type=int, choices=range(1, 10), default=1)
    update.add_argument("--minimum-free-gib", type=float, default=5.0)
    restore = commands.add_parser("reassemble")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reassemble":
        print(json.dumps({"restored": [str(p) for p in reassemble(args.manifest, args.output)]}))
    elif args.command == "refresh":
        if args.minimum_free_gib < 0:
            parser.error("Free disk reserve must be nonnegative")
        print(json.dumps(refresh(args.run, args.output, compresslevel=args.compresslevel,
                                 minimum_free_gib=args.minimum_free_gib)), flush=True)
    else:
        if args.chunk_size_mib < 0 or args.minimum_free_gib < 0:
            parser.error("Chunk size and free disk reserve must be nonnegative")
        print(json.dumps(package(args.run, args.output, include_branches=not args.baseline_only,
            slot=args.slot, chunk_size_mib=args.chunk_size_mib, compresslevel=args.compresslevel,
            minimum_free_gib=args.minimum_free_gib)), flush=True)


if __name__ == "__main__":
    main()
