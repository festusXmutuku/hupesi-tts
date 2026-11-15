"""Copy into the release package EXACTLY the files for the bundled model (VieNeu-TTS v3 Turbo int8
+ MOSS codec), according to the list in apps/model_catalog.py.

Previously build_desktop_exe.bat used robocopy to copy the entire models/ directory except for a
few known folders, so anything else in the dev machine's cache also ended up in the release: build
1.0.1 almost shipped ~650 MB of Kokoro models from another branch. Copying by list ensures only the
files the app needs are included, and cloned voices (custom_voices_*.json, real-person voices) from
the dev machine are never included.

Usage (called by build_desktop_exe.bat):
    python copy_release_models.py dist/Hupesi-TTS/models

Copies into huggingface_hub's cache layout (refs/main + snapshots/<sha>/...), excluding blobs:
offline reads only need files in the snapshot. shutil.copy2 follows symlinks, so machines with
Developer Mode (where snapshots are symlinks to blobs) still produce real files.
Exits with code 1 if the cache is missing files - run `python fetch_models.py` first.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (script lives in scripts/)
sys.path.insert(0, str(ROOT))

from apps.model_catalog import BUNDLED_MODEL_ID, get  # noqa: E402

SOURCE_HUB = ROOT / "models" / "hf_cache" / "hub"


def repo_dir(hub: Path, repo: str) -> Path:
    return hub / ("models--" + repo.replace("/", "--"))


def pinned_revision(repo_path: Path):
    """SHA of the model version in the cache: `refs/main` if present, otherwise the single
    snapshot (fetch_models.py removes old revisions). Caches downloaded by commit SHA on clean CI
    machines sometimes did not have refs/main (fetch_models.py now writes it)."""
    ref = repo_path / "refs" / "main"
    if ref.is_file():
        return ref.read_text(encoding="utf-8").strip()
    snaps = [d.name for d in (repo_path / "snapshots").glob("*") if d.is_dir()]
    return snaps[0] if len(snaps) == 1 else None


def plan_copy(source_hub: Path, dest_hub: Path):
    """([(source, destination)], {repo: sha}, [missing files])."""
    spec = get(BUNDLED_MODEL_ID)
    pairs, refs, missing = [], {}, []
    for source in spec.sources:
        if source.patterns:
            raise SystemExit(f"{BUNDLED_MODEL_ID}: source {source.repo} uses glob patterns and cannot be copied by list.")
        sha = pinned_revision(repo_dir(source_hub, source.repo))
        if sha is None:
            missing.append(f"{source.repo} (not downloaded, or multiple versions in cache)")
            continue
        refs[source.repo] = sha
        for rel in source.allow_patterns:
            src = repo_dir(source_hub, source.repo) / "snapshots" / sha / rel
            if src.is_file():
                pairs.append((src, repo_dir(dest_hub, source.repo) / "snapshots" / sha / rel))
            else:
                missing.append(f"{source.repo}/{rel}")
    return pairs, refs, missing


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python copy_release_models.py <destination models directory>")
        return 2
    dest = Path(argv[1])
    dest_hub = dest / "hf_cache" / "hub"
    pairs, refs, missing = plan_copy(SOURCE_HUB, dest_hub)
    if missing:
        print("[ERROR] Model cache is missing files:")
        for m in missing:
            print("    -", m)
        print('        Run "python fetch_models.py" and build again.')
        return 1
    if dest.exists():
        shutil.rmtree(dest)
    for repo, sha in refs.items():
        ref = repo_dir(dest_hub, repo) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(sha, encoding="utf-8")
    total = 0
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += dst.stat().st_size
    print(f"    Copied {len(pairs)} files for {BUNDLED_MODEL_ID} ({total / 1024 / 1024:.0f} MB).")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv))
