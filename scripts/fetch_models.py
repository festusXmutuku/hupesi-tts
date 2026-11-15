"""Pre-download models into `models/hf_cache` for bundling with a release.

    python fetch_models.py              # CPU/ONNX int8 path (default)
    python fetch_models.py --fp32       # also download fp32 (maximum quality, larger)
    python fetch_models.py --check      # report only, do not download

Why this is needed: on first run the app must download ~286 MB from HuggingFace. For typical
users that means 5-15 minutes of silence, and it is a failure point if HuggingFace is blocked,
overloaded, or the repo is renamed. Pre-downloading and bundling removes that risk entirely.

The file list follows `_GRAPH_FILES` / `_CODEC_FILES` in
`src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py`. If you change it there, update it here too.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The default Windows console is cp1252 — printing accented text can raise
# UnicodeEncodeError and kill the script. Entry points in apps/ handle this the same way.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]  # repo root (script lives in scripts/)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.runtime_paths import setup_hf_home  # noqa: E402  (must run before huggingface_hub)

# File and repo lists come from apps/model_catalog.py — the single source of truth,
# shared with the runtime downloader in apps/model_downloader.py.
from apps.model_catalog import (  # noqa: E402
    V3_REPO, CODEC_REPO, GRAPH_FILES, CODEC_FILES, ROOT_FILES,
)


def human(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true", help="also download the fp32 build (onnx_update)")
    ap.add_argument("--check", action="store_true", help="report only, do not download")
    args = ap.parse_args()

    cache = setup_hf_home()
    print(f"HF_HOME = {cache}")

    if args.check:
        total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
        print(f"Current cache: {human(total)}")
        return 0

    from huggingface_hub import HfApi, hf_hub_download

    # PIN THE REVISION. Without pinning, each run may land on a different revision and the cache ends
    # up with multiple snapshots where NO single snapshot has all files: in an offline package, Voice
    # Cloning breaks because `speaker_encoder.onnx` is in an old revision while `refs/main` points to
    # a newer one. On Windows without Developer Mode, symlinks cannot be created so each snapshot is a
    # full copy — both missing files and doubling disk usage.
    api = HfApi()
    revisions: dict[str, str] = {}
    for repo in (V3_REPO, CODEC_REPO):
        revisions[repo] = api.model_info(repo).sha
        print(f"Pinning {repo} @ {revisions[repo][:12]}")

    jobs: list[tuple[str, str, str | None]] = []
    for fn in ROOT_FILES:
        jobs.append((V3_REPO, fn, None))
    for fn in GRAPH_FILES:
        jobs.append((V3_REPO, fn, "onnx_int8"))
    if args.fp32:
        for fn in GRAPH_FILES:
            jobs.append((V3_REPO, fn, "onnx_update"))
    for fn in CODEC_FILES:
        jobs.append((CODEC_REPO, fn, None))

    failed: list[str] = []
    for i, (repo, fn, sub) in enumerate(jobs, 1):
        label = f"{repo}/{sub + '/' if sub else ''}{fn}"
        try:
            p = hf_hub_download(repo, fn, repo_type="model", subfolder=sub,
                                revision=revisions[repo])
            print(f"[{i}/{len(jobs)}] OK   {label}  ({human(Path(p).stat().st_size)})")
        except Exception as e:
            print(f"[{i}/{len(jobs)}] MISS {label}  -> {e}")
            failed.append(label)

    prune_old_revisions(revisions)
    pin_main_refs(revisions)

    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"\nTotal cache: {human(total)} at {cache}")
    if failed:
        print(f"Failed to download {len(failed)} file(s):")
        for f in failed:
            print(f"  - {f}")
        print("These files will be downloaded at runtime if they are actually needed.")
    return 0


def pin_main_refs(revisions: dict[str, str]) -> None:
    """Write ``refs/main`` = the pinned revision for each repo.

    When downloading by commit SHA (``revision=<sha>``), huggingface_hub does NOT write refs/main,
    but the app reads offline via ``hf_hub_download(..., local_files_only=True)`` with the default
    revision "main" - without refs/main files cannot be found and the app must go online. Dev
    machines may not show the issue because refs/main remains from older downloads; on a clean CI
    build machine, bundled Linux/macOS builds can still require network access on first launch, and
    copy_release_models.py reports "not downloaded" (2026-09-28)."""
    hub = Path(os.environ["HF_HOME"]) / "hub"
    for repo, sha in revisions.items():
        ref = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(sha, encoding="utf-8")


def prune_old_revisions(keep: dict[str, str]) -> None:
    """Delete all old revisions, keeping only the newly pinned revision.

    Necessary because without symlinks each extra revision is a full copy —
    measured in practice: 199 MB of duplication went straight into customer packages.
    """
    from huggingface_hub import scan_cache_dir

    try:
        info = scan_cache_dir()
    except Exception as e:
        print(f"Skipping old-revision cleanup: {e}")
        return

    doomed = [
        rev.commit_hash
        for repo in info.repos
        if repo.repo_id in keep
        for rev in repo.revisions
        if rev.commit_hash != keep[repo.repo_id]
    ]
    if not doomed:
        return

    strategy = info.delete_revisions(*doomed)
    freed = strategy.expected_freed_size_str
    # Delete BEFORE printing: a console-encoding error must not break the cleanup step.
    strategy.execute()
    print(f"Removed {len(doomed)} old revision(s), freed {freed}")

    # Manual cleanup. HF's cache API assumes blobs + symlinks; on Windows without Developer Mode
    # files live directly inside snapshots so its accounting is off and can still miss files
    # (measured in practice: one 50 MB file was left behind).
    import shutil
    stuck: list[Path] = []
    for repo_id, keep_sha in keep.items():
        snap_root = Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        if not snap_root.is_dir():
            continue
        for d in snap_root.iterdir():
            if not d.is_dir() or d.name == keep_sha:
                continue
            # Do NOT use ignore_errors: Windows can lock files mmapped by ONNX Runtime; swallowing
            # errors would make the script report success while tens of MB of junk still ship.
            try:
                shutil.rmtree(d)
                print(f"  removed extra snapshot {repo_id}@{d.name[:12]}")
            except Exception as e:
                stuck.append(d)
                print(f"  COULD NOT delete {repo_id}@{d.name[:12]}: {e}")

    if stuck:
        print("\nJUNK REMAINS IN CACHE — packaging now will waste space.")
        print("This is usually caused by a process holding files. Close all app instances and run again,")
        print("or delete these directories manually:")
        for d in stuck:
            print(f"  {d}")


if __name__ == "__main__":
    sys.exit(main())
