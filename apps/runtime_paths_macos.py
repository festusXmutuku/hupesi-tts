"""macOS-specific paths for packaged builds (.app) - split from runtime_paths.py so
edits here do not affect Windows/Linux.

Why this is needed: on Windows/Linux the "executable folder" is where the user extracted the app,
so writing data there is reasonable. On macOS the executable lives INSIDE
`Hupesi TTS.app/Contents/MacOS/`. Writing there means:
  - updating the app (replacing the .app bundle) would LOSE downloaded models, cloned voices, etc.
  - changing bundle contents after signing may cause macOS to report the app as "damaged".
Therefore all writable data for the .app build lives in ~/Library/Application Support.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "Hupesi TTS"


def is_frozen_app() -> bool:
    return sys.platform == "darwin" and bool(getattr(sys, "frozen", False))


def data_dir() -> Path:
    d = Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def bundled_models_dir() -> Path:
    """Models bundled inside the .app (build_macos.sh --add-data models/hf_cache)."""
    return Path(getattr(sys, "_MEIPASS", "")) / "models" / "hf_cache"


def seed_bundled_models(cache: Path) -> None:
    """First run: copy bundled models from inside the .app into the writable cache.

    Do not point HF_HOME directly into the bundle: huggingface_hub writes lock files and may
    download additional files (fp32, Nano) into the cache, and the .app bundle should not be
    modified. Copy once (~312 MB, a few seconds); on subsequent runs, skip if `hub/` already exists.
    """
    src = bundled_models_dir()
    if not src.is_dir() or (cache / "hub").is_dir():
        return
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".seeding")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=True)
    for entry in tmp.iterdir():
        shutil.move(str(entry), str(cache / entry.name))
    tmp.rmdir()
