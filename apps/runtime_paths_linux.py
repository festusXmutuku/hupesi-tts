"""Linux-specific paths for packaged builds - split from runtime_paths.py so edits here
do not affect Windows/macOS.

A .tar.gz extracted into a user-writable folder can use models in place.
A .deb installed into /opt/hupesi-tts is only writable by root: runtime_paths already
redirects the cache to ~/HupesiTTS/hf_cache, but bundled models in /opt cannot be used
directly (huggingface_hub writes lock files into the cache) - without copying them out, the app
would re-download ~312 MB even though they are already on disk.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def is_frozen_app() -> bool:
    return sys.platform.startswith("linux") and bool(getattr(sys, "frozen", False))


def bundled_models_dir() -> Path:
    """Bundled models: models/hf_cache next to the executable (copied there by build_linux.sh)."""
    return Path(sys.executable).resolve().parent / "models" / "hf_cache"


def seed_bundled_models(cache: Path) -> None:
    """Copy bundled models into the writable cache once. For .tar.gz builds (where the cache is the
    bundled folder itself) or when the cache already has `hub/`, skip."""
    src = bundled_models_dir()
    if not src.is_dir() or src == cache.resolve() or (cache / "hub").is_dir():
        return
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".seeding")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=True)
    for entry in tmp.iterdir():
        shutil.move(str(entry), str(cache / entry.name))
    tmp.rmdir()
