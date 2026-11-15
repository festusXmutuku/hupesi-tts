"""Linux (and macOS) specific parts of the automatic GPU installer - split from gpu_installer.py
so edits here do not affect the Windows code path.

gpu_installer.py only calls into this module on `sys.platform != "win32"` branches.
"""
from __future__ import annotations

import sys
from pathlib import Path


def base_stdlib_roots(venv_home: Path) -> list[Path]:
    """Standard library directories of the base Python, computed from `home` in pyvenv.cfg.

    Unlike Windows: on Linux/macOS `home` is the base Python's `bin/` directory
    (e.g. `.../base-python/3.12.3/python/bin`), while the standard library lives in
    `../lib/python3.12`. Joining directly to `home` as on Windows points to a non-existent
    directory, and the packaged build cannot find missing modules - a real-world error seen as
    "Could not import module 'PretrainedConfig'", rooted in `No module named 'filecmp'`.
    """
    tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    prefix = venv_home.parent if venv_home.name == "bin" else venv_home
    lib = prefix / "lib" / tag
    # C extension modules of the standard library (e.g. _json.so) live in lib-dynload.
    return [p for p in (lib, lib / "lib-dynload") if p.is_dir()]


def clean_child_env(env: dict) -> None:
    """Restore `LD_LIBRARY_PATH` to the state before the PyInstaller bootloader modified it.

    On Linux the bootloader points `LD_LIBRARY_PATH` into `_internal/` and stores the original in
    `LD_LIBRARY_PATH_ORIG`. If not restored, the base Python and CUDA probes running in child
    processes may load bundled .so files (libstdc++, libssl...) instead of system/torch libraries -
    a class of errors that only appears on other machines.
    """
    orig = env.pop("LD_LIBRARY_PATH_ORIG", None)
    if orig is not None:
        env["LD_LIBRARY_PATH"] = orig
    elif getattr(sys, "frozen", False):
        env.pop("LD_LIBRARY_PATH", None)
