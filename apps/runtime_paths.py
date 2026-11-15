"""Shared runtime paths for all entry points (FastAPI, Desktop).

Why this exists: previously each entry point set HF_HOME differently, so running the app in two
different ways would download the same ~300 MB model set into TWO different directories.

Now every entry point calls `setup_hf_home()`, defaulting to `models/hf_cache` INSIDE the
application directory. This allows bundled models to ship with releases, and uninstalling the app
leaves no leftover files in the user's profile.

IMPORTANT: must be called BEFORE `huggingface_hub` is imported for the first time, because that
library reads HF_HOME at import time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """Application root directory.

    PyInstaller `--onedir` packaged build: the directory containing the .exe. Intentionally does NOT
    use `sys._MEIPASS` - that is a temporary extraction directory deleted on exit, so placing the
    cache there would require re-downloading on every launch.
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            # The executable lives inside the .app bundle - data must be stored outside the bundle.
            from apps.runtime_paths_macos import data_dir
            return data_dir()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

# onnxruntime imports these 4 DLLs (verified via the PE import table of
# onnxruntime_pybind11_state.pyd). They are part of Microsoft Visual C++ Redistributable
# 2015-2022 x64 and onnxruntime does NOT bundle them - unlike numpy/sklearn/llvmlite, which
# ship their own msvcp140 copy. On machines missing them, users see an obscure message:
#   "DLL load failed while importing onnxruntime_pybind11_state:
#    A dynamic link library (DLL) initialization routine failed."
# VCRUNTIME140_1.dll only exists in the 2019+ redistributable, so machines with the old 2015
# version may still be missing that specific file.
#
# Packaged .exe builds are not affected: PyInstaller bundles all 4 DLLs.
_VCREDIST_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll")


def missing_vcredist() -> list[str]:
    """VC++ Redistributable DLLs missing on this machine. Empty means all are present."""
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return []
    import ctypes
    missing = []
    for name in _VCREDIST_DLLS:
        try:
            ctypes.WinDLL(name)
        except Exception:
            missing.append(name)
    return missing


def vcredist_error_message(missing: list[str]) -> str:
    return (
        "This machine is missing the Microsoft Visual C++ Redistributable, so the inference "
        "library (onnxruntime) cannot be loaded.\n\n"
        f"Missing: {', '.join(missing)}\n\n"
        f"Download and install the x64 version (free, from Microsoft):\n{VCREDIST_URL}\n\n"
        "Restart the application after installation."
    )


def _is_writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except Exception:
        return False


def setup_hf_home() -> Path:
    """Set the shared HF_HOME and return the cache path.

    Priority order:
      1. HF_HOME explicitly set by the user  -> respect it, do not override.
      2. `<app directory>/models/hf_cache`   -> bundled with releases, clean on uninstall.
      3. `%LOCALAPPDATA%/HupesiTTS/hf_cache` -> when the app directory is read-only
         (installed in Program Files, running from a network drive, write-locked USB, etc.).
    """
    existing = os.environ.get("HF_HOME")
    if existing:
        return Path(existing)

    cache = app_root() / "models" / "hf_cache"
    if not _is_writable(cache):
        base = os.getenv("LOCALAPPDATA") or str(Path.home())
        cache = Path(base) / "HupesiTTS" / "hf_cache"
        cache.mkdir(parents=True, exist_ok=True)

    if sys.platform == "darwin" and getattr(sys, "frozen", False):
        from apps.runtime_paths_macos import seed_bundled_models
        seed_bundled_models(cache)
    elif sys.platform.startswith("linux") and getattr(sys, "frozen", False):
        from apps.runtime_paths_linux import seed_bundled_models as seed_linux
        seed_linux(cache)

    os.environ["HF_HOME"] = str(cache)
    # On Windows without Developer Mode, Hugging Face cannot create symlinks and warns on every
    # download. It still works (just uses more disk space), so there is no need to alarm end users.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache


# Correct MIME types for static files that browsers REQUIRE to be served with the right type.
_WEB_MIME_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


def fix_web_mime_types() -> dict[str, str]:
    r"""Force correct MIME types for static files, regardless of what the Windows registry says.

    Why this is needed: on Windows, `mimetypes.init()` READS THE REGISTRY (HKEY_CLASSES_ROOT\.js
    -> "Content Type") and lets it OVERRIDE Python's defaults. Many programs (old editors, Java/Adobe
    installers, internal tools) overwrite this key with "text/plain". When that happens, Starlette's
    `StaticFiles` serves .js files with `Content-Type: text/plain`, and browsers REFUSE to execute them:

        Failed to load module script: Expected a JavaScript-or-Wasm module script
        but the server responded with a MIME type of "text/plain".

    The result is a completely BLANK window with no visible error - React never runs. This only occurs
    on machines with a polluted registry, so dev machines work fine and the issue is very hard to diagnose.

    `mimetypes.add_type()` runs `init()` first and then overrides, so calling this once at startup is
    sufficient. Returns the extensions that actually had to be fixed, for logging.
    """
    import mimetypes

    fixed: dict[str, str] = {}
    for ext, correct in _WEB_MIME_TYPES.items():
        before = mimetypes.guess_type("x" + ext)[0]
        if before != correct:
            fixed[ext] = before or "(none)"
        mimetypes.add_type(correct, ext)
    return fixed


def https_context():
    """SSL context for urllib downloads: trust the operating system certificate store
    AND certifi's certificate bundle.

    Windows only installs root certificates on demand (schannel fetches them via Windows Update),
    while Python's OpenSSL only reads roots already present in the store. Machines that have never
    encountered the "USERTrust ECC" root (api.github.com's new chain, 2026-10) therefore report
    CERTIFICATE_VERIFY_FAILED, even though curl and browsers can connect.
    """
    import ssl

    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        pass
    return ctx
