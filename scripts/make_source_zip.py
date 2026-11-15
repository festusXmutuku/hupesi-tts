"""Package the source distribution so users can download, extract, and run run_desktop_app.bat.

    python make_source_zip.py

Output: dist/Hupesi-TTS-source.zip

Excluded items that MUST NOT be included in a release:
  - `.venv` / `.xpu_venv`: environments created by uv use trampolines that hardcode the
    interpreter path on the build machine. Copying them to another machine causes
    "uv trampoline failed to spawn Python child process". Users must let run_desktop_app.bat
    create a fresh environment.
  - `dist/`, `build/`: build artifacts, large and useless for a source distribution.
  - `node_modules`: reinstall with npm when needed.
  - `.env`: may contain private tokens.
Keep `app-desktop/hupesi-tts-app/dist` (the prebuilt React UI) so run_stream.bat and
desktop_app.py can run immediately without Node.js.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (script lives in scripts/)
TOP = "Hupesi-TTS"

# Directory names excluded at any depth
SKIP_DIR_NAMES = {
    ".git", ".venv", ".xpu_venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".husky",
    "outputs", "merged_models_cache", ".idea", ".vscode",
}
# Paths excluded relative to the repo root (do not exclude app-desktop/hupesi-tts-app/dist)
# CLAUDE.md: internal notes; gpu_venv/: self-installed GPU env (several GB); addons/: self-installed
# Kokoro voice separation tools (includes GPL espeak-ng, must not be redistributed); public/: public repo clone.
SKIP_ROOT_PATHS = {"dist", "build", "wheels", "models", "gpu_venv", "addons", "public"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".spec", ".zip", ".log"}
# chan-doan.txt: diagnostic report from the build machine.
SKIP_NAMES = {".env", "Hupesi-TTS.spec", "CLAUDE.md", "chan-doan.txt"}

HUONG_DAN = """CIT VOICE STUDIO - USER GUIDE
====================================================================

REQUIREMENTS: Windows 10/11 + Internet connection (only needed for the first run).
No need to install Python or Node.js.

HOW TO RUN
--------------------------------------------------------------------
1. Extract this ENTIRE folder to a drive (for example: D:\\Hupesi-TTS).
   DO NOT run it directly from inside the .zip file.

2. Double-click:  run_desktop_app.bat

3. The first run may take a while (5-15 minutes) because the machine needs to download:
   - The uv tool and Python 3.12
   - Required libraries
   - Voice models (several hundred MB)
   Subsequent runs take only a few seconds.

4. The app opens in its own window (or in a browser; see COMMON ISSUES).

STOPPING THE APP
--------------------------------------------------------------------
Double-click:  stop.bat

OTHER LAUNCH FILES
--------------------------------------------------------------------
run_desktop_app.bat Desktop app (dedicated window)
run_stream.bat      Server only, open in a browser - port 8001
run_gpu.bat         GPU version - ONLY for machines with an NVIDIA GPU
stop.bat            Stop all running processes

MACHINES WITH AN NVIDIA GPU
--------------------------------------------------------------------
The default build runs on the CPU and is already about ~2x real time, so
MOST USERS DO NOT NEED THE GPU BUILD.

Only use run_gpu.bat if you work with long scripts or produce podcasts in
many passes: the GPU build processes the whole script in one pass and can
be much faster in that mode.

Before running run_gpu.bat:
  - The first setup must download about 3.5 GB (torch CUDA libraries).
  - At least 12 GB of free disk space is required.
  - A recent NVIDIA driver is required. The script will check and ask before downloading.
  - If the machine does not have an NVIDIA GPU, use run_desktop_app.bat.

COMMON ISSUES
--------------------------------------------------------------------
* Windows blocks .bat files downloaded from the Internet:
  Right-click the .zip file BEFORE extracting it > Properties > check Unblock > OK.

* "uv sync failed":
  Check your network connection, temporarily disable VPN/proxy, and try again.

* "DLL load failed while importing onnxruntime_pybind11_state":
  Your machine is missing Microsoft Visual C++ Redistributable. Download and install
  the x64 version (free, from Microsoft), then run again:
  https://aka.ms/vs/17/release/vc_redist.x64.exe
  The packaged .exe build does not have this issue because it includes those files.

* The app opens in a BROWSER instead of its own window:
  This is not an error. Microsoft Edge WebView2 Runtime is missing, so the app
  automatically falls back to the browser. All features still work normally.
  To get a dedicated window, install WebView2 Runtime (free, from Microsoft):
  https://go.microsoft.com/fwlink/p/?LinkId=2124703

* The window opens but shows a BLANK page:
  Fixed in this release. The cause was a Windows registry entry incorrectly set by
  other software (.js registered as "text/plain"), causing the browser to refuse to
  run the UI code. The app now enforces the correct type at startup.

* "Port is already in use":
  Run stop.bat first, then run run_desktop_app.bat again.

====================================================================
Author: Festus Mutuku (https://github.com/festusxmutukugithub)
Developed based on VieNeu-TTS (Pham Nguyen Ngoc Bao) - Apache 2.0
"""


def bundled_model_files() -> set:
    """Paths (relative to ROOT) of bundled model files - same list as
    copy_release_models.py. Do not include the entire models/hf_cache: the dev machine cache
    may contain other models (Kokoro, fp32, etc.) and duplicate blobs."""
    import copy_release_models as crm

    pairs, refs, missing = crm.plan_copy(crm.SOURCE_HUB, crm.SOURCE_HUB)
    if missing:
        raise SystemExit("ERROR: model cache is missing files: " + ", ".join(missing)
                         + ". Run first: python fetch_models.py")
    files = {src.relative_to(ROOT).as_posix() for src, _ in pairs}
    files |= {(crm.repo_dir(crm.SOURCE_HUB, repo) / "refs" / "main").relative_to(ROOT).as_posix()
              for repo in refs}
    return files


def skipped(rel: Path, with_models: bool, model_files: frozenset = frozenset()) -> bool:
    parts = rel.parts
    if any(p in SKIP_DIR_NAMES for p in parts):
        return True
    if any(p.endswith(".egg-info") for p in parts):
        return True
    posix = rel.as_posix()
    if with_models and posix.startswith("models/hf_cache/"):
        return posix not in model_files   # bundled models only
    if any(posix == s or posix.startswith(s + "/") for s in SKIP_ROOT_PATHS):
        return True
    return rel.name in SKIP_NAMES or rel.suffix.lower() in SKIP_SUFFIXES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-models", action="store_true",
                    help="bundle models/hf_cache (~312 MB) so it runs immediately without downloading")
    args = ap.parse_args()

    if args.with_models and not (ROOT / "models" / "hf_cache").exists():
        print("ERROR: models/hf_cache is missing. Run first: python fetch_models.py")
        return 1

    suffix = "-source-with-models" if args.with_models else "-source"
    out = ROOT / "dist" / f"{TOP}{suffix}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    model_files = frozenset(bundled_model_files()) if args.with_models else frozenset()
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if skipped(rel, args.with_models, model_files):
                continue
            zf.write(path, f"{TOP}/{rel.as_posix()}")
            count += 1
        note = HUONG_DAN
        if args.with_models:
            note = note.replace(
                "3. The first run may take a while (5-15 minutes) because the machine needs to download:\n"
                "   - The uv tool and Python 3.12\n"
                "   - Required libraries\n"
                "   - Voice models (several hundred MB)\n"
                "   Subsequent runs take only a few seconds.",
                "3. This package ALREADY INCLUDES VOICE MODELS, so nothing needs to be downloaded.\n"
                "   The first run only downloads the uv tool, Python 3.12, and required libraries\n"
                "   (a few minutes). Subsequent runs take only a few seconds.")
        zf.writestr(f"{TOP}/HUONG-DAN.txt", note.replace("\n", "\r\n"))
        count += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"OK: {out}")
    print(f"    {count} files, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
