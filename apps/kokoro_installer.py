"""Install the phonemizer for Kokoro into a separate venv, ONLY when the user clicks Download.

Why not bundled: `kokoro-onnx` depends on `phonemizer` (GPLv3+) and
`espeakng-loader` (which ships espeak-ng, GPLv3). The Hupesi TTS installer does not
contain any GPL components; these packages are downloaded from PyPI by the user's machine
when they choose the Kokoro model - similar to how `apps/gpu_installer.py` installs PyTorch.

Two tiers, each with its own marker file in the venv:
  * Base (`install_deps`, when downloading Kokoro): espeak - sufficient for English, Spanish,
    French, Italian, Portuguese, and Hindi. ~22 MB.
  * Language pack (`install_pack`, when the user adds one): espeak reads Japanese and
    Chinese poorly, so `misaki` (Apache, Kokoro's original phonemizer) is used with its own
    dictionaries - fugashi + unidic-lite for Japanese (~250 MB on disk, Kokoro v1.0's
    default `cutlet` branch), jieba + pypinyin + cn2an for Chinese. Do NOT install the
    `misaki[ja]` extra: it pulls in `pyopenjtalk`, which has no Windows wheel (must be built
    with cmake) and is not used by the cutlet branch - see
    `kokoro_engine._ensure_pyopenjtalk_importable`.

Reuses the same mechanism as the GPU installer (system Python extracted to avoid junction
`os error 448`, bundled `uv`, `_BaseStdlibFinder` for missing stdlib modules) but with a
separate venv and WITHOUT numpy/onnxruntime. All packages are installed with `--no-deps
--require-hashes` using locks in `apps/kokoro_locks.py` (exact versions, exact hashes,
no packages outside the list), and UV_*/PIP_* environment variables are stripped so no
other package source can sneak in. The venv's site-packages are appended to the END of
`sys.path` - app modules always win.

Do NOT hold `model_lock` here - this is only called from the background thread in
`apps/model_downloader.py`, before the apply step.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

from apps import gpu_installer
from apps.kokoro_locks import BASE_LOCK, PACK_LOCKS
from apps.runtime_paths import _is_writable, app_root

KOKORO_ONNX_VERSION = "0.6.1"
PHONEMIZER_VERSION = "3.4.0"
ESPEAKNG_LOADER_VERSION = "0.2.4"
# Base set, installed with --no-deps: this is the FULL dependency tree of phonemizer 3.4.0
# (joblib pulls in cloudpickle). Do not let uv resolve dependencies on its own: a newer
# release of a transitive dependency after the app's release date must not be fetched.
# kokoro-onnx also requires onnxruntime + numpy - the app already has them, so they are
# intentionally omitted here. Hashes live in apps/kokoro_locks.py.
BASE_PACKAGES = (
    f"kokoro-onnx=={KOKORO_ONNX_VERSION}", f"phonemizer=={PHONEMIZER_VERSION}",
    f"espeakng-loader=={ESPEAKNG_LOADER_VERSION}", "attrs==26.1.0", "cloudpickle==3.1.2",
    "dlinfo==2.0.0", "joblib==1.6.0", "typing-extensions==4.16.0",
)
# Languages readable with just the base set (espeak). Matches kokoro_engine.LANGS.
BASE_LANGUAGES = frozenset({"en", "es", "fr", "it", "pt", "hi"})
# Language packs, installed with --no-deps: this is the FULL dependency tree needed for
# `misaki.ja` (cutlet branch) and `misaki.zh` (v1.0) - verify when changing versions.
# unidic-lite and jieba are pure-Python sdists: uv builds them, no compiler needed.
PACK_PACKAGES = {
    "ja": ("misaki==0.9.4", "addict==2.4.0", "fugashi==1.5.2", "unidic-lite==1.0.8",
           "jaconv==0.5.0", "mojimoji==0.0.13"),
    "zh": ("misaki==0.9.4", "addict==2.4.0", "ordered-set==4.1.0", "jieba==0.42.1",
           "pypinyin==0.55.0", "cn2an==0.5.24", "proces==0.1.7"),
}
_PACK_LABEL = {"ja": "Japanese", "zh": "Chinese"}
# Test sentences written with \\u escapes so Windows command lines do not depend on code page.
_PACK_PROBE = {
    "ja": ("from misaki import ja", "ja.JAG2P()('\\u3053\\u3093\\u306b\\u3061\\u306f')[0]"),
    "zh": ("from misaki import zh", "zh.ZHG2P()('\\u4f60\\u597d')[0]"),
}
_PYOPENJTALK_STUB = ("import sys, types; "
                     "sys.modules.setdefault('pyopenjtalk', types.ModuleType('pyopenjtalk')); ")
_MARKER_NAME = ".kokoro_deps_ready"
_WHAT = "phonemizer cho Kokoro"
_HINT = "Check your network connection and available disk space, then try again."
_INDEX = ("--index-url", "https://pypi.org/simple")


def _marker_value() -> str:
    return json.dumps({
        "python": gpu_installer._runtime_tag(), "kokoro_onnx": KOKORO_ONNX_VERSION,
        "phonemizer": PHONEMIZER_VERSION, "espeakng_loader": ESPEAKNG_LOADER_VERSION,
        "schema": 1,
    })


def _pack_marker_value(code: str) -> str:
    return json.dumps({"python": gpu_installer._runtime_tag(),
                       "packages": list(PACK_PACKAGES[code]), "schema": 1})


def _pack_marker(code: str) -> Path:
    return venv_dir() / f".kokoro_pack_{code}"


def venv_dir() -> Path:
    """Next to the app if writable, otherwise %LOCALAPPDATA% (installed to a read-only folder)."""
    d = app_root() / "addons" / "kokoro" / gpu_installer._runtime_tag()
    if _is_writable(d.parent):
        return d
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "HupesiTTS" / "addons" / "kokoro" / gpu_installer._runtime_tag()


def _base_python_present(venv: Path) -> bool:
    """The venv borrows the base Python in `gpu_venv/base-python` (see `install_deps`). If the
    user deletes the gpu_venv folder, the venv's python remains but can no longer run."""
    home = gpu_installer._venv_home(venv)
    return home is not None and home.is_dir()


def deps_ready() -> bool:
    """Installed AND the venv is still usable: marker matches, site-packages exist, and the base
    Python exists. If any of these is missing, the UI must prompt for reinstall instead of ready."""
    venv = venv_dir()
    try:
        if (venv / _MARKER_NAME).read_text(encoding="utf-8") != _marker_value():
            return False
    except OSError:
        return False
    return (gpu_installer._venv_python(venv).exists()
            and gpu_installer._venv_site_packages(venv) is not None
            and _base_python_present(venv))


def pack_ready(code: str) -> bool:
    if code not in PACK_PACKAGES or not deps_ready():
        return False
    try:
        return _pack_marker(code).read_text(encoding="utf-8") == _pack_marker_value(code)
    except OSError:
        return False


def ready_languages() -> set[str]:
    """Languages that can be read with the phonemizer currently installed on this machine."""
    if not deps_ready():
        return set()
    return set(BASE_LANGUAGES) | {c for c in PACK_PACKAGES if pack_ready(c)}


def available() -> bool:
    """Kokoro can be used on this machine: already installed, or `uv` is available to install it.

    macOS: not supported yet (the base Python installer and espeak-ng only target Windows/Linux),
    even if `uv` is already present - otherwise the Download button would appear and fail midway."""
    if sys.platform == "darwin":
        return False
    return deps_ready() or gpu_installer.find_uv() is not None


def activate() -> bool:
    """Append the venv's site-packages to the END of sys.path. Safe to call multiple times."""
    if not deps_ready():
        return False
    site = gpu_installer._venv_site_packages(venv_dir())
    if site is None:
        return False
    if str(site) not in sys.path:
        sys.path.append(str(site))
    gpu_installer._append_base_stdlib(venv_dir())
    importlib.invalidate_caches()
    return True


def _verify_in_process() -> None:
    """Actually load in the app process and phonemize a test sentence. The .exe build may be
    missing modules that a standalone Python has, so testing inside the venv is not enough."""
    from kokoro_onnx.tokenizer import Tokenizer

    if not Tokenizer().phonemize("Hello world.", "en-us"):
        raise RuntimeError("English phonemizer returned no result.")


def _verify_pack_in_process(code: str) -> None:
    from apps.kokoro_engine import _ensure_pyopenjtalk_importable

    if code == "ja":
        _ensure_pyopenjtalk_importable()
        from misaki import ja

        ok = ja.JAG2P()("こんにちは")[0]
    else:
        from misaki import zh

        ok = zh.ZHG2P(version=None)("你好")[0]
    if not ok:
        raise RuntimeError(f"{_PACK_LABEL[code]} phonemizer returned no result.")


def run_selftest() -> Path:
    """Diagnostics for the .exe build: load the phonemizer in the app process and write the
    result to `kokoro-selftest.txt` next to the app. Enable with `CIT_KOKORO_SELFTEST=1`.

    A successful install in a standalone Python does not guarantee it loads in the packaged
    build (missing stdlib modules or submodules of bundled packages, e.g. `numpy.typing`)."""
    import traceback

    out = app_root() / "kokoro-selftest.txt"
    packs = [c for c in PACK_PACKAGES if pack_ready(c)]
    lines = [f"deps_ready={deps_ready()} activate={activate()} packs={packs} venv={venv_dir()}"]
    lines.append(f"torch_kokoro_device={os.getenv('CIT_KOKORO_TORCH_DEVICE', 'cuda')}")
    mods = ["phonemizer", "espeakng_loader", "joblib", "numpy.typing",
            "kokoro_onnx", "kokoro_onnx.tokenizer", "apps.kokoro_engine"]
    if "ja" in packs:
        mods += ["fugashi", "unidic_lite"]
    if "zh" in packs:
        mods += ["jieba", "pypinyin", "cn2an", "misaki.zh"]
    # Kokoro GPU: torch + transformers from the GPU venv, model code bundled in the app.
    from apps import gpu_installer as _gpu

    if _gpu.activate_gpu_venv():
        mods += ["torch", "transformers", "apps._kokoro_torch.model"]
    for mod in mods:
        try:
            m = importlib.import_module(mod)
            lines.append(f"OK   {mod} {getattr(m, '__version__', '')}")
        except BaseException:
            lines.append(f"FAIL {mod}\n{traceback.format_exc()}")
    checks = [("Anh", _verify_in_process)] + [
        (c, lambda c=c: _verify_pack_in_process(c)) for c in packs]
    for name, fn in checks:
        try:
            fn()
            lines.append(f"OK   phonemize {name}")
        except BaseException:
            lines.append(f"FAIL phonemize {name}\n{traceback.format_exc()}")
    try:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass
    return out


def _runner(check):
    def run(args, **kw):
        # strip_package_env: the machine's UV_*/PIP_* variables (extra indexes, config files, ...)
        # must not inject any other package source into this installation.
        return gpu_installer._run(args, check, what=_WHAT, hint=_HINT,
                                  strip_package_env=True, **kw)
    return run


def _pip_install_locked(run, uv: str, py: str, lock: str, name: str) -> None:
    """Install exactly the locked packages: --no-deps (pull nothing outside the list) and
    --require-hashes (downloaded files must match hashes recorded at release time)."""
    req = venv_dir() / f".kokoro-{name}.lock.txt"
    req.write_text(lock.strip() + "\n", encoding="utf-8")
    try:
        run([uv, "pip", "install", "--python", py, "--no-deps", "--require-hashes",
             "-r", str(req), *_INDEX])
    finally:
        req.unlink(missing_ok=True)


def install_deps(on_progress: Optional[Callable[[str], None]] = None,
                 check_cancel: Optional[Callable[[], None]] = None) -> None:
    """Create the venv + install the base set, then load it into the app. Does nothing if already installed and loadable."""
    report = on_progress or (lambda _m: None)
    check = check_cancel or (lambda: None)
    check()
    if activate():
        try:
            _verify_in_process()
            return
        except Exception:
            pass  # marker exists but packages are broken/missing: reinstall

    uv = gpu_installer.find_uv()
    if not uv:
        raise RuntimeError(
            "This application build is missing the tool required to install the Kokoro phonemizer. "
            "Please update or reinstall the latest version of Hupesi TTS."
        )
    run = _runner(check)

    venv = venv_dir()
    venv.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(venv.parent).free < 300 * 1024 ** 2:
        raise RuntimeError("At least 300 MB of free space is required to install the Kokoro phonemizer. Free up disk space and try again.")
    (venv / _MARKER_NAME).unlink(missing_ok=True)

    report("Preparing Python for the phonemizer...")
    if venv.exists() and not _base_python_present(venv):
        # Base Python was deleted (user removed gpu_venv): the venv is broken, rebuild from scratch.
        try:
            shutil.rmtree(venv)
        except OSError as exc:
            raise RuntimeError(
                "Could not remove the old Kokoro phonemizer. Restart the application and "
                f"click Try again. Details: {exc}"
            ) from exc
    if not gpu_installer._venv_python(venv).exists():
        # Share the base Python with the GPU installer: if GPU is already installed, no re-download.
        base_python = gpu_installer._install_base_python(uv, check, report, label=_WHAT)
        run([uv, "venv", str(venv), "--python", str(base_python), "--no-python-downloads"])
    py = str(gpu_installer._venv_python(venv))

    report("Installing phonemizer (about 22 MB)...")
    # --no-deps: kokoro-onnx requires onnxruntime + numpy, which the app already has; a second
    # copy in the venv wastes space and risks being loaded by mistake.
    venv.mkdir(parents=True, exist_ok=True)
    _pip_install_locked(run, uv, py, BASE_LOCK, "base")

    report("Verifying phonemizer...")
    run([py, "-c",
         "import espeakng_loader as l; "
         "from phonemizer.backend.espeak.wrapper import EspeakWrapper as W; "
         "W.set_data_path(l.get_data_path()); W.set_library(l.get_library_path()); "
         "import phonemizer; assert phonemizer.phonemize('hello', 'en-us').strip()"],
        timeout=120)
    check()
    (venv / _MARKER_NAME).write_text(_marker_value(), encoding="utf-8")

    report("Loading phonemizer into the application...")
    try:
        if not activate():
            raise RuntimeError("Could not find the newly installed library.")
        _verify_in_process()
    except Exception as exc:
        raise RuntimeError(
            "The Kokoro phonemizer was installed but could not be loaded into the app. "
            f"Restart the application and click Try again. Details: {exc}"
        ) from exc


def install_pack(code: str, on_progress: Optional[Callable[[str], None]] = None,
                 check_cancel: Optional[Callable[[], None]] = None) -> None:
    """Install the phonemizer for a language pack ("ja"/"zh") into the same venv. Requires the
    base set - installs it if missing. Does nothing if already installed and loadable."""
    if code not in PACK_PACKAGES:
        raise ValueError(f"Unknown language pack \"{code}\".")
    report = on_progress or (lambda _m: None)
    check = check_cancel or (lambda: None)
    ten = _PACK_LABEL[code]

    install_deps(on_progress=report, check_cancel=check)
    if pack_ready(code):
        try:
            _verify_pack_in_process(code)
            return
        except Exception:
            pass

    uv = gpu_installer.find_uv()
    if not uv:
        raise RuntimeError(
            f"This application build is missing the tool required to install the {ten} phonemizer. "
            "Please update or reinstall the latest version of Hupesi TTS."
        )
    run = _runner(check)
    venv = venv_dir()
    if shutil.disk_usage(venv.parent).free < 400 * 1024 ** 2:
        raise RuntimeError(f"At least 400 MB of free space is required to install {ten}. Free up disk space and try again.")
    _pack_marker(code).unlink(missing_ok=True)
    py = str(gpu_installer._venv_python(venv))

    report(f"Installing {ten} phonemizer...")
    _pip_install_locked(run, uv, py, PACK_LOCKS[code], code)

    report(f"Verifying {ten} phonemizer...")
    imp, probe = _PACK_PROBE[code]
    run([py, "-c", f"{_PYOPENJTALK_STUB}{imp}; assert {probe}"], timeout=300)
    check()
    _pack_marker(code).write_text(_pack_marker_value(code), encoding="utf-8")

    report(f"Loading {ten} phonemizer into the application...")
    try:
        importlib.invalidate_caches()
        _verify_pack_in_process(code)
    except Exception as exc:
        raise RuntimeError(
            f"The {ten} phonemizer was installed but could not be loaded into the app. "
            f"Restart the application and click Try again. Details: {exc}"
        ) from exc
