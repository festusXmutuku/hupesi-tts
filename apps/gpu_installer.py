"""Automatically create a separate venv and install PyTorch/CUDA for packaged builds (.exe) when needed.

Why this is needed: the .exe build (PyInstaller --onedir) intentionally does not bundle torch - it goes
against the torch-free commitment and torch+cuda (several GB) does not fit a onedir bundle. When a user
selects a model that requires torch (v3 Turbo GPU) on an NVIDIA machine, we create a separate virtualenv
next to the exe, install the GPU dependency set (equivalent to `gpu-app`) into it, and load that venv's
site-packages into the current process's `sys.path`.

Does NOT hold `model_lock` here - this module is only called from the background thread in
`apps/model_downloader.py`, before the apply step (the only step that holds the lock).
"""
from __future__ import annotations

import importlib
import json
import os
from contextlib import contextmanager
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from apps.runtime_paths import app_root, _is_writable, https_context

_TORCH_VERSION = "2.8.0"
_TORCHAUDIO_VERSION = "2.8.0"
_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"
_MARKER_NAME = ".gpu_deps_ready"
_DLL_HANDLES = {}


def find_uv() -> Optional[str]:
    """Prefer the tool bundled with the app; do not require the user to configure PATH."""
    root = Path(getattr(sys, "_MEIPASS", app_root()))
    bundled = root / "tools" / ("uv.exe" if sys.platform == "win32" else "uv")
    return str(bundled) if bundled.is_file() else shutil.which("uv")


def _runtime_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _marker_value() -> str:
    return json.dumps({"python": _runtime_tag(), "torch": _TORCH_VERSION, "schema": 1})


def _verify_cuda() -> None:
    import torch
    import torchaudio  # noqa: F401
    import transformers  # noqa: F401
    import safetensors  # noqa: F401
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch could not detect an NVIDIA GPU. Please update your NVIDIA driver and try again.")
    value = (torch.ones(1, device="cuda") + 1).item()
    if value != 2:
        raise RuntimeError("A test computation on the GPU failed.")


@contextmanager
def _external_dll_search():
    """Child Python processes must not use DLLs from the app's PyInstaller bundle."""
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        yield
        return
    import ctypes
    kernel = ctypes.windll.kernel32
    old = ctypes.create_unicode_buffer(32768)
    kernel.GetDllDirectoryW(len(old), old)
    kernel.SetDllDirectoryW(None)
    try:
        yield
    finally:
        kernel.SetDllDirectoryW(old.value or None)


def _run(args, check_cancel, *, timeout=3600, what="GPU support",
         hint="Check your network, disk space, and driver, then try again.", strip_package_env=False):
    """Run in the background, support cancellation, and hide the console window on Windows.

    `what`/`hint` only change the error message: the Kokoro installer (apps/kokoro_installer.py) reuses
    this function but is unrelated to GPU or drivers.
    `strip_package_env`: remove all machine UV_*/PIP_* variables (UV_EXTRA_INDEX_URL, UV_FIND_LINKS,
    PIP_INDEX_URL, etc.) so no alternative package sources can interfere - the Kokoro installer enables
    this; the GPU installer keeps the previous behavior."""
    import time
    check_cancel()
    env = os.environ.copy()
    if strip_package_env:
        for key in [k for k in env if k.upper().startswith(("UV_", "PIP_"))]:
            del env[key]
    # Do not let the child Python inherit the packaged app's Python environment.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env["UV_NO_PROGRESS"] = "1"
    env["UV_NO_CONFIG"] = "1"
    storage = gpu_venv_dir().parent
    env["UV_CACHE_DIR"] = str(storage / "cache")
    env["UV_PYTHON_INSTALL_DIR"] = str(storage / "python")
    if sys.platform != "win32":
        from apps.gpu_installer_linux import clean_child_env
        clean_child_env(env)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with _external_dll_search():
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                env=env, creationflags=flags)
    deadline = time.monotonic() + timeout
    try:
        while True:
            check_cancel()
            if time.monotonic() > deadline:
                raise RuntimeError(f"Installing {what} timed out. Check your network and try again.")
            try:
                out, err = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=flags)
        else:
            proc.terminate()
        proc.communicate()
        raise
    if proc.returncode:
        raise RuntimeError(f"Could not install/verify {what}. {hint}\n{(err or out).strip()[-1500:]}")
    return out


def gpu_venv_dir() -> Path:
    """Separate venv directory for GPU dependencies - next to the exe, or %LOCALAPPDATA% if read-only."""
    d = app_root() / "gpu_venv" / _runtime_tag()
    if _is_writable(d.parent):
        return d
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "HupesiTTS" / "gpu_venv" / _runtime_tag()


def _python_install_dir() -> Path:
    """Where uv installs its own Python (matches `UV_PYTHON_INSTALL_DIR` in `_run`).

    New builds no longer install anything here; this is only used to clean up junctions left by older builds."""
    return gpu_venv_dir().parent / "python"


def _base_python_root() -> Path:
    """Self-downloaded and extracted base Python, NOT managed by uv (see `_install_base_python`)."""
    return gpu_venv_dir().parent / "base-python"


def _is_link_dir(p: Path) -> bool:
    try:
        return p.is_symlink() or p.is_junction()
    except OSError:
        return False


def _remove_minor_links(install_dir: Path) -> None:
    """Remove `cpython-3.12-...` junctions left by older builds (which used `uv python install`).

    If junctions remain, uv scanning the Python install directory fails with `os error 448` (see
    `_install_base_python`). Removing a junction only removes the link, not the target directory.
    """
    if not install_dir.is_dir():
        return
    for entry in install_dir.iterdir():
        if entry.name.startswith("cpython-") and _is_link_dir(entry):
            if entry.is_junction():
                os.rmdir(entry)
            else:
                entry.unlink()


def _venv_home(venv_dir: Path) -> Optional[Path]:
    try:
        for line in (venv_dir / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home":
                return Path(value.strip())
    except OSError:
        pass
    return None


def _venv_built_from_base_python(venv_dir: Path) -> bool:
    """Venvs from older builds pointed `home` to uv-managed Python (via junction) - permanently broken
    on machines that block junctions, and must be rebuilt from the self-extracted base Python."""
    home = _venv_home(venv_dir)
    return home is not None and _base_python_root() in home.parents


def _pick_download(entries: list, version: str) -> Optional[dict]:
    """Select a regular CPython build (not freethreaded/debug) for this machine."""
    import platform

    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(
        platform.machine().lower(), platform.machine().lower()
    )
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    ok = [
        e for e in entries
        if e.get("implementation") == "cpython" and e.get("variant") == "default"
        and e.get("os") == os_name and e.get("arch") == arch and e.get("url")
        and str(e.get("version", "")).startswith(version)
    ]
    if not ok:
        return None
    return max(ok, key=lambda e: tuple(int(x) for x in str(e["version"]).split(".") if x.isdigit()))


_ASTRAL_MIRROR = "https://releases.astral.sh/github/"
_GITHUB = "https://github.com/astral-sh/"


def _download(url: str, dest: Path, check_cancel, report, label: str = "GPU") -> None:
    """Download with a User-Agent (releases.astral.sh returns 403 for urllib's default UA);
    if Astral's mirror fails, fall back to the same file on GitHub."""
    urls = [url]
    if url.startswith(_ASTRAL_MIRROR):
        urls.append(_GITHUB + url[len(_ASTRAL_MIRROR):])
    for i, u in enumerate(urls):
        try:
            _download_one(u, dest, check_cancel, report, label)
            return
        except OSError:  # URLError/HTTPError are both OSError
            if i == len(urls) - 1:
                raise


def _download_one(url: str, dest: Path, check_cancel, report, label: str = "GPU") -> None:
    import urllib.request

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "hupesi-tts (python-urllib)"})
    with urllib.request.urlopen(req, timeout=60, context=https_context()) as resp, \
            open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            check_cancel()
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                report(f"Downloading Python for {label}... {done >> 20}/{total >> 20} MB")
    tmp.replace(dest)


def _install_base_python(uv: str, check_cancel, report=lambda _m: None, label: str = "GPU") -> Path:
    """Download the standalone Python build matching the app's version, extract it into a regular directory,
    and return the path to `python.exe`.

    Why not use `uv python install`: uv 0.11 always creates an additional `cpython-3.12-...` junction
    pointing to the patch build and then reads back through it. On Windows 11 with Redirection Guard enabled,
    traversing a junction created by a standard user account is blocked -> "Failed to create Python minor
    version link directory ... untrusted mount point (os error 448)", and uv has no option to disable
    junctions. So we only ask uv for the download URL (`uv python list --only-downloads`, which writes
    nothing to disk), and handle download + extraction ourselves.

    Requests the app's exact patch version; if unavailable, falls back to the latest build of the same minor
    version - same ABI, so native extensions still load correctly.
    """
    import json
    import tarfile

    root = _base_python_root()
    exe_rel = "python/python.exe" if sys.platform == "win32" else "python/bin/python3"
    for existing in sorted(root.glob("*/" + exe_rel)) if root.is_dir() else []:
        return existing

    patch = f"{_runtime_tag()}.{sys.version_info.micro}"
    chosen = None
    for request in (patch, _runtime_tag()):
        out = _run([uv, "python", "list", request, "--only-downloads", "--output-format", "json"],
                   check_cancel, timeout=120)
        try:
            chosen = _pick_download(json.loads(out or "[]"), request)
        except ValueError:
            chosen = None
        if chosen:
            break
    if not chosen:
        raise RuntimeError(f"No suitable Python build was found for {label}.")

    target = root / str(chosen["version"])
    archive = root / "python.tar.gz"
    staging = root / (target.name + ".extract")
    root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    report(f"Downloading Python for {label}...")
    _download(chosen["url"], archive, check_cancel, report, label)
    report(f"Extracting Python for {label}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(staging, filter="data")
    archive.unlink(missing_ok=True)
    if target.exists():
        shutil.rmtree(target)
    _replace_retrying(staging, target)
    exe = target / exe_rel
    if not exe.exists():
        raise RuntimeError("The downloaded Python package has an unexpected structure.")
    return exe


def _replace_retrying(src: Path, dst: Path, attempts: int = 30, delay: float = 0.5) -> None:
    """Rename the just-extracted directory. On Windows, Defender may scan newly created files and hold the
    directory locked for a few seconds, causing `replace` to raise "Access is denied" (WinError 5)
    immediately after extraction; retry for up to ~15 seconds before reporting a real error."""
    import time

    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _venv_site_packages(venv_dir: Path) -> Optional[Path]:
    if sys.platform == "win32":
        lib = venv_dir / "Lib" / "site-packages"
        return lib if lib.exists() else None
    for p in (venv_dir / "lib").glob("python*/site-packages"):
        return p
    return None


def gpu_deps_ready() -> bool:
    """Marker indicating whether the GPU venv was successfully installed in a previous run."""
    try:
        return (gpu_venv_dir() / _MARKER_NAME).read_text(encoding="utf-8") == _marker_value()
    except OSError:
        return False


class _BaseStdlibFinder:
    """Find missing standard library modules in the base Python, using the FULL dotted name.

    Simply adding `Lib` to `sys.path` is not enough: if `unittest` was loaded from the .exe bundle,
    `unittest.mock` is only searched in `unittest.__path__` (inside `_internal`), never in `sys.path`.
    This finder resolves `Lib/unittest/` and then looks for `mock`. Placed at the END of `sys.meta_path`
    so it is only consulted when both FrozenImporter and PathFinder decline - bundled app modules always win.
    """

    def __init__(self, roots: list[Path]):
        self.roots = roots

    def find_spec(self, fullname, path=None, target=None):
        from importlib.machinery import PathFinder

        parts = fullname.split(".")
        dirs = [str(r.joinpath(*parts[:-1])) for r in self.roots]
        return PathFinder.find_spec(fullname, [d for d in dirs if os.path.isdir(d)])


def _append_base_stdlib(venv_dir: Path) -> None:
    """Packaged .exe builds only include the standard library modules the APP imports. torch and
    transformers need additional ones (e.g. `unittest.mock` -> "No module named 'unittest.mock'"
    even after GPU deps are installed). Missing modules are loaded from the base Python's standard
    library - same version as the app, see `_install_base_python`.

    When running from source, the interpreter already has the full standard library.
    """
    if not getattr(sys, "frozen", False):
        return
    home = _venv_home(venv_dir)
    if home is None:
        return
    if sys.platform == "win32":
        roots = [home / s for s in ("Lib", "DLLs") if (home / s).is_dir()]
    else:
        from apps.gpu_installer_linux import base_stdlib_roots
        roots = base_stdlib_roots(home)
    if roots and not any(isinstance(f, _BaseStdlibFinder) for f in sys.meta_path):
        sys.meta_path.append(_BaseStdlibFinder(roots))


def activate_gpu_venv() -> bool:
    """Load the GPU venv's site-packages into the current process's sys.path.

    Must be called BEFORE the first `import torch` in the process (at app startup, and again after
    installation completes). Safe to call multiple times - no-op if already loaded or no venv exists.
    """
    if not gpu_deps_ready():
        return False
    site = _venv_site_packages(gpu_venv_dir())
    if site is None:
        return False
    site_str = str(site)
    if site_str not in sys.path:
        sys.path.insert(0, site_str)
    _append_base_stdlib(gpu_venv_dir())
    # torch bundles native DLLs in torch/lib - on Windows, PATH alone is not enough from Python 3.8+,
    # so directories must be registered explicitly with os.add_dll_directory.
    torch_lib = site / "torch" / "lib"
    if sys.platform == "win32" and torch_lib.exists():
        try:
            if str(torch_lib) not in _DLL_HANDLES:
                _DLL_HANDLES[str(torch_lib)] = os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
    return True


def run_gpu_selftest() -> Path:
    """Diagnostic: import the GPU stack IN THE APP PROCESS and write results to `gpu-selftest.txt`
    next to the app. Enable with `CIT_GPU_SELFTEST=1`.

    Verification in the standalone Python process (`install_gpu_deps`) can pass while loading in the
    .exe still fails - e.g. missing `unittest.mock` - so a test at the actual load site is needed.
    """
    import traceback

    out = app_root() / "gpu-selftest.txt"
    lines = [f"gpu_deps_ready={gpu_deps_ready()} venv={gpu_venv_dir()}"]
    for mod in ("torch", "torchaudio", "safetensors", "transformers",
                "transformers.models.auto", "unittest.mock",
                "transformers.configuration_utils",
                "vieneu._v3_turbo_engine.configuration_v3_turbo",
                "vieneu._v3_turbo_engine.inference_v3_turbo", "vieneu.v3_turbo_serve.engine"):
        try:
            m = importlib.import_module(mod)
            lines.append(f"OK   {mod} {getattr(m, '__version__', '')}")
        except BaseException:
            lines.append(f"FAIL {mod}\n{traceback.format_exc()}")
    # transformers lazy attributes: `import transformers` may succeed but accessing `PretrainedConfig`
    # actually imports configuration_utils (the real error users encounter).
    try:
        from transformers import PretrainedConfig  # noqa: F401
        lines.append("OK   transformers.PretrainedConfig")
    except BaseException:
        lines.append(f"FAIL transformers.PretrainedConfig\n{traceback.format_exc()}")
    try:
        import torch
        lines.append(f"cuda={torch.cuda.is_available()}")
    except BaseException:
        pass
    try:
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass
    return out


def install_gpu_deps(on_progress: Optional[Callable[[str], None]] = None,
                     check_cancel: Optional[Callable[[], None]] = None) -> None:
    """Download GPU libraries only when the user selects them; runs in the model download worker."""
    report = on_progress or (lambda message: None)
    check = check_cancel or (lambda: None)
    check()
    activate_gpu_venv()
    try:
        _verify_cuda()
        return
    except Exception:
        pass

    uv = find_uv()
    if not uv:
        raise RuntimeError("This app build is missing the GPU installer tool. Please update or reinstall Hupesi TTS.")

    venv = gpu_venv_dir()
    venv.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(venv.parent).free < 12 * 1024 ** 3:
        raise RuntimeError("At least 12 GB of free space is required to download and install GPU support. Free up disk space and try again.")
    marker = venv / _MARKER_NAME
    marker.unlink(missing_ok=True)
    report("Preparing Python for GPU support...")
    _remove_minor_links(_python_install_dir())
    if venv.exists() and not _venv_built_from_base_python(venv):
        shutil.rmtree(venv)
    if not _venv_python(venv).exists():
        base_python = _install_base_python(uv, check, report)
        _run([uv, "venv", str(venv), "--python", str(base_python), "--no-python-downloads"], check)
    py = str(_venv_python(venv))
    _run([py, "-c", f"import sys; assert sys.version_info[:2] == {sys.version_info[:2]!r}, 'GPU Python version does not match the app'"], check, timeout=60)
    report("Downloading and installing PyTorch/CUDA (several GB, may take a while)...")
    _run([uv, "pip", "install", "--python", py,
          f"torch=={_TORCH_VERSION}", f"torchaudio=={_TORCHAUDIO_VERSION}",
          "--index-url", _PYTORCH_INDEX], check)
    report("Downloading speech support libraries...")
    _run([uv, "pip", "install", "--python", py,
          "transformers==4.57.6", "safetensors", "--index-url", "https://pypi.org/simple"], check)
    report("Verifying GPU computation on NVIDIA CUDA...")
    _run([py, "-c", "import torch, torchaudio, transformers, safetensors; "
          "assert torch.cuda.is_available(), 'GPU is not ready. Please update your NVIDIA driver'; "
          "assert (torch.ones(1, device='cuda') + 1).item() == 2"], check, timeout=120)
    check()
    marker.write_text(_marker_value(), encoding="utf-8")
    try:
        report("Loading GPU support into the app...")
        if not activate_gpu_venv():
            raise RuntimeError("The newly installed GPU libraries were not found.")
        importlib.invalidate_caches()
        _verify_cuda()
    except Exception as exc:
        # The standalone Python process verified successfully. Keep the marker so the next app launch
        # loads the venv before any imports (avoiding stale modules already in memory).
        raise RuntimeError(f"GPU support was downloaded but could not be loaded into the app. Restart the application and click Retry. Details: {exc}") from exc
