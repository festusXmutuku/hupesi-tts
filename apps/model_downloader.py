"""Download models in a background thread with progress reporting and cancellation.

Why this is needed: previously selecting FP32 and clicking load triggered a ~453 MB
`hf_hub_download` directly inside the HTTP request handler under `model_lock` - blocking all
other synthesis requests, with the UI showing only a spinner for several minutes.

Most important invariant: `model_lock` is NEVER held during download. Only the final apply step
acquires the lock, and it takes only a few seconds.
"""
from __future__ import annotations

import threading
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from fnmatch import fnmatch

from apps import model_catalog

# Download job state. One job at a time.
_state_lock = threading.Lock()
_cancel = threading.Event()
_thread: Optional[threading.Thread] = None

_STATE_RONG: Dict[str, Any] = {
    "state": "idle",          # idle | downloading | applying | ready | error | cancelled
    "model_id": None,
    "pack": None,             # Kokoro language pack being downloaded ("ja"/"zh"), None = full model
    "percent": 0.0,
    "done_bytes": 0,
    "total_bytes": 0,
    "current_file": "",
    "message": "",
}

_state: Dict[str, Any] = dict(_STATE_RONG)


class Cancelled(Exception):
    """User clicked Cancel mid-download."""


def _reset_state() -> None:
    with _state_lock:
        _state.clear()
        _state.update(_STATE_RONG)


def get_state() -> Dict[str, Any]:
    """Return a copy of the state - callers cannot mutate the real state through this."""
    with _state_lock:
        return dict(_state)


def _set_state(**kw: Any) -> None:
    with _state_lock:
        _state.update(kw)
        total = _state.get("total_bytes") or 0
        done = _state.get("done_bytes") or 0
        _state["percent"] = min(100.0, done * 100.0 / total) if total > 0 else 0.0


# Progress for the source (repo) currently being downloaded. `snapshot_download` only provides a
# FILE COUNT progress bar, so displayed bytes are derived: bytes from completed sources plus the
# proportional share of the current source. One job at a time (see `start()`), so these two globals
# do not need their own lock.
_nguon_base_bytes = 0
_nguon_bytes = 0


def _dat_nguon(base_bytes: int, bytes_nguon: int) -> None:
    global _nguon_base_bytes, _nguon_bytes
    _nguon_base_bytes, _nguon_bytes = int(base_bytes), int(bytes_nguon)


class _ProgressTqdm:
    """tqdm replacement for `snapshot_download` to read progress and support cancellation.

    huggingface_hub runs `snapshot_download` via `tqdm.contrib.concurrent.thread_map`, so this class
    must behave like tqdm in three ways - missing any of them breaks downloads, often silently:

      * `get_lock` / `set_lock` are **classmethods**: `ensure_lock()` calls them before opening the
        executor. Missing them causes the download to fail immediately with
        ``type object '_ProgressTqdm' has no attribute 'get_lock'``.
      * Iterable AND must yield each element: `thread_map` wraps `ex.map(...)` in the progress bar and
        then `list(...)`s it. Returning an empty iterator means the executor is never consumed - no files
        are downloaded and no error is raised.
      * `total` here is the FILE COUNT, not bytes: hf does not pass `tqdm_class` down to individual file
        downloads (see `snapshot_download` docstring), so there is no byte-level progress to read.
        Displayed bytes are derived from `_dat_nguon()`.

    `snapshot_download` does not accept a cancellation flag, so raising from `update()` is the only way
    to stop it - cancellation takes effect at file boundaries.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = kwargs.get("total") or 0
        self.n = 0
        self._iterable = args[0] if args else kwargs.get("iterable")
        desc = kwargs.get("desc")
        if desc:
            _set_state(current_file=str(desc))

    @classmethod
    def get_lock(cls):
        lock = getattr(cls, "_lock", None)
        if lock is None:
            lock = threading.RLock()
            cls._lock = lock
        return lock

    @classmethod
    def set_lock(cls, lock: Any) -> None:
        cls._lock = lock

    def update(self, n: int = 1) -> None:
        if _cancel.is_set():
            raise Cancelled()
        self.n += n
        if self.total > 0:
            xong = min(self.n, self.total) / self.total
            _set_state(done_bytes=_nguon_base_bytes + int(_nguon_bytes * xong))

    def __iter__(self):
        if self._iterable is None:
            return
        for item in self._iterable:
            self.update(1)
            yield item

    # huggingface_hub uses the progress bar as a context manager and calls a few helper methods; ignore them.
    def __enter__(self) -> "_ProgressTqdm":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        pass

    def set_description(self, desc: Optional[str] = None, **_: Any) -> None:
        if desc:
            _set_state(current_file=str(desc))

    def set_postfix(self, *a: Any, **k: Any) -> None:
        pass

    def refresh(self, *a: Any, **k: Any) -> None:
        pass

    def reset(self, total: Optional[int] = None) -> None:
        self.total = total or 0
        self.n = 0


# web_stream points this to `_friendly_load_error` on import. Default is `str` so the module works standalone in tests.
apply_error_formatter = str


def _chuoi_loi_mang(e: BaseException) -> str:
    """Convert technical huggingface_hub errors into actionable user messages."""
    ten = type(e).__name__
    nhan_dang_mang = (
        "ConnectionError", "LocalEntryNotFoundError", "HfHubHTTPError",
        "ReadTimeout", "ConnectTimeout", "Timeout", "SSLError", "ProxyError",
    )
    if ten in nhan_dang_mang or "connection" in str(e).lower():
        return (
            "Could not connect to Hugging Face to download the model. "
            "Check your network or proxy and click Retry. "
            "The currently loaded model continues to work normally."
        )
    return f"Could not download the model: {e}"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Wrap two huggingface_hub functions as module-level names so tests can patch them without touching the library.
def _model_info(repo: str, revision: Optional[str] = None):
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, revision=revision, files_metadata=True)


def _prime_symlink_check(repo: str) -> None:
    """Probe once, in a SINGLE thread, whether the repo cache directory supports symlinks.

    Why this is needed: huggingface_hub's `are_symlinks_supported()` writes `True` to its cache before
    actually testing, and it is not thread-safe. `snapshot_download` runs multiple threads, so a later
    thread may read that temporary `True`, call `os.symlink`, and fail with
    `[WinError 1314] A required privilege is not held by the client` on Windows machines without
    Developer Mode. Probing here first ensures download threads read the correct result and fall back to
    copying files when needed.

    Clear the cached result before probing: a previous failed download may have left a stale `True`,
    causing "Retry" to fail until the app is restarted.
    """
    try:
        from huggingface_hub import constants
        from huggingface_hub import file_download as fd

        folder = Path(constants.HF_HUB_CACHE) / fd.repo_folder_name(repo_id=repo, repo_type="model")
        folder.mkdir(parents=True, exist_ok=True)
        fd._are_symlinks_supported_in_dir.pop(str(folder.expanduser().resolve()), None)
        fd.are_symlinks_supported(folder)
    except Exception as e:
        # If huggingface_hub's internal API changes, ignore - worst case we fall back to the old behavior.
        print(f"[Model download] Could not pre-check symlink support for {repo}: {e}")


def _snapshot_download(set_main_ref: bool = True, **kwargs):
    """`set_main_ref=False` for sources pinned to a fixed revision in the catalog (Supertonic, Kokoro):
    that sha is not the repo's `main`, and `is_downloaded` checks the exact sha."""
    from huggingface_hub import snapshot_download

    _prime_symlink_check(kwargs["repo_id"])
    result = snapshot_download(**kwargs)
    if set_main_ref and kwargs.get("revision"):
        _write_main_ref(kwargs["repo_id"], kwargs["revision"])
    return result


def _write_main_ref(repo: str, sha: str) -> None:
    """Point `refs/main` to the just-downloaded revision.

    When downloading by sha (required, see step 3 in `_chay`), huggingface_hub does not write
    `refs/main`. But all offline lookups - `model_catalog.is_downloaded()` and model loading - query
    `main`, so a model downloaded for the first time via the Download button would be reported as
    "not downloaded" even though all files are present. This sha is `main` at download time (from
    `model_info`).
    """
    from huggingface_hub import constants
    from huggingface_hub import file_download as fd

    ref = Path(constants.HF_HUB_CACHE) / fd.repo_folder_name(repo_id=repo, repo_type="model") / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(sha)


def _hf_cache_dir() -> Path:
    import os

    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _doc_gon(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB" if n < 1024 ** 3 else f"{n / (1024 ** 3):.1f} GB"


def _du_cho_trong_o(total_bytes: int) -> tuple[bool, str]:
    """Check for free space before downloading, instead of failing after downloading hundreds of MB."""
    can = int(total_bytes * 1.2)
    try:
        con = shutil.disk_usage(str(_hf_cache_dir())).free
    except Exception:
        return True, ""  # cannot measure: try anyway, do not block falsely
    if con < can:
        return False, (
            f"Only {_doc_gon(con)} of disk space is available, but about {_doc_gon(can)} is needed to download this model. "
            "Free up space and try again."
        )
    return True, ""


def _khop(ten_file: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch(ten_file, p) for p in patterns)


def _can_tu_cai_gpu_deps(spec) -> bool:
    """Use the same condition as Settings, including when running via the standalone Python backend."""
    return "torch" in spec.requires and model_catalog.frozen_gpu_autoinstall_possible()


def _chay(spec, auto_apply: bool, apply_fn) -> None:
    """Worker body. Runs synchronously - `start()` is what dispatches it to a background thread.

    Does NOT touch `model_lock` at any step except the final `apply_fn`.
    """
    _cancel.clear()
    _set_state(
        state="downloading", model_id=spec.id, pack=None, percent=0.0, done_bytes=0,
        total_bytes=0, current_file="", message="", phase="model",
    )

    try:
        # 1. Pin sha + measure actual total bytes. If the network fails here, the cache is untouched.
        ke_hoach = []
        tong = 0
        for src in spec.sources:
            info = _model_info(src.repo, src.revision)
            byte_repo = sum(
                getattr(s, "size", 0) or 0
                for s in getattr(info, "siblings", [])
                if _khop(getattr(s, "rfilename", ""), src.allow_patterns)
            )
            tong += byte_repo
            ke_hoach.append((src, src.revision or info.sha, byte_repo))

        if tong <= 0:
            tong = spec.approx_bytes  # no size metadata: fall back to the estimated size
            # If per-source sizes are unknown, split evenly so the progress bar still moves.
            deu = tong // max(len(ke_hoach), 1)
            ke_hoach = [(src, sha, deu) for src, sha, _ in ke_hoach]
        _set_state(total_bytes=tong)

        # 2. Enough disk space?
        du, ly_do = _du_cho_trong_o(tong)
        if not du:
            _set_state(state="error", message=ly_do)
            return

        # 3. Download. `revision=sha` is required: without pinning, files can land in different revisions,
        #    `speaker_encoder.onnx` can get stuck on an old sha, and voice cloning breaks offline.
        da_xong = 0
        for src, sha, byte_repo in ke_hoach:
            _dat_nguon(da_xong, byte_repo)
            _snapshot_download(
                repo_id=src.repo,
                repo_type="model",
                revision=sha,
                allow_patterns=list(src.allow_patterns),
                tqdm_class=_ProgressTqdm,
                max_workers=4,
                **({"set_main_ref": False} if src.revision else {}),
            )
            da_xong += byte_repo
            _set_state(done_bytes=da_xong)

        # 3b. The .exe does not bundle PyTorch - models that require torch (v3 Turbo GPU) get a separate
        #     venv created and installed here, BEFORE holding model_lock in the apply step.
        #     The standalone Python backend uses the same flow.
        if _can_tu_cai_gpu_deps(spec):
            _set_state(current_file="", message="Installing GPU support...", phase="gpu_setup")
            from apps import gpu_installer
            def check_cancel():
                if _cancel.is_set():
                    raise Cancelled()
            gpu_installer.install_gpu_deps(
                on_progress=lambda m: _set_state(current_file=m, message=m),
                check_cancel=check_cancel,
            )
            check_cancel()

        # 3c. Kokoro requires a GPL phonemizer - not bundled, installed on the user's machine now
        #     (apps/kokoro_installer.py).
        if "kokoro_deps" in spec.requires:
            _set_state(current_file="", message="Installing the Kokoro phonemizer...", phase="addon_setup")
            from apps import kokoro_installer
            def check_cancel_addon():
                if _cancel.is_set():
                    raise Cancelled()
            kokoro_installer.install_deps(
                on_progress=lambda m: _set_state(current_file=m, message=m),
                check_cancel=check_cancel_addon,
            )
            check_cancel_addon()

        # 4. Apply. This is the ONLY place that holds model_lock (inside apply_fn).
        #    An error here is NOT a download error - the files are already fully in the cache.
        #    So it has its own branch and uses `apply_error_formatter`, which web_stream points to
        #    `_friendly_load_error` (that function knows how to detect missing VC++ Redistributable).
        if auto_apply and apply_fn is not None:
            _set_state(state="applying", phase="model", percent=100.0, current_file="",
                       message="Loading the downloaded model...")
            try:
                apply_fn(spec.id)
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                _set_state(state="error", current_file="",
                           message=apply_error_formatter(e))
                return

        _set_state(state="ready", percent=100.0, current_file="",
                   message=f"Ready: {spec.name}")

    except Cancelled:
        _set_state(state="cancelled", current_file="",
                   message="Download cancelled. Partially downloaded files are kept and can be resumed later.")
    except Exception as e:  # noqa: BLE001 - convert any error into a user-readable message
        _set_state(state="error", current_file="", message=_chuoi_loi_mang(e))
    finally:
        _cancel.clear()


def _chay_pack(spec, pack, auto_apply: bool, apply_fn) -> None:
    """Download a Kokoro language pack: the pack's voices + its phonemizer, then (optionally, via
    `apply_fn`) reload Kokoro so the new voices appear.

    Voices are downloaded at the EXACT sha of the model currently in cache, not the latest `main`:
    a different sha would place voices in a different snapshot, while the engine reads the voice
    directory next to `af_heart.bin` in the current snapshot.
    """
    _cancel.clear()
    _set_state(
        state="downloading", model_id=spec.id, pack=pack.code, percent=0.0, done_bytes=0,
        total_bytes=0, current_file="", message="", phase="model",
    )

    def check_cancel():
        if _cancel.is_set():
            raise Cancelled()

    try:
        sha = model_catalog.KOKORO_REVISION
        if not model_catalog.is_downloaded(spec):
            _set_state(state="error", message="Please download the Kokoro model before adding a language.")
            return

        info = _model_info(model_catalog.KOKORO_REPO, revision=sha)
        tong = sum(
            getattr(s, "size", 0) or 0
            for s in getattr(info, "siblings", [])
            if _khop(getattr(s, "rfilename", ""), pack.patterns)
        )
        _set_state(total_bytes=tong)
        _dat_nguon(0, tong)
        _snapshot_download(
            repo_id=model_catalog.KOKORO_REPO, repo_type="model", revision=sha,
            allow_patterns=list(pack.patterns), tqdm_class=_ProgressTqdm, max_workers=4,
            set_main_ref=False,
        )
        _set_state(done_bytes=tong)
        check_cancel()

        _set_state(current_file="", message=f"Installing {pack.name.lower()} phonemizer...", phase="addon_setup")
        from apps import kokoro_installer

        kokoro_installer.install_pack(
            pack.code,
            on_progress=lambda m: _set_state(current_file=m, message=m),
            check_cancel=check_cancel,
        )
        check_cancel()

        if auto_apply and apply_fn is not None:
            _set_state(state="applying", phase="model", percent=100.0, current_file="",
                       message="Reloading the model...")
            try:
                apply_fn(spec.id)
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001
                # The pack is already installed (marker written) - only the reload failed. Make this clear so
                # the user does not re-download the whole pack, just restart the app.
                ten = pack.name.lower()
                _set_state(state="error", current_file="",
                           message=f"Finished installing {ten}, but the model could not be reloaded. "
                                   f"Restart the application to use the new voices. "
                                   f"Details: {apply_error_formatter(e)}")
                return

        _set_state(state="ready", percent=100.0, current_file="",
                   message=f"Added {pack.name.lower()}")

    except Cancelled:
        _set_state(state="cancelled", current_file="",
                   message="Download cancelled. Partially downloaded files are kept and can be resumed later.")
    except Exception as e:  # noqa: BLE001 - convert any error into a user-readable message
        _set_state(state="error", current_file="", message=_chuoi_loi_mang(e))
    finally:
        _cancel.clear()


def start(model_id: str, auto_apply: bool = True, apply_fn=None,
          pack: Optional[str] = None) -> tuple[bool, str]:
    """Accept a download job. Returns `(True, "")` if dispatched to a background thread.

    `pack` ("ja"/"zh"): download only that Kokoro language pack."""
    global _thread

    spec = model_catalog.get(model_id)
    if spec is None:
        return False, f"No model named \"{model_id}\" was found."
    goi = None
    if pack is not None:
        goi = model_catalog.kokoro_pack(pack)
        if not model_catalog.is_kokoro(spec.id) or goi is None:
            return False, f"Model \"{model_id}\" does not have a language pack named \"{pack}\"."

    hien_tai = get_state()
    if hien_tai["state"] in ("downloading", "applying"):
        return False, (
            f"Currently downloading {hien_tai['model_id']}; cancel it first."
        )

    if goi is not None:
        target, args = _chay_pack, (spec, goi, auto_apply, apply_fn)
    else:
        target, args = _chay, (spec, auto_apply, apply_fn)
    _set_state(pack=pack)
    _thread = threading.Thread(target=target, args=args, name="model-downloader", daemon=True)
    _thread.start()
    return True, ""


def cancel() -> bool:
    """Set the cancellation flag. Returns False if nothing is currently running."""
    if get_state()["state"] not in ("downloading", "applying"):
        return False
    _cancel.set()
    return True
