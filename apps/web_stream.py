"""
Hupesi TTS - Comprehensive Backend API Server (FastAPI + VieNeu-TTS)
==========================================================================
Provides the full set of API features equivalent to the Web Studio (Gradio UI):
- Real-time TTS streaming (WAV 48kHz)
- High-quality batch/standard TTS generation (WAV / MP3)
- Use saved cloned voices from microphone or audio files with a usage-rights commitment in the app
- Multi-speaker conversation & podcast
- Read and extract text from PDF / TXT documents
- Convert WAV audio to MP3 format
- Manage and switch models (v3 Turbo int8/fp32/GPU, v3 Nano)
"""
import base64
import io
import json
import os
import re
import sys
import time
import wave
import shutil
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

if sys.platform == "win32":
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if sys.stderr is None:
        sys.stderr = io.StringIO()
    else:
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


import numpy as np
import soundfile as sf
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add the src path to sys.path (supports PyInstaller frozen builds)
if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys._MEIPASS)
else:
    ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# Shared Hugging Face cache. Must run BEFORE `from vieneu import Vieneu` because
# huggingface_hub reads HF_HOME at import time.
from apps.runtime_paths import setup_hf_home, fix_web_mime_types
setup_hf_home()

# Must run BEFORE StaticFiles serves the first file. See the detailed explanation in
# runtime_paths.fix_web_mime_types: a dirty Windows registry can cause .js files to be returned as
# "text/plain", causing the browser to refuse to run modules and showing a blank UI with no error.
_mime_fixed = fix_web_mime_types()

# If PyTorch/CUDA was automatically installed into a separate venv on a previous run (packaged .exe, see
# apps/gpu_installer.py), add it to sys.path NOW - before the first `import torch`
# in the process, anywhere (including inside `vieneu`).
from apps.gpu_installer import activate_gpu_venv, run_gpu_selftest
activate_gpu_venv()
if os.getenv("CIT_GPU_SELFTEST") == "1":
    run_gpu_selftest()
if os.getenv("CIT_KOKORO_SELFTEST") == "1":
    from apps.kokoro_installer import run_selftest as run_kokoro_selftest
    run_kokoro_selftest()

from vieneu import Vieneu
from apps import ai_label, model_catalog, model_downloader
from apps.app_version import APP_VERSION

SAMPLE_RATE = 48_000

DOCS_DESCRIPTION = """
### 🤖 Integration Guide for AI Agents & Developers

Hupesi TTS generates speech on the user's machine. The public API includes:
- **Generate WAV/MP3 audio**, stream WAV PCM16, and generate audio with SRT subtitles.
- **Available models**: VieNeu v3 Turbo (48 kHz), v3 Nano (24 kHz), Supertonic 3 (44.1 kHz, 31 languages), or Kokoro (24 kHz, 8 languages, no Vietnamese). Check `/health` and get voices from `/voices`; do not hard-code voices or sample rates.
- **Use saved cloned voices** via `voice_id`. Create voices in the **Voice Cloning** tab using a microphone or an audio file with a usage-rights commitment; voices and commitments are stored only on this machine. There is no public file-based cloning API (`/api/tts/clone` has been removed). Supertonic 3 and Kokoro only provide built-in voices.
- **Emotions**: VieNeu v3 Turbo uses `[cuoi]`, `[hang giong]`, `[tho dai]` (Vietnamese tags); Supertonic uses `<laugh>`, `<breath>`, `<sigh>`; Kokoro has no emotion tags. The pause tag `[nghi 2s]` (e.g. `[nghi 2s]` for a 2-second pause) works with every model.
- **Conversation & Podcast**: Automatically stitches turns between multiple speakers.

Only send synthesis requests when `/health` returns `status: "ok"`. `/stream` returns data progressively; latency depends on the model and machine, and Supertonic and Kokoro return audio in segments. When Kokoro is running, Vietnamese text is rejected (400). Model management, voice creation/deletion, and history APIs are internal and not provided for external software.

🔑 **Calling from other machines** (when enabled in the app's Settings): include the header `X-API-Key: <key>`. Calls from this machine do not require a key.

---
📥 **[Download hupesi-tts.md](/hupesi-tts.md?download=1)** *(Download the file to load into Cursor, Claude, ChatGPT, AutoGen, etc.)*  
📄 **[View the Markdown documentation directly (/hupesi-tts.md)](/hupesi-tts.md)** *(Open, read, and copy directly in the browser)*  
📋 **[Standard OpenAPI JSON specification (/openapi.json)](/openapi.json)**
"""

app = FastAPI(
    title="Hupesi TTS TTS API",
    version=APP_VERSION,
    description=DOCS_DESCRIPTION,
)
from apps.server_settings import router as server_settings_router, setup_port
# API noi bo (chi giao dien cua app dung) an khoi /docs: tai lieu cho phan mem
# khac chi gom cac API tao giong. Van chay binh thuong, chi khong liet ke.
app.include_router(server_settings_router, include_in_schema=False)
from apps import pronunciation_settings
app.include_router(pronunciation_settings.router, include_in_schema=False)
from vieneu_utils.pause_tags import split_pauses
from vieneu_utils.subtitles import build_units, display_text, to_srt, unit_cues
from vieneu_utils import dubbing
from vieneu_utils.srt import SrtError, parse_srt
from vieneu_utils.time_stretch import time_stretch
from vieneu_utils.core_utils import V3_GAP_SILENCE, pause_pad_samples, split_into_sentences
from vieneu_utils.subtitles import CLOSING_MARKS, _NON_LATIN_STOP
from starlette.background import BackgroundTask
from apps import history_store
app.include_router(history_store.router, include_in_schema=False)
from apps import gpu_memory
app.include_router(gpu_memory.router, include_in_schema=False)
gpu_memory.engine_getter = lambda: vieneu
from apps import remote_access
app.include_router(remote_access.router, include_in_schema=False)
from apps import app_update
app.include_router(app_update.router, include_in_schema=False)
from apps import terms
app.include_router(terms.router, include_in_schema=False)
from apps import ui_prefs
app.include_router(ui_prefs.router, include_in_schema=False)

@app.get("/hupesi-tts.md", response_class=FileResponse, tags=["Documentation"])
async def get_hupesi_tts_doc(download: bool = False):
    """Download or view the hupesi-tts.md Markdown file with API integration instructions for AI agents."""
    from apps.runtime_paths import app_root
    # Packaged build: next to the executable, then inside the bundle (--add-data). Source: docs/.
    candidates = (app_root() / "hupesi-tts.md", ROOT_DIR / "hupesi-tts.md",
                  ROOT_DIR / "docs" / "hupesi-tts.md")
    doc_path = next((p for p in candidates if p.exists()), candidates[-1])
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail="File hupesi-tts.md does not exist.")
    disposition = "attachment" if download else "inline"
    return FileResponse(
        str(doc_path),
        media_type="text/markdown; charset=utf-8",
        filename="hupesi-tts.md",
        headers={
            "Content-Disposition": f'{disposition}; filename="hupesi-tts.md"',
            "Access-Control-Allow-Origin": "*",
        },
    )

# Enable CORS for desktop app & local web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Browsers on other origins can only read headers listed here.
    expose_headers=[ai_label.HEADER],
)
# Add BEFORE remote access => runs inside it, only attaching the header to actual audio responses.
app.add_middleware(ai_label.AiHeaderMiddleware)
# Add AFTER CORS => runs OUTERMOST: blocks other machines before reaching any route.
app.add_middleware(remote_access.RemoteAccessMiddleware)

# Global State
vieneu = None
current_model_info = {
    "name": "VieNeu-TTS-v3-Turbo (int8)",
    "backend": "onnx",
    "precision": "int8",
    "device": "cpu",
    "sample_rate": SAMPLE_RATE,
}
model_lock = threading.Lock()

DIST_DIR = ROOT_DIR / "app-desktop" / "hupesi-tts-app" / "dist"

# Mount static assets if modern frontend build exists
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

if _mime_fixed:
    print("[MIME] Registry Windows khai bao sai kieu file tinh, da sua: "
          + ", ".join(f"{e}={v}" for e, v in _mime_fixed.items()))


def cuda_is_available() -> bool:
    """Whether this machine has an NVIDIA GPU usable through torch.

    The default installation is torch-free, so a failed `import torch` is normal -
    in that case, treat it as no GPU. Used for runtime decisions (model selection at
    startup, `gpu.available` display) - NOT for showing the GPU model download button,
    because torch may not be installed yet (see `_effective_has_cuda`).
    """
    global _torch_import_logged
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception as e:
        # Having no torch is normal; but if the GPU package is installed and still broken,
        # leave a trace in the log, otherwise it will just look like "no GPU".
        if not _torch_import_logged and activate_gpu_venv():
            _torch_import_logged = True
            print(f"[GPU] GPU support package is installed but torch could not be loaded: {e!r}")
        return False


_torch_import_logged = False


def _effective_has_cuda() -> bool:
    """For `/api/models` display: treat as having a GPU if torch reports one, OR if the machine has a
    physical NVIDIA GPU even though torch is not installed yet (the .exe installs it on demand, see gpu_installer.py)."""
    if cuda_is_available():
        return True
    from apps.model_catalog import check_has_physical_nvidia_gpu
    return check_has_physical_nvidia_gpu()


def gpu_name() -> Optional[str]:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def _custom_voices_path() -> Path:
    """Separate JSON file for voices saved by the user (different from built-in voices in the
    installation package, which may be read-only) - uses the same directory selection as
    `setup_hf_home()`/`gpu_installer.gpu_venv_dir()`."""
    from apps.runtime_paths import app_root, _is_writable

    filename = "custom_voices_v3_nano.json" if "nano" in current_model_info.get("name", "").lower() else "custom_voices_v3_turbo.json"
    d = app_root() / "models"
    if _is_writable(d):
        return d / filename
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    p = Path(base) / "HupesiTTS" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _supertonic_dir() -> Path:
    """Supertonic 3 snapshot folder in the HF cache. Read-only cache access: if not downloaded,
    tell the user to click Download instead of pulling ~400 MB while switching models."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(
            model_catalog.SUPERTONIC_REPO, revision=model_catalog.SUPERTONIC_REVISION,
            allow_patterns=list(model_catalog.SUPERTONIC_FILES), local_files_only=True))
    except Exception as e:
        raise RuntimeError(
            "Supertonic 3 model is not downloaded. Open the Models section and click Download (~400 MB).") from e


def _load_kokoro(gpu: bool = False):
    """Kokoro multilingual models. The phonemizer (which includes GPL components) is not bundled in the
    installer - it must be installed on the user's machine beforehand by `kokoro_installer` (when clicking Download).

    `gpu`: hexgrad PyTorch weights on CUDA (apps/_kokoro_torch). The bundled onnxruntime only has CPU and
    the GPU build shares the same module name, so GPU must go through the torch path."""
    from apps import kokoro_installer
    from apps.kokoro_engine import KokoroTTS
    from huggingface_hub import hf_hub_download

    if not kokoro_installer.activate():
        raise RuntimeError(
            "The Kokoro phonemizer is not installed. Open Settings and click Download on the Kokoro model "
            "so the app can install it automatically."
        )
    # Dev machines without a GPU: CIT_KOKORO_TORCH_DEVICE=cpu runs the PyTorch path on CPU
    # to test the full flow (download, load, synthesize) - slow, for testing only.
    device = os.getenv("CIT_KOKORO_TORCH_DEVICE", "cuda")
    if gpu and device == "cuda" and not cuda_is_available():
        raise RuntimeError(
            "No usable NVIDIA GPU was found, or PyTorch/CUDA has not finished loading. "
            "Click Download again on the Kokoro GPU model so the app can install it automatically."
        )
    try:
        if gpu:
            model_path = hf_hub_download(model_catalog.KOKORO_TORCH_REPO, "kokoro-v1_0.pth", revision=model_catalog.KOKORO_TORCH_REVISION, local_files_only=True)
            config = hf_hub_download(model_catalog.KOKORO_TORCH_REPO, "config.json", revision=model_catalog.KOKORO_TORCH_REVISION, local_files_only=True)
        else:
            model_path = hf_hub_download(model_catalog.KOKORO_REPO, "onnx/model.onnx", revision=model_catalog.KOKORO_REVISION, local_files_only=True)
            config = None
        heart = hf_hub_download(model_catalog.KOKORO_REPO, "voices/af_heart.bin", revision=model_catalog.KOKORO_REVISION, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            "The Kokoro model is not downloaded. Open Settings and click Download on the Kokoro model."
        ) from e
    cache = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "kokoro"
    return KokoroTTS(model_path, Path(heart).parent, cache_dir=cache,
                     languages=kokoro_installer.ready_languages(),
                     torch_config=config, device=device)


def load_model_instance(model_name: str = "VieNeu-TTS-v3-Turbo (int8)", device: str = "auto"):
    global vieneu, current_model_info
    with model_lock:
        print(f"⏳ Loading model: {model_name} (Device: {device})...")
        t0 = time.time()

        if model_name == model_catalog.SUPERTONIC_ID:
            from apps.supertonic_engine import LANGUAGES, SupertonicTTS
            vieneu = SupertonicTTS(_supertonic_dir())
            current_model_info = {
                "name": model_name, "backend": "onnx", "precision": "fp32",
                "device": "cpu", "sample_rate": vieneu.sample_rate,
                "languages": [{"code": k, "name": v} for k, v in LANGUAGES.items()],
            }
        elif model_catalog.is_kokoro(model_name):
            # Before the "gpu" branch below: the Kokoro GPU id also contains the word "GPU".
            gpu = model_name == model_catalog.KOKORO_GPU_ID
            vieneu = _load_kokoro(gpu=gpu)
            # Like Supertonic: list languages (only those with the phonemizer installed),
            # so the API `language` field and the UI language selector share the same values.
            # No "vi": the API rejects Vietnamese text (_require_language).
            current_model_info = {
                "name": model_name, "backend": vieneu.backend, "precision": "fp32",
                "device": vieneu.device, "sample_rate": vieneu.sample_rate,
                "languages": [{"code": k, "name": v} for k, v in vieneu.languages.items()],
                "supports_emotions": False,
                # Kokoro prosody is deterministic (only slight vocoder noise), so "Stable"
                # does not produce any audible difference.
                "supports_seed": False,
            }
            if gpu:
                current_model_info["gpu_name"] = gpu_name()
        elif model_name == "VieNeu-TTS-v3-Nano (preview)":
            vieneu = Vieneu(mode="v3nano")
            current_model_info = {
                "name": model_name, "backend": "onnx", "precision": "fp32",
                "device": "cpu", "sample_rate": 24_000,
            }
        elif "gpu" in model_name.lower():
            # v3 Turbo runs on PyTorch/CUDA. This is the ONLY path that uses the GPU for v3:
            # the ONNX engine hardcodes CPUExecutionProvider. It also enables static
            # batching (vieneu.v3_turbo_serve) - where the GPU wins the most, used for
            # multi-turn Conversation/Podcast mode.
            if not cuda_is_available():
                import sys
                if getattr(sys, "frozen", False):
                    raise RuntimeError(
                        "No usable NVIDIA GPU was found, or PyTorch/CUDA has not finished loading. "
                        "Click Download again on the GPU model so the app can install it automatically."
                    )
                raise RuntimeError(
                    "No usable NVIDIA GPU was found. This installation is torch-free. "
                    "Open Settings and choose Install GPU support to download the required libraries."
                )
            vieneu = Vieneu(mode="v3turbo", backend="pytorch", device="cuda")
            # Prevent VRAM from growing with the number of CUDA graphs (4 GB cards can hang): apps/gpu_memory.py
            gpu_memory.apply(vieneu)
            current_model_info = {
                "name": "VieNeu-TTS-v3-Turbo (GPU)",
                "backend": "pytorch",
                "precision": "fp32",
                "device": "cuda",
                "gpu_name": gpu_name(),
                "sample_rate": 48_000,
            }
        elif "int8" in model_name.lower():
            vieneu = Vieneu(mode="v3turbo", backend="onnx", precision="int8")
            current_model_info = {
                "name": "VieNeu-TTS-v3-Turbo (int8)",
                "backend": "onnx",
                "precision": "int8",
                "device": "cpu",
                "sample_rate": 48_000,
            }
        elif "fp32" in model_name.lower():
            vieneu = Vieneu(mode="v3turbo", backend="onnx", precision="fp32")
            current_model_info = {
                "name": "VieNeu-TTS-v3-Turbo (fp32)",
                "backend": "onnx",
                "precision": "fp32",
                "device": "cpu",
                "sample_rate": 48_000,
            }
        else:
            vieneu = Vieneu(backend="onnx")
            current_model_info = {
                "name": "VieNeu-TTS-v3-Turbo (int8)",
                "backend": "onnx",
                "precision": "int8",
                "device": "cpu",
                "sample_rate": 48_000,
            }

        elapsed = time.time() - t0
        print(f"✅ Model is ready in {elapsed:.2f}s! Backbone: {current_model_info['name']}")
        _remember_model(current_model_info["name"])

        # Reload custom voices saved by the user in a previous session (if the engine supports it).
        # This file is separate from built-in voices in the installation package, see _custom_voices_path().
        if hasattr(vieneu, "load_extra_voices"):
            try:
                so_giong = vieneu.load_extra_voices(_custom_voices_path())
                if so_giong:
                    print(f"📢 Loaded {so_giong} previously saved custom voice(s).")
            except Exception as e:
                print(f"⚠️ Could not load saved custom voices: {e}")


def _last_model_path() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "last-model.json"


def _remember_model(model_id: str) -> None:
    """Record the model that just finished loading so the next app launch loads the same one. Previously the app
    always fell back to Turbo int8 even if the user had switched to a different model in Settings."""
    try:
        path = _last_model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": model_id}), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        print(f"⚠️ Could not save the current model: {e}")


def _saved_model() -> Optional[str]:
    """Model used last time, if it still exists in the catalog and all files are downloaded."""
    try:
        model_id = json.loads(_last_model_path().read_text(encoding="utf-8"))["model"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    spec = model_catalog.get(model_id)
    if spec is None or spec not in model_catalog.MODELS:
        return None
    try:
        return model_id if model_catalog.is_downloaded(spec) else None
    except Exception:
        return None


def _friendly_load_error(e: BaseException) -> str:
    """Convert a technical error into a message the user can understand and act on."""
    from apps.runtime_paths import missing_vcredist, vcredist_error_message
    msg = str(e).lower()

    if "no module named 'torch'" in msg or "no module named 'torchaudio'" in msg or "no module named 'neucodec'" in msg:
        if model_catalog.frozen_gpu_autoinstall_possible():
            # Normally model_downloader installs this automatically before reaching here (see
            # apps/gpu_installer.py) - reaching here means the previous installation failed or was incomplete.
            return (
                "Could not load PyTorch/CUDA. Click Download again on this model so the app "
                "can install it automatically (requires internet, about 1.5 GB, may take a few minutes)."
            )
        return (
            "GPU support is not installed and no automatic installer was found. "
            "Please update or reinstall the full latest Hupesi TTS package."
        )

    if "onnxruntime" in msg or "dll load failed" in msg:
        miss = missing_vcredist()
        if miss:
            return vcredist_error_message(miss)
    root = _root_cause(e)
    if root is not e:
        # transformers lazily loads modules and wraps the real error as "Could not import
        # module 'PretrainedConfig'" - without the root cause, diagnosis is impossible.
        return f"Could not load model: {e} (cause: {type(root).__name__}: {root})"
    return f"Could not load model: {e}"


def _root_cause(e: BaseException) -> BaseException:
    seen = set()
    while id(e) not in seen:
        seen.add(id(e))
        nxt = e.__cause__ or e.__context__
        if nxt is None:
            break
        e = nxt
    return e


# Errors when loading a just-downloaded model must go through this translation function; otherwise,
# a machine missing VC++ Redist would receive a meaningless "Could not download model" message.
model_downloader.apply_error_formatter = _friendly_load_error


def _startup_load() -> None:
    """Load the model into RAM on each app launch, reusing files already downloaded to disk."""
    global model_status
    model_status = {"state": "loading", "message": "Loading speech model..."}
    # Model the user selected last time (fully downloaded). If none: use GPU when an NVIDIA GPU is usable,
    # otherwise CPU int8 - default (torch-free) installations fall back to CPU.
    # If the saved model fails to load, fall back to CPU int8 as below.
    name = _saved_model() or (
        "VieNeu-TTS-v3-Turbo (GPU)" if cuda_is_available() else "VieNeu-TTS-v3-Turbo (int8)")
    def update_loading_message(model_name: str) -> None:
        spec = model_catalog.get(model_name)
        cached = spec is not None and model_catalog.is_downloaded(spec)
        model_status["message"] = (
            f"Loading {model_name} from disk into memory..."
            if cached else f"Preparing {model_name}, downloading missing files if needed..."
        )

    try:
        update_loading_message(name)
        load_model_instance(name)
    except Exception as e:
        if name != "VieNeu-TTS-v3-Turbo (int8)":
            print(f"⚠️ Could not load {name} ({e}). Falling back to CPU int8.")
            try:
                update_loading_message("VieNeu-TTS-v3-Turbo (int8)")
                load_model_instance("VieNeu-TTS-v3-Turbo (int8)")
            except Exception as e2:
                model_status = {"state": "error", "message": _friendly_load_error(e2)}
                return
        else:
            model_status = {"state": "error", "message": _friendly_load_error(e)}
            return
    model_status = {"state": "ok", "message": "Ready"}


# Load the model in a BACKGROUND thread, without blocking on import.
#
# Previously the load ran at module level, so uvicorn only opened the port AFTER the model
# was ready. As a result, the app window showed nothing while waiting, users could not tell
# whether the machine was working or frozen - and the "loading" branch of /health as well as
# the UI loading screen became dead code because they never ran.
#
# Now the server opens the port immediately, /health reports "loading", and the UI shows a real loading screen.
model_status: Dict[str, str] = {"state": "loading", "message": "Starting..."}
# Tests set CIT_SKIP_MODEL_AUTOLOAD=1 (tests/conftest.py): this thread would load the REAL model
# and overwrite the `vieneu` variable mocked by tests - breaking tests intermittently mid-run;
# on machines without cache it would also download ~300 MB.
if os.getenv("CIT_SKIP_MODEL_AUTOLOAD") != "1":
    threading.Thread(target=_startup_load, name="model-loader", daemon=True).start()


def _pcm16(audio_f32: np.ndarray) -> bytes:
    """Convert a float32 numpy array to PCM16 bytes."""
    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def _open_output(target, sr: int, fmt: str = "WAV"):
    """Open an audio file/buffer for WRITING and attach the "AI-generated" tags (apps/ai_label.py).
    All audio produced by the server goes through here or ``wav_file_to_mp3``."""
    kw = {"subtype": "PCM_16"} if fmt == "WAV" else {}
    out = sf.SoundFile(target, "w", samplerate=sr, channels=1, format=fmt, **kw)
    ai_label.label(out)
    return out


def _wav_bytes(audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Create high-quality WAV bytes from numpy audio."""
    buf = io.BytesIO()
    with _open_output(buf, sr) as out:
        out.write(np.asarray(audio_f32, dtype=np.float32))
    return buf.getvalue()


def _mp3_bytes(audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Create MP3 bytes from numpy audio."""
    buf = io.BytesIO()
    try:
        with _open_output(buf, sr, "MP3") as out:
            out.write(np.asarray(audio_f32, dtype=np.float32))
    except Exception:
        # libsndfile has no MP3 encoder: return WAV
        return _wav_bytes(audio_f32, sr)
    return buf.getvalue()


def _apply_speed(audio_f32: np.ndarray, speed: float) -> np.ndarray:
    """Speed up/slow down audio if different from 1.0 while preserving pitch (WSOLA, see
    ``vieneu_utils.time_stretch`` - librosa's phase vocoder makes voices raspy)."""
    if abs(speed - 1.0) < 0.02 or len(audio_f32) == 0:
        return audio_f32
    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    return time_stretch(audio_f32, float(speed), sr)


def _reads_vietnamese() -> bool:
    """Whether the running model can read Vietnamese. VieNeu (no language list) and
    Supertonic (has "vi") can; Kokoro cannot."""
    codes = {item["code"] for item in current_model_info.get("languages", [])}
    return not codes or "vi" in codes


def _dict_applies() -> bool:
    """The Pronunciation Dictionary writes readings using Vietnamese spelling ("KTX" -> "ky tuc xa").
    Feeding that to a model that cannot read Vietnamese (Kokoro) produces strange sounds, so skip it."""
    return _reads_vietnamese()


# Characters unique to Vietnamese (a-breve, d-stroke, o-horn, u-horn, and vowels with Vietnamese tone marks).
# Excludes e, a, o, e-circumflex, a-tilde, o-tilde, i, o, u... because French/Spanish/Italian/Portuguese also use them.
_VIETNAMESE_ONLY_RE = re.compile(
    "[\u0103\u0111\u01a1\u01b0\u1ea1\u1ea3\u1ea5\u1ea7\u1ea9\u1eab\u1ead\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u1ebb\u1ebd\u1eb9\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u1ec9\u0129\u1ecb\u1ecf\u1ecd\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u1edb\u1edd\u1edf\u1ee1\u1ee3\u1ee7\u0169\u1ee5\u1ee9\u1eeb\u1eed\u1eef\u1ef1\u1ef3\u1ef7\u1ef9\u1ef5]",
    re.IGNORECASE)
_WORD_RE = re.compile(r"[^\W\d_]+")
# Text is considered Vietnamese when at least 1/4 of (lowercase) words contain characters unique to
# Vietnamese. Vietnamese sentences usually far exceed this threshold, while English/French sentences are near 0.
_VIETNAMESE_WORD_RATIO = 0.25


def _looks_vietnamese(text: str) -> bool:
    """Count by WORD RATIO, not character count: an English sentence mentioning a few Vietnamese place names
    can contain several Vietnamese-specific characters and still be English. Capitalized proper names are excluded,
    so only lowercase words are counted; if there are no lowercase words (all caps, or a single word), count all words."""
    # Strip tags like [nghi 2s], [cuoi]... before counting: characters inside tags are not spoken.
    words = _WORD_RE.findall(re.sub(r"\[[^\]\n]*\]", " ", text or ""))
    lower = [w for w in words if w[0].islower()] or words
    viet = sum(1 for w in lower if _VIETNAMESE_ONLY_RE.search(w))
    return viet > 0 and viet >= _VIETNAMESE_WORD_RATIO * len(lower)


def _require_language(text: str) -> None:
    """Models that cannot read Vietnamese (Kokoro) receiving Vietnamese text -> 400.
    Without this check, API clients would receive audio of a foreign voice misreading everything
    with no error reported (the app is set to Kokoro while the client still sends Vietnamese).
    Foreign text with Vietnamese proper names is still allowed (`_looks_vietnamese`)."""
    if _reads_vietnamese():
        return
    if _looks_vietnamese(text):
        raise HTTPException(
            status_code=400,
            detail="The current model (Kokoro) cannot read Vietnamese. Please switch to "
                   "VieNeu-TTS v3 Turbo in Hupesi TTS Settings.",
        )


def _text_segments(text: str, use_dict: bool = True) -> List[tuple]:
    """Split text at ``[nghi 2s]`` pause tags -> ``[(segment, seconds_after), ...]``.

    The Pronunciation Dictionary is applied AFTER splitting, on each segment: if applied before,
    a dictionary entry could corrupt the word "nghi" inside the tag.
    """
    _require_language(text)
    segments = split_pauses(text)
    if use_dict and _dict_applies():
        segments = [(pronunciation_settings.apply(seg), p) for seg, p in segments]
    if not any(seg for seg, _ in segments):
        raise HTTPException(status_code=400, detail="Text has no readable characters")
    return segments


def _synthesize(text: str, sr: int, infer_one, speed: float = 1.0, use_dict: bool = True) -> np.ndarray:
    """Synthesize each segment with ``infer_one(segment)`` then insert the exact silence duration from the pause tag.

    Speed applies to speech segments, not silence: "[nghi 2s]" remains
    2 seconds even at 1.5x speed.
    """
    parts = []
    for seg, pause in _text_segments(text, use_dict):
        if seg:
            wav = infer_one(seg)
            if wav is not None and len(wav) > 0:
                parts.append(_apply_speed(np.asarray(wav, dtype=np.float32), speed))
        if pause > 0:
            parts.append(np.zeros(int(round(sr * pause)), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


# Long text is synthesized in chunks of this size and written incrementally to file: RAM only needs to hold
# one chunk (~4 minutes of audio), not the entire piece. Previously the whole piece stayed in RAM -
# a ~2-hour piece exhausted backend memory and made the app window go blank (user report, 2026-09).
# Text shorter than this threshold is synthesized as before, in a single call.
LONG_TEXT_BLOCK_CHARS = 3000
_PARAGRAPH_SPLIT = re.compile(r"\n+")


def _text_blocks(text: str, limit: int = LONG_TEXT_BLOCK_CHARS) -> List[tuple]:
    """Split long text into ``[(chunk, silence_seconds_after), ...]`` at sentence boundaries.

    The silence between two chunks matches the gap the engine would naturally insert at that boundary
    (paragraph endings are longer than sentence endings), so chunk boundaries are inaudible.
    """
    if len(text) <= limit:
        return [(text, 0.0)]
    units = []  # (paragraph index, sentence)
    for pi, para in enumerate(p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()):
        # split_into_sentences only knows Latin sentence punctuation: long Japanese/Chinese/Hindi text
        # (Kokoro) would become ONE huge chunk held entirely in RAM. Split first at
        # CJK/Devanagari sentence marks - Vietnamese text does not contain these marks, so it splits as before.
        for piece in (x for x in _NON_LATIN_STOP.split(para) if x.strip()):
            for sent in split_into_sentences(piece) or [piece.strip()]:
                units.append((pi, sent))

    blocks, cur, cur_len = [], [], 0
    for pi, sent in units:
        if cur and cur_len + len(sent) + 1 > limit:
            gap = V3_GAP_SILENCE["para"] if pi != cur[-1][0] else V3_GAP_SILENCE["sentence"]
            blocks.append((_join_units(cur), gap))
            cur, cur_len = [], 0
        cur.append((pi, sent))
        cur_len += len(sent) + 1
    if cur:
        blocks.append((_join_units(cur), 0.0))
    return blocks


def _join_units(units: List[tuple]) -> str:
    paras: Dict[int, List[str]] = {}
    for pi, sent in units:
        paras.setdefault(pi, []).append(sent)
    return "\n".join(_join_sentences(sents) for sents in paras.values())


def _join_sentences(sents: List[str]) -> str:
    """Join sentences with spaces, except after Japanese/Chinese sentence-ending punctuation (which do not use spaces)."""
    out = ""
    for sent in sents:
        if out and not out.rstrip(CLOSING_MARKS).endswith(("。", "！", "？")):
            out += " "
        out += sent
    return out


def _silence(sr: int, seconds: float) -> np.ndarray:
    return np.zeros(int(round(sr * seconds)), dtype=np.float32)


def _synthesize_to_file(text: str, sr: int, infer_one, path, speed: float = 1.0,
                        use_dict: bool = True) -> int:
    """Like ``_synthesize`` but writes incrementally to a 16-bit WAV file and returns the number of samples written.

    Short text produces exactly the same bytes as ``_wav_bytes(_synthesize(...))``.
    """
    written = 0
    with _open_output(str(path), sr) as out:
        for seg, pause in _text_segments(text, use_dict):
            if seg:
                prev_tail, gap_before = None, 0.0
                for block, gap in _text_blocks(seg):
                    wav = infer_one(block)
                    if wav is not None and len(wav) > 0:
                        wav = _apply_speed(np.asarray(wav, dtype=np.float32), speed)
                        pad = _natural_pad(prev_tail, wav, sr, gap_before)
                        if pad:
                            out.write(np.zeros(pad, dtype=np.float32))
                            written += pad
                        out.write(wav)
                        written += len(wav)
                        # Only the tail is needed to measure silence: do not keep the entire ~4-minute chunk.
                        prev_tail = wav[-sr * 10:]
                    gap_before = gap
            if pause > 0:
                out.write(_silence(sr, pause))
                written += int(round(sr * pause))
    return written


def _synthesize_with_cues_to_file(text: str, sr: int, infer_one, path, speed: float = 1.0,
                                  use_dict: bool = True):
    """Like ``_synthesize_with_cues`` but writes incrementally to a WAV file. Returns ``(sample_count, cues)``."""
    _require_language(text)
    units = build_units(text)
    if not any(display_text(u.text) for u in units):
        raise HTTPException(status_code=400, detail="Text has no readable characters")
    cues, cursor = [], 0
    with _open_output(str(path), sr) as out:
        for wav, cue_text in _unit_audio(units, sr, infer_one, speed, use_dict):
            if cue_text is not None:
                cues.extend(unit_cues(cue_text, cursor / sr, (cursor + len(wav)) / sr))
            out.write(wav)
            cursor += len(wav)
    return cursor, cues


def _natural_pad(prev_wav, next_wav: np.ndarray, sr: int, gap: float) -> int:
    """Number of silence samples to insert at a natural sentence/paragraph/chunk boundary. ``gap`` is the TOTAL
    minimum pause like the engine (``V3_GAP_SILENCE``): existing silence at the tail of the previous audio
    and the head of the next audio is counted, and only the shortfall is padded."""
    if prev_wav is None or gap <= 0:
        return 0
    return pause_pad_samples(prev_wav, next_wav, sr, gap)


def _unit_audio(units, sr: int, infer_one, speed: float, use_dict: bool):
    """Synthesize each subtitle unit, yielding ``(audio, subtitle_text | None)`` in order: speech audio
    with its text, or silence (text ``None``). Sentence/paragraph breaks only pad the remaining gap;
    pause tags insert the exact number of seconds."""
    prev, pending = None, 0.0
    for unit in units:
        if unit.text.strip():
            speak = pronunciation_settings.apply(unit.text) if use_dict and _dict_applies() else unit.text
            wav = infer_one(speak)
            if wav is not None and len(wav) > 0:
                wav = _apply_speed(_trim_edges(np.asarray(wav, dtype=np.float32), sr), speed)
                pad = _natural_pad(prev, wav, sr, pending)
                if pad:
                    yield np.zeros(pad, dtype=np.float32), None
                yield wav, unit.text
                prev = wav
        if unit.tag_pause:
            if unit.pause_after > 0:
                yield _silence(sr, unit.pause_after), None
            prev, pending = None, 0.0
        else:
            pending = unit.pause_after


# Convert WAV -> MP3 in chunks: shared with the history MP3 download button.
wav_file_to_mp3 = history_store.wav_file_to_mp3


def _temp_audio_path(suffix: str = ".wav") -> Path:
    fd, name = tempfile.mkstemp(prefix="cit-tts-", suffix=suffix)
    os.close(fd)
    return Path(name)


def _remove_quietly(*paths) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def _trim_edges(wav: np.ndarray, sr: int, margin_s: float = 0.04) -> np.ndarray:
    """Trim leading/trailing silence from a synthesis unit (keeping a small margin) so pauses and
    subtitle timestamps are determined by us, not mixed with excess model silence."""
    if len(wav) == 0:
        return wav
    peak = float(np.max(np.abs(wav)))
    loud = np.flatnonzero(np.abs(wav) > max(1e-3, 0.02 * peak))
    if len(loud) == 0:
        return wav[:0]
    m = int(sr * margin_s)
    return wav[max(0, loud[0] - m): min(len(wav), loud[-1] + 1 + m)]


def _synthesize_with_cues(text: str, sr: int, infer_one, speed: float = 1.0, use_dict: bool = True):
    """Like ``_synthesize`` but synthesizes sentence by sentence and returns subtitles.

    Timestamps are measured from the actual audio duration of each sentence (see vieneu_utils.subtitles).
    Returns ``(wav, cues)``.
    """
    _require_language(text)
    units = build_units(text)
    if not any(display_text(u.text) for u in units):
        raise HTTPException(status_code=400, detail="Text has no readable characters")
    parts, cues, cursor = [], [], 0
    for wav, cue_text in _unit_audio(units, sr, infer_one, speed, use_dict):
        if cue_text is not None:
            cues.extend(unit_cues(cue_text, cursor / sr, (cursor + len(wav)) / sr))
        parts.append(wav)
        cursor += len(wav)
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return audio, cues


# -------------------------------------------------------------
# Web & Static UI Routes
# -------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def ui():
    if (DIST_DIR / "index.html").exists():
        return FileResponse(DIST_DIR / "index.html", media_type="text/html")
    return Response("Hupesi TTS UI not found. Build the frontend first.", status_code=404, media_type="text/plain")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # CIT logo: Vite copies public/favicon.ico into dist/, but only /assets is mounted so we must serve it manually.
    ico = DIST_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(ico, media_type="image/x-icon")
    svg ='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="82" font-size="82">🐆</text></svg>'
    return Response(svg, media_type="image/svg+xml")


def require_model():
    """Fail fast when the model has not finished loading.

    Without this, every endpoint would throw AttributeError on `None` and users
    would get a confusing 500 error instead of knowing they just need to wait.
    """
    if vieneu is None:
        raise HTTPException(
            status_code=503,
            detail=model_status.get("message", "Model is loading, please wait."),
        )


@app.get("/health")
async def health():
    """Model status, actual sample rate, and supported languages. Wait for `status: ok` before generating audio.

    `supports_cloning` refers to voice creation capability in the app, not an API that accepts sample files.
    """
    return {
        "status": model_status.get("state", "loading" if vieneu is None else "ok"),
        "message": model_status.get("message", ""),
        "sample_rate": current_model_info.get("sample_rate", SAMPLE_RATE),
        "engine": current_model_info.get("name", "VieNeu-TTS v3 Turbo (ONNX int8 CPU)"),
        "backend": current_model_info.get("backend", "onnx"),
        "precision": current_model_info.get("precision", "int8"),
        "device": current_model_info.get("device", "cpu"),
        # Multilingual models (Supertonic, Kokoro) list codes usable for `language`;
        # VieNeu only reads Vietnamese (with some English) so the list is empty.
        "languages": current_model_info.get("languages", []),
        "supports_cloning": vieneu is None or hasattr(vieneu, "add_voice"),
        # Emotion tags: VieNeu [cuoi]..., Supertonic <laugh>...; Kokoro has none (tags are stripped).
        "supports_emotions": current_model_info.get("supports_emotions", True),
        # Each generation has different prosody, so a fixed seed ("Stable") is meaningful.
        "supports_seed": current_model_info.get("supports_seed", True),
    }


@app.get("/voices")
async def voices():
    """Voices for the running model, including saved cloned voices if the model supports them.

    Use `id` as `voice_id`. The first voice is the default when `voice_id` is omitted.
    Fetch the list again after switching models; do not use VieNeu voices with Supertonic/Kokoro.
    Kokoro voices also include `language`: the language the voice was trained on.
    """
    require_model()
    try:
        vs = list(vieneu.list_preset_voices())
        # Put the model's default voice first: the UI preselects the first voice in the list,
        # and the API also uses it when voice_id is omitted. Previously the order followed `featured`
        # in the voice file, so the UI selected a different voice than the API default.
        default_voice = getattr(vieneu, "_default_voice", None)
        vs.sort(key=lambda item: (item[1] if isinstance(item, (tuple, list)) and len(item) == 2
                                  else str(item)) != default_voice)
        presets = getattr(vieneu, "_preset_voices", {})
        khz = current_model_info.get("sample_rate", SAMPLE_RATE) // 1000
        out = []
        for item in vs:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                label, vid = item
                name_str = label
                v_id = vid
            else:
                name_str = str(item)
                v_id = str(item)

            # v3 Turbo and v3 Nano presets include `gender` and a "Gender / Region / Style"
            # description; use them directly instead of guessing from the name. String guessing is
            # only a fallback for older engines (v1/v2) without metadata.
            preset = presets.get(v_id) if isinstance(presets.get(v_id), dict) else {}
            mo_ta = str(preset.get("description") or "").strip()
            gioi_tinh = str(preset.get("gender") or "").strip().lower()
            phan = [p.strip() for p in mo_ta.split("·")]
            vung = phan[1].lower() if len(phan) > 2 else ""

            if gioi_tinh not in ("male", "female"):
                name_lower = name_str.lower()
                is_female = any(x in name_lower for x in ["huy\u1ec1n", "linh", "mai", "nga", "n\u1eef", "female", "th\u1ea3o", "h\u01b0\u01a1ng", "ng\u1ecdc", "lan"])
                is_male = any(x in name_lower for x in ["\u0111\u1ee9c", "nam", "tu\u1ea5n", "h\u00f9ng", "b\u1ea3o", "qu\u00e2n", "male", "v\u0129nh", "minh"])
                gioi_tinh = "female" if is_female else ("male" if is_male else "neutral")

            if preset.get("language"):
                region = None  # Kokoro voice: each voice is a single language, no regional accent
            elif vung in ("nam", "mi\u1ec1n nam"):
                region = "South"
            elif vung in ("b\u1eafc", "mi\u1ec1n b\u1eafc"):
                region = "North"
            elif vung == "\u0111a ng\u00f4n ng\u1eef":
                region = None  # Supertonic voice: no Vietnamese regional accent
            else:
                name_lower = name_str.lower()
                region = "South" if ("nam" in name_lower or "sg" in name_lower or "s\u00e0i g\u00f2n" in name_lower) else "North"

            entry = {
                "id": v_id,
                "isCloned": bool(preset.get("is_custom", False)),
                # Voice selected by the VieNeu authors (listed first), marked with a star in the UI.
                "featured": preset.get("featured") is not None,
                "name": v_id,
                "gender": gioi_tinh,
                "region": region,
                "description": f"{mo_ta} · {khz} kHz" if mo_ta else f"Voices {v_id} ({khz} kHz)",
            }
            if preset.get("language"):
                # Language the voice was trained on (Kokoro), using codes like `languages` in /health.
                # The UI automatically selects that language when the user picks this voice.
                entry["language"] = str(preset["language"])
            out.append(entry)
        return out or [{"id": "Minh \u0110\u1ee9c", "name": "Minh \u0110\u1ee9c", "gender": "male", "region": "North", "description": "Standard Northern voice"}]
    except Exception as e:
        return [{"id": "default", "name": f"⚠️ {e}", "gender": "neutral", "region": "North", "description": str(e)}]


# -------------------------------------------------------------
# Streaming TTS (Chunked Realtime)
# -------------------------------------------------------------
STREAM_RESPONSE = {
    200: {
        "description": "Mono PCM16 WAV stream: WAV header followed by audio data; sample rate is in the WAV header and /health.",
        "headers": {ai_label.HEADER: {
            "description": "Always \"true\": AI-generated audio. The stream has no in-file tag, "
                           "so callers should label it themselves when saving or publishing.",
            "schema": {"type": "string"},
        }},
        "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}},
    }
}

AUDIO_RESPONSE = {
    200: {
        "description": "WAV or MP3 file. Check the actual Content-Type; X-Sample-Rate indicates the model sample rate.",
        "headers": {ai_label.HEADER: {
            "description": "Always \"true\": AI-generated audio; the file also carries an \"AI-generated\" tag.",
            "schema": {"type": "string"},
        }},
        "content": {
            mime: {"schema": {"type": "string", "format": "binary"}}
            for mime in ("audio/wav", "audio/mpeg")
        },
    }
}


@app.get("/stream", response_class=StreamingResponse, responses=STREAM_RESPONSE)
async def stream(text: str, voice_id: Optional[str] = None, language: Optional[str] = None):
    """Stream synthesized speech as WAV using a built-in or saved voice.

    `language`: code from `/health` (Supertonic; Kokoro reads in the voice's language); VieNeu ignores it. Does not accept sample voice files.
    This endpoint has no speed, temperature, or seed parameters.
    """
    require_model()
    _require_voice(voice_id)  # validate before sending the stream's 200 header
    lang_kw = _language_kwargs(language)
    engine = vieneu
    segments = _text_segments(text)
    sample_rate = engine.sample_rate
    def gen():
        h = io.BytesIO()
        with wave.open(h, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.setnframes(1_000_000_000)
        yield h.getvalue()

        t0 = time.perf_counter()
        first_at = None
        n_chunks = 0
        emitted = 0
        for seg, pause in segments:
            chunks = engine.infer_stream(seg, voice=voice_id or None, **lang_kw) if seg else ()
            for chunk in chunks:
                if chunk is None or len(chunk) == 0:
                    continue
                if first_at is None:
                    first_at = time.perf_counter() - t0
                    print(f"⚡ TTFA: {first_at * 1000:.0f} ms")
                n_chunks += 1
                emitted += len(chunk)
                yield _pcm16(chunk)
            if pause > 0:  # [nghi Ns] pause tag: emit silence directly in the stream
                silence = np.zeros(int(round(sample_rate * pause)), dtype=np.float32)
                emitted += len(silence)
                yield _pcm16(silence)
        if first_at is not None:
            gen_time = time.perf_counter() - t0
            audio_s = emitted / sample_rate
            rtf = gen_time / audio_s if audio_s else 0
            print(f"✅ {n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s | RTF {rtf:.3f}")

    return StreamingResponse(gen(), media_type="audio/wav")


class StreamReq(BaseModel):
    text: str = Field(description="Text to synthesize; supports pause tags like [nghi 2s].")
    voice_id: Optional[str] = Field(default=None, description="id from GET /voices; omit to use the running model's default voice.")
    language: Optional[str] = Field(default=None, description="Language code from GET /health for Supertonic; omit to auto-detect from text. Kokoro always reads in the voice's language, VieNeu ignores it.")


@app.post("/stream", response_class=StreamingResponse, responses=STREAM_RESPONSE)
async def stream_post(req: StreamReq):
    """Like GET /stream, but sends text, voice_id, and language as JSON."""
    return await stream(req.text, req.voice_id, req.language)


# -------------------------------------------------------------
# High-Quality TTS Synthesis (Batch / Full WAV / MP3)
# -------------------------------------------------------------
class TtsGenerateReq(BaseModel):
    text: str = Field(description="Text to synthesize, must not be empty. Can include pause tags like [nghi 2s].")
    voice_id: Optional[str] = Field(default=None, description="id from GET /voices, including saved cloned voices. Omit for the default voice; does not accept sample files.")
    speed: float = Field(default=1.0, description="Speed factor, must be greater than 0; recommended 0.75-1.5. Does not stretch pause tags.")
    temperature: float = Field(default=0.8, description="Randomness for VieNeu v3 Turbo; the UI allows 0.1-1.5. Nano, Supertonic, and Kokoro ignore it.")
    silence_duration: float = Field(default=0.2, description="Legacy compatibility parameter, has no effect in generate and generate-with-subtitles. Use pause tags like [nghi 2s].")
    format: str = Field(default="wav", description="wav or mp3. generate may return WAV if the machine cannot encode MP3; check Content-Type. With subtitles, read format from the JSON.")
    # False when text has already been replaced (Pronunciation Dictionary "Preview" button) - replacing
    # twice would read incorrectly, e.g. a case-insensitive "ai" entry could corrupt surrounding text.
    use_pronunciation_dict: bool = Field(default=True, description="Apply the app's pronunciation dictionary. Set false if the client has already replaced words.")
    # A fixed seed makes the same text + same voice produce identical output each time ("Stable").
    # None: sample randomly as before, different prosody each time.
    seed: Optional[int] = Field(default=None, description="Fix randomness to reproduce results with the same text, voice, model, settings, and environment; does not improve quality by itself.")
    # Language code ("en", "ja"...) for multilingual models (Supertonic, Kokoro). If omitted,
    # the model auto-detects from text; VieNeu ignores this parameter.
    language: Optional[str] = Field(default=None, description="Language code from GET /health (Supertonic); omit to auto-detect from text. Kokoro always reads in the voice's language, VieNeu ignores it.")


def _language_kwargs(language: Optional[str]) -> dict:
    """``{"lang": ...}`` when the running model supports multiple languages, otherwise empty:
    clients that already send ``language`` continue to work when the app switches back to VieNeu."""
    codes = {item["code"] for item in current_model_info.get("languages", [])}
    if not codes or not language:
        return {}
    if language not in codes:
        raise HTTPException(
            status_code=400,
            detail=f"The running model does not support language '{language}'. "
                   f"Valid codes: {', '.join(sorted(codes))} (see GET /health).")
    return {"lang": language}


def _require_voice(voice_id: Optional[str]) -> None:
    """Voice name not available in the running model -> 400 with a helpful message. Previously the engine
    threw ValueError mid-request and the API returned 500 "Voice '...' not found", making API clients
    think the server was broken. Empty = default voice, always valid."""
    if not voice_id:
        return
    presets = getattr(vieneu, "_preset_voices", None)
    if not isinstance(presets, dict):
        return  # older engine (v1/v2) has no catalog: let the engine report the error
    resolve = getattr(vieneu, "resolve_voice_name", None)
    known = resolve(voice_id) if callable(resolve) else (voice_id if voice_id in presets else None)
    if known is None:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice_id}' is not available in the running model. See the voice list at GET /voices "
                   "(voice_id is the voice name, accent-sensitive), or omit it to use the default voice.",
        )


def _tts_infer(seed: Optional[int] = None, **kwargs):
    """``infer_one(segment)`` helper for text synthesis paths. If ``seed`` is provided, the random
    generator is reset before EACH segment so a segment always produces the same audio regardless of
    its position: Turbo ONNX samples with ``np.random``, GPU with ``torch``, and Nano takes a separate
    ``seed`` parameter. Voice name is validated at construction time (before synthesis)."""
    _require_voice(kwargs.get("voice"))
    if seed is None:
        return lambda seg: vieneu.infer(seg, **kwargs)
    import inspect

    seed = int(seed) % (2 ** 32)
    extra = {"seed": seed} if "seed" in inspect.signature(vieneu.infer).parameters else {}

    def run(seg):
        np.random.seed(seed)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                torch.manual_seed(seed)
            except Exception:
                pass
        return vieneu.infer(seg, **kwargs, **extra)

    return run


class _ClientGone(Exception):
    """The browser cancelled the request (Stop button) during synthesis."""


def _cancellable(infer_one, cancelled: Optional[threading.Event]):
    """Stop at the next segment when ``cancelled`` is set (user clicked Stop)."""
    if cancelled is None:
        return infer_one

    def run(seg):
        if cancelled.is_set():
            raise _ClientGone()
        return infer_one(seg)

    return run


async def _run_while_connected(request: Request, work):
    """Run ``work(cancelled)`` in a threadpool while monitoring the connection: if the browser disconnects
    (Stop button cancels fetch), set ``cancelled`` so ``work`` stops at the next segment.

    Previously the server did not know the user had stopped: it would synthesize the entire piece (keeping
    the CPU busy the whole time) and still save it to history, so "stopped" items would appear in History."""
    import asyncio
    from starlette.concurrency import run_in_threadpool

    cancelled = threading.Event()

    async def watch():
        while not await request.is_disconnected():
            await asyncio.sleep(0.5)
        cancelled.set()

    watcher = asyncio.create_task(watch())
    try:
        return await run_in_threadpool(work, cancelled)
    except _ClientGone:
        print("⏹️ User stopped - discarding the current synthesis, not saving to history.")
        raise HTTPException(status_code=499, detail="Stopped by request; not saved to History.")
    finally:
        watcher.cancel()


@app.post("/api/tts/generate", response_class=FileResponse, responses=AUDIO_RESPONSE)
async def generate_tts(req: TtsGenerateReq):
    """Generate a WAV/MP3 file from text using a built-in or saved cloned voice.

    Returns binary audio, not JSON. Does not create a new voice from a sample file.
    Headers X-Duration-Seconds, X-Elapsed-Seconds, X-RTF, and X-Sample-Rate describe the result.
    """
    require_model()
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    t0 = time.time()
    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    wav_path = _temp_audio_path(".wav")
    try:
        samples = _synthesize_to_file(
            req.text.strip(),
            sr,
            _tts_infer(req.seed, voice=req.voice_id or None, temperature=req.temperature,
                   **_language_kwargs(req.language)),
            wav_path,
            speed=req.speed,
            use_dict=req.use_pronunciation_dict,
        )
    except HTTPException:
        _remove_quietly(wav_path)
        raise
    except Exception as e:
        _remove_quietly(wav_path)
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")

    if samples == 0:
        _remove_quietly(wav_path)
        raise HTTPException(status_code=500, detail="Could not generate audio data")

    duration_sec = samples / sr
    elapsed = time.time() - t0
    rtf = elapsed / duration_sec if duration_sec > 0 else 0

    # Stream directly from file (FileResponse reads incrementally) instead of building the whole piece in RAM.
    path, media_type = wav_path, "audio/wav"
    if req.format.lower() == "mp3":
        mp3_path = wav_path.with_suffix(".mp3")
        if wav_file_to_mp3(wav_path, mp3_path):
            _remove_quietly(wav_path)
            path, media_type = mp3_path, "audio/mpeg"

    headers = {
        "X-Duration-Seconds": f"{duration_sec:.2f}",
        "X-Elapsed-Seconds": f"{elapsed:.2f}",
        "X-RTF": f"{rtf:.3f}",
        "X-Sample-Rate": str(sr),
    }

    return FileResponse(path, media_type=media_type, headers=headers,
                        background=BackgroundTask(_remove_quietly, path))


class TtsToHistoryReq(TtsGenerateReq):
    subtitles: bool = False
    voice_name: str = ""
    mode: str = "tts"


@app.post("/api/tts/generate-to-history", include_in_schema=False)
async def generate_tts_to_history(req: TtsToHistoryReq, request: Request):
    return await _run_while_connected(request, lambda cancelled: _generate_to_history(req, cancelled))


def _generate_to_history(req: TtsToHistoryReq, cancelled: Optional[threading.Event] = None):
    """Generate speech and save it directly to disk history; returns only the history item metadata.

    For the UI: audio does not need to pass through the browser and be uploaded back to save.
    A ~2-hour piece (~690 MB WAV) made the app window go blank when the browser held, played,
    and uploaded that blob at the same time (user report, 2026-09). The UI plays and downloads
    file qua ``/api/history/{id}/...``.
    """
    require_model()
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    t0 = time.time()
    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    text = req.text.strip()
    infer_one = _cancellable(
        _tts_infer(req.seed, voice=req.voice_id or None, temperature=req.temperature,
                   **_language_kwargs(req.language)), cancelled)
    wav_path = history_store.temp_file(".wav")
    try:
        if req.subtitles:
            samples, cues = _synthesize_with_cues_to_file(
                text, sr, infer_one, wav_path, speed=req.speed, use_dict=req.use_pronunciation_dict)
        else:
            samples, cues = _synthesize_to_file(
                text, sr, infer_one, wav_path, speed=req.speed, use_dict=req.use_pronunciation_dict), None
        if samples == 0:
            raise HTTPException(status_code=500, detail="Could not generate audio data")
        duration_sec = samples / sr
        srt = to_srt(cues) if cues else None
        if cancelled is not None and cancelled.is_set():
            raise _ClientGone()  # stopped just as synthesis finished: still do not save
        entry = history_store.add_file(
            wav_path, "wav", text=text, voice_id=req.voice_id or "",
            voice_name=req.voice_name or req.voice_id or "",
            duration_sec=duration_sec, subtitle_srt=srt,
        )
    except (HTTPException, _ClientGone):
        _remove_quietly(wav_path)
        raise
    except Exception as e:
        _remove_quietly(wav_path)
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")
    elapsed = time.time() - t0
    return {
        "entry": entry,
        "sample_rate": sr,
        "duration_seconds": round(duration_sec, 3),
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration_sec, 3) if duration_sec else 0,
        "srt": srt,
    }


class SrtParseReq(BaseModel):
    srt: str


def _parse_srt_or_400(content: str):
    try:
        return parse_srt(content or "")
    except SrtError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/srt/parse", include_in_schema=False)
async def srt_parse(req: SrtParseReq):
    """Read a subtitle file ahead of time so the UI can display the sentence list."""
    cues = _parse_srt_or_400(req.srt)
    return {
        "cues": [{"start": round(c.start, 3), "end": round(c.end, 3), "text": c.text} for c in cues],
        "duration_seconds": round(max(c.end for c in cues), 3),
    }


class DubSrtReq(BaseModel):
    srt: str
    voice_id: Optional[str] = None
    voice_name: str = ""
    speed: float = 1.0
    temperature: float = 0.8
    use_pronunciation_dict: bool = True
    language: Optional[str] = None


@app.post("/api/tts/dub-srt-to-history", include_in_schema=False)
async def dub_srt_to_history(req: DubSrtReq, request: Request):
    return await _run_while_connected(request, lambda cancelled: _dub_srt_to_history(req, cancelled))


def _dub_srt_to_file(cues, sr: int, infer_one, path, speed: float = 1.0,
                     use_dict: bool = True) -> tuple:
    """Dub according to subtitle timestamps, writing incrementally to a WAV file that matches the timeline.

    Returns ``(sample_count, DubStats)``. Each sentence is synthesized separately with silence trimmed on both
    ends so it starts at the correct cue; frame alignment follows ``vieneu_utils.dubbing``."""
    stats = dubbing.DubStats()
    cursor = 0  # number of samples written
    with _open_output(str(path), sr) as out:
        for cue in cues:
            speak = pronunciation_settings.apply(cue.text) if use_dict and _dict_applies() else cue.text
            wav = infer_one(speak)
            if wav is None or len(wav) == 0:
                continue
            wav = _trim_edges(np.asarray(wav, dtype=np.float32), sr)
            if len(wav) == 0:
                continue
            # The user-selected speed and the extra speed-up needed to fit the frame are combined into ONE
            # time-stretch pass: applying two stretches in a row compounds audio artifacts.
            rate = dubbing.fit_rate(len(wav) / speed / sr, cue.end - cue.start)
            wav = _apply_speed(wav, speed * rate)
            start = dubbing.place(cue.start, cursor / sr, stats.cues > 0)
            start_sample = int(round(start * sr))
            if start_sample > cursor:
                out.write(np.zeros(start_sample - cursor, dtype=np.float32))
                cursor = start_sample
            stats.record(cue, cursor / sr, len(wav) / sr, rate)
            out.write(wav)
            cursor += len(wav)
        # Audio must extend at least to the end of the last subtitle cue to match video duration.
        tail = int(round(max(c.end for c in cues) * sr))
        if stats.cues and tail > cursor:
            out.write(np.zeros(tail - cursor, dtype=np.float32))
            cursor = tail
    return cursor, stats


def _dub_srt_to_history(req: DubSrtReq, cancelled: Optional[threading.Event] = None):
    require_model()
    cues = _parse_srt_or_400(req.srt)
    _require_language("\n".join(c.text for c in cues))
    t0 = time.time()
    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    infer_one = _cancellable(
        _tts_infer(None, voice=req.voice_id or None, temperature=req.temperature,
                   **_language_kwargs(req.language)), cancelled)
    wav_path = history_store.temp_file(".wav")
    try:
        samples, stats = _dub_srt_to_file(
            cues, sr, infer_one, wav_path, speed=req.speed, use_dict=req.use_pronunciation_dict)
        if stats.cues == 0:
            raise HTTPException(status_code=500, detail="Could not generate audio data")
        duration_sec = samples / sr
        srt = to_srt(stats.finish())
        if cancelled is not None and cancelled.is_set():
            raise _ClientGone()
        entry = history_store.add_file(
            wav_path, "wav", text="\n".join(c.text for c in cues), voice_id=req.voice_id or "",
            voice_name=req.voice_name or req.voice_id or "",
            duration_sec=duration_sec, subtitle_srt=srt,
        )
    except (HTTPException, _ClientGone):
        _remove_quietly(wav_path)
        raise
    except Exception as e:
        _remove_quietly(wav_path)
        raise HTTPException(status_code=500, detail=f"Dubbing error: {e}")
    elapsed = time.time() - t0
    return {
        "entry": entry,
        "sample_rate": sr,
        "duration_seconds": round(duration_sec, 3),
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration_sec, 3) if duration_sec else 0,
        "srt": srt,
        "stats": {
            "cues": stats.cues,
            "sped_up": stats.sped_up,
            "pushed": stats.pushed,
            "max_push_seconds": round(stats.max_push_seconds, 2),
        },
    }


@app.post("/api/tts/generate-with-subtitles")
async def generate_tts_with_subtitles(req: TtsGenerateReq):
    """Like ``/api/tts/generate`` but returns JSON containing audio (base64) + SRT subtitles.

    Synthesizes sentence by sentence to measure real timestamps for each sentence; subtitles show the
    ORIGINAL text (before pronunciation dictionary replacement), with emotion and pause tags removed.
    """
    require_model()
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    t0 = time.time()
    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    wav_path = _temp_audio_path(".wav")
    mp3_path = wav_path.with_suffix(".mp3")
    try:
        samples, cues = _synthesize_with_cues_to_file(
            req.text.strip(),
            sr,
            _tts_infer(req.seed, voice=req.voice_id or None, temperature=req.temperature,
                   **_language_kwargs(req.language)),
            wav_path,
            speed=req.speed,
            use_dict=req.use_pronunciation_dict,
        )
        if samples == 0:
            raise HTTPException(status_code=500, detail="Could not generate audio data")
        fmt = "wav"
        if req.format.lower() == "mp3" and wav_file_to_mp3(wav_path, mp3_path):
            fmt = "mp3"
        audio_data = (mp3_path if fmt == "mp3" else wav_path).read_bytes()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")
    finally:
        _remove_quietly(wav_path, mp3_path)
    duration_sec = samples / sr
    elapsed = time.time() - t0
    return {
        "audio_base64": base64.b64encode(audio_data).decode("ascii"),
        "format": fmt,
        "sample_rate": sr,
        "duration_seconds": round(duration_sec, 3),
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration_sec, 3) if duration_sec else 0,
        "srt": to_srt(cues),
        "cues": [{"start": round(c.start, 3), "end": round(c.end, 3), "text": c.text} for c in cues],
    }


# The engine uses at most 8 seconds of speech from the sample file (after trimming silence at both ends);
# 30 seconds is ample without having to read an hours-long recording.
REF_AUDIO_MAX_READ_SECONDS = 30.0


def _normalize_ref_audio(file_bytes: bytes, original_filename: str = "sample.wav") -> str:
    """
    Convert arbitrary sample audio data (WAV, MP3, WebM, OGG, M4A, FLAC, AAC...)
    into a normalized 16-bit PCM mono WAV file so Vieneu and SoundFile can always load it.
    """
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out_path = tmp_out.name
    tmp_out.close()
    # Only read the beginning: the engine uses at most 8 seconds of speech. A 60-minute recording
    # was previously read in full and silence-trimmed over the entire buffer - slow and memory-intensive.
    max_s = REF_AUDIO_MAX_READ_SECONDS

    # 1. Try reading with soundfile (WAV, FLAC, OGG, MP3...)
    try:
        with sf.SoundFile(io.BytesIO(file_bytes)) as f:
            data = f.read(frames=int(f.samplerate * max_s), dtype="float32", always_2d=True)
            sr = f.samplerate
        sf.write(tmp_out_path, data.mean(axis=1), sr, format="WAV", subtype="PCM_16")
        return tmp_out_path
    except Exception:
        pass

    # 2. Try reading with librosa (supports more formats via audioread/soxr/soundfile)
    try:
        import librosa
        data, sr = librosa.load(io.BytesIO(file_bytes), sr=48000, mono=True, duration=max_s)
        sf.write(tmp_out_path, data, 48000, format="WAV", subtype="PCM_16")
        return tmp_out_path
    except Exception:
        pass

    # 3. Try reading with pydub if available (requires ffmpeg - usually not installed on user machines)
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))[: int(max_s * 1000)]
        seg = seg.set_channels(1).set_frame_rate(48000)
        seg.export(tmp_out_path, format="wav")
        return tmp_out_path
    except Exception:
        pass

    # Previously the original file was saved as-is and the engine reported "Format not recognised" -
    # giving users a confusing error for M4A/MP4/AAC/WMA/WebM files (on machines without ffmpeg).
    _remove_quietly(tmp_out_path)
    suffix = Path(original_filename).suffix.lower() or "unknown"
    raise HTTPException(
        status_code=400,
        detail=(f"Could not read the sample audio file (format {suffix}). "
                "Please use an MP3, WAV, FLAC, or OGG file, or select the file through the app UI "
                "so it can be converted automatically."),
    )


# -------------------------------------------------------------
# Microphone and file uploads share the same local usage-rights commitment; identity is not verified.
# Direct cloning from uploaded files (/api/tts/clone) was removed in 1.0.2.
# -------------------------------------------------------------
# The UI trims sample clips to at most 15 seconds; allow half a second for decoding tolerance.
MAX_REF_SECONDS = 15.5


def _duration_seconds(wav_path: str) -> float:
    try:
        return sf.info(wav_path).duration
    except Exception:
        return 0.0


@app.post("/api/voices/custom", include_in_schema=False)
async def add_custom_voice(
    name: str = Form(...),
    ref_audio: UploadFile = File(...),
    denoise: bool = Form(True),
    input_source: Literal["recording", "file"] = Form("recording"),
    rights_confirmed: bool = Form(False),
    rights_version: str = Form(""),
):
    """Clone a voice from a sample audio file and save it permanently as `name` for later use."""
    require_model()
    ten = (name or "").strip()
    if not ten:
        raise HTTPException(status_code=400, detail="Voice name must not be empty")
    if not hasattr(vieneu, "add_voice"):
        raise HTTPException(
            status_code=400,
            detail="The current model does not support voice cloning. Please switch to VieNeu-TTS v3 Turbo.",
        )
    if ten in getattr(vieneu, "_preset_voices", {}):
        raise HTTPException(status_code=409, detail="A voice with this name already exists. Please choose a different name.")
    if not rights_confirmed:
        raise HTTPException(status_code=400, detail="Please confirm the right to clone and use this voice before saving.")
    if rights_version != terms.TERMS_VERSION:
        raise HTTPException(status_code=409, detail="The commitment has been updated. Reopen the Voice Cloning tab and confirm again.")

    file_bytes = await ref_audio.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The sample audio file contains no data")
    tmp_ref_path = _normalize_ref_audio(file_bytes, ref_audio.filename or "sample.wav")

    try:
        duration = _duration_seconds(tmp_ref_path)
        if not 3 <= duration <= MAX_REF_SECONDS:
            raise HTTPException(status_code=400, detail="The sample must be 3-15 seconds long. A natural 3-8 second reading without background music is recommended.")
        # The commitment and voice data are stored atomically in the same file; deleting the voice also
        # deletes the commitment. The original file, IP address, or user identity is not stored.
        authorization = {
            "confirmed": True,
            "statement": terms.VOICE_RIGHTS_STATEMENT,
            "terms_version": terms.TERMS_VERSION,
            "accepted_at": datetime.now(timezone.utc).isoformat(),
            "input_source": input_source,
            "verification": "user_declaration",
        }
        with model_lock:
            if ten in getattr(vieneu, "_preset_voices", {}):
                raise HTTPException(status_code=409, detail="A voice with this name already exists. Please choose a different name.")
            voice_id = vieneu.add_voice(
                ten, tmp_ref_path, denoise=denoise, save=True, save_path=_custom_voices_path(),
                authorization=authorization,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving custom voice: {e}")
    finally:
        _remove_quietly(tmp_ref_path)

    return {"status": "ok", "voice_id": voice_id, "name": ten}


@app.get("/api/voices/consent", include_in_schema=False)
def consent_info():
    """Voice-rights commitment shown in the Voice Cloning tab, including the terms version."""
    return {"rights_statement": terms.VOICE_RIGHTS_STATEMENT, "terms_version": terms.TERMS_VERSION}


@app.delete("/api/voices/custom/{name}", include_in_schema=False)
async def remove_custom_voice(name: str):
    """Delete a saved custom voice."""
    require_model()
    if not hasattr(vieneu, "remove_voice"):
        raise HTTPException(
            status_code=400,
            detail="The current model does not support deleting custom voices.",
        )
    preset = getattr(vieneu, "_preset_voices", {}).get(name)
    if preset is None:
        raise HTTPException(status_code=404, detail="Custom voice not found.")
    if not preset.get("is_custom", False):
        raise HTTPException(status_code=400, detail="Built-in voices cannot be deleted.")
    # The cache key is derived from voice data, so the path must be resolved before deletion.
    preview = _voice_preview_path(name)
    try:
        vieneu.remove_voice(name, save=True, save_path=_custom_voices_path())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting custom voice: {e}")
    preview.unlink(missing_ok=True)
    return {"status": "ok"}


# -------------------------------------------------------------
# Voice preview before selection
# -------------------------------------------------------------
VOICE_PREVIEW_TEXT = (
    "Xin ch\u00e0o, \u0111\u00e2y l\u00e0 gi\u1ecdng \u0111\u1ecdc m\u1eabu c\u1ee7a Hupesi TTS. "
    "Ch\u00fac b\u1ea1n m\u1ed9t ng\u00e0y l\u00e0m vi\u1ec7c th\u1eadt hi\u1ec7u qu\u1ea3."
)


def _preview_text(language: Optional[str] = None, voice_id: Optional[str] = None) -> tuple:
    """``(sample sentence, language parameter)``. Supertonic reads the sample in the language selected
    in the UI (English if none). Kokoro ALWAYS uses a sample in the voice's own language: the engine
    ignores `lang` and reads in the voice's language, so an English sentence for a Japanese voice would
    produce meaningless sounds. The sample text is part of the cache key (`_voice_preview_path`), so
    changing the sentence changes the file."""
    codes = {item["code"] for item in current_model_info.get("languages", [])}
    if not codes:
        return VOICE_PREVIEW_TEXT, {}
    kokoro = model_catalog.is_kokoro(current_model_info.get("name"))
    if kokoro:
        from apps.kokoro_engine import PREVIEW_TEXTS
    else:
        from apps.supertonic_engine import PREVIEW_TEXTS

    if kokoro or language not in PREVIEW_TEXTS or language not in codes:
        preset = getattr(vieneu, "_preset_voices", {}).get(voice_id)
        language = preset.get("language") if isinstance(preset, dict) else None
        if language not in PREVIEW_TEXTS:
            language = "en"
    return PREVIEW_TEXTS[language], {"lang": language}


def voice_preview_dir() -> Path:
    return Path(os.getenv("LOCALAPPDATA") or Path.home()) / "HupesiTTS" / "voice_previews"


def _voice_fingerprint(voice_id: str) -> str:
    """Hash voice data (voice vector, reference code, style). The 3.8.3 voice set kept old names but
    changed the data for 8 voices; a key based only on the name would keep playing the old cached sample."""
    import hashlib

    preset = getattr(vieneu, "_preset_voices", {}).get(voice_id)
    h = hashlib.sha1()
    for field in ("speaker_emb", "codes", "style"):
        value = preset.get(field) if isinstance(preset, dict) else None
        if isinstance(value, np.ndarray):
            h.update(field.encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def _voice_preview_path(voice_id: str, text: str = VOICE_PREVIEW_TEXT) -> Path:
    """One file per (model, voice, voice data, sample sentence): Turbo and Nano share voice names but
    sound different, and changing voice data or the sample sentence automatically invalidates the old cache."""
    import hashlib

    key = "\n".join([
        current_model_info.get("name", ""),
        str(current_model_info.get("sample_rate", SAMPLE_RATE)),
        voice_id,
        _voice_fingerprint(voice_id),
        # When the engine changes, reading/pausing behavior changes too: old cached samples become stale.
        "engine vieneu 3.8.3",
        text,
    ])
    return voice_preview_dir() / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.wav"


@app.get("/api/voices/preview", include_in_schema=False)
async def voice_preview(voice_id: str, language: Optional[str] = None):
    """Synthesize the sample sentence with `voice_id`. The first call generates audio (a few seconds);
    subsequent calls play back from the on-disk cache. Does not apply the pronunciation dictionary so
    the cached sample remains stable."""
    require_model()
    if voice_id not in getattr(vieneu, "_preset_voices", {}):
        raise HTTPException(status_code=404, detail="This voice was not found in the current model.")

    text, lang_kw = _preview_text(language, voice_id)
    path = _voice_preview_path(voice_id, text)
    if path.is_file():
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    try:
        wav = _synthesize(
            text, sr, lambda seg: vieneu.infer(seg, voice=voice_id, **lang_kw), use_dict=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating voice preview: {e}")
    if len(wav) == 0:
        raise HTTPException(status_code=500, detail="Could not generate preview audio")

    data = _wav_bytes(wav, sr)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    except OSError as e:
        # If the cache cannot be written, playback still works; it just has to be regenerated next time.
        print(f"[Preview] Could not save voice preview cache: {e}")
    return Response(content=data, media_type="audio/wav", headers={"Cache-Control": "no-store"})


# -------------------------------------------------------------
# Multi-Speaker Conversation & Podcast Mode
# -------------------------------------------------------------
class DialogueTurn(BaseModel):
    speaker: str = Field(description="Speaker name for this turn.")
    text: str = Field(description="Turn content; supports pause and emotion tags depending on the model.")
    voice_id: Optional[str] = Field(default=None, description="id from /voices; takes priority over speaker_mapping. If both are omitted, the default voice is used.")


class ConversationReq(BaseModel):
    script: Optional[str] = Field(default=None, description="Multi-line script in the form [MC]: Hello. When script has content, it takes priority over turns.")
    turns: Optional[List[DialogueTurn]] = Field(default=None, description="List of dialogue turns; used when no script content is provided.")
    speaker_mapping: Optional[Dict[str, str]] = Field(default=None, description="Map speaker names to voice ids from /voices.")
    silence_duration: float = Field(default=0.4, description="Pause between turns in seconds; effective minimum is 0.05 seconds.")
    speed: float = Field(default=1.0, description="Speed factor greater than 0; recommended 0.75-1.5. Does not stretch pauses between turns or pause tags.")
    temperature: float = Field(default=0.8, description="Randomness for VieNeu v3 Turbo; Nano, Supertonic, and Kokoro ignore it.")
    seed: Optional[int] = Field(default=None, description="Fix randomness as in /api/tts/generate.")
    format: str = Field(default="wav", description="wav or mp3.")
    language: Optional[str] = Field(default=None, description="Code from /health, applied to all turns on Supertonic. Kokoro: each voice reads in its own language. VieNeu ignores it.")


@app.post("/api/tts/conversation", response_class=Response, responses=AUDIO_RESPONSE)
async def generate_conversation(req: ConversationReq):
    """Generate a complete multi-speaker conversation / podcast."""
    require_model()
    turns_list: List[DialogueTurn] = []

    # 1. Parse from raw script if turns list is not provided
    if req.script and req.script.strip():
        # Example syntax: [Speaker 1]: Hello | or Speaker 1: Hello
        pattern = re.compile(r"^\s*(?:\[([^\]]+)\]|([^:]+))\s*:\s*(.+)$", re.MULTILINE)
        matches = pattern.findall(req.script)
        if matches:
            for m in matches:
                spk = (m[0] or m[1]).strip()
                content = m[2].strip()
                if spk and content:
                    turns_list.append(DialogueTurn(speaker=spk, text=content))
        else:
            # Fallback: split by lines
            lines = [line.strip() for line in req.script.split("\n") if line.strip()]
            for i, line in enumerate(lines):
                spk = f"Speaker {(i % 2) + 1}"
                turns_list.append(DialogueTurn(speaker=spk, text=line))
    elif req.turns:
        turns_list = req.turns

    if not turns_list:
        raise HTTPException(status_code=400, detail="No dialogue turns found in the script")

    mapping = req.speaker_mapping or {}
    lang_kw = _language_kwargs(req.language)
    sr = current_model_info.get("sample_rate", SAMPLE_RATE)
    silence_arr = np.zeros(int(sr * max(0.05, req.silence_duration)), dtype=np.float32)

    def turn_voice(turn):
        spk = turn.speaker.strip()
        return turn.voice_id or mapping.get(spk) or mapping.get(spk.lower()) or None

    # Validate voices for all turns BEFORE synthesis: previously a turn with an invalid voice was silently
    # skipped (the loop below swallowed errors), producing incomplete audio with no warning.
    for turn in turns_list:
        try:
            _require_voice(turn_voice(turn))
            _require_language(turn.text)
        except HTTPException as e:
            raise HTTPException(status_code=400, detail=f"Turn for '{turn.speaker.strip()}': {e.detail}") from e

    all_wavs = []
    t0 = time.time()

    for i, turn in enumerate(turns_list):
        spk_name = turn.speaker.strip()
        voice_id = turn_voice(turn)

        try:
            # Speed is applied per turn (inside _synthesize) - pauses between turns
            # and [nghi Ns] pause tags keep their exact configured duration.
            chunk_wav = _synthesize(
                turn.text.strip(),
                sr,
                _tts_infer(req.seed, voice=voice_id, temperature=req.temperature, **lang_kw),
                speed=req.speed,
            )
            if chunk_wav is None or len(chunk_wav) == 0:
                raise ValueError("The model returned no audio")
            all_wavs.append(chunk_wav)
            if i < len(turns_list) - 1:
                all_wavs.append(silence_arr)
        except Exception as e:
            print(f"⚠️ Dialogue segment error {i+1} ({spk_name}): {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not generate turn {i+1} for '{spk_name}'. Please try again; no audio was exported because the conversation is incomplete.",
            ) from e

    full_wav = np.concatenate(all_wavs)

    duration_sec = len(full_wav) / sr
    elapsed = time.time() - t0

    if req.format.lower() == "mp3":
        audio_data = _mp3_bytes(full_wav, sr)
        # _mp3_bytes returns WAV if the machine lacks an MP3 encoder; MIME type must match the data.
        media_type = "audio/wav" if audio_data.startswith(b"RIFF") else "audio/mpeg"
    else:
        audio_data = _wav_bytes(full_wav, sr)
        media_type = "audio/wav"

    headers = {
        "X-Duration-Seconds": f"{duration_sec:.2f}",
        "X-Elapsed-Seconds": f"{elapsed:.2f}",
        "X-Turns-Count": str(len(turns_list)),
        "X-Sample-Rate": str(sr),
    }

    return Response(content=audio_data, media_type=media_type, headers=headers)


# -------------------------------------------------------------
# Document & PDF Text Extraction
# -------------------------------------------------------------
@app.post("/api/pdf/extract", include_in_schema=False)
async def extract_pdf_text(file: UploadFile = File(...)):
    """Extract text content from PDF, Word (.docx), or TXT files."""
    filename = file.filename or ""
    content = await file.read()
    ext = Path(filename).suffix.lower()

    if ext == ".doc":
        # Legacy Word 97-2003 binary format: reading it would require Word/LibreOffice on the machine.
        raise HTTPException(
            status_code=415,
            detail="Legacy .doc files (Word 97-2003) are not supported. Open the file in Word, "
                   "choose Save As -> .docx format, then upload it again.",
        )

    if ext == ".docx":
        from vieneu_utils.docx_extract import DocxError, extract_docx
        try:
            text, page_count = extract_docx(content)
        except DocxError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not text.strip():
            raise HTTPException(status_code=400, detail="The .docx file contains no readable text.")
        return {"text": text, "page_count": page_count, "pages": [text], "filename": filename}

    if ext == ".txt":
        try:
            text = content.decode("utf-8")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
        return {"text": text, "page_count": 1, "pages": [text]}

    # Try PyMuPDF (fitz) or pypdf
    pages_text = []
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            pages_text.append(page.get_text())
    except Exception:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not decode PDF: {e}")

    full_text = "\n\n".join([p.strip() for p in pages_text if p.strip()])
    return {
        "text": full_text,
        "page_count": len(pages_text),
        "pages": pages_text,
        "filename": filename,
    }


# -------------------------------------------------------------
# Convert WAV to MP3 API
# -------------------------------------------------------------
@app.post("/api/audio/mp3", include_in_schema=False)
async def convert_to_mp3_endpoint(file: UploadFile = File(...)):
    """Accept a WAV file and convert it to high-quality MP3."""
    try:
        content = await file.read()
        data, sr = sf.read(io.BytesIO(content), dtype="float32")
        mp3_data = _mp3_bytes(data, sr)
        return Response(content=mp3_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MP3 conversion error: {e}")


# -------------------------------------------------------------
# Model Management API
# -------------------------------------------------------------
def _torch_installed() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _mo_ta_model(spec, *, has_torch: bool, has_cuda: bool) -> Dict[str, Any]:
    """One entry in `/api/models`.

    `available` keeps its old meaning (= runnable on this machine) so older frontends behave the same;
    `compatible` and `downloaded` are the new, separate flags.
    """
    from apps import kokoro_installer

    can_autoinstall = model_catalog.frozen_gpu_autoinstall_possible()
    can_kokoro = kokoro_installer.available()
    ok, ly_do = model_catalog.compatible(
        spec, has_torch=has_torch, has_cuda=has_cuda, can_autoinstall_torch=can_autoinstall,
        has_kokoro_deps=can_kokoro,
    )
    da_tai = model_catalog.is_downloaded(spec)
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "supports_cloning": spec.supports_cloning,
        "supports_streaming": spec.supports_streaming,
        "precision": spec.precision,
        "device": spec.device,
        "sample_rate": spec.sample_rate,
        "available": ok,
        "compatible": ok,
        "downloaded": da_tai,
        "needs_gpu_setup": "torch" in spec.requires and can_autoinstall and not cuda_is_available(),
        "needs_addon_setup": "kokoro_deps" in spec.requires and can_kokoro and not kokoro_installer.deps_ready(),
        # Additional language packs (Kokoro: Japanese, Chinese).
        "packs": [
            {"code": p.code, "name": p.name, "approx_bytes": p.approx_bytes,
             "installed": kokoro_installer.pack_ready(p.code) and model_catalog.kokoro_pack_downloaded(p)}
            for p in model_catalog.KOKORO_PACKS
        ] if model_catalog.is_kokoro(spec.id) else [],
        "approx_bytes": spec.approx_bytes,
        "unavailable_reason": ly_do,
        "note": spec.note,
    }


@app.get("/api/models", include_in_schema=False)
async def list_available_models():
    """List models with two separate flags: runnable on this machine, and already downloaded.

    Combining those into one flag was why users could select FP32 and stare at a spinner for
    minutes without knowing a 453 MB download was in progress. `has_cuda` uses the "effective"
    check: treat as having a GPU if torch reports one, OR if the machine has a physical GPU even
    though torch is not installed yet - so the GPU model download button is not incorrectly
    disabled on .exe builds that have not installed torch.
    """
    has_cuda = _effective_has_cuda()
    has_torch = _torch_installed()
    return {
        "current": current_model_info,
        "gpu": {"available": cuda_is_available(), "name": gpu_name()},
        "models": [
            _mo_ta_model(s, has_torch=has_torch, has_cuda=has_cuda)
            for s in model_catalog.MODELS
        ],
    }


class LoadModelReq(BaseModel):
    model_name: str
    device: str = "auto"


# Download/install/switch models: only from the app UI on this machine. CORS is open ("*"), so without
# Origin/Host checks a foreign web page open in the browser could call these endpoints - e.g. forcing the
# machine to download hundreds of MB of models or install Python packages from PyPI for Kokoro.
_THIS_MACHINE = [Depends(remote_access.require_this_machine)]


@app.post("/api/models/load", include_in_schema=False, dependencies=_THIS_MACHINE)
async def switch_model(req: LoadModelReq):
    """Switch to the selected model."""
    try:
        load_model_instance(req.model_name, req.device)
        return {"status": "ok", "current": current_model_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model: {e}")


class DownloadReq(BaseModel):
    model_id: str
    auto_apply: bool = True
    # Kokoro language packs ("ja"/"zh"): download only the pack's voices and phonemizer.
    pack: Optional[str] = None


def _reload_kokoro_if_current(_model_id: str) -> None:
    """After installing a language pack: reload Kokoro if it is currently in use so new voices and
    languages appear. If a different model is in use, leave it unchanged - do not switch the user's model."""
    if model_catalog.is_kokoro(current_model_info.get("name")):
        load_model_instance(current_model_info["name"])


@app.post("/api/models/download", status_code=202, include_in_schema=False,
          dependencies=_THIS_MACHINE)
async def start_model_download(req: DownloadReq):
    """Accept a model download job in a background thread. Returns immediately without waiting for completion."""
    ok, ly_do = model_downloader.start(
        req.model_id, auto_apply=req.auto_apply,
        apply_fn=_reload_kokoro_if_current if req.pack else load_model_instance,
        pack=req.pack,
    )
    if not ok:
        raise HTTPException(status_code=409, detail=ly_do)
    return {"status": "accepted", "model_id": req.model_id}


@app.get("/api/models/download", include_in_schema=False)
async def model_download_status():
    return model_downloader.get_state()


@app.post("/api/models/download/cancel", status_code=202, include_in_schema=False,
          dependencies=_THIS_MACHINE)
async def cancel_model_download():
    if not model_downloader.cancel():
        raise HTTPException(status_code=409, detail="No download job is currently running.")
    return {"status": "cancelling"}
def _openapi_with_api_key():
    """Them nut "Authorize" (khoa X-API-Key) vao /docs de thu API tu may khac.
    Moi API cong khai deu ghi "khong khoa HOAC co khoa": cung may thi khong can."""
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Only required when calling from other machines. Get the key in Settings -> Allow access from other devices.",
        }
    }
    for path, operations in schema.get("paths", {}).items():
        if path in remote_access.PUBLIC_PATHS:
            for operation in operations.values():
                operation["security"] = [{}, {"ApiKey": []}]
    app.openapi_schema = schema
    return schema


app.openapi = _openapi_with_api_key


def main():
    host = remote_access.listen_host()
    remote_access.mark_listening(host)
    uvicorn.run(app, host=host, port=setup_port(), ws="none")  # app does not use WebSocket


if __name__ == "__main__":
    main()
