"""Model catalog: which models consist of which files and what they require to run.

Single source of truth for both `fetch_models.py` (pre-download during packaging) and
`apps/model_downloader.py` (download at runtime). Previously file lists were duplicated in
multiple places and drifted out of sync.

Do NOT import torch / onnxruntime / transformers at top level: this module is called by
`/api/models` on every request, and the default installation is torch-free.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

V3_REPO = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
NANO_REPO = "pnnbao-ump/VieNeu-TTS-v3-Nano"
NANO_FILES: Tuple[str, ...] = (
    "text_encoder.onnx", "duration_predictor.onnx", "vector_estimator.onnx",
    "codec_decoder.onnx", "config.json", "constants.npz",
    "speaker_encoder.onnx", "codec_encoder.onnx", "reference_encoder.onnx", "denoiser.onnx",
)
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
CODEC_REPO_TORCH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
# Supertonic 3 (Supertone, OpenRAIL-M): non-Vietnamese voices, apps/supertonic_engine.py. The repo
# is no longer maintained (2026-07); pin the revision so files do not change under the decoder.
SUPERTONIC_REPO = "supertone-oss-archive/supertonic-3"
SUPERTONIC_REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"
SUPERTONIC_ID = "Supertonic 3 (31 ng\u00f4n ng\u1eef)"
SUPERTONIC_FILES: Tuple[str, ...] = (
    "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx", "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx", "onnx/tts.json", "onnx/unicode_indexer.json",
    *(f"voice_styles/{v}{i}.json" for v in "FM" for i in range(1, 6)),
    # OpenRAIL-M license and model card (license, authorship) always ship with the weights.
    # Only audio_samples/ and img/ from the repo are excluded (~13 MB of demo files not used by the app).
    "LICENSE", "README.md", "config.json",
)

# Kokoro-82M (hexgrad, Apache-2.0) ONNX build: one model reading native voices in 8 languages
# (no Vietnamese). Only the fp32 build is used - the 8-bit build (model_quantized/q8) measured
# ~10x slower on an i7-8650U (RTF 5.7 vs 0.5) because ONNX Runtime handles dynamic quantized
# matmul very poorly on CPU. The first letter of the voice code indicates the language (see
# apps/kokoro_engine.LANGS). The phonemizer (espeak-ng GPL, misaki + Japanese/Chinese dictionaries)
# is NOT bundled with the installer - see apps/kokoro_installer.py.
KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
# Pin revision: new users always download the exact tested version; upstream repo changes have no effect.
# To upgrade: change the sha here, retest, then release.
KOKORO_REVISION = "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
KOKORO_ID = "Kokoro-82M (Multilingual)"
# GPU build: original PyTorch weights from the author (hexgrad), running apps/_kokoro_torch on
# CUDA in the GPU venv. Voices share the same .bin files from KOKORO_REPO - byte-for-byte identical
# to hexgrad's voices/*.pt (verified) - so language packs serve both builds.
KOKORO_TORCH_REPO = "hexgrad/Kokoro-82M"
KOKORO_TORCH_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_GPU_ID = "Kokoro-82M (Multilingual, GPU)"
KOKORO_IDS = (KOKORO_ID, KOKORO_GPU_ID)
# Base pack: English + languages espeak can read (Spanish, French, Hindi, Italian, Portuguese) -
# 13 non-English voices add only ~7 MB and share the same phonemizer.
KOKORO_VOICE_PREFIXES = ("af", "am", "bf", "bm", "ef", "em", "ff", "hf", "hm",
                         "if", "im", "pf", "pm")


@dataclass(frozen=True)
class KokoroPack:
    """Additional language pack for Kokoro, downloaded only when the user requests it: voices +
    a separate phonemizer (Japanese dictionary ~250 MB on disk). Not a ModelSpec - cannot be loaded
    on its own, only adds voices to an already downloaded Kokoro model."""
    code: str                        # language code, matches kokoro_engine.LANGS
    name: str
    voice_prefixes: Tuple[str, ...]
    marker: str                      # one voice file used as the "downloaded" marker
    approx_bytes: int

    @property
    def patterns(self) -> Tuple[str, ...]:
        return tuple(f"voices/{p}_*.bin" for p in self.voice_prefixes)


KOKORO_PACKS: Tuple[KokoroPack, ...] = (
    KokoroPack("ja", "Japanese", ("jf", "jm"), "voices/jf_alpha.bin", 52 * 1024 * 1024),
    KokoroPack("zh", "Chinese", ("zf", "zm"), "voices/zf_xiaoxiao.bin", 25 * 1024 * 1024),
)


def is_kokoro(model_id: Optional[str]) -> bool:
    return model_id in KOKORO_IDS


def kokoro_pack(code: str) -> Optional[KokoroPack]:
    return next((p for p in KOKORO_PACKS if p.code == code), None)

# Must match `_GRAPH_FILES` / `_CODEC_FILES` in
# src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py (vendored copy, not modified).
# tests/test_model_catalog.py guards against drift.
GRAPH_FILES: Tuple[str, ...] = (
    "vieneu_prefill.onnx", "vieneu_decode_step.onnx",
    "vieneu_acoustic_cached.onnx", "vieneu_backbone_shared.data",
    "vieneu_v3_heads.npz", "config.json", "tokenizer.json",
)
CODEC_FILES: Tuple[str, ...] = (
    "moss_audio_tokenizer_decode_full.onnx", "moss_audio_tokenizer_decode_shared.data",
    "moss_audio_tokenizer_decode_step.onnx", "codec_browser_onnx_meta.json",
    "moss_audio_tokenizer_encode.onnx", "moss_audio_tokenizer_encode.data",
)
# At the repo root, used for voice cloning. Always downloaded; otherwise voice cloning would
# still need network access on first use - exactly what we want to avoid.
ROOT_FILES: Tuple[str, ...] = ("config.json", "denoiser.onnx", "speaker_encoder.onnx")


@dataclass(frozen=True)
class RepoFiles:
    """A file source on HF: either a fixed file list or glob patterns.

    ONNX uses `files` because the list is known exactly. The PyTorch path (v3 GPU) uses `patterns`
    because it loads via `AutoModel.from_pretrained(trust_remote_code=True)` with no fixed manifest -
    in that case `markers` must name a few key files to check whether the download is complete.
    """
    repo: str
    subfolder: Optional[str] = None
    files: Optional[Tuple[str, ...]] = None
    patterns: Optional[Tuple[str, ...]] = None
    markers: Tuple[str, ...] = ()
    # Pin a fixed revision (sha). None = use the latest sha at download time.
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        if bool(self.files) == bool(self.patterns):
            raise ValueError("RepoFiles requires exactly ONE of `files` or `patterns`")
        if self.patterns and not self.markers:
            raise ValueError("A `patterns` source must specify `markers`")

    @property
    def allow_patterns(self) -> Tuple[str, ...]:
        """String(s) passed directly to `snapshot_download(allow_patterns=...)`."""
        if self.patterns:
            return self.patterns
        prefix = f"{self.subfolder}/" if self.subfolder else ""
        return tuple(prefix + f for f in (self.files or ()))

    @property
    def marker_paths(self) -> Tuple[str, ...]:
        """Files that must exist in the cache for the download to be considered complete."""
        return self.markers or self.allow_patterns


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    sources: Tuple[RepoFiles, ...]
    requires: Tuple[str, ...] = ()
    approx_bytes: int = 0
    precision: str = "int8"
    device: str = "cpu"
    sample_rate: int = 48_000
    supports_cloning: bool = True
    supports_streaming: bool = True
    note: Optional[str] = None


# Model bundled with releases (copy_release_models.py copies its files); other models are downloaded in Settings.
BUNDLED_MODEL_ID = "VieNeu-TTS-v3-Turbo (int8)"

_ONNX_ROOT = RepoFiles(repo=V3_REPO, files=ROOT_FILES)
_ONNX_CODEC = RepoFiles(repo=CODEC_REPO, files=CODEC_FILES)

MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        id="VieNeu-TTS-v3-Turbo (int8)",
        name="VieNeu-TTS v3 Turbo (INT8 CPU - Recommended)",
        description="Very fast, int8-compressed and optimized for CPU, 48 kHz audio.",
        sources=(
            RepoFiles(repo=V3_REPO, subfolder="onnx_int8", files=GRAPH_FILES),
            _ONNX_ROOT,
            _ONNX_CODEC,
        ),
        requires=(),
        approx_bytes=312 * 1024 * 1024,
        precision="int8",
        device="cpu",
    ),
    ModelSpec(
        id="VieNeu-TTS-v3-Turbo (GPU)",
        name="VieNeu-TTS v3 Turbo (NVIDIA GPU - Fastest)",
        description=(
            "Runs on PyTorch/CUDA at 48 kHz. Batches entire scripts, making it significantly "
            "faster in Conversation/Podcast mode."
        ),
        sources=(
            RepoFiles(
                repo=V3_REPO,
                patterns=("update/*", "*.onnx", "config.json"),
                markers=("update/config.json", "speaker_encoder.onnx"),
            ),
            RepoFiles(
                repo=CODEC_REPO_TORCH,
                patterns=("*.json", "*.py", "*.safetensors"),
                markers=("config.json",),
            ),
        ),
        requires=("torch", "cuda"),
        approx_bytes=1500 * 1024 * 1024,
        precision="fp32",
        device="cuda",
    ),
    ModelSpec(
        id="VieNeu-TTS-v3-Turbo (fp32)",
        name="VieNeu-TTS v3 Turbo (FP32 Studio Quality)",
        description="Maximum studio-quality 48 kHz audio, supports voice cloning and emotion tags.",
        sources=(
            RepoFiles(repo=V3_REPO, subfolder="onnx_update", files=GRAPH_FILES),
            _ONNX_ROOT,
            _ONNX_CODEC,
        ),
        requires=(),
        approx_bytes=453 * 1024 * 1024,
        precision="fp32",
        device="cpu",
        note="Additional download required on first use.",
    ),
    ModelSpec(
        id=SUPERTONIC_ID,
        name="Supertonic 3 (31 languages \u00b7 CPU \u00b7 44.1 kHz)",
        description=(
            "Voices in English, Japanese, Korean, French, German, Spanish, and more (31 languages), "
            "10 male/female voices. Best for non-Vietnamese text."
        ),
        sources=(RepoFiles(repo=SUPERTONIC_REPO, files=SUPERTONIC_FILES, revision=SUPERTONIC_REVISION),),
        approx_bytes=401_301_114,
        precision="fp32",
        device="cpu",
        sample_rate=44_100,
        supports_cloning=False,
        note="No voice cloning. Use VieNeu for Vietnamese.",
    ),
    ModelSpec(
        id=KOKORO_ID,
        name="Kokoro (Multilingual \u00b7 CPU \u00b7 24 kHz)",
        description=(
            "Native voices: English (US, UK), Spanish, French, Italian, Portuguese, Hindi; "
            "Japanese and Chinese available as add-ons. Does not read Vietnamese."
        ),
        sources=(
            RepoFiles(
                repo=KOKORO_REPO,
                revision=KOKORO_REVISION,
                patterns=("onnx/model.onnx",
                          *(f"voices/{p}_*.bin" for p in KOKORO_VOICE_PREFIXES)),
                # One French voice as a marker: if the cache only has English voices (previous version),
                # treat as not downloaded; clicking Download will fetch the missing files.
                markers=("onnx/model.onnx", "voices/af_heart.bin", "voices/ff_siwis.bin"),
            ),
        ),
        requires=("kokoro_deps",),
        approx_bytes=350 * 1024 * 1024,
        precision="fp32",
        device="cpu",
        sample_rate=24_000,
        supports_cloning=False,
        note=(
            "First download is about 350 MB for the model plus a phonemizer (~22 MB) "
            "installed on your machine. No voice cloning, no emotion tag support."
        ),
    ),
    ModelSpec(
        id=KOKORO_GPU_ID,
        name="Kokoro (Multilingual \u00b7 NVIDIA GPU \u00b7 24 kHz)",
        description=(
            "Same as multilingual Kokoro but runs on an NVIDIA GPU, much faster for "
            "long text. Does not read Vietnamese."
        ),
        sources=(
            RepoFiles(repo=KOKORO_TORCH_REPO, files=("config.json", "kokoro-v1_0.pth"),
                      revision=KOKORO_TORCH_REVISION),
            RepoFiles(
                repo=KOKORO_REPO,
                revision=KOKORO_REVISION,
                patterns=tuple(f"voices/{p}_*.bin" for p in KOKORO_VOICE_PREFIXES),
                markers=("voices/af_heart.bin", "voices/ff_siwis.bin"),
            ),
        ),
        requires=("torch", "cuda", "kokoro_deps"),
        approx_bytes=340 * 1024 * 1024,
        precision="fp32",
        device="cuda",
        sample_rate=24_000,
        supports_cloning=False,
        note=(
            "First download is about 340 MB for the model, a phonemizer (~22 MB), and GPU support "
            "if not already installed. No voice cloning, no emotion tag support."
        ),
    ),
    ModelSpec(
        id="VieNeu-TTS-v3-Nano (preview)",
        name="VieNeu-TTS v3 Nano (Preview \u00b7 CPU \u00b7 24 kHz)",
        description="Lightweight CPU model supporting Vietnamese speech and voice cloning, 24 kHz audio.",
        sources=(RepoFiles(repo=NANO_REPO, files=NANO_FILES),),
        approx_bytes=419918885,
        precision="fp32",
        device="cpu",
        sample_rate=24_000,
        supports_cloning=True,
        supports_streaming=False,
        note="Lower quality and weaker English than Turbo.",
    ),
)


def check_has_physical_nvidia_gpu() -> bool:
    """Whether this machine has a physical NVIDIA GPU with a valid driver - no torch required.

    Used to decide whether to show the download button for GPU models when torch is not installed
    (the .exe does not bundle torch). On Windows: try loading `nvcuda.dll` (installed by the NVIDIA
    driver, even without torch/CUDA toolkit). On Linux: check for the driver file.
    """
    if sys.platform != "win32":
        from pathlib import Path
        return Path("/proc/driver/nvidia/version").exists()

    import ctypes
    try:
        return bool(ctypes.windll.kernel32.LoadLibraryW("nvcuda.dll"))
    except Exception:
        return False


def frozen_gpu_autoinstall_possible() -> bool:
    """Whether on-demand GPU installation is possible (name kept for compatibility).

    The standalone Python backend also supports installation from Settings, not just PyInstaller.
    Only checks for the actual installer tool, not sys.frozen.
    """
    from apps.gpu_installer import find_uv
    return find_uv() is not None


def get(model_id: str) -> Optional[ModelSpec]:
    for spec in MODELS:
        if spec.id == model_id:
            return spec
    return None


def compatible(
    spec: ModelSpec, *, has_torch: bool, has_cuda: bool, can_autoinstall_torch: bool = False,
    has_kokoro_deps: bool = False,
) -> tuple[bool, Optional[str]]:
    """Whether this machine can run the model, with a user-facing reason when not.

    Accepts flags from outside instead of probing directly, so tests do not need to patch
    torch/hardware and this module stays torch-free. `can_autoinstall_torch=True` (e.g. .exe with
    `uv`) treats missing torch as temporary - the model still appears compatible, and actual
    installation happens during download/apply (`apps/model_downloader.py`).
    `has_kokoro_deps` is similar for Kokoro's phonemizer: already installed, or installable.
    """
    co = {"torch": has_torch or can_autoinstall_torch, "cuda": has_cuda,
          "kokoro_deps": has_kokoro_deps}

    reason_torch = (
        "GPU support is not installed and no automatic installer was found. "
        "Please update or reinstall the full latest Hupesi TTS package."
    )
    _LY_DO = {
        "torch": reason_torch,
        "cuda": "No usable NVIDIA GPU was found on this machine.",
        "kokoro_deps": (
            # macOS: kokoro_installer.available() is always False - not supported, and updating the app won't help.
            "Kokoro is not yet supported on macOS. On Mac, use VieNeu-TTS for Vietnamese "
            "or Supertonic 3 for other languages."
            if sys.platform == "darwin" else
            "The tool needed to install the Kokoro phonemizer was not found. "
            "Please update or reinstall the full latest Hupesi TTS package."
        ),
    }

    for yeu_cau in spec.requires:
        if not co.get(yeu_cau, False):
            return False, _LY_DO.get(yeu_cau, f"Missing {yeu_cau}.")
    return True, None


def kokoro_pack_downloaded(pack: KokoroPack) -> bool:
    """Whether the pack's voices are already in the local HF cache. Does not access the network."""
    import huggingface_hub

    try:
        huggingface_hub.hf_hub_download(KOKORO_REPO, pack.marker, repo_type="model",
                                        revision=KOKORO_REVISION, local_files_only=True)
        return True
    except Exception:
        return False


def is_downloaded(spec: ModelSpec) -> bool:
    """Whether all required files are already in the local HF cache. Does not access the network."""
    import huggingface_hub

    for src in spec.sources:
        for path in src.marker_paths:
            try:
                huggingface_hub.hf_hub_download(
                    src.repo, path, repo_type="model", revision=src.revision,
                    local_files_only=True,
                )
            except Exception:
                return False
    return True
