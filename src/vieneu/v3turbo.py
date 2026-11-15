"""
VieNeu-TTS v3 Turbo backend (PyTorch).
======================================
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo")
    wav = tts.infer("Xin chào", ref_audio="ref.wav")   # clone a voice
    wav = tts.infer("Xin chào", voice="Xuân Vĩnh")     # preset voice
    tts.save(wav, "out.wav")

``style`` is DEPRECATED on v3 Turbo: the speaking style is already implied by the
reference voice (speaker embedding + reference codes), so generation is always the
natural style. The argument is still accepted everywhere for backward compatibility,
but it is ignored.
"""
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from .base import BaseVieneuTTS
from ._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
from vieneu_utils.phonemize_text import (
    phonemize_text_with_emotions,
    normalize_to_chunks_v3_with_gaps,
)
from vieneu_utils.core_utils import (
    join_audio_chunks,
    gaps_to_silence,
    max_expected_frames,
    pause_pad_samples,
    BABBLE_MAX_RETRIES,
    strip_encoder_pad_frame,
)


def _is_cuda_oom(e: BaseException) -> bool:
    """Out-of-VRAM error, detected by class name/message to avoid importing torch here
    (the CPU path must remain torch-free). Also covers CUDA graph capture errors."""
    if type(e).__name__ == "OutOfMemoryError":
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _cap_frames(sampling: dict, cap: int) -> dict:
    """Return a copy of ``sampling`` with ``max_new_frames`` capped at ``cap``.

    Limits extra "babbling" when the stop token is missed on a short chunk (see
    :func:`vieneu_utils.core_utils.max_expected_frames`); long chunks are unaffected
    because the cap always exceeds the default ``max_new_frames``.
    """
    out = dict(sampling)
    out["max_new_frames"] = min(out.get("max_new_frames", 300), cap)
    return out

logger = logging.getLogger("Vieneu.V3Turbo")


def _featured_rank(v: dict) -> Optional[int]:
    """``featured`` from a voices JSON entry as a positive int, else ``None``."""
    try:
        r = int(v.get("featured"))
    except (TypeError, ValueError):
        return None
    return r if r > 0 else None


def sorted_voices(presets: Dict[str, dict]) -> List[Tuple[str, dict]]:
    """``(name, entry)`` pairs: editors' picks first (by ``featured`` rank), then the
    rest in insertion order."""
    return sorted(presets.items(),
                  key=lambda kv: (kv[1].get("featured") is None, kv[1].get("featured") or 0))


def voice_label(name: str, v: dict) -> str:
    """Dropdown / CLI label: ``⭐ Name — description`` for editors' picks."""
    label = f"{name} — {v['description']}" if v.get("description") else name
    return f"⭐ {label}" if v.get("featured") is not None else label


class V3TurboVieNeuTTS(BaseVieneuTTS):
    """VieNeu-TTS v3 Turbo (PyTorch)."""

    @staticmethod
    def _preclean_reference_audio(
        ref_audio: Union[str, Path],
        *,
        top_db: int = 30,
        out_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Trim silence at the edges and save a cleaned reference clip.
        """
        import os
        import tempfile
        import soundfile as sf

        src = Path(ref_audio)
        if not src.exists():
            return str(src)

        wav, sr = sf.read(str(src), dtype="float32", always_2d=False)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)

        try:
            import librosa
            wav_trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
        except Exception:
            nonzero = np.flatnonzero(np.abs(wav) > 1e-5)
            if nonzero.size == 0:
                wav_trimmed = wav
            else:
                start = max(int(nonzero[0]) - int(0.01 * sr), 0)
                end = min(int(nonzero[-1]) + int(0.01 * sr), len(wav))
                wav_trimmed = wav[start:end]

        if out_path is None:
            try:
                secure_dir = Path.home() / ".cache" / "vieneu" / "tmp"
            except Exception:
                secure_dir = Path("./outputs/tmp")

            secure_dir.mkdir(parents=True, exist_ok=True)
            try:
                secure_dir.chmod(0o700)
            except Exception:
                pass

            fd, out_path = tempfile.mkstemp(
                prefix="temp_clone_optimized_",
                suffix=".wav",
                dir=str(secure_dir)
            )
            os.close(fd)
            out_path = Path(out_path)

        sf.write(str(out_path), wav_trimmed, sr)
        return str(out_path)

    def __init__(
        self,
        backbone_repo: str = "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        model_subfolder: str = "update",
        moss_tokenizer: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        device: str = "auto",
        dtype: str = "auto",
        backend: str = "auto",   # "auto" → ONNX on CPU, PyTorch on GPU; "onnx"|"pytorch" to force
        onnx_repo: Optional[str] = None,
        onnx_dir: Optional[str] = None,
        precision: str = "fp32",   # ONNX/CPU backbone: "fp32" (default, maximum quality) | "int8" (~3x faster/frame, 4x smaller; requires VNNI-capable CPU to avoid distortion)
        onnx_subfolder: Optional[str] = None,   # manual subfolder override; None → derived from `precision`
        threads: int = 0,   # ONNX/CPU intra-op threads; 0 = engine default (~physical cores, capped at 8). Set explicitly to fine-tune.
        max_batch_size: int = 32,   # GPU/PyTorch: max chunks merged into one forward pass (static batching). Actual batch = min(chunk_count, max_batch_size). Ignored on CPU/ONNX.
        max_streams: int = 16,   # GPU/PyTorch: number of concurrent `infer_stream` calls served (continuous batching, one CUDA graph). Measured on RTX 3060: 8 streams first chunk ~135 ms, 16 ~200 ms, 32 ~260 ms. Ignored on CPU/ONNX.
        babble_retries: int = BABBLE_MAX_RETRIES,   # chunks <= 3 words that "babble" (more phoneme clusters than words) are regenerated up to N times; 0 = disable
        **kwargs: Any,
    ):
        super().__init__()
        self.sample_rate = 48_000
        self.babble_retries = max(0, int(babble_retries))

        # `precision` only applies to the ONNX/CPU path (selects int8 vs fp32 graph subfolder).
        # The PyTorch/GPU path uses torch fp32/bf16 and is unrelated.
        if onnx_subfolder is None:
            onnx_subfolder = {"int8": "onnx_int8", "fp32": "onnx_update"}.get(str(precision).lower(), "onnx_update")

        if device in (None, "auto"):
            try:
                import torch
                dev_type = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev_type = "cpu"
        else:
            dev_type = "cuda" if "cuda" in str(device).lower() else str(device).lower()
        use_onnx = backend == "onnx" or (backend == "auto" and dev_type == "cpu")

        if use_onnx:
            # Torch-free CPU engine. Reads its ONNX graphs from `onnx_subfolder` in the
            # model repo (uploaded separately).
            from ._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (ONNX/CPU) from: {backbone_repo}/{onnx_subfolder} ...")
            self.engine = OnnxV3LiteEngine(
                checkpoint_path=backbone_repo,
                onnx_repo=onnx_repo,
                onnx_dir=onnx_dir,
                onnx_subfolder=onnx_subfolder,
                threads=threads,
            )
            self.backend = "onnx"
        else:
            from ._v3_turbo_engine import VieNeuTTSv3Turbo
            logger.info(f"⏳ Loading VieNeu-TTS v3 Turbo (PyTorch) from: {backbone_repo}/{model_subfolder} ...")
            self.engine = VieNeuTTSv3Turbo(
                checkpoint_path=backbone_repo,
                model_subfolder=model_subfolder,
                moss_tokenizer_path=moss_tokenizer,
                device=device,
                dtype=dtype,
            )
            self.backend = "pytorch"
        self.engine.babble_retries = self.babble_retries   # guard runs at the engine layer
        logger.info(f"✅ VieNeu-TTS v3 Turbo ready (backend={self.backend})")

        # Style is deprecated on v3 Turbo: it is implied by the reference (speaker
        # embedding + ref codes), so every generation uses the natural style. Kept
        # only as voice metadata / for backward-compatible call signatures.
        self.default_style = "tu_nhien"
        self._preset_voices: dict = {}
        # Old names that still resolve to a (renamed) preset, e.g. "Minh Quân Pro" →
        # "Hải Đăng". Filled from the voices JSON ("aliases" per entry) so
        # existing API clients and saved scripts keep working after a rename.
        self._voice_aliases: dict = {}
        self._default_voice: Optional[str] = None
        # Enrolled references, keyed by clip CONTENT (blake2b of the file bytes) +
        # enrol flags. Enrolling = denoise + x-vector + codec encode ≈ 2.7 s at
        # 5-10 cores on a 6-core CPU (measured 2026-09-15), and a document's
        # chunks / repeated requests reuse the same clip. Content-keyed so a
        # fresh temp path per upload still hits.
        from collections import OrderedDict
        self._ref_cache: "OrderedDict[tuple, Tuple[np.ndarray, Optional[np.ndarray]]]" = OrderedDict()
        self.backbone_repo = backbone_repo
        self._load_v3_voices()
        self._load_repo_voices(backbone_repo)

        # Static-batching (GPU/PyTorch). Built lazily on the first batch; None on CPU/ONNX.
        self.max_batch_size = max(1, int(max_batch_size))
        self._batch_engine = None
        # Streaming scheduler (GPU/PyTorch): one worker thread + one CUDA graph
        # serving all running `infer_stream` calls. Built lazily on the first stream.
        self.max_streams = max(1, int(max_streams))
        self._stream_sched = None
        self._stream_lock = threading.Lock()

    # ── Preset voices (speaker embedding + reference codes) ─────────────────────
    def _load_v3_voices(self, path: Optional[Union[str, Path]] = None, *, set_default: bool = True) -> None:
        """Load voices from a JSON file (built-in by default; same schema reused by
        ``load_extra_voices`` to merge in a second file without touching this one).

        Each preset carries a 192-d ``speaker_emb`` and pre-encoded ``codes``.
        """
        import json
        path = Path(path) if path else Path(__file__).parent / "assets" / "voices_v3_turbo.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, v in data.get("presets", {}).items():
            # A legacy custom voice file may also contain built-in voices: keep the installed ones.
            if not set_default and name in self._preset_voices:
                continue
            emb = v.get("speaker_emb")
            codes = v.get("codes")
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "is_custom": not set_default or v.get("is_custom", False),
                "authorization": v.get("authorization"),
                # User-saved voices are never marked as featured.
                "featured": _featured_rank(v) if set_default else None,
                "aliases": list(v.get("aliases") or []) if set_default else [],
                "speaker_emb": np.asarray(emb, dtype=np.float32) if emb is not None else None,
                # Voices cloned with an older version may carry a trailing padding frame
                # (stray sound in every read): strip_encoder_pad_frame removes it on load.
                "codes": strip_encoder_pad_frame(np.asarray(codes, dtype=np.int64)) if codes is not None else None,
            }
            if set_default:
                self._register_aliases(name, v.get("aliases"))
        if set_default:
            self._default_voice = data.get("default_voice")
        logger.info(f"📢 Loaded {len(data.get('presets', {}))} voices from {path}")

    def load_extra_voices(self, path: Union[str, Path]) -> int:
        """Merge additional presets from ``path`` (same schema as the built-in file),
        without touching ``default_voice``. Used to restore custom voices the user
        saved in a previous session (see ``add_voice(..., save=True, save_path=...)``).
        Returns the number of newly loaded voices; 0 if the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            return 0
        before = len(self._preset_voices)
        self._load_v3_voices(path, set_default=False)
        return len(self._preset_voices) - before

    def _load_repo_voices(self, backbone_repo: Optional[str]) -> None:
        """Voices shipped WITH a model (fine-tunes): ``voices_v3_turbo.json`` at the root of
        the model folder / Hub repo, same layout as the built-in file. They are added on
        top of the built-ins (same name = override) and the file's ``default_voice`` wins.
        Missing file = nothing happens."""
        if not backbone_repo:
            return
        import json
        path = None
        local = Path(backbone_repo)
        if local.is_dir():
            if (local / "voices_v3_turbo.json").is_file():
                path = local / "voices_v3_turbo.json"
        else:
            # Cache-only read: the main model repo does not have this file (404), so
            # hitting the network on every startup would cause useless timeouts offline.
            try:
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(backbone_repo, "voices_v3_turbo.json", local_files_only=True))
            except Exception:
                path = None
        if path is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Ignoring unreadable voices file {path}: {e}")
            return
        n = 0
        for name, v in data.get("presets", {}).items():
            emb, codes = v.get("speaker_emb"), v.get("codes")
            if emb is None:
                continue
            self._preset_voices[name] = {
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "featured": _featured_rank(v),
                "aliases": list(v.get("aliases") or []),
                "speaker_emb": np.asarray(emb, dtype=np.float32),
                "codes": strip_encoder_pad_frame(np.asarray(codes, dtype=np.int64)) if codes is not None else None,
            }
            self._register_aliases(name, v.get("aliases"))
            n += 1
        if data.get("default_voice") in self._preset_voices:
            self._default_voice = data["default_voice"]
        if n:
            logger.info("📢 Loaded %d extra voice(s) shipped with the model.", n)

    def list_preset_voices(self) -> List[tuple]:
        """Return ``[(label, voice_id), ...]`` for the built-in voices.

        Editors' picks (``featured`` 1..N in the voices JSON) come first in that
        order with a ⭐ prefix, then the remaining voices in file order.
        """
        return [(voice_label(n, v), n) for n, v in sorted_voices(self._preset_voices)]

    def _register_aliases(self, name: str, aliases) -> None:
        for a in aliases or []:
            if a and a != name:
                self._voice_aliases[str(a)] = name

    def resolve_voice_name(self, name: Optional[str]) -> Optional[str]:
        """Canonical preset name for ``name`` (itself, or the preset an alias points
        to); ``None`` if unknown. A real preset always wins over an alias."""
        if name is None:
            return None
        if name in self._preset_voices:
            return name
        target = self._voice_aliases.get(name)
        return target if target in self._preset_voices else None

    def get_preset_voice(self, voice_name: Optional[str] = None) -> dict:
        name = self.resolve_voice_name(voice_name or self._default_voice)
        if name is None:
            raise ValueError(f"Voice '{voice_name or self._default_voice}' not found. "
                             f"Available: {list(self._preset_voices)}")
        return self._preset_voices[name]

    REF_CACHE_MAX = 32   # distinct clips kept (LRU); each entry is a few hundred KB

    def _enroll_reference(self, ref_audio: Union[str, Path], denoise: bool, use_ref_codes: bool
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Trim → engine.prepare_reference, memoised on the clip's content."""
        import hashlib
        import os
        key = None
        try:
            # blake2b: fast content fingerprint (not a security boundary, but keeps
            # the weak-hash linter quiet); 16-byte digest is plenty for a cache key.
            digest = hashlib.blake2b(Path(ref_audio).read_bytes(), digest_size=16).hexdigest()
            key = (digest, bool(denoise), bool(use_ref_codes))
        except OSError:
            pass   # unreadable path: let the engine raise its own error below
        if key is not None and key in self._ref_cache:
            self._ref_cache.move_to_end(key)
            return self._ref_cache[key]
        clean_ref = self._preclean_reference_audio(ref_audio)
        try:
            out = self.engine.prepare_reference(str(clean_ref), denoise=denoise, use_ref_codes=use_ref_codes)
        finally:
            if clean_ref and Path(clean_ref).resolve() != Path(ref_audio).resolve():
                try:
                    os.remove(clean_ref)
                except Exception:
                    pass
        if key is not None:
            self._ref_cache[key] = out
            while len(self._ref_cache) > self.REF_CACHE_MAX:
                self._ref_cache.popitem(last=False)
        return out

    def encode_reference(self, ref_audio: Union[str, Path], denoise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Enroll a voice from a wav → ``(speaker_emb, ref_codes)``.
        """
        return self._enroll_reference(ref_audio, denoise=denoise, use_ref_codes=True)

    def denoise(self, ref_audio: Union[str, Path], out_path: Optional[Union[str, Path]] = None,
                max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Clean up a reference clip and return ``(wav, sample_rate)`` at 44.1 kHz.

        Pass ``out_path`` to also save the cleaned wav. ``max_seconds`` optionally
        trims the clip first. Use the result as a nicer reference for cloning.
        """
        import soundfile as sf
        den = getattr(self.engine, "denoiser", None)
        if den is None:
            raise RuntimeError("Denoiser is not available on this backend.")
        wav, sr = sf.read(str(ref_audio))
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        wav = np.asarray(wav, dtype=np.float32)
        if max_seconds and len(wav) > int(max_seconds * sr):
            wav = wav[: int(max_seconds * sr)]
        clean = den.denoise(wav, sr)          # -> float32 mono @ 44100
        if out_path is not None:
            sf.write(str(out_path), clean, 44100)
        return clean, 44100

    def add_voice(self, name: str, ref_audio: Union[str, Path], *, denoise: bool = True,
                  use_ref_codes: bool = True, description: str = "", gender: str = "",
                  style: Any = None, save: bool = False,
                  save_path: Optional[Union[str, Path]] = None,
                  authorization: Optional[dict] = None) -> str:
        """Register a custom voice under ``name`` for use as ``infer(..., voice=name)``.

        ``ref_audio`` is enrolled once (denoised + trimmed, then speaker embedding +
        reference codes are extracted). Pass ``denoise=False`` if the clip is already
        clean. Set ``save=True`` to persist it to the voices file for later sessions —
        pass ``save_path`` to write to a separate file instead of the built-in one
        (e.g. a writable location outside the installed package).

        ``style`` is deprecated: it is stored as metadata only and never changes how
        the voice is synthesized (the style is implied by the reference itself).
        """
        if not name or not str(name).strip():
            raise ValueError("Voice name must not be empty.")
        speaker_emb, ref_codes = self._enroll_reference(
            ref_audio, denoise=denoise, use_ref_codes=use_ref_codes)
        previous_voice = self._preset_voices.get(name)
        previous_default = self._default_voice
        self._preset_voices[name] = {
            "is_custom": True,
            "authorization": authorization,
            "description": description,
            "gender": gender,
            "style": style or self.default_style,   # metadata only (deprecated)
            "speaker_emb": np.asarray(speaker_emb, dtype=np.float32),
            "codes": None if ref_codes is None else np.asarray(ref_codes, dtype=np.int64),
        }
        if not self._default_voice:
            self._default_voice = name
        if save:
            try:
                self.save_voices(save_path, custom_only=save_path is not None)
            except Exception:
                if previous_voice is None:
                    self._preset_voices.pop(name, None)
                else:
                    self._preset_voices[name] = previous_voice
                self._default_voice = previous_default
                raise
        logger.info(f"➕ Added voice '{name}'.")
        return name

    def remove_voice(self, name: str, save: bool = False, save_path: Optional[Union[str, Path]] = None) -> None:
        """Remove a registered voice by name."""
        name = self.resolve_voice_name(name) or name
        previous_voice = self._preset_voices.pop(name, None)
        previous_aliases = self._voice_aliases
        previous_default = self._default_voice
        self._voice_aliases = {a: t for a, t in self._voice_aliases.items() if t != name}
        if self._default_voice == name:
            self._default_voice = next(iter(self._preset_voices), None)
        if save:
            try:
                self.save_voices(save_path, custom_only=save_path is not None)
            except Exception:
                if previous_voice is not None:
                    self._preset_voices[name] = previous_voice
                self._voice_aliases = previous_aliases
                self._default_voice = previous_default
                raise

    def save_voices(self, path: Optional[Union[str, Path]] = None, *, custom_only: bool = False) -> str:
        """Persist the current voices (speaker embedding + codes) to a JSON file."""
        import json
        import os
        import tempfile
        path = Path(path) if path else (Path(__file__).parent / "assets" / "voices_v3_turbo.json")
        presets = {}
        for n, v in self._preset_voices.items():
            if custom_only and not v.get("is_custom", False):
                continue
            emb = v.get("speaker_emb")
            codes = v.get("codes")
            presets[n] = {
                "is_custom": v.get("is_custom", False),
                "description": v.get("description", ""),
                "gender": v.get("gender", ""),
                "style": v.get("style", self.default_style),
                "speaker_emb": [round(float(x), 6) for x in np.asarray(emb).reshape(-1)] if emb is not None else None,
                "codes": np.asarray(codes, dtype=int).tolist() if codes is not None else None,
            }
            if v.get("featured") is not None:
                presets[n]["featured"] = v["featured"]
            if v.get("authorization") is not None:
                presets[n]["authorization"] = v["authorization"]
            aliases = [a for a, t in self._voice_aliases.items() if t == n]
            if aliases:
                presets[n]["aliases"] = aliases
        data = {"meta": {"note": "v3 turbo voices: speaker embedding + reference codes; "
                                 "`featured` 1..N marks the editors' picks in display order"},
                "default_voice": self._default_voice, "presets": presets}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Replace only after the complete file is written, preserving existing
        # saved voices if serialization or disk writes fail.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(data, temporary, ensure_ascii=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        logger.info(f"💾 Saved {len(presets)} voices → {path}")
        return str(path)

    def _resolve_ref(self, voice, ref_audio, denoise, use_ref_codes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Resolve the requested voice to ``(speaker_emb, ref_codes)``.

        Precedence: cloned ``ref_audio`` → preset ``voice`` (name or dict) → default preset.
        """
        if ref_audio is not None:
            return self._enroll_reference(ref_audio, denoise=denoise, use_ref_codes=use_ref_codes)
        preset = None
        if isinstance(voice, str):
            name = self.resolve_voice_name(voice)
            preset = self._preset_voices.get(name) if name else None
            if preset is None:
                raise ValueError(f"Voice '{voice}' not found. Available: {list(self._preset_voices)}")
        elif isinstance(voice, dict):
            preset = voice
        elif self._default_voice:
            preset = self._preset_voices[self._default_voice]
        if preset is None:
            raise ValueError("Provide a preset `voice` name or a `ref_audio` to clone.")
        codes = preset.get("codes") if use_ref_codes else None
        return preset["speaker_emb"], codes

    # ── Batched inference (GPU/PyTorch) ──────────────────────────────────────
    def _get_batch_engine(self):
        """Lazily build the static-batching engine. Returns ``None`` on CPU/ONNX.

        The batching runtime (``vieneu.v3_turbo_serve``) is pure PyTorch/CUDA, so it
        only applies to the ``pytorch`` backend. Importing it is deferred until the
        first batched call to keep the torch-free CPU path from touching torch.
        """
        if self.backend != "pytorch":
            return None
        if self._batch_engine is None:
            from .v3_turbo_serve import V3TurboBatchEngine
            self._batch_engine = V3TurboBatchEngine(self.engine)
            self._batch_engine.babble_retries = self.babble_retries
        return self._batch_engine

    def _fused_available(self) -> bool:
        """The batch engine runs frames as CUDA graphs here (see fused.py)."""
        eng = self._get_batch_engine()
        return bool(eng is not None and getattr(eng, "use_fused", False)
                    and getattr(self.engine.device, "type", "") == "cuda")

    def _get_stream_scheduler(self):
        """Lazily build the continuous-batching stream scheduler (CUDA only).

        Returns ``None`` on CPU/ONNX or when the fused graphs are disabled
        (``VIENEU_FUSED_FRAME=0``); ``infer_stream`` then takes the engine's
        own single-sequence path.
        """
        if not self._fused_available():
            return None
        with self._stream_lock:
            if self._stream_sched is None:
                from .v3_turbo_serve.stream import V3TurboStreamScheduler
                self._stream_sched = V3TurboStreamScheduler(
                    self._get_batch_engine(), max_streams=self.max_streams,
                )
                logger.info(f"✅ v3 Turbo stream scheduler ready (max_streams={self.max_streams})")
        return self._stream_sched

    def _infer_chunks(
        self,
        chunks: List[str],
        speaker_emb: np.ndarray,
        ref_codes: Optional[np.ndarray],
        use_ref_codes: bool,
        batch_size: int,
        sampling: dict,
    ) -> List[np.ndarray]:
        """Synthesize one waveform per chunk (no join, no watermark).

        On the PyTorch/GPU backend with ``batch_size > 1`` and more than one chunk,
        the chunks are pushed through the static-batching engine in groups of
        ``batch_size`` (chunks share each forward step — the throughput win). Groups
        are formed over chunks SORTED by phoneme length (length bucketing): batching
        similar-length prompts together minimizes left-padding, which cuts wasted
        prefill compute and padding-induced numeric noise. On CPU (ONNX) or a single
        chunk, each chunk goes through the single-sequence engine sequentially.
        Output order always matches ``chunks``.
        """
        n = len(chunks)
        engine = self._get_batch_engine() if (batch_size > 1 and n > 1) else None
        # A single chunk (or batch_size=1) used to take the single-sequence
        # engine. On CUDA the batch engine's fused CUDA graph beats it even
        # at B=1 — measured 2.3 s → 0.38 s for one sentence — so every chunk
        # goes through the batch engine there; groups of ``batch_size`` still
        # decide how many share a forward.
        if engine is None and self._fused_available():
            engine = self._get_batch_engine()
        phs = [phonemize_text_with_emotions(c) for c in chunks]

        def _one(ph: str) -> np.ndarray:
            return self.engine.infer(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                use_ref_codes=use_ref_codes,
                **_cap_frames(sampling, max_expected_frames(ph)),
            )

        if engine is None:
            wavs: List[np.ndarray] = [_one(ph) for ph in phs]
            return wavs

        order = sorted(range(n), key=lambda i: len(phs[i]))
        wavs = [None] * n

        def _run_group(idxs):
            reqs = [{
                "phonemes": phs[j],
                "speaker_emb": speaker_emb, "ref_codes": ref_codes,
                "use_ref_codes": use_ref_codes,
            } for j in idxs]
            # Shared cap for the whole group = cap of the longest row — the group has
            # already been bucketed by phoneme length so the cap stays close for each row.
            group_cap = max(max_expected_frames(phs[j]) for j in idxs)
            try:
                out = engine.generate_batch(reqs, **_cap_frames(sampling, group_cap))
            except Exception as e:
                # Out of VRAM: drop all held graphs (returning the memory pool), split the
                # group in half, and retry — slower but without breaking the whole job. A
                # single-sentence group that still fails raises the real error.
                if not _is_cuda_oom(e):
                    raise
                engine.trim_fused(0)
                if len(idxs) == 1:
                    raise
                logger.warning(f"⚠️ Out of VRAM for a batch of {len(idxs)} sentences - clearing memory and retrying with a smaller batch")
                half = len(idxs) // 2
                _run_group(idxs[:half])
                _run_group(idxs[half:])
                return
            for j, w in zip(idxs, out):
                wavs[j] = w

        for i in range(0, n, batch_size):
            _run_group(order[i:i + batch_size])
        return wavs

    def infer(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: ignored — style is already in the ref code (always natural)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,   # GPU: chunk/forward cap (None → self.max_batch_size; 1 → disable batching)
        **kwargs: Any,
    ) -> np.ndarray:
        """Synthesize ``text`` into one 48 kHz waveform.

        ``style`` is deprecated and ignored (kept only so older code keeps running):
        the reading style comes from the reference voice, so output is always the
        natural style.
        """
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)

        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        if not chunks:
            return np.array([], dtype=np.float32)

        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )
        # GPU merges chunks into the same forward pass; CPU/1-chunk runs sequentially (see _infer_chunks).
        all_wavs = self._infer_chunks(
            chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Silence by boundary type: paragraph break > sentence end > mid-sentence break.
        final_wav = join_audio_chunks(
            all_wavs, self.sample_rate, silence_ps=gaps_to_silence(gaps)
        )
        return self._apply_watermark(final_wav) if apply_watermark else final_wav

    def infer_stream(
        self,
        text: str,
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: ignored (see `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        **kwargs: Any,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize ``text`` and yield 48 kHz float32 audio as it is generated.

        On CUDA every call goes through one shared scheduler (see
        ``v3_turbo_serve/stream.py``): concurrent calls from different threads
        share a CUDA graph and one codec session, so a server can stream to
        many listeners at once — first audio in ~110-135 ms with up to 8
        streams on an RTX 3060. On CPU/ONNX the engine's own frame-level
        streaming runs, one call at a time.
        """
        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=max_chars)
        pauses = gaps_to_silence(gaps)
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
        )
        sched = self._get_stream_scheduler()
        if sched is not None:
            per_chunk = self._stream_chunks_gpu(sched, chunks, speaker_emb, ref_codes,
                                                use_ref_codes, max_new_frames, sampling)
        else:
            per_chunk = self._stream_chunks_engine(chunks, speaker_emb, ref_codes,
                                                   use_ref_codes, max_new_frames, sampling)
        sr = self.sample_rate
        last_out: Optional[np.ndarray] = None   # last emitted audio segment of the previous chunk
        for ci, subs in enumerate(per_chunk):
            first = True
            for sub in subs:
                if sub is None or len(sub) == 0:
                    continue
                if first and last_out is not None:
                    # Pause between two text chunks = TOTAL by boundary type, same
                    # table as join_audio_chunks. The previous audio chunk was already
                    # emitted so it cannot be trimmed; pad with zeros only as needed
                    # (if the model's tail naturally runs longer, keep it).
                    pad = pause_pad_samples(last_out, sub, sr, pauses[ci - 1])
                    if pad > 0:
                        yield np.zeros(pad, dtype=np.float32)
                first = False
                last_out = sub
                yield self._apply_watermark(sub) if apply_watermark else sub

    def _stream_chunks_engine(self, chunks, speaker_emb, ref_codes, use_ref_codes,
                              max_new_frames, sampling):
        """One audio iterator per text chunk from the engine's own streaming
        (PyTorch single path or ONNX); a full ``infer`` per chunk if it has none."""
        stream_fn = getattr(self.engine, "infer_stream", None)
        for chunk in chunks:
            ph = phonemize_text_with_emotions(chunk)
            gen_kwargs = dict(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                use_ref_codes=use_ref_codes,
                max_new_frames=min(max_new_frames, max_expected_frames(ph)), **sampling,
            )
            yield stream_fn(**gen_kwargs) if stream_fn is not None else (self.engine.infer(**gen_kwargs),)

    def _stream_chunks_gpu(self, sched, chunks, speaker_emb, ref_codes, use_ref_codes,
                           max_new_frames, sampling):
        """One audio iterator per text chunk through the stream scheduler.

        Chunk ``i+1`` is submitted the moment chunk ``i``'s last frame is
        generated (its tail audio is still being decoded), so one call holds
        one scheduler slot and the join between chunks costs a prefill, not a
        whole chunk's wait. Abandoning the iterator cancels whatever is queued.
        """
        def submit(ci):
            ph = phonemize_text_with_emotions(chunks[ci])
            return sched.submit(
                phonemes=ph, speaker_emb=speaker_emb, ref_codes=ref_codes,
                use_ref_codes=use_ref_codes,
                max_new_frames=min(max_new_frames, max_expected_frames(ph)), **sampling,
            )

        state = {"next": None}

        def audio_of(handle, ci):
            nxt = None
            try:
                for sub in handle:
                    if nxt is None and ci + 1 < len(chunks) and handle.gen_done.is_set():
                        nxt = submit(ci + 1)
                    yield sub
                if nxt is None and ci + 1 < len(chunks):
                    nxt = submit(ci + 1)
            finally:
                handle.close()
                state["next"] = nxt

        if not chunks:
            return
        handle = submit(0)
        try:
            for ci in range(len(chunks)):
                state["next"] = None
                yield audio_of(handle, ci)
                handle = state["next"]
                if handle is None and ci + 1 < len(chunks):
                    handle = submit(ci + 1)
        finally:
            for h in (handle, state["next"]):
                if h is not None:
                    h.close()

    def infer_batch(
        self,
        texts: List[str],
        ref_audio: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, dict]] = None,
        style: Any = None,   # DEPRECATED: ignored (see `infer`)
        denoise: bool = True,
        use_ref_codes: bool = True,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_chars: int = 256,
        apply_watermark: bool = True,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        """Synthesize many texts, returning one waveform each (same order as ``texts``).

        All texts share one voice (resolved once). On the PyTorch/GPU backend the
        chunks from *every* text are flattened and batched together in groups of
        ``batch_size`` — so even many short texts fill the batch and share forward steps.
        On CPU (ONNX) this runs sequentially. Output equals per-text ``infer`` (equivalent,
        not bit-identical to the single path, since sampling runs batched).
        """
        if not texts:
            return []

        speaker_emb, ref_codes = self._resolve_ref(voice, ref_audio, denoise, use_ref_codes)
        bs = self.max_batch_size if batch_size is None else max(1, int(batch_size))
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_frames=max_new_frames, repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
        )

        # Chunk each text, track which text each chunk belongs to (owner), and keep gaps for joining.
        per_text_gaps: List[Any] = []
        flat_chunks: List[str] = []
        owner: List[int] = []
        for ti, t in enumerate(texts):
            chunks, gaps = normalize_to_chunks_v3_with_gaps(t, max_chars=max_chars)
            per_text_gaps.append(gaps)
            for c in chunks:
                flat_chunks.append(c)
                owner.append(ti)

        empty = np.array([], dtype=np.float32)
        if not flat_chunks:
            return [empty for _ in texts]

        flat_wavs = self._infer_chunks(
            flat_chunks, speaker_emb, ref_codes, use_ref_codes, bs, sampling
        )

        # Group wavs back to the correct text (flat_wavs order matches flat_chunks/owner), then join each text.
        grouped: List[List[np.ndarray]] = [[] for _ in texts]
        for w, ti in zip(flat_wavs, owner):
            grouped[ti].append(w)

        results: List[np.ndarray] = []
        for ti in range(len(texts)):
            if not grouped[ti]:
                results.append(empty)
                continue
            joined = join_audio_chunks(
                grouped[ti], self.sample_rate, silence_ps=gaps_to_silence(per_text_gaps[ti])
            )
            results.append(self._apply_watermark(joined) if apply_watermark else joined)
        return results

    def close(self) -> None:
        if self._stream_sched is not None:
            self._stream_sched.close()
            self._stream_sched = None
        self.engine = None
        self._batch_engine = None
