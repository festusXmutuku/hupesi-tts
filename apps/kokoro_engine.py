"""Multilingual TTS engine using Kokoro-82M (ONNX, CPU) for web_stream.

Does not inherit from `BaseVieneuTTS`: that class is tied to VieNeu's codec and voice format.
This engine only provides the parts used by `web_stream.py` - `infer`, `infer_stream`,
`list_preset_voices`, `_preset_voices`, `sample_rate` - and intentionally does NOT provide
`add_voice` / `load_extra_voices`: Kokoro does not support voice cloning, and web_stream uses the
absence of those attributes to reject cloning requests clearly.

One model, 8 languages (no Vietnamese). For web_stream, Kokoro is a multilingual model like
Supertonic: `languages` lists the readable language codes (/health). Unlike Supertonic, each Kokoro
voice is trained for ONE language (the first letter of the voice code, per Kokoro convention) and
always reads in that language - `infer(lang=...)` is accepted but ignored (see `_lang_for`).
English/Spanish/French/Italian/Portuguese/Hindi are phonemized with espeak-ng (via `kokoro_onnx`);
Japanese and Chinese are phonemized poorly by espeak, so `misaki` is used instead - the same
phonemizer Kokoro was trained with.

`kokoro_onnx` and `misaki` are only available after `apps/kokoro_installer.py` installs them (not
bundled), so imports happen inside functions.
"""
from __future__ import annotations

import re
import sys
import threading
import types
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Union

import numpy as np

from vieneu_utils.subtitles import NON_LATIN_STOP_PATTERN

SAMPLE_RATE = 24_000


class Lang(NamedTuple):
    code: str              # code sent to the UI ("en", "ja", ...)
    espeak: Optional[str]  # None = phonemize with misaki
    name: str              # display language name
    region: str            # display region ("" if none)
    suffix: str            # voice name suffix to avoid collisions across languages


# First letter of voice code -> language (VOICES.md, hexgrad/Kokoro-82M).
LANGS = {
    "a": Lang("en", "en-us", "English", "US", ""),
    "b": Lang("en", "en-gb", "English", "UK", ""),
    "e": Lang("es", "es", "Spanish", "", "ES"),
    "f": Lang("fr", "fr-fr", "French", "", "FR"),
    "h": Lang("hi", "hi", "Hindi", "", "HI"),
    "i": Lang("it", "it", "Italian", "", "IT"),
    "p": Lang("pt", "pt-br", "Portuguese", "Brazil", "PT"),
    "j": Lang("ja", None, "Japanese", "", "JA"),
    "z": Lang("zh", None, "Chinese", "", "ZH"),
}
_GENDER = {"f": ("female", "Female"), "m": ("male", "Male")}
# Language code -> phonemization settings when text is written in a script different from the
# voice's language (`_lang_for`). "en" uses the US voice entry (first record).
_BY_CODE: dict = {}
for _lang in LANGS.values():
    _BY_CODE.setdefault(_lang.code, _lang)
# Display order in the "Text language" selector, with display names.
LANGUAGES = {code: _BY_CODE[code].name for code in ("en", "es", "fr", "it", "pt", "hi", "ja", "zh")}
# Voice preview samples (play button in the voice list): shared with Supertonic, plus Chinese
# (Supertonic does not support Chinese).
PREVIEW_TEXTS = {
    "en": "Hello, this is a sample voice from Hupesi TTS. Have a wonderful day.",
    "es": "Hola, esta es una voz de muestra de Hupesi TTS. Que tengas un d\u00eda maravilloso.",
    "fr": "Bonjour, ceci est une voix d'exemple de Hupesi TTS. Passez une excellente journ\u00e9e.",
    "it": "Ciao, questa \u00e8 una voce di esempio di Hupesi TTS. Buona giornata.",
    "pt": "Ol\u00e1, esta \u00e9 uma voz de exemplo do Hupesi TTS. Tenha um \u00f3timo dia.",
    "hi": "\u0928\u092e\u0938\u094d\u0924\u0947, \u092f\u0939 Hupesi TTS \u0915\u0940 \u090f\u0915 \u0928\u092e\u0942\u0928\u093e \u0906\u0935\u093e\u091c\u093c \u0939\u0948\u0964 \u0906\u092a\u0915\u093e \u0926\u093f\u0928 \u0936\u0941\u092d \u0939\u094b\u0964",
    "ja": "\u3053\u3093\u306b\u3061\u306f\u3002Hupesi TTS\u306e\u30b5\u30f3\u30d7\u30eb\u97f3\u58f0\u3067\u3059\u3002\u7d20\u6575\u306a\u4e00\u65e5\u3092\u304a\u904e\u3054\u3057\u304f\u3060\u3055\u3044\u3002",
    "zh": "\u4f60\u597d\uff0c\u8fd9\u662fHupesi TTS\u7684\u793a\u4f8b\u58f0\u97f3\u3002\u795d\u4f60\u4eca\u5929\u8fc7\u5f97\u6109\u5feb\u3002",
}

# Best voices according to the authors' benchmark come first; the rest follow language order in LANGS, then name.
_ORDER = (
    "af_heart", "af_bella", "af_nicole", "bf_emma", "af_aoede", "af_kore",
    "af_sarah", "am_fenrir", "am_michael", "am_puck", "bm_george", "bm_fable",
)
_LANG_ORDER = "abefhipjz"

# Emotion tags ([cuoi], [tho dai]... from VieNeu; <laugh> <breath> <sigh> from Supertonic, found in
# foreign-language sample snippets): Kokoro does not understand them and espeak would read the inner text
# aloud. Pause tags are already split out by web_stream before reaching here.
_TAG_RE = re.compile(r"\[[^\]\n]*\]|<(?:laugh|breath|sigh)>", re.IGNORECASE)
# Scripts specific to Japanese/Chinese/Hindi (`_lang_for`).
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
# Japanese/Chinese do not use spaces after sentence punctuation; Hindi ends sentences with "।".
# Closing punctuation after sentence-ending marks (「…。」) stays with that sentence - same pattern as subtitles.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|" + NON_LATIN_STOP_PATTERN)
# espeak inserts language-switch flags around loanwords, e.g. "(en)vˈɔɪs(fr)"; the model would read them as noise.
_LANG_FLAG_RE = re.compile(r"\([a-z]{2,3}(?:-[a-z]+)?\)")
# Latin segments inside Japanese/Chinese sentences ("CIT", "AI"): read with English espeak.
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*(?:\s+[A-Za-z][A-Za-z0-9'\-]*)*")


def _create_session(model_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _torch_kokoro(config_path: Path, weights_path: Path, voices_npz: Path, device: str):
    """`kokoro_onnx.Kokoro` with the inference step running in PyTorch (GPU).

    The onnxruntime bundled in the app is CPU-only, and the GPU build shares the same module name so it
    cannot be loaded in parallel. This replaces only `_infer` - phonemization, batching into ≤510 phonemes,
    silence trimming, and punctuation pauses still come from kokoro_onnx (pinned to 0.6.1), so both paths
    produce the same reading style. Does not call `Kokoro.__init__`/`_setup`: those require an ONNX session.
    """
    import torch
    from kokoro_onnx import Kokoro
    from kokoro_onnx.tokenizer import Tokenizer

    from apps._kokoro_torch.model import KModel

    class _TorchKokoro(Kokoro):
        def __init__(self):
            self.model = KModel(str(config_path), str(weights_path)).to(device).eval()
            self.voices = np.load(voices_npz)
            self.tokenizer = Tokenizer(vocab=self.model.vocab)
            vocab = self.tokenizer.vocab
            self._stops = frozenset(vocab[m] for m in ".!?" if m in vocab)
            self._spaces = frozenset(vocab[m] for m in " " if m in vocab)
            self.has_timings = True

        def _infer(self, tokens, style, speed):
            ids = torch.tensor([[0, *tokens, 0]], dtype=torch.long, device=device)
            ref = torch.from_numpy(np.asarray(style, dtype=np.float32).reshape(1, -1)).to(device)
            audio, dur = self.model.forward_with_tokens(ids, ref, float(speed))
            return audio.float().cpu().numpy().ravel(), dur.cpu().numpy().ravel()

    return _TorchKokoro()


def _is_voice_key(key: str) -> bool:
    return len(key) > 3 and key[0] in LANGS and key[1] in _GENDER and key[2] == "_"


def _display_name(key: str) -> str:
    name = key.split("_", 1)[1].capitalize()
    suffix = LANGS[key[0]].suffix
    return f"{name} ({suffix})" if suffix else name


def _sort_key(key: str):
    if key in _ORDER:
        return (0, _ORDER.index(key), key)
    return (1, _LANG_ORDER.index(key[0]), key)


def _ensure_pyopenjtalk_importable() -> None:
    """`misaki.ja` imports `pyopenjtalk` at module top level even though the default (cutlet) branch
    does not use it. The original package has no Windows wheel (requires building with cmake), so if
    missing we install an empty module stub - accidentally hitting the pyopenjtalk branch will raise a
    clear AttributeError."""
    try:
        import pyopenjtalk  # noqa: F401
    except ImportError:
        sys.modules.setdefault("pyopenjtalk", types.ModuleType("pyopenjtalk"))


def _voices_npz(voices_dir: Path, cache_dir: Path, keys: list) -> Path:
    """Combined .npz file containing voices in `keys`, named after that exact list.

    A fixed name is not used: kokoro_onnx keeps the .npz file OPEN for the engine's lifetime, so on
    Windows reloading Kokoro with a new voice set (e.g. after installing the Japanese pack) while
    overwriting the old file would fail with `[WinError 5] Access is denied`. Using different names
    avoids touching the open file; old voice-set files are cleaned up when no longer referenced.
    """
    import hashlib

    tag = hashlib.sha1(",".join(keys).encode()).hexdigest()[:12]
    npz = cache_dir / f"kokoro-voices-{tag}.npz"
    if not npz.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.fromfile(voices_dir / f"{k}.bin", dtype=np.float32).reshape(-1, 1, 256)
                  for k in keys}
        tmp = npz.with_name(npz.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(npz)
    for old in cache_dir.glob("kokoro-voices*.npz"):
        if old != npz:
            try:
                old.unlink()
            except OSError:
                pass  # engine still holds it: will be cleaned up on the next load
    return npz


class KokoroTTS:
    """Kokoro-82M fp32 via `kokoro_onnx`. `voices_dir` contains `<code>.bin` files (float32 510x1x256)."""

    sample_rate = SAMPLE_RATE

    def __init__(self, model_path: Union[str, Path], voices_dir: Union[str, Path],
                 cache_dir: Union[str, Path], languages: Optional[set] = None,
                 torch_config: Union[str, Path, None] = None, device: str = "cuda"):
        """`languages`: only show voices for these languages - languages with the phonemizer installed
        (Japanese/Chinese are optional add-on packs). None = all voices with available files.

        `torch_config` (config.json from hexgrad/Kokoro-82M): run with PyTorch on `device`, in which case
        `model_path` is `kokoro-v1_0.pth`. If None: run with ONNX on CPU."""
        from kokoro_onnx import Kokoro

        keys = sorted((p.stem for p in Path(voices_dir).glob("*.bin")
                       if _is_voice_key(p.stem)
                       and (languages is None or LANGS[p.stem[0]].code in languages)),
                      key=_sort_key)
        if not keys:
            raise RuntimeError("No Kokoro voices were found. Please download the model again.")
        self._keys = {_display_name(k): k for k in keys}
        self._preset_voices = {}
        for name, key in self._keys.items():
            lang = LANGS[key[0]]
            gioi, gioi_vi = _GENDER[key[1]]
            tieng = f"{lang.name} ({lang.region})" if lang.region else lang.name
            # "Gender · Kokoro · Language": same "Gender · Region · Style" layout that /voices and the UI
            # split on "·" (part 3 is shown as the label).
            self._preset_voices[name] = {
                "gender": gioi,
                "language": lang.code,
                "description": f"{gioi_vi} \u00b7 Kokoro \u00b7 {tieng}",
            }
        self._default_voice = next(iter(self._keys))
        # Readable languages = languages of the currently visible voices (filtered by installed phonemizers),
        # in LANGUAGES order. web_stream reports this list in /health.
        have = {LANGS[k[0]].code for k in keys}
        self.languages = {c: n for c, n in LANGUAGES.items() if c in have}
        self._g2p = {}
        self._g2p_lock = threading.Lock()
        # One lock per misaki phonemizer, held for the duration of a call: fugashi's MeCab tagger (and jieba)
        # is not safe for concurrent use by multiple requests.
        self._g2p_call_locks = {code: threading.Lock() for code in ("ja", "zh")}

        # kokoro_onnx reads voices from a SINGLE .npz file; Hugging Face stores each voice as a separate file.
        # Merge once into a writable directory (the HF cache may be read-only).
        npz = _voices_npz(Path(voices_dir), Path(cache_dir), keys)
        if torch_config is not None:
            self.backend, self.device = "pytorch", device
            self.kokoro = _torch_kokoro(Path(torch_config), Path(model_path), npz, device)
        else:
            self.backend, self.device = "onnx", "cpu"
            self.kokoro = Kokoro.from_session(_create_session(Path(model_path)), str(npz))

    # -- voices ----------------------------------------------------------------
    def list_preset_voices(self):
        return [(f"{n} - {v['description']}", n) for n, v in self._preset_voices.items()]

    def resolve_voice_name(self, voice) -> Optional[str]:
        """Display name of the voice (also accepts the original code like "af_heart"), or None if not found.
        web_stream uses this to return 400 for unknown voices before synthesis."""
        if voice in self._keys:
            return voice
        return next((name for name, key in self._keys.items() if key == voice), None)

    def voice_language(self, voice=None) -> str:
        return LANGS[self._resolve(voice)[0]].code

    def _lang_for(self, key: str, text: str) -> Lang:
        """Phonemization settings for a synthesis call: ALWAYS based on the voice's language.

        The request `language` is not used: Conversation/Batch/SRT tabs send a single code for all turns,
        so a Japanese voice in an "English" conversation was previously phonemized via espeak en-us - producing
        meaningless sounds. The only exception is scripts that the voice's language cannot read: text containing
        kana -> Japanese, Han characters -> Chinese, Devanagari -> Hindi (if that language pack is installed).
        A Japanese voice reads Han characters as Japanese (kanji), not as Chinese."""
        own = LANGS[key[0]]
        if _KANA_RE.search(text):
            script = "ja"
        elif _HAN_RE.search(text):
            script = "zh"
        elif _DEVANAGARI_RE.search(text):
            script = "hi"
        else:
            return own
        if script == own.code or (own.code == "ja" and script == "zh"):
            return own
        if script not in self.languages:
            return own  # language pack not installed: reading in the voice's language is better than failing
        return _BY_CODE[script]

    def _resolve(self, voice) -> str:
        if voice is None or voice == "":
            return self._keys[self._default_voice]
        if isinstance(voice, str):
            if voice in self._keys:
                return self._keys[voice]
            if voice in self._keys.values():   # also accept the original code "af_heart" for API callers
                return voice
        raise ValueError(f"Voice '{voice}' is not available in the Kokoro model. "
                         f"Available: {', '.join(self._keys)}")

    # -- phonemization ---------------------------------------------------------
    def _misaki(self, code: str):
        """misaki phonemizer for "ja"/"zh", created once (dictionary loading takes ~1 second)."""
        with self._g2p_lock:
            if code not in self._g2p:
                if code == "ja":
                    _ensure_pyopenjtalk_importable()
                    from misaki import ja

                    self._g2p[code] = ja.JAG2P()
                else:
                    import logging

                    import jieba
                    from misaki import zh

                    jieba.setLogLevel(logging.WARNING)
                    # version=None: the romanization style Kokoro v1.0 was trained with
                    # ("1.1" is for the Kokoro-82M-v1.1-zh model).
                    self._g2p[code] = zh.ZHG2P(version=None)
            return self._g2p[code]

    def _espeak(self, text: str, lang: str) -> str:
        return _LANG_FLAG_RE.sub("", self.kokoro.tokenizer.phonemize(text, lang))

    def _phonemes(self, text: str, lang: Lang) -> str:
        if lang.espeak:
            return self._espeak(text, lang.espeak)
        misaki = self._misaki(lang.code)
        call_lock = self._g2p_call_locks[lang.code]

        def g2p(part: str) -> str:
            with call_lock:
                return misaki(part)[0]

        out, pos = [], 0
        for m in _LATIN_RE.finditer(text):
            if m.start() > pos:
                out.append(g2p(text[pos:m.start()]))
            out.append(self._espeak(m.group(), "en-us"))
            pos = m.end()
        if pos < len(text):
            out.append(g2p(text[pos:]))
        return " ".join(p.strip() for p in out if p and p.strip())

    # -- synthesis -------------------------------------------------------------
    def infer(self, text: str, voice=None, lang: Optional[str] = None, speed: float = 1.0,
              **_ignored) -> np.ndarray:
        """Synthesize `text` using `voice`, in that voice's own language (`_lang_for`).

        `lang` (passed down by web_stream's `_language_kwargs`, like Supertonic) is accepted so API
        clients do not error, but is ignored: to read in a language, choose a voice of that language.
        `temperature` and VieNeu-specific parameters are also ignored."""
        key = self._resolve(voice)
        clean = re.sub(r"\s+", " ", _TAG_RE.sub(" ", text or "")).strip()
        if not clean:
            return np.zeros(0, dtype=np.float32)
        phonemes = self._phonemes(clean, self._lang_for(key, clean))
        if not phonemes.strip():
            return np.zeros(0, dtype=np.float32)
        audio, _sr = self.kokoro.create(phonemes, voice=key, speed=float(speed), is_phonemes=True)
        return np.asarray(audio, dtype=np.float32)

    def infer_stream(self, text: str, voice=None, **kwargs) -> Iterator[np.ndarray]:
        """Kokoro does not stream frames; yield one sentence at a time so playback can start early."""
        for sentence in _SENTENCE_RE.split(text or ""):
            wav = self.infer(sentence, voice=voice, **kwargs)
            if wav.size:
                yield wav

    def close(self) -> None:
        self.kokoro = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
