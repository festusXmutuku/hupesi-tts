"""
Supertonic 3 (Supertone, OpenRAIL-M model): multilingual TTS for 31 languages, running on CPU via ONNX.

Intended for non-Vietnamese voices; Vietnamese should still use VieNeu (Supertonic reads Vietnamese
slowly and does not normalize numbers/abbreviations well). The model reads Unicode characters directly,
so it needs no phonemizer - no espeak, no GPL - and only requires `onnxruntime` + `numpy` which the app
already has.

Inference flow (following Supertone's MIT sample code, `py/helper.py`): normalize text -> character IDs
via `unicode_indexer.json` -> `duration_predictor` estimates duration -> `text_encoder` -> denoise with
`vector_estimator` over `steps` iterations from random noise -> `vocoder` produces audio.
Noise comes from `np.random`, so `web_stream`'s `_tts_infer(seed=...)` can produce deterministic results.

Implements the same engine interface that web_stream uses for VieNeu (`infer`, `infer_stream`, voice
listing, `sample_rate`) and intentionally does NOT provide `add_voice`: Supertonic does not support voice
cloning, so `POST /api/voices/custom` returns 400 automatically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from unicodedata import normalize

import numpy as np

# Languages supported by the model (from the model README). Order = display order in the UI.
LANGUAGES: Dict[str, str] = {
    "en": "English", "vi": "Vietnamese", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "it": "Italian", "ru": "Russian", "id": "Indonesian", "hi": "Hindi",
    "ar": "Arabic", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "uk": "Ukrainian", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "el": "Greek", "cs": "Czech", "sk": "Slovak", "hu": "Hungarian",
    "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian", "sl": "Slovenian",
    "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian",
}
# Voice preview sample text per language (play button in the voice list).
PREVIEW_TEXTS: Dict[str, str] = {
    "en": "Hello, this is a sample voice from Hupesi TTS. Have a wonderful day.",
    "vi": "Xin ch\u00e0o, \u0111\u00e2y l\u00e0 gi\u1ecdng \u0111\u1ecdc m\u1eabu c\u1ee7a Hupesi TTS. Ch\u00fac b\u1ea1n m\u1ed9t ng\u00e0y t\u1ed1t l\u00e0nh.",
    "ja": "\u3053\u3093\u306b\u3061\u306f\u3002Hupesi TTS\u306e\u30b5\u30f3\u30d7\u30eb\u97f3\u58f0\u3067\u3059\u3002\u7d20\u6575\u306a\u4e00\u65e5\u3092\u304a\u904e\u3054\u3057\u304f\u3060\u3055\u3044\u3002",
    "ko": "\uc548\ub155\ud558\uc138\uc694. Hupesi TTS\uc758 \uc0d8\ud50c \uc74c\uc131\uc785\ub2c8\ub2e4. \uc88b\uc740 \ud558\ub8e8 \ubcf4\ub0b4\uc138\uc694.",
    "fr": "Bonjour, ceci est une voix d'exemple de Hupesi TTS. Passez une excellente journ\u00e9e.",
    "de": "Hallo, dies ist eine Beispielstimme von Hupesi TTS. Einen sch\u00f6nen Tag noch.",
    "es": "Hola, esta es una voz de muestra de Hupesi TTS. Que tengas un d\u00eda maravilloso.",
    "pt": "Ol\u00e1, esta \u00e9 uma voz de exemplo do Hupesi TTS. Tenha um \u00f3timo dia.",
    "it": "Ciao, questa \u00e8 una voce di esempio di Hupesi TTS. Buona giornata.",
    "ru": "\u0417\u0434\u0440\u0430\u0432\u0441\u0442\u0432\u0443\u0439\u0442\u0435, \u044d\u0442\u043e \u043f\u0440\u0438\u043c\u0435\u0440 \u0433\u043e\u043b\u043e\u0441\u0430 Hupesi TTS. \u0425\u043e\u0440\u043e\u0448\u0435\u0433\u043e \u0432\u0430\u043c \u0434\u043d\u044f.",
    "id": "Halo, ini adalah contoh suara dari Hupesi TTS. Semoga harimu menyenangkan.",
    "hi": "\u0928\u092e\u0938\u094d\u0924\u0947, \u092f\u0939 Hupesi TTS \u0915\u0940 \u090f\u0915 \u0928\u092e\u0942\u0928\u093e \u0906\u0935\u093e\u091c\u093c \u0939\u0948\u0964 \u0906\u092a\u0915\u093e \u0926\u093f\u0928 \u0936\u0941\u092d \u0939\u094b\u0964",
    "ar": "\u0645\u0631\u062d\u0628\u064b\u0627\u060c \u0647\u0630\u0627 \u0635\u0648\u062a \u062a\u062c\u0631\u064a\u0628\u064a \u0645\u0646 Hupesi TTS. \u0623\u062a\u0645\u0646\u0649 \u0644\u0643 \u064a\u0648\u0645\u064b\u0627 \u0631\u0627\u0626\u0639\u064b\u0627.",
    "tr": "Merhaba, bu Hupesi TTS'dan \u00f6rnek bir ses. \u0130yi g\u00fcnler dilerim.",
    "nl": "Hallo, dit is een voorbeeldstem van Hupesi TTS. Nog een fijne dag.",
    "pl": "Dzie\u0144 dobry, to jest przyk\u0142adowy g\u0142os z Hupesi TTS. Mi\u0142ego dnia.",
    "uk": "\u041f\u0440\u0438\u0432\u0456\u0442, \u0446\u0435 \u0437\u0440\u0430\u0437\u043e\u043a \u0433\u043e\u043b\u043e\u0441\u0443 Hupesi TTS. \u0413\u0430\u0440\u043d\u043e\u0433\u043e \u0432\u0430\u043c \u0434\u043d\u044f.",
    "sv": "Hej, det h\u00e4r \u00e4r en exempelr\u00f6st fr\u00e5n Hupesi TTS. Ha en underbar dag.",
    "da": "Hej, dette er en eksempelstemme fra Hupesi TTS. Hav en dejlig dag.",
    "fi": "Hei, t\u00e4m\u00e4 on Hupesi TTSn esimerkki\u00e4\u00e4ni. Mukavaa p\u00e4iv\u00e4\u00e4.",
    "el": "\u0393\u03b5\u03b9\u03b1 \u03c3\u03b1\u03c2, \u03b1\u03c5\u03c4\u03ae \u03b5\u03af\u03bd\u03b1\u03b9 \u03bc\u03b9\u03b1 \u03b4\u03bf\u03ba\u03b9\u03bc\u03b1\u03c3\u03c4\u03b9\u03ba\u03ae \u03c6\u03c9\u03bd\u03ae \u03c4\u03bf\u03c5 Hupesi TTS. \u039a\u03b1\u03bb\u03ae \u03c3\u03b1\u03c2 \u03bc\u03ad\u03c1\u03b1.",
    "cs": "Dobr\u00fd den, toto je uk\u00e1zkov\u00fd hlas z Hupesi TTS. P\u0159eji hezk\u00fd den.",
    "sk": "Dobr\u00fd de\u0148, toto je uk\u00e1\u017ekov\u00fd hlas z Hupesi TTS. Prajem pekn\u00fd de\u0148.",
    "hu": "Hell\u00f3, ez a Hupesi TTS mintahangja. Sz\u00e9p napot k\u00edv\u00e1nok.",
    "ro": "Bun\u0103 ziua, aceasta este o voce demonstrativ\u0103 de la Hupesi TTS. O zi minunat\u0103.",
    "bg": "\u0417\u0434\u0440\u0430\u0432\u0435\u0439\u0442\u0435, \u0442\u043e\u0432\u0430 \u0435 \u043f\u0440\u0438\u043c\u0435\u0440\u0435\u043d \u0433\u043b\u0430\u0441 \u043e\u0442 Hupesi TTS. \u041f\u0440\u0438\u044f\u0442\u0435\u043d \u0434\u0435\u043d.",
    "hr": "Bok, ovo je ogledni glas iz Hupesi TTS Studija. Ugodan dan.",
    "sl": "Pozdravljeni, to je vzor\u010dni glas iz Hupesi TTS. Lep dan \u017eelim.",
    "et": "Tere, see on Hupesi TTS n\u00e4idish\u00e4\u00e4l. Ilusat p\u00e4eva.",
    "lv": "Sveiki, \u0161\u012b ir Hupesi TTS parauga balss. Jauku dienu.",
    "lt": "Sveiki, tai Hupesi TTS balso pavyzdys. Geros dienos.",
}
VOICES: Tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
DEFAULT_VOICE = "F1"
DEFAULT_STEPS = 8          # number of denoising steps, Supertone default
DEFAULT_SPEED = 1.05       # base speed from Supertone's sample code
CHUNK_SILENCE_S = 0.3      # pause between two chunks of a long text
MODEL_FILES: Tuple[str, ...] = (
    "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx", "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx", "onnx/tts.json", "onnx/unicode_indexer.json",
    *(f"voice_styles/{v}.json" for v in VOICES),
    "LICENSE", "README.md", "config.json",
)

# Supertonic tags are <laugh>, <breath>, <sigh> (inserted directly by the UI, kept as-is). The app's
# legacy bracket tags are also accepted: [cuoi] -> <laugh>, [tho dai] -> <sigh>, [lay hoi] -> <breath>;
# other bracket tags are removed, because Supertonic normalization turns "[" "]" into spaces and would
# read them out loud.
_EMOTION_TAGS = {"c\u01b0\u1eddi": "<laugh>", "th\u1edf d\u00e0i": "<sigh>", "l\u1ea5y h\u01a1i": "<breath>"}
_BRACKET_TAG = re.compile(r"\[([^\]]*)\]")
_VI_LETTERS = re.compile(
    r"[\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0\u00e0\u1ea3\u00e3\u00e1\u1ea1\u1eb1\u1eb3\u1eb5\u1eaf\u1eb7\u1ea7\u1ea9\u1eab\u1ea5\u1ead\u00e8\u1ebb\u1ebd\u00e9\u1eb9\u1ec1\u1ec3\u1ec5\u1ebf\u1ec7\u00ec\u1ec9\u0129\u00ed\u1ecb\u00f2\u1ecf\u00f5\u00f3\u1ecd\u1ed3\u1ed5\u1ed7\u1ed1\u1ed9\u1edd\u1edf\u1ee1\u1edb\u1ee3\u00f9\u1ee7\u0169\u00fa\u1ee5\u1eeb\u1eed\u1eef\u1ee9\u1ef1\u1ef3\u1ef7\u1ef9\u00fd\u1ef5]", re.IGNORECASE)
_KANA = re.compile(r"[\u3040-\u30ff]")
_HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")

_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+")
_REPLACE = {
    "\u2013": "-", "\u2011": "-", "\u2014": "-", "_": " ", "\u201c": '"', "\u201d": '"', "\u2018": "'",
    "\u2019": "'", "\u00b4": "'", "`": "'", "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
    "\u2192": " ", "\u2190": " ",
}
_SENTENCE_END = re.compile(r"[.!?;:,'\"')\]}…。\u300d\u300f\u3011\u3009\u300b\u203a\u00bb]$")
_PARAGRAPHS = re.compile(r"\n\s*\n+")
_SENTENCES = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!etc\.)(?<!e\.g\.)"
    r"(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!St\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
    r"|(?<=[\u3002\uff01\uff1f])")  # CJK sentence punctuation not followed by spaces


def detect_language(text: str) -> str:
    """Guess language when the caller does not specify one: sufficient for common app use cases."""
    if _VI_LETTERS.search(text):
        return "vi"
    if _KANA.search(text):
        return "ja"
    if _HANGUL.search(text):
        return "ko"
    return "en"


def convert_tags(text: str) -> str:
    """Convert the app's emotion tags to Supertonic tags; remove other bracket tags."""
    def repl(m: re.Match) -> str:
        tag = _EMOTION_TAGS.get(m.group(1).strip().lower())
        return f" {tag} " if tag else " "
    return _BRACKET_TAG.sub(repl, text)


def preprocess(text: str, lang: str) -> str:
    """Normalize text like Supertone's sample code, then wrap with language tags ``<lang>...</lang>``."""
    text = _EMOJI.sub("", normalize("NFKD", text))
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for k, v in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
        text = text.replace(k, v)
    text = re.sub(r" ([,.!?;:'])", r"\1", text)
    for q in ('"', "'"):
        while q * 2 in text:
            text = text.replace(q * 2, q)
    text = re.sub(r"\s+", " ", text).strip()
    if not _SENTENCE_END.search(text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def chunk_text(text: str, max_len: int = 300) -> List[str]:
    """Split by paragraphs then sentences, keeping each chunk <= ``max_len`` characters when possible."""
    chunks: List[str] = []
    for para in (p.strip() for p in _PARAGRAPHS.split(text.strip())):
        if not para:
            continue
        cur = ""
        for sent in (s for s in _SENTENCES.split(para) if s.strip()):
            if cur and len(cur) + len(sent) + 1 > max_len:
                chunks.append(cur.strip())
                cur = sent
            else:
                cur = f"{cur} {sent}" if cur else sent
        if cur.strip():
            chunks.append(cur.strip())
    return chunks


def _mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    max_len = int(max_len or lengths.max())
    ids = np.arange(max_len)
    return (ids < lengths[:, None]).astype(np.float32).reshape(-1, 1, max_len)


class SupertonicTTS:
    """Supertonic 3 via onnxruntime. ``model_dir`` is the HF repo snapshot directory."""

    def __init__(self, model_dir, steps: int = DEFAULT_STEPS, sessions=None):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "onnx" / "tts.json").read_text(encoding="utf-8"))
        self.sample_rate = int(cfg["ae"]["sample_rate"])
        self._base_chunk = int(cfg["ae"]["base_chunk_size"])
        self._compress = int(cfg["ttl"]["chunk_compress_factor"])
        self._latent_dim = int(cfg["ttl"]["latent_dim"])
        self.steps = int(steps)
        self._indexer = np.asarray(
            json.loads((model_dir / "onnx" / "unicode_indexer.json").read_text(encoding="utf-8")),
            dtype=np.int64)
        self._styles = {v: self._load_style(model_dir / "voice_styles" / f"{v}.json") for v in VOICES}
        self._dp, self._enc, self._est, self._voc = sessions or self._load_sessions(model_dir / "onnx")
        self._default_voice = DEFAULT_VOICE
        self._preset_voices = {
            v: {"gender": "female" if v[0] == "F" else "male",
                "description": f"{'Female' if v[0] == 'F' else 'Male'} \u00b7 Multilingual \u00b7 Supertonic 3"}
            for v in VOICES
        }

    @staticmethod
    def _load_sessions(onnx_dir: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        return tuple(
            ort.InferenceSession(str(onnx_dir / f"{name}.onnx"), sess_options=opts,
                                 providers=["CPUExecutionProvider"])
            for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"))

    @staticmethod
    def _load_style(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = json.loads(path.read_text(encoding="utf-8"))
        def arr(key):
            dims = data[key]["dims"]
            return np.asarray(data[key]["data"], dtype=np.float32).reshape(1, dims[1], dims[2])
        return arr("style_ttl"), arr("style_dp")

    # --- voice listing (same interface as the VieNeu engine) --------------------
    def list_preset_voices(self) -> List[tuple]:
        return [(f"{'Female' if v[0] == 'F' else 'Male'} {v[1]} ({v})", v) for v in VOICES]

    def resolve_voice_name(self, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        key = str(name).strip().upper()
        return key if key in self._styles else None

    # --- synthesis --------------------------------------------------------------
    def _encode(self, text: str, lang: str) -> np.ndarray:
        codes = np.fromiter((ord(c) for c in preprocess(text, lang)), dtype=np.int64)
        codes = codes[codes < len(self._indexer)]
        ids = self._indexer[codes]
        return ids[ids >= 0]  # characters unknown to the model: drop them instead of breaking the sentence

    def _infer_chunk(self, text: str, lang: str, style, speed: float) -> np.ndarray:
        ids = self._encode(text, lang)
        if len(ids) == 0:
            return np.zeros(0, dtype=np.float32)
        text_ids = ids.reshape(1, -1)
        text_mask = _mask(np.array([len(ids)]))
        ttl, dp = style
        dur, *_ = self._dp.run(None, {"text_ids": text_ids, "style_dp": dp, "text_mask": text_mask})
        dur = np.asarray(dur, dtype=np.float32).reshape(-1) / speed
        emb, *_ = self._enc.run(None, {"text_ids": text_ids, "style_ttl": ttl, "text_mask": text_mask})

        wav_len = (dur * self.sample_rate).astype(np.int64)
        chunk = self._base_chunk * self._compress
        latent_len = (wav_len + chunk - 1) // chunk
        latent_mask = _mask(latent_len)
        xt = np.random.randn(1, self._latent_dim * self._compress, int(latent_len.max())).astype(np.float32)
        xt *= latent_mask
        total = np.array([self.steps], dtype=np.float32)
        for step in range(self.steps):
            xt, *_ = self._est.run(None, {
                "noisy_latent": xt, "text_emb": emb, "style_ttl": ttl, "text_mask": text_mask,
                "latent_mask": latent_mask, "current_step": np.array([step], dtype=np.float32),
                "total_step": total,
            })
        wav, *_ = self._voc.run(None, {"latent": xt})
        return np.asarray(wav, dtype=np.float32).reshape(-1)[: int(wav_len[0])]

    def _chunks(self, text: str, voice: Optional[str], lang: Optional[str], speed: float) -> Iterator[np.ndarray]:
        name = self.resolve_voice_name(voice or self._default_voice)
        if name is None:
            raise ValueError(f"Voice '{voice}' not found")
        text = convert_tags(text)
        lang = lang if lang in LANGUAGES else detect_language(text)
        max_len = 120 if lang in ("ja", "ko") else 300
        for part in chunk_text(text, max_len=max_len):
            wav = self._infer_chunk(part, lang, self._styles[name], speed)
            if len(wav):
                yield wav

    def infer_stream(self, text: str, voice: Optional[str] = None, lang: Optional[str] = None,
                     speed: float = DEFAULT_SPEED, **_) -> Iterator[np.ndarray]:
        """Synthesize each chunk and yield it immediately (no frame-level streaming like VieNeu Turbo)."""
        silence = np.zeros(int(self.sample_rate * CHUNK_SILENCE_S), dtype=np.float32)
        for i, wav in enumerate(self._chunks(text, voice, lang, speed)):
            yield np.concatenate([silence, wav]) if i else wav

    def infer(self, text: str, voice: Optional[str] = None, lang: Optional[str] = None,
              speed: float = DEFAULT_SPEED, **_) -> np.ndarray:
        """Synthesize the full text. ``temperature`` and VieNeu-specific parameters are ignored."""
        parts = list(self.infer_stream(text, voice=voice, lang=lang, speed=speed))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        self._dp = self._enc = self._est = self._voc = None
