import re
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ─── Regex ───────────────────────────────────────────────────────────────────

RE_NEWLINE          = re.compile(r'[\r\n]+')  # shared by both v1 and v2
RE_SENTENCE_FINDALL = re.compile(r'[^.!?]+[.!?]*|[.!?]+')

# A "word" for chunk packing: treat an entire <en>...</en> tag as an unsplittable
# token (tags containing spaces like "<en>u s d</en>" would break if split with
# .split()). Used when splitting ALREADY-normalized text (with <en> inserted) into chunks.
RE_TOKEN_KEEP_EN = re.compile(r'<en>.*?</en>|\S+', re.IGNORECASE | re.DOTALL)


def _tokenize_keep_en(s: str) -> List[str]:
    """Split ``s`` into tokens, keeping each ``<en>...</en>`` span intact."""
    return RE_TOKEN_KEEP_EN.findall(s)

# v1 only
RE_SENTENCE_END = re.compile(r'(?<=[\.\!\?\…])\s+')
RE_MINOR_PUNCT  = re.compile(r'(?<=[\,\;\:\-\–\u2014])\s+')

# ─── Quote/bracket-aware sentence splitting ──────────────────────────────────
# A sentence-ending punctuation mark INSIDE a pair of brackets/quotes is NOT a sentence boundary:
#   e.g. Yes ... something like: "What next? How long do I have to do this?", right?
# is ONE sentence, not three. Splitting with a plain regex (RE_SENTENCE_END) would break this
# into fragments, and the final fragment starting with ", right?" would begin with a comma - not a sentence.
#
# Single quotes ' and ’ are intentionally OMITTED from the list: they overlap with apostrophes in
# "don't" / "l’ordre", which would open a quote that never closes and swallow the rest.
_OPEN_TO_CLOSE = {
    '(': ')', '[': ']', '{': '}',
    '“': '”', '‘': '’', '«': '»', '‹': '›', '「': '」', '『': '』',
}
_OPENERS = frozenset(_OPEN_TO_CLOSE)
_CLOSERS = frozenset(_OPEN_TO_CLOSE.values())
_SYMMETRIC_QUOTE = '"'   # same character opens and closes -> use a toggle flag
_SENT_END_CHARS  = frozenset('.!?…')
# A closing mark immediately AFTER sentence-ending punctuation still belongs to that sentence: e.g. `...?"` or `(really!)`
_TRAILING_CLOSE = _CLOSERS | frozenset('"\'’”')

# v2 noise cleanup
_NOISE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'([.!?])[.,;:]+'), r'\1'),
    (re.compile(r'[.,;:]+([.!?])'), r'\1'),
    (re.compile(r'\s+[,;]\s+'),     ' '),
    (re.compile(r' {2,}'),          ' '),
]
_MULTI_PUNCT = re.compile(r'([.!?])\s*[.!?]+')

# ─── Data class ──────────────────────────────────────────────────────────────

@dataclass
class PhoneChunk:
    text: str
    is_sentence_end: bool  # True = real sentence end | False = artificial split

# ─── Audio utils ─────────────────────────────────────────────────────────────

# MINIMUM pause (seconds) between two chunks depending on the split BOUNDARY: paragraph
# break (line break) -> longest pause, end of sentence (.!?) -> medium pause, in-sentence
# break (,;: or forced split) -> short pause. This is the ACTUAL audible pause (trailing
# silence of the previous chunk + inserted zeros + leading silence of the next chunk):
# ``join_audio_chunks`` keeps the model-generated audio intact and only inserts zeros when
# the total silence at the gap is below this value; naturally longer trailing silence is
# kept. History: the original table (0.35/0.18/0.04) was ADDITIONAL padding without
# measuring trailing silence, so the real pause depended on the voice (v3 Turbo presets
# naturally produced ~300 ms of silence before EOS, voice clones almost none -> chunks
# "overlapped"); v3.5.1 trimmed edges then compensated to match the table (0.55/0.32/0.14)
# - edge trimming was removed 09/2026 because it discarded natural tails unnecessarily
# (the trimmed portion was < -45 dB). Current values follow the pauses the model NATURALLY
# produces when a whole sentence stays in one chunk (measured on 23 v3 Turbo presets,
# 09/2026): comma ~0.30-0.45 s, end of sentence ~0.50-0.65 s.
V3_GAP_SILENCE = {"para": 0.70, "sentence": 0.50, "minor": 0.30}

# Chunk edge trim/fade (``trim_and_fade``): still used in the v3 Nano engine for flow-model
# output; the v3 Turbo path in join_audio_chunks no longer trims edges (09/2026).
EDGE_THRESH_DB = -45.0   # "voiced" threshold on the mean|x| envelope over a 10 ms window
EDGE_KEEP_S = 0.04       # silence kept at each end after trimming
EDGE_FADE_S = 0.015      # cosine fade at both edges to avoid clicks

# Reasonable maximum number of audio frames for ONE chunk based on phoneme length - an upper
# bound symmetric to the MIN_FRAMES_PER_PHONE=0.25 lower-bound guard in v3_turbo_serve.engine.
# Measured on the pnnbao-ump/vi-tts-v3-finetune-mix dataset (129,790 rows, 2026-08):
# frames/phoneme-character p50=0.53, p99=1.10, p99.9=1.50; the cap 24 + 2.0*len covers
# 99.9915% of rows (11/129,790 exceed it). Very short rows have a high ratio due to fixed
# overhead (len<15 bin: max 25 frames) - that is covered by the slack. Short chunks (1-2 words)
# often miss the stop token and "babble on" - this cap cuts off the fabricated part instead of
# running to max_new_frames (300); normal long chunks have a cap > 300 so they are unaffected.
# Markup (<en>...</en>, <|emotion_k|>) takes up characters but not frames, so it is stripped
# before measuring.
MAX_FRAMES_PER_PHONE = 2.0
_FRAME_CAP_SLACK = 24            # frame allowance for lead-in / fixed overhead
_FRAME_MARKUP_RE = re.compile(r"<\|emotion_\d+\|>|</?en>")

# ── Reference clip encoding: odd tail (issue #198) ──────────────────────────
# MOSS codec: 1 frame = 3840 samples @ 48 kHz. If a clip is not frame-aligned, the encoder
# pads the last frame and that frame ALWAYS yields codebook-0 = 455 - not silence (482) but
# a clearly audible short sound (RMS ~0.03 vs real tail ~0.001). It sticks to ref_codes for
# every cloned voice and to the labels of 100% of fine-tune rows, so the model learns it and
# emits it at the end of sentences. Fix: zero-pad to a full frame BEFORE encoding (making the
# last frame the true code of the tail + silence) and force the frame count = samples / 3840
# (the ONNX path always returns one extra frame). For codes saved with the old version, drop
# the trailing 455 frame when loading.
CODEC_SAMPLES_PER_FRAME = 3840
ENCODER_PAD_CODE = 455


def pad_to_codec_frame(wav, sr: int = 48_000):
    """Zero-pad at the end so the length is a multiple of one codec frame (applies only @ 48 kHz)."""
    n = len(wav)
    if sr != 48_000 or n % CODEC_SAMPLES_PER_FRAME == 0:
        return wav
    return np.concatenate([np.asarray(wav, dtype=np.float32),
                           np.zeros(CODEC_SAMPLES_PER_FRAME - n % CODEC_SAMPLES_PER_FRAME, np.float32)])


def strip_encoder_pad_frame(codes):
    """Remove the encoder-padded trailing frame (codebook-0 == 455) from ``(T, n_vq)`` codes
    saved with the old version. A real frame rarely has this code, and losing 80 ms of reference
    context is harmless - keeping it means every generation hears a stray sound at the end of
    the context."""
    if codes is None:
        return None
    arr = np.asarray(codes)
    if arr.ndim == 2 and arr.shape[0] > 1 and int(arr[-1, 0]) == ENCODER_PAD_CODE:
        return arr[:-1]
    return arr


# The codec runs at 12.5 frames/second. For very short chunks the linear phoneme formula is
# still too generous (e.g. a single short word -> 40 frames = 3.2s of pure fabrication), so we add an extra
# cap by SYLLABLE count: 13 frames (~1s) for 1 syllable, +5 frames per additional syllable,
# applied to chunks with <= SYLLABLE_CAP_MAX_SYL syllables. Count by syllables, NOT by
# words: "notification" is one word but 5 syllables, read in nearly 1s - a fixed 13-frame
# cap for "one word" would cut it off (see syllable_count). Short-chunk measurements
# (2026-09): 1 syllable 6-9 frames, 2 syllables 6-10, 3-4 syllables 12-15 - this cap still
# leaves >= 1.5x headroom. Not applied when there is an emotion cue (laughter/sighs take
# real frames).
SINGLE_WORD_MAX_FRAMES = 13      # cap for a 1-syllable chunk (~1s @ 12.5 frames/s)
SYLLABLE_CAP_PER_EXTRA = 5       # +frames for each additional syllable
SYLLABLE_CAP_MAX_SYL = 4         # longer chunks use the phoneme-based formula
_SINGLE_WORD_MAX_PHONES = 24     # maximum reasonable phonemes for ONE syllable


def is_cue_only(phonemes: str) -> bool:
    """Chunk contains only an emotion cue (e.g. laughter/sigh/throat-clearing tags), with no syllables."""
    ph = phonemes or ""
    return "<|emotion_" in ph and not any(ch.isalpha() for ch in _FRAME_MARKUP_RE.sub("", ph))


def max_expected_frames(phonemes: str) -> int:
    """Maximum reasonable number of frames for a chunk with this ``phonemes`` string."""
    stripped = _FRAME_MARKUP_RE.sub("", phonemes or "")
    eff_len = len(stripped)
    cap = _FRAME_CAP_SLACK + int(np.ceil(MAX_FRAMES_PER_PHONE * eff_len))
    if is_cue_only(phonemes):
        # Standalone cue: measured over 48 generations (2026-09), natural laughter / sighs /
        # throat-clearing voices last 5-12 frames; EOS-miss cases jump straight to 19-60 frames.
        # Treat as a 1-syllable chunk: ~1s cap, hitting the cap => regenerate (babble_suspect).
        return min(cap, SINGLE_WORD_MAX_FRAMES)
    if "<|emotion_" not in (phonemes or ""):
        syl = max(1, syllable_count(phonemes))      # empty string / no vowels -> treat as 1 syllable
        # An abnormally long "syllable" (> _SINGLE_WORD_MAX_PHONES phonemes per syllable) is
        # due to normalization sticking words together, not a real syllable -> let the usual formula handle it.
        if syl <= SYLLABLE_CAP_MAX_SYL and eff_len <= _SINGLE_WORD_MAX_PHONES * syl:
            cap = min(cap, SINGLE_WORD_MAX_FRAMES + SYLLABLE_CAP_PER_EXTRA * (syl - 1))
    return cap


def gaps_to_silence(gaps: List[str]) -> List[float]:
    """Map a list of boundary types -> list of TOTAL pause durations (seconds) for ``join_audio_chunks``."""
    return [V3_GAP_SILENCE.get(g, V3_GAP_SILENCE["sentence"]) for g in gaps]


def edge_silence(
    wav: np.ndarray, sr: int, thresh_db: float = EDGE_THRESH_DB, win_s: float = 0.01
) -> Tuple[int, int]:
    """``(lead, tail)``: number of silent samples at the start and end of ``wav`` (mean|x|
    envelope over a ``win_s`` window, below ``thresh_db`` is silence). All-silent wav -> ``(len, 0)``."""
    n_samp = int(wav.size)
    win = max(1, int(win_s * sr))
    n_win = n_samp // win
    if n_win == 0:
        return n_samp, 0
    env = np.abs(wav[: n_win * win]).reshape(n_win, win).mean(1)
    above = np.flatnonzero(env > 10 ** (thresh_db / 20))
    if not above.size:
        return n_samp, 0
    return int(above[0]) * win, n_samp - (int(above[-1]) + 1) * win


def trim_and_fade(
    wav: np.ndarray,
    sr: int,
    thresh_db: float = EDGE_THRESH_DB,
    keep_s: float = EDGE_KEEP_S,
    fade_s: float = EDGE_FADE_S,
) -> np.ndarray:
    """Trim model-generated silence at both ends (keeping ``keep_s`` at each end), then apply a
    cosine ``fade_s`` fade at both edges so joins do not click and the pause is determined only
    by ``join_audio_chunks``. Returns a copy; empty wav is returned unchanged."""
    if wav.size == 0:
        return wav
    lead, tail = edge_silence(wav, sr, thresh_db)
    keep = int(keep_s * sr)
    a = max(0, lead - keep)
    b = wav.size - max(0, tail - keep)
    out = np.array(wav[a:b], dtype=np.float32, copy=True)
    n = min(int(fade_s * sr), out.size // 2)
    if n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)
        out[:n] *= ramp
        out[-n:] *= ramp[::-1]
    return out


def pause_pad_samples(prev_wav: np.ndarray, next_wav: np.ndarray, sr: int, pause_s: float) -> int:
    """Number of zero samples to insert between ``prev_wav`` and ``next_wav`` so the ACTUAL pause
    (trailing silence of prev + zeros + leading silence of next) reaches ``pause_s``; 0 if already
    sufficient. Used for streaming, where the previous chunk has already been played and can no
    longer be trimmed (if the tail is entirely silent, the whole tail is counted)."""
    lead_prev, tail = edge_silence(prev_wav, sr)
    if lead_prev == prev_wav.size:          # prev is entirely silent
        tail = prev_wav.size
    lead, _ = edge_silence(next_wav, sr)
    return max(0, int(pause_s * sr) - tail - lead)


def join_audio_chunks(
    chunks: List[np.ndarray],
    sr: int,
    silence_p: float = 0.0,
    crossfade_p: float = 0.0,
    silence_ps: Optional[List[float]] = None,
) -> np.ndarray:
    """Join audio chunks.

    ``silence_ps`` (optional, v3 path): ``silence_ps[i]`` is the MINIMUM pause (seconds)
    between chunk ``i`` and ``i+1`` based on the boundary type. Each chunk's audio is kept
    INTACT (no edge trimming, no fade); :func:`pause_pad_samples` measures trailing silence
    of the previous chunk + leading silence of the next chunk and only inserts zeros for the
    remainder - voice clones with short tails are padded to reach the target, presets with
    long tails keep their natural rhythm. When ``silence_ps`` is provided, ``silence_p`` and
    ``crossfade_p`` are ignored; gaps missing a value pause for 0 (direct join).

    Without ``silence_ps`` (v1/v2 path): insert ``silence_p`` seconds of zeros, or crossfade
    ``crossfade_p`` seconds, or join directly - unchanged from before.
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    if silence_ps is not None:
        parts: List[np.ndarray] = [chunks[0]]
        for i in range(1, len(chunks)):
            pause_s = silence_ps[i - 1] if i - 1 < len(silence_ps) else 0.0
            pad = pause_pad_samples(chunks[i - 1], chunks[i], sr, pause_s)
            if pad > 0:
                parts.append(np.zeros(pad, dtype=np.float32))
            parts.append(chunks[i])
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    if len(chunks) == 1:
        return chunks[0]

    silence_samples   = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    # Collect parts and concatenate ONCE. Appending each chunk into a growing array copies the
    # entire joined portion every iteration: a 2-hour piece / ~1000 sentences ran out of RAM
    # ("Unable to allocate 117 MiB") after joining only ~10 minutes (user report, 2026-09).
    parts: List[np.ndarray] = [chunks[0]]
    total = len(chunks[0])

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        if silence_samples <= 0 and crossfade_samples > 0:
            overlap = min(total, len(next_chunk), crossfade_samples)
            if overlap > 0:
                # The tail to blend may span several short parts: merge them first.
                while len(parts[-1]) < overlap:
                    tail = parts.pop()
                    parts[-1] = np.concatenate([parts[-1], tail])
                last = parts[-1]
                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in  = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                blended  = last[-overlap:] * fade_out + next_chunk[:overlap] * fade_in
                parts[-1] = last[:-overlap]
                parts.append(blended)
                parts.append(next_chunk[overlap:])
                total += len(next_chunk) - overlap
                continue
        if silence_samples > 0:
            parts.append(np.zeros(silence_samples, dtype=np.float32))
            total += silence_samples
        parts.append(next_chunk)
        total += len(next_chunk)

    return np.concatenate(parts)

# ─── v1: split raw text ──────────────────────────────────────────────────────

def _scan_sentences(text: str, quote_aware: bool = True) -> Tuple[List[str], bool]:
    """Scan ``text`` once and split at ``.!?…`` marks NOT inside brackets/quotes.

    Returns ``(sentences, balanced)``; ``balanced=False`` means the text has unbalanced
    brackets/quotes (missing closing mark) - the caller should rescan with ``quote_aware=False``.
    """
    sentences: List[str] = []
    n = len(text)
    start = i = 0
    depth = 0          # bracket nesting depth ( [ { “ « …
    in_quote = False   # inside "…" (straight double quote, symmetric)

    while i < n:
        ch = text[i]
        if quote_aware and ch == _SYMMETRIC_QUOTE:
            in_quote = not in_quote
        elif quote_aware and ch in _OPENERS:
            depth += 1
        elif quote_aware and ch in _CLOSERS:
            if depth:
                depth -= 1
        elif ch in _SENT_END_CHARS and depth == 0 and not in_quote:
            j = i + 1
            while j < n and text[j] in _SENT_END_CHARS:   # consume "?!", "..."
                j += 1
            while j < n and text[j] in _TRAILING_CLOSE:   # consume a closing mark attached after
                j += 1
            # Only a sentence boundary when followed by whitespace or end of text;
            # this prevents splitting decimals/times like "3.5" / "8.30".
            if j >= n or text[j].isspace():
                sentences.append(text[start:j])
                start = i = j
                continue
            i = j
            continue
        i += 1

    if start < n:
        sentences.append(text[start:])

    return [s.strip() for s in sentences if s.strip()], (depth == 0 and not in_quote)


def split_into_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences, NOT splitting inside brackets/quotes.

    Used on RAW text (before normalization): the sea-g2p normalizer removes all
    brackets/quotes (``"…"`` -> ``,``), so after normalization there is no way to
    distinguish a sentence-ending ``?`` from a ``?`` inside a quoted phrase.

    If the text has unbalanced brackets (missing closing mark), rescan ignoring
    brackets so a stray quote does not swallow the rest into one giant sentence.
    """
    if not text:
        return []
    sentences, balanced = _scan_sentences(text, quote_aware=True)
    if not balanced:
        sentences, _ = _scan_sentences(text, quote_aware=False)
    return sentences


# ─── Split long fragments with no punctuation: prefer connectors ──────────────
_CONN_WORDS = frozenset(
    "và nhưng or song rồi nên vì nếu khi để do bởi".split()
)
# A two-word pair is ONE connector: cut before the whole pair. Pairs also act as a BLOCK rule -
# do not cut inside the pair or immediately after its first word.
_CONN_PAIRS = frozenset([
    ("sau", "khi"), ("trước", "khi"), ("trong", "khi"), ("mỗi", "khi"),
    ("đến", "khi"), ("tới", "khi"), ("cho", "nên"), ("cho", "đến"),
    ("bởi", "vì"), ("nếu", "như"), ("tuy", "nhiên"), ("thế", "nhưng"),
    ("vì", "vậy"), ("vì", "thế"), ("do", "đó"), ("sau", "đó"),
])
_CONN_STRIP = "\"'“”‘’()[]«»…"


def _conn_key(token: str) -> str:
    return token.strip(_CONN_STRIP).lower()


# The chunk character limit is RELATIVE, not hard: if the remainder after a cut is too
# short (<= slack characters, including attached punctuation), merge it into the previous
# chunk even if that exceeds the limit. Slack is 15 for typical limits (>= 120); smaller
# limits scale it down proportionally (max_chars // 8) to avoid loosening too much.
CHUNK_TAIL_SLACK = 15


def _tail_slack(max_chars: int) -> int:
    return min(CHUNK_TAIL_SLACK, max_chars // 8)


def _fits(cur_len: int, add_len: int, max_chars: int) -> bool:
    """Whether ``add_len`` characters (joined by one space) fit into a chunk of length ``cur_len``:
    within the limit, or the added part is short enough to use the slack."""
    total = cur_len + 1 + add_len if cur_len else add_len
    slack = _tail_slack(max_chars)
    return total <= max_chars or (add_len <= slack and total <= max_chars + slack)


# Words the normalizer uses to read numbers. Cutting between these words would split a number.
# Some are also ordinary words - blocking a bit extra is better than splitting a year like 2031.
_NUMBER_WORDS = frozenset(
    "không một mốt hai ba bốn tư năm lăm sáu bảy tám chín mười mươi trăm nghìn ngàn "
    "triệu tỷ tỉ linh lẻ phẩy chấm".split()
)


def _is_number_word(token: str) -> bool:
    key = _conn_key(token)
    return key in _NUMBER_WORDS or key.isdigit()


def _span_len(words: List[str], start: int, end: int) -> int:
    """Length of ``" ".join(words[start:end])``."""
    if end <= start:
        return 0
    return sum(len(w) for w in words[start:end]) + (end - start - 1)


def _balanced_cut(words: List[str], start: int, target: float, max_chars: int, min_left: int) -> int:
    """Choose ``end`` so that ``words[start:end]`` <= ``max_chars`` and is CLOSEST to ``target``.
    Prefer cutting before a connector (``_CONN_WORDS``/``_CONN_PAIRS``, left side >= ``min_left``);
    otherwise the word boundary closest to the target, as long as it does not fall inside a connector
    pair or split a number. If no valid spot remains (a huge token, overlapping pairs), cut at the
    limit and back away from pairs."""
    best_nat = best_plain = None
    end_cap = start + 1
    for j in range(start + 1, len(words)):
        left = _span_len(words, start, j)
        if left > max_chars:
            break
        end_cap = j
        key, prev = _conn_key(words[j]), _conn_key(words[j - 1])
        nxt = _conn_key(words[j + 1]) if j + 1 < len(words) else ""
        d = abs(left - target)
        inside_pair = (prev, key) in _CONN_PAIRS
        natural = (key in _CONN_WORDS or (key, nxt) in _CONN_PAIRS) and prev not in _CONN_WORDS
        if natural and not inside_pair and left >= min_left and (best_nat is None or d < best_nat[0]):
            best_nat = (d, j)
        plain_ok = not inside_pair and not (_is_number_word(words[j - 1]) and _is_number_word(words[j]))
        if plain_ok and (best_plain is None or d < best_plain[0]):
            best_plain = (d, j)
    if best_nat is not None:
        return best_nat[1]
    if best_plain is not None:
        return best_plain[1]
    end = end_cap
    while end > start + 1 and (_conn_key(words[end - 1]), _conn_key(words[end])) in _CONN_PAIRS:
        end -= 1
    return end


def _split_long_part(part: str, max_chars: int) -> List[str]:
    """Split a fragment longer than ``max_chars`` (with no remaining punctuation to anchor on) into
    pieces <= ``max_chars`` by WORDS, distributed EVENLY: knowing ``k = ceil(len / max_chars)``
    pieces are needed, each piece targets ``len / k`` characters and cuts at the nearest valid point
    (see :func:`_balanced_cut`). Previously this was greedy - filling to the limit and leaving a
    small remainder alone - which produced unbalanced splits and could cut at bad spots.

    A connector is accepted only when the left piece is >= ``max_chars // 3``. A final remainder
    shorter than the slack (``_tail_slack``) is still merged into the previous piece even if it
    exceeds the limit (relative limit). ``<en>...</en>`` tokens always stay intact."""
    words = _tokenize_keep_en(part)
    min_left = max_chars // 3
    pieces: List[str] = []
    start = 0
    while start < len(words):
        rest = _span_len(words, start, len(words))
        if rest <= max_chars:
            pieces.append(" ".join(words[start:]))
            break
        # Relative limit: if the remainder after filling to the limit is just a tiny tail,
        # merge it instead of leaving a fragment on its own (see ``_fits``).
        full = start + 1
        while full < len(words) and _span_len(words, start, full + 1) <= max_chars:
            full += 1
        if _fits(_span_len(words, start, full), _span_len(words, full, len(words)), max_chars):
            pieces.append(" ".join(words[start:]))
            break
        n_pieces = -(-rest // max_chars)          # ceil
        end = _balanced_cut(words, start, rest / n_pieces, max_chars, min_left)
        pieces.append(" ".join(words[start:end]))
        start = end
    return pieces


def pack_sentences_into_chunks(sentences: List[str], max_chars: int = 256) -> List[str]:
    """Pack the given SENTENCES into chunks ~<= ``max_chars`` (greedy, preserving order).

    Only sentences longer than ``max_chars`` are split further - first at in-sentence
    punctuation (``,;:``), and finally by words (preferring cuts before connectors, see
    :func:`_split_long_part`).

    The limit is RELATIVE: an added sentence/fragment shorter than ``_tail_slack(max_chars)``
    (15 characters at typical limits) is still merged into the open chunk even if it exceeds
    the limit by that much - avoiding tiny fragments left alone and then attached to the next sentence.
    """
    final_chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if buffer:
                final_chunks.append(buffer)
                buffer = ""

            sub_parts = RE_MINOR_PUNCT.split(sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                if _fits(len(buffer), len(part), max_chars):
                    buffer = (buffer + ' ' + part) if buffer else part
                else:
                    if buffer:
                        final_chunks.append(buffer)
                    buffer = part
                    if len(buffer) > max_chars:
                        pieces = _split_long_part(buffer, max_chars)
                        final_chunks.extend(pieces[:-1])
                        buffer = pieces[-1] if pieces else ""
        else:
            if buffer and not _fits(len(buffer), len(sentence), max_chars):
                final_chunks.append(buffer)
                buffer = sentence
            else:
                buffer = (buffer + ' ' + sentence) if buffer else sentence

    if buffer:
        final_chunks.append(buffer)

    return [c.strip() for c in final_chunks if c.strip()]


def split_text_into_chunks(text: str, max_chars: int = 256) -> List[str]:
    """Split raw text (not yet phonemized) into chunks <= max_chars."""
    if not text:
        return []

    final_chunks: List[str] = []
    for para in RE_NEWLINE.split(text.strip()):
        para = para.strip()
        if para:
            final_chunks.extend(
                pack_sentences_into_chunks(split_into_sentences(para), max_chars)
            )
    return final_chunks


def _classify_gap(chunk: str) -> str:
    """Classify the boundary IMMEDIATELY AFTER ``chunk`` based on its final punctuation:
    end of sentence (``.!?``) -> ``"sentence"``; otherwise (``,;:`` or a forced in-sentence
    split) -> ``"minor"``. The ``"para"`` boundary (paragraph break) is assigned separately by the caller."""
    c = chunk.rstrip()
    return "sentence" if c and c[-1] in ".!?" else "minor"


def split_text_into_chunks_with_gaps(
    text: str, max_chars: int = 256
) -> Tuple[List[str], List[str]]:
    """Like :func:`split_text_into_chunks` but also returns the boundary type BETWEEN chunks so
    audio can be joined with different pause lengths.

    Returns ``(chunks, gaps)`` where ``gaps[i] in {"para","sentence","minor"}`` is the
    boundary between ``chunks[i]`` and ``chunks[i+1]`` (``len(gaps) == len(chunks)-1``):
      * ``"para"``     - different PARAGRAPHS (separated by ``\\n``) -> long pause
      * ``"sentence"`` - end of sentence (left chunk ends with ``.!?``) -> medium pause
      * ``"minor"``    - in-sentence break (``,;:`` / forced split)      -> almost continuous
    """
    if not text:
        return [], []

    paragraphs = RE_NEWLINE.split(text.strip())
    chunks: List[str] = []
    gaps: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_chunks = split_text_into_chunks(para, max_chars=max_chars)
        if not para_chunks:
            continue
        if chunks:                       # boundary with the PREVIOUS paragraph is a paragraph break
            gaps.append("para")
        for j, ch in enumerate(para_chunks):
            if j > 0:                    # boundary within the SAME paragraph: based on punctuation
                gaps.append(_classify_gap(para_chunks[j - 1]))
            chunks.append(ch)

    return chunks, gaps

# ─── v2 helpers ──────────────────────────────────────────────────────────────

def _pick_strongest(m: re.Match) -> str:
    s = m.group(0)
    return '!' if '!' in s else '?' if '?' in s else '.'


def _clean_phoneme_noise(text: str) -> str:
    for pattern, repl in _NOISE_RULES:
        text = pattern.sub(repl, text)
    return _MULTI_PUNCT.sub(_pick_strongest, text).strip()


def _find_best_split(text: str, max_size: int) -> Tuple[int, bool]:
    mid = max_size // 2
    best_comma_pos, best_comma_dist = -1, max_size
    best_space_pos, best_space_dist = -1, max_size

    for i in range(min(max_size, len(text))):
        ch = text[i]
        if ch == ',':
            d = abs(i - mid)
            if d < best_comma_dist:
                best_comma_dist, best_comma_pos = d, i
        elif ch == ' ':
            d = abs(i - mid)
            if d < best_space_dist:
                best_space_dist, best_space_pos = d, i

    if best_comma_pos != -1:
        return best_comma_pos, True
    if best_space_pos != -1:
        return best_space_pos, False
    return -1, False


def _smart_split_body(text: str, max_chunk_size: int) -> List[str]:
    result: List[str] = []
    stack = [text.strip()]

    while stack:
        seg = stack.pop()
        if not seg:
            continue
        if len(seg) <= max_chunk_size:
            result.append(seg)
            continue

        pos, _ = _find_best_split(seg, max_chunk_size)
        if pos != -1:
            left  = seg[:pos].rstrip()
            right = seg[pos + 1:].lstrip()
        else:
            cut = max_chunk_size
            while cut > 0 and seg[cut - 1] != ' ':
                cut -= 1
            if cut == 0:
                cut = max_chunk_size
            left  = seg[:cut].rstrip()
            right = seg[cut:].lstrip()

        if right:
            stack.append(right)
        if left:
            stack.append(left)

    return result


def _split_sentence(sent: str, max_chunk_size: int) -> List[PhoneChunk]:
    sent = sent.strip()
    if not sent:
        return []

    if sent[-1] in '.!?':
        body, punct = sent[:-1].rstrip(), sent[-1]
    else:
        body, punct = sent, '.'

    if not body:
        return []

    if len(sent) <= max_chunk_size:
        return [PhoneChunk(text=body + punct, is_sentence_end=True)]

    sub_chunks = _smart_split_body(body, max_chunk_size)
    if not sub_chunks:
        return [PhoneChunk(text=punct, is_sentence_end=True)]

    last_idx = len(sub_chunks) - 1
    return [
        PhoneChunk(
            text=chunk + (punct if i == last_idx else '.'),
            is_sentence_end=(i == last_idx),
        )
        for i, chunk in enumerate(sub_chunks)
        if chunk
    ]

# ─── v2: split phoneme string ────────────────────────────────────────────────

def split_into_chunks_v2(
    full_phones: str,
    max_chunk_size: int = 256,
    min_chunk_size: int = 10,
) -> List[PhoneChunk]:
    """
    Split a phoneme string into PhoneChunk objects.
      is_sentence_end=True  -> real sentence end -> needs silence
      is_sentence_end=False -> artificial split  -> no silence needed
    """
    if not full_phones:
        return []

    full_phones = _clean_phoneme_noise(full_phones)

    raw_parts: List[PhoneChunk] = []
    for para in RE_NEWLINE.split(full_phones):
        para = para.strip()
        if not para:
            continue
        for sent in RE_SENTENCE_FINDALL.findall(para):
            sent = sent.strip()
            if sent:
                raw_parts.extend(_split_sentence(sent, max_chunk_size))

    if not raw_parts:
        return []

    merged: List[PhoneChunk] = []
    i, n = 0, len(raw_parts)
    while i < n:
        cur = raw_parts[i]
        while len(cur.text) < min_chunk_size and i + 1 < n:
            nxt       = raw_parts[i + 1]
            candidate = cur.text.rstrip('.!?').rstrip() + ' ' + nxt.text
            if len(candidate) <= max_chunk_size:
                cur = PhoneChunk(text=candidate, is_sentence_end=nxt.is_sentence_end)
                i += 1
            else:
                break
        merged.append(cur)
        i += 1

    if len(merged) >= 2 and len(merged[-1].text) < min_chunk_size:
        last      = merged.pop()
        candidate = merged[-1].text.rstrip('.!?').rstrip() + ' ' + last.text
        if len(candidate) <= max_chunk_size:
            merged[-1] = PhoneChunk(text=candidate, is_sentence_end=last.is_sentence_end)
        else:
            merged.append(last)

    return merged


def get_silence_duration_v2(chunk: PhoneChunk) -> float:
    """
    Silence after a chunk (seconds).
      is_sentence_end=False -> 0.0s
      ends with '!'/'?'     -> 0.4s
      ends with '.'         -> 0.3s
    """
    if not chunk.is_sentence_end:
        return 0.0
    return 0.4 if chunk.text.strip()[-1] in '!?' else 0.3

# ─── Misc ────────────────────────────────────────────────────────────────────

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')

# ── Babble guard (very short chunks "babble on") ──────────────────────────────
# 1-3 syllable chunks occasionally miss the stop token and fabricate extra words to
# "complete" the sentence. Measured on 160 short chunks on GPU (2026-09): ~3-6% affected,
# almost only chunks with <= 2 syllables. EOS probability gives no advance warning and "silent"
# codec codes also appear in pauses between syllables, so this cannot be caught DURING the
# generation loop; instead, after generation count energy bursts: one syllable = one burst,
# more bursts than syllables => fabrication => regenerate.
BABBLE_MAX_SYLLABLES = 3       # only check chunks with <= N syllables (longer ones merge bursts, not countable)
BABBLE_MAX_RETRIES = 2
_BURST_THRESH_DB = -18.0       # a burst = RMS above this threshold relative to peak RMS (breaths ~ -25 dB not counted)
_BURST_MIN_GAP_MS = 60         # two bursts closer than this are one burst (aspirated onset)
_BURST_MIN_MS = 30
# ɚ ɝ  r-colored vowels (computer, fire, hour, bird - American voices)
# ᵻ ᵿ  reduced vowels used by eSpeak for unstressed syllables (director -> dᵻɹˈɛktɚ)
# Missing these four symbols would drop entire syllables from the count. Vietnamese does not
# use any of them, so adding them does not affect Vietnamese.
_IPA_VOWELS = set("aeiouyæɐɑɒɔəɘɛɜɤɯɵøœʉʊʌɪɨɚɝᵻᵿ")


def syllable_count(phonemes: str) -> int:
    """Estimate the number of syllables from a SEA-G2P phoneme string.

    Vietnamese: each word is one syllable. English words (<en>) may have multiple syllables,
    so they are counted by vowel groups ('sˈækaɪ' -> 2). Markup and punctuation are stripped.
    """
    stripped = _FRAME_MARKUP_RE.sub("", phonemes or "")
    total = 0
    for tok in stripped.split():
        # A new vowel group starts only after a real CONSONANT; length marks (ː), stress
        # marks (ˈ ˌ), and tone digits do not split groups.
        groups, in_v, consonant_seen = 0, False, True
        for ch in tok:
            if ch in _IPA_VOWELS:
                if not in_v and consonant_seen:
                    groups += 1
                in_v, consonant_seen = True, False
            elif ch in "ːˈˌ" or ch.isdigit():
                # Stress marks precede the syllable they stress, so a stress mark appearing
                # AFTER the token already has a vowel group is a syllable boundary - even with
                # no consonant in between (e.g. create/radio/computer cases). The groups > 0
                # condition keeps Vietnamese intact: sea-g2p places exactly ONE stress mark per
                # syllable and always BEFORE the first vowel group, so groups is still 0 then.
                if ch in "ˈˌ" and groups > 0:
                    in_v, consonant_seen = False, True
                else:
                    in_v = False
            else:
                in_v, consonant_seen = False, True
        if any(ch.isalpha() for ch in tok):
            total += max(1, groups)
    return total


def count_speech_bursts(wav: np.ndarray, sr: int) -> int:
    """Number of energy bursts (approximate syllable count) in a mono waveform."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    hop = max(1, int(sr * 0.010))
    n = len(wav) // hop
    if n == 0:
        return 0
    env = np.sqrt((wav[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    peak = float(env.max())
    if peak <= 1e-6:
        return 0
    on = env > peak * (10 ** (_BURST_THRESH_DB / 20))
    min_gap = max(1, _BURST_MIN_GAP_MS // 10)
    min_len = max(1, _BURST_MIN_MS // 10)
    bursts, start, last_on = [], None, None
    for i, o in enumerate(on):
        if o:
            if start is None:
                start = i
            elif i - last_on > min_gap:          # gap longer than threshold -> new burst
                bursts.append((start, last_on)); start = i
            last_on = i
    if start is not None:
        bursts.append((start, last_on))
    return sum(1 for a, b in bursts if (b - a + 1) >= min_len)


def babble_suspect(wav: np.ndarray, sr: int, phonemes: str, cap_frames: int,
                   n_frames: Optional[int] = None, frames_per_sec: float = 12.5):
    """-> (suspect, syllables, bursts, n_frames) for ONE just-generated chunk.

    Only considers chunks with <= BABBLE_MAX_SYLLABLES syllables and no emotion cue. Suspects
    "babbling on" when: burst count > syllable count, OR a chunk with <= 2 syllables runs up
    against the frame cap (A/B on 720 chunks: every fabricated-word case was a 1-syllable chunk
    at 12-13/13 frames, while normal 1-syllable chunks reach EOS at 6-9 frames).
    """
    if n_frames is None:
        n_frames = int(round(len(wav) / (sr / frames_per_sec)))
    if is_cue_only(phonemes):
        # Cannot count bursts reliably (a laugh is many bursts) - use only the cap-hit rule.
        return n_frames >= cap_frames - 1, 0, 0, n_frames
    syl = syllable_count(phonemes)
    if syl == 0 or syl > BABBLE_MAX_SYLLABLES or "<|emotion_" in (phonemes or ""):
        return False, syl, 0, 0
    bursts = count_speech_bursts(wav, sr)
    hit_cap = syl <= 2 and n_frames >= cap_frames - 1
    return (bursts > syl) or hit_cap, syl, bursts, n_frames


def babble_prefer(new, old) -> bool:
    """Whether regenerated result ``new`` is preferable to ``old`` (tuples from babble_suspect)."""
    (n_bad, _, n_b, n_len), (o_bad, _, o_b, o_len) = new, old
    if n_bad != o_bad:
        return not n_bad
    return n_b < o_b or (n_b == o_b and n_len < o_len)


def babble_log_line(best, tries: int, cap: int) -> str:
    what = f"chunk with {best[1]} syllable(s): {best[2]} speech bursts" if best[1] else "standalone cue"
    return (f"babble guard: {what}, {best[3]}/{cap} frames after {tries} retr{'y' if tries == 1 else 'ies'}"
            + (" - still suspect" if best[0] else ""))
