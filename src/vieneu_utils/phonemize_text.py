"""
Phonemization module for VieNeu-TTS.
Delegates all normalization and G2P logic to the sea-g2p library,
which provides a unified, tested, and maintained Vietnamese G2P pipeline.
"""

import functools
import logging
import re
from typing import Optional
from sea_g2p import SEAPipeline, G2P, Normalizer, punc_norm

logger = logging.getLogger("Vieneu.Phonemizer")

# Split paragraphs by newline - used to normalize per paragraph (a natural boundary)
# before chunking, keeping input to regex backtracking at a safe level.
RE_NEWLINE_SPLIT = re.compile(r"[\r\n]+")

# ---------------------------------------------------------------------------
# Inline non-verbal cues (emotion tokens) - v3 Turbo emotion checkpoint
# ---------------------------------------------------------------------------
# The emotion checkpoint was trained with three non-verbal cues embedded directly
# in the PHONEME stream as special tokens. In the *text* they appear as bracketed
# tags; phonemization must leave them as the matching <|emotion_k|> token instead
# of spelling the bracketed words out. The mapping + spacing reproduce the
# training data (`phones` column of VieNeu-TTS-1000h-in-the-wild-coded) EXACTLY.
#
#   [chuckle]      / [cười]       -> <|emotion_1|>  (laugh)
#   [sigh]         / [thở dài]    -> <|emotion_2|>  (sigh)
#   [clear throat] / [hắng voice] -> <|emotion_3|>  (throat clear)
_EMOTION_TAG_TO_K = {
    "chuckle": 1, "cười": 1, "cuoi": 1,
    "sigh": 2, "thở dài": 2, "tho dai": 2,
    "clear throat": 3, "hắng voice": 3, "hang giong": 3, "hắc voice": 3, "hac giong": 3,
}
# Split on a [bracketed tag] or an already-resolved <|emotion_k|> token.
_EMOTION_SPLIT_RE = re.compile(r"(\[[^\]]+\]|<\|emotion_\d+\|>)")
# Punctuation that stays attached to the preceding emotion token (no space),
# mirroring the training phones, e.g. "... <|emotion_2|>. ...".
_ATTACHING_PUNCT = set(".,!?;:…)]}\"'’”")


def _emotion_tag_token(tag: str) -> Optional[str]:
    """Map a raw ``[tag]`` / ``<|emotion_k|>`` string to its ``<|emotion_k|>`` form.

    Returns ``None`` for an unrecognized bracketed span (caller phonemizes it as
    ordinary text).
    """
    t = tag.strip()
    if t.startswith("<|"):
        return t  # already an explicit emotion token - pass through unchanged
    inner = t[1:-1].strip().lower()  # drop the surrounding [ ]
    k = _EMOTION_TAG_TO_K.get(inner)
    return f"<|emotion_{k}|>" if k is not None else None

# ---------------------------------------------------------------------------
# Always-punc_norm normalizer wrapper
# ---------------------------------------------------------------------------
# VieNeu-TTS ALWAYS enables punc_norm (sea-g2p >= 0.7.6): short sentences (<5 words) have
# their final punctuation forced to ".", and long sentences missing a terminal punctuation
# mark get "." appended. Use this wrapper instead of sea_g2p.Normalizer everywhere so we do
# not have to pass punc_norm=True at every call site and to guarantee this "always-on"
# behavior for future call sites as well.
class PuncNormalizer:
    """``sea_g2p.Normalizer`` with punc_norm defaulting to True."""

    def __init__(self, lang: str = "vi") -> None:
        self._n = Normalizer(lang=lang)

    def normalize(self, text, punc_norm: bool = True):
        return self._n.normalize(text, punc_norm=punc_norm)

    def normalize_batch(self, texts, punc_norm: bool = True):
        return self._n.normalize_batch(texts, punc_norm=punc_norm)


# ---------------------------------------------------------------------------
# Shared singletons (instantiation is lazy-safe and thread-safe via GIL)
# ---------------------------------------------------------------------------
_pipeline: SEAPipeline = None
_g2p: G2P = None
_normalizer: PuncNormalizer = None

def _get_pipeline() -> SEAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SEAPipeline(lang="vi")
    return _pipeline

def _get_g2p() -> G2P:
    global _g2p
    if _g2p is None:
        _g2p = G2P(lang="vi")
    return _g2p

def _get_normalizer() -> PuncNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = PuncNormalizer()
    return _normalizer

# ---------------------------------------------------------------------------
# Public API  (same signatures as before - callers don't need to change)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _phonemize_cached(text: str, punc_norm: bool = True) -> str:
    """Cached single-text phonemization (normalize + G2P), with punc_norm enabled by default."""
    return _get_pipeline().run(text, punc_norm=punc_norm)


def phonemize_text(text: str) -> str:
    """Normalize and phonemize a single Vietnamese/bilingual text string."""
    return _phonemize_cached(text)


# Finalize end punctuation for ONE chunk using sea-g2p's punc_norm rules (single source
# of truth, not reimplemented): short sentences (< 5 words) are forced to a single ".",
# long sentences get "." appended if they do not already end with , . ! ?. Applied uniformly
# to text and phoneme chunks (including chunks ending with an emotion token
# "<|emotion_k|>" -> "<|emotion_k|>.").


def phonemize_text_with_emotions(text: str) -> str:
    """Phonemize ``text`` while preserving inline non-verbal cues as emotion tokens.

    Same as :func:`phonemize_text`, but inline cues ``[cười]``/``[thở dài]``/
    ``[hắng voice]`` (or the English ``[chuckle]``/``[sigh]``/``[clear throat]``,
    or an explicit ``<|emotion_k|>``) are kept as ``<|emotion_1|>``/``<|emotion_2|>``/
    ``<|emotion_3|>`` in the phoneme stream instead of being spelled out. Used by the
    v3 Turbo emotion checkpoint. Spacing matches the training data exactly: one
    space before the token, with following punctuation attached.
    """
    if "[" not in text and "<|emotion_" not in text:
        return _phonemize_cached(text)  # fast path: no cues -> plain cached phonemize
    out = ""
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(text)):
        token = _emotion_tag_token(part) if i % 2 == 1 else None
        if token is not None:
            out = (out + " " + token) if out else token
            continue
        # Fragment between emotion tokens: do NOT force punc_norm to avoid inserting "."
        # mid-sentence - keep spacing/format matching the emotion checkpoint's training data.
        # (The whole chunk has already been split in a sentence-aware way.)
        ph = _phonemize_cached(part, punc_norm=False) if part and part.strip() else ""
        if not ph:
            continue
        if not out:
            out = ph
        elif ph[0] in _ATTACHING_PUNCT:
            out += ph          # punctuation attaches to the previous token/phones
        else:
            out += " " + ph
    # Finalize end punctuation at the whole-chunk level (inner fragments used punc_norm=False).
    return punc_norm(out)


def phonemize_batch(
    texts: list[str],
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
    **kwargs,
) -> list[str]:
    """
    Phonemize multiple texts with bilingual support.

    Args:
        texts:          List of input strings.
        skip_normalize: If True, assume the texts are already normalized
                        (i.e. only run G2P, not the normalizer).
        phoneme_dict:   Optional custom {word: phoneme} dict that overrides
                        the built-in dictionary for specific words.
    """
    if not texts:
        return []

    g2p = _get_g2p()

    # punc_norm is ALWAYS enabled at the G2P layer: even if text is already normalized
    # (skip_normalize) or missing punctuation, the phone string still ends with a valid ".".
    if skip_normalize:
        # Texts are pre-normalized - only run the G2P layer
        return g2p.phonemize_batch(texts, punc_norm=True, phoneme_dict=phoneme_dict)
    else:
        # Full pipeline: normalize (punc_norm) then G2P (punc_norm)
        normalizer = _get_normalizer()
        normalized = [normalizer.normalize(t) for t in texts]
        return g2p.phonemize_batch(normalized, punc_norm=True, phoneme_dict=phoneme_dict)


def phonemize_with_dict(
    text: str,
    phoneme_dict: dict = None,
    skip_normalize: bool = False,
) -> str:
    """
    Phonemize a single text, optionally with a custom word→phoneme mapping.

    When phoneme_dict is None and skip_normalize is False, the result is
    cached via lru_cache for performance.
    """
    if phoneme_dict is not None:
        # Custom dict supplied - skip cache to avoid cross-contamination
        return phonemize_batch(
            [text], skip_normalize=skip_normalize, phoneme_dict=phoneme_dict
        )[0]
    if skip_normalize:
        # punc_norm is still enabled at the G2P layer even if the text is already normalized.
        return _get_g2p().phonemize_batch([text], punc_norm=True)[0]
    return _phonemize_cached(text)


def _normalize_sentence_keep_cues(sentence: str, normalizer) -> str:
    """Normalize ONE sentence, keeping inline emotion cues as ``<|emotion_k|>``."""
    if "[" not in sentence and "<|emotion_" not in sentence:
        return normalizer.normalize(sentence, punc_norm=False)
    parts = []
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(sentence)):
        if i % 2 == 1:                       # emotion tag
            tok = _emotion_tag_token(part)
            parts.append(tok if tok is not None else part)
        elif part.strip():
            parts.append(normalizer.normalize(part, punc_norm=False))
    return " ".join(p for p in parts if p)


def _normalized_sentences_by_para(
    text: str,
    skip_normalize: bool = False,
    keep_cues: bool = False,
) -> list[list[str]]:
    """Raw text -> list of PARAGRAPHS, each paragraph a list of normalized SENTENCES.

    The order is INTENTIONAL: *split sentences first, normalize after*: the normalizer removes all
    brackets/quotes (``"…"``/``(…)`` -> ``,``), so splitting after normalization would mistake a
    ``?`` inside a quoted phrase for a sentence end and produce fragments. Chunk packing length is
    still measured AFTER normalization (in :func:`~vieneu_utils.core_utils.pack_sentences_into_chunks`),
    so chunks do not grow when the normalizer expands text.

    ``punc_norm`` is NOT enabled at the sentence level: the "< 5 words force final punctuation to
    ``.``" rule would turn a short question into a statement mid-chunk, losing question intonation.
    Final punctuation is fixed only once at the CHUNK level.
    """
    from vieneu_utils.core_utils import split_into_sentences

    paragraphs = [p for p in RE_NEWLINE_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    normalizer = None if skip_normalize else _get_normalizer()
    out: list[list[str]] = []
    for para in paragraphs:
        sentences = split_into_sentences(para)
        if not sentences:
            continue
        if skip_normalize:
            out.append(sentences)
        elif keep_cues:
            out.append([_normalize_sentence_keep_cues(s, normalizer) for s in sentences])
        else:
            out.append(normalizer.normalize_batch(sentences, punc_norm=False))
    return out


def normalize_to_chunks(
    text: str,
    max_chars: int = 256,
    skip_normalize: bool = False,
) -> list[str]:
    """Split text into chunks <= max_chars, cutting at SENTENCE boundaries.

    Split sentences on RAW text (quote/bracket-aware) -> normalize each sentence ->
    pack by normalized length -> finalize end punctuation for each chunk. See
    :func:`_normalized_sentences_by_para` for why the order must be this way.

    Normalizing per sentence also keeps input to the normalizer's backtracking regexes at a
    safe level for DOCX-sized text.
    """
    from vieneu_utils.core_utils import pack_sentences_into_chunks

    if not text:
        return []

    chunks: list[str] = []
    for sentences in _normalized_sentences_by_para(text, skip_normalize=skip_normalize):
        chunks.extend(pack_sentences_into_chunks(sentences, max_chars=max_chars))
    return [punc_norm(c) for c in chunks]


def normalize_to_chunks_v3(
    text: str, max_chars: int = 256, min_chunk_chars: int = 20
) -> list[str]:
    """Chunk for the v3 path EXACTLY LIKE v2-gpu: cut by normalized TEXT length.

    The v3 path previously cut at the PHONEME level (``chunk_phonemes``), but phonemes are
    ~1.2-1.4x longer than text so the same ``max_chars`` produced more fragments than v2-gpu.
    This function cuts by text length like ``normalize_to_chunks`` (v2-gpu) to match chunk
    counts, WHILE preserving inline emotion cues (e.g. ``[cười]``/``<|emotion_k|>`` ->
    ``<|emotion_k|>``) that ``normalize_to_chunks`` would normally drop (``[...]`` markers
    are removed during normalization).

    Chunks shorter than ``min_chunk_chars`` are merged into an adjacent chunk (see
    :func:`_merge_short_chunks`) - 1-2 word chunks on their own make the AR model prone to
    hallucination.

    Returns a list of TEXT chunks (each chunk may contain ``<|emotion_k|>``); the caller
    phonemizes each chunk with :func:`phonemize_text_with_emotions`.
    """
    return normalize_to_chunks_v3_with_gaps(
        text, max_chars=max_chars, min_chunk_chars=min_chunk_chars
    )[0]


_EMOTION_TOKEN_RE = re.compile(r"<\|emotion_\d+\|>")


def _effective_len(chunk: str) -> int:
    """The "readable" length of a chunk: emotion tokens take up characters but are not words."""
    return len(_EMOTION_TOKEN_RE.sub("", chunk).strip())


def _merge_short_chunks(
    chunks: list[str], gaps: list[str], min_chars: int
) -> tuple[list[str], list[str]]:
    """Merge chunks shorter than ``min_chars`` into an adjacent chunk and update ``gaps`` accordingly.

    1-2 word chunks (often a heading line or short dialogue after splitting on newlines) standing
    alone leave the AR model with too little text context, so the stop token may be missed -> hallucination.
    Merge preference: prefer a boundary that is NOT a paragraph break, then the shorter neighbor, then the
    right side (attaching a heading to the content that follows sounds more natural). When forced to merge
    across a paragraph break, that boundary's ``para`` pause gives way to the sentence-break pause determined
    by punctuation - a deliberate trade-off to avoid hallucination.

    A merged chunk may exceed ``max_chars`` by at most ``min_chars + 1`` characters. A lone chunk (the
    entire text is too short) is kept as-is - the phoneme-based frame cap handles that case.
    """
    while len(chunks) > 1:
        short = [i for i, c in enumerate(chunks) if _effective_len(c) < min_chars]
        if not short:
            break
        i = min(short, key=lambda k: _effective_len(chunks[k]))
        sides = []                                   # (priority, side)
        if i < len(chunks) - 1:
            sides.append(((gaps[i] != "para", -len(chunks[i + 1])), "R"))
        if i > 0:
            sides.append(((gaps[i - 1] != "para", -len(chunks[i - 1])), "L"))
        side = max(sides, key=lambda t: t[0])[1]     # tie -> "R" (listed first)
        if side == "R":
            chunks[i] = chunks[i] + " " + chunks.pop(i + 1)
            gaps.pop(i)
        else:
            chunks[i - 1] = chunks[i - 1] + " " + chunks.pop(i)
            gaps.pop(i - 1)
    return chunks, gaps


def normalize_to_chunks_v3_with_gaps(
    text: str, max_chars: int = 256, min_chunk_chars: int = 20
) -> tuple[list[str], list[str]]:
    """Like :func:`normalize_to_chunks_v3` but also returns the boundary type BETWEEN chunks so
    audio can be joined with context-dependent pause lengths.

    Returns ``(chunks, gaps)`` where ``gaps[i] in {"para","sentence","minor"}`` is the
    boundary between ``chunks[i]`` and ``chunks[i+1]`` (``len(gaps) == len(chunks)-1``).
    Use with :func:`vieneu_utils.core_utils.gaps_to_silence` +
    ``join_audio_chunks(..., silence_ps=...)``.

    ``sentence``/``minor`` boundaries are reclassified on the ``punc_norm``-ed chunks
    (punc_norm may force end punctuation for short sentences) to match real audio intonation;
    ``para`` (paragraph break) is preserved - even on paths with emotion cues.
    """
    from vieneu_utils.core_utils import pack_sentences_into_chunks, _classify_gap

    if not text:
        return [], []

    keep_cues = "[" in text or "<|emotion_" in text
    chunks: list[str] = []
    gaps: list[str] = []
    for sentences in _normalized_sentences_by_para(text, keep_cues=keep_cues):
        para_chunks = pack_sentences_into_chunks(sentences, max_chars=max_chars)
        if not para_chunks:
            continue
        if chunks:                   # boundary with the PREVIOUS paragraph is a paragraph break
            gaps.append("para")
        for j, ch in enumerate(para_chunks):
            if j > 0:                # boundary within the SAME paragraph: reclassified later
                gaps.append("sentence")
            chunks.append(ch)

    chunks = [punc_norm(c) for c in chunks]
    gaps = [g if g == "para" else _classify_gap(chunks[i]) for i, g in enumerate(gaps)]
    # Merge AFTER punc_norm: each piece has already finalized end punctuation, so merged text
    # preserves intonation; gaps for the remaining boundaries are unchanged.
    return _merge_short_chunks(chunks, gaps, min_chunk_chars)


def phonemize_to_chunks(
    text: str,
    max_chars: int = 256,
    min_chunk_size: int = 10,
    source_max_chars: Optional[int] = None,
    skip_normalize: bool = False,
    phoneme_dict: dict = None,
):
    """
    Convert long raw text into bounded phoneme chunks.

    Order (chunk AFTER normalization): normalize per PARAGRAPH (punc_norm=True)
    -> phonemize -> split at the PHONEME level (``split_into_chunks_v2``). Because chunks
    are cut after normalization + phonemization, chunk length reflects the actual phoneme
    string and is always <= ``max_chars`` (no inflation from splitting before normalization).
    Every chunk always ends with valid terminal punctuation.

    Normalizing per paragraph (split by newline) keeps input to backtracking regexes at a
    safe level for large text. ``source_max_chars`` is kept for signature compatibility but
    is no longer used (length-based splitting is now done at the phoneme level).
    """
    from vieneu_utils.core_utils import split_into_chunks_v2

    if not text:
        return []

    if skip_normalize:
        normalized_units = [text]
    else:
        normalizer = _get_normalizer()
        paragraphs = [p for p in RE_NEWLINE_SPLIT.split(text) if p.strip()] or [text]
        normalized_units = normalizer.normalize_batch(paragraphs, punc_norm=True)

    phonemes = phonemize_batch(
        normalized_units,
        skip_normalize=True,
        phoneme_dict=phoneme_dict,
    )

    phone_chunks = []
    for chunk_phonemes in phonemes:
        phone_chunks.extend(
            split_into_chunks_v2(
                chunk_phonemes,
                max_chunk_size=max_chars,
                min_chunk_size=min_chunk_size,
            )
        )
    return phone_chunks


# ---------------------------------------------------------------------------
# CLI helper (python -m vieneu_utils.phonemize_text "some text")
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    test_text = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "SP500 price today is 4,200.5 points."
    )
    print(f"Output: {phonemize_text(test_text)}")