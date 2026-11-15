"""
SRT subtitles for TTS audio - timestamps are measured from the ACTUAL audio, not estimated.

Approach: split the text into "reading units" (sentences, merging sentences that are too short);
the caller synthesizes each unit separately and measures its duration. The engine already
generates chunks independently and concatenates them, so reading by sentence does not change
quality. Short sentences (< ``MIN_UNIT_CHARS``) are merged for the same reason the engine does:
a 1-2 word chunk on its own makes the AR model prone to hallucination.

Subtitles display the user's ORIGINAL TEXT ("KTX", not the pronunciation-dictionary reading form)
and strip emotion tags. Long sentences are split into multiple cues (time is divided
proportionally by character count within that sentence) so each cue is at most 2 readable lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vieneu_utils.core_utils import V3_GAP_SILENCE, split_into_sentences
from vieneu_utils.pause_tags import split_pauses

MIN_UNIT_CHARS = 20
LINE_CHARS = 42          # common subtitle standard: ~42 characters per line
CUE_MAX_CHARS = 2 * LINE_CHARS

# App bracket tags, VieNeu emotion tokens, and Supertonic <laugh>/<breath>/<sigh> tags.
_EMOTION_TAG = re.compile(r"\[[^\]]*\]|<\|emotion_\d+\|>|<(?:laugh|breath|sigh)>", re.IGNORECASE)
_NEWLINES = re.compile(r"[\r\n]+")
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:…])")
# Preferred cue break points: after secondary punctuation.
_SOFT_BREAK = re.compile(r"(?<=[,;:])\s+")
# Sentence-ending punctuation that does not require a following space: Japanese/Chinese (。！？) and
# Hindi (।) — Kokoro voices. split_into_sentences only recognizes Latin punctuation, so a whole
# paragraph would otherwise become one cue. Closing brackets/quotes AFTER sentence punctuation still
# belong to that sentence: in 「…。」 cutting right after 。 would leave 」 as a separate sentence.
# At most 2 consecutive closing marks.
CLOSING_MARKS = "」』）)\"'”’》】"
_STOP = "[。！？।]"
_CLOSE = "[" + re.escape(CLOSING_MARKS) + "]"
NON_LATIN_STOP_PATTERN = "|".join(
    rf"(?<={_STOP}{_CLOSE * n})(?!{_CLOSE})\s*" for n in range(3))
_NON_LATIN_STOP = re.compile(NON_LATIN_STOP_PATTERN)
# Han/kana characters: roughly one character ~ one word, so count double when comparing to MIN_UNIT_CHARS.
_CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
# Japanese/Chinese punctuation is not followed by a space; merged sentences must not insert one either.
_CJK_PUNCT_END = ("。", "！", "？", "、", "，")


@dataclass
class Unit:
    """One TTS call. ``pause_after`` = seconds of pause after the unit.

    ``tag_pause`` False: natural sentence/paragraph break, ``pause_after`` is the MINIMUM total pause
    (silence already present at audio edges is included, see ``core_utils.pause_pad_samples``).
    True: pause tag, insert exactly ``pause_after`` seconds.
    """
    text: str
    pause_after: float
    tag_pause: bool = False


@dataclass
class Cue:
    start: float
    end: float
    text: str


def display_text(text: str) -> str:
    """Text shown in subtitles: remove emotion/unknown tags and collapse whitespace."""
    text = _SPACES.sub(" ", _EMOTION_TAG.sub(" ", text)).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)  # remove stray space before punctuation


def _effective_len(text: str) -> int:
    shown = display_text(text)
    return len(shown) + len(_CJK_CHAR.findall(shown))


def _join(a: str, b: str) -> str:
    return f"{a}{b}" if a.rstrip(CLOSING_MARKS).endswith(_CJK_PUNCT_END) else f"{a} {b}"


def build_units(text: str) -> list[Unit]:
    """Split text (which may contain pause tags like ``[nghỉ Ns]`` / ``[pause Ns]``) into reading units."""
    units: list[Unit] = []
    for seg, pause in split_pauses(text):
        seg_units: list[Unit] = []
        paragraphs = [p for p in _NEWLINES.split(seg) if p.strip()]
        for para in paragraphs:
            sentences = [
                s for piece in _NON_LATIN_STOP.split(para) if piece.strip()
                for s in (split_into_sentences(piece) or [piece.strip()])
            ] or [para.strip()]
            for i, sentence in enumerate(sentences):
                gap = V3_GAP_SILENCE["para"] if i == len(sentences) - 1 else V3_GAP_SILENCE["sentence"]
                prev = seg_units[-1] if seg_units else None
                # Merge into the previous unit if the previous unit is still too short.
                if prev is not None and _effective_len(prev.text) < MIN_UNIT_CHARS:
                    prev.text = _join(prev.text, sentence)
                    prev.pause_after = gap
                else:
                    seg_units.append(Unit(sentence, gap))
        # If the last unit is too short, merge it back into the previous unit (same pause-tag segment).
        if len(seg_units) > 1 and _effective_len(seg_units[-1].text) < MIN_UNIT_CHARS:
            last = seg_units.pop()
            seg_units[-1].text = _join(seg_units[-1].text, last.text)
            seg_units[-1].pause_after = last.pause_after
        if seg_units:
            seg_units[-1].pause_after = pause  # end of segment: pause for the tag's exact duration
            seg_units[-1].tag_pause = True
            units.extend(seg_units)
        elif units:
            units[-1].pause_after += pause     # pause tag with no preceding text
            units[-1].tag_pause = True
        elif pause > 0:
            units.append(Unit("", pause, tag_pause=True))  # pause tag at start of text: silence before first sentence
    # The final unit gets the last segment's pause: 0 if the text does not end with a pause tag.
    return units


def _wrap(text: str) -> str:
    """At most 2 lines, broken at the word boundary closest to the middle."""
    if len(text) <= LINE_CHARS:
        return text
    mid = len(text) // 2
    left = text.rfind(" ", 0, mid + 1)
    right = text.find(" ", mid)
    candidates = [i for i in (left, right) if i > 0]
    if not candidates:
        return text
    cut = min(candidates, key=lambda i: abs(i - mid))
    return f"{text[:cut].strip()}\n{text[cut:].strip()}"


def _split_long(text: str) -> list[str]:
    """Split a long sentence into parts <= CUE_MAX_CHARS: prefer breaks after commas, then by words."""
    if len(text) <= CUE_MAX_CHARS:
        return [text]
    pieces: list[str] = []
    buf = ""
    for part in _SOFT_BREAK.split(text):
        for word in [part] if len(part) <= CUE_MAX_CHARS else part.split(" "):
            cand = f"{buf} {word}".strip()
            if buf and len(cand) > CUE_MAX_CHARS:
                pieces.append(buf)
                buf = word
            else:
                buf = cand
    if buf:
        pieces.append(buf)
    return pieces


def unit_cues(text: str, start: float, end: float) -> list[Cue]:
    """Cues for a reading unit spanning [start, end]; long sentences are split proportionally by character count."""
    shown = display_text(text)
    if not shown:
        return []
    pieces = _split_long(shown)
    total = sum(len(p) for p in pieces)
    cues, t = [], start
    for p in pieces:
        dur = (end - start) * len(p) / total
        cues.append(Cue(t, t + dur, _wrap(p)))
        t += dur
    cues[-1].end = end
    return cues


def _ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [f"{i}\n{_ts(c.start)} --> {_ts(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)
