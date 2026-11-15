"""
Read SRT (and simple WebVTT) subtitle files for time-aligned dubbing.

Handles common file variations: BOM, CRLF line endings, timestamps using ``,`` or ``.``,
missing cue numbers, multi-line text, formatting tags like ``<i>``/``<font>`` and ``{\\an8}``.
Returned cues are sorted by start time; cues with no text left after tag removal are skipped.
"""

from __future__ import annotations

import re

from vieneu_utils.subtitles import Cue

_TIME = r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_TIMING = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")
_HTML_TAG = re.compile(r"<[^>]*>")
_ASS_TAG = re.compile(r"\{[^}]*\}")
_MUSIC = re.compile(r"[♪♫]")
_SPACES = re.compile(r"\s+")
_BLOCK_SPLIT = re.compile(r"\n\s*\n")


class SrtError(ValueError):
    """The file contains no valid cues."""


def _seconds(h, m, s, ms) -> float:
    # "5" means 500 ms, "05" means 50 ms: the field is always the fractional part of a second.
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def clean_text(text: str) -> str:
    """Text to speak/display: remove formatting tags and music symbols, and collapse lines into one."""
    text = _MUSIC.sub(" ", _ASS_TAG.sub(" ", _HTML_TAG.sub(" ", text)))
    return _SPACES.sub(" ", text).strip()


def parse_srt(content: str) -> list[Cue]:
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in _BLOCK_SPLIT.split(content):
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            m = _TIMING.match(line)
            if m:
                start = _seconds(*m.groups()[:4])
                end = _seconds(*m.groups()[4:])
                text = clean_text(" ".join(lines[i + 1:]))
                if text:
                    cues.append(Cue(start, max(start, end), text))
                break
    if not cues:
        raise SrtError("The subtitle file is not a valid SRT or contains no dialogue.")
    cues.sort(key=lambda c: c.start)
    return cues
