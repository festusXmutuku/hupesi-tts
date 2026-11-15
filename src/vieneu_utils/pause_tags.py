"""
Pause tag ``[nghỉ 2s]`` / ``[pause 2s]`` - insert a silence of an exact duration between two spoken segments.

The tag is not sent to the model: the longest pause the pipeline creates on its own is a 0.7s
paragraph break (see ``V3_GAP_SILENCE`` in core_utils), and the normalizer can also read an
unknown bracketed tag out loud (for example "[nghỉ 2s]"). Instead, the caller splits the text
at the tag, synthesizes each segment normally, and inserts the requested silence.

Syntax (case-insensitive, with or without Vietnamese diacritics):
    [nghỉ 2s]  [nghỉ 1.5s]  [nghỉ 1,5 giây]  [nghỉ 500ms]  [pause 2s]  [nghỉ]
No unit -> seconds; an empty ``[nghỉ]`` / ``[pause]`` -> ``DEFAULT_PAUSE_S``. A malformed tag
(e.g. ``[nghỉ abc]``) is removed (0s pause) so it is not spoken.
"""

from __future__ import annotations

import re

DEFAULT_PAUSE_S = 1.0
# Cap per tag: a typo like "[nghỉ 200s]" must not create minutes of silence.
MAX_PAUSE_S = 10.0

_TAG_RE = re.compile(r"\[\s*(?:nghỉ|nghi|pause)(?![^\W\d_])([^\]]*)\]", re.IGNORECASE)
_ARG_RE = re.compile(
    r"^\s*(\d+(?:[.,]\d+)?)\s*(ms|s|giây|giay|sec|seconds?)?\s*$", re.IGNORECASE
)


def parse_pause(arg: str) -> float:
    """Convert the argument after the pause keyword in the tag to seconds (capped). Invalid syntax -> 0."""
    if not arg.strip():
        return DEFAULT_PAUSE_S
    m = _ARG_RE.match(arg)
    if not m:
        return 0.0
    value = float(m.group(1).replace(",", "."))
    if (m.group(2) or "").lower() == "ms":
        value /= 1000.0
    return min(max(value, 0.0), MAX_PAUSE_S)


def has_pause_tags(text: str) -> bool:
    return bool(text) and _TAG_RE.search(text) is not None


def split_pauses(text: str) -> list[tuple[str, float]]:
    """Split ``text`` into ``[(segment, pause_after_segment_seconds), ...]``.

    A segment may be empty (tag at the start of the text, or two adjacent tags - in that case the
    seconds are accumulated into the same pause). No tags -> ``[(text, 0.0)]``.
    """
    out: list[tuple[str, float]] = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        seg = text[pos:m.start()].strip()
        pause = parse_pause(m.group(1))
        if not seg and out:
            prev_seg, prev_pause = out[-1]
            out[-1] = (prev_seg, prev_pause + pause)
        else:
            out.append((seg, pause))
        pos = m.end()
    tail = text[pos:].strip()
    if tail or not out:
        out.append((tail, 0.0))
    return out
