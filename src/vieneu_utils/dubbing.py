"""
Align synthesized speech to subtitle timing (SRT-based dubbing).

Each sentence is spoken at the user-selected speed. If a sentence is longer than its slot,
it is sped up to fit, up to ``MAX_FIT_RATE`` times; if it is still too long, it spills into
the following silence and, if it reaches the next sentence, pushes that sentence back. Text
is never cut: cutting loses content, while a shift of a few hundred milliseconds is usually
imperceptible to viewers.

Pure computation with no engine dependency, so it can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vieneu_utils.subtitles import Cue

MAX_FIT_RATE = 1.5
# Minimum gap between two sentences when the previous one overruns into the next.
MIN_GAP_SECONDS = 0.05
# Shifts below this threshold are not counted as "pushed" (sample-rounding tolerance).
PUSH_EPSILON = 0.005


def fit_rate(audio_seconds: float, slot_seconds: float, max_rate: float = MAX_FIT_RATE) -> float:
    """Speed-up factor needed for ``audio_seconds`` to fit ``slot_seconds`` (1.0 if it already fits)."""
    if audio_seconds <= 0 or audio_seconds <= slot_seconds:
        return 1.0
    if slot_seconds <= 0:
        return max_rate
    return min(max_rate, audio_seconds / slot_seconds)


def place(cue_start: float, cursor: float, has_prev: bool) -> float:
    """Actual start time: the subtitle start, unless the previous sentence is still playing."""
    if not has_prev:
        return max(cue_start, cursor)
    return max(cue_start, cursor + MIN_GAP_SECONDS)


@dataclass
class DubStats:
    cues: int = 0
    sped_up: int = 0
    pushed: int = 0
    max_push_seconds: float = 0.0
    placed: list[Cue] = field(default_factory=list)

    def record(self, original: Cue, start: float, audio_seconds: float, rate: float) -> None:
        self.cues += 1
        if rate > 1.0:
            self.sped_up += 1
        push = start - original.start
        if push > PUSH_EPSILON:
            self.pushed += 1
            self.max_push_seconds = max(self.max_push_seconds, push)
        # Adjusted subtitle: shown for at least its original duration and long enough for the spoken sentence.
        end = start + max(audio_seconds, original.end - original.start)
        self.placed.append(Cue(start, end, original.text))

    def finish(self) -> list[Cue]:
        """Adjusted subtitle timing; cues do not overlap the next cue."""
        for cur, nxt in zip(self.placed, self.placed[1:]):
            cur.end = min(cur.end, nxt.start)
        return self.placed
