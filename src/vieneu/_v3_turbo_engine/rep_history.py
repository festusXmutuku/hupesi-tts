"""Sliding-window repetition-penalty history, shared by every v3 Turbo path.

Why a sliding window instead of an accumulating set: each codebook has only 1024 codes, so after
a few hundred frames an unbounded set penalizes nearly half the codebook — including valid codes
that must legitimately repeat (silence, sustained vowels) — causing the voice to drift toward the
end of long chunks. The real goal of the penalty is to break LOCAL loops, so only codes appearing
within the most recent ``window`` frames are penalized.

Each channel (codebook) receives exactly 1 code per frame, so a ``window`` measured in frames is
also the maximum number of codes kept per channel. ``window <= 0`` preserves the old behavior
(permanent/unbounded penalty).
"""
from __future__ import annotations

from collections import Counter, deque

# ~2.5 s of audio at 25 frame/s: long enough to catch any local loop, short enough that a
# finished vowel/silence is no longer unfairly penalized.
DEFAULT_REP_WINDOW = 64


class _ChannelWindow:
    """History for one codebook. Externally it behaves like ``set`` (iter/len/bool/add) so existing
    sampling functions work unchanged; internally it is a sliding window."""

    __slots__ = ("_seen", "_order", "_window")

    def __init__(self, window: int):
        self._seen: Counter = Counter()   # code -> occurrences within the window
        self._order: deque = deque()      # codes in generation order, for FIFO eviction
        self._window = int(window)

    def add(self, code: int) -> None:
        self._seen[code] += 1
        if self._window > 0:
            self._order.append(code)
            if len(self._order) > self._window:
                old = self._order.popleft()
                if self._seen[old] <= 1:
                    del self._seen[old]
                else:
                    self._seen[old] -= 1

    def __iter__(self):
        return iter(self._seen)

    def __len__(self) -> int:
        return len(self._seen)

    def __bool__(self) -> bool:
        return bool(self._seen)


class RepetitionHistory:
    """Per-codebook sliding-window history: ``hist[ch]`` is the view for channel ``ch``.

    Drop-in replacement for ``[set() for _ in range(n_vq)]`` at all existing call sites.
    """

    __slots__ = ("channels",)

    def __init__(self, n_channels: int, window: int = DEFAULT_REP_WINDOW):
        self.channels = [_ChannelWindow(window) for _ in range(n_channels)]

    def __getitem__(self, ch: int) -> _ChannelWindow:
        return self.channels[ch]

    def __len__(self) -> int:
        return len(self.channels)
