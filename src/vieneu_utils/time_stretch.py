"""
Change speech speed while preserving pitch using WSOLA (Waveform Similarity Overlap-Add).

This previously used ``librosa.effects.time_stretch`` (phase vocoder), which reconstructs
the phase of each frequency band and can make speech sound metallic/robotic, increasingly
so at higher speeds (reported 2026-10). WSOLA cuts and splices real waveform segments from
the original audio, shifting each segment slightly to best match the previous one, so the
voice stays more natural around 1.2-1.5x. It only requires NumPy, so it does not pull
librosa/torch into the CPU path.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 30.0      # frame length: enough to cover several pitch periods of speech
TOLERANCE_MS = 12.0  # maximum shift when searching for the best waveform match
SEARCH_RATE = 12000  # search for alignment on a downsampled signal, then refine for speed


def _best_offset(region: np.ndarray, target: np.ndarray, step: int) -> int:
    """Return the position in ``region`` where a window the length of ``target`` best matches ``target``."""
    coarse = np.correlate(region[::step], target[::step], mode="valid")
    best = int(np.argmax(coarse)) * step
    if step == 1:
        return best
    lo = max(0, best - step)
    hi = min(len(region) - len(target), best + step)
    fine = np.correlate(region[lo:hi + len(target)], target, mode="valid")
    return lo + int(np.argmax(fine))


def time_stretch(audio: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """Speed up by ``rate`` (``rate`` < 1 slows down); output length is approximately ``len / rate``."""
    x = np.asarray(audio, dtype=np.float32)
    if abs(rate - 1.0) < 0.02 or len(x) == 0:
        return x
    n = max(64, int(sr * FRAME_MS / 1000)) & ~1
    hop = n // 2
    tol = max(1, int(sr * TOLERANCE_MS / 1000))
    step = max(1, sr // SEARCH_RATE)
    # Periodic Hann window: 50% overlap sums to 1.
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)

    out_len = int(round(len(x) / rate))
    frames = out_len // hop + 1
    pad = tol + hop
    xp = np.concatenate([np.zeros(pad, np.float32), x,
                         np.zeros(2 * n + tol + int(np.ceil(hop * rate)), np.float32)])
    y = np.zeros(frames * hop + n, dtype=np.float32)

    prev = pad
    for k in range(frames):
        nominal = pad + int(round(k * hop * rate))
        if k == 0:
            pos = nominal
        else:
            # "Natural continuation" of the previous frame: the new frame should best match it.
            target = xp[prev + hop: prev + hop + n]
            lo = max(0, nominal - tol)
            hi = min(len(xp) - n, nominal + tol)
            pos = lo + _best_offset(xp[lo:hi + n], target, step)
        y[k * hop: k * hop + n] += xp[pos: pos + n] * win
        prev = pos
    return y[:out_len]
