# Nano audio helpers adapted from pnnbao97/VieNeu-TTS, Apache-2.0.
# Revision: d8ca4f91b9abb6be82ad1d8b1fdf0841e4b4ed29
from typing import List, Optional, Tuple
import numpy as np

V3_GAP_SILENCE = {"para": 0.55, "sentence": 0.32, "minor": 0.14}

EDGE_THRESH_DB = -45.0

EDGE_KEEP_S = 0.04

EDGE_FADE_S = 0.015

def gaps_to_silence(gaps: List[str]) -> List[float]:
    """Map a list of boundary types to a list of TOTAL pause durations (seconds) for ``join_audio_chunks``."""
    return [V3_GAP_SILENCE.get(g, V3_GAP_SILENCE["sentence"]) for g in gaps]

def edge_silence(
    wav: np.ndarray, sr: int, thresh_db: float = EDGE_THRESH_DB, win_s: float = 0.01
) -> Tuple[int, int]:
    """``(lead, tail)``: number of silent samples at the start and end of ``wav`` (mean |x| envelope
    over a ``win_s`` window; below ``thresh_db`` is silence). All-silent wav -> ``(len, 0)``."""
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
    """Trim model-generated silence at both ends (keeping ``keep_s`` on each side), then apply a
    ``fade_s`` cosine fade at both edges so joins do not click and pauses are controlled only by
    ``join_audio_chunks``. Returns a copy; an empty wav is returned unchanged."""
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
    (trailing silence of previous + zeros + leading silence of next) reaches ``pause_s``; 0 if already enough.
    Used for streaming, where the previous chunk has already been emitted so it cannot be trimmed
    (if the final segment is entirely silent, the whole segment counts as trailing silence)."""
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
    """Concatenate audio chunks.

    ``silence_ps`` (optional, v3 path): ``silence_ps[i]`` is the desired TOTAL pause (seconds)
    between chunk ``i`` and ``i+1`` based on boundary type. Each chunk is :func:`trim_and_fade`'d
    (trimming model-generated silence, keeping 40 ms, fading edges) and just enough zeros are inserted
    so trailing-silence + zeros + leading-silence = ``silence_ps[i]`` — making pauses independent of
    the voice (presets may emit a long tail, clones often almost none). When ``silence_ps`` is
    provided, ``silence_p``/``crossfade_p`` are ignored; missing gap values default to 0 (trim + join).

    Without ``silence_ps`` (v1/v2 path): insert ``silence_p`` seconds of zeros, or crossfade
    ``crossfade_p`` seconds, or join directly — behavior unchanged.
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    if silence_ps is not None:
        trimmed = [trim_and_fade(c, sr) for c in chunks]
        parts: List[np.ndarray] = [trimmed[0]]
        for i in range(1, len(trimmed)):
            pause_s = silence_ps[i - 1] if i - 1 < len(silence_ps) else 0.0
            pad = pause_pad_samples(trimmed[i - 1], trimmed[i], sr, pause_s)
            if pad > 0:
                parts.append(np.zeros(pad, dtype=np.float32))
            parts.append(trimmed[i])
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    if len(chunks) == 1:
        return chunks[0]

    silence_samples   = int(sr * silence_p)
    crossfade_samples = int(sr * crossfade_p)
    final_wav = chunks[0]

    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        if silence_samples > 0:
            silence   = np.zeros(silence_samples, dtype=np.float32)
            final_wav = np.concatenate([final_wav, silence, next_chunk])
        elif crossfade_samples > 0:
            overlap = min(len(final_wav), len(next_chunk), crossfade_samples)
            if overlap > 0:
                fade_out  = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                fade_in   = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                blended   = final_wav[-overlap:] * fade_out + next_chunk[:overlap] * fade_in
                final_wav = np.concatenate([final_wav[:-overlap], blended, next_chunk[overlap:]])
            else:
                final_wav = np.concatenate([final_wav, next_chunk])
        else:
            final_wav = np.concatenate([final_wav, next_chunk])

    return final_wav
