"""Timestamp sanitization, interval merging, and 1 kHz sine beep synthesis."""

from __future__ import annotations

import numpy as np


def sanitize_timestamp(
    start: float,
    end: float,
    duration_s: float,
    min_span_s: float = 0.05,
) -> tuple[float, float]:
    """Clamp word timestamps to ``[0, duration_s]`` and enforce a minimum span."""
    if duration_s <= 0.0:
        return 0.0, 0.0

    s = max(0.0, min(float(start), duration_s))
    e = max(0.0, min(float(end), duration_s))

    if e <= s:
        e = min(duration_s, s + min_span_s)
        if e <= s:
            s = max(0.0, duration_s - min_span_s)
            e = duration_s
    elif (e - s) < min_span_s and duration_s >= min_span_s:
        e = min(duration_s, s + min_span_s)
        if (e - s) < min_span_s:
            s = max(0.0, e - min_span_s)

    return float(s), float(e)


def merge_beep_intervals(
    intervals: list[tuple[float, float]],
    duration_s: float,
    pad_s: float = 0.10,
    merge_gap_s: float = 0.05,
) -> list[tuple[float, float]]:
    """Pad each interval by ``+-pad_s``, clamp to ``[0, duration_s]``, and merge close spans."""
    if not intervals or duration_s <= 0.0:
        return []

    padded: list[tuple[float, float]] = []
    for start, end in intervals:
        s = max(0.0, float(start) - pad_s)
        e = min(duration_s, float(end) + pad_s)
        if e > s:
            padded.append((s, e))

    if not padded:
        return []

    padded.sort(key=lambda iv: (iv[0], iv[1]))
    merged: list[tuple[float, float]] = [padded[0]]

    for cur_s, cur_e in padded[1:]:
        prev_s, prev_e = merged[-1]
        if cur_s - prev_e <= merge_gap_s:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))

    return merged


def apply_beeps(
    pcm: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    freq_hz: float = 1000.0,
    amplitude: float = 0.3,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Replace each ``(start_s, end_s)`` span with a 1 kHz sine beep and 5 ms fades."""
    out = np.array(pcm, dtype=np.float32, copy=True)
    n_samples = out.shape[0]
    if n_samples == 0 or not intervals or sample_rate <= 0:
        return out

    fade_samples_default = max(1, int(round((fade_ms / 1000.0) * sample_rate)))

    for start_s, end_s in intervals:
        s_idx = max(0, min(n_samples, int(round(start_s * sample_rate))))
        e_idx = max(0, min(n_samples, int(round(end_s * sample_rate))))
        span_len = e_idx - s_idx
        if span_len <= 0:
            continue

        t = np.arange(span_len, dtype=np.float32) / float(sample_rate)
        beep = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

        fade_len = min(fade_samples_default, span_len // 2)
        if fade_len > 0:
            ramp_in = np.linspace(0.0, 1.0, fade_len, endpoint=False, dtype=np.float32)
            ramp_out = np.linspace(1.0, 0.0, fade_len, endpoint=False, dtype=np.float32)

            orig_head = out[s_idx : s_idx + fade_len]
            orig_tail = out[e_idx - fade_len : e_idx]

            out[s_idx : s_idx + fade_len] = orig_head * (1.0 - ramp_in) + beep[:fade_len] * ramp_in
            if span_len > 2 * fade_len:
                out[s_idx + fade_len : e_idx - fade_len] = beep[fade_len : span_len - fade_len]
            out[e_idx - fade_len : e_idx] = orig_tail * (1.0 - ramp_out) + beep[-fade_len:] * ramp_out
        else:
            out[s_idx:e_idx] = beep

    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
