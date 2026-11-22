"""Unit tests for censor.beep timestamp sanitization, merging, and 1 kHz sine beeps."""

from __future__ import annotations

import numpy as np

from censor.beep import apply_beeps, merge_beep_intervals, sanitize_timestamp


def test_sanitize_timestamp_clamps_and_fixes_inverted() -> None:
    s, e = sanitize_timestamp(-0.5, 12.0, duration_s=5.0)
    assert s == 0.0 and e == 5.0

    # Zero-length or inverted span -> minimum 50 ms span
    s2, e2 = sanitize_timestamp(2.0, 2.0, duration_s=5.0, min_span_s=0.05)
    assert s2 == 2.0 and abs((e2 - s2) - 0.05) < 1e-6

    s3, e3 = sanitize_timestamp(2.5, 2.1, duration_s=5.0, min_span_s=0.05)
    assert s3 == 2.5 and abs((e3 - s3) - 0.05) < 1e-6

    # Near end of clip
    s4, e4 = sanitize_timestamp(4.99, 4.99, duration_s=5.0, min_span_s=0.05)
    assert 0.0 <= s4 < e4 <= 5.0


def test_merge_beep_intervals_padding_and_adjacent_gap() -> None:
    # Two words at [1.0, 1.2] and [1.43, 1.6]:
    # After +-0.10 s pad: [0.9, 1.3] and [1.33, 1.7] -> gap = 0.03 s <= 0.05 s -> merged!
    merged = merge_beep_intervals([(1.0, 1.2), (1.43, 1.6)], duration_s=3.0, pad_s=0.10, merge_gap_s=0.05)
    assert len(merged) == 1
    assert abs(merged[0][0] - 0.9) < 1e-6
    assert abs(merged[0][1] - 1.7) < 1e-6

    # Distant words stay separate
    merged2 = merge_beep_intervals([(0.5, 0.7), (2.0, 2.2)], duration_s=3.0, pad_s=0.10, merge_gap_s=0.05)
    assert len(merged2) == 2


def test_apply_beeps_frequency_amplitude_and_untouched_clean() -> None:
    sr = 16000
    duration_s = 2.0
    pcm = np.zeros(int(sr * duration_s), dtype=np.float32)
    pcm[: int(0.4 * sr)] = 0.25  # clean prefix

    out = apply_beeps(pcm, sample_rate=sr, intervals=[(0.5, 1.0)], freq_hz=1000.0, amplitude=0.3, fade_ms=5.0)

    # Clean prefix untouched
    assert np.allclose(out[: int(0.4 * sr)], 0.25)

    # Inside middle of beep span, peak amplitude matches ~0.3 and dominant FFT bin is 1000 Hz
    mid = out[int(0.6 * sr) : int(0.9 * sr)]
    assert 0.28 <= float(np.max(np.abs(mid))) <= 0.31

    freqs = np.fft.rfftfreq(len(mid), d=1.0 / sr)
    mag = np.abs(np.fft.rfft(mid))
    dominant_freq = float(freqs[int(np.argmax(mag))])
    assert abs(dominant_freq - 1000.0) < 10.0

    # Fade boundaries start/end smoothly near original signal (0.0)
    assert abs(float(out[int(0.5 * sr)])) < 0.02
