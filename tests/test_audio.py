"""Unit tests for censor.audio PyAV streaming decode and 120 s cap."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from censor.audio import AudioDecodeError, AudioTooLongError, decode_audio_bytes, pcm_to_wav_bytes


def _make_wav_bytes(duration_s: float, sr: int = 16000, freq_hz: float = 440.0) -> bytes:
    t = np.arange(int(round(duration_s * sr)), dtype=np.float32) / float(sr)
    pcm = (0.2 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_wav_roundtrip() -> None:
    wav_bytes = _make_wav_bytes(1.5, sr=22050)
    asr_pcm, out_pcm, out_sr, dur = decode_audio_bytes(wav_bytes, filename="test.wav")

    assert out_sr == 44100
    assert abs(dur - 1.5) < 0.05
    assert asr_pcm.dtype == np.float32
    assert out_pcm.dtype == np.float32
    assert abs(len(asr_pcm) / 16000.0 - 1.5) < 0.05

    reencoded = pcm_to_wav_bytes(out_pcm, out_sr)
    assert reencoded[:4] == b"RIFF"


def test_decode_rejects_audio_over_120s() -> None:
    wav_bytes = _make_wav_bytes(3.0, sr=8000)
    with pytest.raises(AudioTooLongError):
        decode_audio_bytes(wav_bytes, filename="too_long.wav", max_duration_s=2.0)


def test_decode_rejects_empty_and_corrupt_bytes() -> None:
    with pytest.raises(AudioDecodeError):
        decode_audio_bytes(b"", filename="empty.wav")

    with pytest.raises(AudioDecodeError):
        decode_audio_bytes(b"not a valid audio container at all", filename="bad.wav")
