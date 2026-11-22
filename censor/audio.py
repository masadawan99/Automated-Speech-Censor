"""Streaming PyAV audio decoding (<=120 s hard cap) and WAV byte export."""

from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile

import av
import numpy as np
import soundfile as sf

MAX_DURATION_S: float = 120.0
ASR_SAMPLE_RATE: int = 16000
OUT_SAMPLE_RATE: int = 44100


class AudioTooLongError(ValueError):
    """Raised when input audio exceeds the maximum allowed duration."""


class AudioDecodeError(ValueError):
    """Raised when input audio bytes cannot be decoded into a valid audio stream."""


def _resample_frames(
    container: av.container.InputContainer,
    max_duration_s: float,
    asr_sr: int,
    out_sr: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        raise AudioDecodeError("No audio stream found in the provided file.")

    stream = audio_streams[0]

    # Early header check if duration metadata is well-formed and clearly exceeds cap
    if container.duration is not None and container.duration > 0:
        header_dur_s = float(container.duration) / float(av.time_base)
        if header_dur_s > max_duration_s + 2.0:
            raise AudioTooLongError(
                f"Audio duration ({header_dur_s:.1f} s) exceeds the {max_duration_s:.0f} s limit."
            )

    resampler_asr = av.AudioResampler(format="fltp", layout="mono", rate=asr_sr)
    resampler_out = av.AudioResampler(format="fltp", layout="mono", rate=out_sr)

    asr_chunks: list[np.ndarray] = []
    out_chunks: list[np.ndarray] = []
    max_out_samples = int(round(max_duration_s * out_sr))
    total_out_samples = 0

    def _collect(resampler: av.AudioResampler, frame: av.AudioFrame | None, chunks: list[np.ndarray]) -> int:
        added = 0
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            if arr.size > 0:
                chunks.append(arr)
                added += int(arr.size)
        return added

    for packet in container.demux(stream):
        for frame in packet.decode():
            _collect(resampler_asr, frame, asr_chunks)
            total_out_samples += _collect(resampler_out, frame, out_chunks)
            if total_out_samples > max_out_samples:
                approx_dur = total_out_samples / float(out_sr)
                raise AudioTooLongError(
                    f"Audio exceeds the {max_duration_s:.0f} s limit (>{approx_dur:.1f} s decoded)."
                )

    # Flush remaining buffered samples from both resamplers
    _collect(resampler_asr, None, asr_chunks)
    total_out_samples += _collect(resampler_out, None, out_chunks)
    if total_out_samples > max_out_samples:
        approx_dur = total_out_samples / float(out_sr)
        raise AudioTooLongError(
            f"Audio exceeds the {max_duration_s:.0f} s limit (>{approx_dur:.1f} s decoded)."
        )

    if not out_chunks or total_out_samples == 0:
        raise AudioDecodeError("Decoded audio contains zero samples.")

    asr_pcm = np.clip(np.concatenate(asr_chunks), -1.0, 1.0).astype(np.float32, copy=False)
    out_pcm = np.clip(np.concatenate(out_chunks), -1.0, 1.0).astype(np.float32, copy=False)
    duration_s = float(out_pcm.size) / float(out_sr)
    return asr_pcm, out_pcm, duration_s


def decode_audio_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    max_duration_s: float = MAX_DURATION_S,
    asr_sr: int = ASR_SAMPLE_RATE,
    out_sr: int = OUT_SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Decode raw audio bytes in memory via PyAV with a streaming duration cap.

    Falls back to a temporary file only if the container format requires a seekable
    filesystem descriptor, and always deletes that temporary file in a ``finally:`` block.

    Returns:
        ``(asr_pcm_16k, out_pcm_44k, out_sr, duration_s)``
    """
    if not audio_bytes:
        raise AudioDecodeError("Empty audio input.")

    try:
        with av.open(io.BytesIO(audio_bytes)) as container:
            asr_pcm, out_pcm, duration_s = _resample_frames(
                container, max_duration_s=max_duration_s, asr_sr=asr_sr, out_sr=out_sr
            )
            return asr_pcm, out_pcm, out_sr, duration_s
    except (AudioTooLongError, AudioDecodeError):
        raise
    except Exception as mem_exc:
        # Some MP4/M4A containers need a seekable file path; always clean up via finally:
        suffix = Path(filename).suffix or ".audio"
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            with av.open(tmp_path) as container:
                asr_pcm, out_pcm, duration_s = _resample_frames(
                    container, max_duration_s=max_duration_s, asr_sr=asr_sr, out_sr=out_sr
                )
                return asr_pcm, out_pcm, out_sr, duration_s
        except (AudioTooLongError, AudioDecodeError):
            raise
        except Exception as file_exc:
            raise AudioDecodeError(
                f"Could not decode audio '{filename}': {file_exc}"
            ) from mem_exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float32 waveform in ``[-1, 1]`` into 16-bit PCM WAV bytes."""
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
