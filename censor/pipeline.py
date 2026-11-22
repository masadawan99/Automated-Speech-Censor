"""End-to-end audio analysis and beep rendering pipeline (Spec Section 10 contract)."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import time
from typing import Any

import numpy as np

from censor.audio import MAX_DURATION_S, decode_audio_bytes, pcm_to_wav_bytes
from censor.beep import apply_beeps, merge_beep_intervals, sanitize_timestamp
from censor.detector import BaseDetector
from censor.lexicon import ProfanityLexicon


@dataclass
class WordEvent:
    word: str      # Original ASR token (stripped)
    norm: str      # Lowercased alphanumeric normalized token
    start: float   # Sanitized start time (seconds, clamped)
    end: float     # Sanitized end time (seconds, end > start)
    tier: str      # "CLEAN" | "TIER_A" | "TIER_B"
    conf: float    # 0.0 (CLEAN), 0.99 (TIER_A, display only), or classifier p in [0, 1] (TIER_B)
    window: str    # +-5-word context snippet (for debugging/UI)


@dataclass
class AnalysisResult:
    audio_hash: str
    sample_rate: int
    audio_pcm: np.ndarray
    duration_s: float
    asr_time_s: float
    clf_time_s: float
    words: list[WordEvent]


def extract_context_window(words: list[str], idx: int, radius: int = 5) -> str:
    """Return a ``+-radius`` word context window centered on ``words[idx]``."""
    lo = max(0, idx - radius)
    hi = min(len(words), idx + radius + 1)
    return " ".join(words[lo:hi])


def _transcribe_words(asr_model: Any, asr_pcm: np.ndarray) -> list[tuple[str, float, float]]:
    """Run ``faster-whisper`` (or compatible mock) and extract ``(word, start, end)`` tuples."""
    segments, _info = asr_model.transcribe(
        asr_pcm,
        language="en",
        word_timestamps=True,
        vad_filter=True,
    )
    raw_words: list[tuple[str, float, float]] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        for w in seg_words:
            text = str(getattr(w, "word", "")).strip()
            if not text:
                continue
            raw_words.append((text, float(w.start), float(w.end)))
    return raw_words


def analyze_audio(
    audio_bytes: bytes,
    filename: str,
    asr_model: Any,
    detector: BaseDetector,
    lexicon: ProfanityLexicon,
    max_duration_s: float = MAX_DURATION_S,
) -> AnalysisResult:
    """Decode audio, run Whisper ASR with word timestamps, and score Tier B candidates."""
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    asr_pcm, out_pcm, out_sr, duration_s = decode_audio_bytes(
        audio_bytes,
        filename=filename,
        max_duration_s=max_duration_s,
    )

    t0_asr = time.perf_counter()
    try:
        raw_words = _transcribe_words(asr_model, asr_pcm)
    finally:
        del asr_pcm
        gc.collect()
    asr_time_s = time.perf_counter() - t0_asr

    tokens = [w[0] for w in raw_words]
    events: list[WordEvent] = []
    tier_b_indices: list[int] = []
    tier_b_windows: list[str] = []

    for idx, (word_str, raw_start, raw_end) in enumerate(raw_words):
        s_clean, e_clean = sanitize_timestamp(raw_start, raw_end, duration_s=duration_s)
        norm, tier = lexicon.classify_word(word_str)
        window = extract_context_window(tokens, idx, radius=5)

        if tier == "TIER_A":
            conf = 0.99
        elif tier == "TIER_B":
            conf = 0.0  # populated in batch below
            tier_b_indices.append(idx)
            tier_b_windows.append(window)
        else:
            conf = 0.0

        events.append(
            WordEvent(
                word=word_str,
                norm=norm,
                start=s_clean,
                end=e_clean,
                tier=tier,
                conf=conf,
                window=window,
            )
        )

    t0_clf = time.perf_counter()
    if tier_b_windows:
        probs = detector.predict_proba(tier_b_windows)
        for ev_idx, prob in zip(tier_b_indices, probs):
            events[ev_idx].conf = float(min(1.0, max(0.0, prob)))
    clf_time_s = time.perf_counter() - t0_clf

    return AnalysisResult(
        audio_hash=audio_hash,
        sample_rate=out_sr,
        audio_pcm=out_pcm,
        duration_s=duration_s,
        asr_time_s=asr_time_s,
        clf_time_s=clf_time_s,
        words=events,
    )


def render_censored_audio(
    analysis: AnalysisResult,
    threshold: float,
) -> tuple[bytes, list[dict[str, Any]]]:
    """Apply threshold rule to pre-computed word events and synthesize beeped WAV bytes.

    Rule: ``beeped = (w.tier == "TIER_A") or (w.tier == "TIER_B" and w.conf >= threshold)``.
    """
    flagged_rows: list[dict[str, Any]] = []
    beep_spans: list[tuple[float, float]] = []

    for w in analysis.words:
        if w.tier == "CLEAN":
            continue
        beeped = (w.tier == "TIER_A") or (w.tier == "TIER_B" and w.conf >= threshold)
        if beeped:
            beep_spans.append((w.start, w.end))
        flagged_rows.append(
            {
                "word": w.word,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "tier": w.tier,
                "conf": round(w.conf, 4),
                "beeped": beeped,
            }
        )

    merged_spans = merge_beep_intervals(
        beep_spans,
        duration_s=analysis.duration_s,
        pad_s=0.10,
        merge_gap_s=0.05,
    )
    censored_pcm = apply_beeps(
        analysis.audio_pcm,
        sample_rate=analysis.sample_rate,
        intervals=merged_spans,
        freq_hz=1000.0,
        amplitude=0.3,
        fade_ms=5.0,
    )
    wav_bytes = pcm_to_wav_bytes(censored_pcm, sample_rate=analysis.sample_rate)
    return wav_bytes, flagged_rows
