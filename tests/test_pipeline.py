"""End-to-end tests for censor.pipeline using mock ASR & Detector."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from censor.audio import AudioTooLongError
from censor.detector import BaseDetector
from censor.lexicon import ProfanityLexicon
from censor.pipeline import analyze_audio, render_censored_audio


@dataclass
class _MockWord:
    word: str
    start: float
    end: float


@dataclass
class _MockSegment:
    words: list[_MockWord]


class _MockASR:
    def __init__(self, words: list[_MockWord]) -> None:
        self.words = words
        self.calls = 0

    def transcribe(self, audio, language="en", word_timestamps=True, vad_filter=True):
        self.calls += 1
        return [_MockSegment(words=self.words)], {"language": language}


class _MockDetector(BaseDetector):
    def __init__(self, scores: dict[str, float], threshold: float = 0.45) -> None:
        self.scores = scores
        self.threshold = threshold
        self.model_name = "mock-detector"
        self.backend = "mock"
        self.seen_windows: list[str] = []

    def predict_proba(self, texts: list[str]) -> list[float]:
        self.seen_windows.extend(texts)
        out = []
        for t in texts:
            low = t.lower()
            if "profane" in low:
                out.append(self.scores.get("profane", 0.85))
            else:
                out.append(self.scores.get("benign", 0.15))
        return out


def _make_wav(duration_s: float = 4.0, sr: int = 16000) -> bytes:
    pcm = np.zeros(int(sr * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture(scope="module")
def lexicon() -> ProfanityLexicon:
    csv_path = Path(__file__).resolve().parent.parent / "lexicon" / "profanity_lexicon.csv"
    return ProfanityLexicon.from_csv(csv_path)


def test_pipeline_tier_a_always_beeped_clean_never_and_slider_rerender(
    lexicon: ProfanityLexicon,
) -> None:
    words = [
        _MockWord("Hello", 0.1, 0.2),
        _MockWord("class", 0.3, 0.4),          # CLEAN (Scunthorpe)
        _MockWord("hell", 0.5, 0.6),           # CLEAN (mild exclusion)
        _MockWord("fuck", 0.7, 0.8),           # TIER_A (skips classifier)
        _MockWord("balls", 0.9, 1.0),          # TIER_B benign (window has no "profane")
        _MockWord("went", 1.1, 1.2),
        _MockWord("to", 1.3, 1.4),
        _MockWord("the", 1.5, 1.6),
        _MockWord("park", 1.7, 1.8),
        _MockWord("yesterday", 1.9, 2.0),
        _MockWord("afternoon", 2.1, 2.2),
        _MockWord("profane", 2.3, 2.4),        # >5 words away from balls, right next to prick
        _MockWord("prick", 2.5, 2.7),          # TIER_B profane (window contains "profane")
    ]
    asr = _MockASR(words)
    detector = _MockDetector(scores={"profane": 0.82, "benign": 0.18}, threshold=0.45)

    analysis = analyze_audio(
        audio_bytes=_make_wav(4.0),
        filename="sample.wav",
        asr_model=asr,
        detector=detector,
        lexicon=lexicon,
    )

    assert asr.calls == 1
    # Classifier must ONLY be called for the 2 Tier B tokens, never for Tier A ('fuck') or CLEAN
    assert len(detector.seen_windows) == 2

    ev_by_word = {w.word: w for w in analysis.words}
    assert ev_by_word["fuck"].tier == "TIER_A" and ev_by_word["fuck"].conf == 0.99
    assert ev_by_word["class"].tier == "CLEAN" and ev_by_word["class"].conf == 0.0
    assert ev_by_word["hell"].tier == "CLEAN" and ev_by_word["hell"].conf == 0.0
    assert "profane" not in ev_by_word["balls"].window
    assert ev_by_word["balls"].tier == "TIER_B" and abs(ev_by_word["balls"].conf - 0.18) < 1e-6
    assert ev_by_word["prick"].tier == "TIER_B" and abs(ev_by_word["prick"].conf - 0.82) < 1e-6

    # Default threshold 0.45 -> fuck (TIER_A) and prick (0.82 >= 0.45) beeped; balls (0.18) not beeped
    wav_default, rows_default = render_censored_audio(analysis, threshold=detector.threshold)
    assert wav_default[:4] == b"RIFF"
    by_word_def = {r["word"]: r for r in rows_default}
    assert by_word_def["fuck"]["beeped"] is True
    assert by_word_def["prick"]["beeped"] is True
    assert by_word_def["balls"]["beeped"] is False
    assert "class" not in by_word_def and "hell" not in by_word_def

    # Slider = 1.0 -> Tier A ('fuck') STILL beeped; Tier B ('prick', 0.82) un-beeped; ASR not re-called
    _, rows_max = render_censored_audio(analysis, threshold=1.0)
    by_word_max = {r["word"]: r for r in rows_max}
    assert by_word_max["fuck"]["beeped"] is True
    assert by_word_max["prick"]["beeped"] is False
    assert asr.calls == 1

    # Slider = 0.0 -> Tier B ('balls') beeped, CLEAN ('class', 'hell') STILL never beeped
    _, rows_min = render_censored_audio(analysis, threshold=0.0)
    by_word_min = {r["word"]: r for r in rows_min}
    assert by_word_min["balls"]["beeped"] is True
    assert "class" not in by_word_min and "hell" not in by_word_min


def test_pipeline_enforces_max_duration(lexicon: ProfanityLexicon) -> None:
    asr = _MockASR([])
    detector = _MockDetector(scores={})
    with pytest.raises(AudioTooLongError):
        analyze_audio(
            audio_bytes=_make_wav(3.0),
            filename="long.wav",
            asr_model=asr,
            detector=detector,
            lexicon=lexicon,
            max_duration_s=1.5,
        )
    assert asr.calls == 0
