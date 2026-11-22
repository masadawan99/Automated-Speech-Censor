"""Automated Speech Censor core package."""

from censor.audio import (
    ASR_SAMPLE_RATE,
    MAX_DURATION_S,
    OUT_SAMPLE_RATE,
    AudioDecodeError,
    AudioTooLongError,
    decode_audio_bytes,
    pcm_to_wav_bytes,
)
from censor.beep import apply_beeps, merge_beep_intervals, sanitize_timestamp
from censor.detector import (
    BaseDetector,
    LexiconOnlyDetector,
    OnnxDistilBertDetector,
    SklearnTfidfDetector,
    load_detector,
)
from censor.lexicon import ProfanityLexicon, normalize_token
from censor.pipeline import (
    AnalysisResult,
    WordEvent,
    analyze_audio,
    extract_context_window,
    render_censored_audio,
)

__all__ = [
    "ASR_SAMPLE_RATE",
    "MAX_DURATION_S",
    "OUT_SAMPLE_RATE",
    "AnalysisResult",
    "AudioDecodeError",
    "AudioTooLongError",
    "BaseDetector",
    "LexiconOnlyDetector",
    "OnnxDistilBertDetector",
    "ProfanityLexicon",
    "SklearnTfidfDetector",
    "WordEvent",
    "analyze_audio",
    "apply_beeps",
    "decode_audio_bytes",
    "extract_context_window",
    "load_detector",
    "merge_beep_intervals",
    "normalize_token",
    "pcm_to_wav_bytes",
    "render_censored_audio",
    "sanitize_timestamp",
]
