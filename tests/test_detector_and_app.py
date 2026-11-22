"""Unit tests for censor.detector and app.py pure formatting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import format_colored_transcript
from censor.detector import (
    LexiconOnlyDetector,
    OnnxDistilBertDetector,
    SklearnTfidfDetector,
    _softmax_pos_prob,
    load_detector,
)
from censor.pipeline import WordEvent


def test_load_detector_falls_back_to_lexicon_only(tmp_path: Path) -> None:
    det = load_detector(local_dir=tmp_path)
    assert isinstance(det, LexiconOnlyDetector)
    assert det.threshold == 0.5
    assert det.predict_proba(["hello", "dick"]) == [0.5, 0.5]


def test_load_detector_sklearn_from_local_models(tmp_path: Path) -> None:
    texts = [
        "you are a total dick and asshole",
        "stop being a stupid bitch",
        "move your lazy ass now",
        "I am taking a machine learning class",
        "Dick Grayson is a comic character",
        "we should assess the accuracy",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    pipe.fit(texts, labels)

    joblib.dump(pipe, tmp_path / "best_classical_pipeline.pkl")
    (tmp_path / "detector_config.json").write_text(
        json.dumps({"model": "test-lr", "threshold": 0.37}),
        encoding="utf-8",
    )

    det = load_detector(local_dir=tmp_path)
    assert isinstance(det, SklearnTfidfDetector)
    assert abs(det.threshold - 0.37) < 1e-6
    probs = det.predict_proba(["stupid dick", "machine learning class"])
    assert len(probs) == 2
    assert probs[0] > probs[1]


def test_format_colored_transcript_escaping_and_colors() -> None:
    events = [
        WordEvent("Hello", "hello", 0.0, 0.2, "CLEAN", 0.0, "Hello f***"),
        WordEvent("f***", "f", 0.3, 0.5, "TIER_A", 0.99, "Hello f*** class"),
        WordEvent("Dick", "dick", 0.6, 0.8, "TIER_B", 0.20, "Dick went home"),
        WordEvent("ass", "ass", 0.9, 1.1, "TIER_B", 0.85, "lazy ass"),
    ]
    md = format_colored_transcript(events, threshold=0.50)
    assert ":red-background[f\\*\\*\\* (0.99)]" in md
    assert ":orange-background[Dick (0.20)]" in md
    assert ":red-background[ass (0.85)]" in md
    assert "Hello" in md
    assert "<span" not in md


def test_onnx_detector_softmax_and_mock_session(tmp_path: Path) -> None:
    p2 = _softmax_pos_prob(np.array([[-2.0, 2.0], [2.0, -2.0]], dtype=np.float32))
    assert p2[0] > 0.95 and p2[1] < 0.05

    p1 = _softmax_pos_prob(np.array([[3.0], [-3.0]], dtype=np.float32))
    assert p1[0] > 0.90 and p1[1] < 0.10

    mock_input_ids = mock.MagicMock()
    mock_input_ids.name = "input_ids"
    mock_mask = mock.MagicMock()
    mock_mask.name = "attention_mask"

    mock_session = mock.MagicMock()
    mock_session.get_inputs.return_value = [mock_input_ids, mock_mask]
    mock_session.run.return_value = [np.array([[-1.5, 1.5], [1.5, -1.5]], dtype=np.float32)]

    mock_enc = mock.MagicMock()
    mock_enc.ids = [101, 2023, 102, 0]
    mock_enc.attention_mask = [1, 1, 1, 0]
    mock_tok = mock.MagicMock()
    mock_tok.encode_batch.return_value = [mock_enc, mock_enc]

    with (
        mock.patch("onnxruntime.InferenceSession", return_value=mock_session),
        mock.patch("tokenizers.Tokenizer.from_file", return_value=mock_tok),
    ):
        det = OnnxDistilBertDetector(
            onnx_path=tmp_path / "dummy.onnx",
            tokenizer_path=tmp_path / "tokenizer.json",
            threshold=0.42,
        )
        probs = det.predict_proba(["bad window", "clean window"])
        assert len(probs) == 2
        assert probs[0] > 0.90 and probs[1] < 0.10
        assert det.threshold == 0.42

