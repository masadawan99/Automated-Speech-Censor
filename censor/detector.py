"""Context profanity detector backends (ONNX INT8 DistilBERT, joblib TF-IDF, lexicon fallback).

Zero PyTorch dependency in runtime inference so Streamlit Cloud stays lean and OOM-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _softmax_pos_prob(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return _sigmoid(logits)
    if logits.shape[1] == 1:
        return _sigmoid(logits[:, 0])
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    return probs[:, 1]


class BaseDetector(ABC):
    """Abstract profanity context classifier returning probabilities in ``[0, 1]``."""

    threshold: float = 0.5
    model_name: str = "base"
    backend: str = "base"

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> list[float]:
        """Return ``P(profane/obscene = 1)`` for each context window string."""


class LexiconOnlyDetector(BaseDetector):
    """Fallback detector used when trained model weights are unavailable locally or on HF."""

    def __init__(self, threshold: float = 0.5, default_tier_b_conf: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.default_tier_b_conf = float(default_tier_b_conf)
        self.model_name = "lexicon-only"
        self.backend = "lexicon_only"

    def predict_proba(self, texts: list[str]) -> list[float]:
        return [self.default_tier_b_conf for _ in texts]


class SklearnTfidfDetector(BaseDetector):
    """Joblib-serialized scikit-learn ``Pipeline(TfidfVectorizer, classifier)`` detector."""

    def __init__(
        self,
        pipeline_path: str | Path,
        threshold: float = 0.5,
        model_name: str = "classical-tfidf",
    ) -> None:
        self.pipeline: Any = joblib.load(pipeline_path)
        self.threshold = float(threshold)
        self.model_name = model_name
        self.backend = "sklearn_tfidf"

    def predict_proba(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        probs = self.pipeline.predict_proba(texts)
        return [float(p) for p in probs[:, 1]]


class OnnxDistilBertDetector(BaseDetector):
    """ONNX Runtime INT8 DistilBERT detector using ``onnxruntime`` + ``tokenizers``."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        threshold: float = 0.5,
        max_length: int = 128,
        intra_op_threads: int = 2,
        model_name: str = "distilbert-int8-onnx",
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = intra_op_threads
        sess_opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        self.threshold = float(threshold)
        self.model_name = model_name
        self.backend = "onnx_int8"

    def predict_proba(self, texts: list[str]) -> list[float]:
        if not texts:
            return []

        encodings = self.tokenizer.encode_batch(texts)
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

        ort_inputs: dict[str, np.ndarray] = {}
        if "input_ids" in self.input_names:
            ort_inputs["input_ids"] = input_ids
        if "attention_mask" in self.input_names:
            ort_inputs["attention_mask"] = attention_mask
        if "token_type_ids" in self.input_names:
            ort_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self.session.run(None, ort_inputs)
        logits = np.asarray(outputs[0], dtype=np.float32)
        probs = _softmax_pos_prob(logits)
        return [float(min(1.0, max(0.0, p))) if not math.isnan(float(p)) else 0.5 for p in probs]


def _download_hf_file(
    repo_id: str,
    filename: str,
    token: str | None,
    cache_dir: Path | None = None,
) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token or None,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(local)
    except Exception:
        return None


def _resolve_artifact(
    filename: str,
    local_dir: Path | None,
    hf_repo_id: str | None,
    hf_token: str | None,
) -> Path | None:
    if local_dir is not None:
        candidate = local_dir / filename
        if candidate.is_file():
            return candidate
    if hf_repo_id:
        return _download_hf_file(hf_repo_id, filename, hf_token)
    return None


def _read_detector_config(
    local_dir: Path | None,
    hf_repo_id: str | None,
    hf_token: str | None,
    filename: str = "detector_config.json",
) -> dict[str, Any]:
    cfg_path = _resolve_artifact(filename, local_dir, hf_repo_id, hf_token)
    if cfg_path is None:
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_detector(
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    local_dir: str | Path | None = "models",
    preferred_backend: str = "auto",
) -> BaseDetector:
    """Load the deployed context detector from local ``models/`` or private HF Hub.

    Preference order for ``preferred_backend="auto"``:
      1. ONNX INT8 DistilBERT (``distilbert_int8.onnx`` + ``tokenizer.json``,
         configured via ``onnx_detector_config.json`` with fallback to ``detector_config.json``)
      2. Classical TF-IDF pipeline (``best_classical_pipeline.pkl``,
         configured via ``detector_config.json``)
      3. ``LexiconOnlyDetector`` (never crashes if weights are not uploaded yet)
    """
    local_path = Path(local_dir) if local_dir else None
    classical_cfg = _read_detector_config(local_path, hf_repo_id, hf_token, "detector_config.json")

    if preferred_backend in ("auto", "onnx"):
        onnx_path = _resolve_artifact("distilbert_int8.onnx", local_path, hf_repo_id, hf_token)
        tok_path = _resolve_artifact("tokenizer.json", local_path, hf_repo_id, hf_token)
        if onnx_path is not None and tok_path is not None:
            onnx_cfg = _read_detector_config(local_path, hf_repo_id, hf_token, "onnx_detector_config.json")
            cfg = onnx_cfg or classical_cfg
            return OnnxDistilBertDetector(
                onnx_path=onnx_path,
                tokenizer_path=tok_path,
                threshold=float(cfg.get("threshold", 0.5)),
                model_name=str(cfg.get("model", "")) or "distilbert-int8-onnx",
            )

    if preferred_backend in ("auto", "classical", "sklearn"):
        pkl_path = _resolve_artifact("best_classical_pipeline.pkl", local_path, hf_repo_id, hf_token)
        if pkl_path is not None:
            return SklearnTfidfDetector(
                pipeline_path=pkl_path,
                threshold=float(classical_cfg.get("threshold", 0.5)),
                model_name=str(classical_cfg.get("model", "")) or "classical-tfidf",
            )

    return LexiconOnlyDetector(threshold=float(classical_cfg.get("threshold", 0.5)))
