"""End-to-end audio evaluation script on TTS clips and real-world recordings."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import numpy as np

from censor import ProfanityLexicon, analyze_audio, load_detector, normalize_token, render_censored_audio


def _f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * (precision * recall) / (b2 * precision + recall)


def _compute_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(_f_beta(p, r, beta=1.0), 4),
        "f2": round(_f_beta(p, r, beta=2.0), 4),
    }


def evaluate_tts(
    tts_dir: Path,
    asr_model,
    detector,
    lexicon: ProfanityLexicon,
    threshold: float | None = None,
) -> dict:
    """Evaluate end-to-end pipeline on ``data/e2e_tts/manifest.json``."""
    manifest_path = tts_dir / "manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    tau = detector.threshold if threshold is None else float(threshold)

    tp = fp = fn = 0
    tier_b_tp = tier_b_fn = 0
    boundary_errors_ms: list[float] = []
    rtfs: list[float] = []
    failures: list[dict] = []

    for rec in records:
        audio_path = tts_dir / rec["audio_file"]
        audio_bytes = audio_path.read_bytes()

        t0 = time.perf_counter()
        analysis = analyze_audio(
            audio_bytes=audio_bytes,
            filename=audio_path.name,
            asr_model=asr_model,
            detector=detector,
            lexicon=lexicon,
        )
        _wav, flagged_rows = render_censored_audio(analysis, threshold=tau)
        elapsed = time.perf_counter() - t0
        rtfs.append(elapsed / max(0.01, analysis.duration_s))

        # Ground-truth profane spans from TTS word boundaries
        expected_set = {normalize_token(w) for w in rec["expected_beep_words"]}
        gt_spans = []
        for wb in rec.get("word_boundaries", []):
            norm_wb = normalize_token(wb["word"])
            if norm_wb in expected_set:
                gt_spans.append((norm_wb, float(wb["start"]), float(wb["end"])))

        pred_beeped = [r for r in flagged_rows if r["beeped"]]
        matched_pred: set[int] = set()

        for gt_word, gt_s, gt_e in gt_spans:
            is_tier_b = gt_word in lexicon.tier_b
            best_idx = None
            best_overlap = 0.0
            for idx, pr in enumerate(pred_beeped):
                if idx in matched_pred:
                    continue
                # Pad predicted span by +-0.10 s (actual beep coverage) for time-overlap check
                ov = min(gt_e, pr["end"] + 0.10) - max(gt_s, pr["start"] - 0.10)
                if ov > best_overlap:
                    best_overlap = ov
                    best_idx = idx
            if best_idx is not None and best_overlap > 0.0:
                matched_pred.add(best_idx)
                tp += 1
                if is_tier_b:
                    tier_b_tp += 1
                pr = pred_beeped[best_idx]
                err_start = abs((pr["start"] - 0.10) - gt_s) * 1000.0
                err_end = abs((pr["end"] + 0.10) - gt_e) * 1000.0
                boundary_errors_ms.append(0.5 * (err_start + err_end))
            else:
                fn += 1
                if is_tier_b:
                    tier_b_fn += 1
                failures.append(
                    {
                        "clip_id": rec["clip_id"],
                        "category": rec["category"],
                        "type": "FN_MISSED_PROFANITY",
                        "word": gt_word,
                        "text": rec["text"],
                    }
                )

        for idx, pr in enumerate(pred_beeped):
            if idx not in matched_pred:
                fp += 1
                failures.append(
                    {
                        "clip_id": rec["clip_id"],
                        "category": rec["category"],
                        "type": "FP_FALSE_BEEP",
                        "word": pr["word"],
                        "text": rec["text"],
                    }
                )

    tier_b_recall = tier_b_tp / (tier_b_tp + tier_b_fn) if (tier_b_tp + tier_b_fn) > 0 else 0.0
    return {
        "n_clips": len(records),
        "threshold": tau,
        "overall": _compute_prf(tp, fp, fn),
        "tier_b_recall": round(tier_b_recall, 4),
        "tier_b_tp": tier_b_tp,
        "tier_b_fn": tier_b_fn,
        "mean_boundary_error_ms": round(float(np.mean(boundary_errors_ms)), 2) if boundary_errors_ms else 0.0,
        "median_boundary_error_ms": round(float(np.median(boundary_errors_ms)), 2) if boundary_errors_ms else 0.0,
        "mean_rtf": round(float(np.mean(rtfs)), 4) if rtfs else 0.0,
        "failures": failures,
    }


def evaluate_real(
    real_dir: Path,
    asr_model,
    detector,
    lexicon: ProfanityLexicon,
    threshold: float | None = None,
) -> dict:
    """Evaluate end-to-end pipeline on ``data/e2e_real/labels.csv``."""
    labels_csv = real_dir / "labels.csv"
    if not labels_csv.is_file():
        return {"n_clips": 0, "note": "labels.csv not found"}

    rows: list[dict] = []
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fname = (row.get("file") or "").strip()
            if fname and not fname.startswith("#"):
                rows.append(row)

    if not rows:
        return {"n_clips": 0, "note": "No labeled real recordings yet in data/e2e_real/labels.csv"}

    tau = detector.threshold if threshold is None else float(threshold)
    tp = fp = fn = 0
    tier_b_tp = tier_b_fn = 0
    rtfs: list[float] = []
    failures: list[dict] = []

    for row in rows:
        audio_path = real_dir / row["file"].strip()
        if not audio_path.is_file():
            failures.append({"file": row["file"], "type": "MISSING_FILE"})
            continue

        raw_expected = (row.get("profane_words") or "").replace(";", " ").split()
        gt_counts = Counter(normalize_token(w) for w in raw_expected if normalize_token(w))

        t0 = time.perf_counter()
        analysis = analyze_audio(
            audio_bytes=audio_path.read_bytes(),
            filename=audio_path.name,
            asr_model=asr_model,
            detector=detector,
            lexicon=lexicon,
        )
        _wav, flagged_rows = render_censored_audio(analysis, threshold=tau)
        rtfs.append((time.perf_counter() - t0) / max(0.01, analysis.duration_s))

        pred_counts = Counter(normalize_token(r["word"]) for r in flagged_rows if r["beeped"])

        for w, g_cnt in gt_counts.items():
            p_cnt = pred_counts.get(w, 0)
            matched = min(g_cnt, p_cnt)
            missed = g_cnt - matched
            tp += matched
            fn += missed
            if w in lexicon.tier_b:
                tier_b_tp += matched
                tier_b_fn += missed
            if missed > 0:
                failures.append({"file": audio_path.name, "type": "FN_MISSED_PROFANITY", "word": w})

        for w, p_cnt in pred_counts.items():
            extra = p_cnt - gt_counts.get(w, 0)
            if extra > 0:
                fp += extra
                failures.append({"file": audio_path.name, "type": "FP_FALSE_BEEP", "word": w})

    tier_b_recall = tier_b_tp / (tier_b_tp + tier_b_fn) if (tier_b_tp + tier_b_fn) > 0 else 0.0
    return {
        "n_clips": len(rows),
        "threshold": tau,
        "overall": _compute_prf(tp, fp, fn),
        "tier_b_recall": round(tier_b_recall, 4),
        "tier_b_tp": tier_b_tp,
        "tier_b_fn": tier_b_fn,
        "mean_rtf": round(float(np.mean(rtfs)), 4) if rtfs else 0.0,
        "failures": failures,
    }


def main() -> None:
    default_repo = os.environ.get("HF_REPO_ID", "").strip()
    default_user = os.environ.get("HF_USER", "").strip()
    if default_repo and "/" not in default_repo and default_user:
        default_repo = f"{default_user}/{default_repo}"
    default_token = (
        os.environ.get("HF_TOKEN_streamlitt")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_TOKEN_kaggle")
        or ""
    ).strip()

    parser = argparse.ArgumentParser(description="Run end-to-end audio evaluation.")
    parser.add_argument("--asr-size", default="base.en")
    parser.add_argument("--tts-dir", type=Path, default=Path("data/e2e_tts"))
    parser.add_argument("--real-dir", type=Path, default=Path("data/e2e_real"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--hf-repo-id", default=default_repo)
    parser.add_argument("--hf-token", default=default_token)
    parser.add_argument("--out-json", type=Path, default=Path("data/e2e_eval_results.json"))
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    lexicon = ProfanityLexicon.from_csv()
    detector = load_detector(
        hf_repo_id=args.hf_repo_id or None,
        hf_token=args.hf_token or None,
        local_dir=args.models_dir,
    )
    asr_model = WhisperModel(args.asr_size, device="cpu", compute_type="int8", cpu_threads=2)

    results: dict = {
        "asr_model": f"faster-whisper-{args.asr_size}",
        "detector": detector.model_name,
        "detector_backend": detector.backend,
        "threshold": detector.threshold,
    }
    if (args.tts_dir / "manifest.json").is_file():
        results["tts"] = evaluate_tts(args.tts_dir, asr_model, detector, lexicon)
    else:
        results["tts"] = {"n_clips": 0, "note": "Run scripts/gen_tts_testset.py first"}

    results["real"] = evaluate_real(args.real_dir, asr_model, detector, lexicon)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
