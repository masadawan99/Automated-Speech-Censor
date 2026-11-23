# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Automated Speech Censor — Notebook 04: End-to-End Audio Pipeline Evaluation (TTS + Real Clips)
#
# Runs the exact `censor/` deployment pipeline (`faster-whisper` + ONNX INT8 / TF-IDF detector + two-tier lexicon + 1 kHz beep renderer) across:
# 1. **~120 TTS clips** (`data/e2e_tts/manifest.json`) with exact word boundaries.
# 2. **20–30 real-world recordings** (`data/e2e_real/labels.csv`).
#
# Reports word-level beep **Precision, Recall, F1, F2**, **Tier B recall separately**, **beep boundary timing error (ms)**, **RTF**, **synthetic-vs-real gap**, and **failure-case taxonomy**.

# %%
import os
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from IPython.display import display
except ImportError:
    display = print

import pandas as pd

REPO_ROOT = Path("..").resolve() if Path("../censor").exists() else Path(".").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faster_whisper import WhisperModel
from censor import ProfanityLexicon, load_detector
from scripts.run_e2e_eval import evaluate_real, evaluate_tts

ASR_SIZE = os.environ.get("ASR_MODEL_SIZE", "base.en")
TTS_DIR = REPO_ROOT / "data" / "e2e_tts"
REAL_DIR = REPO_ROOT / "data" / "e2e_real"
MODELS_DIR = REPO_ROOT / "models"

hf_user = os.environ.get("HF_USER", "").strip()
hf_repo_id = os.environ.get("HF_REPO_ID", "").strip()
if hf_repo_id and "/" not in hf_repo_id and hf_user:
    hf_repo_id = f"{hf_user}/{hf_repo_id}"
hf_token = (
    os.environ.get("HF_TOKEN_streamlitt")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HF_TOKEN_kaggle")
    or ""
).strip()

lexicon = ProfanityLexicon.from_csv(REPO_ROOT / "lexicon" / "profanity_lexicon.csv")
detector = load_detector(
    hf_repo_id=hf_repo_id or None,
    hf_token=hf_token or None,
    local_dir=MODELS_DIR,
)
asr_model = WhisperModel(ASR_SIZE, device="cpu", compute_type="int8", cpu_threads=2)

print(f"ASR: faster-whisper-{ASR_SIZE} | Detector: {detector.model_name} ({detector.backend}) | tau*={detector.threshold}")

# %% [markdown]
# ## 1. Run End-to-End Evaluation on TTS & Real Clips

# %%
tts_res = evaluate_tts(TTS_DIR, asr_model, detector, lexicon) if (TTS_DIR / "manifest.json").exists() else {"n_clips": 0}
real_res = evaluate_real(REAL_DIR, asr_model, detector, lexicon)

summary_rows = []
for split_name, res in [("TTS Synthetic", tts_res), ("Real Recordings", real_res)]:
    if res.get("n_clips", 0) > 0 and "overall" in res:
        ov = res["overall"]
        summary_rows.append({
            "Split": split_name,
            "Clips": res["n_clips"],
            "Precision": ov["precision"],
            "Recall": ov["recall"],
            "F1": ov["f1"],
            "F2": ov["f2"],
            "Tier_B_Recall": res["tier_b_recall"],
            "Mean_RTF": res["mean_rtf"],
            "Mean_Boundary_Err_ms": res.get("mean_boundary_error_ms", None),
        })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

# %% [markdown]
# ## 2. Synthetic-vs-Real Gap & Failure-Case Taxonomy

# %%
for name, res in [("TTS", tts_res), ("Real", real_res)]:
    fails = res.get("failures", [])
    print(f"=== {name} Failures ({len(fails)}) ===")
    if fails:
        fdf = pd.DataFrame(fails)
        display(fdf["type"].value_counts())
        display(fdf.head(20))
