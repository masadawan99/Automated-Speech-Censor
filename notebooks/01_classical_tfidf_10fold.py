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
# # Automated Speech Censor — Notebook 01: Classical TF-IDF & 10-Fold CV
#
# > **Live Cloud Deployment URL:** `https://YOUR-APP-NAME.streamlit.app` *(Update with your live Streamlit Community Cloud URL before Google Classroom submission)*
#
# **Course:** AIC354 Machine Learning Fundamentals (COMSATS Lahore, Fall 2026)  
# **Compute:** Kaggle CPU (no GPU needed — saves your weekly Kaggle GPU quota for Notebook 02)  
# **Task:** Binary Profanity / Obscenity Classification on Jigsaw Toxic Comment Classification Challenge (`obscene` target, 159,571 train rows; 63,978 official scored test rows)  
# **Evaluation Protocol:**
# - **10-fold `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`** inside `Pipeline(TfidfVectorizer, clf)` via `cross_validate` (zero TF-IDF data leakage).
# - Refit on full `train.csv` and score on the official labelled test set (`test_labels.csv` where `obscene != -1`, 63,978 rows).
# - Report **Accuracy**, **Precision**, **Recall**, **F1**, and **F2** (`beta=2`, weighting Recall 2x Precision). **Recall is prioritized over Precision** because missed profanity in speech is worse than a false beep.
# - Highlight **The Accuracy Trap** via `DummyClassifier(strategy="most_frequent")` (~94.71% accuracy with 0.00 recall).
# - Compare **Variation 1 (no class weights)** vs. **Variation 2 (`class_weight='balanced'`)** (for LR, calibrated LinearSVC, RF; `ComplementNB` has no `class_weight` and appears in Variation 1 only).

# %%
import json
import os
from pathlib import Path
import re
import time
import warnings

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from IPython.display import display
except ImportError:
    display = print

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


def get_secret(key: str, default: str = "") -> str:
    """Read secret from Kaggle UserSecretsClient first, then environment/.env."""
    try:
        from kaggle_secrets import UserSecretsClient

        val = UserSecretsClient().get_secret(key)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.environ.get(key, default).strip()


HF_USER = get_secret("HF_USER")
HF_REPO_ID = get_secret("HF_REPO_ID")
if HF_REPO_ID and "/" not in HF_REPO_ID and HF_USER:
    HF_REPO_ID = f"{HF_USER}/{HF_REPO_ID}"
HF_TOKEN_WRITE = get_secret("HF_TOKEN_kaggle") or get_secret("HF_TOKEN")

OUT_DIR = Path("artifacts_nb01")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Configured HF_REPO_ID set={bool(HF_REPO_ID)} | HF_TOKEN_kaggle set={bool(HF_TOKEN_WRITE)}")

# %% [markdown]
# ## 1. Load Jigsaw Dataset (Kaggle Input or Automatic Hugging Face Mirror)
#
# - If you attached the Kaggle competition dataset (`jigsaw-toxic-comment-classification-challenge`), this cell finds `train.csv[.zip]`, `test.csv[.zip]`, and `test_labels.csv[.zip]` anywhere under `/kaggle/input`.
# - If no Kaggle dataset is attached (with **Internet = ON**), it automatically downloads the exact CSVs from the ungated Hugging Face mirror `thesofakillers/jigsaw-toxic-comment-classification-challenge`.

# %%
def _find_kaggle_file(stem: str) -> Path | None:
    kaggle_root = Path("/kaggle/input")
    if not kaggle_root.exists():
        return None
    for ext in (".csv", ".csv.zip"):
        matches = sorted(kaggle_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def load_jigsaw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_path = _find_kaggle_file("train")
    te_path = _find_kaggle_file("test")
    tl_path = _find_kaggle_file("test_labels")

    if tr_path and te_path and tl_path:
        print(f"Loading Jigsaw CSVs from Kaggle input: {tr_path.parent}")
        train_df = pd.read_csv(tr_path)
        test_df = pd.read_csv(te_path)
        test_labels_df = pd.read_csv(tl_path)
    else:
        print("Kaggle input not found; downloading CSVs from HF mirror thesofakillers/jigsaw-toxic-comment-classification-challenge...")
        from huggingface_hub import hf_hub_download

        repo = "thesofakillers/jigsaw-toxic-comment-classification-challenge"
        train_df = pd.read_csv(hf_hub_download(repo_id=repo, filename="train.csv", repo_type="dataset"))
        test_df = pd.read_csv(hf_hub_download(repo_id=repo, filename="test.csv", repo_type="dataset"))
        test_labels_df = pd.read_csv(hf_hub_download(repo_id=repo, filename="test_labels.csv", repo_type="dataset"))

    merged_test = test_df.merge(test_labels_df, on="id")
    scored_test_df = merged_test[merged_test["obscene"] != -1].reset_index(drop=True)
    return train_df, scored_test_df


train_df, scored_test_df = load_jigsaw_data()
train_df["comment_text"] = train_df["comment_text"].fillna("").astype(str)
scored_test_df["comment_text"] = scored_test_df["comment_text"].fillna("").astype(str)

X_train = train_df["comment_text"].values
y_train = train_df["obscene"].astype(int).values
X_test = scored_test_df["comment_text"].values
y_test = scored_test_df["obscene"].astype(int).values

print(f"Train shape: {len(X_train):,} | Positive (obscene=1): {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"Test  shape: {len(X_test):,} | Positive (obscene=1): {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# %% [markdown]
# ## 2. Metric Helper & Baselines (DummyClassifier + Lexicon-Only)

# %%
def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
    }


def lexicon_predict(texts, tier_a_words: set[str], tier_b_words: set[str], include_tier_b: bool) -> np.ndarray:
    target_set = tier_a_words | tier_b_words if include_tier_b else tier_a_words
    preds = np.zeros(len(texts), dtype=int)
    for i, txt in enumerate(texts):
        tokens = re.findall(r"[a-z]+", txt.lower())
        if any(t in target_set for t in tokens):
            preds[i] = 1
    return preds


LEXICON_PATH = Path("lexicon/profanity_lexicon.csv")
if not LEXICON_PATH.exists():
    LEXICON_PATH = Path("../lexicon/profanity_lexicon.csv")

if LEXICON_PATH.exists():
    lex_df = pd.read_csv(LEXICON_PATH)
    tier_a_set = set(lex_df.loc[lex_df["tier"] == "TIER_A", "word"].str.lower())
    tier_b_set = set(lex_df.loc[lex_df["tier"] == "TIER_B", "word"].str.lower())
else:
    tier_a_set = {"fuck", "fucking", "fucked", "fucker", "shit", "shitty", "bullshit", "cunt", "asshole", "motherfucker"}
    tier_b_set = {"ass", "dick", "bitch", "cock", "pussy", "prick", "bastard", "whore", "slut", "piss"}

results_rows = []

# Baseline 1: DummyClassifier (Accuracy Trap)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_test_m = compute_metrics(y_test, dummy.predict(X_test))
results_rows.append({"Group": "Baseline", "Model": "DummyClassifier (most_frequent)", "Variation": "Baseline", **dummy_test_m})

# Baseline 2: Lexicon-only (Tier A only, and Tier A + Tier B)
lex_a_pred = lexicon_predict(X_test, tier_a_set, tier_b_set, include_tier_b=False)
lex_ab_pred = lexicon_predict(X_test, tier_a_set, tier_b_set, include_tier_b=True)
results_rows.append({"Group": "Baseline", "Model": "Lexicon-Only (Tier A only)", "Variation": "Baseline", **compute_metrics(y_test, lex_a_pred)})
results_rows.append({"Group": "Baseline", "Model": "Lexicon-Only (Tier A + Tier B)", "Variation": "Baseline", **compute_metrics(y_test, lex_ab_pred)})

display(pd.DataFrame(results_rows))

# %% [markdown]
# ## 3. 10-Fold Stratified CV: Variation 1 (Unweighted) vs. Variation 2 (`class_weight='balanced'`)

# %%
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
scoring = {
    "acc": "accuracy",
    "prec": "precision",
    "rec": "recall",
    "f1": "f1",
}


def make_tfidf(max_features: int = 50000, min_df: int = 3) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        strip_accents="unicode",
    )


model_specs = [
    # Variation 1: No class weights
    ("Variation 1 (Unweighted)", "LogisticRegression", Pipeline([
        ("tfidf", make_tfidf(50000, 3)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])),
    ("Variation 1 (Unweighted)", "Calibrated LinearSVC", Pipeline([
        ("tfidf", make_tfidf(50000, 3)),
        ("clf", CalibratedClassifierCV(LinearSVC(random_state=SEED), method="sigmoid", cv=3)),
    ])),
    ("Variation 1 (Unweighted)", "ComplementNB", Pipeline([
        ("tfidf", make_tfidf(50000, 3)),
        ("clf", ComplementNB()),
    ])),
    ("Variation 1 (Unweighted)", "RandomForest", Pipeline([
        ("tfidf", make_tfidf(5000, 5)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    ])),
    # Variation 2: class_weight='balanced' (ComplementNB excluded as it has no class_weight)
    ("Variation 2 (Balanced)", "LogisticRegression (balanced)", Pipeline([
        ("tfidf", make_tfidf(50000, 3)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])),
    ("Variation 2 (Balanced)", "Calibrated LinearSVC (balanced)", Pipeline([
        ("tfidf", make_tfidf(50000, 3)),
        ("clf", CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), method="sigmoid", cv=3)),
    ])),
    ("Variation 2 (Balanced)", "RandomForest (balanced)", Pipeline([
        ("tfidf", make_tfidf(5000, 5)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=30, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED)),
    ])),
]

cv_rows = []
fitted_pipelines = {}

for variation, name, pipe in model_specs:
    t0 = time.time()
    cv_out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_prec = float(np.mean(cv_out["test_prec"]))
    cv_rec = float(np.mean(cv_out["test_rec"]))
    cv_f2 = (5 * cv_prec * cv_rec) / (4 * cv_prec + cv_rec) if (cv_prec + cv_rec) > 0 else 0.0
    cv_rows.append({
        "Variation": variation,
        "Model": name,
        "CV_Acc": float(np.mean(cv_out["test_acc"])),
        "CV_Prec": cv_prec,
        "CV_Rec": cv_rec,
        "CV_F1": float(np.mean(cv_out["test_f1"])),
        "CV_F2": cv_f2,
        "CV_Time_s": round(time.time() - t0, 1),
    })

    # Refit on full train set and evaluate on official scored test set
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe
    y_pred_test = pipe.predict(X_test)
    test_m = compute_metrics(y_test, y_pred_test)
    results_rows.append({"Group": "Classical ML", "Model": name, "Variation": variation, **test_m})
    print(
        f"[{name}] CV F1={cv_rows[-1]['CV_F1']:.4f} | "
        f"Test Acc={test_m['Accuracy']:.4f} P={test_m['Precision']:.4f} "
        f"R={test_m['Recall']:.4f} F1={test_m['F1']:.4f} F2={test_m['F2']:.4f}"
    )

cv_df = pd.DataFrame(cv_rows)
test_df_summary = pd.DataFrame(results_rows)
display(cv_df)
display(test_df_summary.sort_values("F2", ascending=False))

# %% [markdown]
# ## 4. Variation 1 vs. Variation 2 Chart & Best Classical Model Selection

# %%
fig, ax = plt.subplots(figsize=(10, 5))
plot_df = test_df_summary[test_df_summary["Group"] == "Classical ML"].copy()
plot_melted = plot_df.melt(
    id_vars=["Model", "Variation"],
    value_vars=["Precision", "Recall", "F1", "F2"],
    var_name="Metric",
    value_name="Score",
)
sns.barplot(data=plot_melted, x="Model", y="Score", hue="Metric", ax=ax)
plt.xticks(rotation=25, ha="right")
plt.title("Official Jigsaw Test Set: Variation 1 (Unweighted) vs Variation 2 (Balanced)")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(OUT_DIR / "classical_var1_vs_var2.png", dpi=150)
plt.show()

classical_only = test_df_summary[test_df_summary["Group"] == "Classical ML"]
best_row = classical_only.sort_values("F2", ascending=False).iloc[0]
BEST_MODEL_NAME = str(best_row["Model"])
best_pipe = fitted_pipelines[BEST_MODEL_NAME]
print(
    f"Best Classical Model (selected by Recall-weighted F2): {BEST_MODEL_NAME} | "
    f"Test F2={best_row['F2']:.4f}, F1={best_row['F1']:.4f}, Acc={best_row['Accuracy']:.4f}"
)

# %% [markdown]
# ## 5. Out-of-Fold 10-Fold CV Threshold Tuning (`τ*` for F2) & Confusion Matrix

# %%
oof_probs = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
thresholds = np.linspace(0.05, 0.95, 91)
oof_f2s = [fbeta_score(y_train, (oof_probs >= t).astype(int), beta=2.0, zero_division=0) for t in thresholds]
tau_star = float(thresholds[int(np.argmax(oof_f2s))])
print(f"Classical OOF-tuned threshold tau* = {tau_star:.2f} (OOF F2 = {max(oof_f2s):.4f})")

test_probs = best_pipe.predict_proba(X_test)[:, 1]
m_05 = compute_metrics(y_test, (test_probs >= 0.5).astype(int))
m_tau = compute_metrics(y_test, (test_probs >= tau_star).astype(int))
print("Test @ 0.50:", {k: round(v, 4) for k, v in m_05.items()})
print(f"Test @ {tau_star:.2f}:", {k: round(v, 4) for k, v in m_tau.items()})

cm = confusion_matrix(y_test, (test_probs >= tau_star).astype(int))
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Clean", "Obscene"], yticklabels=["Clean", "Obscene"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"{BEST_MODEL_NAME} @ tau*={tau_star:.2f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "classical_confusion_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Step 9 Unseen-Text Inference & Export Artifacts (`.pkl` + `detector_config.json`)

# %%
sample_windows = [
    "I am taking a machine learning class today",
    "Dick Grayson was the original Robin",
    "Don't be such a dick to the intern",
    "Move your lazy ass out of the doorway",
    "Stop acting like a total bitch right now",
]
sample_probs = best_pipe.predict_proba(sample_windows)[:, 1]
for text, p in zip(sample_windows, sample_probs):
    print(f"p={p:.4f} (pred={int(p >= tau_star)}) | {text}")

tfidf_vec = best_pipe.named_steps["tfidf"]
clf_step = best_pipe.named_steps["clf"]

joblib.dump(tfidf_vec, OUT_DIR / "tfidf_vectorizer.pkl")
joblib.dump(clf_step, OUT_DIR / "best_model.pkl")
joblib.dump(best_pipe, OUT_DIR / "best_classical_pipeline.pkl")

cfg = {
    "model": BEST_MODEL_NAME,
    "backend": "sklearn_tfidf",
    "threshold": round(tau_star, 4),
    "metric": "F2",
    "test_metrics_at_0_5": {k: round(v, 4) for k, v in m_05.items()},
    "test_metrics_at_tau_star": {k: round(v, 4) for k, v in m_tau.items()},
}
(OUT_DIR / "detector_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
test_df_summary.to_csv(OUT_DIR / "nb01_model_comparison.csv", index=False)
print("Saved artifacts to", OUT_DIR)

# %% [markdown]
# ## 7. Push Classical Artifacts to Private Hugging Face Hub Repo (`HF_TOKEN_kaggle`)

# %%
if HF_TOKEN_WRITE and HF_REPO_ID and "YOUR_HF_USER" not in HF_REPO_ID:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN_WRITE)
    for fname in ("tfidf_vectorizer.pkl", "best_model.pkl", "best_classical_pipeline.pkl", "detector_config.json"):
        api.upload_file(
            path_or_fileobj=str(OUT_DIR / fname),
            path_in_repo=fname,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
    print(f"Uploaded classical artifacts to https://huggingface.co/{HF_REPO_ID}")
else:
    print("Skipped HF upload: ensure HF_REPO_ID and HF_TOKEN_kaggle are configured.")
