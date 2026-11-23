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
# # Automated Speech Censor — Notebook 02: Pre-trained & Fine-Tuned Transformers + ONNX INT8 Export
#
# > **Live Cloud Deployment URL:** `https://YOUR-APP-NAME.streamlit.app` *(Update with your live Streamlit Community Cloud URL before Google Classroom submission)*
#
# **Compute:** Kaggle GPU T4 x2 with `CUDA_VISIBLE_DEVICES=0` pinned before importing `torch` for clean single-T4 `fp16` training (avoiding `DataParallel` quirks).  
# **Models Evaluated:**
# 1. **In-Domain Pre-trained (no fine-tuning):** `unitary/toxic-bert` (`obscene` head; explicitly note Unitary already fine-tuned it on Jigsaw 2018 `train.csv` — NOT zero-shot).
# 2. **Cross-Domain Zero-Shot Transfer:** `cardiffnlp/twitter-roberta-base-offensive` (trained on TweetEval/OLID; `offensive` is broader than Jigsaw `obscene`).
# 3. **Fine-Tuned Transformer:** `distilbert/distilbert-base-uncased` (`max_len=128`, `batch_size=32`, `lr=2e-5`, `2` epochs, `fp16`, 90/10 stratified val split for F2 threshold tuning `τ*`).
# 4. **ONNX INT8 Quantization & Parity Check:** Export `distilbert_int8.onnx` (~65 MB) via `onnxruntime.quantization.quantize_dynamic` for zero-PyTorch deployment on Streamlit Cloud.

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Must be set before importing torch

import json
from pathlib import Path
import shutil
import subprocess
import sys
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

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

warnings.filterwarnings("ignore")
SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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

OUT_DIR = Path("artifacts_nb02")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Configured HF_REPO_ID set={bool(HF_REPO_ID)} | HF_TOKEN_kaggle set={bool(HF_TOKEN_WRITE)}")

# %% [markdown]
# ## 1. Load Jigsaw Dataset (Kaggle Input or Automatic HF Mirror) & 90/10 Stratified Split

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

tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    train_df["comment_text"].tolist(),
    train_df["obscene"].astype(int).tolist(),
    test_size=0.10,
    stratify=train_df["obscene"].astype(int).tolist(),
    random_state=SEED,
)
test_texts = scored_test_df["comment_text"].tolist()
test_labels = scored_test_df["obscene"].astype(int).tolist()

print(f"Train: {len(tr_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
    }

# %% [markdown]
# ## 2. Evaluate Pre-Trained Baselines (`unitary/toxic-bert` & `cardiffnlp/twitter-roberta-base-offensive`)

# %%
@torch.no_grad()
def predict_hf_probs(model_id: str, texts: list[str], mode: str, batch_size: int = 64, max_len: int = 128) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE).eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], truncation=True, padding=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = mdl(**enc).logits
        if mode == "sigmoid_obscene":
            obscene_idx = [k for k, v in mdl.config.id2label.items() if v.lower() == "obscene"][0]
            p = torch.sigmoid(logits[:, obscene_idx]).float().cpu().numpy()
        elif mode == "softmax_offensive":
            p = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        else:
            p = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        probs.append(p)
    del mdl
    torch.cuda.empty_cache()
    return np.concatenate(probs)


transformer_rows = []

p_toxic_bert = predict_hf_probs("unitary/toxic-bert", test_texts, mode="sigmoid_obscene")
m_toxic = compute_metrics(test_labels, (p_toxic_bert >= 0.5).astype(int))
transformer_rows.append({"Model": "unitary/toxic-bert (in-domain, NOT zero-shot)", "Threshold": 0.5, **m_toxic})

p_tweet_rob = predict_hf_probs("cardiffnlp/twitter-roberta-base-offensive", test_texts, mode="softmax_offensive")
m_rob = compute_metrics(test_labels, (p_tweet_rob >= 0.5).astype(int))
transformer_rows.append({"Model": "cardiffnlp/twitter-roberta-base-offensive (cross-domain zero-shot)", "Threshold": 0.5, **m_rob})

display(pd.DataFrame(transformer_rows))

# %% [markdown]
# ## 3. Fine-Tune `distilbert/distilbert-base-uncased` (2 Epochs, `fp16`, `max_len=128`, `batch_size=32`)

# %%
MODEL_ID = "distilbert/distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc


train_loader = DataLoader(TextDataset(tr_texts, tr_labels), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(TextDataset(val_texts, val_labels), batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(TextDataset(test_texts, test_labels), batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    for step, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            out = model(**batch)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += float(loss.item())
        if step % 1000 == 0:
            print(f"Epoch {epoch} step {step}/{len(train_loader)} loss={running_loss/step:.4f}")
    print(f"Epoch {epoch} done in {time.time() - t0:.1f}s | avg_loss={running_loss/len(train_loader):.4f}")

SAVE_PT_DIR = OUT_DIR / "distilbert_fp32"
model.save_pretrained(SAVE_PT_DIR)
tokenizer.save_pretrained(SAVE_PT_DIR)

# %% [markdown]
# ## 4. Tune Validation F2 Threshold `τ*` & Evaluate on Official Test Set

# %%
@torch.no_grad()
def predict_loader(mdl, loader) -> np.ndarray:
    mdl.eval()
    probs = []
    for batch in loader:
        inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = mdl(**inputs).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


val_probs = predict_loader(model, val_loader)
thresholds = np.linspace(0.05, 0.95, 91)
val_f2s = [fbeta_score(val_labels, (val_probs >= t).astype(int), beta=2.0, zero_division=0) for t in thresholds]
tau_star = float(thresholds[int(np.argmax(val_f2s))])
print(f"Validation-tuned tau* = {tau_star:.2f} (Val F2 = {max(val_f2s):.4f})")

test_probs_fp32 = predict_loader(model, test_loader)
m_ft_05 = compute_metrics(test_labels, (test_probs_fp32 >= 0.5).astype(int))
m_ft_tau = compute_metrics(test_labels, (test_probs_fp32 >= tau_star).astype(int))
transformer_rows.append({"Model": "DistilBERT fine-tuned (FP32 PyTorch)", "Threshold": 0.5, **m_ft_05})
transformer_rows.append({"Model": "DistilBERT fine-tuned (FP32 PyTorch)", "Threshold": round(tau_star, 2), **m_ft_tau})
display(pd.DataFrame(transformer_rows))

# %% [markdown]
# ## 5. Export ONNX FP32 -> Quantize INT8 -> Parity Check & 15-Sentence Tier B Sanity Probe

# %%
try:
    import onnx  # noqa: F401
    import onnxruntime as ort
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"])
    import onnxruntime as ort

from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer

model_cpu = model.to("cpu").eval()
dummy_enc = tokenizer("test sentence for export", return_tensors="pt", padding="max_length", max_length=MAX_LEN)
fp32_onnx = OUT_DIR / "distilbert_fp32.onnx"
int8_onnx = OUT_DIR / "distilbert_int8.onnx"

torch.onnx.export(
    model_cpu,
    (dummy_enc["input_ids"], dummy_enc["attention_mask"]),
    str(fp32_onnx),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    },
    opset_version=14,
)

quantize_dynamic(
    model_input=str(fp32_onnx),
    model_output=str(int8_onnx),
    weight_type=QuantType.QInt8,
)
print(f"FP32 ONNX size: {fp32_onnx.stat().st_size / 1e6:.1f} MB | INT8 ONNX size: {int8_onnx.stat().st_size / 1e6:.1f} MB")

fast_tok = Tokenizer.from_file(str(SAVE_PT_DIR / "tokenizer.json"))
fast_tok.enable_truncation(max_length=MAX_LEN)
fast_tok.enable_padding(pad_id=0, pad_token="[PAD]")
sess = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])

onnx_probs = []
for i in range(0, len(test_texts), 128):
    encs = fast_tok.encode_batch(test_texts[i : i + 128])
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    sh = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(sh) / np.sum(np.exp(sh), axis=1, keepdims=True)
    onnx_probs.append(p[:, 1])
test_probs_int8 = np.concatenate(onnx_probs)

m_int8_05 = compute_metrics(test_labels, (test_probs_int8 >= 0.5).astype(int))
m_int8_tau = compute_metrics(test_labels, (test_probs_int8 >= tau_star).astype(int))
transformer_rows.append({"Model": "DistilBERT fine-tuned (ONNX INT8)", "Threshold": 0.5, **m_int8_05})
transformer_rows.append({"Model": "DistilBERT fine-tuned (ONNX INT8)", "Threshold": round(tau_star, 2), **m_int8_tau})
summary_df = pd.DataFrame(transformer_rows)
summary_df.to_csv(OUT_DIR / "nb02_transformer_comparison.csv", index=False)
display(summary_df)

probe_sentences = [
    ("Don't be such a dick to him", 1),
    ("Move your lazy ass out of here", 1),
    ("Stop whining like a spoiled bitch", 1),
    ("He is an arrogant prick who lies", 1),
    ("That lying bastard stole my wallet", 1),
    ("He yelled suck my cock outside", 1),
    ("Quit acting like a total pussy", 1),
    ("She called her coworker a slut", 1),
    ("Dick Grayson was the original Robin", 0),
    ("I am taking a machine learning class", 0),
    ("The rooster or cock crowed early", 0),
    ("She adopted a female dog or bitch", 0),
    ("A wild ass walked the mountain trail", 0),
    ("Please pass me a fruit cocktail", 0),
    ("We must assess the model recall", 0),
]
p_encs = fast_tok.encode_batch([s for s, _ in probe_sentences])
p_ids = np.array([e.ids for e in p_encs], dtype=np.int64)
p_mask = np.array([e.attention_mask for e in p_encs], dtype=np.int64)
p_logits = sess.run(None, {"input_ids": p_ids, "attention_mask": p_mask})[0]
p_sh = p_logits - np.max(p_logits, axis=1, keepdims=True)
p_probs = (np.exp(p_sh) / np.sum(np.exp(p_sh), axis=1, keepdims=True))[:, 1]
for (sent, y_exp), prob in zip(probe_sentences, p_probs):
    print(f"exp={y_exp} pred={int(prob >= tau_star)} p={prob:.3f} | {sent}")

# %% [markdown]
# ## 6. Known Domain-Shift & Casing Limitations
# - **Training vs. Inference Domain Shift:** DistilBERT is fine-tuned on full written Wikipedia talk comments (`train.csv`), whereas at runtime it receives `±5`-word spoken ASR transcript snippets without punctuation richness.
# - **Uncased Tokenization:** `distilbert-base-uncased` lowercases all text, so capitalization cues from Whisper (e.g., proper name `"Dick"` vs. `"dick"`) cannot be leveraged directly by the classifier.
#
# ## 7. Export `onnx_detector_config.json` & Push ONNX INT8 Artifacts to Private HF Hub (`HF_TOKEN_kaggle`)

# %%
shutil.copy2(SAVE_PT_DIR / "tokenizer.json", OUT_DIR / "tokenizer.json")
cfg = {
    "model": "distilbert-base-uncased-jigsaw-obscene-int8",
    "backend": "onnx_int8",
    "threshold": round(tau_star, 4),
    "metric": "F2",
    "test_metrics_at_0_5": {k: round(v, 4) for k, v in m_int8_05.items()},
    "test_metrics_at_tau_star": {k: round(v, 4) for k, v in m_int8_tau.items()},
}
(OUT_DIR / "onnx_detector_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

if HF_TOKEN_WRITE and HF_REPO_ID and "YOUR_HF_USER" not in HF_REPO_ID:
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN_WRITE)
    for fname in ("distilbert_int8.onnx", "tokenizer.json", "onnx_detector_config.json"):
        api.upload_file(
            path_or_fileobj=str(OUT_DIR / fname),
            path_in_repo=fname,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
    print(f"Uploaded ONNX INT8 artifacts to https://huggingface.co/{HF_REPO_ID}")
else:
    print("Skipped HF upload: ensure HF_REPO_ID and HF_TOKEN_kaggle are configured.")
