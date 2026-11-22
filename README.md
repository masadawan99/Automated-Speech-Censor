# Automated Speech Censor (Voice Profanity Beeper)

**Course:** AIC354 Machine Learning Fundamentals (COMSATS Lahore, Fall 2026)  
**Target Spec:** [Voice Profanity Beeper Requirements Specification.md](Voice%20Profanity%20Beeper%20Requirements%20Specification.md)  
**Implementation Plan:** [docs/implementation_plan.md](docs/implementation_plan.md)

---

## Architecture Overview

1. **Streaming Audio Decode (`censor/audio.py`):** In-memory PyAV decoding (`16 kHz` mono for ASR + `44.1 kHz` mono for output) with a hard `120 s` streaming cap and immediate `finally:` cleanup for any container fallback temp file.
2. **Word-Level ASR (`faster-whisper`):** CPU `int8` transcription (`cpu_threads=2`, `word_timestamps=True`, `vad_filter=True`).
3. **Two-Tier Lexicon + Context Classifier (`censor/lexicon.py`, `censor/detector.py`):**
   - **Mild intensifiers** (`hell`, `damn`, `crap`, `heck`) and clean words (`class`, `assess`, `Dickens`) stay `CLEAN` (never beeped, even if slider = `0.0`).
   - **Tier A** (unambiguous profanity + `*`-masked ASR tokens) skips the classifier (`conf = 0.99`) and is **always beeped** regardless of the slider.
   - **Tier B** (context-dependent words: `ass`, `dick`, `bitch`, `cock`, etc.) is disambiguated over a `±5`-word context window via **ONNX INT8 DistilBERT** (`onnxruntime` + `tokenizers`, zero `torch` in deployment) or **joblib TF-IDF fallback**; beeped when `conf >= threshold` (default `τ*` tuned for validation F2 in `detector_config.json`).
4. **Click-Free Beep Synthesis (`censor/beep.py`):** Clamps timestamps, enforces 50 ms minimum span, pads `±100 ms`, merges intervals with gap `≤ 50 ms`, and synthesizes a 1 kHz sine beep (amplitude `0.3`) with `5 ms` crossfades.
5. **Streamlit App (`app.py`):** Caches models once via `@st.cache_resource` and caches `AnalysisResult` in `st.session_state` by SHA-256 `audio_hash` so moving the sensitivity slider re-renders WAV audio in milliseconds without re-running Whisper.

---

## Local Setup (Windows / Python 3.12)

```powershell
# 1. Create and activate Python 3.12 venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install runtime + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Rebuild lexicon (if editing scripts/build_lexicon.py)
python scripts/build_lexicon.py

# 4. Run unit tests
pytest -v

# 5. Launch local Streamlit app
streamlit run app.py
```

### Local Fallback HTTPS Tunnel (NFR-4)
Browser microphone access (`st.audio_input`) requires HTTPS outside `localhost`. For an instant public HTTPS URL during a live demo:
```powershell
streamlit run app.py --server.port 8501
cloudflared tunnel --url http://localhost:8501
```

---

## Training & Evaluation Workflow

1. **Kaggle NB01 (`notebooks/01_classical_tfidf_10fold.ipynb`, CPU):**
   - Attach `jigsaw-toxic-comment-classification-challenge`, enable Internet, add Kaggle Secret `HF_TOKEN` (Write), and set `HF_REPO_ID = "<HF_USER>/asc-models"`.
   - Evaluates `DummyClassifier`, Lexicon-only (Tier A & Tier A+B), Variation 1 (unweighted) vs. Variation 2 (`class_weight='balanced'`) across 10-fold CV and official test set (`63,978` rows).
   - Exports `tfidf_vectorizer.pkl`, `best_model.pkl`, `best_classical_pipeline.pkl`, and `detector_config.json` to Hugging Face Hub.
2. **Kaggle NB02 (`notebooks/02_transformers.ipynb`, GPU T4 x2 with `CUDA_VISIBLE_DEVICES=0`):**
   - Evaluates `unitary/toxic-bert` (in-domain pre-trained), `cardiffnlp/twitter-roberta-base-offensive` (cross-domain zero-shot), and fine-tuned `distilbert/distilbert-base-uncased` (`2` epochs, `fp16`).
   - Tunes `τ*` maximizing F2 on the 10% validation split, exports & quantizes `distilbert_int8.onnx` (~65 MB), checks FP32 vs. INT8 parity, and pushes `distilbert_int8.onnx`, `tokenizer.json`, and `detector_config.json` to Hugging Face Hub.
3. **NB03 (`notebooks/03_asr_eval.ipynb`):** Compares `faster-whisper` `tiny.en`, `base.en`, `small.en` (`int8`, CPU) on 200 LibriSpeech `test-clean` clips (WER, RTF, peak RAM) and profanity recall.
4. **TTS & Real-World E2E Eval (`scripts/gen_tts_testset.py`, `scripts/run_e2e_eval.py`, `notebooks/04_e2e_eval.ipynb`):**
   ```powershell
   python scripts/gen_tts_testset.py
   python scripts/run_e2e_eval.py --hf-repo-id "<HF_USER>/asc-models" --hf-token "<READ_TOKEN>"
   ```

---

## Streamlit Community Cloud Deployment

1. Push `app.py`, `requirements.txt`, `.streamlit/config.toml`, `censor/`, and `lexicon/` to a **private GitHub repository**.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing to `app.py` and **select Python 3.12 under Advanced settings**.
3. Under **App Settings → Secrets**, add:
   ```toml
   HF_REPO_ID = "<HF_USER>/asc-models"
   HF_TOKEN = "<your_hf_read_token>"
   ASR_MODEL_SIZE = "base.en"
   ```
4. Immediately open **App Settings → Sharing → set visibility to Public** so anyone can open the `*.streamlit.app` URL without logging in while the GitHub repo stays private.
5. Copy the live `https://<app>.streamlit.app` URL into the **very top markdown cell** of `notebooks/01_classical_tfidf_10fold.ipynb` and `notebooks/02_transformers.ipynb`.
