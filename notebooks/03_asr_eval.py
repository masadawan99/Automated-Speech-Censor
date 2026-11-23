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
# # Automated Speech Censor — Notebook 03: Whisper ASR Evaluation (WER, RTF, Peak RAM & Profanity Recall)
#
# Compares `Systran/faster-whisper-tiny.en`, `Systran/faster-whisper-base.en`, and `Systran/faster-whisper-small.en` (`compute_type="int8"`, CPU, `cpu_threads=2`) on:
# 1. **200 streamed utterances** from `openslr/librispeech_asr` (`test-clean`) using `Audio(decode=False)` + `soundfile` decoding (avoids Windows `datasets >= 4.0` `torchcodec` FFmpeg DLL errors).
# 2. **Profanity-word recall** on a TTS profanity subset (`data/e2e_tts/manifest.json`) to quantify whether smaller Whisper sizes drop or mask profanity.

# %%
import gc
import io
import json
from pathlib import Path
import time

try:
    from IPython.display import display
except ImportError:
    display = print

from datasets import Audio, load_dataset
from faster_whisper import WhisperModel
import jiwer
import numpy as np
import pandas as pd
import psutil
import soundfile as sf

MODEL_SIZES = ["tiny.en", "base.en", "small.en"]
N_LIBRISPEECH = 200
TTS_MANIFEST = Path("../data/e2e_tts/manifest.json")
if not TTS_MANIFEST.exists():
    TTS_MANIFEST = Path("data/e2e_tts/manifest.json")

# %% [markdown]
# ## 1. Stream 200 LibriSpeech `test-clean` Utterances (Windows-Safe `Audio(decode=False)`)

# %%
ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

libri_samples = []
for ex in ds:
    raw_bytes = ex["audio"]["bytes"]
    pcm, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    dur_s = len(pcm) / float(sr)
    libri_samples.append({"pcm": pcm, "sr": sr, "dur_s": dur_s, "text": ex["text"]})
    if len(libri_samples) >= N_LIBRISPEECH:
        break

print(f"Loaded {len(libri_samples)} LibriSpeech utterances (total {sum(x['dur_s'] for x in libri_samples):.1f}s)")

# %% [markdown]
# ## 2. Benchmark WER, RTF, Peak RAM (MB) & Profanity-Word Recall

# %%
process = psutil.Process()
normalizer = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

tts_records = []
if TTS_MANIFEST.exists():
    all_tts = json.loads(TTS_MANIFEST.read_text(encoding="utf-8"))
    tts_records = [r for r in all_tts if r["expected_beep_words"]]
    print(f"Loaded {len(tts_records)} profane TTS clips for profanity recall check.")

asr_rows = []
for size in MODEL_SIZES:
    gc.collect()
    rss_before_mb = process.memory_info().rss / (1024 ** 2)
    model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=2)
    peak_rss_mb = process.memory_info().rss / (1024 ** 2)

    refs, hyps = [], []
    total_audio_s = 0.0
    total_proc_s = 0.0

    for item in libri_samples:
        t0 = time.perf_counter()
        segs, _ = model.transcribe(item["pcm"], language="en", word_timestamps=True, vad_filter=True)
        hyp_text = " ".join(s.text.strip() for s in segs).strip()
        total_proc_s += time.perf_counter() - t0
        total_audio_s += item["dur_s"]
        refs.append(normalizer(item["text"]))
        hyps.append(normalizer(hyp_text))
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 ** 2))

    wer = jiwer.wer(refs, hyps)
    rtf = total_proc_s / max(0.01, total_audio_s)

    prof_hits, prof_total = 0, 0
    if tts_records:
        from censor.audio import decode_audio_bytes

        for rec in tts_records:
            ap = TTS_MANIFEST.parent / rec["audio_file"]
            asr_pcm, _, _, _ = decode_audio_bytes(ap.read_bytes(), filename=ap.name)
            segs, _ = model.transcribe(asr_pcm, language="en", word_timestamps=True, vad_filter=True)
            words_out = [w.word.strip().lower() for s in segs for w in (s.words or [])]
            joined = " ".join(words_out)
            for exp_w in rec["expected_beep_words"]:
                prof_total += 1
                if exp_w.lower() in joined or "*" in joined:
                    prof_hits += 1

    prof_recall = (prof_hits / prof_total) if prof_total > 0 else np.nan
    asr_rows.append({
        "ASR_Model": f"faster-whisper-{size} (int8)",
        "WER": round(wer, 4),
        "RTF": round(rtf, 4),
        "Est_1min_Time_s": round(rtf * 60.0, 2),
        "Peak_RSS_MB": round(peak_rss_mb, 1),
        "Delta_RSS_MB": round(peak_rss_mb - rss_before_mb, 1),
        "Profanity_Word_Recall": round(prof_recall, 4) if not np.isnan(prof_recall) else None,
    })
    del model
    gc.collect()

asr_df = pd.DataFrame(asr_rows)
display(asr_df)
asr_df.to_csv("nb03_asr_benchmark.csv", index=False)
