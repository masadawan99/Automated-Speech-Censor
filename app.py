"""Automated Speech Censor (Voice Profanity Beeper) — Streamlit Web Application."""

from __future__ import annotations

import hashlib
import os
import time

import pandas as pd
import streamlit as st

from censor import (
    AnalysisResult,
    AudioDecodeError,
    AudioTooLongError,
    BaseDetector,
    ProfanityLexicon,
    WordEvent,
    analyze_audio,
    load_detector,
    render_censored_audio,
)

_MD_ESCAPE_CHARS = ("\\", "*", "_", "[", "]", "`", "$")


def _escape_markdown(text: str) -> str:
    escaped = text
    for ch in _MD_ESCAPE_CHARS:
        escaped = escaped.replace(ch, f"\\{ch}")
    return escaped


def get_config_value(key: str, default: str = "") -> str:
    """Safely read a config value from ``st.secrets`` or environment variables."""
    try:
        if key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass
    return os.environ.get(key, default).strip()


def format_colored_transcript(words: list[WordEvent], threshold: float) -> str:
    """Format words using Streamlit native markdown color highlights (no raw HTML)."""
    if not words:
        return "_No speech detected in the audio._"

    parts: list[str] = []
    for w in words:
        safe_word = _escape_markdown(w.word)
        beeped = (w.tier == "TIER_A") or (w.tier == "TIER_B" and w.conf >= threshold)
        if beeped:
            parts.append(f":red-background[{safe_word} ({w.conf:.2f})]")
        elif w.tier == "TIER_B":
            parts.append(f":orange-background[{safe_word} ({w.conf:.2f})]")
        else:
            parts.append(safe_word)
    return " ".join(parts)


@st.cache_resource(show_spinner="Loading profanity lexicon...")
def get_lexicon() -> ProfanityLexicon:
    return ProfanityLexicon.from_csv()


@st.cache_resource(show_spinner="Loading Whisper ASR model (first boot only)...")
def get_asr_model(model_size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=2)


@st.cache_resource(show_spinner="Loading context profanity detector...")
def get_detector(hf_repo_id: str, hf_token: str, preferred_backend: str) -> BaseDetector:
    return load_detector(
        hf_repo_id=hf_repo_id or None,
        hf_token=hf_token or None,
        local_dir="models",
        preferred_backend=preferred_backend or "auto",
    )


def _select_active_audio(
    mic_file,
    uploaded_file,
) -> tuple[bytes, str] | None:
    """Return ``(audio_bytes, filename)`` for whichever input widget changed most recently."""
    mic_bytes = mic_file.getvalue() if mic_file is not None else None
    up_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

    mic_hash = hashlib.sha256(mic_bytes).hexdigest() if mic_bytes else None
    up_hash = hashlib.sha256(up_bytes).hexdigest() if up_bytes else None

    prev_mic_hash = st.session_state.get("mic_hash")
    prev_up_hash = st.session_state.get("upload_hash")

    st.session_state["mic_hash"] = mic_hash
    st.session_state["upload_hash"] = up_hash

    if mic_hash and mic_hash != prev_mic_hash:
        st.session_state["active_source"] = "mic"
    elif up_hash and up_hash != prev_up_hash:
        st.session_state["active_source"] = "upload"

    active = st.session_state.get("active_source")
    if active == "mic" and mic_bytes is not None:
        return mic_bytes, getattr(mic_file, "name", "mic_recording.wav") or "mic_recording.wav"
    if active == "upload" and up_bytes is not None:
        return up_bytes, getattr(uploaded_file, "name", "uploaded_audio.wav") or "uploaded_audio.wav"
    if mic_bytes is not None:
        return mic_bytes, getattr(mic_file, "name", "mic_recording.wav") or "mic_recording.wav"
    if up_bytes is not None:
        return up_bytes, getattr(uploaded_file, "name", "uploaded_audio.wav") or "uploaded_audio.wav"
    return None


def main() -> None:
    st.set_page_config(
        page_title="Automated Speech Censor",
        page_icon="🔇",
        layout="wide",
    )

    st.title("Automated Speech Censor (Voice Profanity Beeper)")
    st.caption(
        "Detect and beep English profanity with a 1 kHz tone using Whisper word timestamps "
        "and a two-tier lexicon + context classifier."
    )

    asr_size = get_config_value("ASR_MODEL_SIZE", "base.en")
    hf_repo_id = get_config_value("HF_REPO_ID", "")
    hf_token = get_config_value("HF_TOKEN", "")
    preferred_backend = get_config_value("DETECTOR_BACKEND", "auto")

    lexicon = get_lexicon()
    detector = get_detector(hf_repo_id, hf_token, preferred_backend)
    asr_model = get_asr_model(asr_size)

    if "threshold" not in st.session_state:
        st.session_state["threshold"] = float(detector.threshold)

    with st.sidebar:
        st.header("Censor Settings")
        threshold = st.slider(
            "Tier B Sensitivity Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="threshold",
            help=(
                "Controls context-dependent (Tier B) words such as 'ass', 'dick', 'bitch', 'cock'. "
                "Lower = stricter (more beeps). Tier A unambiguous profanity is always beeped; "
                "mild words ('hell', 'damn', 'crap') and clean words are never beeped."
            ),
        )
        st.divider()
        st.markdown("**Active Pipeline**")
        st.write(f"- ASR: `faster-whisper-{asr_size}` (`int8`, CPU)")
        st.write(f"- Detector: `{detector.model_name}` (`{detector.backend}`)")
        st.write(f"- Tuned default threshold: `{detector.threshold:.2f}`")
        st.write("- Max duration: `120 s`")
        st.divider()
        st.info(
            "**Privacy Notice:** Uploaded and recorded audio is processed strictly in memory "
            "(any OS container fallback temp file is deleted immediately in a `finally:` block) "
            "and is never stored persistently or sent to third-party APIs."
        )

    tab_mic, tab_upload = st.tabs(["Record Microphone", "Upload Audio File"])
    with tab_mic:
        mic_file = st.audio_input("Record English speech (up to 120 seconds)", sample_rate=44100)
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an audio file (WAV, MP3, M4A/AAC, MP4, WebM, Ogg)",
            type=["wav", "mp3", "m4a", "aac", "mp4", "webm", "ogg"],
        )

    selected = _select_active_audio(mic_file, uploaded_file)
    if selected is None:
        st.info("Record from your microphone or upload an audio clip above to get started.")
        return

    audio_bytes, filename = selected
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if st.session_state.get("last_audio_hash") != audio_hash:
        with st.spinner("Transcribing and analyzing speech..."):
            try:
                analysis = analyze_audio(
                    audio_bytes=audio_bytes,
                    filename=filename,
                    asr_model=asr_model,
                    detector=detector,
                    lexicon=lexicon,
                )
            except AudioTooLongError as exc:
                st.error(f"Audio too long: {exc}")
                return
            except AudioDecodeError as exc:
                st.error(f"Audio decode failed: {exc}")
                return
            except Exception as exc:
                st.error(f"Unexpected processing error: {exc}")
                return

        st.session_state["last_audio_hash"] = audio_hash
        st.session_state["analysis"] = analysis

    analysis: AnalysisResult | None = st.session_state.get("analysis")
    if analysis is None:
        return

    t0_render = time.perf_counter()
    censored_wav, flagged_rows = render_censored_audio(analysis, threshold=threshold)
    render_time_s = time.perf_counter() - t0_render

    beeped_count = sum(1 for row in flagged_rows if row["beeped"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Audio Duration", f"{analysis.duration_s:.1f} s")
    c2.metric("Words Censored", str(beeped_count))
    c3.metric("ASR Time", f"{analysis.asr_time_s:.2f} s")
    c4.metric("Classifier Time", f"{analysis.clf_time_s * 1000:.0f} ms")
    c5.metric("Beep Render", f"{render_time_s * 1000:.0f} ms")

    st.subheader("Cleaned Audio")
    st.audio(censored_wav, format="audio/wav")
    stem = os.path.splitext(filename)[0] or "censored"
    st.download_button(
        label="Download Censored WAV",
        data=censored_wav,
        file_name=f"{stem}_censored.wav",
        mime="audio/wav",
    )

    st.subheader("Transcript")
    st.caption(
        "Red = beeped | Orange = Tier B candidate below threshold (uncensored) | "
        "Plain = clean or mild excluded"
    )
    st.markdown(format_colored_transcript(analysis.words, threshold=threshold))

    st.subheader("Flagged Candidate Words")
    if flagged_rows:
        st.dataframe(pd.DataFrame(flagged_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No Tier A or Tier B profanity candidates detected.")


if __name__ == "__main__":
    main()
