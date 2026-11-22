"""Sequential edge-tts generator (with pyttsx3 fallback) for end-to-end audio evaluation.

Generates ~120 curated clips across 5 categories:
  - tier_a_profane
  - tier_b_profane
  - tier_b_tricky_clean
  - mild_clean
  - general_clean

Saves audio files, word-boundary JSON timings, and ``data/e2e_tts/manifest.json``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import wave

import edge_tts

VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-AU-WilliamNeural",
]


@dataclass(frozen=True)
class PromptSpec:
    clip_id: str
    category: str
    text: str
    expected_beep_words: tuple[str, ...]


def _build_prompts() -> list[PromptSpec]:
    tier_a_sentences = [
        ("What the fuck was that noise outside?", ("fuck",)),
        ("Stop talking bullshit during the meeting.", ("bullshit",)),
        ("He is a complete asshole in traffic.", ("asshole",)),
        ("Why is this fucking computer crashing again?", ("fucking",)),
        ("That entire presentation was pure shit.", ("shit",)),
        ("You fucked up the deployment script.", ("fucked",)),
        ("Tell that dumb motherfucker to leave now.", ("motherfucker",)),
        ("Stop acting like a total dumbass today.", ("dumbass",)),
        ("This shitty microphone keeps cutting out.", ("shitty",)),
        ("Do not be a jackass at dinner.", ("jackass",)),
        ("She called him a worthless dickhead.", ("dickhead",)),
        ("Everything on that server went to shit.", ("shit",)),
        ("He keeps bullshitting the whole team.", ("bullshitting",)),
        ("What a bunch of horseshit excuses.", ("horseshit",)),
        ("That guy is a massive scumbag.", ("scumbag",)),
        ("Quit messing around and fix the fucking bug.", ("fucking",)),
        ("I cannot stand that arrogant cunt.", ("cunt",)),
        ("Stop being a douchebag to your coworkers.", ("douchebag",)),
        ("The whole system turned into a shitshow.", ("shitshow",)),
        ("They fucked over every single customer.", ("fucked",)),
        ("No one cares about your dipshit opinion.", ("dipshit",)),
        ("He shouted fuck when the glass broke.", ("fuck",)),
        ("Why did you leave this bullshit on my desk?", ("bullshit",)),
        ("She is tired of his shitty attitude.", ("shitty",)),
        ("That wanker broke the coffee machine again.", ("wanker",)),
        ("Get your fucking hands off my laptop.", ("fucking",)),
        ("He acted like an absolute asshole yesterday.", ("asshole",)),
        ("This code is fucked beyond repair.", ("fucked",)),
        ("Stop talking shit behind my back.", ("shit",)),
        ("You are such a stubborn fucker.", ("fucker",)),
        ("What a total clusterfuck of a morning.", ("clusterfuck",)),
        ("Don't give me that apeshit excuse again.", ("apeshit",)),
        ("He threw a fit and yelled bullshit.", ("bullshit",)),
        ("Tell the asshole to move his car.", ("asshole",)),
        ("Another fucking delay on the train line.", ("fucking",)),
        ("Don't be such a dick to the new intern.", ("dick",)),
        ("Move your lazy ass out of the doorway.", ("ass",)),
        ("Stop whining like a spoiled bitch.", ("bitch",)),
        ("That lying bastard stole my parking spot.", ("bastard",)),
        ("She called her coworker a greedy whore.", ("whore",)),
        ("Shut your mouth you pathetic slut.", ("slut",)),
        ("Quit acting like a coward and a pussy.", ("pussy",)),
        ("He pissed me off with that rude comment.", ("pissed",)),
        ("You are acting like a real douche right now.", ("douche",)),
        ("He kept bragging like a total cock.", ("cock",)),
    ]

    tier_b_profane_sentences = [
        ("He is an arrogant prick who never listens.", ("prick",)),
        ("Stop being a spineless prick to your friends.", ("prick",)),
        ("That prick refused to pay his share.", ("prick",)),
        ("She told that prick to get lost.", ("prick",)),
        ("Those pricks blocked the entire driveway.", ("pricks",)),
        ("Stop acting like a bunch of selfish pricks.", ("pricks",)),
        ("Why did those pricks delete our database?", ("pricks",)),
        ("Suck my balls and leave me alone.", ("balls",)),
        ("He kicked him square in the balls during the fight.", ("balls",)),
        ("She yelled shut up you cheap hoe.", ("hoe",)),
        ("He insulted her and called her a dirty hoe.", ("hoe",)),
        ("Tell those loud hoes to get out of the lobby.", ("hoes",)),
        ("He keeps bragging about chasing hoes every weekend.", ("hoes",)),
        ("Quit acting like a shameless tramp at the bar.", ("tramp",)),
        ("She called him a worthless tramp to his face.", ("tramp",)),
        ("Get lost you miserable sod.", ("sod",)),
        ("Look at that lazy sod sleeping on the couch.", ("sod",)),
        ("He acted like an ignorant boob on live television.", ("boob",)),
        ("Stop staring like a foolish boob during the meeting.", ("boob",)),
        ("She flipped out and called him a pathetic tit.", ("tit",)),
    ]

    tier_b_tricky_clean_sentences = [
        ("I am attending my machine learning class today.", ()),
        ("Keep your eyes on the tennis balls during practice.", ()),
        ("We should assess the model precision carefully.", ()),
        ("The soccer balls were pumped before the match.", ()),
        ("Be careful not to prick your finger on that rose thorn.", ()),
        ("The nurse felt a tiny pin prick on his arm.", ()),
        ("Use the garden hoe to clear those weeds by the fence.", ()),
        ("Please pass me a glass of fruit cocktail.", ()),
        ("The pilot checked every gauge inside the cockpit.", ()),
        ("Charles Dickens wrote many famous Victorian novels.", ()),
        ("My cat pussyfooting around the kitchen was adorable.", ()),
        ("We visited Scunthorpe during our summer train trip.", ()),
        ("The assistant professor graded every assignment early.", ()),
        ("Santa laughed ho ho ho as he entered the room.", ()),
        ("That classical concert lasted nearly two hours.", ()),
        ("They ordered peacock feathers for the stage costume.", ()),
        ("A small titmouse landed on the garden branch.", ()),
        ("The landscapers laid fresh sod across the front lawn.", ()),
        ("He is a skilled craftsman who repairs brass clocks.", ()),
        ("The weathercock spun rapidly in the evening wind.", ()),
        ("Our class discussed baseline accuracy versus recall.", ()),
        ("We watched the classic Disney movie Lady and the Tramp.", ()),
        ("The shuttlecock flew high over the badminton net.", ()),
        ("That young puppy still has plenty of spunk and energy.", ()),
        ("Please classify these documents by publication date.", ()),
    ]

    mild_clean_sentences = [
        ("What the hell happened to the projector?", ()),
        ("Damn, that espresso shot was strong!", ()),
        ("This old keyboard is total crap now.", ()),
        ("Oh heck, I forgot my notebook at home.", ()),
        ("Why the hell is the printer jammed again?", ()),
        ("Damn it, we missed the early bus.", ()),
        ("Quit complaining about that crappy weather.", ()),
        ("Who in the hell left the lights on?", ()),
        ("That was a damn good soccer match.", ()),
        ("What the heck are they building outside?", ()),
        ("My battery died, oh darn it.", ()),
        ("He muttered damn under his breath.", ()),
        ("That cheap cable is a piece of crap.", ()),
        ("How the hell did you solve that integral?", ()),
        ("Damn, I dropped my keys in the hallway.", ()),
    ]

    general_clean_sentences = [
        ("Good morning everyone, welcome to the project demo.", ()),
        ("Please speak clearly into the laptop microphone.", ()),
        ("The quarterly sales report is ready for review.", ()),
        ("We recorded twenty clean samples in the quiet lab.", ()),
        ("Streamlit caches the trained weights at startup.", ()),
        ("Our evaluation measures precision, recall, and latency.", ()),
        ("Thank you for testing the audio moderation tool.", ()),
        ("Python three point twelve runs smoothly on Windows.", ()),
        ("The afternoon train arrives at platform number four.", ()),
        ("Let us compare classical baselines against transformers.", ()),
        ("Always validate timestamps before rendering audio.", ()),
        ("Coffee and tea are available near the conference room.", ()),
        ("Remember to save your checkpoints before shutting down.", ()),
        ("Cloud hosting requires keeping memory under one gigabyte.", ()),
        ("Have a wonderful weekend and drive safely home.", ()),
    ]

    prompts: list[PromptSpec] = []
    groups = [
        ("tier_a_profane", tier_a_sentences),
        ("tier_b_profane", tier_b_profane_sentences),
        ("tier_b_tricky_clean", tier_b_tricky_clean_sentences),
        ("mild_clean", mild_clean_sentences),
        ("general_clean", general_clean_sentences),
    ]
    idx = 1
    for category, items in groups:
        for text, expected in items:
            prompts.append(
                PromptSpec(
                    clip_id=f"tts_{idx:03d}_{category}",
                    category=category,
                    text=text,
                    expected_beep_words=expected,
                )
            )
            idx += 1
    return prompts


PROMPTS: list[PromptSpec] = _build_prompts()


async def _synthesize_edge_tts(
    text: str,
    voice: str,
    audio_out: Path,
) -> list[dict]:
    communicate = edge_tts.Communicate(text, voice=voice, boundary="WordBoundary")
    word_events: list[dict] = []
    audio_chunks: list[bytes] = []

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            # edge-tts offsets/durations are in 100-nanosecond ticks (1e7 ticks/sec)
            start_s = float(chunk["offset"]) / 10_000_000.0
            dur_s = float(chunk["duration"]) / 10_000_000.0
            word_events.append(
                {
                    "word": str(chunk["text"]),
                    "start": round(start_s, 4),
                    "end": round(start_s + max(0.05, dur_s), 4),
                }
            )

    audio_out.write_bytes(b"".join(audio_chunks))
    return word_events


def _synthesize_pyttsx3_fallback(text: str, audio_out: Path) -> list[dict]:
    """Word-by-word Windows SAPI fallback retaining exact timing boundaries."""
    import numpy as np
    import pyttsx3
    import soundfile as sf

    engine = pyttsx3.init()
    words = [w for w in text.split() if w.strip()]
    tmp_dir = audio_out.parent / "_tmp_sapi"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    segments: list[np.ndarray] = []
    events: list[dict] = []
    target_sr = 22050
    cursor_s = 0.0
    gap = np.zeros(int(0.08 * target_sr), dtype=np.float32)

    try:
        for i, w in enumerate(words):
            tmp_wav = tmp_dir / f"w_{i}.wav"
            engine.save_to_file(w, str(tmp_wav))
            engine.runAndWait()
            if not tmp_wav.exists():
                continue
            data, sr = sf.read(str(tmp_wav), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            target_sr = sr
            gap = np.zeros(int(0.08 * target_sr), dtype=np.float32)
            dur = len(data) / float(sr)
            events.append({"word": w, "start": round(cursor_s, 4), "end": round(cursor_s + dur, 4)})
            segments.append(data)
            segments.append(gap)
            cursor_s += dur + 0.08
            tmp_wav.unlink(missing_ok=True)

        if not segments:
            raise RuntimeError("pyttsx3 generated zero audio segments")
        full = np.concatenate(segments)
        with wave.open(str(audio_out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            pcm16 = (np.clip(full, -1.0, 1.0) * 32767.0).astype(np.int16)
            wf.writeframes(pcm16.tobytes())
    finally:
        for p in tmp_dir.glob("*"):
            p.unlink(missing_ok=True)
        if tmp_dir.exists():
            tmp_dir.rmdir()

    return events


async def generate_dataset(
    out_dir: Path,
    limit: int | None = None,
    delay_s: float = 0.4,
    fallback_pyttsx3: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    selected = PROMPTS[:limit] if limit is not None else PROMPTS
    manifest_records: list[dict] = []

    for idx, spec in enumerate(selected):
        voice = VOICES[idx % len(VOICES)]
        ext = "wav" if fallback_pyttsx3 else "mp3"
        audio_path = clips_dir / f"{spec.clip_id}.{ext}"
        meta_path = clips_dir / f"{spec.clip_id}.json"

        # Skip-if-exists checkpointing
        if audio_path.is_file() and meta_path.is_file() and audio_path.stat().st_size > 0:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            manifest_records.append(record)
            continue

        word_timings: list[dict]
        backend_used = "edge-tts"
        if not fallback_pyttsx3:
            try:
                word_timings = await _synthesize_edge_tts(spec.text, voice, audio_path)
            except Exception as exc:
                print(f"[gen_tts] edge-tts failed for {spec.clip_id}: {exc}. Trying pyttsx3...")
                audio_path = clips_dir / f"{spec.clip_id}.wav"
                word_timings = _synthesize_pyttsx3_fallback(spec.text, audio_path)
                backend_used = "pyttsx3"
        else:
            word_timings = _synthesize_pyttsx3_fallback(spec.text, audio_path)
            backend_used = "pyttsx3"

        record = {
            **asdict(spec),
            "expected_beep_words": list(spec.expected_beep_words),
            "voice": voice if backend_used == "edge-tts" else "sapi5",
            "tts_backend": backend_used,
            "audio_file": str(audio_path.relative_to(out_dir)).replace("\\", "/"),
            "word_boundaries": word_timings,
        }
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        manifest_records.append(record)

        print(f"[{idx + 1}/{len(selected)}] Generated {spec.clip_id} ({backend_used})")
        if delay_s > 0 and backend_used == "edge-tts":
            time.sleep(delay_s)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")
    print(f"[gen_tts] Saved manifest with {len(manifest_records)} clips -> {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS evaluation dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/e2e_tts"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--fallback-pyttsx3", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        generate_dataset(
            out_dir=args.out_dir,
            limit=args.limit,
            delay_s=args.delay,
            fallback_pyttsx3=args.fallback_pyttsx3,
        )
    )


if __name__ == "__main__":
    main()
