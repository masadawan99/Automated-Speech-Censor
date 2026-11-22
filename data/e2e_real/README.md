# Real-World Evaluation Recordings (`data/e2e_real/`)

Record **20 to 30 short English audio clips** (5–20 seconds each, recorded by you and friends in normal speaking environments) and save them inside this folder (`data/e2e_real/`).

Supported formats: `.wav`, `.mp3`, `.m4a`, `.aac`, `.mp4`, `.webm`, `.ogg`, `.flac`.

## Recommended Clip Mix (20–30 clips total)
1. **Tier A profanity (6–8 clips):** Clear unambiguous profanity (e.g., *"what the fuck was that"*, *"this is total bullshit"*, *"stop acting like an asshole"*).
2. **Tier B profane uses (5–7 clips):** Context-dependent words used profanely (e.g., *"don't be a dick to him"*, *"she is such a bitch"*, *"get your ass over here"*).
3. **Tier B tricky-clean clips (5–8 clips):** Same/similar words in harmless contexts or Scunthorpe cases (e.g., *"I am taking a machine learning class"*, *"Dick Grayson is my favorite character"*, *"The rooster or cock crowed at dawn"*, *"Let's assess the model accuracy"*, *"Order a fruit cocktail"*).
4. **Mild-word clean clips (4–7 clips):** Mild intensifiers intentionally left uncensored (e.g., *"what the hell is going on"*, *"damn that was fast"*, *"this old script is crap"*).

## Ground-Truth Labelling (`labels.csv`)
Edit `data/e2e_real/labels.csv` with one row per audio clip:
- `file`: Filename relative to `data/e2e_real/` (e.g. `clip_01.wav`).
- `profane_words`: Space- or semicolon-separated list of words that **should** be beeped (leave empty `""` for clean/mild-only clips).

Example:
```csv
file,profane_words
clip_01.wav,fuck
clip_02.wav,dick asshole
clip_03_clean_class.wav,
clip_04_mild_hell.wav,
```
