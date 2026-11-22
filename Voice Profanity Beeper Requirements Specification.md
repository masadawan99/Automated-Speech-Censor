# Automated Speech Censor (Voice Profanity Beeper): Requirements Specification

*Purpose: input for an architecture-design agent. This document states what must be built and why. It deliberately does not prescribe the implementation; where a direction is suggested it is marked **Proposed** and may be challenged.*

## 1. Project summary

**Automated Speech Censor** is a web application that takes English speech audio (recorded in the browser or uploaded as a file), detects profane words, replaces them with a beep, and returns the cleaned audio to play or download. It is a university machine learning semester project and is also to be framed as a product prototype for streaming, live game chat and TV broadcasting, which already run with a built-in broadcast delay.

## 2. Context and constraints

| Item | Requirement |
| --- | --- |
| Course | Machine Learning semester project (Task 2: select an ML problem on unstructured data) |
| Deadline | Sunday, 29 Nov 2026 at 23:59 (5%/day late penalty; 0 marks after 3 days) |
| Team | Solo (one developer, working with agentic coding tools — allowed provided the student understands the work) |
| Language | English only |
| Processing mode | Offline (file in, cleaned file out). Real-time is explicitly out of scope for the deliverable |
| Training compute | Kaggle notebooks, GPU T4 x2 |
| Deployment budget | Zero. Free tier only; it must run smoothly during the live presentation, not at production scale |
| Instructor guidance | Prefer existing pre-trained models and weights (Hugging Face and similar); fine-tuning them is encouraged |

## 3. Course rubric (must be satisfied)

1. **Problem selection:** an ML problem that uses unstructured data (audio here).
2. **Dataset:** download and use the full dataset for the chosen problem.
3. **Technique:** any of feature extraction, pre-trained transformer, pre-trained LLM, or fine-tuning an existing LLM.
4. **Apply, evaluate and compare:** evaluate multiple ML algorithms, or transformers, or LLMs on the full dataset, using either a train-test split or 10-fold cross-validation.
5. **Deployment:** deploy the model to the cloud and integrate it with a web application.

The instructor's reference example is an Emotion Prediction System: TF-IDF features, 10-fold CV, Random Forest, Naive Bayes, k-NN and SVM compared, best model reported as 78% accuracy. The grader therefore expects a clear **model comparison table with an accuracy score** and a named best model.

## 4. Users and product framing

- **Primary customers:** live-stream platforms, game chat/voice moderation and TV broadcasters that already operate with a delay buffer.
- **Secondary users:** podcasters, YouTubers and educators cleaning recordings.
- The deliverable is a **prototype with a product story**. Product features beyond the MVP (section 6) are roadmap items to be described, not necessarily built.

## 5. Functional requirements (MVP)

| ID | Requirement |
| --- | --- |
| FR-1 | The user can record audio in the browser (microphone) or upload an audio file |
| FR-2 | Supported upload formats: at least WAV, MP3, M4A/AAC, WebM/Ogg (browser recordings). Files are normalised internally |
| FR-3 | Maximum input length is capped (target about 2 minutes) with a clear error beyond that |
| FR-4 | The system transcribes the audio with word-level timestamps |
| FR-5 | The system decides, per word, whether it is profane and assigns a confidence score |
| FR-6 | Every word judged profane is replaced by a 1 kHz sine beep covering the word span with ~100 ms padding each side and 5 ms fades |
| FR-7 | The user can play the cleaned audio in the page and download it as WAV |
| FR-8 | The page shows the transcript with beeped words highlighted and each flagged word's confidence |
| FR-9 | A sensitivity threshold slider (0–1, default 0.5) re-applies the beep decision on the already-computed transcript and scores without re-transcribing |
| FR-10 | The page reports processing time and the number of words censored |
| FR-11 | Uploaded/generated files live only in the app's temp cache and are auto-deleted (Gradio `delete_cache`, ≤10 min); the page states this |

### Roadmap (describe, do not necessarily build)

Live/chunked mode using a delay buffer; manual review screen (un-beep / add words); user-defined word lists; choice of beep, silence or custom sound; Urdu and Roman Urdu support; API access and batch processing; per-channel profiles.

## 6. Machine learning requirements

### 6.1 Speech-to-text

- Must produce word-level timestamps.
- **Proposed:** an existing pre-trained ASR model (Whisper family via faster-whisper or similar) used without training. Compare at least three sizes or models on word error rate and processing speed, and choose one with justification.
- Known risk to verify early: ASR models may soften, drop or mis-transcribe profanity, which would cause misses. The architecture must allow swapping the ASR component.

### 6.2 Profanity detection

- Output per word: a profane/clean decision and a confidence score between 0 and 1.
- **Proposed:** two steps. First, a curated English lexicon (including inflections and common spoken variants) flags candidate words, keeping recall high. Second, a trained text classifier uses the surrounding context to confirm or reject each candidate, reducing false beeps (for example names, "class", mild words in harmless use).
- **Proposed training data:** the Jigsaw Toxic Comment Classification dataset (labels such as `obscene` and `toxic`), used in full. The architect may propose a better-suited public dataset if one is justified.
- A lexicon-only baseline must be included in the comparison.

### 6.3 Models to compare (minimum)

| Group | Candidates |
| --- | --- |
| Classical ML on TF-IDF | Logistic Regression, Naive Bayes, SVM, Random Forest |
| Pre-trained, no fine-tuning | for example `unitary/toxic-bert`, `cardiffnlp/twitter-roberta-base-offensive` (availability to be verified) |
| Fine-tuned transformer | DistilBERT (RoBERTa-base if time allows) |
| Baseline | Lexicon only |

### 6.4 Evaluation requirements

- Use 10-fold cross-validation for the classical models. For fine-tuned transformers a single fixed stratified train-test split is acceptable because of compute limits. The protocol must be stated explicitly and applied consistently within each group.
- Report accuracy (required by the instructor), plus precision, recall and F1. The dataset is imbalanced, so accuracy alone is not sufficient.
- **Recall is prioritised over precision:** a missed profanity is worse than a false beep. The report must say so.
- **ASR evaluation:** word error rate and real-time factor (processing time divided by audio duration) for each ASR candidate.
- **End-to-end audio evaluation:** a test set of roughly 100 to 150 clips, mostly TTS-generated with known text and timings, plus 20 to 30 real recordings (the developer and friends) with profane words labelled by hand. Metrics: word-level beep precision, recall and F1, timing error of beep boundaries, and real-time factor.
- The report must state that synthetic audio is cleaner than real speech and discuss the gap.

## 7. Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | Runs on CPU only in deployment (no GPU available) |
| NFR-2 | A 1-minute clip should be processed in a time suitable for a live demo (target under about 30 s; to be measured and reported) |
| NFR-3 | Free-tier hosting; no payment card should be required. Hugging Face Spaces is the leading candidate; the architect may propose alternatives |
| NFR-4 | A fallback demo path must exist (for example running locally behind a tunnel) in case the free host sleeps, is slow or the venue network fails |
| NFR-5 | Works on desktop and mobile browsers, including microphone permission handling |
| NFR-6 | Model artefacts are loaded once at startup, not per request |
| NFR-7 | Reproducibility: training is done in Kaggle notebooks with fixed seeds, and results are re-creatable from the notebooks |
| NFR-8 | Privacy: no persistent storage of user audio; no third-party transmission of audio |
| NFR-9 | Must be buildable by one person in about one week using AI coding assistants; prefer few moving parts |

## 8. Deliverables

1. Training and evaluation notebooks (Kaggle) with the model comparison table.
2. The selected model(s) packaged for inference.
3. A deployed web application (public URL).
4. A project report following the rubric: problem, dataset, techniques, comparison and evaluation, deployment.
5. A presentation with a live demo and a short product pitch.
6. Google Classroom submission: upload code, dataset, and trained model file, with the live cloud deployment link placed at the very top of the main Jupyter notebook.

## 9. Acceptance criteria

- A user can record or upload a clip with profanity and receive a downloadable cleaned file with the profane words beeped.
- The comparison table covers at least four models across at least two technique groups, with accuracy, precision, recall and F1.
- The deployed URL works from a device other than the developer's.
- End-to-end word-level recall on the audio test set is reported, with the failure cases discussed.
- Audio is not retained after processing.

## 10. What the architect should produce

- Component diagram and data flow from audio input to cleaned audio.
- The choice of ASR, detector and beep-insertion approach, with trade-offs against the constraints above.
- API contract between frontend and backend.
- Model packaging and loading strategy for CPU-only free hosting.
- Repository structure and the split between training notebooks and the serving app.
- A risk list with mitigations (ASR softening profanity, free-host cold starts, timestamp drift, short deadline) and a cut-line showing which requirements are dropped first if time runs out.

## 11. Resolved decisions

- **Team:** Solo developer.
- **Dataset:** Jigsaw Toxic Comment Classification Challenge (`obscene` label, binary classification) via Kaggle competition dataset (fallback: Hugging Face mirror `thesofakillers/jigsaw-toxic-comment-classification-challenge`).
- **Lexicon:** LDNOOBW English bad-word list (`en`, single words) + hand-curated spoken variants/inflections, organised into two tiers (Tier A unambiguous, Tier B context-dependent).
- **ASR:** `faster-whisper` (`Systran/faster-whisper-tiny.en`, `base.en`, `small.en` compared on WER and RTF; best trade-off selected).
- **Hosting:** Hugging Face Spaces (Gradio SDK, free CPU basic tier, no payment card required).
- **Slides:** Prepared by the developer at the end (following the `ppt_Semester Project.pptx` structure; Course Feedback left for the developer).
