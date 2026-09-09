---
name: audio-analyzer
description: "Audio analysis + lyrics transcription + LLM music producer synthesis + original song creation for Suno AI. Analyzes audio files (MP3/WAV/FLAC/AAC/OGG/M4A) returning key/mode/chords/bassline/drums/EQ/LUFS/BPM/melody/structure, auto vocal transcription with chorus detection, deep producer-perspective synthesis integrating audio features and lyric themes, then creates a brand-new original song (title + full lyrics + Suno style prompt ≤200 chars) imitating the reference track's style and production DNA. Keywords: audio analysis, suno prompt, lyrics transcription, song imitation, lyric creation, producer reference, BPM, chord progression, key, bassline, audio reverse engineering, music theory."
version: "7.2"
changelog: "v7.2 — Production-ready release: 1. Dual-branch architecture (Vocal vs Instrumental) with zero-dependency Whisper hallucination gating (instrumentalness >= 0.7 skips Whisper). 2. Three-channel parallel song identity detection (Whisper chorus line search + AcoustID + filename fallback) + LRClib synced lyrics integration. 3. Lyric Normalizer (breath-point comma formatting, bar-aligned phrasing, Metatag structuring). 4. Instrumental Timeline Arrangement Scaffolding (dynamic section cues in Lyrics box for instrumental tracks). 5. Suno-perceptible theory tag whitelist (dorian mode, 2-5-1, syncopated bass). 6. Artist feature deconstruction (artist_feature_map.yaml) converting banned artist names into rich stylistic descriptors. 7. Auto-exclusion generator (auto_exclusion_map.yaml) for orthogonal negative prompts (<=200 chars). 8. MIDI analysis engine (scripts/parse_midi.py) and prompt compiler (scripts/compile_prompt.py)."
---

# Audio Analyzer v7.2 — Multi-Channel Identification + Theory White-list Compiler + Dual-Branch Delivery for Suno

Full pipeline: audio feature extraction → **Whisper hallucination gate → dual-branch pipeline (vocal / instrumental) → multi-channel identity detection (Whisper/AcoustID/Filename) → LRClib normalization & timeline scaffolding → theory tag compiler & artist deconstruction → Suno prompt & structured delivery**.

---

## Pipeline Overview

```
【Input Audio】
    │
    ▼
Step 0: Fast Audio Features & Hallucination Gate (instrumentalness >= 0.7)
    │
    ├─[instrumentalness >= 0.7]─────────────┐
    │                                       │
    │  【Instrumental Branch】               │  【Vocal Branch】
    │  0b: Audio Fingerprint (AcoustID)     │  0a: Whisper (hallucination guarded) → lyrics line query
    │  0c: Filename fallback                │  0b: Audio Fingerprint (parallel)
    │  (Whisper channel physically closed)  │  0c: Filename fallback
    │                                       │
    └───────────────────┬───────────────────┘
                        ▼
            Step 1: Identity & Lyrics Ground Truth
            - LRClib synced lyrics query & Lyric Normalizer
            - Or Instrumental Timeline Scaffolding
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
Step 1d: Cultural & Artist Profile     Step 1e: Theory & MIDI Analysis
- Artist deconstruction dict            - Suno-perceptible theory tag whitelist
- Production & era context              - MIDI/pyin chord & progression analysis
        │                               │
        └───────────────┬───────────────┘
                        ▼
Step 3: Prompt Compiler (Theory + Audio + Culture Triangulation)
- Style prompts (Safe / Recommended / Experimental, target 150-350 chars)
- Orthogonal Negative Prompt (<=200 chars, auto-exclusion map)
                        │
                        ▼
Step 4: Structured Delivery
【Vocal Branch】                        【Instrumental Branch】
① Normalized Metatag Lyrics             ① Instrumental Timeline Scaffolding
② 3 Version Style Prompts               ② 3 Version Style Prompts
③ Negative Prompt (<=200 chars)         ③ Negative Prompt (<=200 chars)
④ (Optional) sketch.mid                 ④ (Optional) sketch.mid
```

---

## Step 0 & Step 1 — Multi-Channel Identification & Audio Analysis

### 1. Run Core Analysis Script
```bash
python scripts/analyze_audio.py <file_path> > analysis.json
```
- **Whisper Hallucination Gate**: When Essentia detects `instrumentalness >= 0.7`, the script automatically bypasses Whisper transcription, eliminating text looping hallucinations over instrumental or solo passages.

### 2. Identify Song & Format Lyrics
```bash
python scripts/identify_song.py <file_path> analysis.json
```
- **Tri-channel Parallel Matching**: Lyric hook search (LRClib synced lyrics) + Audio fingerprinting (AcoustID) + Filename fallback.
- **Vocal Branch**: Fetches lyrics and runs **Lyric Normalizer** (CJK/English punctuation normalization, breath-point commas, 1-2 bar line wrapping), outputting standardized `[Intro] [Verse] [Chorus] [Outro]` metatags.
- **Instrumental Branch**: Automatically generates an **Instrumental Timeline Arrangement Scaffolding** for the Suno Lyrics box to direct dramatic dynamic builds and drops.

---

## Step 1b — Web Knowledge & Artist Deconstruction 🌐

From the identified track and artist, perform web search or query the local artist deconstruction dictionary (`artist_feature_map.yaml`):
- **Compliance Filter**: Suno systematically intercepts artist names in style prompts.
- **Deconstruction Engine**: Translates matched artists into positive, high-fidelity sound, gear, and theory descriptors (e.g. `The Weeknd` → `dark R&B, vintage 80s analog synth bass, soaring falsetto, pulsing arpeggios`).

---

## Step 1e — MIDI & Music Theory Analysis

When a `.mid` file is supplied or local transcription is performed:
```bash
python scripts/parse_midi.py <midi_file_path>
```
Extracted harmony tags are strictly filtered through the **Suno-Perceptible Theory Tag Whitelist (`theory_tags`)**:
- ✅ **Whitelisted**: `2-5-1 jazz progression`, `four-chord pop loop`, `dorian mode`, `minor pentatonic`, `syncopated bass`, `half-time drum beat`.
- ❌ **Blacklisted**: `tritone substitution`, `hypodorian`, `iv6-V7` (academic labels with no prior distribution in Suno's training data).

---

## Step 2 — Present Analysis Report

Parse the JSON and present all sections clearly. Use this structure:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎛️ PRODUCTION QUICK-START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BPM / Key / Time Sig / Swing / LUFS / Dynamic Range

① Loudness & Dynamics
RMS / LUFS / Peak / Crest Factor / Dynamic Range / Headroom + Compression hints

② Frequency Balance (EQ Reference)
7-band breakdown + % + dB; bass vs treble orientation

③ Rhythm & Groove
BPM / Swing type / Time signature / First 16 beats timeline

④ Drum Pattern (First 8 bars)
Kick / Snare / Hi-Hat onsets + density per bar

⑤ Bassline Sequence
Root note frequency table + opening progression + MIDI suggestions

⑥ Chord Progressions
Core loop (with Roman numerals) / Full track chords / Harmonic transitions

⑦ Tonality & Scale
Primary key / Modal flavor (Phrygian/Dorian/etc.) / Relative key / Chroma energy

⑧ Tonal Timeline (Every 15s)
Time bracket → Key / Confidence / Modulation markers

⑨ Melodic Contour
Pitch range / Dominant melodic notes / Opening contour sequence

⑩ Energy Arc & Song Structure
RMS + Spectral Flux timeline (highlighting drops/peaks)
Inferred sections: Intro / Verse / Build / Chorus / Breakdown / Drop / Outro

⑪ Timbre & Spectral Profile
Centroid / Bandwidth / Rolloff / Harmonic ratio / Brightness descriptors

⑫ Lyrics & Vocals
Language / Vocal density / Full lyric timeline / Detected chorus lines / Theme keywords
```

---

## Step 3 — Suno Prompt Compiler

Run the compiler to generate 3-version Style Prompts and orthogonal Negative Prompts:
```bash
python scripts/compile_prompt.py analysis.json "Artist Name"
```

### 1. Character Budget & Priority Order
- **Tag Order**: `Genre(1-2) → Mood(1-2) → Theory Tag(1-2) → Instruments(2-3) → Vocals(1) → Production(1-2)`, placing higher-weighted tags at the beginning.
- **Character Target**: 150–350 characters (8–12 comma-separated tags), avoiding verbose prose that dilutes model attention.

### 2. Auto-Exclusion Negative Prompt (≤200 chars)
Assembled from `auto_exclusion_map.yaml`:
- **Generic Quality Exclusions**: `muddy mix, low quality, harsh distortion, clipping, muffled vocals`
- **Orthogonal Style Conflicts**: Large opposing genres, strictly avoiding negation of same-root positive instruments.

### 3. Multi-Version Prompts
- **🟢 Safe**: Highest confidence tags only (genres + basic mood + tempo).
- **🎯 Recommended**: Balanced accuracy and musical creativity (default).
- **🔥 Experimental**: Introduces power combos and spatial descriptors.

---

## Step 4 — Original Song Delivery

Select the delivery workflow based on the track classification:

### Path A — Instrumental (Pure Music)

Outputs Suno Style Prompt along with the **Instrumental Timeline Arrangement Scaffolding** for the Lyrics box:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎵 ORIGINAL SONG CREATION (INSTRUMENTAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Reference Concept:
   [Producer 1-sentence summary]

📝 INPUT 1 — Lyrics Field (Instrumental Timeline Scaffolding):
[Intro: Atmospheric ambient textures, subtle pulse]
[Build-up: Rising tension, dynamic percussion enters, 128 BPM]
[Drop: Full instrumentation, powerful rhythmic hook, deep sub-bass]
[Verse: Stripped back arrangement, melodic solo lead over soft background chords]
[Climax: Massive dynamic peak, triumphant full-spectrum wall of sound]
[Outro: Slow decrescendo, lingering atmospheric tail, fade out]

🎚️ INPUT 2 — Style of Music Field (≤350 chars):
   [Recommended Style Prompt]

🚫 INPUT 3 — Exclude Styles Field (≤200 chars):
   [Negative Prompt]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Path B — Vocal Music

Outputs structured lyric metatags alongside style and negative prompts:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎵 ORIGINAL SONG CREATION (VOCAL TRACK)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Title: [Original Song Title]
   Concept: [Thematic core statement]

📝 INPUT 1 — Lyrics Field (Direct Paste with Suno Metatags):
[Intro]

[Verse 1]
[Lyric line with breath-point commas]
[Lyric line formatted to 1-2 musical bars]

[Chorus 1]
[Repetitive melodic hook line]

[Verse 2]
...

[Outro]

🎚️ INPUT 2 — Style of Music Field (≤350 chars):
   [Recommended Style Prompt]

🚫 INPUT 3 — Exclude Styles Field (≤200 chars):
   [Negative Prompt]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
