# audio-analyzer-skill

Audio analysis + lyrics transcription + web knowledge fusion + LLM music producer synthesis + original song creation for Suno AI

**Current version: v7.2**

---

## What It Does

Full pipeline for high-fidelity Suno music generation & production reverse-engineering:

1. **Dual-Branch Architecture & Hallucination Gate** *(v7.2 new)* — Zero-dependency gating using `instrumentalness >= 0.7` to cut off Whisper and eliminate instrumental looping text hallucinations.
2. **Multi-Channel Song Identity Detection** *(v7.2 new)* — Parallel tri-channel validation (Whisper hook lines + AcoustID audio fingerprinting + filename fallback) + LRClib synced lyrics matching.
3. **Lyric Normalizer & Structured Metatags** *(v7.2 new)* — Normalizes CJK/English punctuation (breath commas, bar line breaks) and automatically structures songs with `[Intro]`, `[Verse]`, `[Chorus]`, and `[Outro]` tags.
4. **Instrumental Timeline Arrangement Scaffolding** *(v7.2 new)* — Fills the Suno Lyrics box with dynamic arrangement directives (`[Intro: ...]`, `[Build-up: ...]`, `[Drop: ...]`) for purely instrumental tracks.
5. **Suno-Perceptible Music Theory Whitelist** *(v7.2 new)* — Filters musical theory into tokens that Suno actually understands (`2-5-1 jazz progression`, `four-chord pop loop`, `dorian mode`, `syncopated bass`).
6. **Artist Deconstruction Engine** *(v7.2 new)* — `artist_feature_map.yaml` transforms banned artist names into compliant positive sound/gear/theory descriptors.
7. **Orthogonal Auto-Exclusion Negative Prompt** *(v7.2 new)* — `auto_exclusion_map.yaml` auto-generates anti-artifact & orthogonal negative prompts (≤200 chars).
8. **Dual-Engine Audio Analysis** — Essentia (primary) + Librosa (secondary) for production-grade precision.
9. **Prompt Compiler & Multi-Version Output** — Safe / Recommended / Experimental prompt generation strictly within Suno attention budgets (150–350 chars).

---

## Changelog

### v7.2 — Production-Ready Gating & Dual-Branch Architecture
- **Whisper Hallucination Gate**: Automatically stops Whisper when Essentia detects `instrumentalness >= 0.7`, preventing hallucinations on instrumentals and solo sections.
- **Three-channel Identity Detection**: Parallel evaluation of Whisper hook lines, AcoustID fingerprints, and filename cues.
- **LRClib Synced Lyrics Integration**: Pulls timestamps and plain/synced lyrics without requiring API keys.
- **Lyric Normalizer**: Cleans CJK punctuation into breathing and phrase breaks for natural singing cadence in Suno.
- **Instrumental Timeline Scaffolding**: Provides structured arrangement cues in the Suno Lyrics box for instrumental generations.
- **Suno Theory Whitelist**: Adds `theory_tags` to `suno_tag_library.yaml` mapping academic harmony to community tokens.
- **Artist Feature Deconstruction**: `artist_feature_map.yaml` maps artist identities to positive stylistic tags, bypassing Suno name filters.
- **Auto-Exclusion Map**: `auto_exclusion_map.yaml` provides generic quality exclusions and orthogonal genre exclusions.
- **MIDI & Theory Engine**: `scripts/parse_midi.py` parses note-level MIDI and extracts modes/grooves.
- **Prompt Compiler**: `scripts/compile_prompt.py` compiles multi-version Style Prompts + Negative Prompts.

### v6.0 — Dual-Engine Architecture + Product-Grade Outputs
- **Essentia primary engine**: `KeyExtractor` (strength 0.914 vs librosa 0.677), `RhythmExtractor2013` BPM (109.94 vs 112.3), `MusicExtractor` EBU R128 LUFS (-13.34 dB), `ChordsDetection+HPCP` chord histogram with Roman numerals, `Danceability` metric
- **Spotify-like semantic features**: valence / energy / danceability / acousticness / instrumentalness
- **Cinematic/Orchestral genre category** added (Hans Zimmer / John Williams styles)
- **5-dimensional genre inference**: danceability + valence + BPM + key + LUFS
- **4-dimensional Suno mood mapping**: Danceability × Valence × Energy × Key → 8 precise mood words
- **Three-version Suno prompt output** (Step 3c): Safe / Recommended / Experimental
- **Similarity score** (Step 3d): 5-dimension pre-generation quality prediction
- **Suno style tag library** (`suno_tag_library.yaml`): 393 community-validated tags, 7 categories, 15 power combos, 12 forbidden combos
- **Song structure templates** (`suno_structure_templates.yaml`): 7 song form templates, 13 auto-selection rules, 18 annotated Suno structure tags
- **Lyric scaffold templates** (`suno_lyric_scaffolds.md`): 3 complete scaffolds with placeholder annotations
- **Graceful fallback** to librosa-only mode if essentia unavailable

### v5.0 — Web Knowledge Fusion
- Song name detection from filename, web knowledge retrieval from 4 sources (Wikipedia / reviews / producer interviews / lyrics), three-way triangulation protocol, priority override: Wikipedia > reviews > script data

### v4.0 — Multi-Signal Genre Scoring
- 12+ genre scoring system, genre-aware bass/drum tag selection, modal mood cross-validated with genre context

---

## Why v6 Is More Accurate

### Engine Comparison

| Metric | librosa (v4/v5) | Essentia (v6) | Improvement |
|---|---|---|---|
| Key detection | A# minor, conf=0.677 | **D minor, strength=0.914** | +35% confidence |
| BPM | 112.3 | **109.94** | EBU-standard precision |
| LUFS | -16.2 (approx) | **-13.34** (EBU R128) | Industry standard |
| Chords | Basic triads only | **Chord histogram** (Dm 44%, D 14.8%, G 11.5%…) | Full distribution |
| Genre | BPM+hihat rules | **5-dim scoring** (dance+valence+BPM+key+LUFS) | Cinematic recognized |

*Test file: F1 Hans Zimmer theme. librosa misclassified as Trap/Hip-hop; v6 correctly identified as Cinematic/Orchestral.*

### New Semantic Features (Spotify-like)

```json
"spotify_like_features": {
  "valence": 0.28,        // emotional positivity (0=dark, 1=happy)
  "energy": 0.82,         // intensity level
  "danceability": 0.68,   // rhythmic drive
  "acousticness": 0.35,   // acoustic vs electronic ratio
  "instrumentalness": 0.98 // vocal presence estimate
}
```

### Multi-Version Prompt Output (Step 3c)

| Version | Strategy | Character Count |
|---|---|---|
| 🟢 Safe | Highest-confidence tags only | ≤60 chars |
| 🎯 Recommended | Balanced precision + creativity | ≤120 chars |
| 🔥 Experimental | Power combos + niche descriptors | ≤120 chars |

### Similarity Estimation (Step 3d)

Before generating, get a 5-star prediction across:
- Rhythm fit (BPM range)
- Tonal clarity (key strength)
- Harmonic complexity (chord variety)
- Dynamics reachability (LUFS zone)
- Genre clarity (genre score gap)

---

## Included Files

| File | Description |
|---|---|
| `scripts/analyze_audio.py` | Main analysis script (dual-engine + Whisper hallucination gating, 1270 lines) |
| `scripts/identify_song.py` | Multi-channel song identity detection, LRClib synced lyrics query, Lyric Normalizer & Instrumental Timeline Scaffolding |
| `scripts/compile_prompt.py` | Prompt Compiler generating 3-version Style Prompts (150–350 chars) + Orthogonal Negative Prompts (≤200 chars) |
| `scripts/parse_midi.py` | MIDI parser extracting modes, syncopated bass, and whitelisted Suno theory tags |
| `artist_feature_map.yaml` | Positive artist deconstruction dictionary bypassing Suno's artist name filter |
| `auto_exclusion_map.yaml` | Orthogonal negative prompts and universal anti-artifact rules |
| `SKILL.md` | Full pipeline instructions for Claude (v7.2 dual-branch workflow) |
| `suno_tag_library.yaml` | 400+ validated Suno style tags, 8 categories (including theory whitelist), 15 power combos |
| `suno_structure_templates.yaml` | 7 song form templates, 18 annotated structure tags, 13 auto-selection rules |
| `suno_lyric_scaffolds.md` | 3 complete lyric scaffold templates with placeholder annotations |

---

## Supported Formats

`MP3` / `WAV` / `FLAC` / `AAC` / `OGG` / `M4A`

---

## Installation

### 1. Install Python dependencies

```bash
pip install librosa soundfile scipy faster-whisper
pip install essentia   # recommended — enables v6 dual-engine mode
```

> If `essentia` install fails (requires gcc), the skill automatically falls back to librosa-only mode.
> `ffprobe` is required (comes with `ffmpeg`). First Whisper run downloads the small model (~245MB), cached at `/tmp/whisper_models`.

### 2. Install the skill

Place the skill folder under your OpenClaw skills directory:

```
~/.openclaw/skills/audio-analyzer/
```

---

## Usage

### 1. Basic Audio Feature Extraction
```bash
python3 scripts/analyze_audio.py <file_path>
```
Output: JSON with all audio features including `essentia_features`, `spotify_like_features`, `chord_histogram`, and `analysis_engine` fields. Automatically engages Whisper hallucination gate if `instrumentalness >= 0.7`.

### 2. Multi-Channel Identity & Lyric Structuring
```bash
python3 scripts/identify_song.py <file_path> analysis.json
```
Output: Matches track via LRClib / AcoustID, normalizes lyrics with Suno metatags, or outputs an Instrumental Timeline Arrangement Scaffolding.

### 3. Prompt Compiler
```bash
python3 scripts/compile_prompt.py analysis.json [optional_artist_name]
```
Output: Generates 3-version Style Prompts (Safe / Recommended / Experimental) within 150–350 chars alongside orthogonal Negative Prompts (≤200 chars).

### 4. Optional MIDI Theory Analysis
```bash
python3 scripts/parse_midi.py <midi_file_path>
```
Output: Extracts scale modes, rhythm syncopation, and Suno-perceptible whitelisted theory tags.

---

## Output JSON Structure (v7.2)

```json
{
  "analysis_engine": {
    "primary": "essentia",
    "secondary": "librosa",
    "fallback": false
  },
  "essentia_features": {
    "bpm": 109.94,
    "lufs_integrated": -13.34,
    "lufs_range": 8.92,
    "danceability": 1.35,
    "key_extractor": { "key": "D", "scale": "minor", "strength": 0.914 },
    "chord_histogram": { "Dm": 44.0, "D": 14.8, "G": 11.5, "Am": 7.6 }
  },
  "spotify_like_features": {
    "valence": 0.28,
    "energy": 0.82,
    "danceability": 0.68,
    "acousticness": 0.35,
    "instrumentalness": 0.98
  },
  "tonality": { "key": "D", "mode": "minor", "modal_flavor": "Aeolian" },
  "summary": { "tempo_bpm": 110, "key_signature": "D minor" },
  "suno_prompt": { "style_tags": "cinematic orchestral, epic, dark, driving, wide dynamics" }
}
```

---

## Accuracy

| Dimension | Rating | Notes |
|---|---|---|
| BPM | ★★★★★ | Essentia RhythmExtractor2013, EBU standard |
| Key/Mode | ★★★★★ | Essentia KeyExtractor, strength 0.9+ |
| Chords | ★★★★ | ChordsDetection+HPCP, chord histogram with % distribution |
| Bassline | ★★★★ | pyin tracking, first 90s effective |
| Drums | ★★★ | Frequency-band separation, first 8 bars effective |
| Lyrics | ★★★ | faster-whisper small, English best |
| Genre | ★★★★★ | Wikipedia (v5) + 5-dim Essentia scoring (v6) |
| Style Tags | ★★★★★ | Web knowledge fusion × Essentia data × Claude triangulation |
| Suno Prompt | ★★★★★ | 3-version output + similarity pre-check |
