# audio-analyzer-skill

Audio analysis + lyrics transcription + web knowledge fusion + LLM music producer synthesis + original song creation for Suno AI

**Current version: v6.0**

---

## What It Does

Full pipeline for high-fidelity Suno music generation:

1. **Song Identity Detection** — Auto-extract song name & artist from filename
2. **Dual-Engine Audio Analysis** *(v6 new)* — Essentia (primary) + Librosa (secondary) for production-grade precision
3. **Web Knowledge Retrieval** — Fetch Wikipedia genre labels, music reviews, producer interviews, and lyrics analysis
4. **Lyrics Transcription** — Auto vocal transcription via Whisper with chorus detection
5. **Knowledge Fusion & Style Tag Validation** — Three-way triangulation: web knowledge × audio data × Claude judgment
6. **Multi-Version Suno Prompt** *(v6 new)* — 3 prompt variants: Safe / Recommended / Experimental
7. **Similarity Estimation** *(v6 new)* — 5-dimension score predicting how well Suno can replicate the source track
8. **Music Producer Synthesis** — Deep LLM analysis from a producer's perspective
9. **Original Song Creation** — Brand-new title + full lyrics + Suno style prompt imitating the reference track's DNA

---

## Changelog

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
| `scripts/analyze_audio.py` | Main analysis script (dual-engine, 1266 lines) |
| `SKILL.md` | Full pipeline instructions for Claude |
| `suno_tag_library.yaml` | 393 validated Suno style tags, 7 categories, feature→tag mapping rules |
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

### Basic

```bash
python3 ~/.openclaw/skills/audio-analyzer/scripts/analyze_audio.py <file_path>
```

Output: JSON with all audio features including `essentia_features`, `spotify_like_features`, `chord_histogram`, and `analysis_engine` fields.

### With Song Name (enables web knowledge fusion)

```bash
# Rename file to song name before analysis
mv recording.mp3 "The Weeknd - Blinding Lights.mp3"
python3 analyze_audio.py "The Weeknd - Blinding Lights.mp3"
```

Claude will auto-detect the title and fetch Wikipedia + reviews for style fusion.

---

## Output JSON Structure (v6)

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
