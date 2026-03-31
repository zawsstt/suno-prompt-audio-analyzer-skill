# audio-analyzer-skill

Audio analysis + lyrics transcription + web knowledge fusion + LLM music producer synthesis + original song creation for Suno AI

**Current version: v5.0**

---

## What It Does

Full pipeline for high-fidelity Suno music generation:

1. **Song Identity Detection** *(v5 new)* — Auto-extract song name & artist from filename
2. **Audio Feature Extraction** — BPM, key, chords, bassline, drums, EQ, LUFS, melody, structure
3. **Web Knowledge Retrieval** *(v5 new)* — Fetch Wikipedia genre labels, music reviews, producer interviews, and lyrics analysis for the detected song
4. **Lyrics Transcription** — Auto vocal transcription via Whisper with chorus detection
5. **Knowledge Fusion & Style Tag Validation** *(v5 new)* — Three-way triangulation: web knowledge × audio data × Claude judgment to produce the most accurate Suno style prompt
6. **Music Producer Synthesis** — Deep LLM analysis from a producer's perspective, integrating all data sources
7. **Original Song Creation** — Brand-new title + full lyrics + Suno style prompt imitating the reference track's production DNA

---

## Why v5 Is More Accurate

Pure audio analysis can measure BPM, key, and frequency data — but it can't know that a song uses a "Juno-106 synth pad" or belongs to "new wave". **Web knowledge fills this gap:**

| Source | What It Provides |
|---|---|
| Wikipedia | Precise genre labels defined by music scholars (highest authority) |
| Music reviews | Timbre/texture descriptors ("pulsing synth arpeggios", "gated reverb drums") |
| Producer interviews | Gear & techniques ("Roland TR-808", "pitched vocal sample") |
| Lyrics analysis | Theme, imagery, writing style |

**Example — The Weeknd - Blinding Lights:**

| Version | Suno Style Prompt |
|---|---|
| Script only | `driving, melancholic, four-on-the-floor` |
| v5 with web fusion | `synth-pop, new wave, pulsing synth arpeggios, gated reverb drums, 80s aesthetic, melancholic, driving` |

The v5 prompt contains vocabulary directly from how humans describe the song — which is exactly what Suno was trained on.

> **How to enable:** Simply name your audio file after the song (e.g. `The Weeknd - Blinding Lights.mp3`). Claude will auto-detect the title and fetch knowledge automatically.

---

## Supported Formats

`MP3` / `WAV` / `FLAC` / `AAC` / `OGG` / `M4A`

---

## Installation

### 1. Install Python dependencies

```bash
pip install librosa soundfile scipy faster-whisper
```

> `ffprobe` is required (comes with `ffmpeg`). First Whisper run downloads the small model (~245MB), cached at `/tmp/whisper_models`.

### 2. Install the skill

Place the skill folder under your OpenClaw skills directory:

```
~/.openclaw/skills/audio-analyzer/
```

---

## Usage

### Basic

Run the analysis script on any audio file:

```bash
python3 ~/.openclaw/skills/audio-analyzer/scripts/analyze_audio.py <file_path>
```

Or download from URL first:

```bash
curl -sL -o /tmp/files/$(date +%s).wav "<URL>"
```

### Recommended: Name Your File After the Song

For best results, rename the audio file to the song name before uploading:

```
# Format: Artist - Title.ext
The Weeknd - Blinding Lights.mp3
Kendrick Lamar - Not Like Us.mp3
Daft Punk - Get Lucky.flac

# Or just title
Blinding Lights.mp3
```

Claude will auto-detect the song identity and fetch web knowledge to enhance the Suno prompt.

Then ask Claude:

- `分析这首歌` / `analyze this audio`
- `帮我仿写这首歌的风格`
- `给我生成 Suno prompt`
- `这首歌的 BPM 和调性是什么`

---

## Pipeline (v5)

```
Step 0   Song identity detection    →  filename → song name + artist
Step 1   Run analysis script        →  raw JSON (audio features + whisper lyrics)
Step 1b  Web knowledge retrieval    →  Wikipedia, reviews, producer info, lyrics analysis
Step 2   Present analysis report    →  structured sections ①–⑫
Step 3   Knowledge fusion + style validation  →  triangulate web + audio + Claude
Step 4   Original song creation     →  new title + full lyrics + Suno prompt
```

---

## Output Sections

| Section | Content |
|---|---|
| ① 响度 & 动态 | RMS / LUFS / Crest Factor / Dynamic Range |
| ② 频段能量 | 7-band EQ reference bar chart |
| ③ 节奏 & 律动 | BPM / Swing / Time Signature |
| ④ 鼓组 Pattern | Kick / Snare / HiHat timeline |
| ⑤ Bassline | Root note sequence + MIDI suggestions |
| ⑥ 和弦进行 | Core loop + Roman numerals + transitions |
| ⑦ 调性 & 音阶 | Key / Mode / Relative key / Chroma energy |
| ⑧ 调性时间轴 | Key changes every 15 seconds |
| ⑨ 旋律轮廓 | Range / Dominant notes / Opening contour |
| ⑩ 能量弧度 & 结构 | RMS+Flux timeline / Segment detection |
| ⑪ 音色 & 频谱 | Centroid / Bandwidth / Rolloff / Brightness |
| ⑫ 歌词识别 | Whisper transcript + chorus candidates |

---

## Suno Prompt Rules

| Mode | Field | Char Limit |
|---|---|---|
| Instrumental (Link A) | Style only | ≤ 200 chars |
| With Lyrics (Link B) | Style of Music | ≤ 120 chars |
| With Lyrics (Link B) | Lyrics | ≤ 3000 chars |

---

## Accuracy

| Dimension | Accuracy | Notes |
|---|---|---|
| BPM | ★★★★ | ±1 BPM |
| Key / Mode | ★★★ | KS algorithm; modulating songs less accurate |
| Chords | ★★★ | Triads only, no extensions |
| Bassline | ★★★★ | pyin tracking, first 90s |
| Drums | ★★★ | Band separation, first 8 bars |
| Lyrics (Whisper) | ★★★ | English best; degrades with heavy mix |
| **Genre Detection** | **★★★★★** | **v5: Wikipedia (highest authority) + v4 multi-signal scoring** |
| **Style Tags** | **★★★★★** | **v5: Web knowledge fusion × audio data × Claude triangulation** |
| LLM Synthesis | ★★★★ | Quality scales with available data |
| Imitation Lyrics | ★★★★ | Style-matched, 100% original content |
| **Suno Prompt** | **★★★★★** | **v5: Human music knowledge + audio science + Claude judgment** |

---

## Changelog

### v5.0 (2026-03-31)
- **Step 0**: Song identity detection from filename (supports `Artist - Title.ext` and `Title.ext` formats)
- **Step 1b**: Web knowledge retrieval pipeline with 4 source priority tiers (Wikipedia > reviews > producer interviews > lyrics analysis)
- **Step 3a**: Three-way knowledge fusion protocol — web knowledge overrides script when they conflict
- Wikipedia genre labels now take highest priority over script-based genre detection
- Music review timbre/texture descriptors fused directly into Suno style prompt

### v4.0 (2026-03-31)
- Multi-signal genre scoring system (12+ genres × 5+ signal dimensions each)
- Genre-aware bass tag selection: `808 bass` only in hip-hop context; DnB sub bass → `rolling bass`
- Pattern-aware drum tags: trap hi-hats vs. four-on-the-floor vs. breakbeat drums
- Modal mood cross-validated with genre context (e.g. Phrygian × EDM → `dark and ominous`, not `brooding`)
- Added `genre_scores` field to JSON output for confidence transparency
- Added `llm_synthesis_hint` field for Claude's style validation step
- Script: 776 → 941 lines

### v3.0 (prior)
- Initial release with audio feature extraction, Whisper transcription, LLM synthesis, and Suno prompt generation

---

## Keywords

`分析音频` `suno prompt` `歌词识别` `仿写歌曲` `创作歌词` `模仿创作` `编曲参考` `BPM` `和弦进行` `调性` `bassline` `audio analysis` `lyrics transcription` `imitation composition` `original song creation` `style tags` `genre detection` `web knowledge fusion`
