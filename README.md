# audio-analyzer-skill

Audio analysis + lyrics transcription + LLM music producer synthesis + original song creation for Suno AI

---

## What It Does

Full pipeline for music production reference analysis:

1. **Audio Feature Extraction** — BPM, key, chords, bassline, drums, EQ, LUFS, melody, structure
2. **Lyrics Transcription** — Auto vocal transcription via Whisper with chorus detection
3. **Music Producer Synthesis** — Deep LLM analysis from a producer's perspective, integrating audio features and lyric themes
4. **Original Song Creation** — Brand-new title + full lyrics + Suno style prompt (≤200 chars) imitating the reference track's style and production DNA

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

Run the analysis script on any audio file:

```bash
python3 ~/.openclaw/skills/audio-analyzer/scripts/analyze_audio.py <file_path>
```

Or download from URL first:

```bash
curl -sL -o /tmp/files/$(date +%s).wav "<URL>"
```

Then ask Claude:

- `分析这首歌` / `analyze this audio`
- `帮我仿写这首歌的风格`
- `给我生成 Suno prompt`
- `这首歌的 BPM 和调性是什么`

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

| Dimension | Accuracy |
|---|---|
| BPM | ★★★★ (±1 BPM) |
| Key / Mode | ★★★ |
| Chords | ★★★ (triads only) |
| Bassline | ★★★★ (first 90s) |
| Drums | ★★★ (first 8 bars) |
| Lyrics (Whisper) | ★★★ (English best) |
| LLM Synthesis | ★★★★ |
| Imitation Lyrics | ★★★★ |
| Suno Prompt | ★★★★ |

---

## Keywords

`分析音频` `suno prompt` `歌词识别` `仿写歌曲` `创作歌词` `模仿创作` `编曲参考` `BPM` `和弦进行` `调性` `bassline` `audio analysis` `lyrics transcription` `imitation composition` `original song creation`
