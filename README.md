# audio-analyzer-skill

Audio analysis + lyric forced alignment + Suno prompt compilation for original song creation.

**Current version: v7.3**

---

## The question this project answers

> A reference audio file contains thousands of measurable things. Which of them
> can be turned into words that Suno will actually read?

Suno reads three text fields — **Style**, **Lyrics**, **Exclude**. It does not
read LUFS, chroma, Roman numerals, drum timestamps or MFCC vectors. Anything
measured that cannot be written into one of those three fields is report-only.

So the pipeline answers four questions and nothing else:

1. Vocal or instrumental?
2. What is the style character?
3. What is the speed feel?
4. New lyrics, or structure only?

---

## What v7.3 fixes

v7.2's main path was broken in ways that produced confidently wrong tags. The
short version:

| # | v7.2 behaviour | Consequence | v7.3 fix |
|---|---|---|---|
| 1 | Whisper was gated on `instrumentalness >= 0.7`, but that score was computed as `0.95 whenever has_lyrics is False`, and `has_lyrics` was only set *after* Whisper ran | The score was always 0.95, Whisper never ran, the vocal branch was unreachable, and every song was treated as instrumental | The gate is now acoustic and runs **before** transcription. It returns `vocal` / `instrumental` / `unknown`, and only a confident `instrumental` skips transcription. `unknown` escalates. |
| 2 | `detect_mode()` returned `Phrygian` whenever b6 and b7 were strong | Natural minor **is** b6 + b7, so almost every minor song was labelled Phrygian and the compiler wrote `phrygian tension` into the prompt | Phrygian requires a measured flat 2nd ≥ 0.35 that also beats the natural 2nd. Natural minor resolves to Aeolian, and the decision is recorded in `modal_evidence`. |
| 3 | Two compilers: `generate_suno_prompt()` capped Style at 120/200 chars, `compile_prompt.py` documented 150–350 | The prompt depended on which script the agent happened to call | One compiler, one budget per field, defined once in `suno_contract.py`. `analyze_audio.py` builds no prompts. |
| 4 | `chord_to_roman()` only recognised `Am`, but Essentia emits `A:min` | Every minor chord was read as major; the whole Roman-numeral block skewed major | Chord parsing handles `A:min`, `A:maj`, flats, and `N`/`NC`. |
| 5 | `suno_tag_library.yaml`'s `theory_tags` whitelist was never read | Tags were emitted from a guess chain, with a `four-chord pop loop` fallback | Triggers are evaluated against real evidence. No evidence means an empty list, never a default tag. |
| 6 | `auto_exclusion_map.yaml` was keyed `cinematic_orchestral` while the compiler tested `"cinematic_orchestral" in "cinematic, orchestral"` | Orthogonal exclusion almost never fired — only `lo-fi` matched, by substring accident | Canonical keys with explicit `matches` alias lists. Matching is exact after normalisation. |
| 7 | Generic exclusions were appended unconditionally | `harsh distortion` cancelled a requested distorted guitar; `muffled vocals` cancelled an intentionally muffled vocal | Every negative tag passes a same-root check. Conflicts are dropped and the reason is recorded. |
| 8 | Over-long output was cut with `[:197] + "..."` | The trailing ellipsis became a style token | Whole tags are dropped, least-important first. No ellipsis is ever emitted. |
| 9 | After identifying the track, `format_structured_lyrics()` repunctuated the reference song's real lyrics and handed them to Suno | The "original song" output was a cover | The reference contributes **statistics only**: syllable density, hook length, perspective, rhyme, section shape. Reference words are withheld unless you explicitly opt into cover mode. |
| 10 | `generate_instrumental_timeline()` claimed to read the energy curve but was a hard-coded Intro/Build/Drop/Climax arc | A lo-fi loop and a Zimmer score got the identical seven-part structure, and the measured `energy_drop_times_sec` was discarded | Section labels require evidence. `Drop` needs a measured drop; `Climax` needs a peak clearly above the median. No evidence means one neutral line. |
| 11 | The README advertised "three-channel identity detection", but `get_acoustid_fingerprint()` was never called and there was no AcoustID query | The claim was false; a filename match alone got 0.9 confidence with no duration check | Channels report honestly what ran. AcoustID is either really queried or reported unavailable. Every match is duration-checked. |
| 12 | `parse_midi.py` returned `["four-chord pop loop", "syncopated bass"]` when `pretty_midi` was missing | Tags were invented for music that was never analysed | Missing dependency means `available: false` and an empty tag list. Chords are actually detected; drums are excluded from pitch statistics; bass syncopation is measured on the bass only. |
| 13 | Beat-interval variance was reported as "Swing type" | Rhythm instability was presented as swing | Timing steadiness and swing are separate fields. `swing_ratio` is `null` when it cannot be measured. |
| 14 | Fixed `/tmp/_whisper_input.wav` and `tempfile.mktemp` | Concurrent analyses overwrote each other | Unique temp paths under the system temp directory. |
| 15 | Drum density was computed after truncating onsets to 24 | Density was systematically underestimated | Counts use the full window; the onset list is truncated for display only. |
| 16 | A single whole-file `rfft` was used as the EQ reference | A loud chorus erased a quiet verse | Per-frame band energies with a median reference plus a loudness-weighted view. |
| 17 | `load_yaml_simple()` returned `{}` when PyYAML was missing | The artist and exclusion tables silently went empty and the pipeline still produced a prompt | PyYAML is required. The compiler stops with an explicit error. |
| 18 | No tests, no evaluation set, `__pycache__` committed | The five-star ratings rested on one Hans Zimmer file | 108 unit tests, a 20-track listening set, and `__pycache__` removed from the repo. |

### Removed claims

The v7.2 star ratings were deleted. They rested on a single example, and the
"+35% confidence" figure compared librosa's correlation coefficient with
Essentia's `strength` — two different quantities that cannot be expressed as a
percentage improvement. Accuracy claims will be reinstated only after
`eval/score_eval.py` produces real numbers.

---

## Install

```bash
# Core analysis + compilation (required)
pip install librosa soundfile scipy numpy pyyaml faster-whisper

# Optional but recommended: better key, LUFS and chord detection.
# Falls back to librosa cleanly when absent.
pip install essentia

# Lyric timeline (step 4 only)
pip install demucs torch torchaudio transformers uroman

# Optional accuracy upgrades
pip install silero-vad      # real VAD for the vocal gate
pip install pretty_midi     # MIDI chord analysis
```

`ffmpeg` and `ffprobe` must be on PATH.

`PyYAML` is required, not optional. Without it the compiler cannot read the
artist and exclusion tables and will stop rather than produce a prompt from an
empty configuration.

`ACOUSTID_API_KEY` is optional; it only enables the AcoustID identity channel.

### Install the skill

Place the folder under your skills directory:

```
~/.openclaw/skills/audio-analyzer/
```

---

## Usage

### The single entry point

```bash
python scripts/run_pipeline.py song.mp3 --out-dir out
python scripts/run_pipeline.py song.mp3 --artist "The Weeknd" --out-dir out
python scripts/run_pipeline.py song.mp3 --style-limit 200 --exclude-limit 200
python scripts/run_pipeline.py song.mp3 --align-lyrics lyrics.txt --out-dir out
```

This runs analysis → identification → compilation and prints the three Suno
fields ready to paste. It writes `analysis.json`, `identity.json` and
`suno_prompt.json`.

### Individual steps

```bash
# 1. Measurement and evidence (builds no prompts)
python scripts/analyze_audio.py song.mp3 > analysis.json
python scripts/analyze_audio.py song.mp3 --vocal-presence vocal
python scripts/analyze_audio.py song.mp3 --skip-whisper

# 2. Identity and lyrics policy
python scripts/identify_song.py song.mp3 analysis.json --out identity.json

# 3. The only compiler
python scripts/compile_prompt.py analysis.json "The Weeknd" --identity identity.json

# 4. Lyric timeline for video (optional, after the song exists)
python scripts/align_lyrics.py song.mp3 lyrics.txt analysis.json
python scripts/align_lyrics.py song.mp3 lyrics.txt --from-lrc existing.lrc
python scripts/align_lyrics.py song.mp3 lyrics.txt --vocals stems/vocals.wav

# Optional: MIDI harmony evidence
python scripts/parse_midi.py sketch.mid --tag-library suno_tag_library.yaml
```

---

## Output

### `suno_prompt.json`

```json
{
  "mode": "vocal",
  "vocal_presence": { "presence": "vocal", "score": 0.71, "decided_by": "acoustic" },
  "style": {
    "prompt": "R&B, neo-soul, mandarin vocals, warm bass, rolling hi-hats, hypnotic, four-chord pop loop, mid-tempo",
    "char_count": 108, "limit": 200, "tags": [], "dropped": []
  },
  "exclude": {
    "prompt": "hardcore trap, blast beats, muddy mix, low quality, clipping",
    "char_count": 63, "limit": 200, "dropped_same_root": []
  },
  "lyrics_plan": { "policy": "write_new", "note": "Write a NEW lyric. ..." },
  "theory_tags": [
    { "tag": "four-chord pop loop", "why": "progression matched i-VI-III-VII" }
  ],
  "artist_deconstruction": { "requested": "the weeknd", "matched_key": "The Weeknd" },
  "report_only": { "_comment": "Report-only. These values MUST NOT be pasted into Suno." },
  "warnings": []
}
```

### `analysis.json` (selected fields)

```json
{
  "analysis_engine": { "primary": "essentia", "secondary": "librosa", "fallback": false },
  "vocal_presence": {
    "presence": "vocal", "score": 0.71, "confidence": 0.42,
    "reason": "Sustained pitched activity in the vocal range dominates the mix.",
    "decided_by": "acoustic"
  },
  "tonality": {
    "key": "A minor", "modal_flavor": "Aeolian (natural minor)", "pentatonic": "minor",
    "modal_evidence": { "b2": 0.05, "natural_2": 0.88, "flat_2_confirmed": false }
  },
  "rhythm": {
    "tempo_bpm": 104.0, "rhythm_stability": "steady", "swing_type": "straight",
    "swing_ratio": 1.0, "bass_syncopation_ratio": 0.52,
    "beat_times_sec": [0.58, 1.16, 1.74], "downbeat_times_sec": [0.58, 2.9]
  },
  "frequency_bands": {
    "sub_bass_20_60hz": { "energy_pct": 6.2, "energy_pct_loudweighted": 7.1,
                          "energy_pct_global": 9.4 }
  },
  "drum_pattern": { "kick_per_bar": 2.0, "half_time": false, "sixteenth_hats": false },
  "structure": { "energy_drop_times_sec": [40.0], "energy_peak_times_sec": [80.0] },
  "theory_evidence": { "roman_loop": ["i", "VI", "III", "VII"], "half_time": false },
  "lyrics": { "has_lyrics": true, "language": "zh", "lang_suno_tag": "mandarin vocals" }
}
```

`lyrics.timeline` is a report. It must never be used as a video timeline and must
never be pasted into the Lyrics field of an original song.

### `mv_shots.json`

```json
{
  "contract": "The timeline is locked. Only the `visual` field may be edited. `start`, `end` and `lyric` MUST NOT be changed.",
  "shot_count": 42,
  "needs_review_count": 3,
  "shots": [
    { "start": 0.0, "end": 9.8, "section": "Instrumental", "lyric": null,
      "energy": "low", "visual": "", "needs_review": false },
    { "start": 9.8, "end": 13.4, "section": "Verse 1", "lyric": "把夜色唱成海岸",
      "energy": "mid", "visual": "", "needs_review": false,
      "alignment_score": 0.87, "snap_shift_sec": -0.04 }
  ]
}
```

A line is flagged `needs_review` when its alignment score is low, when it lands
where there is no vocal energy, or when it needed a large snap. Nothing is
silently accepted.

---

## Tests

```bash
pip install pytest numpy pyyaml
python -m pytest tests -q
```

108 tests covering the specific v7.2 regressions: natural minor must not be
labelled Phrygian, `A:min` must parse as minor, an absent vocal measurement must
not become `instrumental`, exclusions must not cancel requested instruments, no
output may contain an ellipsis, and structure labels must require energy evidence.

## Evaluation

```bash
python eval/score_eval.py
```

`eval/manifest.yaml` defines 20 slots with the coverage requirements and the
ground truth to confirm by ear. Accuracy claims stay out of this README until
that table has real numbers.

---

## Files

| File | Role |
|---|---|
| `scripts/run_pipeline.py` | **Entry point.** Runs steps 1–3 in order. |
| `scripts/analyze_audio.py` | Measurement and evidence only. |
| `scripts/compile_prompt.py` | **The only compiler.** |
| `scripts/identify_song.py` | Identity channels and lyrics policy. |
| `scripts/align_lyrics.py` | Forced alignment → LRC, SRT, MV shot list. |
| `scripts/parse_midi.py` | MIDI harmony evidence. |
| `scripts/vocal_gate.py` | Acoustic vocal/instrumental/unknown decision. |
| `scripts/music_theory.py` | Chord parsing, Roman numerals, mode detection. |
| `scripts/theory_triggers.py` | Evidence-gated theory tags. |
| `scripts/exclusion_rules.py` | Canonical genre matching, same-root filtering. |
| `scripts/structure_plan.py` | Energy-derived section labels. |
| `scripts/lyric_utils.py` | Lyric parsing, transferable style statistics. |
| `scripts/shot_builder.py` | Beat snapping, shot merging/splitting, export. |
| `scripts/suno_contract.py` | Field budgets, slot order, tag vocabulary. |
| `scripts/yaml_io.py` | Explicit config loading. |
| `auto_exclusion_map.yaml` | Canonical exclusion keys with `matches` aliases. |
| `artist_feature_map.yaml` | Artist → compliant descriptors, with aliases. |
| `suno_tag_library.yaml` | Tag vocabulary and the theory whitelist. |
| `suno_structure_templates.yaml` | Section tag reference. |
| `suno_lyric_scaffolds.md` | Lyric scaffold reference. |
| `tests/` | 108 unit tests. |
| `eval/` | 20-track listening set and scoring script. |

---

## Supported formats

`MP3` / `WAV` / `FLAC` / `AAC` / `OGG` / `M4A`
