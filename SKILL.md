---
name: audio-analyzer
description: "Audio analysis + lyric forced alignment + Suno prompt compilation for original song creation. Analyzes a reference audio file, decides vocal-vs-instrumental from acoustics, extracts only the properties Suno can actually read, and compiles the three Suno fields (Style, Lyrics, Exclude). Also builds a time-locked lyric timeline (LRC + SRT + MV shot list) for a finished song from its known lyrics. Keywords: audio analysis, suno prompt, style prompt, exclude styles, forced alignment, LRC, SRT, music video shots, BPM, key, mode, chord progression, original song."
version: "7.3"
changelog: >
  v7.3 — Correctness release. Rewrites the vocal gate so it no longer depends on
  transcription (the v7.2 gate was always closed, so the vocal branch was dead);
  requires a measured flat 2nd before emitting a Phrygian label; collapses the two
  conflicting compilers into one entry point with a single character budget;
  connects the theory whitelist and the exclusion map to the tags the compiler
  actually emits; stops shipping the reference song's lyrics as the "original"
  output; derives instrumental scaffolding from measured energy; replaces the
  fake tri-channel identification with honest channel reporting; and adds an
  optional Demucs + CTC forced-alignment step that produces a locked MV shot list.
---

# Audio Analyzer v7.3 — What Suno Actually Reads

## The only question this skill answers

> Of everything measurable in a reference audio file, what can be turned into
> words that Suno will actually read?

Suno reads three text fields. It does not read LUFS, chroma, Roman numerals,
drum timestamps or MFCC vectors. Anything measured that cannot be written into
**Style**, **Lyrics** or **Exclude** is report-only cost, not output.

So the pipeline answers exactly four things:

| # | Question | Where it lands |
|---|---|---|
| 1 | Vocal or instrumental? | Style (`instrumental` / language tag) + which Lyrics path |
| 2 | What is the style character? | Style (genre, mood, 2–3 instruments) |
| 3 | What is the speed feel? | Style (one tempo word) |
| 4 | New lyrics, or structure only? | Lyrics |

Everything else goes into the report and **must not be pasted into Suno**.

---

## Pipeline

```
audio file
    │
    ▼
Step 1  analyze_audio.py ──► analysis.json
        • acoustic vocal gate (BEFORE transcription)
        • key / mode (Phrygian requires a measured flat 2nd)
        • BPM, beat grid, downbeats, real swing ratio, bass syncopation
        • band energies (per-frame median + loudness-weighted)
        • energy curve, drops, peaks
        • transcription, ONLY when the gate does not say "instrumental"
        │
        ▼
Step 2  identify_song.py ──► identity.json
        • reports which identity channels actually ran
        • duration-checks every match
        • extracts transferable lyric STATISTICS, never reference words
        │
        ▼
Step 3  compile_prompt.py ──► suno_prompt.json
        • the ONE compiler
        • one budget per field; whole tags dropped, never truncated
        │
        ▼
Step 4  align_lyrics.py (optional, after the song exists)
        ──► lyrics.lrc + lyrics.srt + mv_shots.json
```

### Just run it

```bash
python scripts/run_pipeline.py song.mp3 --out-dir out
```

`run_pipeline.py` is the single entry point. Do not call the steps out of order
and do not use any other script to build a Suno prompt.

---

## Step 1 — Analysis

```bash
python scripts/analyze_audio.py song.mp3 > analysis.json
python scripts/analyze_audio.py song.mp3 --vocal-presence vocal   # override the gate
python scripts/analyze_audio.py song.mp3 --skip-whisper           # analysis only
```

### The vocal gate

The gate returns one of three states and **never reads lyrics**:

- `vocal` — sustained pitched activity in the vocal range
- `instrumental` — no sustained pitched activity
- `unknown` — ambiguous, or not enough audio to judge

**Only a confident `instrumental` skips transcription.** `unknown` always runs
it, because a false "instrumental" loses the entire lyric pipeline while a false
"vocal" costs one transcription pass.

If a transcription is inconclusive, the presence is upgraded to `vocal` — never
downgraded to `instrumental`. A failed transcription is not evidence of an
instrumental track.

Read the decision in `analysis.json → vocal_presence`:

```json
"vocal_presence": {
  "presence": "vocal",
  "score": 0.71,
  "confidence": 0.42,
  "reason": "Sustained pitched activity in the vocal range dominates the mix.",
  "decided_by": "acoustic",
  "evidence": { "voiced_frame_fraction": 0.72, "band_tonality": 0.9 }
}
```

### Modal flavour

`tonality.modal_flavor` is only `Phrygian` when `tonality.modal_evidence` shows a
flat 2nd that is present **and** stronger than the natural 2nd. Natural minor
(b6 + b7) resolves to `Aeolian (natural minor)`. The evidence block always
records the decision:

```json
"modal_evidence": {
  "b2": 0.05, "natural_2": 0.88, "flat_2_confirmed": false,
  "decision": "b6+b7 but no flat 2nd (b2=0.05 <= natural 2nd=0.88) -> natural minor"
}
```

### Report-only fields

These are measured, reported, and must never enter a Suno field:

LUFS · dynamic range · crest factor · chroma profile · MFCC means/stds ·
Roman numerals · chord histogram · beat-interval CV · onset timestamp lists ·
spectral centroid.

`compile_prompt.py` returns them under `report_only` with an explicit warning.

---

## Step 2 — Identification and lyrics policy

```bash
python scripts/identify_song.py song.mp3 analysis.json --out identity.json
```

Channels are reported honestly under `channels`. A channel that could not run
says so. Filename parsing is labelled a hint, not an identification, and never
produces a confidence score on its own.

- **AcoustID** runs only when `fpcalc` is installed and `ACOUSTID_API_KEY` is
  set. Every match is duration-checked.
- **LRClib** runs without a key and is duration-checked when the duration is known.

### Reference lyrics never become your lyrics

`lyrics_policy` is `write_new` by default. The reference contributes only:

- syllable density and spread
- average line length
- hook length and repetition count
- narrative perspective
- rhyme behaviour
- section proportions

These appear under `lyric_style_reference`, and `structure_scaffold` turns them
into a Suno-ready skeleton with placeholder counts and **no reference words**.

`--allow-reference-lyrics` switches to `cover_reference` and says loudly that
the result is a cover, not an original work.

---

## Step 3 — Prompt compilation

```bash
python scripts/compile_prompt.py analysis.json "The Weeknd" --identity identity.json
```

There is exactly one compiler. It reads `analysis.json` plus three config files
and returns:

```json
{
  "mode": "vocal",
  "style":   { "prompt": "...", "char_count": 189, "limit": 200, "tags": [], "dropped": [] },
  "exclude": { "prompt": "...", "char_count": 78,  "limit": 200, "dropped_same_root": [] },
  "lyrics_plan": { "policy": "write_new", "note": "..." },
  "theory_tags": [ { "tag": "four-chord pop loop", "why": "progression matched i-VI-III-VII" } ],
  "artist_deconstruction": { "requested": "the weeknd", "matched_key": "The Weeknd" },
  "report_only": { "_comment": "Report-only. These values MUST NOT be pasted into Suno." },
  "warnings": []
}
```

### Field budgets

One budget per field, defined once in `scripts/suno_contract.py`:

| Field | Default | Override |
|---|---|---|
| Style of Music | 200 | `--style-limit N` |
| Exclude Styles | 200 | `--exclude-limit N` |

When a budget is exceeded, **whole tags are dropped**, least-important first
(production → mood/artist → instruments → tempo). Nothing is truncated and no
ellipsis is ever emitted — a trailing `...` is itself a style token.

`dropped` lists exactly what was removed, so the choice is auditable.

### Theory tags are triggers, not defaults

A theory tag is emitted only when its evidence exists. No evidence means an
empty list, never a fallback tag.

| Tag | Requires |
|---|---|
| `four-chord pop loop` | the measured Roman loop actually contains `i-VI-III-VII` (or a listed variant), cyclically |
| `phrygian tension` | measured flat 2nd ≥ 0.35 **and** stronger than the natural 2nd |
| `dorian mode` / `mixolydian groove` | the matching modal flavour |
| `minor` / `major pentatonic` | ≥4 of 5 pentatonic degrees present, ≤1 foreign degree |
| `syncopated bass` | bass-line syncopation ratio > 0.4 (measured on the bass only) |
| `swung 8ths` | measured swing ratio ≥ 1.2 |
| `straight 16ths` | swing ratio in [0.9, 1.1] **and** measured 16th hi-hats |
| `half-time drum beat` | measured half-time kick pattern |

Artist deconstruction contributes sound and vocal descriptors only. The artist
map's `theory` field is deliberately ignored: it describes the artist in general,
not this track, so using it would reintroduce an unmeasured label.

### Exclusions

`auto_exclusion_map.yaml` keys are canonical, and each declares a `matches` alias
list containing the exact strings the compiler emits. Matching is exact after
normalisation — never a substring accident.

Every negative tag passes a same-root check against the positive tags. A negative
that shares an instrument family with a positive is dropped, and the reason is
recorded in `exclude.dropped_same_root`. That is why `harsh distortion`
disappears when a distorted guitar was requested, and `muffled vocals` disappears
for any vocal track.

---

## Step 4 — Lyric timeline for video

```bash
python scripts/align_lyrics.py song.mp3 lyrics.txt analysis.json
python scripts/align_lyrics.py song.mp3 lyrics.txt --from-lrc existing.lrc
python scripts/align_lyrics.py song.mp3 lyrics.txt --vocals stems/vocals.wav
```

This is deliberately **not** wired into the instrumental gate: a human supplied
the words, so "is there a vocal?" is already answered.

It does not re-transcribe. It aligns **your** words to the vocal stem and snaps
line heads to the beat:

1. **Separate the vocal stem** with Demucs `htdemucs`. The accompaniment is kept
   only for beat finding.
2. **CTC forced alignment** of the known lyric sheet (default
   `facebook/mms-300m`). Structure tags are stripped first and restored
   afterwards, so the aligner is never asked to find acoustics for the word
   "Chorus". Choruses are aligned every time they occur — lines are never
   de-duplicated. Non-Latin lyrics are romanised with `uroman`.
3. **Snap line heads** to the nearest beat, preferring a downbeat within 250 ms.
   The maximum shift is 200 ms: a slightly early or late syllable still reads as
   on-beat, but a 600 ms drag does not.
4. **Leave instrumental gaps wordless** and flag anything uncertain.

### Outputs

`lyrics.lrc` · `lyrics.srt` · `mv_shots.json`

```json
{
  "contract": "The timeline is locked. Only the `visual` field may be edited. `start`, `end` and `lyric` MUST NOT be changed.",
  "shots": [
    {
      "start": 32.10, "end": 36.40, "section": "Chorus",
      "lyric": "把夜色唱成海岸", "energy": "high", "visual": "",
      "needs_review": false, "alignment_score": 0.87, "snap_shift_sec": -0.04
    }
  ]
}
```

Shot rules: a line longer than 6 s is split at the nearest beat; a line shorter
than 1.5 s is merged into the next shot; gaps become wordless shots.

**Low-confidence lines are flagged, not silently accepted.** A line is marked
`needs_review` when its alignment score is below 0.5, when it lands where there
is no vocal energy, or when it needed a large snap. Fix those by hand.

`--verify-whisper` adds a read-only transcription diff that reports lines that
may be missing from or extra in the recording. It never changes a timestamp.

### Missing dependencies

Demucs and a CTC acoustic model are heavy, so this step is optional. When
something is missing the script names it, tells you how to install it, and exits
non-zero. **It never falls back to Whisper transcription** — that would rewrite
your lyrics, which is the exact failure this step exists to avoid.

---

## Dependencies

```bash
# Core analysis + compilation (required)
pip install librosa soundfile scipy numpy pyyaml faster-whisper
pip install essentia            # optional: better key/LUFS/chords, falls back cleanly

# Lyric timeline (optional, step 4 only)
pip install demucs torch torchaudio transformers uroman

# Optional accuracy upgrades
pip install silero-vad          # real VAD for the vocal gate
pip install pretty_midi         # MIDI chord analysis
```

`ffmpeg` / `ffprobe` must be on PATH. `PyYAML` is **required** — without it the
compiler stops with an explicit error instead of silently producing a prompt from
empty config tables.

`ACOUSTID_API_KEY` is optional and only enables the AcoustID channel.

---

## Accuracy

No accuracy claims are published for v7.3 yet. `eval/manifest.yaml` defines a
20-track listening set covering vocal/instrumental, mode (including true Phrygian
vs natural minor), half-time vs full-time, and the full tempo range.

```bash
python eval/score_eval.py
```

Fill in `eval/audio/` and confirm the expected values by ear, then paste the
resulting table here. **Claims follow the measurement, not the other way round.**
The v7.2 five-star ratings were removed because they rested on a single example
and compared librosa's correlation coefficient with Essentia's `strength` — two
different quantities that cannot be expressed as a "+35%" improvement.

---

## Files

| File | Role |
|---|---|
| `scripts/run_pipeline.py` | **The entry point.** Runs steps 1–3 in order. |
| `scripts/analyze_audio.py` | Measurement and evidence only. Builds no prompts. |
| `scripts/compile_prompt.py` | **The only compiler.** Produces Style / Exclude / Lyrics plan. |
| `scripts/identify_song.py` | Identity channels and the lyrics policy. |
| `scripts/align_lyrics.py` | Forced alignment → LRC, SRT, MV shot list. |
| `scripts/parse_midi.py` | MIDI harmony evidence. Emits no tags without evidence. |
| `scripts/vocal_gate.py` | The acoustic vocal/instrumental/unknown decision. |
| `scripts/music_theory.py` | Chord parsing, Roman numerals, mode detection. |
| `scripts/theory_triggers.py` | Evidence-gated theory tags. |
| `scripts/exclusion_rules.py` | Canonical genre matching and same-root filtering. |
| `scripts/structure_plan.py` | Energy-derived section labels. |
| `scripts/lyric_utils.py` | Lyric parsing and transferable style statistics. |
| `scripts/shot_builder.py` | Beat snapping, shot merging/splitting, LRC/SRT export. |
| `scripts/suno_contract.py` | Field budgets, slot order, tag vocabulary. |
| `scripts/yaml_io.py` | Explicit config loading. Missing PyYAML is never silent. |
| `auto_exclusion_map.yaml` | Canonical exclusion keys with explicit `matches` aliases. |
| `artist_feature_map.yaml` | Artist → compliant descriptors, with aliases. |
| `suno_tag_library.yaml` | Tag vocabulary and the theory whitelist. |
| `suno_structure_templates.yaml` | Section tag reference. |
| `suno_lyric_scaffolds.md` | Lyric scaffold reference. |
| `tests/` | 108 unit tests. `python -m pytest tests -q` |
| `eval/` | The 20-track listening set and its scoring script. |
