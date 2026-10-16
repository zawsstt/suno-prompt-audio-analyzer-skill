#!/usr/bin/env python3
"""
parse_midi.py -- MIDI analysis that only reports what it actually measured.

v7.2 problems
-------------
1. When `pretty_midi` was missing the except-branch returned
   `theory_tags = ["four-chord pop loop", "syncopated bass"]`. It invented tags
   for music it had not analysed at all, and those tags went straight into a
   Suno prompt.
2. Even with pretty_midi present, no chord was ever detected, yet a major-key
   scale inference emitted "four-chord pop loop".
3. Drum-track notes were mixed into the pitch statistics, so the inferred key
   was contaminated by noise hits.
4. "syncopated bass" was measured across ALL notes, so a busy hi-hat could
   produce a bass claim.

v7.3 contract
-------------
* Missing dependency => `available: false`, empty `theory_tags`, explicit error.
* Chord progressions are actually detected from sounding notes, then converted
  to Roman numerals and handed to the shared trigger evaluator.
* Drum tracks are excluded from every pitch statistic.
* Bass syncopation is measured on the bass track only.
* Phrygian requires a measured flat 2nd (shared with the analyzer).

Usage:
    python parse_midi.py <midi_file_path> [--output-json <path>]
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_theory import (  # noqa: E402
    KEY_NAMES,
    chord_to_roman,
    detect_mode_detailed,
    detect_pentatonic,
    ks_key,
    match_chord,
)

BASS_PROGRAMS = set(range(32, 40))  # GM acoustic + synth bass


def _empty_result(reason, available=False):
    return {
        "available": available,
        "error": reason,
        "tempo_bpm": None,
        "tracks": [],
        "detected_key": None,
        "scale_mode": None,
        "modal_evidence": {},
        "pentatonic": None,
        "chord_progression_raw": [],
        "roman_numerals": [],
        "roman_loop": [],
        "bass_syncopation_ratio": None,
        "theory_evidence": {},
        # The critical v7.3 change: an unanalysed file yields NO theory tags.
        "theory_tags": [],
        "theory_tag_reasons": [],
    }


def _pick_bass_track(pitched_tracks):
    """Choose the bass track by GM program first, then by lowest median pitch."""
    for t in pitched_tracks:
        if t["program"] in BASS_PROGRAMS:
            return t
    candidates = [t for t in pitched_tracks if t["median_pitch"] is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda t: t["median_pitch"])


def parse_midi_file(midi_path, library=None):
    """
    Parse a MIDI file and return measured evidence plus whitelisted theory tags.

    `library` is the parsed suno_tag_library.yaml. When omitted, theory tags are
    still produced for evidence that does not need the whitelist (mode / groove)
    using the built-in tag names, but the trigger evaluation is skipped.
    """
    try:
        import pretty_midi
    except ImportError:
        return _empty_result(
            "pretty_midi is not installed. Install it with: pip install pretty_midi. "
            "No theory tags were emitted because nothing was measured."
        )

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as exc:
        return _empty_result(f"Could not read MIDI file: {exc}")

    result = _empty_result("", available=True)
    result.pop("error", None)

    tempos, _ = pm.get_tempo_changes()
    bpm = round(float(tempos[0]), 1) if len(tempos) else 120.0
    result["tempo_bpm"] = bpm

    # ---- split tracks: drums are NEVER allowed into pitch statistics --------
    pitched_tracks = []
    drum_track_count = 0

    for idx, instrument in enumerate(pm.instruments):
        notes = list(instrument.notes)
        info = {
            "name": instrument.name or f"Track {idx}",
            "is_drum": bool(instrument.is_drum),
            "program": int(instrument.program),
            "note_count": len(notes),
        }
        if notes:
            pitches = [n.pitch for n in notes]
            info["min_pitch"] = min(pitches)
            info["max_pitch"] = max(pitches)
            info["median_pitch"] = sorted(pitches)[len(pitches) // 2]
        else:
            info["median_pitch"] = None

        if instrument.is_drum:
            drum_track_count += 1
        elif notes:
            pitched_tracks.append({**info, "notes": notes})
        result["tracks"].append(info)

    result["drum_tracks_excluded"] = drum_track_count

    if not pitched_tracks:
        result["error"] = ("No pitched (non-drum) tracks found; key and chord "
                           "analysis is not possible.")
        return result

    # ---- key + mode from pitched notes only --------------------------------
    pitch_hist = collections.Counter()
    for t in pitched_tracks:
        for n in t["notes"]:
            pitch_hist[n.pitch % 12] += max(0.05, n.end - n.start)

    chroma = [pitch_hist.get(i, 0.0) for i in range(12)]
    total = sum(chroma) or 1.0
    chroma = [c / total for c in chroma]

    root_name, mode, ks_score = ks_key(chroma)
    root_idx = KEY_NAMES.index(root_name)
    modal, modal_evidence = detect_mode_detailed(chroma, root_idx)

    result["detected_key"] = root_name
    result["scale_mode"] = mode
    result["modal_flavor"] = modal
    result["modal_evidence"] = modal_evidence
    result["ks_confidence"] = ks_score

    degree_labels = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]
    rotated = [chroma[(i + root_idx) % 12] for i in range(12)]
    peak = max(rotated) or 1e-9
    strong_degrees = [degree_labels[i] for i in range(12) if rotated[i] / peak > 0.5]
    result["strong_scale_degrees"] = strong_degrees
    result["pentatonic"] = detect_pentatonic(strong_degrees)

    # ---- real chord detection ---------------------------------------------
    beat = 60.0 / bpm
    window = beat * 2  # analyse two beats at a time
    end_time = max((n.end for t in pitched_tracks for n in t["notes"]), default=0.0)

    chord_sequence = []
    t_cursor = 0.0
    while t_cursor < end_time:
        t_next = t_cursor + window
        weights = [0.0] * 12
        for t in pitched_tracks:
            for n in t["notes"]:
                overlap = min(n.end, t_next) - max(n.start, t_cursor)
                if overlap > 0:
                    weights[n.pitch % 12] += overlap
        if sum(weights) > 0:
            chord, score = match_chord(weights)
            if chord and score >= 0.25:
                if not chord_sequence or chord_sequence[-1]["chord"] != chord:
                    chord_sequence.append({"chord": chord, "start": round(t_cursor, 3),
                                           "score": score})
        t_cursor = t_next

    result["chord_progression_raw"] = chord_sequence[:32]

    roman_loop = []
    for entry in chord_sequence[:12]:
        r = chord_to_roman(entry["chord"], root_name, mode)
        if r:
            roman_loop.append(r)
    result["roman_numerals"] = roman_loop
    result["roman_loop"] = roman_loop

    # ---- bass syncopation, measured on the BASS track only -----------------
    bass_track = _pick_bass_track(pitched_tracks)
    bass_syncopation = None
    if bass_track and len(bass_track["notes"]) >= 8:
        offbeat = 0
        for n in bass_track["notes"]:
            phase = (n.start % beat) / beat
            if 0.3 < phase < 0.7:
                offbeat += 1
        bass_syncopation = round(offbeat / len(bass_track["notes"]), 4)
    result["bass_syncopation_ratio"] = bass_syncopation
    result["bass_track"] = bass_track["name"] if bass_track else None

    # ---- hook motifs (real repetition, not a guess) -------------------------
    if bass_track:
        intervals = [n.pitch for n in bass_track["notes"][:24]]
        if len(intervals) >= 8:
            pattern = tuple(intervals[:4])
            repeats = sum(1 for i in range(len(intervals) - 4)
                          if tuple(intervals[i:i + 4]) == pattern)
            result["hook_detected"] = repeats >= 1

    # ---- theory tags via the shared trigger evaluator -----------------------
    evidence = {
        "modal_flavor": modal,
        "modal_evidence": modal_evidence,
        "roman_loop": roman_loop,
        "pentatonic": result["pentatonic"],
        "bass_syncopation_ratio": bass_syncopation,
        "swing_ratio": None,          # MIDI gives no audio swing measurement
        "half_time": None,            # not measured from note data
        "sixteenth_hats": None,
        "polyrhythm": None,
    }
    result["theory_evidence"] = evidence

    if library:
        from theory_triggers import evaluate_theory_tags
        fired = evaluate_theory_tags(evidence, library, max_tags=3)
        result["theory_tags"] = [f["tag"] for f in fired]
        result["theory_tag_reasons"] = fired
    else:
        # No whitelist loaded: emit only the tags whose evidence is unambiguous
        # and self-contained, and say why.
        tags, reasons = [], []
        if modal.startswith("Dorian"):
            tags.append("dorian mode")
            reasons.append({"tag": "dorian mode", "why": "dorian detected in MIDI notes"})
        if modal == "Phrygian" and modal_evidence.get("flat_2_confirmed"):
            tags.append("phrygian tension")
            reasons.append({"tag": "phrygian tension",
                            "why": "flat 2nd confirmed in MIDI notes"})
        if modal.startswith("Mixolydian"):
            tags.append("mixolydian groove")
            reasons.append({"tag": "mixolydian groove", "why": "mixolydian detected"})
        if result["pentatonic"] == "minor":
            tags.append("minor pentatonic")
            reasons.append({"tag": "minor pentatonic", "why": "minor pentatonic degree set"})
        result["theory_tags"] = tags
        result["theory_tag_reasons"] = reasons
        result["theory_tag_note"] = ("suno_tag_library.yaml was not supplied; only "
                                     "self-contained mode tags were emitted.")

    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="MIDI theory analysis.")
    ap.add_argument("midi_file")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--tag-library", default=None,
                    help="path to suno_tag_library.yaml (enables trigger evaluation)")
    args = ap.parse_args(argv)

    library = None
    if args.tag_library:
        from yaml_io import load_yaml
        library = load_yaml(args.tag_library, required=True)

    res = parse_midi_file(args.midi_file, library=library)
    text = json.dumps(res, indent=2, ensure_ascii=False)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0 if res.get("available") else 2


if __name__ == "__main__":
    sys.exit(main())
