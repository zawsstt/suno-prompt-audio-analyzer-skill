#!/usr/bin/env python3
"""
compile_prompt.py -- THE compiler. There is exactly one.

v7.2 problems this replaces
---------------------------
1. Two compilers existed. `analyze_audio.generate_suno_prompt()` capped Style at
   120/200 characters; `compile_prompt.py` and SKILL.md documented 150-350.
   Whichever one the agent happened to call won. There was no single entry point.
2. The exclusion map was keyed `cinematic_orchestral` while the compiler tested
   `"cinematic_orchestral" in "cinematic, orchestral"`, so orthogonal exclusion
   almost never fired.
3. Generic exclusions were appended unconditionally, cancelling instruments the
   positive prompt had just requested.
4. Over-long output was cut with `[:197] + "..."`, and the ellipsis became a
   style token.
5. The artist table required an exact string match, so `the weeknd` never
   matched `The Weeknd`, and the identified artist was never passed in at all.
6. Theory tags were emitted from a guess chain with no evidence check.

v7.3 contract
-------------
`compile_suno_prompt()` is the only place a Suno prompt is produced. It answers
four questions and nothing else:

    vocal or instrumental | style character | speed feel | lyrics plan

LUFS, chroma, MFCC, Roman numerals and drum timestamps stay in the report.
One budget per field. Whole tags are dropped, never truncated.

Usage:
    python scripts/compile_prompt.py analysis.json [--artist "The Weeknd"]
                                                   [--style-limit 200]
                                                   [--exclude-limit 200]
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exclusion_rules import build_exclusion_prompt  # noqa: E402
from structure_plan import build_instrumental_timeline, structure_summary  # noqa: E402
from suno_contract import (  # noqa: E402
    EXCLUDE_LIMIT_DEFAULT,
    GENRE_TO_SUNO_TAGS,
    PRESENCE_INSTRUMENTAL,
    PRESENCE_UNKNOWN,
    PRESENCE_VOCAL,
    STYLE_LIMIT_DEFAULT,
    fit_slots,
    language_tag,
    tempo_feel_tag,
)
from theory_triggers import evaluate_theory_tags  # noqa: E402
from yaml_io import YamlInvalid, YamlUnavailable, load_yaml  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Report-only fields that must never reach a Suno prompt.
REPORT_ONLY_FIELDS = [
    "lufs", "lufs_integrated", "lufs_approx", "chord_roman_analysis",
    "mfcc_means", "mfcc_stds", "chroma_profile", "beat_interval_cv",
    "kick_onsets_sec", "snare_onsets_sec", "hihat_onsets_sec",
    "spectral_centroid_hz", "crest_factor_db", "dynamic_range_db",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(repo_root=REPO_ROOT):
    """
    Load the three config tables. Missing PyYAML is FATAL here: without these
    tables the compiler cannot do its job, and silently producing a prompt from
    an empty config is exactly the v7.2 bug.
    """
    root = repo_root
    try:
        artist = load_yaml(os.path.join(root, "artist_feature_map.yaml"), required=False)
        exclusion = load_yaml(os.path.join(root, "auto_exclusion_map.yaml"), required=True)
        library = load_yaml(os.path.join(root, "suno_tag_library.yaml"), required=False)
    except YamlUnavailable as exc:
        raise SystemExit(f"error: {exc}")
    except YamlInvalid as exc:
        raise SystemExit(f"error: {exc}")
    return artist, exclusion, library


def normalize_artist_key(name):
    """'the weeknd' / 'The-Weeknd' / 'THE WEEKND' -> 'theweeknd'."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def build_artist_index(artist_config):
    """Map every accepted spelling of an artist onto its deconstruction entry."""
    index = {}
    mappings = (artist_config or {}).get("artist_mappings") or {}
    for canonical, spec in mappings.items():
        if not isinstance(spec, dict):
            continue
        index[normalize_artist_key(canonical)] = (canonical, spec)
        for alias in spec.get("aliases") or []:
            index[normalize_artist_key(alias)] = (canonical, spec)
    return index


def lookup_artist(artist_name, index):
    if not artist_name:
        return None
    return index.get(normalize_artist_key(artist_name))


# ---------------------------------------------------------------------------
# Vocal mode
# ---------------------------------------------------------------------------
def resolve_vocal_mode(analysis):
    """
    Decide the branch from `vocal_presence`, which the analyzer derives from
    acoustics. Falls back to the legacy `lyrics.has_lyrics` flag only as a
    clearly-labelled weak signal.
    """
    vp = analysis.get("vocal_presence")
    if isinstance(vp, dict) and vp.get("presence"):
        return vp["presence"], vp, "analyzer acoustic gate"

    lyrics = analysis.get("lyrics") or {}
    if lyrics.get("has_lyrics") is True:
        return (PRESENCE_VOCAL,
                {"presence": PRESENCE_VOCAL, "confidence": None,
                 "reason": "legacy has_lyrics flag"},
                "legacy has_lyrics")

    if lyrics.get("has_lyrics") is False:
        # A False here means "transcription produced nothing", which is NOT the
        # same as "this track is instrumental". Do not route to instrumental.
        return (PRESENCE_UNKNOWN,
                {"presence": PRESENCE_UNKNOWN, "confidence": None,
                 "reason": "legacy has_lyrics was False; not treated as instrumental"},
                "legacy has_lyrics")

    return (PRESENCE_UNKNOWN,
            {"presence": PRESENCE_UNKNOWN, "confidence": None,
             "reason": "no vocal evidence in the analysis"},
            "missing")


# ---------------------------------------------------------------------------
# Evidence for theory triggers
# ---------------------------------------------------------------------------
def theory_evidence(analysis):
    tonality = analysis.get("tonality") or {}
    rhythm = analysis.get("rhythm") or {}
    drums = analysis.get("drum_pattern") or {}
    chord = analysis.get("chord_progression") or {}

    roman_loop = chord.get("roman_loop")
    if not roman_loop:
        roman = analysis.get("essentia_features", {}).get("chord_roman_analysis") or {}
        if roman:
            ordered = sorted(roman.items(), key=lambda kv: kv[1].get("pct", 0), reverse=True)
            roman_loop = [v.get("roman") for _, v in ordered if v.get("roman")]

    return {
        "modal_flavor": tonality.get("modal_flavor"),
        "modal_evidence": tonality.get("modal_evidence") or {},
        "roman_loop": roman_loop,
        "pentatonic": tonality.get("pentatonic"),
        "bass_syncopation_ratio": rhythm.get("bass_syncopation_ratio"),
        "swing_ratio": rhythm.get("swing_ratio"),
        "half_time": drums.get("half_time"),
        "sixteenth_hats": drums.get("sixteenth_hats"),
        "polyrhythm": rhythm.get("polyrhythm"),
    }


# ---------------------------------------------------------------------------
# Style slots
# ---------------------------------------------------------------------------
def genre_tags_for(analysis):
    genres = (analysis.get("production_style") or {}).get("likely_genres") or []
    tags = []
    for g in genres:
        for t in GENRE_TO_SUNO_TAGS.get(g, []):
            if t not in tags:
                tags.append(t)
    return tags[:2] or ["electronic"]


def mood_tags_for(analysis, modal_flavor=""):
    """4-D mood mapping: danceability x valence x energy x mode."""
    f = analysis.get("spotify_like_features") or {}
    dance = f.get("danceability", 0.5) or 0.5
    valence = f.get("valence", 0.5) or 0.5
    energy = f.get("energy", 0.5) or 0.5
    mode = (analysis.get("tonality") or {}).get("mode", "minor")

    if dance > 0.6 and valence > 0.5 and energy > 0.6:
        word = "euphoric"
    elif dance > 0.6 and valence > 0.5:
        word = "groovy"
    elif dance > 0.6 and valence <= 0.5 and energy > 0.6:
        word = "driving"
    elif dance > 0.6:
        word = "hypnotic"
    elif valence > 0.5 and energy > 0.6:
        word = "uplifting"
    elif valence > 0.5:
        word = "serene"
    elif energy > 0.6:
        word = "intense" if mode == "minor" else "powerful"
    else:
        word = "brooding" if "aeolian" in str(modal_flavor).lower() else "contemplative"
    return [word]


def instrument_tags_for(analysis, genre_tags):
    """2-3 instrument tags, all from measured evidence."""
    bands = analysis.get("frequency_bands") or {}
    sub_pct = (bands.get("sub_bass_20_60hz") or {}).get("energy_pct", 0) or 0
    bass_pct = (bands.get("bass_60_250hz") or {}).get("energy_pct", 0) or 0
    drums = analysis.get("drum_pattern") or {}
    kick_pb = drums.get("kick_per_bar", 0) or 0
    hihat_pb = drums.get("hihat_per_bar", 0) or 0

    joined = " ".join(genre_tags).lower()
    is_orch = "orchestral" in joined
    is_hiphop = any(k in joined for k in ("trap", "hip-hop", "boom bap"))
    is_dnb = any(k in joined for k in ("drum and bass", "breakbeat"))

    tags = []
    if is_orch:
        tags.append("orchestral strings")
    elif is_hiphop:
        if sub_pct > 12:
            tags.append("808 bass")
        elif bass_pct > 25:
            tags.append("punchy bass")
    elif is_dnb:
        if sub_pct > 12:
            tags.append("heavy sub bass")
        else:
            tags.append("rolling bass")
    else:
        if sub_pct > 15:
            tags.append("heavy sub bass")
        elif sub_pct > 8:
            tags.append("deep sub bass")
        elif bass_pct > 20:
            tags.append("warm bass")

    if not is_orch:
        if kick_pb >= 3.5:
            tags.append("four-on-the-floor kick")
        elif hihat_pb >= 5:
            tags.append("rolling hi-hats")
        elif kick_pb >= 1.5:
            tags.append("punchy kick")

    return tags[:3]


def production_tags_for(analysis):
    crest = (analysis.get("loudness") or {}).get("crest_factor_db", 12) or 12
    hp = analysis.get("harmonic_percussive") or {}
    h_ratio = hp.get("harmonic_ratio", 0.5) or 0.5
    brightness = (analysis.get("spectral") or {}).get("brightness", "mid")

    if h_ratio > 0.85:
        tag = "atmospheric"
    elif crest < 7:
        tag = "heavily compressed"
    elif crest > 16:
        tag = "wide dynamics"
    elif brightness == "dark":
        tag = "warm analog production"
    else:
        tag = "clean modern production"
    return [tag]


# ---------------------------------------------------------------------------
# Lyrics plan
# ---------------------------------------------------------------------------
def lyrics_plan_for(analysis, mode, identity=None):
    """
    Decide what goes into the Lyrics box.

    Original song => reference WORDS never appear here.
    """
    plan = {
        "policy": None,
        "scaffold": None,
        "style_reference": None,
        "note": None,
    }

    if mode == PRESENCE_INSTRUMENTAL:
        plan["policy"] = "structure_only"
        plan["scaffold"] = build_instrumental_timeline(analysis)
        plan["note"] = ("Instrumental track: paste the scaffold into the Lyrics box. "
                        "Sections are named only where the energy curve supports them.")
        return plan

    if mode == PRESENCE_UNKNOWN:
        plan["policy"] = "human_decision"
        plan["note"] = ("Vocal presence could not be established. Decide manually: "
                        "write a new lyric, or use the instrumental scaffold.")
        plan["scaffold"] = build_instrumental_timeline(analysis)
        plan["alternative"] = "write_new"
        return plan

    plan["policy"] = "write_new"
    plan["note"] = ("Write a NEW lyric. The reference contributes style statistics only; "
                    "its words must not be reused.")

    ref = (identity or {}).get("lyric_style_reference")
    if ref:
        plan["style_reference"] = ref
        plan["scaffold"] = (identity or {}).get("structure_scaffold")

    return plan


# ---------------------------------------------------------------------------
# Main compile
# ---------------------------------------------------------------------------
def compile_suno_prompt(analysis,
                        artist_name="",
                        identity=None,
                        style_limit=STYLE_LIMIT_DEFAULT,
                        exclude_limit=EXCLUDE_LIMIT_DEFAULT,
                        repo_root=REPO_ROOT,
                        config=None):
    """
    Compile the three Suno fields from an analysis dict.

    Returns a dict with `style`, `exclude`, `lyrics_plan`, `mode` and a full
    `provenance` block recording what was used and what was dropped.
    """
    warnings = []
    artist_config, exclusion_config, library = config if config else load_config(repo_root)

    mode, vocal_presence, mode_source = resolve_vocal_mode(analysis)

    # ---- slots ------------------------------------------------------------
    genre_tags = genre_tags_for(analysis)
    modal_flavor = (analysis.get("tonality") or {}).get("modal_flavor", "")

    theory = []
    if library:
        theory = evaluate_theory_tags(theory_evidence(analysis), library, max_tags=2)
    else:
        warnings.append("suno_tag_library.yaml missing or empty; no theory tags emitted.")
    theory_tags = [t["tag"] for t in theory]

    artist_tags = []
    artist_match = None
    if artist_name:
        index = build_artist_index(artist_config)
        hit = lookup_artist(artist_name, index)
        if hit:
            canonical, spec = hit
            artist_match = {"requested": artist_name, "matched_key": canonical}
            artist_tags.extend((spec.get("production_sound") or [])[:2])
            artist_tags.extend((spec.get("vocals") or [])[:1])
            # NOTE: `spec["theory"]` is deliberately NOT used. It describes the
            # artist in general, not this track, so emitting it would reintroduce
            # exactly the v7.2 bug of shipping an unmeasured theory label.
            artist_match["theory_tags_ignored"] = spec.get("theory") or []
        else:
            warnings.append(
                f"artist '{artist_name}' is not in artist_feature_map.yaml; "
                "no deconstruction applied."
            )

    instrument_tags = instrument_tags_for(analysis, genre_tags)
    mood_tags = mood_tags_for(analysis, modal_flavor)
    production_tags = production_tags_for(analysis)

    lyrics = analysis.get("lyrics") or {}
    language = lyrics.get("language") if mode != PRESENCE_INSTRUMENTAL else None
    lang_tag = language_tag(language) if language else None

    tempo_tag = tempo_feel_tag((analysis.get("summary") or {}).get("tempo_bpm"))

    if mode == PRESENCE_INSTRUMENTAL:
        vocal_slot = ["instrumental"]
    elif mode == PRESENCE_VOCAL:
        vocal_slot = [lang_tag] if lang_tag else ["vocals"]
    else:
        # Unknown: never claim instrumental, never claim a language we did not
        # hear. The Style field simply does not commit.
        vocal_slot = []
        warnings.append(
            "Vocal presence is unknown; the Style prompt does not commit to either "
            "and the Lyrics field is left for a human decision."
        )

    slots = {
        "genre": genre_tags,
        "vocal_mode": vocal_slot,
        "instruments": instrument_tags,
        "mood": mood_tags,
        "tempo": [tempo_tag] if tempo_tag else [],
        "production": production_tags,
    }

    # Theory + artist tags slot in right after genre/mood, before instruments.
    slots["mood"] = mood_tags + theory_tags + artist_tags

    kept, dropped = fit_slots(slots, style_limit)
    style_prompt = ", ".join(kept)

    # ---- artist-name leak guard -------------------------------------------
    leak = artist_name_leak(style_prompt, artist_name)
    if leak:
        style_prompt = strip_leak(style_prompt, leak)
        kept = [t for t in style_prompt.split(", ") if t]
        warnings.append(f"removed artist name '{leak}' from the style prompt.")

    # ---- exclusions -------------------------------------------------------
    exclusion = build_exclusion_prompt(
        positive_tags=kept,
        genre_tags=genre_tags,
        exclusion_map=exclusion_config,
        limit=exclude_limit,
    )

    plan = lyrics_plan_for(analysis, mode, identity=identity)

    return {
        "mode": mode,
        "vocal_presence": vocal_presence,
        "mode_source": mode_source,
        "style": {
            "prompt": style_prompt,
            "tags": kept,
            "char_count": len(style_prompt),
            "limit": style_limit,
            "dropped": dropped,
        },
        "exclude": exclusion,
        "lyrics_plan": plan,
        "theory_tags": theory,
        "artist_deconstruction": artist_match,
        "report_only": report_only_block(analysis),
        "structure_evidence": structure_summary(analysis),
        "provenance": {
            "genre_tags": genre_tags,
            "mood_tags": mood_tags,
            "instrument_tags": instrument_tags,
            "production_tags": production_tags,
            "tempo_tag": tempo_tag,
            "language_tag": lang_tag,
            "artist_tags_used": artist_tags,
            "compiler": "compile_prompt.py v7.3 (single compiler)",
        },
        "warnings": warnings,
    }


def report_only_block(analysis):
    """
    Explicitly park the measurements that must never enter a Suno field.
    Kept in the output so a human can still use them, clearly labelled.
    """
    summary = analysis.get("summary") or {}
    return {
        "_comment": "Report-only. These values MUST NOT be pasted into Suno.",
        "lufs": summary.get("lufs"),
        "dynamic_range_db": summary.get("dynamic_range_db"),
        "key_full": summary.get("key_full"),
        "spectral_centroid_hz": (analysis.get("spectral") or {}).get("centroid_hz"),
        "chord_roman_analysis": (analysis.get("essentia_features") or {}).get("chord_roman_analysis"),
        "beat_interval_cv": (analysis.get("rhythm") or {}).get("beat_interval_cv"),
    }


def artist_name_leak(text, artist_name):
    """Return the artist spelling that leaked into `text`, if any."""
    if not artist_name:
        return None
    norm_text = normalize_artist_key(text)
    norm_artist = normalize_artist_key(artist_name)
    if norm_artist and norm_artist in norm_text:
        return artist_name
    return None


def strip_leak(prompt, leak):
    parts = [p.strip() for p in prompt.split(",")]
    norm_leak = normalize_artist_key(leak)
    return ", ".join(p for p in parts if norm_leak not in normalize_artist_key(p))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Compile Suno Style / Exclude / Lyrics fields.")
    ap.add_argument("analysis_json")
    ap.add_argument("artist", nargs="?", default="", help="identified artist name (optional)")
    ap.add_argument("--identity", default=None, help="identity.json from identify_song.py")
    ap.add_argument("--style-limit", type=int, default=STYLE_LIMIT_DEFAULT)
    ap.add_argument("--exclude-limit", type=int, default=EXCLUDE_LIMIT_DEFAULT)
    args = ap.parse_args(argv)

    with open(args.analysis_json, "r", encoding="utf-8") as fh:
        analysis = json.load(fh)

    identity = None
    if args.identity and os.path.exists(args.identity):
        with open(args.identity, "r", encoding="utf-8") as fh:
            identity = json.load(fh)

    result = compile_suno_prompt(
        analysis,
        artist_name=args.artist,
        identity=identity,
        style_limit=args.style_limit,
        exclude_limit=args.exclude_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
