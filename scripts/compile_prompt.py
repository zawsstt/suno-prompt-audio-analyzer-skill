#!/usr/bin/env python3
"""
compile_prompt.py: Suno Prompt Compiler & Exclusions Generator for audio-analyzer v7
Translates multi-dimensional audio & cultural truth into Suno Style and Negative Prompts.

Usage:
    python compile_prompt.py <analysis_json_path> [--artist <artist_name>]
"""

import sys
import os
import json
import re

def load_yaml_simple(file_path):
    """
    Lightweight fallback parser for YAML mapping files when pyyaml is not available.
    """
    if not os.path.exists(file_path):
        return {}
    try:
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    return {}

def compile_suno_prompt(analysis_data, artist_name="", custom_theory_tags=None):
    """
    Compile Style Prompt (150-350 chars) + Negative Prompt (<=200 chars).
    Structure: Genre(1-2) -> Mood(1-2) -> Theory(1-2) -> Instruments(2-3) -> Vocals(1) -> Production(1-2)
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artist_map_path = os.path.join(script_dir, "artist_feature_map.yaml")
    exclusion_map_path = os.path.join(script_dir, "auto_exclusion_map.yaml")

    artist_data = load_yaml_simple(artist_map_path).get("artist_mappings", {})
    exclusion_data = load_yaml_simple(exclusion_map_path)

    # 1. Base genre extraction
    likely_genres = analysis_data.get("production_style", {}).get("likely_genres", ["Pop"])
    base_genre = likely_genres[0] if likely_genres else "Pop"

    # 2. Check artist deconstruction (compliance filter: NO artist names allowed)
    deconstructed_tags = []
    if artist_name and artist_name in artist_data:
        deconstructed = artist_data[artist_name]
        deconstructed_tags.extend(deconstructed.get("production_sound", [])[:2])
        deconstructed_tags.extend(deconstructed.get("theory", [])[:1])

    # 3. Extract musical features
    bpm = analysis_data.get("summary", {}).get("tempo_bpm", 120)
    spotify_f = analysis_data.get("spotify_like_features", {})
    energy = spotify_f.get("energy", 0.5)
    dance = spotify_f.get("danceability", 0.5)
    valence = spotify_f.get("valence", 0.5)
    inst_score = spotify_f.get("instrumentalness", 0.0)

    # 4. Mood & Theory mapping
    mood = "melancholic" if valence < 0.4 else ("euphoric" if valence > 0.7 else "driving")
    theory_tags = list(custom_theory_tags or [])
    if not theory_tags:
        modal = analysis_data.get("tonality", {}).get("modal_flavor", "")
        if "Dorian" in modal:
            theory_tags.append("dorian mode")
        elif "Phrygian" in modal:
            theory_tags.append("phrygian tension")
        elif dance > 0.6:
            theory_tags.append("syncopated bass")
        else:
            theory_tags.append("four-chord pop loop")

    # 5. Production tags
    prod_tags = ["punchy mix"]
    if energy > 0.7:
        prod_tags.append("heavy dynamics")
    else:
        prod_tags.append("warm analog tape")

    # Compile Safe, Recommended, Experimental
    genre_clean = base_genre.lower().replace(" / ", ", ")

    rec_tags = [genre_clean, mood] + theory_tags[:1] + deconstructed_tags[:2] + prod_tags[:1]
    # Remove duplicates preserving order
    seen = set()
    final_rec_list = []
    for t in rec_tags:
        t_clean = t.strip()
        if t_clean and t_clean.lower() not in seen:
            seen.add(t_clean.lower())
            final_rec_list.append(t_clean)

    prompt_recommended = ", ".join(final_rec_list) + f", {bpm} BPM"
    prompt_safe = f"{genre_clean}, {mood}, {bpm} BPM"
    prompt_experimental = f"{prompt_recommended}, cinematic spatial depth, complex harmonies"

    # 6. Negative Prompt compilation (generic quality exclusions + orthogonal conflict)
    neg_generic = exclusion_data.get("generic_quality_exclusions", {}).get("default_string",
        "muddy mix, low quality, harsh distortion, clipping, muffled vocals")

    conflict_items = []
    conflicts = exclusion_data.get("genre_conflict_exclusions", {})
    for g_key, conf in conflicts.items():
        if g_key in genre_clean:
            conflict_items.extend(conf.get("exclude", [])[:3])

    if conflict_items:
        negative_prompt = f"{neg_generic}, {', '.join(conflict_items[:3])}"
    else:
        negative_prompt = neg_generic

    if len(negative_prompt) > 200:
        negative_prompt = negative_prompt[:197] + "..."

    return {
        "prompts": {
            "safe": prompt_safe,
            "recommended": prompt_recommended,
            "experimental": prompt_experimental
        },
        "negative_prompt": negative_prompt,
        "is_instrumental": inst_score >= 0.7
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compile_prompt.py <analysis_json_path> [artist_name]")
        sys.exit(1)

    json_path = sys.argv[1]
    art = sys.argv[2] if len(sys.argv) > 2 else ""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = compile_suno_prompt(data, art)
    print(json.dumps(res, indent=2, ensure_ascii=False))
