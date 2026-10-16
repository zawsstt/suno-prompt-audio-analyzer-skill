"""End-to-end compiler behaviour on synthetic analysis dicts.

These run without librosa: the compiler only consumes the evidence block.
"""

import pytest

from compile_prompt import (
    REPO_ROOT,
    build_artist_index,
    compile_suno_prompt,
    load_config,
    normalize_artist_key,
)


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT)


def vocal_analysis(**overrides):
    base = {
        "file_info": {"duration_sec": 213.4},
        "summary": {"tempo_bpm": 104.0, "lufs": -11.2, "dynamic_range_db": 7.4,
                    "key_full": "A minor (Aeolian (natural minor))"},
        "vocal_presence": {
            "presence": "vocal", "score": 0.71, "confidence": 0.42,
            "reason": "Sustained pitched activity in the vocal range dominates the mix.",
            "decided_by": "acoustic",
        },
        "lyrics": {"has_lyrics": True, "language": "zh"},
        "production_style": {"likely_genres": ["R&B / Neo-soul", "Pop / Indie Pop"]},
        "spotify_like_features": {"valence": 0.32, "energy": 0.55, "danceability": 0.62},
        "frequency_bands": {
            "sub_bass_20_60hz": {"energy_pct": 6.2},
            "bass_60_250hz": {"energy_pct": 24.0},
        },
        "drum_pattern": {"kick_per_bar": 2.0, "hihat_per_bar": 6.0, "snare_per_bar": 2.0,
                         "half_time": False, "sixteenth_hats": False},
        "loudness": {"crest_factor_db": 11.0},
        "harmonic_percussive": {"harmonic_ratio": 0.72},
        "spectral": {"brightness": "mid", "centroid_hz": 2100},
        "tonality": {
            "mode": "minor",
            "modal_flavor": "Aeolian (natural minor)",
            "modal_evidence": {"b2": 0.05, "natural_2": 0.88},
            "pentatonic": "minor",
        },
        "rhythm": {"bass_syncopation_ratio": 0.52, "swing_ratio": 1.0,
                   "beat_interval_cv": 0.021},
        "chord_progression": {"roman_loop": ["i", "VI", "III", "VII"]},
        "essentia_features": {"chord_roman_analysis": {"A:min": {"pct": 44.0, "roman": "i"}}},
        "structure": {"energy_drop_times_sec": [], "energy_peak_times_sec": []},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Vocal mode
# ---------------------------------------------------------------------------
def test_vocal_track_is_not_treated_as_instrumental(config):
    result = compile_suno_prompt(vocal_analysis(), config=config)
    assert result["mode"] == "vocal"
    assert "instrumental" not in result["style"]["prompt"].lower()
    assert result["lyrics_plan"]["policy"] == "write_new"


def test_unknown_presence_does_not_claim_instrumental(config):
    analysis = vocal_analysis(vocal_presence={"presence": "unknown", "reason": "ambiguous"})
    result = compile_suno_prompt(analysis, config=config)

    assert result["mode"] == "unknown"
    assert "instrumental" not in result["style"]["prompt"].lower()
    assert result["lyrics_plan"]["policy"] == "human_decision"
    assert any("unknown" in w.lower() for w in result["warnings"])


def test_legacy_has_lyrics_false_is_not_instrumental(config):
    """A failed transcription is not evidence that a track is instrumental."""
    analysis = vocal_analysis()
    analysis.pop("vocal_presence")
    analysis["lyrics"] = {"has_lyrics": False}

    result = compile_suno_prompt(analysis, config=config)
    assert result["mode"] == "unknown"
    assert "instrumental" not in result["style"]["prompt"].lower()


def test_instrumental_track_gets_structure_only(config):
    analysis = vocal_analysis(vocal_presence={
        "presence": "instrumental", "score": 0.1,
        "reason": "No sustained pitched activity in the vocal range."})
    result = compile_suno_prompt(analysis, config=config)

    assert result["mode"] == "instrumental"
    assert "instrumental" in result["style"]["prompt"].lower()
    assert result["lyrics_plan"]["policy"] == "structure_only"


# ---------------------------------------------------------------------------
# Theory tags
# ---------------------------------------------------------------------------
def test_natural_minor_never_yields_phrygian_tension(config):
    result = compile_suno_prompt(vocal_analysis(), config=config)
    tags = [t["tag"] for t in result["theory_tags"]]
    assert "phrygian tension" not in tags
    assert "phrygian" not in result["style"]["prompt"].lower()


def test_four_chord_loop_requires_the_loop(config):
    without = vocal_analysis()
    without["chord_progression"] = {"roman_loop": ["i", "iv"]}
    without["essentia_features"] = {}
    result = compile_suno_prompt(without, config=config)
    assert "four-chord pop loop" not in [t["tag"] for t in result["theory_tags"]]


def test_no_theory_evidence_means_no_theory_tags(config):
    analysis = vocal_analysis()
    analysis["tonality"]["modal_flavor"] = ""
    analysis["tonality"]["modal_evidence"] = {}
    analysis["tonality"]["pentatonic"] = None
    analysis["chord_progression"] = {}
    analysis["essentia_features"] = {}
    analysis["rhythm"] = {}

    result = compile_suno_prompt(analysis, config=config)
    assert result["theory_tags"] == []


# ---------------------------------------------------------------------------
# Budget discipline
# ---------------------------------------------------------------------------
def test_style_prompt_respects_the_budget(config):
    result = compile_suno_prompt(vocal_analysis(), config=config, style_limit=60)
    assert result["style"]["char_count"] <= 60
    assert "..." not in result["style"]["prompt"]
    assert "\u2026" not in result["style"]["prompt"]


def test_over_budget_tags_are_dropped_whole(config):
    result = compile_suno_prompt(vocal_analysis(), config=config, style_limit=45)
    for tag in result["style"]["tags"]:
        assert tag in result["style"]["prompt"]
    assert result["style"]["dropped"]


def test_default_budget_is_single_sourced(config):
    """v7.2 had 120, 200 and 350 in different files."""
    from suno_contract import STYLE_LIMIT_DEFAULT
    result = compile_suno_prompt(vocal_analysis(), config=config)
    assert result["style"]["limit"] == STYLE_LIMIT_DEFAULT


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
def test_exclude_never_contains_an_ellipsis(config):
    result = compile_suno_prompt(vocal_analysis(), config=config)
    assert "..." not in result["exclude"]["prompt"]
    assert "\u2026" not in result["exclude"]["prompt"]


def test_positive_instruments_are_not_negated(config):
    result = compile_suno_prompt(vocal_analysis(), config=config)
    negatives = result["exclude"]["prompt"].lower()
    # The positive prompt asked for vocals in a language; "muffled vocals" must go.
    assert "muffled vocals" not in negatives


# ---------------------------------------------------------------------------
# Artist deconstruction
# ---------------------------------------------------------------------------
def test_artist_lookup_is_case_insensitive():
    index = build_artist_index({
        "artist_mappings": {
            "The Weeknd": {"aliases": ["the weeknd", "weeknd"],
                           "production_sound": ["vintage analog synth bass"],
                           "theory": ["minor pentatonic"],
                           "vocals": ["soaring falsetto"]}
        }
    })
    assert normalize_artist_key("the weeknd") in index
    assert normalize_artist_key("THE WEEKND") in index
    assert normalize_artist_key("The-Weeknd") in index


def test_artist_name_never_reaches_the_prompt(config):
    for spelling in ("The Weeknd", "the weeknd", "THE WEEKND", "the-weeknd"):
        result = compile_suno_prompt(vocal_analysis(), artist_name=spelling, config=config)
        assert "weeknd" not in result["style"]["prompt"].lower()


def test_unknown_artist_produces_a_warning_not_a_guess(config):
    result = compile_suno_prompt(vocal_analysis(), artist_name="Nobody At All", config=config)
    assert any("not in artist_feature_map" in w for w in result["warnings"])
    assert result["artist_deconstruction"] is None


def test_artist_deconstruction_contributes_tags(config):
    result = compile_suno_prompt(vocal_analysis(), artist_name="the weeknd",
                                 config=config, style_limit=400)
    assert result["artist_deconstruction"]["matched_key"] == "The Weeknd"
    assert "falsetto" in result["style"]["prompt"].lower()


# ---------------------------------------------------------------------------
# Report-only separation
# ---------------------------------------------------------------------------
def test_report_only_values_stay_out_of_the_prompt(config):
    result = compile_suno_prompt(vocal_analysis(), config=config)
    prompt = result["style"]["prompt"]

    for leaked in ("-11.2", "LUFS", "lufs", "mfcc", "chroma", "0.021"):
        assert leaked not in prompt

    assert "MUST NOT be pasted" in result["report_only"]["_comment"]
    assert result["report_only"]["lufs"] == -11.2


def test_structure_evidence_is_reported_for_the_human(config):
    result = compile_suno_prompt(vocal_analysis(), config=config)
    assert "has_drop" in result["structure_evidence"]
    assert result["structure_evidence"]["has_drop"] is False
