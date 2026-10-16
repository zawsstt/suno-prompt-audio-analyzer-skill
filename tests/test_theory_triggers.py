"""Theory tags must be triggered by evidence, never emitted as a default."""

from theory_triggers import (
    contains_cyclic,
    evaluate_theory_tags,
    normalize_roman,
    roman_loop_string,
)

AEOLLIAN = "Aeolian (natural minor)"


def test_empty_evidence_emits_nothing(tag_library):
    """
    v7.2 fell through to `theory_tags.append("four-chord pop loop")` when nothing
    matched. Absent evidence must produce an empty list, not a default tag.
    """
    assert evaluate_theory_tags({}, tag_library) == []


def test_natural_minor_never_produces_phrygian(tag_library):
    evidence = {
        "modal_flavor": AEOLLIAN,
        "modal_evidence": {"b2": 0.05, "natural_2": 0.90},
    }
    tags = [t["tag"] for t in evaluate_theory_tags(evidence, tag_library)]
    assert "phrygian tension" not in tags


def test_phrygian_needs_a_confirmed_flat_second(tag_library):
    weak = {
        "modal_flavor": "Phrygian",
        "modal_evidence": {"b2": 0.20, "natural_2": 0.60},
    }
    assert "phrygian tension" not in [t["tag"] for t in evaluate_theory_tags(weak, tag_library)]

    strong = {
        "modal_flavor": "Phrygian",
        "modal_evidence": {"b2": 0.85, "natural_2": 0.15},
    }
    fired = evaluate_theory_tags(strong, tag_library)
    assert "phrygian tension" in [t["tag"] for t in fired]


def test_phrygian_without_measurement_is_not_emitted(tag_library):
    """Modal flavour alone is not enough; the b2 numbers must be present."""
    evidence = {"modal_flavor": "Phrygian", "modal_evidence": {}}
    assert "phrygian tension" not in [t["tag"] for t in evaluate_theory_tags(evidence, tag_library)]


def test_four_chord_pop_loop_requires_the_actual_loop(tag_library):
    with_loop = {"roman_loop": ["i", "VI", "III", "VII"]}
    assert "four-chord pop loop" in [t["tag"] for t in evaluate_theory_tags(with_loop, tag_library)]

    without = {"roman_loop": ["i", "iv", "bII"]}
    assert "four-chord pop loop" not in [t["tag"] for t in evaluate_theory_tags(without, tag_library)]


def test_cyclic_loop_matching(tag_library):
    """A loop rotated by the analysis window must still match."""
    rotated = {"roman_loop": ["III", "VII", "i", "VI"]}
    assert "four-chord pop loop" in [t["tag"] for t in evaluate_theory_tags(rotated, tag_library)]


def test_syncopated_bass_requires_a_bass_measurement(tag_library):
    """v7.2 measured offbeats across all notes and called it a bass tag."""
    assert "syncopated bass" not in [
        t["tag"] for t in evaluate_theory_tags({"syncopation_level": "high"}, tag_library)
    ]
    fired = evaluate_theory_tags({"bass_syncopation_ratio": 0.55}, tag_library)
    assert "syncopated bass" in [t["tag"] for t in fired]


def test_swing_tag_requires_a_measured_ratio(tag_library):
    assert "swung 8ths" not in [t["tag"] for t in evaluate_theory_tags({}, tag_library)]
    assert "swung 8ths" in [
        t["tag"] for t in evaluate_theory_tags({"swing_ratio": 1.45}, tag_library)
    ]


def test_straight_16ths_requires_ratio_and_hats(tag_library):
    assert "straight 16ths" not in [
        t["tag"] for t in evaluate_theory_tags({"swing_ratio": 1.0}, tag_library)
    ]
    fired = evaluate_theory_tags({"swing_ratio": 1.0, "sixteenth_hats": True}, tag_library)
    assert "straight 16ths" in [t["tag"] for t in fired]


def test_half_time_requires_the_flag(tag_library):
    assert "half-time drum beat" not in [
        t["tag"] for t in evaluate_theory_tags({"half_time": False}, tag_library)
    ]
    assert "half-time drum beat" in [
        t["tag"] for t in evaluate_theory_tags({"half_time": True}, tag_library)
    ]


def test_max_tags_respected(tag_library):
    evidence = {
        "modal_flavor": "Dorian",
        "modal_evidence": {"b2": 0.9, "natural_2": 0.1},
        "roman_loop": ["ii", "V", "I"],
        "bass_syncopation_ratio": 0.6,
        "swing_ratio": 1.4,
        "half_time": True,
    }
    assert len(evaluate_theory_tags(evidence, tag_library, max_tags=2)) <= 2


def test_missing_library_emits_nothing():
    assert evaluate_theory_tags({"modal_flavor": "Dorian"}, None) == []
    assert evaluate_theory_tags({"modal_flavor": "Dorian"}, {}) == []


def test_roman_helpers():
    assert normalize_roman("V7") == "V"
    assert normalize_roman("iv6") == "iv"
    assert roman_loop_string(["i7", "VI", "III", "VII"]) == "i-VI-III-VII"
    assert contains_cyclic(["i", "VI", "III", "VII"], ["III", "VII", "i"]) is True
    assert contains_cyclic(["i", "iv"], ["VI", "III"]) is False
