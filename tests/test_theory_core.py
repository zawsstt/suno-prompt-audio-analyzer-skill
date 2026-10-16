"""Regression tests for the two theory bugs that produced wrong Suno tags."""

import numpy as np
import pytest

from music_theory import (
    chord_to_roman,
    detect_mode,
    detect_mode_detailed,
    detect_pentatonic,
    parse_chord,
)


def chroma(weights_by_degree):
    """Build a 12-element chroma vector from {scale_degree: weight}."""
    v = np.zeros(12)
    for degree, w in weights_by_degree.items():
        v[degree % 12] = w
    return v


# ---------------------------------------------------------------------------
# BUG 1 (v7.2): natural minor was labelled Phrygian
# ---------------------------------------------------------------------------
def test_natural_minor_is_not_phrygian():
    """
    v7.2: `if m6 > M6 and m7 > M7: return "Phrygian"`.
    Natural minor IS b6 + b7, so this returned Phrygian for almost every minor
    song, and the compiler then wrote "phrygian tension" into the prompt.
    """
    v = chroma({0: 1.0, 2: 0.85, 3: 1.0, 5: 0.9, 7: 1.0, 8: 1.1, 10: 1.2})
    mode, evidence = detect_mode_detailed(v, 0)

    assert mode == "Aeolian (natural minor)"
    assert evidence["flat_2_confirmed"] is False
    assert "no flat 2nd" in evidence["decision"]


def test_natural_minor_with_dominant_b6_b7_is_still_aeolian():
    """
    The exact shape that triggered the v7.2 bug: b6 and b7 are the two loudest
    degrees in the chroma vector.
    """
    v = chroma({0: 0.9, 2: 0.7, 3: 1.0, 5: 0.8, 7: 1.0, 8: 1.35, 10: 1.4})
    assert detect_mode(v, 0) == "Aeolian (natural minor)"


def test_real_phrygian_requires_a_flat_second():
    v = chroma({0: 1.0, 1: 1.0, 3: 1.0, 5: 0.9, 7: 1.0, 8: 1.0, 10: 1.0})
    mode, evidence = detect_mode_detailed(v, 0)

    assert mode == "Phrygian"
    assert evidence["flat_2_confirmed"] is True
    assert evidence["b2"] > evidence["natural_2"]


def test_weak_flat_second_does_not_trigger_phrygian():
    """A passing b2 must not be enough to claim a Phrygian mode."""
    v = chroma({0: 1.0, 1: 0.30, 2: 0.55, 3: 1.0, 5: 0.9, 7: 1.0, 8: 1.1, 10: 1.1})
    mode, evidence = detect_mode_detailed(v, 0)
    assert mode == "Aeolian (natural minor)"
    assert evidence["flat_2_confirmed"] is False


def test_dorian_and_mixolydian_still_detected():
    dorian = chroma({0: 1.0, 2: 0.8, 3: 1.0, 5: 0.9, 7: 1.0, 9: 1.0, 10: 1.0})
    assert detect_mode(dorian, 0) == "Dorian"

    mixo = chroma({0: 1.0, 2: 0.8, 4: 1.0, 5: 0.9, 7: 1.0, 9: 0.9, 10: 1.0})
    assert detect_mode(mixo, 0) == "Mixolydian"


def test_major_modes_unaffected():
    major = chroma({0: 1.0, 2: 0.8, 4: 1.0, 5: 0.9, 7: 1.0, 9: 0.9, 11: 1.0})
    assert detect_mode(major, 0) == "Ionian (natural major)"


# ---------------------------------------------------------------------------
# BUG 2 (v7.2): Essentia's "A:min" was parsed as a major chord
# ---------------------------------------------------------------------------
def test_essentia_colon_spelling_parsed():
    assert parse_chord("A:min") == ("A", "min")
    assert parse_chord("A:maj") == ("A", "maj")
    assert parse_chord("C#:min") == ("C#", "min")


def test_flat_roots_normalised():
    assert parse_chord("Bb") == ("A#", "maj")
    assert parse_chord("Eb:min") == ("D#", "min")


def test_no_chord_marker_returns_none():
    for token in ("N", "NC", "no_chord", "?", ""):
        assert parse_chord(token) == (None, None)


def test_minor_chord_gives_lowercase_numeral():
    """The v7.2 bug: 'A:min' produced an uppercase (major) numeral."""
    assert chord_to_roman("A:min", "A", "minor") == "i"
    assert chord_to_roman("Am", "A", "minor") == "i"
    # A minor is the vi of C major -- this is the case v7.2 got wrong.
    assert chord_to_roman("A:min", "C", "major") == "vi"
    assert chord_to_roman("Am", "C", "major") == "vi"
    # A minor is a borrowed tonic minor in A major.
    assert chord_to_roman("A:min", "A", "major") == "i"


def test_major_chord_gives_uppercase_numeral():
    assert chord_to_roman("A:maj", "A", "minor") == "I"
    assert chord_to_roman("A", "A", "minor") == "I"


def test_common_progression_in_a_minor():
    loop = [chord_to_roman(c, "A", "minor")
            for c in ("A:min", "F:maj", "C:maj", "G:maj")]
    assert loop == ["i", "VI", "III", "VII"]


def test_chromatic_flat_two_chord():
    assert chord_to_roman("Bb", "A", "minor") == "bII"


def test_unparseable_chord_returns_none():
    assert chord_to_roman("N", "A", "minor") is None


# ---------------------------------------------------------------------------
# Pentatonic evidence
# ---------------------------------------------------------------------------
def test_pentatonic_requires_a_tight_fit():
    assert detect_pentatonic(["1", "b3", "4", "5", "b7"]) == "minor"
    assert detect_pentatonic(["1", "2", "3", "5", "6"]) == "major"
    # Too many foreign degrees -> no claim.
    assert detect_pentatonic(["1", "b2", "b3", "3", "4", "b5", "5", "b7"]) is None
    assert detect_pentatonic([]) is None
    assert detect_pentatonic(None) is None
