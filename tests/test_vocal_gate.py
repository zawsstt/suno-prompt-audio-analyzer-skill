"""The vocal gate must never be able to silently close the vocal branch."""

from vocal_gate import (
    PRESENCE_INSTRUMENTAL,
    PRESENCE_UNKNOWN,
    PRESENCE_VOCAL,
    decide_vocal_presence,
)


def test_missing_features_escalate_to_unknown_not_instrumental():
    """
    The v7.2 failure mode: an absent measurement was treated as "instrumental",
    which skipped Whisper forever. Absent must mean unknown.
    """
    result = decide_vocal_presence({})
    assert result["presence"] == PRESENCE_UNKNOWN
    assert result["decided_by"] == "insufficient_evidence"
    assert result["score"] is None


def test_insufficient_audio_escalates_to_unknown():
    result = decide_vocal_presence({"insufficient_audio": True})
    assert result["presence"] == PRESENCE_UNKNOWN


def test_no_vocal_score_is_ever_read_from_lyrics():
    """A 'has_lyrics' key must have no effect on the decision."""
    with_lyrics = decide_vocal_presence({
        "voiced_frame_fraction": 0.05,
        "band_tonality": 0.15,
        "vocal_band_ratio": 0.12,
        "has_lyrics": False,
    })
    without = decide_vocal_presence({
        "voiced_frame_fraction": 0.05,
        "band_tonality": 0.15,
        "vocal_band_ratio": 0.12,
    })
    assert with_lyrics["presence"] == without["presence"] == PRESENCE_INSTRUMENTAL


def test_clear_vocal_profile():
    result = decide_vocal_presence({
        "voiced_frame_fraction": 0.72,
        "band_tonality": 0.90,
        "vocal_band_ratio": 0.55,
    })
    assert result["presence"] == PRESENCE_VOCAL
    assert result["score"] >= 0.60


def test_clear_instrumental_profile():
    result = decide_vocal_presence({
        "voiced_frame_fraction": 0.05,
        "band_tonality": 0.18,
        "vocal_band_ratio": 0.14,
    })
    assert result["presence"] == PRESENCE_INSTRUMENTAL
    assert result["score"] <= 0.35


def test_ambiguous_profile_is_unknown_so_whisper_runs():
    result = decide_vocal_presence({
        "voiced_frame_fraction": 0.40,
        "band_tonality": 0.50,
        "vocal_band_ratio": 0.40,
    })
    assert result["presence"] == PRESENCE_UNKNOWN
    assert "ambiguous" in result["reason"].lower()


def test_vad_high_speech_ratio_wins():
    result = decide_vocal_presence({
        "vad_speech_ratio": 0.70,
        "voiced_frame_fraction": 0.30,
        "band_tonality": 0.40,
        "vocal_band_ratio": 0.35,
    })
    assert result["presence"] == PRESENCE_VOCAL
    assert result["decided_by"] == "vad"


def test_vad_low_ratio_alone_does_not_claim_instrumental():
    """
    VAD frequently misses sustained singing. A low speech ratio with a
    vocal-ish acoustic profile must stay unknown, not become instrumental.
    """
    result = decide_vocal_presence({
        "vad_speech_ratio": 0.04,
        "voiced_frame_fraction": 0.65,
        "band_tonality": 0.85,
        "vocal_band_ratio": 0.50,
    })
    assert result["presence"] == PRESENCE_UNKNOWN
    assert result["decided_by"] == "vad"


def test_vad_low_ratio_with_dead_acoustics_is_instrumental():
    result = decide_vocal_presence({
        "vad_speech_ratio": 0.0,
        "voiced_frame_fraction": 0.04,
        "band_tonality": 0.10,
        "vocal_band_ratio": 0.10,
    })
    assert result["presence"] == PRESENCE_INSTRUMENTAL


def test_evidence_block_is_always_present():
    for payload in ({}, {"insufficient_audio": True},
                    {"voiced_frame_fraction": 0.7, "band_tonality": 0.9}):
        result = decide_vocal_presence(payload)
        assert "evidence" in result
        assert "reason" in result
        assert result["presence"] in {PRESENCE_VOCAL, PRESENCE_INSTRUMENTAL, PRESENCE_UNKNOWN}
