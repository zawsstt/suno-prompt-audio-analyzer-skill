"""Beat snapping, shot merging/splitting, and export formats."""

from shot_builder import (
    build_shots,
    snap_to_grid,
    to_lrc,
    to_srt,
)


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------
def test_snap_to_nearest_beat_within_tolerance():
    beats = [0.0, 1.0, 2.0, 3.0, 4.0]
    result = snap_to_grid(2.15, beats)
    assert result["time"] == 2.0
    assert result["shifted"] is True
    assert result["target"] == "beat"


def test_no_snap_beyond_the_tolerance():
    """A 500 ms drag would look wrong even if it lands on a beat."""
    beats = [0.0, 1.0, 2.0, 3.0, 4.0]
    result = snap_to_grid(2.5, beats, max_shift=0.2)
    assert result["shifted"] is False
    assert result["time"] == 2.5


def test_downbeat_is_preferred_when_close():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    downbeats = [0.0, 4.0]
    result = snap_to_grid(3.9, beats, downbeats)
    assert result["target"] == "downbeat"
    assert result["time"] == 4.0


def test_beat_used_when_no_downbeat_is_near():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    downbeats = [0.0, 8.0]
    result = snap_to_grid(2.1, beats, downbeats)
    assert result["target"] == "beat"


def test_snap_without_a_grid_is_a_noop():
    result = snap_to_grid(5.0, [], [])
    assert result["shifted"] is False
    assert result["time"] == 5.0


# ---------------------------------------------------------------------------
# Shot construction
# ---------------------------------------------------------------------------
def test_short_line_merges_into_the_next_shot():
    lines = [
        {"section": "Verse", "text": "short", "start": 0.0, "end": 0.8, "score": 0.9},
        {"section": "Verse", "text": "the following line", "start": 1.0, "end": 3.0, "score": 0.9},
    ]
    shots = build_shots(lines, min_shot=1.5, max_shot=6.0)
    assert len(shots) == 1
    assert shots[0]["start"] == 0.0
    assert shots[0]["lyric"] == "the following line"


def test_long_line_is_split_at_a_beat():
    beats = [i * 0.5 for i in range(0, 28)]
    lines = [{"section": "Chorus", "text": "a very long line", "start": 0.0,
              "end": 13.0, "score": 0.9}]
    shots = build_shots(lines, beat_times=beats, min_shot=1.5, max_shot=6.0)

    assert len(shots) > 1
    for shot in shots:
        assert shot["end"] - shot["start"] <= 6.0 + 1e-6
    assert shots[0]["start"] == 0.0
    assert shots[-1]["end"] == 13.0


def test_instrumental_gaps_become_wordless_shots():
    lines = [{"section": "Verse", "text": "only line", "start": 10.0,
              "end": 12.0, "score": 0.9}]
    shots = build_shots(lines, total_duration=20.0, min_shot=1.5, max_shot=6.0)

    assert len(shots) == 3
    assert shots[0]["lyric"] is None
    assert shots[0]["section"] == "Instrumental"
    assert shots[-1]["lyric"] is None
    assert shots[1]["lyric"] == "only line"


def test_low_alignment_score_is_flagged_not_silently_accepted():
    lines = [{"section": "Verse", "text": "uncertain", "start": 1.0,
              "end": 4.0, "score": 0.2}]
    shots = build_shots(lines, min_shot=1.5, max_shot=6.0)
    assert shots[0]["needs_review"] is True
    assert any("alignment score" in r for r in shots[0]["review_reasons"])


def test_line_aligned_onto_silence_is_flagged():
    lines = [{"section": "Verse", "text": "misplaced", "start": 10.0,
              "end": 12.0, "score": 0.95}]
    regions = [{"start": 0.0, "end": 5.0}, {"start": 15.0, "end": 20.0}]
    shots = build_shots(lines, vocal_regions=regions, min_shot=1.5, max_shot=6.0)
    lyric_shots = [s for s in shots if s["lyric"]]
    assert lyric_shots[0]["needs_review"] is True
    assert any("vocal energy" in r for r in lyric_shots[0]["review_reasons"])


def test_confident_line_is_not_flagged():
    lines = [{"section": "Verse", "text": "solid", "start": 1.0,
              "end": 4.0, "score": 0.92}]
    regions = [{"start": 0.0, "end": 20.0}]
    shots = build_shots(lines, vocal_regions=regions, min_shot=1.5, max_shot=6.0)
    assert shots[0]["needs_review"] is False


def test_shots_are_sorted_and_non_overlapping():
    lines = [
        {"section": "A", "text": "one", "start": 5.0, "end": 8.0, "score": 0.9},
        {"section": "A", "text": "two", "start": 1.0, "end": 4.0, "score": 0.9},
        {"section": "A", "text": "three", "start": 9.0, "end": 12.0, "score": 0.9},
    ]
    shots = build_shots(lines, min_shot=1.5, max_shot=6.0)
    for i in range(1, len(shots)):
        assert shots[i]["start"] >= shots[i - 1]["end"] - 1e-9


def test_no_shots_without_timings():
    lines = [{"section": "A", "text": "no timing", "start": None, "end": None, "score": None}]
    assert build_shots(lines) == []


def test_energy_label_comes_from_the_curve():
    curve = [{"time_sec": float(i * 5), "rms_db": -30 + i} for i in range(12)]
    lines = [
        {"section": "A", "text": "quiet part", "start": 0.0, "end": 3.0, "score": 0.9},
        {"section": "B", "text": "loud part", "start": 50.0, "end": 53.0, "score": 0.9},
    ]
    from shot_builder import build_energy_lookup
    shots = build_shots(lines, energy_lookup=build_energy_lookup(curve),
                        min_shot=1.5, max_shot=6.0)
    assert shots[0]["energy"] == "low"
    assert shots[-1]["energy"] == "high"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def test_lrc_and_srt_formats():
    shots = [
        {"start": 32.10, "end": 36.40, "section": "Chorus", "lyric": "line one",
         "energy": "high", "visual": "", "needs_review": False},
        {"start": 36.40, "end": 40.00, "section": "Instrumental", "lyric": None,
         "energy": "mid", "visual": "", "needs_review": False},
        {"start": 40.00, "end": 43.00, "section": "Chorus", "lyric": "line two",
         "energy": "high", "visual": "", "needs_review": False},
    ]
    lrc = to_lrc(shots)
    srt = to_srt(shots)

    assert "[00:32.10]line one" in lrc
    assert "[00:40.00]line two" in lrc
    # instrumental shots carry no subtitle line
    assert "Instrumental" not in lrc
    assert lrc.count("\n") == 2

    assert "00:00:32,100 --> 00:00:36,400" in srt
    assert "line one" in srt
    # SRT numbering skips the instrumental shot
    assert "\n2\n00:00:40,000" in srt


def test_shot_list_contract_is_declared():
    from shot_builder import SHOT_LIST_CONTRACT
    assert "locked" in SHOT_LIST_CONTRACT.lower()
    assert "visual" in SHOT_LIST_CONTRACT
