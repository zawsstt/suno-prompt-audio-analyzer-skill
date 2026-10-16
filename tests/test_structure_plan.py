"""Structure labels must come from measured energy, never from a fixed arc."""

from structure_plan import build_instrumental_timeline, energy_shape, structure_summary


def make_analysis(curve_db, drops=None, peaks=None, bpm=120, genres=None):
    return {
        "energy_curve": [
            {"time": f"{int(i * 5 // 60):02d}:{int(i * 5 % 60):02d}",
             "time_sec": float(i * 5),
             "rms_db": float(db)}
            for i, db in enumerate(curve_db)
        ],
        "structure": {
            "energy_drop_times_sec": drops or [],
            "energy_peak_times_sec": peaks or [],
        },
        "summary": {"tempo_bpm": bpm},
        "production_style": {"likely_genres": genres or ["Cinematic / Orchestral"]},
        "file_info": {"duration_sec": len(curve_db) * 5.0},
    }


DYNAMIC_CURVE = (
    [-30, -30, -30]              # quiet head
    + [-24, -24, -24, -24, -24]  # build
    + [-30]                      # drop
    + [-18] * 7                  # full section
    + [-14] * 5                  # climax
    + [-26, -26, -26]            # quiet tail
)


def test_no_energy_curve_means_no_invented_structure():
    """
    v7.2 always emitted Intro/Build/Drop/Verse/Bridge/Climax/Outro regardless of
    the music. With no evidence we must emit one neutral line instead.
    """
    analysis = {"summary": {"tempo_bpm": 120},
                "production_style": {"likely_genres": ["Cinematic / Orchestral"]}}
    timeline = build_instrumental_timeline(analysis)

    for invented in ("Drop", "Climax", "Build", "Bridge"):
        assert invented not in timeline
    assert "No measured energy curve" in timeline


def test_flat_track_gets_a_single_neutral_section():
    analysis = make_analysis([-20] * 24)
    timeline = build_instrumental_timeline(analysis)
    lines = [l for l in timeline.splitlines() if l.strip()]

    assert len(lines) == 1
    for invented in ("Drop", "Climax", "Intro", "Outro", "Build"):
        assert invented not in timeline


def test_drop_is_named_only_when_measured():
    analysis = make_analysis(DYNAMIC_CURVE, drops=[40.0], peaks=[80.0])
    timeline = build_instrumental_timeline(analysis)
    assert "Drop" in timeline
    assert "Climax" in timeline


def test_no_drop_evidence_means_no_drop_label():
    analysis = make_analysis(DYNAMIC_CURVE, drops=[], peaks=[80.0])
    timeline = build_instrumental_timeline(analysis)
    assert "Drop" not in timeline


def test_climax_requires_a_clear_peak():
    """A flat-ish track with one tiny bump must not be called a climax."""
    gentle = [-22] * 12 + [-20] * 10 + [-22] * 2
    analysis = make_analysis(gentle, drops=[], peaks=[60.0])
    timeline = build_instrumental_timeline(analysis)
    assert "Climax" not in timeline


def test_lofi_loop_and_cinematic_score_differ():
    lofi = build_instrumental_timeline(
        make_analysis([-20] * 24, genres=["Lo-fi / Boom-bap"]), genre_label="lo-fi hip-hop")
    cinematic = build_instrumental_timeline(
        make_analysis(DYNAMIC_CURVE, drops=[40.0], peaks=[80.0]),
        genre_label="cinematic orchestral")

    assert lofi != cinematic
    assert "Drop" not in lofi
    assert "Drop" in cinematic


def test_section_count_is_bounded():
    drops = [float(i * 10) for i in range(1, 20)]
    peaks = [float(i * 10 + 5) for i in range(1, 20)]
    analysis = make_analysis([-20 + (i % 7) for i in range(40)], drops=drops, peaks=peaks)
    timeline = build_instrumental_timeline(analysis, max_sections=4)
    assert len([l for l in timeline.splitlines() if l.strip()]) <= 4


def test_energy_shape_reports_evidence():
    shape = energy_shape(make_analysis(DYNAMIC_CURVE, drops=[40.0], peaks=[80.0]))
    assert shape["available"] is True
    assert shape["has_drop"] is True
    assert shape["has_climax"] is True
    assert shape["starts_quiet"] is True
    assert shape["ends_quiet"] is True
    assert shape["median_db"] is not None


def test_structure_summary_exposes_reasoning():
    summary = structure_summary(make_analysis(DYNAMIC_CURVE, drops=[40.0]))
    assert summary["has_drop"] is True
    assert summary["drops_sec"] == [40.0]
    assert "reason" in summary
