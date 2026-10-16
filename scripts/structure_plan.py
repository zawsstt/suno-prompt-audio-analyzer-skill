#!/usr/bin/env python3
"""
structure_plan.py -- describe song structure from measured energy, nothing else.

v7.2 problem
------------
`generate_instrumental_timeline()` documented itself as using the energy curve,
but the body was a hard-coded Hollywood arc:

    [Intro] [Build-up] [Drop] [Verse] [Bridge] [Climax] [Outro]

Only the genre name and the BPM were substituted. A lo-fi loop and a Zimmer
score received the identical seven-part arc, and `energy_drop_times_sec`, which
the analyzer had already measured, was thrown away.

v7.3 contract
-------------
A section label is only emitted when the energy curve supports it:
    * "Drop" requires a measured drop
    * "Climax" requires a peak that is clearly above the track median
    * "Intro" / "Outro" require a low-energy head / tail
When there is no evidence we emit a single neutral line instead of inventing
a structure. Invented structure is worse than no structure: Suno will obey it.
"""

# How far above the median a peak must sit to be called a climax (in dB).
_CLIMAX_MARGIN_DB = 3.0
# How far below the median a head/tail must sit to be called intro/outro.
_QUIET_MARGIN_DB = 2.0
# Minimum segment length we are willing to name.
_MIN_SEGMENT_SEC = 6.0


def _curve_points(analysis):
    """Normalise the 5-second energy curve into [(t, rms_db), ...]."""
    points = []
    for entry in (analysis or {}).get("energy_curve") or []:
        t = entry.get("time_sec")
        if t is None:
            stamp = entry.get("time")
            if isinstance(stamp, str) and ":" in stamp:
                try:
                    m, s = stamp.split(":", 1)
                    t = int(m) * 60 + float(s)
                except Exception:
                    t = None
        db = entry.get("rms_db")
        if t is None or db is None:
            continue
        points.append((float(t), float(db)))
    points.sort()
    return points


def _median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def energy_shape(analysis):
    """
    Summarise the measured energy shape. Pure evidence, no labels yet.

    Returns:
        {
          "available": bool,
          "median_db": float | None,
          "duration_sec": float | None,
          "drops": [float, ...],
          "peaks": [float, ...],
          "has_drop": bool,
          "has_climax": bool,
          "starts_quiet": bool,
          "ends_quiet": bool,
          "reason": str,
        }
    """
    structure = (analysis or {}).get("structure") or {}
    drops = [float(x) for x in (structure.get("energy_drop_times_sec") or [])]
    peaks = [float(x) for x in (structure.get("energy_peak_times_sec") or [])]

    points = _curve_points(analysis)
    duration = (analysis.get("file_info") or {}).get("duration_sec")

    if not points:
        return {
            "available": False,
            "median_db": None,
            "duration_sec": duration,
            "drops": drops,
            "peaks": peaks,
            "has_drop": False,
            "has_climax": False,
            "starts_quiet": False,
            "ends_quiet": False,
            "reason": "No energy curve in the analysis; structure cannot be inferred.",
        }

    median_db = _median([p[1] for p in points])
    peak_db = max(p[1] for p in points)
    head_db = _median([p[1] for p in points[:max(1, len(points) // 8)]])
    tail_db = _median([p[1] for p in points[-max(1, len(points) // 8):]])

    has_drop = bool(drops)
    has_climax = peak_db is not None and median_db is not None and \
        (peak_db - median_db) >= _CLIMAX_MARGIN_DB

    return {
        "available": True,
        "median_db": round(median_db, 2) if median_db is not None else None,
        "peak_db": round(peak_db, 2),
        "duration_sec": duration,
        "drops": drops,
        "peaks": peaks,
        "has_drop": has_drop,
        "has_climax": has_climax,
        "starts_quiet": head_db is not None and median_db is not None and
        head_db <= median_db - _QUIET_MARGIN_DB,
        "ends_quiet": tail_db is not None and median_db is not None and
        tail_db <= median_db - _QUIET_MARGIN_DB,
        "reason": "Derived from the measured 5-second RMS curve.",
    }


def _segment_label(seg, shape, index, total):
    """Name one segment, but only from evidence."""
    db = seg["mean_db"]
    median = shape.get("median_db")
    if median is None:
        return "Section"

    if seg.get("contains_drop") and db >= median:
        return "Drop"
    if seg.get("contains_peak") and shape.get("has_climax"):
        return "Climax"
    if index == 0 and shape.get("starts_quiet"):
        return "Intro"
    if index == total - 1 and shape.get("ends_quiet"):
        return "Outro"
    if db < median - 1.0:
        return "Breakdown"
    if seg.get("rising"):
        return "Build"
    return "Section"


def _descriptor(seg, shape):
    median = shape.get("median_db") or 0.0
    delta = seg["mean_db"] - median
    if delta <= -2.0:
        return "stripped back, low energy"
    if delta >= 2.0:
        return "full arrangement, high energy"
    return "steady mid-level energy"


def build_instrumental_timeline(analysis, genre_label=None, max_sections=6):
    """
    Build Suno Lyrics-box scaffolding for an instrumental track.

    Returns a string. When the analysis carries no usable energy evidence the
    result is a single neutral line -- never a fabricated arc.
    """
    shape = energy_shape(analysis)
    bpm = (analysis or {}).get("summary", {}).get("tempo_bpm")
    genre = genre_label
    if not genre:
        genres = (analysis or {}).get("production_style", {}).get("likely_genres") or []
        genre = genres[0] if genres else None

    head_bits = [b for b in (genre, f"{round(bpm)} BPM" if bpm else None) if b]
    head = ", ".join(head_bits)

    if not shape["available"]:
        return (f"[Instrumental: {head}]\n" if head else "[Instrumental]\n") + \
            "[No measured energy curve -- arrangement left to the model]"

    points = _curve_points(analysis)
    duration = shape.get("duration_sec") or (points[-1][0] if points else 0)

    # Boundaries = drops and peaks, plus the track ends.
    boundaries = sorted(set([0.0] + shape["drops"] + shape["peaks"] + [float(duration)]))
    boundaries = [b for b in boundaries if 0 <= b <= float(duration) + 1e-6]

    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < _MIN_SEGMENT_SEC:
            continue
        window = [p for p in points if start <= p[0] <= end]
        if not window:
            continue
        mean_db = sum(p[1] for p in window) / len(window)
        half = len(window) // 2 or 1
        rising = (sum(p[1] for p in window[half:]) / max(len(window) - half, 1)) > \
                 (sum(p[1] for p in window[:half]) / half)
        segments.append({
            "start": start,
            "end": end,
            "mean_db": mean_db,
            "rising": rising,
            "contains_drop": any(start <= d <= end for d in shape["drops"]),
            "contains_peak": any(start <= p <= end for p in shape["peaks"]),
        })

    if not segments:
        return (f"[Instrumental: {head}]\n" if head else "[Instrumental]\n") + \
            "[Energy is essentially flat -- no section breaks detected]"

    if len(segments) > max_sections:
        # Keep the most dynamic segments, preserving order.
        ranked = sorted(segments, key=lambda s: abs(s["mean_db"] - (shape["median_db"] or 0)),
                        reverse=True)[:max_sections]
        segments = sorted(ranked, key=lambda s: s["start"])

    lines = []
    for idx, seg in enumerate(segments):
        label = _segment_label(seg, shape, idx, len(segments))
        desc = _descriptor(seg, shape)
        stamp = f"{int(seg['start'] // 60):02d}:{int(seg['start'] % 60):02d}"
        bits = [desc]
        if head and idx == 0:
            bits.insert(0, head)
        lines.append(f"[{label} @ {stamp}: {', '.join(bits)}]")

    return "\n".join(lines)


def structure_summary(analysis):
    """Short human-readable evidence summary, for reports and provenance."""
    shape = energy_shape(analysis)
    return {
        "has_drop": shape["has_drop"],
        "has_climax": shape["has_climax"],
        "starts_quiet": shape["starts_quiet"],
        "ends_quiet": shape["ends_quiet"],
        "drops_sec": shape["drops"][:10],
        "peaks_sec": shape["peaks"][:10],
        "median_rms_db": shape.get("median_db"),
        "reason": shape["reason"],
    }
