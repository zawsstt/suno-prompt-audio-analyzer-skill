#!/usr/bin/env python3
"""
shot_builder.py -- turn aligned lyric lines into a time-locked shot list.

Pure logic only: no audio, no ML. Everything here is unit-testable.

Why a shot list instead of an LRC
---------------------------------
A video model handed "one audio file + a sheet of words with no timing" cannot
guess where the beat is. Handing it an LRC is barely better, because it will
reflow the timings. The contract is therefore:

    the timeline is LOCKED, the model may only fill in `visual`.

Shot rules (from the spec):
    * a line longer than 6s is split at the nearest beat
    * a line shorter than 1.5s is merged into the next shot
    * instrumental gaps become wordless shots
    * a line whose span does not sit on vocal energy is flagged for review
    * start times snap to the nearest beat, or to a downbeat when one is close
"""

MIN_SHOT_SEC = 1.5
MAX_SHOT_SEC = 6.0
DEFAULT_SNAP_SEC = 0.2
DEFAULT_DOWNBEAT_WINDOW = 0.25


# ---------------------------------------------------------------------------
# Beat grid
# ---------------------------------------------------------------------------
def snap_to_grid(t, beat_times, downbeat_times=None,
                 max_shift=DEFAULT_SNAP_SEC,
                 downbeat_window=DEFAULT_DOWNBEAT_WINDOW):
    """
    Move `t` onto the nearest beat.

    A downbeat is preferred when it is within `downbeat_window`, because a
    picture cut landing on the first beat of a bar reads as intentional.
    A move larger than `max_shift` is refused: a lyric head can be slightly
    early or late without looking wrong, but dragging it 600 ms will.

    Returns {"time": float, "shifted": bool, "shift": float, "target": "downbeat"|"beat"|None}
    """
    t = float(t)
    result = {"time": t, "shifted": False, "shift": 0.0, "target": None}

    if not beat_times:
        return result

    if downbeat_times:
        db = min(downbeat_times, key=lambda x: abs(x - t))
        if abs(db - t) <= downbeat_window:
            result.update({"time": round(db, 3), "shifted": True,
                           "shift": round(db - t, 4), "target": "downbeat"})
            return result

    b = min(beat_times, key=lambda x: abs(x - t))
    if abs(b - t) <= max_shift:
        result.update({"time": round(b, 3), "shifted": True,
                       "shift": round(b - t, 4), "target": "beat"})
    return result


def _split_long(start, end, beat_times, max_shot=MAX_SHOT_SEC):
    """Split [start, end] at the beat nearest each midpoint until all fit."""
    if end - start <= max_shot:
        return [(start, end)]

    mid = (start + end) / 2.0
    candidates = [b for b in (beat_times or []) if start + 0.3 < b < end - 0.3]
    cut = min(candidates, key=lambda x: abs(x - mid)) if candidates else mid

    if cut <= start or cut >= end:
        return [(start, end)]

    return (_split_long(start, cut, beat_times, max_shot)
            + _split_long(cut, end, beat_times, max_shot))


def _in_vocal(t, vocal_regions):
    if not vocal_regions:
        return True  # no information: do not punish the line
    for seg in vocal_regions:
        if seg["start"] <= t <= seg["end"]:
            return True
    return False


def build_energy_lookup(energy_curve):
    """
    Build a callable mapping a timestamp to 'low' | 'mid' | 'high'.

    Percentile based, so it works for a quiet lo-fi loop and a loud orchestral
    track without hard-coded dB thresholds.
    """
    points = []
    for entry in energy_curve or []:
        t = entry.get("time_sec")
        if t is None:
            t = _parse_mmss(entry.get("time"))
        db = entry.get("rms_db")
        if t is None or db is None:
            continue
        points.append((float(t), float(db)))

    if not points:
        return lambda _t: "mid"

    points.sort()
    vals = sorted(p[1] for p in points)
    lo = vals[int(len(vals) * 0.33)]
    hi = vals[int(len(vals) * 0.66)]

    def lookup(t):
        # nearest sample in time
        nearest = min(points, key=lambda p: abs(p[0] - t))
        db = nearest[1]
        if db <= lo:
            return "low"
        if db >= hi:
            return "high"
        return "mid"

    return lookup


def _parse_mmss(s):
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    try:
        m, sec = s.split(":", 1)
        return int(m) * 60 + float(sec)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shot construction
# ---------------------------------------------------------------------------
def build_shots(aligned_lines,
                beat_times=None,
                downbeat_times=None,
                vocal_regions=None,
                total_duration=None,
                energy_lookup=None,
                min_shot=MIN_SHOT_SEC,
                max_shot=MAX_SHOT_SEC,
                snap_sec=DEFAULT_SNAP_SEC,
                low_confidence_threshold=0.5):
    """
    Build the locked shot list.

    `aligned_lines`: [{"section": str|None, "text": str,
                       "start": float|None, "end": float|None,
                       "score": float|None}]
    `vocal_regions`: [{"start": float, "end": float}]
    `energy_lookup`: callable(t) -> "low"|"mid"|"high"

    Returns a list of shot dicts, sorted and non-overlapping.
    """
    beat_times = sorted(beat_times or [])
    downbeat_times = sorted(downbeat_times or [])
    energy_lookup = energy_lookup or (lambda _t: "mid")

    usable = []
    for line in aligned_lines or []:
        if line.get("start") is None or line.get("end") is None:
            continue
        if line["end"] <= line["start"]:
            continue
        usable.append(dict(line))

    if not usable:
        return []

    # ---- 1. snap each start to the grid ------------------------------------
    for line in usable:
        snap = snap_to_grid(line["start"], beat_times, downbeat_times, max_shift=snap_sec)
        line["_raw_start"] = line["start"]
        line["start"] = snap["time"]
        line["snap_shift_sec"] = snap["shift"]
        line["snapped_to"] = snap["target"]

    # ---- 2. merge lines shorter than min_shot into the next shot -----------
    merged = []
    i = 0
    while i < len(usable):
        cur = usable[i]
        if (cur["end"] - cur["start"]) < min_shot and i + 1 < len(usable):
            nxt = usable[i + 1]
            nxt["start"] = min(nxt["start"], cur["start"])
            nxt["section"] = nxt.get("section") or cur.get("section")
            nxt.setdefault("merged_from", []).append(cur.get("text"))
            i += 1
            continue
        if (cur["end"] - cur["start"]) < min_shot and merged:
            prev = merged[-1]
            prev["end"] = max(prev["end"], cur["end"])
            prev.setdefault("merged_from", []).append(cur.get("text"))
            i += 1
            continue
        merged.append(cur)
        i += 1

    # ---- 3. split lines longer than max_shot ------------------------------
    pieces = []
    for line in merged:
        for (s, e) in _split_long(line["start"], line["end"], beat_times, max_shot):
            piece = dict(line)
            piece["start"], piece["end"] = round(s, 3), round(e, 3)
            pieces.append(piece)

    pieces.sort(key=lambda x: x["start"])

    # ---- 4. enforce monotonic, non-overlapping ----------------------------
    for idx in range(1, len(pieces)):
        if pieces[idx]["start"] < pieces[idx - 1]["end"]:
            pieces[idx]["start"] = pieces[idx - 1]["end"]
        if pieces[idx]["end"] <= pieces[idx]["start"]:
            pieces[idx]["end"] = pieces[idx]["start"] + 0.1

    # ---- 5. insert instrumental shots for the gaps ------------------------
    timeline = []
    cursor = 0.0
    for piece in pieces:
        if piece["start"] - cursor >= min_shot:
            timeline.append(_instrumental_shot(cursor, piece["start"], energy_lookup))
        timeline.append(piece)
        cursor = max(cursor, piece["end"])

    if total_duration and (total_duration - cursor) >= min_shot:
        timeline.append(_instrumental_shot(cursor, total_duration, energy_lookup))

    # ---- 6. finalise ------------------------------------------------------
    shots = []
    for item in timeline:
        if item.get("_instrumental"):
            shots.append(item["shot"])
            continue
        mid = (item["start"] + item["end"]) / 2.0
        score = item.get("score")
        needs_review = False
        reasons = []
        if score is not None and score < low_confidence_threshold:
            needs_review = True
            reasons.append(f"low alignment score ({score:.2f})")
        if not _in_vocal(mid, vocal_regions):
            needs_review = True
            reasons.append("no vocal energy across this line")
        if abs(item.get("snap_shift_sec") or 0.0) > snap_sec:
            needs_review = True
            reasons.append("large snap shift")

        shots.append({
            "start": round(float(item["start"]), 3),
            "end": round(float(item["end"]), 3),
            "section": item.get("section"),
            "lyric": item.get("text"),
            "energy": energy_lookup(mid),
            "visual": "",
            "needs_review": needs_review,
            "review_reasons": reasons,
            "alignment_score": round(score, 3) if isinstance(score, (int, float)) else None,
            "snap_shift_sec": item.get("snap_shift_sec"),
            "snapped_to": item.get("snapped_to"),
        })

    shots.sort(key=lambda s: s["start"])
    return shots


def _instrumental_shot(start, end, energy_lookup):
    mid = (start + end) / 2.0
    return {
        "_instrumental": True,
        "shot": {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "section": "Instrumental",
            "lyric": None,
            "energy": energy_lookup(mid),
            "visual": "",
            "needs_review": False,
            "review_reasons": [],
            "alignment_score": None,
            "snap_shift_sec": None,
            "snapped_to": None,
        },
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _fmt_lrc_time(t):
    t = max(0.0, float(t))
    m = int(t // 60)
    s = t - m * 60
    return f"[{m:02d}:{s:05.2f}]"


def _fmt_srt_time(t):
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_lrc(shots):
    """One LRC line per lyric shot. Instrumental shots are not emitted."""
    lines = []
    for shot in shots:
        if not shot.get("lyric"):
            continue
        lines.append(f"{_fmt_lrc_time(shot['start'])}{shot['lyric']}")
    return "\n".join(lines) + ("\n" if lines else "")


def to_srt(shots):
    """SRT for subtitles. Uses the locked shot timings."""
    blocks = []
    index = 0
    for shot in shots:
        if not shot.get("lyric"):
            continue
        index += 1
        blocks.append(
            f"{index}\n{_fmt_srt_time(shot['start'])} --> {_fmt_srt_time(shot['end'])}\n"
            f"{shot['lyric']}\n"
        )
    return "\n".join(blocks)


SHOT_LIST_CONTRACT = (
    "The timeline is locked. Only the `visual` field may be edited. "
    "`start`, `end` and `lyric` MUST NOT be changed."
)
