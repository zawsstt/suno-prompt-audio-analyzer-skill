#!/usr/bin/env python3
"""
music_theory.py -- pure harmony helpers shared by the analyzer and the MIDI
parser.

Two v7.2 bugs lived in code that only the analyzer had:

1. `detect_mode()` returned "Phrygian" whenever b6 and b7 were strong. Natural
   minor IS b6 + b7, so ordinary minor songs were labelled Phrygian and the
   compiler then wrote "phrygian tension" into the Suno prompt. The whitelist
   itself said the tag requires "minor with flat 2nd"; the code never checked.

2. `chord_to_roman()` only recognised "Am". Essentia emits "A:min", which was
   parsed as a MAJOR chord, skewing the whole Roman-numeral block toward major.

Both now live here, with the evidence check made explicit, so the analyzer and
the MIDI parser cannot drift apart.
"""

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E",
}

# A flat 2nd must reach this normalised level AND beat the natural 2nd before
# we are willing to call a minor key Phrygian.
B2_EVIDENCE_MIN = 0.35
B2_OVER_NATURAL2 = 1.10


def parse_chord(chord_name):
    """
    Parse a chord symbol into (root, quality).

    Handles Essentia's colon spelling and flats:
        "A:min" -> ("A", "min")     "A:maj" -> ("A", "maj")
        "Am"    -> ("A", "min")     "C"     -> ("C", "maj")
        "Bb"    -> ("A#", "maj")    "N"     -> (None, None)
    """
    import re

    if not chord_name:
        return None, None

    s = str(chord_name).strip()
    if not s or s.upper() in {"N", "NC", "NO_CHORD", "X", "?"}:
        return None, None

    if ":" in s:
        root_part, _, qual_part = s.partition(":")
        root = root_part.strip()
        qual = qual_part.strip().lower()
    else:
        m = re.match(r"^([A-Ga-g])([#b]?)(.*)$", s)
        if not m:
            return None, None
        root = m.group(1).upper() + m.group(2)
        rest = m.group(3).strip()
        low = rest.lower()
        if low.startswith("maj") or rest.startswith("M"):
            qual = "maj"
        elif low.startswith("min") or low.startswith("m"):
            qual = "min"
        elif low.startswith("dim") or low in {"o", "\u00b0"}:
            qual = "dim"
        elif low.startswith("aug") or low in {"+", "#5"}:
            qual = "aug"
        elif low.startswith("sus"):
            qual = "sus"
        elif low == "":
            qual = "maj"
        else:
            qual = low

    if len(root) >= 2:
        key = root[0].upper() + root[1]
        root = _FLAT_TO_SHARP.get(key, key)
    elif root:
        root = root[0].upper()

    if root not in KEY_NAMES:
        return None, None

    if qual.startswith("min") or qual == "m":
        quality = "min"
    elif qual.startswith("dim"):
        quality = "dim"
    elif qual.startswith("aug"):
        quality = "aug"
    elif qual.startswith("sus"):
        quality = "sus"
    else:
        quality = "maj"

    return root, quality


def chord_to_roman(chord_name, root_key, root_scale):
    """Convert a chord symbol to a Roman numeral relative to a key."""
    chord_root, quality = parse_chord(chord_name)
    if chord_root is None:
        return None

    root_idx = KEY_NAMES.index(root_key) if root_key in KEY_NAMES else 0
    chord_idx = KEY_NAMES.index(chord_root)
    interval = (chord_idx - root_idx) % 12

    if root_scale == "minor":
        diatonic = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}
        chromatic = {1: "bII", 4: "III", 6: "bV", 9: "VI"}
    else:
        diatonic = {0: "I", 2: "II", 4: "III", 5: "IV", 7: "V", 9: "VI", 11: "VII"}
        chromatic = {1: "bII", 3: "bIII", 6: "bV", 8: "bVI", 10: "bVII"}

    degree = diatonic.get(interval) or chromatic.get(interval) or f"?{interval}"

    is_minor_chord = quality in {"min", "dim"}
    if is_minor_chord and degree == degree.upper():
        degree = degree.lower()
    elif not is_minor_chord and degree == degree.lower() and degree.isalpha():
        degree = degree.upper()

    suffix = ""
    if quality == "dim":
        suffix = "o"
    elif quality == "aug":
        suffix = "+"

    return degree + suffix


def detect_mode_detailed(chroma_mean, root_idx):
    """
    Modal flavour WITH the evidence that justified it.

    Phrygian requires a measured flat 2nd that is present and stronger than the
    natural 2nd. Without that measurement the answer is Aeolian, not Phrygian.
    """
    import numpy as np

    d = np.roll(chroma_mean, -root_idx)
    d = d / (d.max() + 1e-9)

    b2, nat2 = float(d[1]), float(d[2])
    m3, M3 = float(d[3]), float(d[4])
    m6, M6 = float(d[8]), float(d[9])
    m7, M7 = float(d[10]), float(d[11])

    evidence = {
        "b2": round(b2, 4), "natural_2": round(nat2, 4),
        "b3": round(m3, 4), "natural_3": round(M3, 4),
        "b6": round(m6, 4), "natural_6": round(M6, 4),
        "b7": round(m7, 4), "natural_7": round(M7, 4),
    }

    if M3 > m3:
        mode = "Ionian (natural major)" if M7 > m7 else "Mixolydian"
        evidence["decision"] = "major 3rd present"
        return mode, evidence

    has_flat_2 = b2 >= B2_EVIDENCE_MIN and b2 > nat2 * B2_OVER_NATURAL2
    evidence["flat_2_confirmed"] = bool(has_flat_2)

    if m6 > M6 and m7 > M7:
        if has_flat_2:
            evidence["decision"] = f"b6+b7 with confirmed b2 ({b2:.2f} vs {nat2:.2f})"
            return "Phrygian", evidence
        evidence["decision"] = (f"b6+b7 but no flat 2nd "
                                f"(b2={b2:.2f} <= natural 2nd={nat2:.2f}) -> natural minor")
        return "Aeolian (natural minor)", evidence

    if M6 > m6:
        evidence["decision"] = "natural 6th present -> dorian"
        return "Dorian", evidence

    evidence["decision"] = "minor 3rd, no distinguishing 6th/7th"
    return "Aeolian (natural minor)", evidence


def detect_mode(chroma_mean, root_idx):
    """Backwards-compatible wrapper returning just the mode string."""
    return detect_mode_detailed(chroma_mean, root_idx)[0]


def detect_pentatonic(strong_degrees):
    """
    Infer a pentatonic set from strong scale degrees, or return None.

    Requires a tight fit: at least 4 of the 5 pentatonic degrees and no more
    than one foreign degree.
    """
    s = set(strong_degrees or [])
    minor_pent = {"1", "b3", "4", "5", "b7"}
    major_pent = {"1", "2", "3", "5", "6"}
    if len(s & minor_pent) >= 4 and len(s - minor_pent) <= 1:
        return "minor"
    if len(s & major_pent) >= 4 and len(s - major_pent) <= 1:
        return "major"
    return None


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key profiles
# ---------------------------------------------------------------------------
KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def ks_key(chroma_mean):
    """Correlate a chroma vector against the Krumhansl-Schmuckler profiles."""
    import numpy as np

    best_key, best_mode, best_score = 0, "major", -999.0
    for i in range(12):
        r_maj = float(np.corrcoef(chroma_mean, np.roll(KS_MAJOR, -i))[0, 1])
        r_min = float(np.corrcoef(chroma_mean, np.roll(KS_MINOR, -i))[0, 1])
        if r_maj > best_score:
            best_score, best_key, best_mode = r_maj, i, "major"
        if r_min > best_score:
            best_score, best_key, best_mode = r_min, i, "minor"
    return KEY_NAMES[best_key], best_mode, round(best_score, 4)


# ---------------------------------------------------------------------------
# Chord templates -- used by the MIDI parser's real chord detection
# ---------------------------------------------------------------------------
CHORD_TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus":  [0, 5, 7],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
}


def match_chord(pitch_class_weights):
    """
    Match a 12-element pitch-class weight vector against the chord templates.

    Returns (chord_symbol, score) or (None, score). `score` is the fraction of
    the template covered minus a penalty for foreign pitch classes, so a
    passing tone cannot invent a chord.
    """
    total = sum(pitch_class_weights) or 1e-9
    norm = [w / total for w in pitch_class_weights]

    best, best_score = None, 0.0
    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            template = {(root + i) % 12 for i in intervals}
            inside = sum(norm[p] for p in template)
            outside = 1.0 - inside
            coverage = inside / len(template) * len(template)
            score = coverage - outside * 1.5
            if score > best_score:
                best_score = score
                best = f"{KEY_NAMES[root]}{'' if quality == 'maj' else quality}"

    if best_score < 0.25:
        return None, round(best_score, 3)
    return best, round(best_score, 3)
