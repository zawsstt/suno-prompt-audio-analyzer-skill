#!/usr/bin/env python3
"""
theory_triggers.py -- turn music-theory labels into Suno tags ONLY when the
analysis actually produced the evidence for them.

v7.2 problem
------------
`suno_tag_library.yaml` declared a `theory_tags` whitelist with explicit
`trigger` conditions, but no script ever read it. The compiler instead emitted
tags from a chain of `if/elif` guesses:

    elif "Phrygian" in modal: theory_tags.append("phrygian tension")
    ...
    else: theory_tags.append("four-chord pop loop")

So a tag could be emitted with zero supporting evidence, and `detect_mode()`
labelled nearly every minor song as Phrygian because it never checked for a
flat second.

v7.3 contract
-------------
  * Every theory tag requires a named evidence key to be present AND to
    satisfy its condition.
  * A missing evidence key means "not measured", which means "do not emit".
    It is never treated as zero and it never falls back to a default tag.
  * Phrygian requires a measured flat 2nd that is stronger than the natural
    2nd. Natural minor (b6 + b7) resolves to Aeolian.
  * The chord-progression triggers are matched against the real Roman-numeral
    loop, cyclically, so `i-VI-III-VII` is only emitted when that loop exists.
"""

# ---------------------------------------------------------------------------
# Roman numeral helpers
# ---------------------------------------------------------------------------

_ROMAN_TOKENS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii",
    "I", "II", "III", "IV", "V", "VI", "VII",
    "bII", "bIII", "bV", "bVI", "bVII",
}


def normalize_roman(token):
    """
    Normalise a Roman numeral to the form used in trigger strings.

    Accepts 'i', 'i7', 'I7', 'V7', 'iv6', 'bVII7' and strips quality suffixes
    and the 'o'/'+' marks so that 'V7' and 'v' compare consistently.
    """
    if token is None:
        return None
    t = str(token).strip()
    if not t:
        return None
    t = t.replace("o", "").replace("+", "")
    t = t.replace("7", "").replace("6", "").replace("9", "").replace("sus4", "")
    t = t.replace("maj", "").replace("min", "")
    t = t.strip()
    return t or None


def roman_loop_string(romans):
    """Join a list of Roman numerals into a 'i-VI-III-VII' style loop."""
    parts = [normalize_roman(r) for r in romans]
    return "-".join(p for p in parts if p)


def _split_pattern(pattern):
    return [p for p in (normalize_roman(x) for x in str(pattern).split("-")) if p]


def contains_cyclic(loop, pattern):
    """
    True when `pattern` appears as a contiguous run inside `loop`, wrapping
    around the end. Used so that a 4-chord loop rotated by the analysis window
    still matches its trigger.
    """
    if not loop or not pattern:
        return False
    if len(pattern) > len(loop):
        return False
    doubled = list(loop) + list(loop[:len(pattern) - 1])
    n = len(pattern)
    for i in range(len(loop)):
        if doubled[i:i + n] == list(pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Evidence-based trigger evaluation
# ---------------------------------------------------------------------------

def _mode_matches(evidence, *needles):
    modal = (evidence.get("modal_flavor") or "").lower()
    return any(n.lower() in modal for n in needles)


def _evaluate_progression(entry, evidence):
    loop = evidence.get("roman_loop")
    if not loop:
        return None
    triggers = entry.get("trigger") or []
    for pattern in triggers:
        pat = _split_pattern(pattern)
        if pat and contains_cyclic(loop, pat):
            return f"progression matched {pattern}"
    return None


def _evaluate_mode(entry, evidence):
    tag = entry.get("tag")
    modal_evidence = evidence.get("modal_evidence") or {}

    if tag == "phrygian tension":
        # Requires a MEASURED flat 2nd that beats the natural 2nd.
        if not _mode_matches(evidence, "phrygian"):
            return None
        b2 = modal_evidence.get("b2")
        nat2 = modal_evidence.get("natural_2")
        if b2 is None or nat2 is None:
            return None
        if b2 < 0.35 or b2 <= nat2 * 1.1:
            return None
        return f"phrygian with b2={b2:.2f} > natural 2nd={nat2:.2f}"

    if tag == "dorian mode":
        if _mode_matches(evidence, "dorian"):
            return "dorian detected"
        return None

    if tag == "mixolydian groove":
        if _mode_matches(evidence, "mixolydian"):
            return "mixolydian detected"
        return None

    if tag == "minor pentatonic":
        if evidence.get("pentatonic") == "minor":
            return "minor pentatonic degree set"
        return None

    if tag == "major pentatonic":
        if evidence.get("pentatonic") == "major":
            return "major pentatonic degree set"
        return None

    return None


def _evaluate_rhythm(entry, evidence):
    tag = entry.get("tag")

    if tag == "syncopated bass":
        # Must be measured ON THE BASS, not on all notes.
        ratio = evidence.get("bass_syncopation_ratio")
        if ratio is None:
            return None
        if ratio > 0.4:
            return f"bass syncopation {ratio:.2f}"
        return None

    if tag == "half-time drum beat":
        if evidence.get("half_time") is True:
            return "half-time kick pattern"
        return None

    if tag == "swung 8ths":
        ratio = evidence.get("swing_ratio")
        if ratio is None:
            return None
        if ratio >= 1.2:
            return f"swing ratio {ratio:.2f}"
        return None

    if tag == "straight 16ths":
        ratio = evidence.get("swing_ratio")
        if ratio is None or not (0.9 <= ratio <= 1.1):
            return None
        if evidence.get("sixteenth_hats") is True:
            return "straight 16th hi-hats"
        return None

    if tag == "polyrhythm 3:4":
        if evidence.get("polyrhythm") is True:
            return "3:4 polyrhythm measured"
        return None

    return None


def evaluate_theory_tags(evidence, library, max_tags=2):
    """
    Evaluate the whitelist against measured evidence.

    `evidence` keys (all optional -- absent means "not measured"):
        modal_flavor            str   e.g. "Aeolian (natural minor)"
        modal_evidence          dict  {"b2": float, "natural_2": float}
        roman_loop              list  e.g. ["i", "VI", "III", "VII"]
        pentatonic              "minor" | "major" | None
        bass_syncopation_ratio  float
        swing_ratio             float | None
        half_time               bool | None
        sixteenth_hats          bool | None
        polyrhythm              bool | None

    `library` is the parsed `suno_tag_library.yaml`.

    Returns a list of dicts: {"tag": str, "why": str}. Empty when nothing fired.
    """
    theory = (library or {}).get("theory_tags") or {}
    if not theory:
        return []

    out = []
    seen = set()

    groups = (
        ("progressions", _evaluate_progression),
        ("modes_and_scales", _evaluate_mode),
        ("rhythm_and_groove", _evaluate_rhythm),
    )

    for group_name, evaluator in groups:
        for entry in theory.get(group_name) or []:
            if not isinstance(entry, dict):
                continue
            tag = entry.get("tag")
            if not tag or tag in seen:
                continue
            why = evaluator(entry, evidence)
            if why:
                seen.add(tag)
                out.append({"tag": tag, "why": why, "group": group_name})

    return out[:max_tags]
