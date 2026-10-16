#!/usr/bin/env python3
"""
lyric_utils.py -- parse a lyric sheet and extract the parts that are actually
transferable to a NEW song.

v7.2 problem
------------
`format_structured_lyrics()` took the reference track's real lyrics, changed a
few punctuation marks, wrapped them in `[Verse]`/`[Chorus]`, and handed them to
Suno's Lyrics box. That is a cover, not an original song.

v7.3 contract
-------------
The reference contributes STYLE, never WORDS. What is transferable:
    syllable density, line length, hook length, repeated-hook count,
    narrative perspective, rhyme behaviour, section proportions.

This module is the only place allowed to look at reference lyric text, and its
output contains no reference words.
"""

import re

# ---------------------------------------------------------------------------
# Structure tags
# ---------------------------------------------------------------------------
_SECTION_TAG_RE = re.compile(r"^\s*[\[\(]\s*([^\]\)]+?)\s*[\]\)]\s*$")
_INLINE_TAG_RE = re.compile(r"[\[\(]\s*([^\]\)]+?)\s*[\]\)]")

# Section labels that are structural rather than lyric content.
_KNOWN_SECTION_WORDS = {
    "intro", "outro", "verse", "pre-chorus", "prechorus", "chorus", "bridge",
    "hook", "refrain", "break", "drop", "build", "build-up", "buildup",
    "instrumental", "solo", "interlude", "ad-lib", "adlib", "skit", "vamp",
    "tag", "coda", "ending", "fade", "spoken", "rap", "final chorus",
}


def _is_section_label(text):
    t = (text or "").strip().lower()
    if not t:
        return False
    # "Chorus", "Verse 2", "Pre-Chorus 1", "Final Chorus", "Instrumental Break"
    words = re.findall(r"[a-z\-]+", t)
    if not words:
        return False
    return words[0] in _KNOWN_SECTION_WORDS


def parse_lyric_sheet(text):
    """
    Split a lyric sheet into ordered sections.

    Structure tags are RECORDED but never treated as lyric content, so they can
    be handed back later without being fed to a forced aligner.

    Lines are kept in order and are NEVER de-duplicated: a chorus that appears
    three times must be aligned three times.

    Returns:
        {
          "sections": [{"label": "Verse 1", "lines": ["...", ...]}, ...],
          "flat_lines": [{"section": "Verse 1", "text": "..."}, ...],
          "section_labels": ["Verse 1", "Chorus 1", ...],
        }
    """
    if not text:
        return {"sections": [], "flat_lines": [], "section_labels": []}

    sections = []
    current = {"label": None, "lines": []}

    def flush():
        if current["lines"] or current["label"]:
            sections.append({"label": current["label"], "lines": list(current["lines"])})

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _SECTION_TAG_RE.match(line)
        if m and _is_section_label(m.group(1)):
            flush()
            current = {"label": m.group(1).strip(), "lines": []}
            continue

        # Inline tags such as "Hello [Chorus]" -- record the tag, keep the text.
        inline = [t for t in _INLINE_TAG_RE.findall(line) if _is_section_label(t)]
        stripped = _INLINE_TAG_RE.sub("", line).strip()
        if inline:
            flush()
            current = {"label": inline[-1].strip(), "lines": []}
        if stripped:
            current["lines"].append(stripped)

    flush()

    if not sections:
        return {"sections": [], "flat_lines": [], "section_labels": []}

    if sections and sections[0]["label"] is None:
        sections[0]["label"] = "Unlabelled"

    flat = []
    for sec in sections:
        for line in sec["lines"]:
            flat.append({"section": sec["label"], "text": line})

    return {
        "sections": sections,
        "flat_lines": flat,
        "section_labels": [s["label"] for s in sections],
    }


def strip_structure_tags(text):
    """
    Remove structure tags for feeding a forced aligner. A CTC aligner will
    otherwise try to find acoustics for the literal word "Chorus".
    """
    if not text:
        return ""
    out = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if _SECTION_TAG_RE.match(line) and _is_section_label(
                _SECTION_TAG_RE.match(line).group(1)):
            continue
        line = _INLINE_TAG_RE.sub("", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Syllables
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def estimate_syllables(text):
    """
    Rough syllable count.

    CJK: one syllable per ideograph / kana / hangul block.
    Latin: vowel groups per word, minus a silent trailing 'e'.
    """
    if not text:
        return 0

    cjk = len(_CJK_RE.findall(text))

    latin = 0
    for word in _LATIN_WORD_RE.findall(text.lower()):
        groups = _VOWEL_GROUP_RE.findall(word)
        n = len(groups)
        if n > 1 and word.endswith("e") and not word.endswith(("le", "ee", "ye")):
            n -= 1
        latin += max(1, n)

    return cjk + latin


# ---------------------------------------------------------------------------
# Style statistics -- the transferable part of a reference lyric
# ---------------------------------------------------------------------------
_PERSPECTIVE_MARKERS = {
    "first_person_singular": ["i", "i'm", "i've", "i'll", "i'd", "me", "my", "mine", "myself", "我", "我的"],
    "second_person":         ["you", "you're", "you've", "your", "yours", "你", "你的", "妳"],
    "first_person_plural":   ["we", "we're", "we've", "our", "ours", "us", "我们", "咱们"],
    "third_person":          ["he", "she", "they", "him", "her", "them", "his", "their", "他", "她", "他们"],
}

# Common English words that happen to look like the marker list; avoid false hits.
_MARKER_STOP = {"i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "they", "them"}


def _perspective(lines):
    counts = {k: 0 for k in _PERSPECTIVE_MARKERS}
    for line in lines:
        tokens = re.findall(r"[A-Za-z']+", line.lower()) + _CJK_RE.findall(line)
        token_set = set(tokens)
        for key, markers in _PERSPECTIVE_MARKERS.items():
            for mk in markers:
                if mk in token_set:
                    counts[key] += 1
                    break
    total = sum(counts.values())
    if total == 0:
        return {"dominant": "unclear", "counts": counts}
    dominant = max(counts, key=counts.get)
    return {"dominant": dominant, "counts": counts}


def _rhyme_tail(line):
    """Last vowel-ish tail of a line, used for a crude rhyme comparison."""
    words = re.findall(r"[A-Za-z']+", line.lower())
    if words:
        w = re.sub(r"[^a-z]", "", words[-1])
        m = re.search(r"[aeiouy][^aeiouy]*$", w)
        return m.group(0) if m else w[-2:]
    cjk = _CJK_RE.findall(line)
    if cjk:
        return cjk[-1]
    return ""


def _rhyme_behaviour(lines):
    tails = [_rhyme_tail(l) for l in lines]
    tails = [t for t in tails if t]
    if len(tails) < 4:
        return {"scheme_guess": "unclear", "rhyme_density": None}
    pairs = 0
    comparisons = 0
    for i in range(len(tails) - 1):
        comparisons += 1
        if tails[i] and tails[i] == tails[i + 1]:
            pairs += 1
    return {
        "scheme_guess": "rhyming" if pairs / max(comparisons, 1) > 0.3 else "loose/free",
        "rhyme_density": round(pairs / max(comparisons, 1), 3),
    }


def lyric_style_stats(parsed):
    """
    Extract the parts of a reference lyric that CAN be transferred to a new song.

    `parsed` is the output of `parse_lyric_sheet`.

    The returned dict contains NO reference words.
    """
    flat = parsed.get("flat_lines") or []
    lines = [l["text"] for l in flat if l.get("text")]
    if not lines:
        return {"available": False}

    syl = [estimate_syllables(l) for l in lines]
    syl = [s for s in syl if s > 0] or [0]
    char_lens = [len(re.sub(r"\s+", "", l)) for l in lines]

    # Hook: the most repeated line, described by LENGTH not by content.
    counts = {}
    for l in lines:
        key = re.sub(r"\s+", " ", l.strip().lower())
        counts[key] = counts.get(key, 0) + 1
    repeats = sorted((c for c in counts.values() if c >= 2), reverse=True)
    hook_lines = [k for k, c in counts.items() if c >= 2]
    hook_syllables = [estimate_syllables(k) for k in hook_lines] or [0]

    sections = parsed.get("sections") or []
    section_shape = [
        {"label": s.get("label"), "line_count": len(s.get("lines") or []),
         "syllables": sum(estimate_syllables(l) for l in (s.get("lines") or []))}
        for s in sections
    ]

    return {
        "available": True,
        "line_count": len(lines),
        "avg_syllables_per_line": round(sum(syl) / len(syl), 2),
        "syllable_density_spread": round(float(max(syl) - min(syl)), 2),
        "avg_chars_per_line": round(sum(char_lens) / len(char_lens), 2),
        "hook": {
            "repeated_line_count": len(hook_lines),
            "max_repetitions": repeats[0] if repeats else 1,
            "avg_hook_syllables": round(sum(hook_syllables) / len(hook_syllables), 2),
        },
        "perspective": _perspective(lines),
        "rhyme": _rhyme_behaviour(lines),
        "section_shape": section_shape,
    }


def render_structure_only(parsed, title_hint=None):
    """
    Produce a Suno-ready Lyrics scaffold that describes STRUCTURE ONLY.

    Every line is a placeholder. No reference words are copied, so the result
    is a genuinely new song rather than a cover.
    """
    sections = (parsed or {}).get("sections") or []
    if not sections:
        return "[Verse 1]\n(write your lyric here)\n\n[Chorus]\n(write your hook here)"

    out = []
    if title_hint:
        out.append(f"[Title: {title_hint}]")
        out.append("")

    stats = lyric_style_stats(parsed)
    avg_syl = int(round(stats.get("avg_syllables_per_line", 8))) if stats.get("available") else 8
    hook_syl = int(round(stats.get("hook", {}).get("avg_hook_syllables", avg_syl))) \
        if stats.get("available") else avg_syl

    for sec in sections:
        label = sec.get("label") or "Verse"
        n_lines = len(sec.get("lines") or []) or 4
        is_hook = any(w in str(label).lower() for w in ("chorus", "hook", "refrain"))
        target = hook_syl if is_hook else avg_syl
        out.append(f"[{label}]")
        for _ in range(n_lines):
            out.append(f"(~{target} syllables)")
        out.append("")

    return "\n".join(out).strip()
