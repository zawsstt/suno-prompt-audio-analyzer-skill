#!/usr/bin/env python3
"""
suno_contract.py -- single source of truth for what Suno actually reads.

Design rule (v7.3)
------------------
The pipeline exists to answer exactly four questions:

    1. vocal or instrumental
    2. style / genre character
    3. speed feel
    4. lyrics: write new, or provide structure only

Everything else the analyzer measures (LUFS, chroma, MFCC, Roman numerals,
drum timestamps, EQ curves) is REPORT-ONLY. It must never be placed into a
Suno Style, Lyrics or Exclude field.

Consequences enforced here:
  * There is exactly ONE style budget and ONE exclude budget. The old
    120 / 200 / 350 split is gone.
  * When a budget is exceeded we drop WHOLE tags, never truncate a tag.
    A trailing "..." is itself a style token and poisons the prompt.
  * Suno's own documentation order is respected:
        genre -> sub-genre -> instruments -> mood -> tempo/key
        -> vocal type -> production
"""

# ---------------------------------------------------------------------------
# Field budgets
# ---------------------------------------------------------------------------
# Suno changes these from time to time and they differ between the web app and
# the API. Keep the conservative values as defaults and let the caller override
# (run_pipeline.py exposes --style-limit / --exclude-limit). Never invent a
# second budget somewhere else in the codebase.
STYLE_LIMIT_DEFAULT = 200
EXCLUDE_LIMIT_DEFAULT = 200
LYRICS_LIMIT_DEFAULT = 5000

# Hard safety cap on how many exclude tags we will ever emit.
MAX_EXCLUDE_TAGS = 6

# ---------------------------------------------------------------------------
# Slot order == the order Suno sees the tags in.
# ---------------------------------------------------------------------------
# `vocal_mode` sits early on purpose. Suno's own guidance is
# genre -> sub-genre -> instruments -> mood -> tempo -> vocal -> production, but
# the single most consequential thing this pipeline can get wrong is claiming
# "instrumental" for a song that has vocals (that was the v7.2 failure). Keeping
# it near the front means it survives any budget squeeze.
TAG_SLOT_ORDER = [
    "genre",        # 1-2 tags
    "vocal_mode",   # instrumental / language / nothing when unknown
    "instruments",  # 2-3 tags
    "mood",         # mood word + theory tags + artist deconstruction
    "tempo",        # 1 tag, one of the four core answers
    "production",   # 0-1 tag
]

# Order in which tags are sacrificed when the budget is exceeded: least
# informative first. Genre and vocal_mode are sacrificed last, if ever.
DEFAULT_DROP_ORDER = [
    "production",
    "mood",        # drops the last artist tag, then theory, then the mood word
    "instruments",
    "tempo",
    "genre",
    "vocal_mode",
]

# ---------------------------------------------------------------------------
# Internal genre bucket -> Suno-facing tags
# ---------------------------------------------------------------------------
# Keys are the canonical internal genre buckets produced by analyze_audio.py.
# Values are the strings we are willing to send to Suno.
GENRE_TO_SUNO_TAGS = {
    "House / Deep House":                 ["deep house", "house"],
    "Tech House":                         ["tech house"],
    "Trance / Progressive":               ["progressive trance", "trance"],
    "Drum & Bass / Breakbeat":            ["drum and bass", "breakbeat"],
    "Techno / Hard Dance":                ["techno", "hard techno"],
    "EDM / Big Room":                     ["EDM", "big room"],
    "Trap / Hip-hop":                     ["trap", "hip-hop"],
    "Lo-fi / Boom-bap":                   ["lo-fi hip-hop", "boom bap"],
    "R&B / Neo-soul":                     ["R&B", "neo-soul"],
    "Ambient / Cinematic":                ["ambient", "cinematic"],
    "Cinematic / Orchestral":             ["cinematic orchestral", "epic orchestral"],
    "Pop / Indie Pop":                    ["indie pop", "pop"],
    "Rock / Alternative":                 ["alternative rock", "indie rock"],
    "Folk / Acoustic":                    ["indie folk", "acoustic"],
    "Jazz / Soul":                        ["jazz", "neo-soul"],
    "Electronic / Instrumental (mixed)":  ["electronic"],
}

# ---------------------------------------------------------------------------
# Speed feel -- one tag, derived from BPM only. This is what Suno reacts to.
# ---------------------------------------------------------------------------
def tempo_feel_tag(bpm):
    """Return a single Suno-facing speed descriptor for a BPM value."""
    try:
        bpm = float(bpm)
    except (TypeError, ValueError):
        return None
    if bpm < 70:
        return "very slow"
    if bpm < 90:
        return "slow"
    if bpm < 110:
        return "mid-tempo"
    if bpm < 130:
        return "upbeat"
    if bpm < 150:
        return "driving"
    return "fast"


# ---------------------------------------------------------------------------
# Language tag
# ---------------------------------------------------------------------------
LANGUAGE_TO_SUNO_TAG = {
    "en": "english vocals",
    "zh": "mandarin vocals",
    "yue": "cantonese vocals",
    "ja": "japanese vocals",
    "ko": "korean vocals",
    "es": "spanish vocals",
    "fr": "french vocals",
    "de": "german vocals",
    "pt": "portuguese vocals",
    "it": "italian vocals",
    "ru": "russian vocals",
    "ar": "arabic vocals",
}


def language_tag(language_code):
    """Map an ISO-ish language code to a Suno-facing vocal language tag."""
    if not language_code:
        return None
    code = str(language_code).strip().lower().split("-")[0]
    return LANGUAGE_TO_SUNO_TAG.get(code, f"{code} vocals")


# ---------------------------------------------------------------------------
# Tag hygiene
# ---------------------------------------------------------------------------
import re as _re

_WS_RE = _re.compile(r"\s+")
_PUNCT_TAIL_RE = _re.compile(r"[,;:.\s]+$")


def normalize_tag(tag):
    """
    Canonicalise a tag for dedupe and for Suno output.

    Also strips the '...' artefact: an ellipsis is never allowed to survive
    into a prompt, because Suno treats it as part of a style token.
    """
    if tag is None:
        return None
    t = str(tag).replace("\u2026", "...")
    t = t.replace("...", " ")
    t = _WS_RE.sub(" ", t).strip()
    t = _PUNCT_TAIL_RE.sub("", t).strip()
    if not t:
        return None
    # A tag must contain at least one letter or CJK character.
    if not _re.search(r"[^\W\d_]", t, flags=_re.UNICODE):
        return None
    return t


def dedupe_tags(tags):
    """Order-preserving, case-insensitive dedupe."""
    seen = set()
    out = []
    for raw in tags:
        t = normalize_tag(raw)
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Budget fitting -- drops whole tags, never truncates
# ---------------------------------------------------------------------------
def join_tags(tags):
    return ", ".join(t for t in tags if t)


def fit_budget(tags, limit):
    """
    Greedily keep tags in order until `limit` characters are reached.

    Returns (kept, dropped). Every entry in `dropped` is a COMPLETE tag that
    did not fit; nothing is ever cut in half and no ellipsis is ever added.
    """
    kept, dropped = [], []
    for raw in tags:
        t = normalize_tag(raw)
        if not t:
            continue
        candidate = join_tags(kept + [t])
        if len(candidate) <= limit:
            kept.append(t)
        else:
            dropped.append(t)
    return kept, dropped


def fit_slots(slots, limit, drop_order=None):
    """
    Fit an ordered mapping of slot-name -> [tags] into a character budget.

    Tags are emitted in TAG_SLOT_ORDER (the order Suno reads them). When the
    result exceeds `limit`, whole tags are removed from the tail of the
    least-important slot first, following DEFAULT_DROP_ORDER.

    Nothing is ever truncated mid-tag, and no ellipsis is ever appended: a
    trailing "..." is itself a style token and poisons the prompt.

    Returns (kept_in_emission_order, dropped) where `dropped` is a list of
    {"slot": name, "tag": tag}.
    """
    drop_order = list(drop_order or DEFAULT_DROP_ORDER)

    ordered_names = [n for n in TAG_SLOT_ORDER if n in slots]
    ordered_names += [n for n in slots if n not in TAG_SLOT_ORDER]

    # Dedupe globally, first occurrence wins. Without this the same tag can
    # arrive from two sources (e.g. "minor pentatonic" from both the measured
    # theory trigger and the artist deconstruction) and be emitted twice.
    working = {}
    seen = set()
    for name in ordered_names:
        kept = []
        for tag in dedupe_tags(slots.get(name) or []):
            if tag.lower() in seen:
                continue
            seen.add(tag.lower())
            kept.append(tag)
        working[name] = kept

    def flatten():
        return [t for name in ordered_names for t in working[name]]

    dropped = []
    if len(join_tags(flatten())) > limit:
        for name in drop_order + ordered_names:
            if name not in working:
                continue
            while working[name] and len(join_tags(flatten())) > limit:
                dropped.append({"slot": name, "tag": working[name].pop()})

    return flatten(), dropped


# ---------------------------------------------------------------------------
# Vocal presence vocabulary (shared by analyzer + compiler + identify)
# ---------------------------------------------------------------------------
PRESENCE_VOCAL = "vocal"
PRESENCE_INSTRUMENTAL = "instrumental"
PRESENCE_UNKNOWN = "unknown"

# Only this value may switch the pipeline into the instrumental branch.
# `unknown` MUST be treated as "there might be vocals" so that the vocal
# branch can never be silently disabled.
def is_confidently_instrumental(vocal_presence):
    if not vocal_presence:
        return False
    return vocal_presence.get("presence") == PRESENCE_INSTRUMENTAL


# ---------------------------------------------------------------------------
# Instrument lexicon -- used by exclusion_rules to stop a negative tag from
# negating an instrument that a positive tag already asked for.
# ---------------------------------------------------------------------------
INSTRUMENT_ROOTS = {
    # canonical root: surface forms
    "distort":   ["distortion", "distorted", "distorting", "overdrive",
                  "overdriven", "fuzz", "fuzzed"],
    "vocal":     ["vocal", "vocals", "voice", "voiced", "singing", "singer",
                  "falsetto", "breathy", "whispery", "choir", "chant",
                  "harmonies", "harmony", "shout", "shouted", "screaming"],
    "guitar":    ["guitar", "guitars", "guitar-driven"],
    "drum":      ["drum", "drums", "drumming", "kick", "snare", "hihat",
                  "hi-hat", "hats", "percussion", "taiko", "timpani"],
    "bass":      ["bass", "bassline", "sub", "808", "reese", "subbass"],
    "synth":     ["synth", "synths", "synthesizer", "synthesizers",
                  "synthwave", "arp", "arpeggio", "supersaw"],
    "piano":     ["piano", "rhodes", "wurlitzer", "keys", "keyboard"],
    "string":    ["string", "strings", "violin", "viola", "cello", "orchestra",
                  "orchestral", "symphonic", "chamber"],
    "brass":     ["brass", "trumpet", "trombone", "horn", "saxophone", "sax"],
    "flute":     ["flute", "clarinet", "oboe", "woodwind", "woodwinds"],
    "organ":     ["organ", "hammond"],
    "banjo":     ["banjo", "mandolin", "ukulele"],
    "acoustic":  ["acoustic", "unplugged", "fingerpicked", "nylon"],
    "folk":      ["folk", "bluegrass", "celtic", "americana"],
    "crackle":   ["crackle", "vinyl", "tape", "hiss"],
    "edm":       ["edm", "dubstep", "techno", "house", "trance", "wobble",
                  "bigroom", "big-room"],
    "metal":     ["metal", "metalcore", "djent", "thrash"],
    "autotune":  ["autotune", "auto-tune", "vocoder", "talkbox"],
}

# Build a reverse lookup once.
_SURFACE_TO_ROOT = {}
for _root, _forms in INSTRUMENT_ROOTS.items():
    _SURFACE_TO_ROOT[_root] = _root
    for _f in _forms:
        _SURFACE_TO_ROOT[_f] = _root


def _lemma(word):
    w = word.lower()
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tag_roots(tag):
    """
    Return the set of canonical instrument/timbre roots present in a tag.

    Only words that appear in the instrument lexicon produce a root, so purely
    adjectival tags ("dark", "heavy", "fast") never create false conflicts.
    """
    roots = set()
    if not tag:
        return roots
    words = _re.findall(r"[a-zA-Z][a-zA-Z\-']*", str(tag))
    for i, w in enumerate(words):
        for cand in (_lemma(w), w.lower()):
            r = _SURFACE_TO_ROOT.get(cand)
            if r:
                roots.add(r)
        # crude bigram handling: "hi hat" / "hi-hat"
        if i + 1 < len(words):
            bigram = f"{w.lower()}-{words[i + 1].lower()}"
            r = _SURFACE_TO_ROOT.get(bigram)
            if r:
                roots.add(r)
    return roots


def has_same_root(a, b):
    """True when two tags talk about the same instrument family."""
    return bool(tag_roots(a) & tag_roots(b))
