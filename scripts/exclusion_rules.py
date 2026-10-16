#!/usr/bin/env python3
"""
exclusion_rules.py -- build the Suno "Exclude Styles" string without shooting
the positive prompt in the foot.

v7.2 problems
-------------
1. `auto_exclusion_map.yaml` was keyed `cinematic_orchestral` and `jazz_neo_soul`
   while the compiler tested `if g_key in genre_clean` against strings like
   `"cinematic, orchestral"`. `"cinematic_orchestral" in "cinematic, orchestral"`
   is False, so orthogonal exclusion almost never fired. Only `lo-fi` matched,
   and only because it happened to be a substring.
2. Generic exclusions were appended unconditionally, so `harsh distortion`
   cancelled a requested distorted guitar and `muffled vocals` cancelled an
   intentionally muffled vocal.
3. Over-long output was cut with `[:197] + "..."`, and the trailing ellipsis
   became a style token.

v7.3 contract
-------------
  * Genre matching is done against explicit alias lists (canonical keys), never
    by substring accident.
  * Every negative tag passes a same-root check against the positive tags.
    A negative tag that shares an instrument root with a positive tag is
    dropped and the reason is recorded.
  * Over-long output drops WHOLE tags. No ellipsis, ever.
"""

from suno_contract import (
    MAX_EXCLUDE_TAGS,
    dedupe_tags,
    fit_budget,
    has_same_root,
    normalize_tag,
)

# Generic anti-artefact tags. These are still useful, but they are no longer
# applied unconditionally -- they go through the same same-root filter as the
# genre conflicts.
GENERIC_QUALITY_TAGS = [
    "muddy mix",
    "low quality",
    "harsh distortion",
    "clipping",
    "muffled vocals",
    "noisy background",
    "amateur mix",
]


def _norm(s):
    """Normalise a genre/tag string for comparison."""
    if not s:
        return ""
    t = str(s).lower().replace("_", " ").replace("-", " ")
    return " ".join(t.split())


def canonical_genres(genre_tags, exclusion_map):
    """
    Map the compiler's own genre tags onto canonical exclusion keys.

    `exclusion_map` is the parsed `auto_exclusion_map.yaml`. Each entry under
    `genre_conflict_exclusions` must declare a `matches` alias list containing
    the exact strings the compiler can emit.

    Returns {canonical_key: matched_alias}.
    """
    conflicts = (exclusion_map or {}).get("genre_conflict_exclusions") or {}
    normalized_positive = {_norm(g) for g in (genre_tags or []) if g}

    matched = {}
    for key, spec in conflicts.items():
        if not isinstance(spec, dict):
            continue
        aliases = spec.get("matches") or []
        for alias in aliases:
            if _norm(alias) in normalized_positive:
                matched[key] = alias
                break
    return matched


def filter_negative_tags(negative_tags, positive_tags):
    """
    Drop any negative tag that shares an instrument root with a positive tag.

    Returns (kept, dropped) where `dropped` is a list of
    {"tag": ..., "reason": ...} entries.
    """
    positives = [p for p in (positive_tags or []) if p]
    kept, dropped = [], []
    for raw in negative_tags or []:
        tag = normalize_tag(raw)
        if not tag:
            continue
        conflict = None
        for pos in positives:
            if has_same_root(tag, pos):
                conflict = pos
                break
        if conflict:
            dropped.append({
                "tag": tag,
                "reason": f"same instrument root as positive tag '{conflict}'",
            })
        else:
            kept.append(tag)
    return kept, dropped


def build_exclusion_prompt(positive_tags,
                           genre_tags,
                           exclusion_map,
                           limit=200,
                           include_generic=True):
    """
    Assemble the final Exclude Styles string.

    Returns:
        {
          "prompt": str,
          "tags": [...],
          "dropped_same_root": [...],
          "dropped_over_budget": [...],
          "matched_genre_keys": {...},
          "limit": int,
          "char_count": int,
        }
    """
    exclusion_map = exclusion_map or {}

    generic_spec = exclusion_map.get("generic_quality_exclusions") or {}
    generic_tags = list(generic_spec.get("tags") or GENERIC_QUALITY_TAGS) \
        if include_generic else []

    matched_keys = canonical_genres(genre_tags, exclusion_map)
    conflicts = exclusion_map.get("genre_conflict_exclusions") or {}
    genre_tags_out = []
    for key in matched_keys:
        spec = conflicts.get(key) or {}
        genre_tags_out.extend(spec.get("exclude") or [])

    candidates = dedupe_tags(generic_tags + genre_tags_out)
    kept, dropped_same_root = filter_negative_tags(candidates, positive_tags)

    if len(kept) > MAX_EXCLUDE_TAGS:
        # Keep the genre-specific conflicts first: they carry the most signal.
        head = [t for t in kept if t not in generic_tags]
        tail = [t for t in kept if t in generic_tags]
        kept = (head + tail)[:MAX_EXCLUDE_TAGS]

    kept, over_budget = fit_budget(kept, limit)

    return {
        "prompt": ", ".join(kept),
        "tags": kept,
        "dropped_same_root": dropped_same_root,
        "dropped_over_budget": over_budget,
        "matched_genre_keys": matched_keys,
        "limit": limit,
        "char_count": len(", ".join(kept)),
    }
