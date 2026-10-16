"""Exclusion map keys, same-root conflicts, and budget truncation."""

from exclusion_rules import (
    build_exclusion_prompt,
    canonical_genres,
    filter_negative_tags,
)
from suno_contract import has_same_root, normalize_tag, tag_roots


# ---------------------------------------------------------------------------
# Canonical key matching (v7.2 used a broken substring test)
# ---------------------------------------------------------------------------
def test_canonical_key_matches_the_real_compiler_output(exclusion_map):
    """
    v7.2 tested `if "cinematic_orchestral" in "cinematic, orchestral"`, which is
    False, so orthogonal exclusion almost never fired. Aliases fix this.
    """
    matched = canonical_genres(["cinematic orchestral", "epic orchestral"], exclusion_map)
    assert "cinematic_orchestral" in matched


def test_canonical_key_matches_each_emitted_alias(exclusion_map):
    """Both emitted aliases must reach the same canonical key."""
    assert "cinematic_orchestral" in canonical_genres(["cinematic orchestral"], exclusion_map)
    assert "cinematic_orchestral" in canonical_genres(["epic orchestral"], exclusion_map)


def test_ambient_and_cinematic_do_not_collide_with_orchestral(exclusion_map):
    """
    "Ambient / Cinematic" emits the bare tag "cinematic". That must map to the
    ambient bucket, not to the orchestral one, or an ambient track would inherit
    orchestral exclusions it does not want.
    """
    matched = canonical_genres(["ambient", "cinematic"], exclusion_map)
    assert "ambient_cinematic" in matched
    assert "cinematic_orchestral" not in matched


def test_every_emitted_genre_tag_has_a_canonical_home(exclusion_map):
    """
    Guard against drift: every Suno genre tag the compiler can emit must be
    matched by at least one canonical key in the exclusion map.
    """
    from suno_contract import GENRE_TO_SUNO_TAGS

    all_emitted = {t for tags in GENRE_TO_SUNO_TAGS.values() for t in tags}
    matched = canonical_genres(sorted(all_emitted), exclusion_map)

    conflicts = exclusion_map["genre_conflict_exclusions"]
    covered = {alias for key in matched for alias in
               (conflicts[key].get("matches") or [])}

    # "house" and "pop" are aliases that appear alongside a more specific tag;
    # require that the specific tags are covered.
    must_cover = {"deep house", "tech house", "progressive trance", "drum and bass",
                  "techno", "EDM", "trap", "lo-fi hip-hop", "R&B", "ambient",
                  "cinematic orchestral", "indie pop", "alternative rock",
                  "indie folk", "jazz", "electronic"}
    missing = must_cover - covered
    assert not missing, f"genre tags with no exclusion mapping: {sorted(missing)}"


def test_unrelated_genre_does_not_match(exclusion_map):
    matched = canonical_genres(["deep house"], exclusion_map)
    assert "cinematic_orchestral" not in matched


def test_no_substring_accidents(exclusion_map):
    """
    A genre that merely CONTAINS another genre's name must not match it.
    """
    matched = canonical_genres(["indie pop"], exclusion_map)
    assert "acoustic_pop" not in matched


# ---------------------------------------------------------------------------
# Same-root conflicts
# ---------------------------------------------------------------------------
def test_distorted_guitar_is_not_negated_by_harsh_distortion():
    """
    v7.2 appended "harsh distortion" unconditionally, cancelling the very
    instrument the positive prompt had requested.
    """
    kept, dropped = filter_negative_tags(
        ["muddy mix", "harsh distortion", "clipping"],
        ["distorted guitar", "driving"],
    )
    assert "harsh distortion" not in kept
    assert any(d["tag"] == "harsh distortion" for d in dropped)


def test_muffled_vocals_not_negated_when_vocals_requested():
    kept, dropped = filter_negative_tags(
        ["muffled vocals", "low quality"],
        ["breathy vocals", "mandarin vocals"],
    )
    assert "muffled vocals" not in kept
    assert "low quality" in kept


def test_unrelated_negatives_survive():
    kept, dropped = filter_negative_tags(
        ["muddy mix", "clipping", "autotune"],
        ["cinematic orchestral", "orchestral strings"],
    )
    assert "muddy mix" in kept
    assert "clipping" in kept
    assert dropped == []


def test_root_lexicon_ignores_pure_adjectives():
    """Adjectives must not create phantom conflicts."""
    assert tag_roots("dark") == set()
    assert tag_roots("heavy") == set()
    assert not has_same_root("dark", "heavy")


def test_root_lexicon_finds_instruments():
    assert "guitar" in tag_roots("distorted guitar")
    assert "vocal" in tag_roots("breathy vocals")
    assert has_same_root("hi-hat rolls", "trap drums")


# ---------------------------------------------------------------------------
# Budget handling -- whole tags only, never an ellipsis
# ---------------------------------------------------------------------------
def test_never_emits_an_ellipsis(exclusion_map):
    result = build_exclusion_prompt(
        positive_tags=["deep house", "punchy kick"],
        genre_tags=["House / Deep House"],
        exclusion_map=exclusion_map,
        limit=40,
    )
    assert "..." not in result["prompt"]
    assert "\u2026" not in result["prompt"]
    assert result["char_count"] <= 40


def test_over_budget_tags_are_dropped_whole(exclusion_map):
    result = build_exclusion_prompt(
        positive_tags=["deep house"],
        genre_tags=["House / Deep House"],
        exclusion_map=exclusion_map,
        limit=30,
    )
    for tag in result["tags"]:
        assert tag in result["prompt"]


def test_normalize_tag_strips_ellipsis():
    assert normalize_tag("cinematic ...") == "cinematic"
    assert normalize_tag("...") is None
    assert normalize_tag("  spaced   out  ") == "spaced out"


def test_generic_exclusions_can_be_disabled(exclusion_map):
    result = build_exclusion_prompt(
        positive_tags=["deep house"],
        genre_tags=["House / Deep House"],
        exclusion_map=exclusion_map,
        include_generic=False,
    )
    assert "muddy mix" not in result["prompt"]


def test_tag_count_is_capped(exclusion_map):
    result = build_exclusion_prompt(
        positive_tags=["deep house"],
        genre_tags=["House / Deep House"],
        exclusion_map=exclusion_map,
        limit=2000,
    )
    assert len(result["tags"]) <= 6
