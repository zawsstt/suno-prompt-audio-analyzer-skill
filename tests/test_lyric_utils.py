"""Lyric parsing and the transferable-style extraction that replaced
'paste the reference lyrics into the Lyrics box'."""

from lyric_utils import (
    estimate_syllables,
    lyric_style_stats,
    parse_lyric_sheet,
    render_structure_only,
    strip_structure_tags,
)

SHEET = """[Verse 1]
I walked the wire again
Counting every step I take
[Chorus]
Hold the night, hold the night
We are burning like a star
[Verse 2]
Nothing here is left to lose
Hold the night, hold the night
"""


def test_parse_records_sections_and_order():
    parsed = parse_lyric_sheet(SHEET)
    labels = parsed["section_labels"]
    assert labels == ["Verse 1", "Chorus", "Verse 2"]
    assert len(parsed["flat_lines"]) == 6


def test_duplicate_lines_are_never_deduplicated():
    """
    v7.2's chorus handling was positional. For alignment, a chorus that appears
    twice MUST be aligned twice.
    """
    parsed = parse_lyric_sheet(SHEET)
    texts = [l["text"] for l in parsed["flat_lines"]]
    assert texts.count("Hold the night, hold the night") == 2


def test_structure_tags_are_stripped_for_alignment():
    clean = strip_structure_tags(SHEET)
    assert "[Verse 1]" not in clean
    assert "[Chorus]" not in clean
    # ...but the words themselves survive untouched.
    assert "Hold the night, hold the night" in clean
    assert len(clean.splitlines()) == 6


def test_inline_tags_are_handled():
    parsed = parse_lyric_sheet("Hello there [Chorus]\nSing it loud\n")
    assert parsed["section_labels"] == ["Chorus"]
    assert parsed["flat_lines"][0]["text"] == "Hello there"
    assert parsed["flat_lines"][1]["text"] == "Sing it loud"


def test_empty_sheet_is_safe():
    parsed = parse_lyric_sheet("")
    assert parsed["flat_lines"] == []
    assert lyric_style_stats(parsed) == {"available": False}


def test_syllable_estimation():
    assert estimate_syllables("Hold the night") == 3
    assert estimate_syllables("把夜色唱成海岸") == 7
    assert estimate_syllables("") == 0


def test_style_stats_contain_no_reference_words():
    """
    The whole point: the reference contributes STATISTICS, never WORDS.
    """
    parsed = parse_lyric_sheet(SHEET)
    stats = lyric_style_stats(parsed)
    blob = repr(stats).lower()

    for word in ("walked", "wire", "burning", "star", "counting"):
        assert word not in blob
    assert stats["available"] is True
    assert stats["line_count"] == 6
    assert stats["avg_syllables_per_line"] > 0


def test_style_stats_describe_the_hook_without_quoting_it():
    parsed = parse_lyric_sheet(SHEET)
    stats = lyric_style_stats(parsed)
    assert stats["hook"]["repeated_line_count"] == 1
    assert stats["hook"]["max_repetitions"] == 2
    assert stats["hook"]["avg_hook_syllables"] > 0


def test_perspective_detection():
    parsed = parse_lyric_sheet(SHEET)
    stats = lyric_style_stats(parsed)
    assert stats["perspective"]["dominant"] in {"first_person_singular", "first_person_plural"}


def test_structure_scaffold_has_no_reference_words():
    parsed = parse_lyric_sheet(SHEET)
    scaffold = render_structure_only(parsed, title_hint="New Song")
    assert "Hold the night" not in scaffold
    assert "walked" not in scaffold
    assert "[Verse 1]" in scaffold
    assert "[Chorus]" in scaffold
    assert "[Title: New Song]" in scaffold


def test_structure_scaffold_survives_an_empty_sheet():
    scaffold = render_structure_only({"sections": []})
    assert "[Verse 1]" in scaffold
    assert "write your" in scaffold
