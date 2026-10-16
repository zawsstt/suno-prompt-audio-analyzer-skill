#!/usr/bin/env python3
"""
identify_song.py -- identify a reference track, and keep its WORDS out of the
original-song pipeline.

v7.2 problems
-------------
1. The README and SKILL advertised "three-channel identity detection
   (Whisper hook lines + AcoustID + filename)". `get_acoustid_fingerprint()` was
   defined but never called, there was no AcoustID query at all, and there was
   no Whisper hook search. Only the filename and one LRClib call were real.
2. A filename match was given 0.9 confidence with no duration check.
3. After identifying the track, `format_structured_lyrics()` repunctuated the
   reference song's real lyrics and handed them to Suno. That is a cover, not
   an original song. What is actually transferable -- syllable density, hook
   length, perspective, rhyme, section proportions -- was never extracted.
4. `generate_instrumental_timeline()` claimed to read the energy curve but was
   a hard-coded Intro/Build/Drop/Climax arc.
5. The identified artist was never passed to the compiler.

v7.3 contract
-------------
* Channels are reported honestly. AcoustID is either really queried (fpcalc +
  ACOUSTID_API_KEY) or reported as unavailable. Nothing is called "three-channel"
  unless three channels actually ran.
* Every identity match is duration-checked when a duration is known.
* The default lyrics policy is `write_new`. Reference words are NOT returned.
  `--allow-reference-lyrics` opts into cover mode explicitly and says so loudly.
* Instrumental scaffolding comes from measured energy evidence.

Usage:
    python identify_song.py <audio_file> [analysis.json]
                            [--duration 213.4]
                            [--allow-reference-lyrics]
                            [--out identity.json]
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lyric_utils import (  # noqa: E402
    lyric_style_stats,
    parse_lyric_sheet,
    render_structure_only,
)
from structure_plan import build_instrumental_timeline  # noqa: E402
from suno_contract import (  # noqa: E402
    PRESENCE_INSTRUMENTAL,
    PRESENCE_UNKNOWN,
    PRESENCE_VOCAL,
    is_confidently_instrumental,
)

USER_AGENT = "AudioAnalyzerSkill/7.3"
ACOUSTID_ENDPOINT = "https://api.acoustid.org/v2/lookup"
DURATION_TOLERANCE_SEC = 5.0
DURATION_TOLERANCE_RATIO = 0.03  # 3%


# ---------------------------------------------------------------------------
# Filename channel
# ---------------------------------------------------------------------------
def clean_filename(filepath):
    """
    Parse "Artist - Title" from a filename.

    Handles full-width dashes, underscores and bracketed noise, which v7.2's
    naive `split(" - ")` missed.
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    base = re.sub(r"[\[\(].*?[\]\)]", " ", base).strip()
    base = base.replace("_", " ")

    for sep in (" - ", " \u2013 ", " \u2014 ", " \uff0d ", " \uff0f "):
        if sep in base:
            artist, title = base.split(sep, 1)
            return {"artist": artist.strip(), "title": title.strip(), "raw": base}

    return {"artist": "", "title": base.strip(), "raw": base}


# ---------------------------------------------------------------------------
# AcoustID channel
# ---------------------------------------------------------------------------
def get_acoustid_fingerprint(filepath):
    """Chromaprint fingerprint via fpcalc. Returns (duration, fingerprint)."""
    import subprocess
    try:
        proc = subprocess.run(["fpcalc", "-json", filepath],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            return data.get("duration"), data.get("fingerprint")
    except Exception:
        pass
    return None, None


def acoustid_lookup(filepath, api_key, expected_duration=None):
    """
    Query AcoustID for real. Reports `status` so the caller can tell the
    difference between "no match" and "could not run".
    """
    if not api_key:
        return {"status": "unavailable",
                "reason": "ACOUSTID_API_KEY is not set; skipping the AcoustID channel."}

    duration, fingerprint = get_acoustid_fingerprint(filepath)
    if not fingerprint:
        return {"status": "unavailable",
                "reason": "fpcalc (chromaprint) is not installed; skipping the AcoustID channel."}

    params = {
        "client": api_key,
        "meta": "recordings+releasegroups+compress",
        "duration": int(duration) if duration else 0,
        "fingerprint": fingerprint,
    }
    try:
        url = ACOUSTID_ENDPOINT + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "error", "reason": f"AcoustID request failed: {exc}"}

    if payload.get("status") != "ok":
        return {"status": "error",
                "reason": f"AcoustID returned status={payload.get('status')}"}

    best = None
    for entry in payload.get("results") or []:
        score = float(entry.get("score") or 0)
        for rec in entry.get("recordings") or []:
            cand = {
                "title": rec.get("title"),
                "artist": ", ".join(a.get("name", "") for a in (rec.get("artists") or [])),
                "duration": rec.get("duration"),
                "acoustid_score": round(score, 3),
            }
            if best is None or cand["acoustid_score"] > best["acoustid_score"]:
                best = cand

    if best is None:
        return {"status": "no_match", "reason": "AcoustID returned no recordings."}

    ok, note = duration_ok(best.get("duration"), expected_duration or duration)
    best["duration_verified"] = ok
    best["duration_note"] = note
    best["status"] = "matched"
    return best


def duration_ok(candidate_duration, reference_duration):
    """
    v7.2 gave a filename match 0.9 confidence without ever comparing durations.
    A title match with the wrong length is a different recording.
    """
    if not candidate_duration or not reference_duration:
        return None, "duration unavailable; not verified"
    tol = max(DURATION_TOLERANCE_SEC, reference_duration * DURATION_TOLERANCE_RATIO)
    delta = abs(float(candidate_duration) - float(reference_duration))
    if delta <= tol:
        return True, f"duration matches within {delta:.1f}s"
    return False, f"duration differs by {delta:.1f}s (tolerance {tol:.1f}s)"


# ---------------------------------------------------------------------------
# LRClib channel
# ---------------------------------------------------------------------------
def search_lrclib(title, artist="", duration=None):
    """Query LRClib (free, no key). Returns the raw record or None."""
    if not title:
        return None
    base_headers = {"User-Agent": USER_AGENT}

    def _get(url):
        req = urllib.request.Request(url, headers=base_headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
        return None

    try:
        params = {"track_name": title}
        if artist:
            params["artist_name"] = artist
        if duration:
            params["duration"] = int(duration)
        data = _get("https://lrclib.net/api/get?" + urllib.parse.urlencode(params))
        if data:
            return data
    except Exception:
        pass

    try:
        query = f"{artist} {title}".strip()
        results = _get("https://lrclib.net/api/search?" + urllib.parse.urlencode({"q": query}))
        if isinstance(results, list) and results:
            return results[0]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def identify_and_process(audio_path, analysis_data=None, duration_hint=None,
                         allow_reference_lyrics=False):
    warnings = []
    fn_info = clean_filename(audio_path)

    analysis_data = analysis_data or {}
    lyrics_block = analysis_data.get("lyrics") or {}
    vocal_presence = analysis_data.get("vocal_presence") or {}
    presence = vocal_presence.get("presence") or PRESENCE_UNKNOWN

    duration = (duration_hint
                or (analysis_data.get("file_info") or {}).get("duration_sec"))

    result = {
        "candidate": fn_info,
        "identity": None,
        "vocal_mode": presence,
        "vocal_presence": vocal_presence or None,
        "lyrics_policy": None,
        "channels": {},
        "lyric_style_reference": None,
        "structure_scaffold": None,
        "instrumental_scaffold": None,
        "warnings": warnings,
    }

    # ---- channel 1: filename (evidence only, no confidence on its own) -----
    result["channels"]["filename"] = {
        "status": "parsed" if fn_info["artist"] else "title_only",
        "artist": fn_info["artist"],
        "title": fn_info["title"],
        "note": "A filename is a hint, not an identification.",
    }

    # ---- channel 2: AcoustID ----------------------------------------------
    api_key = os.environ.get("ACOUSTID_API_KEY")
    acoustid = acoustid_lookup(audio_path, api_key, expected_duration=duration)
    result["channels"]["acoustid"] = acoustid

    # ---- channel 3: LRClib ------------------------------------------------
    lrclib = search_lrclib(fn_info["title"], fn_info["artist"], duration)
    if lrclib:
        ok, note = duration_ok(lrclib.get("duration"), duration)
        result["channels"]["lrclib"] = {
            "status": "matched",
            "title": lrclib.get("trackName"),
            "artist": lrclib.get("artistName"),
            "album": lrclib.get("albumName"),
            "duration": lrclib.get("duration"),
            "duration_verified": ok,
            "duration_note": note,
        }
    else:
        result["channels"]["lrclib"] = {"status": "no_match"}

    channels_run = sum(1 for c in result["channels"].values()
                       if c.get("status") in {"matched", "parsed", "title_only"})

    # ---- merge into a single identity --------------------------------------
    identity = None
    if result["channels"]["lrclib"].get("status") == "matched":
        lc = result["channels"]["lrclib"]
        verified = lc.get("duration_verified")
        confidence = 0.75 if verified else (0.5 if verified is None else 0.3)
        identity = {
            "title": lc["title"], "artist": lc["artist"], "album": lc.get("album"),
            "duration": lc.get("duration"), "source": "lrclib",
            "confidence": confidence,
            "duration_verified": verified,
        }
    elif acoustid.get("status") == "matched":
        identity = {
            "title": acoustid.get("title"), "artist": acoustid.get("artist"),
            "album": None, "duration": acoustid.get("duration"),
            "source": "acoustid",
            "confidence": round(0.6 * acoustid.get("acoustid_score", 0.5) + 0.3, 3),
            "duration_verified": acoustid.get("duration_verified"),
        }

    if identity is None:
        warnings.append(
            "The track could not be identified. Filename parsing alone is not an "
            "identification, so no identity is reported. If you need artist "
            "deconstruction, pass --artist explicitly."
        )

    result["identity"] = identity
    result["channels_run"] = channels_run
    result["channels_note"] = ("Only the channels listed above actually ran. "
                               "Unavailable channels are reported, not assumed.")

    # ---- lyrics policy -----------------------------------------------------
    if is_confidently_instrumental(vocal_presence):
        result["lyrics_policy"] = "structure_only"
        result["instrumental_scaffold"] = build_instrumental_timeline(analysis_data)
        result["lyrics_note"] = ("Instrumental track. Use the scaffold in the Lyrics box. "
                                 "Sections are named only where the energy curve supports them.")
        return result

    if presence == PRESENCE_VOCAL and lyrics_block.get("has_lyrics"):
        result["lyrics_policy"] = "write_new"
        result["lyrics_note"] = ("Write a NEW lyric. The reference track's words are not "
                                 "returned. Only transferable style statistics are extracted.")
    else:
        result["lyrics_policy"] = "write_new"
        result["lyrics_note"] = ("No confirmed vocal content. Write a new lyric, or re-run "
                                 "the analyzer with --vocal-presence vocal if this track does sing.")
        return result

    # ---- extract transferable style statistics -----------------------------
    if lrclib and (lrclib.get("syncedLyrics") or lrclib.get("plainLyrics")):
        raw = lrclib.get("syncedLyrics") or lrclib.get("plainLyrics")
        if lrclib.get("syncedLyrics"):
            raw = re.sub(r"\[\d+:\d+(?:\.\d+)?\]", "", raw)
        parsed = parse_lyric_sheet(raw)
        result["lyric_style_reference"] = lyric_style_stats(parsed)
        result["structure_scaffold"] = render_structure_only(
            parsed, title_hint=identity.get("title") if identity else None)

        if allow_reference_lyrics:
            result["reference_lyrics"] = raw
            result["lyrics_policy"] = "cover_reference"
            result["lyrics_note"] = (
                "COVER MODE: the reference lyrics are included because you passed "
                "--allow-reference-lyrics. Suno will reproduce the original song, "
                "not a new one. Do not treat this as an original work."
            )
            warnings.append("Cover mode enabled: reference lyrics are included in the output.")
        else:
            result["reference_lyrics"] = None
            result["lyrics_note"] += (" (Reference lyrics withheld. Pass "
                                      "--allow-reference-lyrics only if you intend a cover.)")

    elif lrclib:
        warnings.append("LRClib matched the track but had no lyrics text.")

    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Identify a reference track and decide the lyrics policy.")
    ap.add_argument("audio_file")
    ap.add_argument("analysis_json", nargs="?", default=None)
    ap.add_argument("--duration", type=float, default=None,
                    help="known duration in seconds, if the analysis is unavailable")
    ap.add_argument("--allow-reference-lyrics", action="store_true",
                    help="include the reference lyrics (cover mode, not original)")
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = ap.parse_args(argv)

    if not os.path.exists(args.audio_file):
        print(f"error: audio not found: {args.audio_file}", file=sys.stderr)
        return 1

    analysis = None
    if args.analysis_json and os.path.exists(args.analysis_json):
        with open(args.analysis_json, "r", encoding="utf-8") as fh:
            analysis = json.load(fh)

    res = identify_and_process(
        args.audio_file, analysis, duration_hint=args.duration,
        allow_reference_lyrics=args.allow_reference_lyrics)

    text = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
