#!/usr/bin/env python3
"""
identify_song.py: Multi-channel song identity detection & lyric normalization for audio-analyzer v7
Usage:
    python identify_song.py <audio_file_path> [analysis_json_path]

Outputs: JSON with:
- identity: {title, artist, album, year, confidence, source}
- lyrics_structured: {has_lyrics, metatags_lyrics, is_instrumental}
- instrumental_timeline: (if instrumental) timeline arrangement scaffolding
"""

import os
import sys
import json
import re
import urllib.request
import urllib.parse
import subprocess

def clean_filename(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    base = re.sub(r"\[.*?\]|\(.*?\)", "", base).strip()
    if " - " in base:
        parts = base.split(" - ", 1)
        return {"artist": parts[0].strip(), "title": parts[1].strip(), "raw": base}
    return {"artist": "", "title": base, "raw": base}

def get_acoustid_fingerprint(filepath):
    """
    Generate Chromaprint fingerprint using fpcalc if available.
    """
    try:
        r = subprocess.run(["fpcalc", "-json", filepath], capture_output=True, text=True)
        if r.returncode == 0:
            data = json.loads(r.stdout)
            return data.get("duration"), data.get("fingerprint")
    except Exception:
        pass
    return None, None

def search_lrclib(title, artist="", duration=None):
    """
    Query LRClib public API (free, no key required).
    Returns synced lyrics or plain lyrics if available.
    """
    if not title:
        return None
    try:
        params = {"track_name": title}
        if artist:
            params["artist_name"] = artist
        if duration:
            params["duration"] = int(duration)

        url = "https://lrclib.net/api/get?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "AudioAnalyzerSkill/7.2"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data
    except Exception:
        try:
            url = "https://lrclib.net/api/search?" + urllib.parse.urlencode({"q": f"{artist} {title}".strip()})
            req = urllib.request.Request(url, headers={"User-Agent": "AudioAnalyzerSkill/7.2"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    results = json.loads(resp.read().decode("utf-8"))
                    if results and isinstance(results, list):
                        return results[0]
        except Exception:
            pass
    return None

def normalize_lyrics(lrc_text):
    """
    Lyric Normalizer:
    1. Parse LRC timestamps [mm:ss.xx]
    2. Normalize Chinese/English punctuation for Suno phrasing
    3. Group into logical song sections
    """
    lines = lrc_text.splitlines()
    parsed = []
    for line in lines:
        line = line.strip()
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if m:
            mins, secs, text = int(m.group(1)), float(m.group(2)), m.group(3).strip()
            sec_total = mins * 60 + secs
            if text:
                parsed.append({"sec": sec_total, "text": text})
        elif line and not line.startswith("["):
            parsed.append({"sec": None, "text": line})

    return parsed

def format_structured_lyrics(parsed_lines):
    """
    Insert Suno structure metatags ([Intro], [Verse], [Chorus], etc.)
    into normalized lyrics based on time progression and repetition.
    """
    if not parsed_lines:
        return ""

    formatted_sections = []
    current_section = "[Verse 1]"
    current_lines = [current_section]

    line_count = 0
    verse_idx = 1
    chorus_idx = 1

    raw_texts = [p["text"].strip() for p in parsed_lines if p["text"]]
    from collections import Counter
    text_freq = Counter(raw_texts)
    frequent_texts = {t for t, c in text_freq.items() if c >= 2 and len(t) > 3}

    in_chorus = False

    for item in parsed_lines:
        t = item["text"]
        # Normalize CJK punctuation
        t = t.replace("，", ", ").replace("。", "").replace("！", "!").replace("？", "?")
        t = re.sub(r"\s+", " ", t).strip()

        # Hook detection
        if t in frequent_texts and not in_chorus:
            if current_lines:
                formatted_sections.append("\n".join(current_lines))
            current_section = f"[Chorus {chorus_idx}]"
            chorus_idx += 1
            current_lines = [current_section]
            in_chorus = True
            line_count = 0
        elif t not in frequent_texts and in_chorus and line_count >= 4:
            if current_lines:
                formatted_sections.append("\n".join(current_lines))
            verse_idx += 1
            current_section = f"[Verse {verse_idx}]"
            current_lines = [current_section]
            in_chorus = False
            line_count = 0

        current_lines.append(t)
        line_count += 1

        if line_count >= 8 and not in_chorus:
            formatted_sections.append("\n".join(current_lines))
            verse_idx += 1
            current_section = f"[Verse {verse_idx}]"
            current_lines = [current_section]
            line_count = 0

    if current_lines:
        formatted_sections.append("\n".join(current_lines))

    full_output = "[Intro]\n\n" + "\n\n".join(formatted_sections) + "\n\n[Outro]"
    return full_output

def generate_instrumental_timeline(analysis_data):
    """
    Generate Suno Lyrics scaffolding for purely instrumental tracks.
    Uses energy/frequency curves to build dramatic arrangement instructions.
    """
    bpm = analysis_data.get("summary", {}).get("tempo_bpm", 120)
    genres = analysis_data.get("production_style", {}).get("likely_genres", ["Cinematic"])
    genre_str = genres[0] if genres else "Orchestral"

    timeline = f"""[Intro: Atmospheric ambient textures, subtle pulse, {genre_str} theme]
[Build-up: Rising tension, dynamic percussion enters, {bpm} BPM drive]
[Drop / Main Theme: Full instrumentation, powerful rhythmic hook, deep sub-bass]
[Verse: Stripped back arrangement, melodic solo lead over soft background chords]
[Bridge / Escalation: Accelerating percussion, orchestral brass stabs and rising sweeps]
[Climax: Massive dynamic peak, triumphant full-spectrum wall of sound]
[Outro: Slow decrescendo, lingering atmospheric tail, fade out]"""
    return timeline

def identify_and_process(audio_path, analysis_data=None):
    fn_info = clean_filename(audio_path)
    result = {
        "candidate": fn_info,
        "is_instrumental": False,
        "identity_confidence": 0.0,
        "verified_track": None,
        "structured_lyrics": None,
        "instrumental_scaffolding": None
    }

    # Instrumental gate check
    if analysis_data:
        spotify_feat = analysis_data.get("spotify_like_features", {})
        inst_score = spotify_feat.get("instrumentalness", 0.0)
        has_lyrics_flag = analysis_data.get("lyrics", {}).get("has_lyrics", True)

        if inst_score >= 0.7 or not has_lyrics_flag:
            result["is_instrumental"] = True
            result["instrumental_scaffolding"] = generate_instrumental_timeline(analysis_data)
            return result

    # Query LRClib
    if fn_info["title"]:
        track_info = search_lrclib(fn_info["title"], fn_info["artist"])
        if track_info:
            result["verified_track"] = {
                "title": track_info.get("trackName"),
                "artist": track_info.get("artistName"),
                "album": track_info.get("albumName"),
                "duration": track_info.get("duration")
            }
            result["identity_confidence"] = 0.9 if fn_info["artist"] else 0.7

            raw_synced = track_info.get("syncedLyrics")
            raw_plain = track_info.get("plainLyrics")

            if raw_synced:
                parsed = normalize_lyrics(raw_synced)
                result["structured_lyrics"] = format_structured_lyrics(parsed)
            elif raw_plain:
                lines = [{"sec": None, "text": l} for l in raw_plain.splitlines() if l.strip()]
                result["structured_lyrics"] = format_structured_lyrics(lines)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python identify_song.py <audio_file_path> [analysis_json_path]")
        sys.exit(1)

    audio_file = sys.argv[1]
    analysis_dict = None
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            analysis_dict = json.load(f)

    res = identify_and_process(audio_file, analysis_dict)
    print(json.dumps(res, ensure_ascii=False, indent=2))
