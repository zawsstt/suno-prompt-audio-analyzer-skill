#!/usr/bin/env python3
"""
run_pipeline.py -- the ONE entry point.

v7.2 had no orchestrator. `analyze_audio.py` produced a large JSON that already
contained a `suno_prompt` built by a compiler capped at 120/200 characters,
while `compile_prompt.py` produced a different prompt at 150-350 characters.
Which answer you got depended on which script the agent happened to call.

v7.3 has a single path:

    analyze_audio.py  ->  evidence only
    identify_song.py  ->  identity + lyrics policy (no reference words)
    compile_prompt.py ->  the three Suno fields

This script just runs them in order and prints the result ready to paste.

Usage:
    python scripts/run_pipeline.py song.mp3
    python scripts/run_pipeline.py song.mp3 --artist "The Weeknd" --out-dir out
    python scripts/run_pipeline.py song.mp3 --align-lyrics lyrics.txt
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _run(module, args, label):
    cmd = [sys.executable, os.path.join(HERE, module)] + args
    print(f"  -> {label}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"     failed (exit {proc.returncode})", file=sys.stderr)
        if proc.stderr:
            print("     " + proc.stderr.strip().replace("\n", "\n     "), file=sys.stderr)
    return proc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the full Suno prompt pipeline.")
    ap.add_argument("audio")
    ap.add_argument("--artist", default="", help="identified artist (optional)")
    ap.add_argument("--out-dir", default=".", help="output directory")
    ap.add_argument("--style-limit", type=int, default=None)
    ap.add_argument("--exclude-limit", type=int, default=None)
    ap.add_argument("--vocal-presence", choices=["vocal", "instrumental", "unknown"],
                    default=None, help="override the acoustic vocal gate")
    ap.add_argument("--skip-whisper", action="store_true",
                    help="do not transcribe (analysis only)")
    ap.add_argument("--no-identify", action="store_true",
                    help="skip track identification")
    ap.add_argument("--allow-reference-lyrics", action="store_true",
                    help="cover mode: include the reference lyrics (not original)")
    ap.add_argument("--align-lyrics", default=None,
                    help="also build a locked lyric timeline from this lyric sheet")
    args = ap.parse_args(argv)

    if not os.path.exists(args.audio):
        print(f"error: audio not found: {args.audio}", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    analysis_path = os.path.join(args.out_dir, "analysis.json")
    identity_path = os.path.join(args.out_dir, "identity.json")
    prompt_path = os.path.join(args.out_dir, "suno_prompt.json")

    print(f"Analyzing {args.audio}")

    # ---- 1. analysis ------------------------------------------------------
    analyze_args = [args.audio]
    if args.vocal_presence:
        analyze_args += ["--vocal-presence", args.vocal_presence]
    if args.skip_whisper:
        analyze_args.append("--skip-whisper")

    proc = _run("analyze_audio.py", analyze_args, "audio analysis")
    if proc.returncode != 0 or not proc.stdout.strip():
        print("error: analysis failed; stopping.", file=sys.stderr)
        return 1
    try:
        analysis = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"error: could not parse analysis output: {exc}", file=sys.stderr)
        return 1
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, ensure_ascii=False, indent=2)

    # ---- 2. identity ------------------------------------------------------
    identity = None
    if not args.no_identify:
        ident_args = [args.audio, analysis_path]
        if args.allow_reference_lyrics:
            ident_args.append("--allow-reference-lyrics")
        proc = _run("identify_song.py", ident_args, "track identification")
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                identity = json.loads(proc.stdout)
            except json.JSONDecodeError:
                identity = None
            if identity:
                with open(identity_path, "w", encoding="utf-8") as fh:
                    json.dump(identity, fh, ensure_ascii=False, indent=2)

    # ---- 3. compile -------------------------------------------------------
    artist = args.artist
    if not artist and identity and identity.get("identity"):
        artist = identity["identity"].get("artist") or ""

    compile_args = [analysis_path, artist]
    if identity:
        compile_args += ["--identity", identity_path]
    if args.style_limit:
        compile_args += ["--style-limit", str(args.style_limit)]
    if args.exclude_limit:
        compile_args += ["--exclude-limit", str(args.exclude_limit)]

    proc = _run("compile_prompt.py", compile_args, "prompt compilation")
    if proc.returncode != 0 or not proc.stdout.strip():
        print("error: prompt compilation failed.", file=sys.stderr)
        return 1
    try:
        compiled = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"error: could not parse compiler output: {exc}", file=sys.stderr)
        return 1
    with open(prompt_path, "w", encoding="utf-8") as fh:
        json.dump(compiled, fh, ensure_ascii=False, indent=2)

    # ---- 4. optional lyric alignment --------------------------------------
    align_result = None
    if args.align_lyrics:
        align_args = [args.audio, args.align_lyrics, analysis_path,
                      "--out-dir", args.out_dir]
        proc = _run("align_lyrics.py", align_args, "lyric timeline alignment")
        align_result = proc.returncode

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 66)
    print("SUNO FIELDS")
    print("=" * 66)

    mode = compiled.get("mode")
    vp = compiled.get("vocal_presence") or {}
    print(f"Vocal mode: {mode}  ({vp.get('reason', 'n/a')})")
    if identity and identity.get("identity"):
        ident = identity["identity"]
        print(f"Reference : {ident.get('artist')} - {ident.get('title')} "
              f"[{ident.get('source')}, confidence {ident.get('confidence')}]")
    print()

    style = compiled.get("style") or {}
    print(f"STYLE OF MUSIC  ({style.get('char_count')}/{style.get('limit')} chars)")
    print(f"  {style.get('prompt')}")

    exc = compiled.get("exclude") or {}
    print(f"\nEXCLUDE STYLES  ({exc.get('char_count')}/{exc.get('limit')} chars)")
    print(f"  {exc.get('prompt') or '(none)'}")

    plan = compiled.get("lyrics_plan") or {}
    print(f"\nLYRICS  policy={plan.get('policy')}")
    if plan.get("scaffold"):
        print("  " + "\n  ".join(str(plan["scaffold"]).splitlines()))

    if compiled.get("warnings"):
        print("\nWARNINGS")
        for w in compiled["warnings"]:
            print(f"  - {w}")

    dropped = style.get("dropped") or []
    if dropped:
        print("\nDROPPED FOR BUDGET (whole tags, never truncated)")
        for d in dropped:
            print(f"  - [{d['slot']}] {d['tag']}")

    same_root = (exc.get("dropped_same_root") or [])
    if same_root:
        print("\nNEGATIVE TAGS REMOVED (same root as a positive tag)")
        for d in same_root:
            print(f"  - {d['tag']}: {d['reason']}")

    print("\nFiles written:")
    for p in (analysis_path, identity_path if identity else None, prompt_path):
        if p:
            print(f"  {p}")
    if args.align_lyrics:
        print(f"  {os.path.join(args.out_dir, 'lyrics.lrc')}")
        print(f"  {os.path.join(args.out_dir, 'lyrics.srt')}")
        print(f"  {os.path.join(args.out_dir, 'mv_shots.json')}")
        if align_result not in (0, None):
            print("  (lyric alignment did not complete -- see the message above)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
