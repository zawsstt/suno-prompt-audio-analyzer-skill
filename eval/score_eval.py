#!/usr/bin/env python3
"""
score_eval.py -- score the pipeline against the 20-track listening set.

    python eval/score_eval.py [--manifest eval/manifest.yaml] [--json]

v7.2 shipped five-star accuracy claims based on one Hans Zimmer file. This
script is how those claims get replaced by measurements.

For every manifest slot with a filled-in `audio:` path it runs the analyzer and
compares:

    vocal_presence   exact match on vocal / instrumental / unknown
    modal_flavor     exact match on the modal name
    phrygian         derived from the flat-2nd evidence flag
    feel             half_time flag
    tempo_band       derived from summary.tempo_bpm

Slots with no audio are reported as `not run`, never as a pass. Nothing is
scored until there is something to score.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from yaml_io import load_yaml  # noqa: E402


def tempo_band(bpm):
    if bpm is None:
        return None
    if bpm < 70:
        return "<70"
    if bpm < 90:
        return "70-90"
    if bpm < 110:
        return "90-110"
    if bpm < 130:
        return "110-130"
    if bpm < 150:
        return "130-150"
    return ">150"


def observed(analysis):
    """Pull the four scored dimensions out of an analysis dict."""
    tonality = analysis.get("tonality") or {}
    drums = analysis.get("drum_pattern") or {}
    modal_evidence = tonality.get("modal_evidence") or {}

    flat2 = modal_evidence.get("flat_2_confirmed")
    return {
        "vocal_presence": (analysis.get("vocal_presence") or {}).get("presence"),
        "modal_flavor": tonality.get("modal_flavor"),
        "phrygian": flat2 if flat2 is not None else None,
        "feel": "half_time" if drums.get("half_time") else "full_time",
        "tempo_band": tempo_band((analysis.get("summary") or {}).get("tempo_bpm")),
    }


def run_analysis(audio_path):
    cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "analyze_audio.py"),
           audio_path, "--skip-whisper"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, (proc.stderr or "analyzer produced no output").strip()[:400]
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"could not parse analyzer output: {exc}"


def compare(expected, got):
    """
    Return {dimension: "pass" | "fail" | "skip"}.

    A null expectation means "this track is meant to be ambiguous here"; the
    dimension is skipped rather than scored.
    """
    out = {}
    for key in ("vocal_presence", "modal_flavor", "phrygian", "feel", "tempo_band"):
        exp = expected.get(key)
        if exp is None:
            out[key] = "skip"
            continue
        out[key] = "pass" if got.get(key) == exp else "fail"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the pipeline on the listening set.")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.yaml"))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    manifest = load_yaml(args.manifest, required=True)
    audio_dir = manifest.get("audio_dir") or os.path.join(HERE, "audio")
    tracks = manifest.get("tracks") or []

    results = []
    totals = {}
    not_run = []

    for track in tracks:
        tid = track.get("id")
        audio = track.get("audio")
        expected = track.get("expected") or {}

        if not audio:
            not_run.append(tid)
            continue
        path = audio if os.path.isabs(audio) else os.path.join(audio_dir, audio)
        if not os.path.exists(path):
            not_run.append(tid)
            results.append({"id": tid, "status": "audio_missing", "path": path})
            continue

        analysis, error = run_analysis(path)
        if analysis is None:
            results.append({"id": tid, "status": "analysis_failed", "error": error})
            continue

        got = observed(analysis)
        verdicts = compare(expected, got)
        results.append({"id": tid, "status": "scored", "expected": expected,
                        "observed": got, "verdicts": verdicts,
                        "note": track.get("note")})
        for dim, verdict in verdicts.items():
            if verdict == "skip":
                continue
            bucket = totals.setdefault(dim, {"pass": 0, "fail": 0})
            bucket[verdict] += 1

    if args.json:
        print(json.dumps({"results": results, "totals": totals,
                          "not_run": not_run}, indent=2, ensure_ascii=False))
        return 0

    print("=" * 70)
    print("SUNO PIPELINE EVALUATION")
    print("=" * 70)

    for entry in results:
        if entry["status"] != "scored":
            print(f"  {entry['id']:<5} {entry['status']}"
                  + (f"  ({entry.get('error') or entry.get('path')})"
                     if entry.get("error") or entry.get("path") else ""))
            continue
        marks = " ".join(f"{k}={v}" for k, v in entry["verdicts"].items())
        print(f"  {entry['id']:<5} {marks}")

    if not_run:
        print(f"\n  Not run ({len(not_run)} of {len(tracks)} slots have no audio yet): "
              f"{', '.join(not_run)}")

    print()
    if not totals:
        print("  No dimension has been scored yet.")
        print("  Fill in eval/audio/ and the `audio:` fields, then re-run.")
        print("  Do not publish accuracy claims until this table has real numbers.")
        return 0

    print(f"  {'dimension':<18}{'pass':>6}{'fail':>6}{'accuracy':>11}")
    print("  " + "-" * 41)
    for dim in ("vocal_presence", "modal_flavor", "phrygian", "feel", "tempo_band"):
        bucket = totals.get(dim)
        if not bucket:
            continue
        total = bucket["pass"] + bucket["fail"]
        acc = bucket["pass"] / total if total else 0.0
        print(f"  {dim:<18}{bucket['pass']:>6}{bucket['fail']:>6}{acc:>10.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
