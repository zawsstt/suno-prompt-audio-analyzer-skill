#!/usr/bin/env python3
"""
align_lyrics.py -- build a locked lyric timeline for a finished song.

    python scripts/align_lyrics.py song.mp3 lyrics.txt [analysis.json]

Why not just re-transcribe with Whisper
---------------------------------------
You already wrote the lyrics. Re-transcribing them can only make them WORSE:
Whisper rewrites words, drops chorus repeats, and turns instrumental sections
into looping nonsense. And a transcript is not a video timeline anyway.

So we do forced alignment of YOUR words against the vocal stem, then snap line
heads onto the beat grid:

    1. separate the vocal stem (Demucs htdemucs)
    2. CTC forced alignment of the known lyric sheet (no rewriting)
    3. snap each line head to the nearest beat / downbeat (max 200 ms)
    4. leave instrumental gaps wordless, flag anything uncertain

Outputs: lyrics.lrc, lyrics.srt, mv_shots.json

Deliberate design choices
-------------------------
* This script is NOT wired into the instrumental gate in analyze_audio.py.
  A human handed us the words, so "is there a vocal?" is already answered.
* Heavy dependencies (demucs, torch, transformers) are optional and are only
  needed here. When they are missing we say exactly what is missing and stop.
  We NEVER fall back to Whisper transcription -- that would rewrite the words.
* Structure tags ([Verse], [Chorus]) are stripped before alignment and restored
  afterwards, so the aligner is never asked to find acoustics for the word
  "Chorus".
* `analysis.json` is optional and is used ONLY for its beat grid. Without it the
  alignment still works, it just does not snap.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lyric_utils import parse_lyric_sheet, strip_structure_tags  # noqa: E402
from shot_builder import (  # noqa: E402
    SHOT_LIST_CONTRACT,
    build_energy_lookup,
    build_shots,
    to_lrc,
    to_srt,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_MISSING_DEP = 2
EXIT_FAILED = 3


# ---------------------------------------------------------------------------
# Dependency reporting -- explicit, never silently degraded
# ---------------------------------------------------------------------------
def _module_available(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def check_dependencies(need_demucs=True, need_aligner=True, need_romanize=False):
    """
    Return a list of human-readable problems. Empty list == ready to run.
    """
    problems = []

    if need_demucs:
        if shutil.which("demucs") is None and not _module_available("demucs"):
            problems.append(
                "Demucs is required to isolate the vocal stem.\n"
                "    install:  pip install demucs\n"
                "    or pass an already separated stem with --vocals path/to/vocals.wav"
            )

    if need_aligner:
        if not _module_available("torch"):
            problems.append("PyTorch is required for CTC forced alignment.\n    install:  pip install torch")
        if not _module_available("transformers"):
            problems.append("transformers is required for the CTC acoustic model.\n    install:  pip install transformers")
        if not _module_available("torchaudio"):
            problems.append("torchaudio is required (forced_align / merge_tokens).\n    install:  pip install torchaudio")

    if need_romanize:
        if not _module_available("uroman"):
            problems.append(
                "uroman is required to romanise non-Latin lyrics before alignment.\n"
                "    install:  pip install uroman"
            )

    return problems


def _looks_non_latin(text):
    for ch in text or "":
        if ord(ch) > 0x2000:
            return True
    return False


def _section_map(line_texts, flat_lines):
    """
    Return the section label for each aligned line, in order.

    Positional mapping is used when the counts line up, because the parser and
    the tag stripper walk the same lines in the same order and skip the same
    ones. Otherwise we fall back to a text search, which is fragile with
    repeated chorus lines but better than returning nothing.
    """
    if len(line_texts) == len(flat_lines):
        return [entry.get("section") for entry in flat_lines]

    sections = []
    for text in line_texts:
        needle = text.strip()
        match = next((e.get("section") for e in flat_lines
                      if (e.get("text") or "").strip() == needle), None)
        sections.append(match)
    return sections


# ---------------------------------------------------------------------------
# Step 1 -- vocal separation
# ---------------------------------------------------------------------------
def separate_vocals(audio_path, out_dir, model="htdemucs", device=None):
    """
    Run Demucs and return the path to the vocal stem.

    Raises RuntimeError with the exact command when separation fails.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
           "-n", model, "-o", out_dir]
    if device:
        cmd += ["-d", device]
    cmd.append(audio_path)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Demucs vocal separation failed.\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr:  {(proc.stderr or '')[-600:]}"
        )

    track = os.path.splitext(os.path.basename(audio_path))[0]
    candidates = [
        os.path.join(out_dir, model, track, "vocals.wav"),
        os.path.join(out_dir, model, track, "vocals.mp3"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise RuntimeError(
        f"Demucs finished but no vocal stem was found. Looked in: {candidates}"
    )


# ---------------------------------------------------------------------------
# Step 2 -- CTC forced alignment of the KNOWN lyric sheet
# ---------------------------------------------------------------------------
def _romanize_lines(lines):
    try:
        import uroman
        u = uroman.Uroman()
        return [u.romanize_string(l) for l in lines]
    except Exception as exc:
        raise RuntimeError(
            "Non-Latin lyrics need romanisation before alignment and uroman "
            f"is unavailable: {exc}\n    install:  pip install uroman"
        )


def _load_aligner(model_id, device, language=None):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()

    if language and hasattr(model, "load_adapter"):
        # MMS-1b-all style language adapters.
        try:
            model.load_adapter(language)
        except Exception:
            pass

    blank_id = 0
    try:
        tok = processor.tokenizer
        if getattr(tok, "pad_token_id", None) is not None:
            blank_id = int(tok.pad_token_id)
    except Exception:
        pass

    return processor, model, blank_id


def _line_token_ids(tokenizer, lines):
    """
    Tokenise every line and remember which token positions belong to which line.

    Unknown characters are dropped, and we report how many, because silently
    dropping a character shifts every subsequent alignment.
    """
    all_ids = []
    line_ranges = []
    dropped = 0
    vocab = getattr(tokenizer, "get_vocab", lambda: {})() or {}

    for line in lines:
        ids = tokenizer(line, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        kept = []
        for i in ids:
            if vocab and i not in vocab.values():
                dropped += 1
                continue
            kept.append(int(i))
        start = len(all_ids)
        all_ids.extend(kept)
        line_ranges.append((start, len(all_ids), dropped))

    return all_ids, line_ranges, dropped


def forced_align_lines(vocals_path, lines, model_id, device="cpu", language=None,
                       max_seconds=None):
    """
    Align each line of the known lyric sheet against the vocal stem.

    Returns [{"start": float, "end": float, "score": float} or None, ...]
    one entry per input line, in input order. Lines with no alignment come back
    as None so the caller can flag them instead of inventing a time.
    """
    import numpy as np
    import torch
    import torchaudio
    import torchaudio.functional as AF

    try:
        import soundfile as sf
    except Exception:
        sf = None

    target_sr = 16000
    if sf is not None:
        wav, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != target_sr:
            mono = torchaudio.functional.resample(
                torch.from_numpy(mono), sr, target_sr).numpy()
    else:
        import librosa
        mono, target_sr = librosa.load(vocals_path, sr=target_sr, mono=True)

    if max_seconds:
        mono = mono[: int(max_seconds * target_sr)]

    if mono.size < target_sr:
        raise RuntimeError("Vocal stem is shorter than one second.")

    processor, model, blank_id = _load_aligner(model_id, device, language)

    waveform = torch.from_numpy(np.ascontiguousarray(mono)).float()
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)

    with torch.no_grad():
        logits = model(waveform.unsqueeze(0).to(device)).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    if log_probs.shape[1] < 2:
        raise RuntimeError("Acoustic model produced too few frames to align.")

    ratio = waveform.shape[0] / log_probs.shape[1]

    all_ids, line_ranges, dropped = _line_token_ids(processor.tokenizer, lines)
    if not all_ids:
        raise RuntimeError(
            "No alignable tokens were produced from the lyric sheet. "
            "Check the language / model pairing."
        )
    if dropped:
        print(f"  warning: {dropped} character(s) were not in the model vocabulary "
              f"and were dropped before alignment.", file=sys.stderr)

    targets = torch.tensor([all_ids], dtype=torch.int32, device=device)
    alignments, scores = AF.forced_align(log_probs, targets, blank=blank_id)
    al = alignments[0].cpu().numpy()
    sc = scores[0].cpu().numpy()

    def frame_to_sec(frame):
        return float(frame * ratio / target_sr)

    results = []
    for (start_idx, end_idx, _) in line_ranges:
        if end_idx <= start_idx:
            results.append(None)
            continue
        frames = np.where((al >= start_idx) & (al < end_idx))[0]
        if frames.size == 0:
            results.append(None)
            continue
        s_frame, e_frame = int(frames[0]), int(frames[-1]) + 1
        line_scores = sc[(al >= start_idx) & (al < end_idx)]
        results.append({
            "start": round(frame_to_sec(s_frame), 3),
            "end": round(frame_to_sec(e_frame), 3),
            "score": round(float(np.mean(line_scores)) if line_scores.size else 0.0, 4),
        })

    return results


# ---------------------------------------------------------------------------
# Step 3 -- vocal activity regions
# ---------------------------------------------------------------------------
def vocal_regions(vocals_path, threshold_db=-42.0, min_gap=0.6, min_segment=0.25,
                  target_sr=16000):
    """
    Energy-based vocal activity segments. Used only to flag lines that were
    aligned onto a section with no vocal energy, and to carve wordless shots.
    """
    import numpy as np
    import librosa

    y, sr = librosa.load(vocals_path, sr=target_sr, mono=True)
    if y.size == 0:
        return []

    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9, ref=1.0)

    active = db > threshold_db
    times = librosa.frames_to_time(np.arange(len(active)), sr=sr, hop_length=hop)

    segments = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = times[i]
        elif not flag and start is not None:
            segments.append({"start": float(start), "end": float(times[i])})
            start = None
    if start is not None:
        segments.append({"start": float(start), "end": float(times[-1])})

    merged = []
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] < min_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    return [s for s in merged if s["end"] - s["start"] >= min_segment]


# ---------------------------------------------------------------------------
# Beat grid
# ---------------------------------------------------------------------------
def beats_from_analysis(analysis_path):
    if not analysis_path or not os.path.exists(analysis_path):
        return [], [], None
    try:
        with open(analysis_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return [], [], None

    rhythm = data.get("rhythm") or {}
    beats = rhythm.get("beat_times_sec") or []
    downbeats = rhythm.get("downbeat_times_sec") or []
    duration = (data.get("file_info") or {}).get("duration_sec")
    if not beats:
        beats = rhythm.get("beat_times_first16") or []
    return list(beats), list(downbeats), duration


def beats_from_audio(audio_path):
    import librosa
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    times = [round(float(t), 3) for t in librosa.frames_to_time(beat_frames, sr=sr)]
    downbeats = times[::4]
    return times, downbeats, float(librosa.get_duration(y=y, sr=sr))


# ---------------------------------------------------------------------------
# Optional verification pass (read-only diff, never rewrites timings)
# ---------------------------------------------------------------------------
def whisper_diff(vocals_path, lyric_lines):
    """
    Read-only transcription used ONLY to report extra/missing lines.
    It never writes to the timeline. Returns None when unavailable.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return {"available": False,
                "reason": "faster-whisper not installed; skipped verification pass."}

    try:
        model = WhisperModel("small", device="cpu", compute_type="int8",
                             download_root=os.path.join(os.path.dirname(vocals_path),
                                                        ".whisper_models"))
        seg_iter, _ = model.transcribe(vocals_path, beam_size=5, vad_filter=True)
        heard = [s.text.strip() for s in seg_iter if s.text and s.text.strip()]
    except Exception as exc:
        return {"available": False, "reason": f"verification pass failed: {exc}"}

    def norm(s):
        return "".join(ch.lower() for ch in s if ch.isalnum())

    heard_norm = {norm(h) for h in heard}
    written_norm = {norm(l) for l in lyric_lines}

    missing = [l for l in lyric_lines if norm(l) and norm(l) not in heard_norm]
    extra = [h for h in heard if norm(h) and norm(h) not in written_norm]

    return {
        "available": True,
        "note": "Read-only check. Timings below were NOT modified by this pass.",
        "possibly_missing_from_recording": missing[:20],
        "possibly_extra_in_recording": extra[:20],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a locked lyric timeline (LRC + SRT + shot list) from a "
                    "finished song and its known lyric sheet.",
    )
    ap.add_argument("audio", help="finished song (mp3/wav/flac/...)")
    ap.add_argument("lyrics", help="lyric sheet, structure tags allowed")
    ap.add_argument("analysis", nargs="?", default=None,
                    help="optional analysis.json (used only for the beat grid)")
    ap.add_argument("--out-dir", default=".", help="where to write outputs (default: .)")
    ap.add_argument("--vocals", default=None,
                    help="already separated vocal stem; skips Demucs")
    ap.add_argument("--demucs-model", default="htdemucs")
    ap.add_argument("--model", default="facebook/mms-300m",
                    help="CTC acoustic model for forced alignment")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--language", default=None,
                    help="MMS language adapter, e.g. 'cmn' for Mandarin")
    ap.add_argument("--from-lrc", default=None,
                    help="skip alignment; use an existing LRC for line timings")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="align only the first N seconds (debugging)")
    ap.add_argument("--min-shot", type=float, default=1.5)
    ap.add_argument("--max-shot", type=float, default=6.0)
    ap.add_argument("--snap-sec", type=float, default=0.2,
                    help="maximum shift when snapping a line head to the grid")
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--verify-whisper", action="store_true",
                    help="run a read-only transcription diff (never changes timings)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.audio):
        print(f"error: audio not found: {args.audio}", file=sys.stderr)
        return EXIT_USAGE
    if not os.path.exists(args.lyrics):
        print(f"error: lyric sheet not found: {args.lyrics}", file=sys.stderr)
        return EXIT_USAGE

    with open(args.lyrics, "r", encoding="utf-8") as fh:
        raw_lyrics = fh.read()

    parsed = parse_lyric_sheet(raw_lyrics)
    flat = parsed.get("flat_lines") or []
    if not flat:
        print("error: no lyric lines found in the sheet.", file=sys.stderr)
        return EXIT_USAGE

    clean_text = strip_structure_tags(raw_lyrics)
    clean_lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

    # ---- dependencies -----------------------------------------------------
    need_aligner = args.from_lrc is None
    need_demucs = args.vocals is None and args.from_lrc is None
    need_romanize = need_aligner and _looks_non_latin(clean_text)

    problems = check_dependencies(need_demucs=need_demucs,
                                  need_aligner=need_aligner,
                                  need_romanize=need_romanize)
    if problems:
        print("Cannot build a lyric timeline -- missing dependencies:\n", file=sys.stderr)
        for p in problems:
            print(f"  - {p}\n", file=sys.stderr)
        print("Refusing to fall back to Whisper transcription: that would rewrite "
              "your lyrics. Install the missing pieces above, or pass "
              "--from-lrc with an existing timing file.", file=sys.stderr)
        return EXIT_MISSING_DEP

    # ---- vocal stem -------------------------------------------------------
    vocals_path = args.vocals
    if need_aligner and vocals_path is None:
        print("Step 1/4  separating vocals (Demucs)...")
        try:
            vocals_path = separate_vocals(
                args.audio, os.path.join(args.out_dir, "_stems"),
                model=args.demucs_model, device=args.device,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_FAILED
        print(f"           vocal stem: {vocals_path}")

    # ---- alignment --------------------------------------------------------
    if args.from_lrc:
        print("Step 2/4  reading line timings from the supplied LRC...")
        lrc_entries = []
        with open(args.from_lrc, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("["):
                    continue
                try:
                    stamp, text = line[1:].split("]", 1)
                    mm, ss = stamp.split(":")
                    lrc_entries.append({"start": int(mm) * 60 + float(ss),
                                        "text": text.strip()})
                except Exception:
                    continue
        if not lrc_entries:
            print("error: no usable timestamps found in the LRC file.", file=sys.stderr)
            return EXIT_USAGE
        aligned = []
        for i, e in enumerate(lrc_entries):
            end = (lrc_entries[i + 1]["start"] if i + 1 < len(lrc_entries)
                   else e["start"] + 3.0)
            aligned.append({"start": e["start"], "end": end, "score": None})
        line_texts = [e["text"] for e in lrc_entries]
    else:
        print(f"Step 2/4  CTC forced alignment ({args.model}) on your own lyrics...")
        lines_for_align = clean_lines
        if need_romanize:
            lines_for_align = _romanize_lines(clean_lines)
        try:
            aligned = forced_align_lines(
                vocals_path, lines_for_align, args.model,
                device=args.device, language=args.language,
                max_seconds=args.max_seconds,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_FAILED
        line_texts = clean_lines

    # ---- restore the section labels the aligner never saw ------------------
    # Structure tags were stripped before alignment, so they have to be put
    # back. Positional mapping is exact because both the parser and the
    # stripper walk the same lines in the same order and skip the same ones.
    sections = _section_map(line_texts, flat)

    aligned_lines = [{"text": t, "section": sections[i]}
                     for i, t in enumerate(line_texts)]

    # ---- assemble ---------------------------------------------------------
    merged = []
    for i, line in enumerate(aligned_lines):
        a = aligned[i] if i < len(aligned) else None
        merged.append({
            "section": line.get("section"),
            "text": line.get("text"),
            "start": a["start"] if a else None,
            "end": a["end"] if a else None,
            "score": a["score"] if a else None,
        })

    unaligned = sum(1 for m in merged if m["start"] is None)
    if unaligned:
        print(f"           {unaligned} line(s) could not be aligned and will be "
              f"flagged for manual review.", file=sys.stderr)

    print("Step 3/4  snapping line heads to the beat grid...")
    beat_times, downbeats, duration = beats_from_analysis(args.analysis)
    grid_source = "analysis.json"
    if not beat_times:
        try:
            beat_times, downbeats, duration = beats_from_audio(args.audio)
            grid_source = "librosa beat tracking"
        except Exception as exc:
            print(f"           no beat grid available ({exc}); alignment only.", file=sys.stderr)
            grid_source = "none"

    regions = []
    if vocals_path:
        try:
            regions = vocal_regions(vocals_path)
        except Exception as exc:
            print(f"           vocal region detection failed ({exc}); skipping.",
                  file=sys.stderr)

    energy_curve = []
    if args.analysis and os.path.exists(args.analysis):
        try:
            with open(args.analysis, "r", encoding="utf-8") as fh:
                energy_curve = (json.load(fh).get("energy_curve") or [])
        except Exception:
            energy_curve = []

    shots = build_shots(
        merged,
        beat_times=[] if args.no_snap else beat_times,
        downbeat_times=[] if args.no_snap else downbeats,
        vocal_regions=regions,
        total_duration=duration,
        energy_lookup=build_energy_lookup(energy_curve),
        min_shot=args.min_shot,
        max_shot=args.max_shot,
        snap_sec=args.snap_sec,
    )

    if not shots:
        print("error: no shots could be built from the alignment.", file=sys.stderr)
        return EXIT_FAILED

    # ---- outputs ----------------------------------------------------------
    print("Step 4/4  writing outputs...")
    os.makedirs(args.out_dir, exist_ok=True)
    lrc_path = os.path.join(args.out_dir, "lyrics.lrc")
    srt_path = os.path.join(args.out_dir, "lyrics.srt")
    shots_path = os.path.join(args.out_dir, "mv_shots.json")

    with open(lrc_path, "w", encoding="utf-8") as fh:
        fh.write(to_lrc(shots))
    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write(to_srt(shots))

    verification = None
    if args.verify_whisper and vocals_path:
        verification = whisper_diff(vocals_path, clean_lines)

    payload = {
        "contract": SHOT_LIST_CONTRACT,
        "source": {
            "audio": os.path.abspath(args.audio),
            "lyrics": os.path.abspath(args.lyrics),
            "analysis": os.path.abspath(args.analysis) if args.analysis else None,
            "vocal_stem": os.path.abspath(vocals_path) if vocals_path else None,
            "alignment_model": None if args.from_lrc else args.model,
            "beat_grid_source": grid_source,
            "snapping": not args.no_snap,
        },
        "rules": {
            "min_shot_sec": args.min_shot,
            "max_shot_sec": args.max_shot,
            "max_snap_sec": args.snap_sec,
        },
        "shot_count": len(shots),
        "needs_review_count": sum(1 for s in shots if s.get("needs_review")),
        "shots": shots,
    }
    if verification:
        payload["verification"] = verification

    with open(shots_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    review = payload["needs_review_count"]
    print(f"\n  lyrics.lrc     {len([s for s in shots if s['lyric']])} timed lines")
    print(f"  lyrics.srt     subtitles")
    print(f"  mv_shots.json  {len(shots)} shots"
          + (f", {review} need review" if review else ""))
    if review:
        print("\n  Lines flagged for manual review:")
        for s in shots:
            if s.get("needs_review"):
                print(f"    {s['start']:7.2f}s  {s.get('lyric')}   [{'; '.join(s['review_reasons'])}]")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
