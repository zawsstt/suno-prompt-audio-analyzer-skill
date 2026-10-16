#!/usr/bin/env python3
"""
vocal_gate.py -- decide vocal vs instrumental WITHOUT looking at lyrics.

Why this module exists
----------------------
v7.2 gated Whisper on `instrumentalness >= 0.7`, but `instrumentalness` was
computed as `0.95 whenever has_lyrics is False`. Since `has_lyrics` only
becomes True after Whisper runs, the score was always 0.95, Whisper never ran,
and the vocal branch was unreachable. Every song was treated as instrumental.

The fix is structural:
  * the decision is made from ACOUSTIC evidence only,
  * it never reads `has_lyrics`, `vocal_density` or anything produced by
    transcription,
  * it returns one of three states, and only `instrumental` may skip Whisper.

`unknown` is the safe default. It routes to the vocal branch. This module is
deliberately conservative: a false "instrumental" loses the whole lyric
pipeline, while a false "vocal" merely costs one Whisper pass.
"""

PRESENCE_VOCAL = "vocal"
PRESENCE_INSTRUMENTAL = "instrumental"
PRESENCE_UNKNOWN = "unknown"

# Score bands. Anything between the two thresholds is `unknown` and MUST run
# Whisper. These are deliberately wide.
VOCAL_THRESHOLD = 0.60
INSTRUMENTAL_THRESHOLD = 0.35

# Weights for the composite acoustic score.
_W_BAND_RATIO = 0.20
_W_VOICED_FRACTION = 0.55
_W_TONALITY = 0.25

# Weights when an explicit VAD backend is available -- the VAD signal is
# strong enough to dominate.
_W_VAD = 0.75
_W_ACOUSTIC_REST = 0.25


def _clip01(x):
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def decide_vocal_presence(features,
                          vocal_threshold=VOCAL_THRESHOLD,
                          instrumental_threshold=INSTRUMENTAL_THRESHOLD):
    """
    Pure decision function -- no audio, no I/O. Fully unit-testable.

    `features` keys (all optional; missing keys never default to "instrumental"):
        vocal_band_ratio     float  median (250-4000 Hz) / (full band) frame energy
        voiced_frame_fraction float fraction of frames with a pitched 80-1100 Hz track
        band_tonality        float 1.0 = highly tonal, 0.0 = noise-like
        vad_speech_ratio     float fraction of speech frames from a real VAD backend
        backend              str   which feature extractor produced this

    Returns:
        {
          "presence": "vocal" | "instrumental" | "unknown",
          "score": float | None,
          "confidence": float,
          "evidence": {...},
          "reason": str,
          "decided_by": "vad" | "acoustic" | "insufficient_evidence"
        }
    """
    evidence = {
        "vocal_band_ratio": features.get("vocal_band_ratio"),
        "voiced_frame_fraction": features.get("voiced_frame_fraction"),
        "band_tonality": features.get("band_tonality"),
        "vad_speech_ratio": features.get("vad_speech_ratio"),
        "backend": features.get("backend"),
    }

    if features.get("insufficient_audio"):
        return {
            "presence": PRESENCE_UNKNOWN,
            "score": None,
            "confidence": 0.0,
            "evidence": evidence,
            "reason": "Not enough audio to judge vocal presence; escalating to transcription.",
            "decided_by": "insufficient_evidence",
        }

    vad = _clip01(features.get("vad_speech_ratio"))
    band = _clip01(features.get("vocal_band_ratio"))
    voiced = _clip01(features.get("voiced_frame_fraction"))
    tonal = _clip01(features.get("band_tonality"))

    if vad is not None:
        # A real VAD backend measured speech. Singing is voiced speech, so a
        # high speech ratio is strong evidence for a vocal track. A very low
        # ratio is NOT evidence of instrumental (VAD often misses sustained
        # singing), so we fall back to the acoustic score for the low end.
        acoustic = None
        if voiced is not None and tonal is not None:
            acoustic = _W_BAND_RATIO * (band if band is not None else 0.5) \
                + _W_VOICED_FRACTION * voiced + _W_TONALITY * tonal
        if vad >= 0.45:
            score = _W_VAD * vad + _W_ACOUSTIC_REST * (acoustic if acoustic is not None else vad)
            return {
                "presence": PRESENCE_VOCAL,
                "score": round(score, 4),
                "confidence": round(min(1.0, 0.5 + vad), 4),
                "evidence": evidence,
                "reason": f"VAD reported speech in {vad:.0%} of frames.",
                "decided_by": "vad",
            }
        if vad <= 0.05 and acoustic is not None and acoustic <= instrumental_threshold:
            return {
                "presence": PRESENCE_INSTRUMENTAL,
                "score": round(acoustic, 4),
                "confidence": round(min(1.0, 0.5 + (instrumental_threshold - acoustic)), 4),
                "evidence": evidence,
                "reason": "VAD found no speech and the acoustic profile is non-vocal.",
                "decided_by": "vad",
            }
        return {
            "presence": PRESENCE_UNKNOWN,
            "score": round(acoustic, 4) if acoustic is not None else None,
            "confidence": 0.0,
            "evidence": evidence,
            "reason": "VAD was inconclusive (low speech ratio, acoustic profile ambiguous).",
            "decided_by": "vad",
        }

    # ---- acoustic-only path -------------------------------------------------
    missing = [k for k, v in (("voiced_frame_fraction", voiced),
                              ("band_tonality", tonal)) if v is None]
    if missing:
        return {
            "presence": PRESENCE_UNKNOWN,
            "score": None,
            "confidence": 0.0,
            "evidence": evidence,
            "reason": "Missing acoustic features (%s); escalating to transcription." % ", ".join(missing),
            "decided_by": "insufficient_evidence",
        }

    score = (_W_BAND_RATIO * (band if band is not None else 0.5)
             + _W_VOICED_FRACTION * voiced
             + _W_TONALITY * tonal)

    if score >= vocal_threshold:
        presence = PRESENCE_VOCAL
        reason = "Sustained pitched activity in the vocal range dominates the mix."
    elif score <= instrumental_threshold:
        presence = PRESENCE_INSTRUMENTAL
        reason = "No sustained pitched activity in the vocal range."
    else:
        presence = PRESENCE_UNKNOWN
        reason = "Acoustic profile is ambiguous; escalating to transcription."

    return {
        "presence": presence,
        "score": round(score, 4),
        "confidence": round(min(1.0, abs(score - 0.5) * 2), 4),
        "evidence": evidence,
        "reason": reason,
        "decided_by": "acoustic",
    }


# ---------------------------------------------------------------------------
# Feature extraction (requires librosa / scipy -- imported lazily)
# ---------------------------------------------------------------------------
def extract_vocal_features(y, sr, max_seconds=45.0, hop=512, n_fft=2048):
    """
    Measure acoustic vocal-likeness. Never looks at lyrics.

    Returns a feature dict suitable for `decide_vocal_presence`.
    """
    import numpy as np
    import librosa
    import scipy.signal

    if y is None or len(y) == 0:
        return {"insufficient_audio": True}

    limit = int(max_seconds * sr)
    seg = y[:limit] if len(y) > limit else y
    if len(seg) < sr:  # less than one second
        return {"insufficient_audio": True}

    def bandpass(sig, lo, hi):
        nyq = sr / 2.0
        lo_n = max(lo / nyq, 1e-3)
        hi_n = min(hi / nyq, 0.999)
        sos = scipy.signal.butter(4, [lo_n, hi_n], btype="band", output="sos")
        return scipy.signal.sosfilt(sos, sig)

    mid = bandpass(seg, 250.0, 4000.0)

    rms_all = librosa.feature.rms(y=seg, frame_length=n_fft, hop_length=hop)[0]
    rms_mid = librosa.feature.rms(y=mid, frame_length=n_fft, hop_length=hop)[0]
    n = min(len(rms_all), len(rms_mid))
    rms_all, rms_mid = rms_all[:n], rms_mid[:n]

    floor = float(np.percentile(rms_all, 20)) + 1e-9
    active = rms_all > max(floor * 1.5, 1e-6)
    if int(active.sum()) < 5:
        return {"insufficient_audio": True}

    ratio = rms_mid[active] / (rms_all[active] + 1e-9)
    vocal_band_ratio = float(np.median(ratio))

    flat = librosa.feature.spectral_flatness(y=mid, n_fft=n_fft, hop_length=hop)[0]
    flat = flat[:n]
    flat_med = float(np.median(flat[active]))
    # Voiced singing is strongly tonal: flatness around 0.001-0.02.
    # Broadband instrumental material sits near 0.05 and above.
    band_tonality = float(np.clip(1.0 - flat_med / 0.05, 0.0, 1.0))

    voiced_frame_fraction = None
    try:
        f0, voiced_flag, _ = librosa.pyin(
            mid, fmin=80.0, fmax=1100.0, sr=sr,
            frame_length=n_fft, hop_length=max(hop, 1024),
        )
        if voiced_flag is not None and len(voiced_flag) > 0:
            voiced_frame_fraction = float(np.mean(voiced_flag))
    except Exception:
        voiced_frame_fraction = None

    return {
        "vocal_band_ratio": vocal_band_ratio,
        "voiced_frame_fraction": voiced_frame_fraction,
        "band_tonality": band_tonality,
        "vad_speech_ratio": None,
        "backend": "librosa-acoustic",
        "insufficient_audio": False,
    }


def try_vad_speech_ratio(wav_path, sr=16000):
    """
    Optional Silero VAD backend. Returns None when unavailable -- we never
    substitute a fake number for a missing VAD.
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        import torch
        import soundfile as sf
        import numpy as np
    except Exception:
        return None

    try:
        data, file_sr = sf.read(wav_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if file_sr != sr:
            import librosa
            mono = librosa.resample(mono, orig_sr=file_sr, target_sr=sr)
        model = load_silero_vad()
        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(mono)), model,
            sampling_rate=sr, return_seconds=True,
        )
        total = len(mono) / float(sr)
        if total <= 0:
            return None
        spoken = sum(max(0.0, s["end"] - s["start"]) for s in stamps)
        return float(min(1.0, spoken / total))
    except Exception:
        return None
