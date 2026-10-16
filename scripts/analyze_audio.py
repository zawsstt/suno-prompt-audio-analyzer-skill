#!/usr/bin/env python3
"""
audio-analyzer v7.3 -- professional audio analysis for Suno prompt synthesis.

Usage: python3 analyze_audio.py <audio_file_path> [--vocal-presence vocal|instrumental|unknown]

What changed in v7.3 (see CHANGELOG in README.md)
------------------------------------------------
1.  Vocal gating is now acoustic and runs BEFORE transcription. The old gate
    read `instrumentalness`, which was itself derived from `has_lyrics`, which
    was only set after Whisper ran -- so it was always 0.95 and the vocal branch
    was unreachable. The gate now returns vocal / instrumental / unknown, and
    only a confident `instrumental` skips Whisper.
2.  `detect_mode()` requires a measured flat 2nd before it will say Phrygian.
    Natural minor (b6 + b7) resolves to Aeolian instead of being mislabelled.
3.  `chord_to_roman()` understands Essentia's `A:min` spelling as well as `Am`.
4.  Beat-interval variance is reported as rhythm STABILITY. A real swing ratio
    is computed separately, and is `null` when it cannot be measured.
5.  Drum density is computed over the whole window; onsets are no longer
    truncated to 24 before dividing by the bar count.
6.  The EQ reference is a per-frame median with a loudness-weighted view, so a
    loud chorus no longer erases a quiet verse.
7.  Temp files use unique paths under the system temp dir (no more fixed
    /tmp/_whisper_input.wav clobbering concurrent runs).
8.  `generate_suno_prompt()` is gone. Prompt compilation lives in exactly one
    place: compile_prompt.py. This script only produces evidence.
9.  Extra evidence is exported for the lyric aligner: full beat grid, downbeats,
    bass syncopation, swing ratio, modal evidence, Roman-numeral loop.
"""

import sys, json, os, tempfile, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vocal_gate import (  # noqa: E402
    PRESENCE_INSTRUMENTAL,
    PRESENCE_UNKNOWN,
    PRESENCE_VOCAL,
    decide_vocal_presence,
    extract_vocal_features,
    try_vad_speech_ratio,
)
from music_theory import (  # noqa: E402
    KEY_NAMES,
    chord_to_roman,
    detect_mode,
    detect_mode_detailed,
    detect_pentatonic,
    ks_key,
    parse_chord,
)

# A flat 2nd must reach this normalised chroma level, and beat the natural 2nd,
# before we are willing to call a minor key Phrygian. (Enforced in music_theory.)

try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


# ── Helper functions ──────────────────────────────────────────────────────────

def f0_to_note(f):
    import numpy as np
    if f is None or f <= 0 or np.isnan(f):
        return "?"
    midi = int(round(12 * np.log2(float(f) / 440) + 69))
    return f"{KEY_NAMES[midi % 12]}{midi // 12 - 1}"


def rle(seq):
    from itertools import groupby
    return [(k, sum(1 for _ in g)) for k, g in groupby(seq)]


LIBROSA_NATIVE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".mp3"}


def make_temp_wav_path(tag="analyzer"):
    """
    Unique temp path in the system temp directory.

    v7.2 used a fixed /tmp/_whisper_input.wav and tempfile.mktemp, so two
    concurrent analyses overwrote each other's audio.
    """
    fd, path = tempfile.mkstemp(prefix=f"{tag}_", suffix=".wav",
                                dir=tempfile.gettempdir())
    os.close(fd)
    return path


def ensure_wav(filepath):
    import subprocess
    ext = os.path.splitext(filepath)[1].lower()
    if ext in LIBROSA_NATIVE_EXTS:
        return filepath, False
    tmp = make_temp_wav_path("convert")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", filepath,
         "-vn", "-acodec", "pcm_s16le", "-ar", "48000", tmp],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {r.stderr.decode()[:300]}")
    return tmp, True


# ─────────────────────────────────────────────────────────────────────────────
# Essentia analysis engine
# ─────────────────────────────────────────────────────────────────────────────

def analyze_essentia(filepath):
    """
    Run Essentia primary analysis. Returns dict with essentia_features.
    Falls back gracefully on any sub-algorithm failure.
    """
    import numpy as np
    from collections import Counter
    result = {"available": True, "fallback_used": False}

    try:
        extractor = es.MusicExtractor(
            lowlevelStats=["mean", "stdev"],
            rhythmStats=["mean", "stdev"],
            tonalStats=["mean", "stdev"],
        )
        features, _ = extractor(filepath)
        result["bpm"] = round(float(features["rhythm.bpm"]), 2)
        result["lufs_integrated"] = round(float(features["lowlevel.loudness_ebu128.integrated"]), 2)
        try:
            result["lufs_range"] = round(float(features["lowlevel.loudness_ebu128.loudness_range"]), 2)
        except Exception:
            result["lufs_range"] = None
        result["danceability_raw"] = round(float(features["rhythm.danceability"]), 4)
        result["chords_key"] = str(features["tonal.chords_key"])
        result["chords_scale"] = str(features["tonal.chords_scale"])
    except Exception as e:
        result["music_extractor_error"] = str(e)
        result["fallback_used"] = True

    audio = None
    try:
        audio = es.MonoLoader(filename=filepath, sampleRate=44100)()
        key, scale, strength = es.KeyExtractor()(audio)
        result["key_extractor"] = {
            "key": str(key), "scale": str(scale), "strength": round(float(strength), 4),
        }
    except Exception as e:
        result["key_extractor_error"] = str(e)

    try:
        if audio is None:
            audio = es.MonoLoader(filename=filepath, sampleRate=44100)()
        bpm_r, _, _, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
        result["bpm_rhythm_extractor"] = round(float(bpm_r), 2)
        if "bpm" in result and abs(float(bpm_r) - result["bpm"]) < 20:
            result["bpm"] = round(float(bpm_r), 2)
        elif "bpm" not in result:
            result["bpm"] = round(float(bpm_r), 2)
    except Exception as e:
        result["rhythm_extractor_error"] = str(e)

    try:
        if audio is None:
            audio = es.MonoLoader(filename=filepath, sampleRate=44100)()

        windowing = es.Windowing(type="blackmanharris62")
        spectrum_algo = es.Spectrum()
        spectral_peaks = es.SpectralPeaks(
            orderBy="magnitude", magnitudeThreshold=1e-05,
            minFrequency=40, maxFrequency=5000,
        )
        hpcp_algo = es.HPCP()
        chord_det = es.ChordsDetection()

        hpcps = []
        for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=2048):
            win = windowing(frame)
            spec = spectrum_algo(win)
            freqs, mags = spectral_peaks(spec)
            hpcps.append(hpcp_algo(freqs, mags))

        if hpcps:
            chords, _ = chord_det(np.array(hpcps))
            counts = Counter(chords)
            total = len(chords)
            result["chord_histogram"] = {
                ch: round(cnt / total * 100, 1) for ch, cnt in counts.most_common(12)
            }
        else:
            result["chord_histogram"] = {}
    except Exception as e:
        result["chord_detection_error"] = str(e)
        result["chord_histogram"] = {}

    return result


def compute_chord_roman_analysis(chord_histogram, root_key, root_scale):
    out = {}
    for chord, pct in (chord_histogram or {}).items():
        roman = chord_to_roman(chord, root_key, root_scale)
        out[chord] = {"pct": pct, "roman": roman}
    return out


def compute_band_energies(y, sr):
    """
    Per-frame band energies with a robust and a perceptual reference.

    v7.2 took a single rfft over the whole file, so a loud chorus erased a quiet
    verse and there was no perceptual weighting at all.

    Reported per band:
        energy_pct             median across frames  <- the reference value
        energy_pct_loudweighted loudness-weighted mean
        energy_pct_global      the old whole-file number, kept for comparison
    """
    import numpy as np
    import librosa

    BANDS = [
        ("sub_bass_20_60hz", 20, 60),
        ("bass_60_250hz", 60, 250),
        ("low_mid_250_500hz", 250, 500),
        ("mid_500_2khz", 500, 2000),
        ("high_mid_2k_6khz", 2000, 6000),
        ("presence_6k_12khz", 6000, 10000),
        ("air_10k_plus", 10000, sr // 2 - 1),
    ]

    n_fft = 4096
    hop = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    frame_energy = S.sum(axis=0) + 1e-12
    weights = frame_energy / frame_energy.sum()

    global_mag = np.abs(np.fft.rfft(y))
    global_freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    global_total = float(np.sum(global_mag ** 2)) + 1e-12

    out = {}
    for name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[name] = {"energy_pct": 0.0, "energy_pct_loudweighted": 0.0,
                         "energy_pct_global": 0.0, "db": -120.0}
            continue
        band_frame = S[mask].sum(axis=0)
        pct_frame = band_frame / frame_energy * 100.0

        gmask = (global_freqs >= lo) & (global_freqs < hi)
        g_pct = float(np.sum(global_mag[gmask] ** 2) / global_total * 100.0) if gmask.any() else 0.0

        band_power = float(np.median(band_frame))
        db = float(librosa.amplitude_to_db(np.array([np.sqrt(max(band_power, 1e-12))]))[0])

        out[name] = {
            "energy_pct": round(float(np.median(pct_frame)), 2),
            "energy_pct_loudweighted": round(float(np.sum(pct_frame * weights)), 2),
            "energy_pct_global": round(g_pct, 2),
            "db": round(db, 1),
        }
    return out


def compute_spotify_like_features(essentia_feat, librosa_data, root_scale):
    """
    Estimate Spotify-like audio features (0.0-1.0 scale).

    v7.3: `instrumentalness` is NO LONGER derived from `has_lyrics`. Deriving it
    from a field that only exists after transcription made it a constant 0.95,
    which permanently closed the vocal branch.
    """
    centroid_hz = librosa_data.get("spectral_centroid_hz", 2000)
    brightness_n = min(1.0, centroid_hz / 8000)
    base_valence = 0.65 if root_scale == "major" else 0.30
    valence = round(min(1.0, max(0.0, base_valence + (brightness_n - 0.5) * 0.20)), 3)

    rms_db = librosa_data.get("rms_db", -20)
    rms_norm = min(1.0, max(0.0, (rms_db + 40) / 40))
    flux_norm = min(1.0, librosa_data.get("spectral_flux_mean", 0.0) * 50)
    energy = round(min(1.0, max(0.0, rms_norm * 0.6 + flux_norm * 0.4)), 3)

    ess_dance_raw = essentia_feat.get("danceability_raw", 1.0)
    danceability = round(min(1.0, max(0.0, ess_dance_raw / 3.0)), 3)

    harmonic_ratio = librosa_data.get("harmonic_ratio", 0.5)
    acousticness = round(min(1.0, max(0.0, harmonic_ratio * (1.0 - brightness_n))), 3)

    vocal_score = librosa_data.get("vocal_score")
    if vocal_score is None:
        instrumentalness = None
    else:
        instrumentalness = round(min(1.0, max(0.0, 1.0 - float(vocal_score))), 3)

    return {
        "valence": valence,
        "energy": energy,
        "danceability": danceability,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "_note": "instrumentalness is None when no acoustic vocal measurement exists; "
                 "it is never inferred from has_lyrics.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rhythm helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_swing_ratio(y, sr, beat_times, min_beats=6):
    """
    Real swing measurement, replacing the v7.2 beat-interval variance proxy.

    For each beat we look for exactly one subdivision onset strictly inside the
    beat and record its normalised position. A straight feel sits near 0.5; a
    swung feel pushes the subdivision later.

    Returns (swing_ratio, swing_position, reason).
        swing_ratio = pos / (1 - pos)   (2.0 == triplet swing)
    """
    import numpy as np
    import librosa

    if beat_times is None or len(beat_times) < min_beats + 1:
        return None, None, "too few beats"

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
    if len(onsets) < min_beats:
        return None, None, "too few onsets"

    positions = []
    for i in range(len(beat_times) - 1):
        b0, b1 = float(beat_times[i]), float(beat_times[i + 1])
        span = b1 - b0
        if span <= 0.05:
            continue
        inner = [t for t in onsets if b0 + 0.15 * span < t < b1 - 0.15 * span]
        if len(inner) != 1:
            continue
        positions.append((inner[0] - b0) / span)

    if len(positions) < min_beats:
        return None, None, f"only {len(positions)} measurable subdivisions"

    pos = float(np.median(positions))
    pos = min(max(pos, 0.05), 0.95)
    return round(pos / (1.0 - pos), 3), round(pos, 4), None


def compute_bass_syncopation_ratio(y, sr, beat_times, bpm):
    """
    Fraction of BASS onsets that fall clearly off the beat.

    v7.2 measured offbeats across ALL notes and called it "syncopated bass",
    so a busy hi-hat or a lead line could produce a bass claim.
    """
    import numpy as np
    import librosa
    import scipy.signal

    if not beat_times or bpm <= 0:
        return None

    sos = scipy.signal.butter(4, 250 / (sr / 2), btype="low", output="sos")
    y_bass = scipy.signal.sosfilt(sos, y)
    onsets = librosa.onset.onset_detect(y=y_bass, sr=sr, units="time", backtrack=False)
    if len(onsets) < 8:
        return None

    beat_len = 60.0 / bpm
    offbeat = 0
    for t in onsets:
        phase = (float(t) % beat_len) / beat_len
        if 0.3 < phase < 0.7:
            offbeat += 1
    return round(offbeat / len(onsets), 4)


def compute_drum_onsets(sig, sr, hop=256, min_dist=0.08):
    """Onset times for a band-limited percussion signal (full window)."""
    import numpy as np
    import librosa
    import scipy.signal

    frames = librosa.util.frame(sig, frame_length=hop * 2, hop_length=hop)
    rms_arr = np.sqrt(np.mean(frames ** 2, axis=0))
    thr = np.percentile(rms_arr, 80)
    peaks, _ = scipy.signal.find_peaks(rms_arr, height=thr,
                                       distance=max(1, int(sr * min_dist / hop)))
    return sorted(round(float(librosa.frames_to_time(p, sr=sr, hop_length=hop)), 3)
                  for p in peaks)


# ─────────────────────────────────────────────────────────────────────────────
# Main analyze function
# ─────────────────────────────────────────────────────────────────────────────

def analyze(filepath, vocal_presence_override=None, skip_whisper=False):
    import numpy as np, librosa, scipy.signal
    from scipy.ndimage import uniform_filter1d
    from collections import Counter
    import subprocess

    result = {}

    # ── 0a. Format normalisation ───────────────────────────────────────────────
    wav_path, is_temp_wav = ensure_wav(filepath)

    # ── 0b. File metadata ──────────────────────────────────────────────────────
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", filepath],
        capture_output=True, text=True)
    if probe.returncode == 0:
        meta = json.loads(probe.stdout)
        fmt = meta.get("format", {})
        astream = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), {})
        result["file_info"] = {
            "format": fmt.get("format_long_name", fmt.get("format_name", "?")),
            "duration_sec": round(float(fmt.get("duration", 0)), 2),
            "size_bytes": int(fmt.get("size", 0)),
            "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000, 1),
            "codec": astream.get("codec_name", "?"),
            "native_sample_rate_hz": int(astream.get("sample_rate", 0)),
            "channels": astream.get("channels", 0),
            "channel_layout": astream.get("channel_layout", "?"),
        }

    # ── 1. Load ────────────────────────────────────────────────────────────────
    SR = 22050
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    ch = result.get("file_info", {}).get("channels", 1)
    result["channel_analysis"] = {
        "type": "stereo" if ch >= 2 else "mono",
        "note": "Stereo correlation skipped for memory efficiency; use a DAW for precise M/S analysis",
    }

    # ── 2. Loudness & dynamics ─────────────────────────────────────────────────
    rms_v = float(np.sqrt(np.mean(y ** 2)))
    peak_v = float(np.max(np.abs(y)))
    rms_db = float(librosa.amplitude_to_db(np.array([rms_v + 1e-9]))[0])
    peak_db = float(librosa.amplitude_to_db(np.array([peak_v + 1e-9]))[0])
    lufs_approx = round(rms_db - 0.691, 2)
    crest_db = round(peak_db - rms_db, 2)
    hop_1s = sr
    rms_wins = [float(np.sqrt(np.mean(y[i:i + hop_1s] ** 2)))
                for i in range(0, len(y) - hop_1s, hop_1s)]
    rms_db_wins = librosa.amplitude_to_db(np.array(rms_wins) + 1e-9)
    dr = round(float(np.percentile(rms_db_wins, 95) - np.percentile(rms_db_wins, 5)), 2)
    result["loudness"] = {
        "rms_db": round(rms_db, 2),
        "peak_db": round(peak_db, 2),
        "lufs_approx": lufs_approx,
        "crest_factor_db": crest_db,
        "dynamic_range_db": dr,
        "headroom_db": round(-peak_db, 2),
        "compression_hint": ("heavily compressed" if crest_db < 8
                             else ("moderate" if crest_db < 14 else "wide dynamics")),
    }

    # ── 3. Frequency band energy (robust + perceptual) ─────────────────────────
    band_e = compute_band_energies(y, sr)
    result["frequency_bands"] = band_e
    result["frequency_bands_reference"] = (
        "energy_pct is the median across STFT frames; energy_pct_global is the old "
        "whole-file number kept for comparison. Loud sections no longer dominate."
    )

    # ── 4. Rhythm & beat grid ──────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm_librosa = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)

    if len(beat_times) > 4:
        ivs = np.diff(beat_times)
        cv = float(np.std(ivs) / np.mean(ivs)) if np.mean(ivs) > 0 else 0.0
        if cv < 0.03:
            stability = "steady"
        elif cv < 0.08:
            stability = "slightly variable"
        else:
            stability = "variable"
    else:
        cv = 0.0
        stability = "unknown"

    swing_ratio, swing_pos, swing_reason = compute_swing_ratio(y, sr, beat_times)
    if swing_ratio is None:
        swing_type = "unknown"
    elif swing_ratio >= 1.2:
        swing_type = "swung"
    elif swing_ratio <= 0.85:
        swing_type = "anticipatory"
    else:
        swing_type = "straight"

    oe = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(oe, max_size=sr // 2)
    tf = int(round(sr / (bpm_librosa / 60) / 512)) if bpm_librosa > 0 else 1
    beat4 = ac[tf * 4] if tf * 4 < len(ac) else 0
    beat3 = ac[tf * 3] if tf * 3 < len(ac) else 0
    time_sig = "4/4" if beat4 >= beat3 else "3/4"

    result["rhythm"] = {
        "tempo_bpm_librosa": round(bpm_librosa, 1),
        "beat_count": len(beat_times),
        "estimated_time_signature": time_sig,
        "rhythm_stability": stability,
        "rhythm_stability_note": "beat-interval coefficient of variation; this is timing "
                                 "steadiness, NOT swing",
        "beat_interval_cv": round(cv, 4),
        "swing_type": swing_type,
        "swing_ratio": swing_ratio,
        "swing_position": swing_pos,
        "swing_note": swing_reason or "median subdivision position inside each beat",
        "beat_times_first16": [round(float(t), 3) for t in beat_times[:16]],
        "beat_times_sec": [round(float(t), 3) for t in beat_times],
        "downbeat_times_sec": [round(float(t), 3) for t in beat_times[::4]],
        "polyrhythm": None,
    }

    # ── 5. Drum pattern ────────────────────────────────────────────────────────
    _, y_perc = librosa.effects.hpss(y)

    def bp_filter(sig, lo, hi, fs):
        nyq = fs / 2
        sos = scipy.signal.butter(4, [max(lo / nyq, 0.001), min(hi / nyq, 0.999)],
                                  btype="band", output="sos")
        return scipy.signal.sosfilt(sos, sig)

    bar_dur = 60.0 / bpm_librosa * 4 if bpm_librosa > 0 else 2.0
    end_s = min(bar_dur * 8, 20.0)
    end_smp = int(end_s * sr)

    kick_all = compute_drum_onsets(bp_filter(y_perc[:end_smp], 40, 120, sr), sr)
    snare_all = compute_drum_onsets(bp_filter(y_perc[:end_smp], 150, 500, sr), sr)
    hihat_all = compute_drum_onsets(
        bp_filter(y_perc[:end_smp], 5000, min(10000, sr // 2 - 100), sr), sr)

    bars = max(end_s / bar_dur, 1e-6)
    kick_pb = round(len(kick_all) / bars, 2)
    snare_pb = round(len(snare_all) / bars, 2)
    hihat_pb = round(len(hihat_all) / bars, 2)

    half_time = bool(1.5 <= kick_pb <= 2.5 and bpm_librosa >= 110)
    sixteenth_hats = bool(hihat_pb >= 8)

    result["drum_pattern"] = {
        "analysis_window": f"first {round(end_s, 1)}s (~8 bars)",
        "kick_onsets_sec": kick_all[:24],
        "snare_onsets_sec": snare_all[:24],
        "hihat_onsets_sec": hihat_all[:32],
        "kick_per_bar": kick_pb,
        "snare_per_bar": snare_pb,
        "hihat_per_bar": hihat_pb,
        "density_note": "counts use the FULL window; the onset lists above are truncated "
                        "for display only",
        "half_time": half_time,
        "sixteenth_hats": sixteenth_hats,
    }

    # ── 6. Bass line ───────────────────────────────────────────────────────────
    y90 = y[:int(90 * sr)]
    sos_lp = scipy.signal.butter(4, 300 / (sr / 2), btype="low", output="sos")
    y_bass = scipy.signal.sosfilt(sos_lp, y90)
    try:
        f0b, vb, _ = librosa.pyin(y_bass, fmin=30, fmax=300, sr=sr,
                                  frame_length=4096, hop_length=1024)
        bass_notes = [f0_to_note(f) for f in f0b[vb]]
        bass_seq = [{"note": k, "frames": v} for k, v in rle(bass_notes)[:24] if k != "?"]
        bass_top = [{"note": k, "count": v}
                    for k, v in Counter(n for n in bass_notes if n != "?").most_common(8)]
    except Exception as e:
        bass_seq, bass_top = [], [{"error": str(e)}]
    result["bassline"] = {"dominant_bass_notes": bass_top, "bass_sequence_start": bass_seq}
    del y_bass, y90

    result["rhythm"]["bass_syncopation_ratio"] = compute_bass_syncopation_ratio(
        y, sr, beat_times, bpm_librosa)

    # ── 7. Chord progression ───────────────────────────────────────────────────
    y_harm, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, bins_per_octave=36)
    beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
    chord_labels = [chord_label(beat_chroma[:, i]) for i in range(beat_chroma.shape[1])]
    chord_seq = [{"chord": k, "beats": v} for k, v in rle(chord_labels)[:32]]
    bigrams = [(chord_seq[i]["chord"], chord_seq[i + 1]["chord"])
               for i in range(len(chord_seq) - 1)]
    common_trans = [{"from": a, "to": b, "count": c} for (a, b), c in Counter(bigrams).most_common(8)]
    unique_chords = list(dict.fromkeys(c["chord"] for c in chord_seq))
    result["chord_progression"] = {
        "sequence": chord_seq[:24],
        "unique_chords_used": unique_chords[:12],
        "common_transitions": common_trans,
        "roman_loop": None,
    }

    # ── 8. Key, mode, scale degrees ────────────────────────────────────────────
    chroma_mean = chroma.mean(axis=1)
    root_name_ks, mode_ks, ks_score = ks_key(chroma_mean)
    root_idx_ks = KEY_NAMES.index(root_name_ks)
    modal, modal_evidence = detect_mode_detailed(chroma_mean, root_idx_ks)
    rel_idx = (root_idx_ks + 3) % 12 if mode_ks == "minor" else (root_idx_ks - 3) % 12
    relative = f"{KEY_NAMES[rel_idx]} {'major' if mode_ks == 'minor' else 'minor'}"
    rotated = np.roll(chroma_mean, -root_idx_ks)
    rotated = rotated / (rotated.max() + 1e-9)
    degree_labels = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7"]
    strong_deg = [degree_labels[i] for i in range(12) if rotated[i] > 0.5]

    result["tonality"] = {
        "key": f"{root_name_ks} {mode_ks}",
        "root": root_name_ks,
        "mode": mode_ks,
        "modal_flavor": modal,
        "modal_evidence": modal_evidence,
        "pentatonic": detect_pentatonic(strong_deg),
        "ks_confidence": ks_score,
        "relative_key": relative,
        "strong_scale_degrees": strong_deg,
        "chroma_profile": {k: round(float(v), 4) for k, v in zip(KEY_NAMES, chroma_mean)},
        "source": "librosa KS",
    }

    # Roman-numeral loop for the theory triggers (needs the key, so it runs here).
    roman_seq = []
    for entry in chord_seq[:12]:
        r = chord_to_roman(entry["chord"], root_name_ks, mode_ks)
        if r:
            roman_seq.append(r)
    result["chord_progression"]["roman_loop"] = roman_seq or None

    # ── 9. Key timeline ────────────────────────────────────────────────────────
    key_tl = []
    prev = None
    for i in range(int(np.ceil(duration / 15))):
        s0 = int(i * 15 * sr)
        s1 = int(min((i + 1) * 15 * sr, len(y)))
        if s1 - s0 < sr:
            continue
        cm = librosa.feature.chroma_cqt(y=y[s0:s1], sr=sr).mean(axis=1)
        rn, rm, rs = ks_key(cm)
        ks = f"{rn} {rm}"
        t0s = f"{int(i * 15 // 60):02d}:{int(i * 15 % 60):02d}"
        entry = {"time": t0s, "key": ks, "conf": rs}
        if ks != prev:
            entry["changed"] = True
            prev = ks
        key_tl.append(entry)
    result["key_timeline"] = key_tl

    # ── 10. Melodic contour ────────────────────────────────────────────────────
    y_harm90 = y_harm[:int(90 * sr)]
    try:
        f0m, vm, _ = librosa.pyin(y_harm90, fmin=80, fmax=2000, sr=sr,
                                  frame_length=2048, hop_length=512)
        mel_notes = [f0_to_note(f) for f in f0m[vm]]
        mel_seq = [{"note": k, "frames": v} for k, v in rle(mel_notes)[:32] if k != "?"]
        mel_top = [{"note": k, "count": v}
                   for k, v in Counter(n for n in mel_notes if n != "?").most_common(10)]
        valid_f0 = f0m[vm & ~np.isnan(f0m)]
        pr = {
            "lowest_note": f0_to_note(valid_f0.min()) if len(valid_f0) > 0 else "?",
            "highest_note": f0_to_note(valid_f0.max()) if len(valid_f0) > 0 else "?",
            "range_hz": (f"{round(float(valid_f0.min()), 1) if len(valid_f0) > 0 else 0}"
                         f"-{round(float(valid_f0.max()), 1) if len(valid_f0) > 0 else 0}"),
        }
    except Exception as e:
        mel_seq, mel_top, pr = [], [], {"error": str(e)}
    del y_harm90
    result["melodic_contour"] = {
        "pitch_range": pr,
        "dominant_melody_notes": mel_top,
        "melody_sequence_start": mel_seq,
    }

    # ── 11. Energy curve ───────────────────────────────────────────────────────
    ecurve, flux_vals = [], []
    for i in range(int(np.ceil(duration / 5))):
        s0 = int(i * 5 * sr)
        s1 = int(min((i + 1) * 5 * sr, len(y)))
        if s1 == s0:
            continue
        chunk = y[s0:s1]
        rv = float(np.sqrt(np.mean(chunk ** 2)))
        rd = float(librosa.amplitude_to_db(np.array([rv + 1e-9]))[0])
        spec = np.abs(librosa.stft(chunk, n_fft=512))
        flux = float(np.mean(np.diff(spec, axis=1) ** 2)) if spec.shape[1] > 1 else 0.0
        flux_vals.append(flux)
        ecurve.append({
            "time": f"{int(i * 5 // 60):02d}:{int(i * 5 % 60):02d}",
            "time_sec": float(i * 5),
            "rms_db": round(rd, 1),
            "spectral_flux": round(flux, 4),
        })
    result["energy_curve"] = ecurve
    spectral_flux_mean = float(np.mean(flux_vals)) if flux_vals else 0.0

    # ── 12. Structure ──────────────────────────────────────────────────────────
    rms_env = librosa.feature.rms(y=y, frame_length=sr // 2, hop_length=sr // 4)[0]
    rms_sm = uniform_filter1d(librosa.amplitude_to_db(rms_env + 1e-9), size=8)
    drops = []
    for i in range(4, len(rms_sm) - 4):
        if rms_sm[i] < rms_sm[i - 4] - 3 and rms_sm[i] < rms_sm[i + 4] - 3:
            drops.append(round(float(librosa.frames_to_time(i, sr=sr, hop_length=sr // 4)), 1))
    pk_idx, _ = scipy.signal.find_peaks(rms_sm, prominence=3, distance=8)
    pk_times = [round(float(librosa.frames_to_time(p, sr=sr, hop_length=sr // 4)), 1)
                for p in pk_idx]
    result["structure"] = {
        "energy_drop_times_sec": drops[:10],
        "energy_peak_times_sec": pk_times[:10],
        "tip": "Drops = likely section breaks; Peaks = climax candidates. "
               "Section names are only applied where this evidence exists.",
    }

    # ── 13. Spectral profile ───────────────────────────────────────────────────
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    sf = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    avg_c = float(np.mean(sc))
    result["spectral"] = {
        "centroid_hz": round(avg_c, 1),
        "bandwidth_hz": round(float(np.mean(sb)), 1),
        "rolloff_hz": round(float(np.mean(sro)), 1),
        "flatness": round(float(np.mean(sf)), 6),
        "zcr": round(float(np.mean(zcr)), 5),
        "brightness": "bright" if avg_c > 3000 else ("mid" if avg_c > 1000 else "dark"),
        "mfcc_means": [round(float(v), 3) for v in mfcc.mean(axis=1)],
        "mfcc_stds": [round(float(v), 3) for v in mfcc.std(axis=1)],
    }

    # ── 14. Harmonic / percussive ratio ────────────────────────────────────────
    he = float(np.mean(y_harm ** 2))
    pe = float(np.mean(y_perc ** 2))
    te = he + pe + 1e-12
    result["harmonic_percussive"] = {
        "harmonic_ratio": round(he / te, 4),
        "percussive_ratio": round(pe / te, 4),
        "dominant": "harmonic" if he > pe else "percussive",
    }

    # ── 15. Essentia ───────────────────────────────────────────────────────────
    ess_feat = {}
    if ESSENTIA_AVAILABLE:
        try:
            ess_feat = analyze_essentia(filepath)
        except Exception as e:
            ess_feat = {"available": False, "error": str(e)}
    else:
        ess_feat = {"available": False, "reason": "essentia not installed"}

    result["analysis_engine"] = {
        "primary": "essentia" if ess_feat.get("available") else "librosa",
        "secondary": "librosa",
        "fallback": not ess_feat.get("available", False),
    }

    ess_out = {"available": ess_feat.get("available", False)}
    if ess_feat.get("available"):
        for src, dst in (("bpm", "bpm"), ("bpm_rhythm_extractor", "bpm_rhythm_extractor"),
                         ("lufs_integrated", "lufs_integrated"), ("lufs_range", "lufs_range"),
                         ("key_extractor", "key_extractor"), ("chord_histogram", "chord_histogram")):
            if src in ess_feat:
                ess_out[dst] = ess_feat[src]
        if "danceability_raw" in ess_feat:
            ess_out["danceability"] = ess_feat["danceability_raw"]
        if "chords_key" in ess_feat:
            ess_out["chords_key"] = ess_feat["chords_key"]
            ess_out["chords_scale"] = ess_feat.get("chords_scale", "")

        if "key_extractor" in ess_feat:
            ke = ess_feat["key_extractor"]
            ess_root, ess_scale = ke["key"], ke["scale"]
            ess_root_idx = KEY_NAMES.index(ess_root) if ess_root in KEY_NAMES else root_idx_ks
            ess_modal, ess_modal_evidence = detect_mode_detailed(chroma_mean, ess_root_idx)
            ess_rel_idx = (ess_root_idx + 3) % 12 if ess_scale == "minor" else (ess_root_idx - 3) % 12
            result["tonality"].update({
                "key": f"{ess_root} {ess_scale}",
                "root": ess_root,
                "mode": ess_scale,
                "modal_flavor": ess_modal,
                "modal_evidence": ess_modal_evidence,
                "ks_confidence": ke["strength"],
                "relative_key": f"{KEY_NAMES[ess_rel_idx]} "
                                f"{'major' if ess_scale == 'minor' else 'minor'}",
                "source": "essentia KeyExtractor",
            })
            roman_seq = []
            for entry in chord_seq[:12]:
                r = chord_to_roman(entry["chord"], ess_root, ess_scale)
                if r:
                    roman_seq.append(r)
            result["chord_progression"]["roman_loop"] = roman_seq or None

        if ess_feat.get("chord_histogram"):
            final_root = result["tonality"]["root"]
            final_scale = result["tonality"]["mode"]
            ess_out["chord_roman_analysis"] = compute_chord_roman_analysis(
                ess_feat["chord_histogram"], final_root, final_scale)

    result["essentia_features"] = ess_out

    # ── 16. Vocal presence gate (BEFORE transcription) ─────────────────────────
    vocal_features = {}
    try:
        vocal_features = extract_vocal_features(y, sr)
    except Exception as e:
        vocal_features = {"insufficient_audio": True, "error": str(e)}

    if vocal_features.get("vad_speech_ratio") is None and wav_path:
        try:
            vad = try_vad_speech_ratio(wav_path)
            if vad is not None:
                vocal_features["vad_speech_ratio"] = vad
        except Exception:
            pass

    if vocal_presence_override in (PRESENCE_VOCAL, PRESENCE_INSTRUMENTAL, PRESENCE_UNKNOWN):
        vocal_presence = {
            "presence": vocal_presence_override,
            "score": None,
            "confidence": 1.0,
            "evidence": vocal_features,
            "reason": "explicit override from the caller",
            "decided_by": "override",
        }
    else:
        vocal_presence = decide_vocal_presence(vocal_features)

    result["vocal_presence"] = vocal_presence
    result["vocal_presence_features"] = vocal_features

    # ── 17. Genre scoring ──────────────────────────────────────────────────────
    sub_pct = band_e["sub_bass_20_60hz"]["energy_pct"]
    bass_pct = band_e["bass_60_250hz"]["energy_pct"]
    mid_pct = band_e["mid_500_2khz"]["energy_pct"]
    air_pct = band_e["air_10k_plus"]["energy_pct"]
    crest = result["loudness"]["crest_factor_db"]
    bpm = ess_feat.get("bpm", bpm_librosa) if ess_feat.get("available") else bpm_librosa
    result["rhythm"]["tempo_bpm"] = round(bpm, 1)
    lufs_val = ess_feat.get("lufs_integrated", lufs_approx) if ess_feat.get("available") else lufs_approx
    dance_raw = ess_feat.get("danceability_raw", 1.0) if ess_feat.get("available") else 1.0
    dance_norm = min(1.0, dance_raw / 3.0)
    final_scale = result["tonality"]["mode"]

    librosa_data_for_spotify = {
        "spectral_centroid_hz": avg_c,
        "rms_db": rms_db,
        "spectral_flux_mean": spectral_flux_mean,
        "harmonic_ratio": round(he / te, 4),
        "vocal_score": vocal_presence.get("score"),
    }
    spotify_features = compute_spotify_like_features(ess_feat, librosa_data_for_spotify, final_scale)
    result["spotify_like_features"] = spotify_features

    genre_scores = {}

    def add(g, score, reason=""):
        genre_scores[g] = genre_scores.get(g, 0) + score

    valence_est = spotify_features["valence"]
    energy_est = spotify_features["energy"]

    if 118 <= bpm <= 135:
        add("House / Deep House", 2, "bpm")
    if kick_pb >= 3.5:
        add("House / Deep House", 2, "four-on-floor kick")
    if 3 < sub_pct < 15:
        add("House / Deep House", 1, "moderate sub")
    if he / te > 0.5:
        add("House / Deep House", 1, "harmonic")
    if avg_c < 4000:
        add("House / Deep House", 1, "warm centroid")

    if 126 <= bpm <= 138:
        add("Tech House", 2, "bpm")
    if kick_pb >= 3.5:
        add("Tech House", 2, "tight kick")
    if avg_c > 3000:
        add("Tech House", 1, "bright")
    if hihat_pb >= 3:
        add("Tech House", 1, "busy hihat")
    if pe / te > 0.35:
        add("Tech House", 1, "percussive")

    if 126 <= bpm <= 145:
        add("Trance / Progressive", 2, "bpm")
    if avg_c > 3500:
        add("Trance / Progressive", 1, "bright")
    if he / te > 0.6:
        add("Trance / Progressive", 2, "harmonic build")
    if air_pct > 5:
        add("Trance / Progressive", 1, "airy highs")
    if kick_pb >= 3.5:
        add("Trance / Progressive", 1, "four-on-floor")

    if bpm >= 160:
        add("Drum & Bass / Breakbeat", 3, "high bpm")
    elif 140 <= bpm < 160:
        add("Drum & Bass / Breakbeat", 2, "bpm")
    if pe / te > 0.35:
        add("Drum & Bass / Breakbeat", 2, "percussive")
    if sub_pct > 5:
        add("Drum & Bass / Breakbeat", 1, "sub bass")
    if avg_c > 2500:
        add("Drum & Bass / Breakbeat", 1, "bright")

    if bpm >= 130:
        add("Techno / Hard Dance", 1, "high bpm")
    if bpm >= 140:
        add("Techno / Hard Dance", 2, "techno bpm")
    if avg_c > 4000:
        add("Techno / Hard Dance", 2, "harsh bright")
    if pe / te > 0.4:
        add("Techno / Hard Dance", 1, "industrial percussive")
    if kick_pb >= 3.5:
        add("Techno / Hard Dance", 1, "kick-driven")

    if 60 <= bpm <= 100:
        add("Trap / Hip-hop", 2, "bpm")
    elif 100 < bpm <= 140:
        add("Trap / Hip-hop", 1, "bpm half-time")
    if sub_pct > 10:
        add("Trap / Hip-hop", 3, "heavy sub")
    if hihat_pb >= 4:
        add("Trap / Hip-hop", 2, "trap hihats")
    if bass_pct > 30:
        add("Trap / Hip-hop", 1, "bass dominant")
    if kick_pb < 2:
        add("Trap / Hip-hop", 1, "sparse kick")

    if 70 <= bpm <= 100:
        add("Lo-fi / Boom-bap", 2, "bpm")
    if he / te > 0.65:
        add("Lo-fi / Boom-bap", 2, "harmonic")
    if avg_c < 2500:
        add("Lo-fi / Boom-bap", 2, "warm/muffled")
    if crest > 10:
        add("Lo-fi / Boom-bap", 1, "dynamic")
    if hihat_pb < 3:
        add("Lo-fi / Boom-bap", 1, "sparse hihat")

    if 60 <= bpm <= 110:
        add("R&B / Neo-soul", 1, "bpm")
    if he / te > 0.7:
        add("R&B / Neo-soul", 2, "harmonic")
    if avg_c < 3000:
        add("R&B / Neo-soul", 1, "warm")
    if sub_pct > 5:
        add("R&B / Neo-soul", 1, "bass presence")
    if mid_pct > 20:
        add("R&B / Neo-soul", 1, "midrange presence")

    if 100 <= bpm <= 130:
        add("Pop / Indie Pop", 1, "bpm")
    if 2000 < avg_c < 4000:
        add("Pop / Indie Pop", 2, "balanced bright")
    if he / te > 0.6:
        add("Pop / Indie Pop", 1, "melodic")
    if crest < 14:
        add("Pop / Indie Pop", 1, "compressed radio mix")
    if air_pct > 3:
        add("Pop / Indie Pop", 1, "polished highs")

    # Rock / Alternative
    if 90 <= bpm <= 170:
        add("Rock / Alternative", 1, "bpm")
    if 0.3 < he / te < 0.7:
        add("Rock / Alternative", 2, "guitar-band balance")
    if mid_pct > 22:
        add("Rock / Alternative", 2, "midrange guitar presence")
    if crest > 10:
        add("Rock / Alternative", 1, "live dynamics")
    if kick_pb >= 2 and hihat_pb >= 3:
        add("Rock / Alternative", 1, "acoustic drum kit")

    # Folk / Acoustic
    if he / te > 0.75 and avg_c < 3000:
        add("Folk / Acoustic", 2, "very harmonic, warm")
    if sub_pct < 3:
        add("Folk / Acoustic", 2, "no sub bass")
    if hihat_pb < 3 and kick_pb < 2:
        add("Folk / Acoustic", 1, "sparse percussion")
    if 70 <= bpm <= 120:
        add("Folk / Acoustic", 1, "bpm")

    # Jazz / Soul
    if he / te > 0.8 and 0.3 < dance_norm < 0.8:
        add("Jazz / Soul", 2, "harmonic with moderate groove")
    if mid_pct > 25 and air_pct < 3:
        add("Jazz / Soul", 1, "warm midrange, soft highs")
    if swing_ratio is not None and 1.1 <= swing_ratio <= 1.8:
        add("Jazz / Soul", 2, "measured swing feel")

    if bpm <= 90:
        add("Ambient / Cinematic", 1, "slow bpm")
    if he / te > 0.85:
        add("Ambient / Cinematic", 3, "very harmonic")
    if avg_c < 2000:
        add("Ambient / Cinematic", 2, "dark/soft centroid")
    if sub_pct < 3 and bass_pct < 20:
        add("Ambient / Cinematic", 1, "low bass presence")
    if crest > 12:
        add("Ambient / Cinematic", 1, "wide dynamics")
    if valence_est < 0.45 and dance_norm < 0.5:
        add("Ambient / Cinematic", 2, "low valence + low dance")
    if lufs_val < -18:
        add("Ambient / Cinematic", 2, "quiet master")

    if he / te > 0.80:
        add("Cinematic / Orchestral", 3, "very harmonic")
    if 800 < avg_c < 3500:
        add("Cinematic / Orchestral", 2, "orchestral centroid")
    if crest > 14:
        add("Cinematic / Orchestral", 2, "wide dynamics")
    if valence_est < 0.6:
        add("Cinematic / Orchestral", 1, "minor-leaning valence")
    if lufs_val < -14:
        add("Cinematic / Orchestral", 1, "dynamic master")
    if bpm < 130:
        add("Cinematic / Orchestral", 1, "non-dance bpm")
    if dance_norm < 0.5:
        add("Cinematic / Orchestral", 2, "low danceability")
    if kick_pb < 2:
        add("Cinematic / Orchestral", 1, "sparse kick")

    if 126 <= bpm <= 140:
        add("EDM / Big Room", 1, "bpm")
    if air_pct > 8:
        add("EDM / Big Room", 2, "bright/shiny highs")
    if avg_c > 4000:
        add("EDM / Big Room", 1, "bright")
    if kick_pb >= 3.5:
        add("EDM / Big Room", 1, "four-on-floor")
    if he / te > 0.5 and sub_pct > 5:
        add("EDM / Big Room", 2, "full spectrum")

    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    THRESHOLD = 4
    genres = [g for g, s in sorted_genres if s >= THRESHOLD]
    if not genres:
        genres = [g for g, s in sorted_genres[:2]]
    if not genres:
        genres = ["Electronic / Instrumental (mixed)"]
    genres = genres[:3]

    result["production_style"] = {
        "likely_genres": genres,
        "genre_scores": {g: s for g, s in sorted_genres[:8]},
        "mastering": ("heavily limited" if crest < 8
                      else ("radio-ready" if crest < 14 else "wide dynamic range")),
        "daw_tips": [
            f"Set project BPM to {round(bpm, 1)}, time signature {time_sig}",
            f"Program in {result['tonality']['root']} {result['tonality']['mode']} — "
            f"scale: {result['tonality']['modal_flavor']}",
            f"Reference key: {result['tonality']['relative_key']} (for modulations)",
            f"Sub bass ({round(sub_pct, 1)}% energy) — "
            f"{'boost sub' if sub_pct < 3 else 'sub already prominent'}",
            f"Mix brightness target: ~{round(avg_c)}Hz spectral centroid",
            f"Aim for ~{round(lufs_val, 1)} LUFS integrated loudness",
            f"Compression depth: crest factor {crest}dB → "
            f"{'use heavy parallel compression' if crest > 14 else 'moderate limiting'}",
        ],
    }

    # ── 18. Summary ────────────────────────────────────────────────────────────
    result["summary"] = {
        "duration": f"{int(duration // 60):02d}:{int(duration % 60):02d}",
        "key_full": f"{result['tonality']['root']} {result['tonality']['mode']} "
                    f"({result['tonality']['modal_flavor']})",
        "relative_key": result["tonality"]["relative_key"],
        "tempo_bpm": round(bpm, 1),
        "time_signature": time_sig,
        "rhythm_stability": stability,
        "swing_type": swing_type,
        "lufs": round(lufs_val, 2),
        "lufs_source": ("essentia EBU R128" if ess_feat.get("available") and
                        "lufs_integrated" in ess_feat else "librosa approx"),
        "lufs_approx": lufs_approx,
        "dynamic_range_db": dr,
        "brightness": result["spectral"]["brightness"],
        "harmonic_dominant": he > pe,
        "likely_genres": genres,
        "vocal_presence": vocal_presence.get("presence"),
    }

    # ── 19. Transcription (only when the gate does not say instrumental) ───────
    if skip_whisper:
        result["lyrics"] = {
            "has_lyrics": None,
            "skipped": True,
            "note": "Transcription skipped at the caller's request (--skip-whisper).",
        }
    elif vocal_presence.get("presence") == PRESENCE_INSTRUMENTAL:
        result["lyrics"] = {
            "has_lyrics": False,
            "is_instrumental": True,
            "skipped": True,
            "note": ("Acoustic gate classified this track as instrumental "
                     f"({vocal_presence.get('reason')}) — transcription skipped."),
        }
    else:
        result["lyrics"] = extract_lyrics(filepath)
        if vocal_presence.get("presence") == PRESENCE_UNKNOWN and \
                result["lyrics"].get("has_lyrics"):
            result["vocal_presence"]["presence"] = PRESENCE_VOCAL
            result["vocal_presence"]["resolved_by"] = "transcription confirmed vocals"
            result["summary"]["vocal_presence"] = PRESENCE_VOCAL

    # ── 20. Theory-tag evidence for the compiler ───────────────────────────────
    result["theory_evidence"] = {
        "modal_flavor": result["tonality"]["modal_flavor"],
        "modal_evidence": result["tonality"]["modal_evidence"],
        "roman_loop": result["chord_progression"]["roman_loop"],
        "pentatonic": result["tonality"]["pentatonic"],
        "bass_syncopation_ratio": result["rhythm"]["bass_syncopation_ratio"],
        "swing_ratio": result["rhythm"]["swing_ratio"],
        "half_time": result["drum_pattern"]["half_time"],
        "sixteenth_hats": result["drum_pattern"]["sixteenth_hats"],
        "polyrhythm": result["rhythm"]["polyrhythm"],
    }

    # ── 21. Prompt compilation is delegated to the single compiler ─────────────
    result["suno_prompt"] = _delegate_prompt_compilation(result)

    if is_temp_wav:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return result


def chord_label(chroma_frame, thr=0.35):
    import numpy as np
    c = chroma_frame / (chroma_frame.max() + 1e-9)
    root = int(np.argmax(c))
    m3, M3 = c[(root + 3) % 12], c[(root + 4) % 12]
    p5, d5, a5 = c[(root + 7) % 12], c[(root + 6) % 12], c[(root + 8) % 12]
    if M3 > thr and p5 > thr:
        return f"{KEY_NAMES[root]}"
    elif m3 > thr and p5 > thr:
        return f"{KEY_NAMES[root]}m"
    elif m3 > thr and d5 > thr:
        return f"{KEY_NAMES[root]}dim"
    elif M3 > thr and a5 > thr:
        return f"{KEY_NAMES[root]}aug"
    return KEY_NAMES[root]


def _delegate_prompt_compilation(result):
    """
    Call the ONE compiler. Prompt construction no longer lives in this file.

    v7.2 had `generate_suno_prompt()` here capped at 120/200 characters while
    compile_prompt.py documented 150-350, so the answer depended on which entry
    point the agent happened to use.
    """
    try:
        from compile_prompt import compile_suno_prompt
        return compile_suno_prompt(result)
    except SystemExit as exc:
        return {"error": str(exc),
                "hint": "Run compile_prompt.py directly for a full error report."}
    except Exception as exc:
        return {"error": f"prompt compilation failed: {exc}",
                "hint": "Analysis above is still complete; re-run compile_prompt.py."}


# ─────────────────────────────────────────────────────────────────────────────
# Lyrics extraction (faster-whisper)
# ─────────────────────────────────────────────────────────────────────────────

def extract_lyrics(path):
    import subprocess, re, collections

    tmp_wav = make_temp_wav_path("whisper")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_wav],
        capture_output=True)
    if r.returncode != 0:
        return {"error": "ffmpeg conversion failed", "detail": r.stderr.decode()[:300]}

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "faster-whisper not installed. Run: pip install faster-whisper"}

    model_root = os.path.join(tempfile.gettempdir(), "whisper_models")
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8",
                             download_root=model_root)
        segments_iter, info = model.transcribe(
            tmp_wav, beam_size=5, language=None,
            vad_filter=False, word_timestamps=False,
            no_speech_threshold=0.6, log_prob_threshold=-0.5)
        segments = list(segments_iter)
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

    noise = {"[Music]", "[MUSIC]", "(Music)", "Music", "MUSIC", "[Applause]",
             "[Silence]", "[silence]", "[inaudible]", "[Inaudible]",
             "[music]", "(music)", "[background music]", "[instrumental]"}
    lyric_segs = []
    for seg in segments:
        text = seg.text.strip()
        if text and text not in noise and len(text) > 3:
            ts = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
            lyric_segs.append({"time": ts, "start_sec": round(seg.start, 1), "text": text})

    if not lyric_segs:
        return {
            "language": info.language,
            "language_prob": round(info.language_probability, 2),
            "has_lyrics": False,
            "note": "No vocal content detected by transcription.",
        }

    full_text = " / ".join(s["text"] for s in lyric_segs)

    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "is", "are", "was", "were", "be", "been", "have", "has",
                 "had", "we", "i", "you", "he", "she", "it", "they", "this", "that",
                 "all", "just", "no", "not", "so", "when", "as", "if", "our", "their",
                 "my", "your", "its", "what", "there", "will", "can", "do", "dont",
                 "im", "set", "one", "like"}
    words = re.findall(r"[a-z']+", full_text.lower())
    word_freq = collections.Counter(w for w in words if w not in stopwords and len(w) > 2)
    top_keywords = [w for w, _ in word_freq.most_common(10)]

    line_texts = [s["text"].lower().strip() for s in lyric_segs]
    line_freq = collections.Counter(line_texts)
    repeated_lines = [l for l, c in line_freq.most_common(5) if c >= 2]

    total_dur_min = lyric_segs[-1]["start_sec"] / 60 if lyric_segs else 1
    vocal_density = round(len(lyric_segs) / max(total_dur_min, 0.1), 1)

    from suno_contract import language_tag
    lang_tag = language_tag(info.language)

    lyric_lower = full_text.lower()
    mood_keywords = {
        "dark": ["dark", "shadow", "dead", "death", "end", "destroy", "void", "silence", "fall"],
        "aggressive": ["fire", "strike", "charge", "attack", "breach", "fight", "violent"],
        "mysterious": ["static", "signal", "unspoken", "locked", "hidden"],
        "emotional": ["heart", "love", "tears", "feel", "soul", "hope", "dream"],
        "epic": ["structure", "falls", "prevails", "unity", "trust", "purpose", "complete"],
    }
    mood_scores = {m: sum(lyric_lower.count(kw) for kw in kws)
                   for m, kws in mood_keywords.items()}
    lyric_mood = max(mood_scores, key=mood_scores.get) if max(mood_scores.values()) > 0 else None

    return {
        "language": info.language,
        "language_prob": round(info.language_probability, 2),
        "has_lyrics": True,
        "lang_suno_tag": lang_tag,
        "vocal_density_per_min": vocal_density,
        "lyric_mood": lyric_mood,
        "top_keywords": top_keywords,
        "repeated_lines": repeated_lines[:3],
        "timeline": lyric_segs,
        "full_text_preview": full_text[:500] + ("..." if len(full_text) > 500 else ""),
        "_warning": "Transcription is for REPORTING only. Never paste it into the Lyrics "
                    "field of an original song, and never use it as a video timeline.",
    }


# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Audio analysis for Suno prompt synthesis.")
    ap.add_argument("audio")
    ap.add_argument("--vocal-presence", choices=["vocal", "instrumental", "unknown"],
                    default=None, help="override the acoustic vocal gate")
    ap.add_argument("--skip-whisper", action="store_true")
    args = ap.parse_args()

    try:
        print(json.dumps(analyze(args.audio,
                                 vocal_presence_override=args.vocal_presence,
                                 skip_whisper=args.skip_whisper),
                         ensure_ascii=False, indent=2, cls=NumpyEncoder))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))
        sys.exit(1)
