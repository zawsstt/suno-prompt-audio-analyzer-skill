#!/usr/bin/env python3
"""
audio-analyzer v6: Professional audio analysis with dual-engine (Essentia + Librosa)
Usage: python3 analyze_audio.py <audio_file_path>
Outputs: JSON with comprehensive production-grade analysis

v6 changes:
- Essentia primary engine: KeyExtractor, RhythmExtractor2013, MusicExtractor (EBU R128 LUFS),
  ChordsDetection+HPCP pipeline, Danceability
- Spotify-like features: valence, energy, danceability, acousticness, instrumentalness
- Chord histogram with pct and Roman numeral analysis
- 5-dimensional genre inference: danceability + valence + bpm + key + lufs
- Cinematic/Orchestral genre category added
- 4-dimensional Suno mood mapping: Danceability + Valence + Energy + Key
- Graceful fallback to librosa-only if essentia unavailable
"""

import sys, json, warnings
warnings.filterwarnings("ignore")

KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
KS_MAJOR  = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
KS_MINOR  = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

# ── Try to import Essentia ────────────────────────────────────────────────────
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


# ── Helper functions ──────────────────────────────────────────────────────────

def ks_key(chroma_mean):
    import numpy as np
    best_key, best_mode, best_score = 0, "major", -999.0
    for i in range(12):
        r_maj = float(np.corrcoef(chroma_mean, np.roll(KS_MAJOR,-i))[0,1])
        r_min = float(np.corrcoef(chroma_mean, np.roll(KS_MINOR,-i))[0,1])
        if r_maj > best_score: best_score, best_key, best_mode = r_maj, i, "major"
        if r_min > best_score: best_score, best_key, best_mode = r_min, i, "minor"
    return KEY_NAMES[best_key], best_mode, round(best_score, 4)

def detect_mode(chroma_mean, root_idx):
    import numpy as np
    d = np.roll(chroma_mean, -root_idx); d = d/(d.max()+1e-9)
    m3,M3,m6,M6,m7,M7 = d[3],d[4],d[8],d[9],d[10],d[11]
    if M3 > m3:
        return "Ionian (natural major)" if M7 > m7 else "Mixolydian"
    else:
        if m6 > M6 and m7 > M7: return "Phrygian"
        elif M6 > m6:            return "Dorian"
        else:                    return "Aeolian (natural minor)"

def chord_label(chroma_frame, thr=0.35):
    import numpy as np
    c = chroma_frame/(chroma_frame.max()+1e-9)
    root = int(np.argmax(c))
    m3,M3 = c[(root+3)%12], c[(root+4)%12]
    p5,d5,a5 = c[(root+7)%12], c[(root+6)%12], c[(root+8)%12]
    if M3>thr and p5>thr:   return f"{KEY_NAMES[root]}"
    elif m3>thr and p5>thr: return f"{KEY_NAMES[root]}m"
    elif m3>thr and d5>thr: return f"{KEY_NAMES[root]}dim"
    elif M3>thr and a5>thr: return f"{KEY_NAMES[root]}aug"
    else: return KEY_NAMES[root]

def f0_to_note(f):
    import numpy as np
    if f is None or f <= 0 or np.isnan(f): return "?"
    midi = int(round(12*np.log2(float(f)/440)+69))
    return f"{KEY_NAMES[midi%12]}{midi//12-1}"

def rle(seq):
    from itertools import groupby
    return [(k, sum(1 for _ in g)) for k,g in groupby(seq)]

LIBROSA_NATIVE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".mp3"}

def ensure_wav(filepath):
    import subprocess, tempfile, os
    ext = os.path.splitext(filepath)[1].lower()
    if ext in LIBROSA_NATIVE_EXTS:
        return filepath, False
    tmp = tempfile.mktemp(suffix=".wav", dir="/tmp")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", filepath,
         "-vn", "-acodec", "pcm_s16le", "-ar", "48000", tmp],
        capture_output=True
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

    # ── MusicExtractor: BPM, LUFS, key, danceability ─────────────────────────
    try:
        extractor = es.MusicExtractor(
            lowlevelStats=['mean','stdev'],
            rhythmStats=['mean','stdev'],
            tonalStats=['mean','stdev']
        )
        features, _ = extractor(filepath)
        result["bpm"] = round(float(features['rhythm.bpm']), 2)
        result["lufs_integrated"] = round(float(features['lowlevel.loudness_ebu128.integrated']), 2)
        try:
            result["lufs_range"] = round(float(features['lowlevel.loudness_ebu128.loudness_range']), 2)
        except Exception:
            result["lufs_range"] = None
        result["danceability_raw"] = round(float(features['rhythm.danceability']), 4)
        result["chords_key"] = str(features['tonal.chords_key'])
        result["chords_scale"] = str(features['tonal.chords_scale'])
    except Exception as e:
        result["music_extractor_error"] = str(e)
        result["fallback_used"] = True

    # ── KeyExtractor: most accurate key detection ─────────────────────────────
    audio = None
    try:
        audio = es.MonoLoader(filename=filepath, sampleRate=44100)()
        key_ext = es.KeyExtractor()
        key, scale, strength = key_ext(audio)
        result["key_extractor"] = {
            "key":      str(key),
            "scale":    str(scale),
            "strength": round(float(strength), 4)
        }
    except Exception as e:
        result["key_extractor_error"] = str(e)

    # ── RhythmExtractor2013: precise BPM ──────────────────────────────────────
    try:
        if audio is None:
            audio = es.MonoLoader(filename=filepath, sampleRate=44100)()
        rhythm_ext = es.RhythmExtractor2013(method="multifeature")
        bpm_r, _, _, _, _ = rhythm_ext(audio)
        result["bpm_rhythm_extractor"] = round(float(bpm_r), 2)
        # Use RhythmExtractor2013 as authoritative BPM if close to MusicExtractor
        if "bpm" in result and abs(float(bpm_r) - result["bpm"]) < 20:
            result["bpm"] = round(float(bpm_r), 2)
        elif "bpm" not in result:
            result["bpm"] = round(float(bpm_r), 2)
    except Exception as e:
        result["rhythm_extractor_error"] = str(e)

    # ── ChordsDetection + HPCP pipeline: chord histogram ─────────────────────
    try:
        if audio is None:
            audio = es.MonoLoader(filename=filepath, sampleRate=44100)()

        windowing     = es.Windowing(type='blackmanharris62')
        spectrum_algo = es.Spectrum()
        spectral_peaks = es.SpectralPeaks(
            orderBy='magnitude', magnitudeThreshold=1e-05,
            minFrequency=40, maxFrequency=5000
        )
        hpcp_algo = es.HPCP()
        chord_det = es.ChordsDetection()

        hpcps = []
        for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=2048):
            win  = windowing(frame)
            spec = spectrum_algo(win)
            freqs, mags = spectral_peaks(spec)
            h = hpcp_algo(freqs, mags)
            hpcps.append(h)

        if hpcps:
            hpcp_matrix = np.array(hpcps)
            chords, _ = chord_det(hpcp_matrix)
            chord_counts = Counter(chords)
            total = len(chords)
            result["chord_histogram"] = {
                ch: round(cnt / total * 100, 1)
                for ch, cnt in chord_counts.most_common(12)
            }
        else:
            result["chord_histogram"] = {}

    except Exception as e:
        result["chord_detection_error"] = str(e)
        result["chord_histogram"] = {}

    return result


def chord_to_roman(chord_name, root_key, root_scale):
    """Convert chord name to Roman numeral relative to a given key."""
    NOTE_IDX = {n: i for i, n in enumerate(KEY_NAMES)}

    # Parse chord root and type
    if len(chord_name) >= 2 and chord_name[1] == '#':
        chord_root = chord_name[:2]
        chord_type = chord_name[2:]
    else:
        chord_root = chord_name[:1]
        chord_type = chord_name[1:]

    is_minor_chord = chord_type.startswith('m') and not chord_type.startswith('maj')

    root_idx  = NOTE_IDX.get(root_key, 0)
    chord_idx = NOTE_IDX.get(chord_root, 0)
    interval  = (chord_idx - root_idx) % 12

    if root_scale == "minor":
        DIATONIC   = {0:"i", 2:"ii", 3:"III", 5:"iv", 7:"v", 8:"VI", 10:"VII"}
        CHROMATIC  = {1:"bII", 4:"III", 6:"bV", 9:"VI"}
    else:
        DIATONIC   = {0:"I", 2:"II", 4:"III", 5:"IV", 7:"V", 9:"VI", 11:"VII"}
        CHROMATIC  = {1:"bII", 3:"bIII", 6:"bV", 8:"bVI", 10:"bVII"}

    degree = DIATONIC.get(interval, CHROMATIC.get(interval, f"?{interval}"))

    if is_minor_chord and degree == degree.upper():
        degree = degree.lower()
    elif not is_minor_chord and degree == degree.lower():
        degree = degree.upper()

    suffix = ""
    if "dim" in chord_type:   suffix = "o"
    elif "aug" in chord_type: suffix = "+"
    elif "7" in chord_type:   suffix = "7"

    return degree + suffix


def compute_chord_roman_analysis(chord_histogram, root_key, root_scale):
    result = {}
    for chord, pct in chord_histogram.items():
        roman = chord_to_roman(chord, root_key, root_scale)
        result[chord] = {"pct": pct, "roman": roman}
    return result


def compute_spotify_like_features(essentia_feat, librosa_data, root_scale):
    """
    Estimate Spotify-like audio features (0.0-1.0 scale).
    """
    # ── Valence ───────────────────────────────────────────────────────────────
    centroid_hz   = librosa_data.get("spectral_centroid_hz", 2000)
    brightness_n  = min(1.0, centroid_hz / 8000)
    base_valence  = 0.65 if root_scale == "major" else 0.30
    valence       = round(min(1.0, max(0.0, base_valence + (brightness_n - 0.5) * 0.20)), 3)

    # ── Energy ────────────────────────────────────────────────────────────────
    rms_db       = librosa_data.get("rms_db", -20)
    rms_norm     = min(1.0, max(0.0, (rms_db + 40) / 40))
    flux_norm    = min(1.0, librosa_data.get("spectral_flux_mean", 0.0) * 50)
    energy       = round(min(1.0, max(0.0, rms_norm * 0.6 + flux_norm * 0.4)), 3)

    # ── Danceability (Essentia raw / 3.0) ─────────────────────────────────────
    ess_dance_raw = essentia_feat.get("danceability_raw", 1.0)
    danceability  = round(min(1.0, max(0.0, ess_dance_raw / 3.0)), 3)

    # ── Acousticness ──────────────────────────────────────────────────────────
    harmonic_ratio = librosa_data.get("harmonic_ratio", 0.5)
    acousticness   = round(min(1.0, max(0.0, harmonic_ratio * (1.0 - brightness_n))), 3)

    # ── Instrumentalness ──────────────────────────────────────────────────────
    has_lyrics     = librosa_data.get("has_lyrics", False)
    vocal_density  = librosa_data.get("vocal_density_per_min", 0)
    if not has_lyrics:
        instrumentalness = 0.95
    else:
        instrumentalness = max(0.0, 0.8 - min(vocal_density / 30.0, 0.8))
    instrumentalness = round(instrumentalness, 3)

    return {
        "valence":          valence,
        "energy":           energy,
        "danceability":     danceability,
        "acousticness":     acousticness,
        "instrumentalness": instrumentalness,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analyze function
# ─────────────────────────────────────────────────────────────────────────────

def analyze(filepath):
    import numpy as np, librosa, scipy.signal, os
    from scipy.ndimage import uniform_filter1d
    from collections import Counter
    result = {}

    # ── 0a. Format normalisation ───────────────────────────────────────────────
    import subprocess
    wav_path, is_temp_wav = ensure_wav(filepath)

    # ── 0b. File metadata ──────────────────────────────────────────────────────
    probe = subprocess.run(
        ["ffprobe","-v","quiet","-print_format","json",
         "-show_format","-show_streams", filepath],
        capture_output=True, text=True)
    if probe.returncode == 0:
        meta = json.loads(probe.stdout)
        fmt = meta.get("format",{})
        astream = next((s for s in meta.get("streams",[]) if s.get("codec_type")=="audio"),{})
        result["file_info"] = {
            "format":               fmt.get("format_long_name", fmt.get("format_name","?")),
            "duration_sec":         round(float(fmt.get("duration",0)),2),
            "size_bytes":           int(fmt.get("size",0)),
            "bitrate_kbps":         round(int(fmt.get("bit_rate",0))/1000,1),
            "codec":                astream.get("codec_name","?"),
            "native_sample_rate_hz": int(astream.get("sample_rate",0)),
            "channels":             astream.get("channels",0),
            "channel_layout":       astream.get("channel_layout","?"),
        }

    # ── 1. Load (22050 Hz mono for librosa analysis) ───────────────────────────
    SR = 22050
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    ch = result.get("file_info",{}).get("channels",1)
    result["channel_analysis"] = {
        "type": "stereo" if ch >= 2 else "mono",
        "note": "Stereo correlation skipped for memory efficiency; use DAW for precise M/S analysis"
    }

    # ── 2. Loudness & dynamics (librosa) ──────────────────────────────────────
    rms_v   = float(np.sqrt(np.mean(y**2)))
    peak_v  = float(np.max(np.abs(y)))
    rms_db  = float(librosa.amplitude_to_db(np.array([rms_v+1e-9]))[0])
    peak_db = float(librosa.amplitude_to_db(np.array([peak_v+1e-9]))[0])
    lufs_approx = round(rms_db - 0.691, 2)
    crest_db    = round(peak_db - rms_db, 2)
    hop_1s      = sr
    rms_wins    = [float(np.sqrt(np.mean(y[i:i+hop_1s]**2))) for i in range(0,len(y)-hop_1s,hop_1s)]
    rms_db_wins = librosa.amplitude_to_db(np.array(rms_wins)+1e-9)
    dr          = round(float(np.percentile(rms_db_wins,95)-np.percentile(rms_db_wins,5)),2)
    result["loudness"] = {
        "rms_db":           round(rms_db,2),
        "peak_db":          round(peak_db,2),
        "lufs_approx":      lufs_approx,
        "crest_factor_db":  crest_db,
        "dynamic_range_db": dr,
        "headroom_db":      round(-peak_db,2),
        "compression_hint": "heavily compressed" if crest_db<8 else ("moderate" if crest_db<14 else "wide dynamics"),
    }

    # ── 3. Frequency band energy ───────────────────────────────────────────────
    fft_mag = np.abs(np.fft.rfft(y))
    freqs   = np.fft.rfftfreq(len(y), d=1.0/sr)
    total_e = np.sum(fft_mag**2)+1e-12
    bands = [
        ("sub_bass_20_60hz",   20,  60),
        ("bass_60_250hz",      60, 250),
        ("low_mid_250_500hz", 250, 500),
        ("mid_500_2khz",      500,2000),
        ("high_mid_2k_6khz", 2000,6000),
        ("presence_6k_12khz",6000,10000),
        ("air_10k_plus",    10000,sr//2-1),
    ]
    band_e = {}
    for name,lo,hi in bands:
        mask = (freqs>=lo)&(freqs<hi)
        e = float(np.sum(fft_mag[mask]**2))
        band_e[name] = {
            "energy_pct": round(e/total_e*100,2),
            "db": round(float(librosa.amplitude_to_db(np.array([np.sqrt(e/(mask.sum()+1))]))[0]),1)
        }
    result["frequency_bands"] = band_e

    # ── 4. Rhythm & beat grid (librosa fallback; Essentia BPM preferred) ──────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm_librosa = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times  = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times)>4:
        ivs = np.diff(beat_times)
        cv  = float(np.std(ivs)/np.mean(ivs))
        swing_hint = "straight" if cv<0.05 else ("light swing" if cv<0.12 else "heavy swing/rubato")
    else:
        cv=0.0; swing_hint="sparse"
    oe    = librosa.onset.onset_strength(y=y, sr=sr)
    ac    = librosa.autocorrelate(oe, max_size=sr//2)
    tf    = int(round(sr/(bpm_librosa/60)/512))
    beat4 = ac[tf*4] if tf*4<len(ac) else 0
    beat3 = ac[tf*3] if tf*3<len(ac) else 0
    time_sig = "4/4" if beat4>=beat3 else "3/4"

    result["rhythm"] = {
        "tempo_bpm_librosa":         round(bpm_librosa,1),
        "beat_count":                len(beat_times),
        "estimated_time_signature":  time_sig,
        "swing_type":                swing_hint,
        "beat_interval_cv":          round(cv,4),
        "beat_times_first16":        [round(float(t),3) for t in beat_times[:16]],
    }

    # ── 5. Drum pattern (kick/snare/hihat) ─────────────────────────────────────
    _, y_perc = librosa.effects.hpss(y)
    def bp_filter(sig, lo, hi, fs):
        nyq  = fs/2
        lo_n = max(lo/nyq, 0.001); hi_n = min(hi/nyq, 0.999)
        sos  = scipy.signal.butter(4,[lo_n,hi_n],btype='band',output='sos')
        return scipy.signal.sosfilt(sos, sig)

    hop     = 256
    bar_dur = 60.0/bpm_librosa*4
    end_s   = min(bar_dur*8, 20.0)
    end_smp = int(end_s*sr)

    def drum_onsets(sig, sr, hop, min_dist=0.08):
        frames  = librosa.util.frame(sig, frame_length=hop*2, hop_length=hop)
        rms_arr = np.sqrt(np.mean(frames**2, axis=0))
        thr     = np.percentile(rms_arr,80)
        peaks,_ = scipy.signal.find_peaks(rms_arr, height=thr, distance=int(sr*min_dist/hop))
        return sorted([round(float(librosa.frames_to_time(p,sr=sr,hop_length=hop)),3) for p in peaks])

    kick_sig  = bp_filter(y_perc[:end_smp], 40,  120, sr)
    snare_sig = bp_filter(y_perc[:end_smp], 150, 500, sr)
    hihat_sig = bp_filter(y_perc[:end_smp], 5000, min(10000, sr//2-100), sr)

    kick_pat  = drum_onsets(kick_sig,  sr, hop)[:24]
    snare_pat = drum_onsets(snare_sig, sr, hop)[:24]
    hihat_pat = drum_onsets(hihat_sig, sr, hop)[:32]

    result["drum_pattern"] = {
        "analysis_window": f"first {round(end_s,1)}s (~8 bars)",
        "kick_onsets_sec":  kick_pat,
        "snare_onsets_sec": snare_pat,
        "hihat_onsets_sec": hihat_pat,
        "kick_per_bar":   round(len(kick_pat)/(end_s/bar_dur),2) if bar_dur>0 else 0,
        "snare_per_bar":  round(len(snare_pat)/(end_s/bar_dur),2) if bar_dur>0 else 0,
        "hihat_per_bar":  round(len(hihat_pat)/(end_s/bar_dur),2) if bar_dur>0 else 0,
    }

    # ── 6. Bass line (low-pass + pyin on first 90s) ────────────────────────────
    y90    = y[:int(90*sr)]
    sos_lp = scipy.signal.butter(4, 300/(sr/2), btype='low', output='sos')
    y_bass = scipy.signal.sosfilt(sos_lp, y90)
    try:
        f0b, vb, _ = librosa.pyin(y_bass, fmin=30, fmax=300, sr=sr,
                                    frame_length=4096, hop_length=1024)
        bass_notes = [f0_to_note(f) for f in f0b[vb]]
        bass_seq   = [{"note":k,"frames":v} for k,v in rle(bass_notes)[:24] if k!="?"]
        bass_top   = [{"note":k,"count":v} for k,v in Counter(n for n in bass_notes if n!="?").most_common(8)]
    except Exception as e:
        bass_seq = []; bass_top = [{"error": str(e)}]
    result["bassline"] = {
        "dominant_bass_notes": bass_top,
        "bass_sequence_start": bass_seq,
    }
    del y_bass, y90

    # ── 7. Chord progression (librosa) ────────────────────────────────────────
    y_harm, _ = librosa.effects.hpss(y)
    chroma     = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, bins_per_octave=36)
    beat_chroma= librosa.util.sync(chroma, beats, aggregate=np.median)
    chord_labels = [chord_label(beat_chroma[:,i]) for i in range(beat_chroma.shape[1])]
    chord_seq    = [{"chord":k,"beats":v} for k,v in rle(chord_labels)[:32]]
    bigrams      = [(chord_seq[i]["chord"], chord_seq[i+1]["chord"]) for i in range(len(chord_seq)-1)]
    common_trans = [{"from":a,"to":b,"count":c} for (a,b),c in Counter(bigrams).most_common(8)]
    unique_chords = list(dict.fromkeys(c["chord"] for c in chord_seq))
    result["chord_progression"] = {
        "sequence":             chord_seq[:24],
        "unique_chords_used":   unique_chords[:12],
        "common_transitions":   common_trans,
    }

    # ── 8. Key, mode, scale degrees (librosa KS, may be overridden by Essentia) ─
    chroma_mean  = chroma.mean(axis=1)
    root_name_ks, mode_ks, ks_score = ks_key(chroma_mean)
    root_idx_ks  = KEY_NAMES.index(root_name_ks)
    modal        = detect_mode(chroma_mean, root_idx_ks)
    rel_idx      = (root_idx_ks+3)%12 if mode_ks=="minor" else (root_idx_ks-3)%12
    relative     = f"{KEY_NAMES[rel_idx]} {'major' if mode_ks=='minor' else 'minor'}"
    rotated      = np.roll(chroma_mean, -root_idx_ks); rotated /= rotated.max()+1e-9
    degree_labels= ["1","b2","2","b3","3","4","b5","5","b6","6","b7","7"]
    strong_deg   = [degree_labels[i] for i in range(12) if rotated[i]>0.5]

    result["tonality"] = {
        "key":                  f"{root_name_ks} {mode_ks}",
        "root":                 root_name_ks,
        "mode":                 mode_ks,
        "modal_flavor":         modal,
        "ks_confidence":        ks_score,
        "relative_key":         relative,
        "strong_scale_degrees": strong_deg,
        "chroma_profile":       {k:round(float(v),4) for k,v in zip(KEY_NAMES,chroma_mean)},
        "source":               "librosa KS",
    }

    # ── 9. Key timeline (per 15s) ──────────────────────────────────────────────
    key_tl = []
    prev = None
    for i in range(int(np.ceil(duration/15))):
        s0 = int(i*15*sr); s1 = int(min((i+1)*15*sr,len(y)))
        if s1-s0 < sr: continue
        cm = librosa.feature.chroma_cqt(y=y[s0:s1],sr=sr).mean(axis=1)
        rn,rm,rs = ks_key(cm)
        ks = f"{rn} {rm}"
        t0s = f"{int(i*15//60):02d}:{int(i*15%60):02d}"
        entry = {"time":t0s,"key":ks,"conf":rs}
        if ks!=prev: entry["changed"]=True; prev=ks
        key_tl.append(entry)
    result["key_timeline"] = key_tl

    # ── 10. Melodic contour (harmonic channel, first 90s) ─────────────────────
    y_harm90 = y_harm[:int(90*sr)]
    try:
        f0m, vm, _ = librosa.pyin(y_harm90, fmin=80, fmax=2000, sr=sr,
                                    frame_length=2048, hop_length=512)
        mel_notes = [f0_to_note(f) for f in f0m[vm]]
        mel_seq   = [{"note":k,"frames":v} for k,v in rle(mel_notes)[:32] if k!="?"]
        mel_top   = [{"note":k,"count":v} for k,v in Counter(n for n in mel_notes if n!="?").most_common(10)]
        valid_f0  = f0m[vm & ~np.isnan(f0m)]
        pr = {
            "lowest_note":  f0_to_note(valid_f0.min()) if len(valid_f0)>0 else "?",
            "highest_note": f0_to_note(valid_f0.max()) if len(valid_f0)>0 else "?",
            "range_hz":     f"{round(float(valid_f0.min()),1) if len(valid_f0)>0 else 0}–{round(float(valid_f0.max()),1) if len(valid_f0)>0 else 0}"
        }
    except Exception as e:
        mel_seq=[]; mel_top=[]; pr={"error":str(e)}
    del y_harm90
    result["melodic_contour"] = {
        "pitch_range":            pr,
        "dominant_melody_notes":  mel_top,
        "melody_sequence_start":  mel_seq,
    }

    # ── 11. Energy / tension curve (per 5s) ───────────────────────────────────
    ecurve   = []
    flux_vals= []
    for i in range(int(np.ceil(duration/5))):
        s0 = int(i*5*sr); s1 = int(min((i+1)*5*sr,len(y)))
        if s1==s0: continue
        chunk = y[s0:s1]
        rv    = float(np.sqrt(np.mean(chunk**2)))
        rd    = float(librosa.amplitude_to_db(np.array([rv+1e-9]))[0])
        spec  = np.abs(librosa.stft(chunk,n_fft=512))
        flux  = float(np.mean(np.diff(spec,axis=1)**2)) if spec.shape[1]>1 else 0.0
        flux_vals.append(flux)
        ecurve.append({"time":f"{int(i*5//60):02d}:{int(i*5%60):02d}",
                       "rms_db":round(rd,1),"spectral_flux":round(flux,4)})
    result["energy_curve"] = ecurve
    spectral_flux_mean = float(np.mean(flux_vals)) if flux_vals else 0.0

    # ── 12. Structure (energy drop / peak detection) ───────────────────────────
    rms_env = librosa.feature.rms(y=y, frame_length=sr//2, hop_length=sr//4)[0]
    rms_sm  = uniform_filter1d(librosa.amplitude_to_db(rms_env+1e-9), size=8)
    drops   = []
    for i in range(4,len(rms_sm)-4):
        if rms_sm[i]<rms_sm[i-4]-3 and rms_sm[i]<rms_sm[i+4]-3:
            drops.append(round(float(librosa.frames_to_time(i,sr=sr,hop_length=sr//4)),1))
    pk_idx,_ = scipy.signal.find_peaks(rms_sm, prominence=3, distance=8)
    pk_times  = [round(float(librosa.frames_to_time(p,sr=sr,hop_length=sr//4)),1) for p in pk_idx]
    result["structure"] = {
        "energy_drop_times_sec": drops[:10],
        "energy_peak_times_sec": pk_times[:10],
        "tip": "Drops = likely section breaks (verse<->chorus); Peaks = climax moments",
    }

    # ── 13. Spectral profile ───────────────────────────────────────────────────
    sc   = librosa.feature.spectral_centroid(y=y,sr=sr)[0]
    sb   = librosa.feature.spectral_bandwidth(y=y,sr=sr)[0]
    sro  = librosa.feature.spectral_rolloff(y=y,sr=sr)[0]
    sf   = librosa.feature.spectral_flatness(y=y)[0]
    zcr  = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
    avg_c = float(np.mean(sc))
    result["spectral"] = {
        "centroid_hz":  round(avg_c,1),
        "bandwidth_hz": round(float(np.mean(sb)),1),
        "rolloff_hz":   round(float(np.mean(sro)),1),
        "flatness":     round(float(np.mean(sf)),6),
        "zcr":          round(float(np.mean(zcr)),5),
        "brightness":   "bright" if avg_c>3000 else ("mid" if avg_c>1000 else "dark"),
        "mfcc_means":   [round(float(v),3) for v in mfcc.mean(axis=1)],
        "mfcc_stds":    [round(float(v),3) for v in mfcc.std(axis=1)],
    }

    # ── 14. Harmonic/Percussive ratio ──────────────────────────────────────────
    he = float(np.mean(y_harm**2)); pe = float(np.mean(y_perc**2)); te = he+pe+1e-12
    result["harmonic_percussive"] = {
        "harmonic_ratio":   round(he/te,4),
        "percussive_ratio": round(pe/te,4),
        "dominant": "harmonic" if he>pe else "percussive",
    }

    # ── 15. Essentia analysis (primary engine) ────────────────────────────────
    ess_feat = {}
    if ESSENTIA_AVAILABLE:
        try:
            ess_feat = analyze_essentia(filepath)
        except Exception as e:
            ess_feat = {"available": False, "error": str(e)}
    else:
        ess_feat = {"available": False, "reason": "essentia not installed"}

    result["analysis_engine"] = {
        "primary":   "essentia" if ess_feat.get("available") else "librosa",
        "secondary": "librosa",
        "fallback":  not ess_feat.get("available", False),
    }

    # Compile essentia_features block
    ess_out = {
        "available": ess_feat.get("available", False),
    }
    if ess_feat.get("available"):
        if "bpm" in ess_feat:
            ess_out["bpm"] = ess_feat["bpm"]
        if "bpm_rhythm_extractor" in ess_feat:
            ess_out["bpm_rhythm_extractor"] = ess_feat["bpm_rhythm_extractor"]
        if "lufs_integrated" in ess_feat:
            ess_out["lufs_integrated"] = ess_feat["lufs_integrated"]
        if "lufs_range" in ess_feat:
            ess_out["lufs_range"] = ess_feat["lufs_range"]
        if "danceability_raw" in ess_feat:
            ess_out["danceability"] = ess_feat["danceability_raw"]
        if "key_extractor" in ess_feat:
            ess_out["key_extractor"] = ess_feat["key_extractor"]
        if "chords_key" in ess_feat:
            ess_out["chords_key"] = ess_feat["chords_key"]
            ess_out["chords_scale"] = ess_feat.get("chords_scale", "")
        if "chord_histogram" in ess_feat:
            ess_out["chord_histogram"] = ess_feat["chord_histogram"]

        # Override tonality with Essentia KeyExtractor (higher accuracy)
        if "key_extractor" in ess_feat:
            ke = ess_feat["key_extractor"]
            ess_root  = ke["key"]
            ess_scale = ke["scale"]
            ess_root_idx = KEY_NAMES.index(ess_root) if ess_root in KEY_NAMES else root_idx_ks
            ess_modal    = detect_mode(chroma_mean, ess_root_idx)
            ess_rel_idx  = (ess_root_idx+3)%12 if ess_scale=="minor" else (ess_root_idx-3)%12
            ess_relative = f"{KEY_NAMES[ess_rel_idx]} {'major' if ess_scale=='minor' else 'minor'}"
            result["tonality"].update({
                "key":         f"{ess_root} {ess_scale}",
                "root":        ess_root,
                "mode":        ess_scale,
                "modal_flavor": ess_modal,
                "ks_confidence": ke["strength"],
                "relative_key": ess_relative,
                "source":      "essentia KeyExtractor",
            })

        # Chord Roman numeral analysis
        if ess_feat.get("chord_histogram"):
            final_root  = result["tonality"]["root"]
            final_scale = result["tonality"]["mode"]
            ess_out["chord_roman_analysis"] = compute_chord_roman_analysis(
                ess_feat["chord_histogram"], final_root, final_scale
            )

    result["essentia_features"] = ess_out

    # ── 16. Production style assessment ───────────────────────────────────────
    sub_pct    = band_e["sub_bass_20_60hz"]["energy_pct"]
    bass_pct   = band_e["bass_60_250hz"]["energy_pct"]
    mid_pct    = band_e["mid_500_2khz"]["energy_pct"]
    hi_mid_pct = band_e["high_mid_2k_6khz"]["energy_pct"]
    air_pct    = band_e["air_10k_plus"]["energy_pct"]
    crest      = result["loudness"]["crest_factor_db"]
    dr_db      = result["loudness"]["dynamic_range_db"]
    hihat_pb   = result["drum_pattern"]["hihat_per_bar"]
    kick_pb    = result["drum_pattern"]["kick_per_bar"]

    # Use Essentia BPM if available, else librosa
    bpm = ess_feat.get("bpm", bpm_librosa) if ess_feat.get("available") else bpm_librosa

    # Update rhythm with authoritative BPM
    result["rhythm"]["tempo_bpm"] = round(bpm, 1)

    # Get Essentia LUFS (EBU R128) if available, else librosa approx
    lufs_val = ess_feat.get("lufs_integrated", lufs_approx) if ess_feat.get("available") else lufs_approx

    # Get danceability for genre inference
    dance_raw  = ess_feat.get("danceability_raw", 1.0) if ess_feat.get("available") else 1.0
    dance_norm = min(1.0, dance_raw / 3.0)

    # Get mode for genre inference
    final_mode  = result["tonality"]["mode"]
    final_scale = final_mode  # "major" or "minor"

    # ── Spotify-like features (before genre scoring, used in scoring) ─────────
    librosa_data_for_spotify = {
        "spectral_centroid_hz": avg_c,
        "rms_db":               rms_db,
        "spectral_flux_mean":   spectral_flux_mean,
        "harmonic_ratio":       round(he/te, 4),
        "has_lyrics":           False,  # placeholder, updated after lyrics
        "vocal_density_per_min":0,
    }
    spotify_features = compute_spotify_like_features(ess_feat, librosa_data_for_spotify, final_scale)

    # ── Multi-signal genre scoring (v6: 5-dimensional) ────────────────────────
    genre_scores = {}

    def add(g, score, reason=""):
        genre_scores[g] = genre_scores.get(g, 0) + score

    valence_est  = spotify_features["valence"]
    energy_est   = spotify_features["energy"]

    # House / Deep House
    if 118 <= bpm <= 135:           add("House / Deep House", 2, "bpm")
    if kick_pb >= 3.5:              add("House / Deep House", 2, "four-on-floor kick")
    if sub_pct > 3 and sub_pct<15:  add("House / Deep House", 1, "moderate sub")
    if he/te > 0.5:                  add("House / Deep House", 1, "harmonic")
    if avg_c < 4000:                 add("House / Deep House", 1, "warm centroid")

    # Tech House
    if 126 <= bpm <= 138:           add("Tech House", 2, "bpm")
    if kick_pb >= 3.5:              add("Tech House", 2, "tight kick")
    if avg_c > 3000:                 add("Tech House", 1, "bright")
    if hihat_pb >= 3:               add("Tech House", 1, "busy hihat")
    if pe/te > 0.35:                 add("Tech House", 1, "percussive")

    # Trance / Progressive
    if 126 <= bpm <= 145:           add("Trance / Progressive", 2, "bpm")
    if avg_c > 3500:                 add("Trance / Progressive", 1, "bright")
    if he/te > 0.6:                  add("Trance / Progressive", 2, "harmonic build")
    if air_pct > 5:                  add("Trance / Progressive", 1, "airy highs")
    if kick_pb >= 3.5:              add("Trance / Progressive", 1, "four-on-floor")

    # Drum & Bass / Breakbeat
    if bpm >= 160:                   add("Drum & Bass / Breakbeat", 3, "high bpm")
    elif 140 <= bpm < 160:          add("Drum & Bass / Breakbeat", 2, "bpm")
    if pe/te > 0.35:                 add("Drum & Bass / Breakbeat", 2, "percussive")
    if sub_pct > 5:                  add("Drum & Bass / Breakbeat", 1, "sub bass")
    if avg_c > 2500:                 add("Drum & Bass / Breakbeat", 1, "bright")

    # Techno / Hard Techno
    if bpm >= 130:                   add("Techno / Hard Dance", 1, "high bpm")
    if bpm >= 140:                   add("Techno / Hard Dance", 2, "techno bpm")
    if avg_c > 4000:                 add("Techno / Hard Dance", 2, "harsh bright")
    if pe/te > 0.4:                  add("Techno / Hard Dance", 1, "industrial percussive")
    if kick_pb >= 3.5:              add("Techno / Hard Dance", 1, "kick-driven")

    # Trap / Hip-hop
    if 60 <= bpm <= 100:            add("Trap / Hip-hop", 2, "bpm")
    elif 100 < bpm <= 140:          add("Trap / Hip-hop", 1, "bpm half-time")
    if sub_pct > 10:                 add("Trap / Hip-hop", 3, "heavy sub")
    if hihat_pb >= 4:               add("Trap / Hip-hop", 2, "trap hihats")
    if bass_pct > 30:               add("Trap / Hip-hop", 1, "bass dominant")
    if kick_pb < 2:                  add("Trap / Hip-hop", 1, "sparse kick (trap)")

    # Lo-fi Hip-hop / Boom-bap
    if 70 <= bpm <= 100:            add("Lo-fi / Boom-bap", 2, "bpm")
    if he/te > 0.65:                 add("Lo-fi / Boom-bap", 2, "harmonic")
    if avg_c < 2500:                 add("Lo-fi / Boom-bap", 2, "warm/muffled")
    if crest > 10:                   add("Lo-fi / Boom-bap", 1, "dynamic")
    if hihat_pb < 3:                 add("Lo-fi / Boom-bap", 1, "sparse hihat")

    # R&B / Neo-soul
    if 60 <= bpm <= 110:            add("R&B / Neo-soul", 1, "bpm")
    if he/te > 0.7:                  add("R&B / Neo-soul", 2, "harmonic")
    if avg_c < 3000:                 add("R&B / Neo-soul", 1, "warm")
    if sub_pct > 5:                  add("R&B / Neo-soul", 1, "bass presence")
    if mid_pct > 20:                 add("R&B / Neo-soul", 1, "midrange presence")

    # Pop / Indie Pop
    if 100 <= bpm <= 130:           add("Pop / Indie Pop", 1, "bpm")
    if avg_c > 2000 and avg_c < 4000: add("Pop / Indie Pop", 2, "balanced bright")
    if he/te > 0.6:                  add("Pop / Indie Pop", 1, "melodic")
    if crest < 14:                   add("Pop / Indie Pop", 1, "compressed radio mix")
    if air_pct > 3:                  add("Pop / Indie Pop", 1, "polished highs")

    # Ambient / Cinematic (v6: added 5-dim scoring)
    if bpm <= 90:                    add("Ambient / Cinematic", 1, "slow bpm")
    if he/te > 0.85:                  add("Ambient / Cinematic", 3, "very harmonic")
    if avg_c < 2000:                  add("Ambient / Cinematic", 2, "dark/soft centroid")
    if sub_pct < 3 and bass_pct < 20: add("Ambient / Cinematic", 1, "low bass presence")
    if crest > 12:                    add("Ambient / Cinematic", 1, "wide dynamics")
    # v6: 5-dim additional signals
    if valence_est < 0.45 and dance_norm < 0.5: add("Ambient / Cinematic", 2, "low valence + low dance")
    if lufs_val < -18:                add("Ambient / Cinematic", 2, "quiet master")

    # Cinematic / Orchestral (v6: new category)
    if he/te > 0.80:                  add("Cinematic / Orchestral", 3, "very harmonic")
    if avg_c > 800 and avg_c < 3500:  add("Cinematic / Orchestral", 2, "orchestral centroid")
    if crest > 14:                    add("Cinematic / Orchestral", 2, "wide dynamics")
    if valence_est < 0.6:             add("Cinematic / Orchestral", 1, "minor-leaning valence")
    if lufs_val < -14:                add("Cinematic / Orchestral", 1, "dynamic master")
    if bpm < 130:                     add("Cinematic / Orchestral", 1, "non-dance bpm")
    if dance_norm < 0.5:             add("Cinematic / Orchestral", 2, "low danceability")
    if kick_pb < 2:                   add("Cinematic / Orchestral", 1, "sparse kick")

    # EDM / Big Room
    if 126 <= bpm <= 140:           add("EDM / Big Room", 1, "bpm")
    if air_pct > 8:                  add("EDM / Big Room", 2, "bright/shiny highs")
    if avg_c > 4000:                 add("EDM / Big Room", 1, "bright")
    if kick_pb >= 3.5:              add("EDM / Big Room", 1, "four-on-floor")
    if he/te > 0.5 and sub_pct > 5: add("EDM / Big Room", 2, "full spectrum")

    # ── Pick top genres ────────────────────────────────────────────────────────
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    THRESHOLD = 4
    genres = [g for g, s in sorted_genres if s >= THRESHOLD]
    if not genres:
        genres = [g for g, s in sorted_genres[:2]]
    if not genres:
        genres = ["Electronic / Instrumental (mixed)"]
    genres = genres[:3]

    genre_scores_detail = {g: s for g, s in sorted_genres[:8]}

    result["production_style"] = {
        "likely_genres": genres,
        "genre_scores":  genre_scores_detail,
        "mastering": "heavily limited" if crest<8 else ("radio-ready" if crest<14 else "wide dynamic range"),
        "daw_tips": [
            f"Set project BPM to {round(bpm,1)}, time signature {time_sig}",
            f"Program in {result['tonality']['root']} {result['tonality']['mode']} — scale: {result['tonality']['modal_flavor']}",
            f"Reference key: {result['tonality']['relative_key']} (for modulations)",
            f"Sub bass ({round(sub_pct,1)}% energy) — {'boost sub' if sub_pct<3 else 'sub already prominent'}",
            f"Mix brightness target: ~{round(avg_c)}Hz spectral centroid",
            f"Aim for ~{round(lufs_val,1)} LUFS integrated loudness",
            f"Compression depth: crest factor {crest}dB → {'use heavy parallel compression' if crest>14 else 'moderate limiting'}",
        ],
    }

    # ── 17. Summary ────────────────────────────────────────────────────────────
    root_final = result["tonality"]["root"]
    mode_final = result["tonality"]["mode"]
    result["summary"] = {
        "duration":          f"{int(duration//60):02d}:{int(duration%60):02d}",
        "key_full":          f"{root_final} {mode_final} ({result['tonality']['modal_flavor']})",
        "relative_key":      result["tonality"]["relative_key"],
        "tempo_bpm":         round(bpm,1),
        "time_signature":    time_sig,
        "swing":             swing_hint,
        "lufs":              round(lufs_val,2),
        "lufs_source":       "essentia EBU R128" if ess_feat.get("available") and "lufs_integrated" in ess_feat else "librosa approx",
        "lufs_approx":       lufs_approx,
        "dynamic_range_db":  dr,
        "brightness":        result["spectral"]["brightness"],
        "harmonic_dominant": he>pe,
        "likely_genres":     genres,
    }

    # ── 18. Lyrics Analysis (faster-whisper) ──────────────────────────────────
    result["lyrics"] = extract_lyrics(filepath)

    # Update Spotify features with actual lyrics data
    has_lyrics_actual  = result["lyrics"].get("has_lyrics", False)
    vocal_density_actual = result["lyrics"].get("vocal_density_per_min", 0)
    librosa_data_for_spotify["has_lyrics"]          = has_lyrics_actual
    librosa_data_for_spotify["vocal_density_per_min"] = vocal_density_actual
    spotify_features = compute_spotify_like_features(ess_feat, librosa_data_for_spotify, final_scale)
    result["spotify_like_features"] = spotify_features

    # ── 19. Suno Prompt Generation ─────────────────────────────────────────────
    result["suno_prompt"] = generate_suno_prompt(result)

    # ── Cleanup temp WAV ───────────────────────────────────────────────────────
    if is_temp_wav:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Lyrics extraction (faster-whisper)
# ─────────────────────────────────────────────────────────────────────────────

def extract_lyrics(path):
    import subprocess, re, collections

    tmp_wav = "/tmp/_whisper_input.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_wav],
        capture_output=True
    )
    if r.returncode != 0:
        return {"error": "ffmpeg conversion failed", "detail": r.stderr.decode()[:300]}

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "faster-whisper not installed. Run: pip install faster-whisper"}

    try:
        model = WhisperModel("small", device="cpu", compute_type="int8",
                             download_root="/tmp/whisper_models")
        segments_iter, info = model.transcribe(
            tmp_wav, beam_size=5, language=None,
            vad_filter=False, word_timestamps=False,
            no_speech_threshold=0.6, log_prob_threshold=-0.5
        )
        segments = list(segments_iter)
    except Exception as e:
        return {"error": str(e)}

    noise = {"[Music]","[MUSIC]","(Music)","Music","MUSIC","[Applause]",
             "[Silence]","[silence]","[inaudible]","[Inaudible]",
             "[music]","(music)","[background music]","[instrumental]"}
    lyric_segs = []
    for seg in segments:
        text = seg.text.strip()
        if text and text not in noise and len(text) > 3:
            ts = f"{int(seg.start//60):02d}:{int(seg.start%60):02d}"
            lyric_segs.append({"time": ts, "start_sec": round(seg.start, 1), "text": text})

    if not lyric_segs:
        return {
            "language": info.language,
            "language_prob": round(info.language_probability, 2),
            "has_lyrics": False,
            "note": "No vocal content detected — likely instrumental",
        }

    full_text = " / ".join(s["text"] for s in lyric_segs)

    stopwords = {"the","a","an","and","or","but","in","on","at","to","for",
                 "of","with","is","are","was","were","be","been","have","has",
                 "had","we","i","you","he","she","it","they","this","that",
                 "all","just","no","not","so","when","as","if","our","their",
                 "my","your","its","what","there","will","can","do","dont",
                 "im","its","its","set","one","like"}
    words     = re.findall(r"[a-z']+", full_text.lower())
    word_freq = collections.Counter(w for w in words if w not in stopwords and len(w) > 2)
    top_keywords = [w for w, _ in word_freq.most_common(10)]

    line_texts = [s["text"].lower().strip() for s in lyric_segs]
    line_freq  = collections.Counter(line_texts)
    repeated_lines = [l for l, c in line_freq.most_common(5) if c >= 2]

    total_dur_min = lyric_segs[-1]["start_sec"] / 60 if lyric_segs else 1
    vocal_density = round(len(lyric_segs) / max(total_dur_min, 0.1), 1)

    lang_tag_map = {
        "en": "English lyrics", "zh": "Chinese lyrics", "ja": "Japanese lyrics",
        "ko": "Korean lyrics",  "es": "Spanish lyrics", "fr": "French lyrics",
        "de": "German lyrics",  "pt": "Portuguese lyrics",
    }
    lang_tag = lang_tag_map.get(info.language, f"{info.language} lyrics")

    lyric_lower = full_text.lower()
    lyric_mood  = None
    mood_keywords = {
        "dark":       ["dark","shadow","dead","death","end","destroy","void","silence","fall"],
        "aggressive": ["fire","strike","charge","attack","breach","fight","violent","exact"],
        "mysterious": ["static","signal","inter","unspoken","recalibrate","locked"],
        "emotional":  ["heart","love","tears","feel","soul","hope","dream"],
        "epic":       ["structure","falls","prevails","unity","trust","purpose","complete"],
    }
    mood_scores = {m: sum(lyric_lower.count(kw) for kw in kws)
                   for m, kws in mood_keywords.items()}
    if max(mood_scores.values()) > 0:
        lyric_mood = max(mood_scores, key=mood_scores.get)

    return {
        "language":               info.language,
        "language_prob":          round(info.language_probability, 2),
        "has_lyrics":             True,
        "lang_suno_tag":          lang_tag,
        "vocal_density_per_min":  vocal_density,
        "lyric_mood":             lyric_mood,
        "top_keywords":           top_keywords,
        "repeated_lines":         repeated_lines[:3],
        "timeline":               lyric_segs,
        "full_text_preview":      full_text[:500] + ("..." if len(full_text) > 500 else ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Suno prompt generator (v6: 4-dimensional mood mapping)
# ─────────────────────────────────────────────────────────────────────────────

def generate_suno_prompt(r):
    """
    Synthesize analysis data into a Suno-compatible style prompt.
    v6: 4-dimensional mood mapping using Danceability + Valence + Energy + Key.
    """

    GENRE_TAG_MAP = {
        "Drum & Bass / Breakbeat":           ["drum and bass", "breakbeat"],
        "Techno / Hard Dance":               ["techno", "hard techno"],
        "Tech House":                        ["tech house"],
        "House / Deep House":                ["deep house"],
        "Trance / Progressive":              ["progressive trance", "trance"],
        "EDM / Big Room":                    ["big room EDM", "EDM"],
        "Trap / Hip-hop":                    ["trap", "hip-hop"],
        "Lo-fi / Boom-bap":                  ["lo-fi hip-hop", "boom bap"],
        "R&B / Neo-soul":                    ["R&B", "neo-soul"],
        "Ambient / Cinematic":               ["cinematic", "ambient"],
        "Cinematic / Orchestral":            ["cinematic orchestral", "epic orchestral"],
        "Pop / Indie Pop":                   ["indie pop", "pop"],
        "Electronic / Instrumental (mixed)": ["electronic"],
    }

    likely       = r.get("production_style", {}).get("likely_genres", [])
    genre_scores = r.get("production_style", {}).get("genre_scores", {})
    genre_tags   = []
    for g in likely:
        for key, tags in GENRE_TAG_MAP.items():
            if key in g or g in key:
                for t in tags:
                    if t not in genre_tags:
                        genre_tags.append(t)
    if not genre_tags:
        genre_tags = ["electronic"]
    genre_tags = genre_tags[:2]

    # ── v6: 4-dimensional mood mapping ────────────────────────────────────────
    spotify_f  = r.get("spotify_like_features", {})
    valence    = spotify_f.get("valence",      0.5)
    energy     = spotify_f.get("energy",       0.5)
    dance      = spotify_f.get("danceability", 0.5)
    mode       = r.get("tonality", {}).get("mode", "minor")
    modal      = r.get("tonality", {}).get("modal_flavor", "")
    bpm        = r.get("summary",  {}).get("tempo_bpm", 120)

    # 4-D mood matrix: (dance, valence, energy, mode) → mood word
    if dance > 0.6 and valence > 0.5 and energy > 0.6:
        mood_word = "euphoric"
    elif dance > 0.6 and valence > 0.5 and energy <= 0.6:
        mood_word = "groovy"
    elif dance > 0.6 and valence <= 0.5 and energy > 0.6:
        mood_word = "driving"
    elif dance > 0.6 and valence <= 0.5 and energy <= 0.6:
        mood_word = "hypnotic"
    elif dance <= 0.6 and valence > 0.5 and energy > 0.6:
        mood_word = "uplifting"
    elif dance <= 0.6 and valence > 0.5 and energy <= 0.6:
        mood_word = "serene"
    elif dance <= 0.6 and valence <= 0.5 and energy > 0.6:
        mood_word = "intense" if mode == "minor" else "powerful"
    else:
        # dance<=0.6, valence<=0.5, energy<=0.6
        if modal in ("Phrygian",):
            mood_word = "dark and ominous"
        elif modal in ("Dorian",):
            mood_word = "melancholic"
        elif modal in ("Aeolian (natural minor)",):
            mood_word = "brooding"
        else:
            mood_word = "contemplative"

    mood_tags = [mood_word]

    # ── Bass character ────────────────────────────────────────────────────────
    sub_pct  = r.get("frequency_bands",{}).get("sub_bass_20_60hz",{}).get("energy_pct", 0)
    bass_pct = r.get("frequency_bands",{}).get("bass_60_250hz",{}).get("energy_pct", 0)
    is_hiphop = any("trap" in t.lower() or "hip" in t.lower() or "boom" in t.lower() for t in genre_tags)
    is_dnb    = any("drum and bass" in t.lower() or "breakbeat" in t.lower() for t in genre_tags)
    is_orch   = any("orchestral" in t.lower() for t in genre_tags)

    bass_tags = []
    if is_orch:
        pass  # no bass tags for orchestral
    elif is_hiphop:
        if sub_pct > 15:   bass_tags = ["heavy 808 bass"]
        elif sub_pct > 8:  bass_tags = ["808 bass"]
        elif bass_pct > 30: bass_tags = ["punchy bass"]
    elif is_dnb:
        if sub_pct > 20:   bass_tags = ["heavy sub bass", "Reese bass"]
        elif sub_pct > 10: bass_tags = ["rolling bass"]
    else:
        if sub_pct > 15:   bass_tags = ["heavy sub bass"]
        elif sub_pct > 8:  bass_tags = ["deep sub bass"]
        elif bass_pct > 40: bass_tags = ["warm deep bass"]
        elif bass_pct > 20: bass_tags = ["punchy bass"]

    # ── Drum feel ─────────────────────────────────────────────────────────────
    dp       = r.get("drum_pattern", {})
    kick_pb  = dp.get("kick_per_bar", 0)
    hihat_pb = dp.get("hihat_per_bar", 0)
    snare_pb = dp.get("snare_per_bar", 0)

    drum_tags = []
    if is_orch:
        pass  # no electronic drum tags for orchestral
    elif is_hiphop:
        if hihat_pb >= 5:   drum_tags = ["rolling trap hi-hats"]
        elif hihat_pb >= 3: drum_tags = ["trap hi-hats"]
        if snare_pb >= 3:   drum_tags += ["snappy snare"]
    elif is_dnb:
        drum_tags = ["breakbeat drums", "jungle snare rolls"]
    else:
        if kick_pb >= 3.5:   drum_tags = ["four-on-the-floor kick"]
        elif kick_pb >= 2:   drum_tags = ["punchy kick"]
        if hihat_pb >= 6:    drum_tags += ["16th hi-hats"]
        elif hihat_pb >= 3:  drum_tags += ["offbeat hi-hats"]
        elif hihat_pb >= 1 and not drum_tags: drum_tags = ["electronic drums"]
    drum_tags = drum_tags[:2]

    # ── Tempo descriptor ──────────────────────────────────────────────────────
    if bpm < 75:     tempo_tag = "downtempo"
    elif bpm < 95:   tempo_tag = "mid-tempo"
    elif bpm < 115:  tempo_tag = "uptempo"
    elif bpm < 130:  tempo_tag = "driving"
    elif bpm < 155:  tempo_tag = "high-energy"
    else:            tempo_tag = "frenetic energy"

    # ── Structural tag ────────────────────────────────────────────────────────
    ec       = r.get("energy_curve", [])
    rms_vals = [e["rms_db"] for e in ec]
    has_drop = False
    if len(rms_vals) > 6:
        global_min_idx = rms_vals.index(min(rms_vals))
        global_max_idx = rms_vals.index(max(rms_vals))
        valley = rms_vals[global_min_idx]
        peak   = rms_vals[global_max_idx]
        avg    = sum(rms_vals) / len(rms_vals)
        if valley < avg - 6 and global_max_idx > global_min_idx and peak > valley + 8:
            has_drop = True
    struct_tags = ["build and drop"] if has_drop else []

    # ── Texture ───────────────────────────────────────────────────────────────
    crest     = r.get("loudness", {}).get("crest_factor_db", 12)
    hp        = r.get("harmonic_percussive", {})
    h_ratio   = hp.get("harmonic_ratio", 0.5)
    brightness= r.get("spectral", {}).get("brightness", "mid")
    is_edm    = any(t.lower() in {"tech house","deep house","trance","edm","techno","drum and bass"} for t in genre_tags)

    texture_tags = []
    if is_orch:
        texture_tags.append("lush orchestration")
    elif h_ratio > 0.8 and is_edm:
        texture_tags.append("lush atmosphere")
    elif h_ratio > 0.8:
        texture_tags.append("atmospheric")
    elif h_ratio < 0.3:
        texture_tags.append("raw percussive")
    if brightness == "dark" and "dark" not in mood_word:
        texture_tags.append("dark production")
    if crest > 16:    texture_tags.append("wide dynamics")
    elif crest < 7:   texture_tags.append("heavily compressed")
    texture_tags = texture_tags[:1]

    # ── Lyrics context ────────────────────────────────────────────────────────
    lyrics     = r.get("lyrics", {})
    has_lyrics = lyrics.get("has_lyrics", False)
    lyric_tags = []
    if has_lyrics:
        lyric_mood = lyrics.get("lyric_mood")
        if lyric_mood and lyric_mood not in mood_tags and lyric_mood not in " ".join(genre_tags):
            lyric_tags.append(lyric_mood)
    else:
        lyric_tags.append("instrumental")

    # ── Assemble prompt ───────────────────────────────────────────────────────
    LIMIT       = 200
    STYLE_LIMIT = 120 if has_lyrics else LIMIT

    all_slots = [
        genre_tags, mood_tags, bass_tags, drum_tags,
        [tempo_tag], struct_tags, lyric_tags, texture_tags,
    ]
    selected = []
    for slot in all_slots:
        for tag in slot:
            candidate = ", ".join(selected + [tag])
            if len(candidate) <= STYLE_LIMIT:
                selected.append(tag)

    prompt = ", ".join(selected)
    while len(prompt) > STYLE_LIMIT:
        selected.pop()
        prompt = ", ".join(selected)

    # ── LLM synthesis hint ────────────────────────────────────────────────────
    spectral = r.get("spectral", {})
    lufs_val = r.get("summary", {}).get("lufs", -14)
    llm_hint = {
        "note": "Claude should use this data to validate or refine the rule-based prompt above.",
        "key_signals": {
            "bpm":                   bpm,
            "key_mode":              r.get("summary", {}).get("key_full", ""),
            "genre_scores_top5":     dict(list(genre_scores.items())[:5]),
            "sub_bass_pct":          round(sub_pct, 1),
            "bass_pct":              round(bass_pct, 1),
            "spectral_centroid_hz":  spectral.get("centroid_hz", 0),
            "brightness":            brightness,
            "harmonic_ratio":        round(h_ratio, 3),
            "kick_per_bar":          kick_pb,
            "hihat_per_bar":         hihat_pb,
            "crest_factor_db":       crest,
            "has_build_drop":        has_drop,
            "lyrics_language":       lyrics.get("language"),
            "lyric_mood":            lyrics.get("lyric_mood"),
            "top_keywords":          lyrics.get("top_keywords", [])[:5],
            "essentia_key":          r.get("essentia_features", {}).get("key_extractor", {}),
            "essentia_lufs":         r.get("essentia_features", {}).get("lufs_integrated"),
            "essentia_danceability": r.get("essentia_features", {}).get("danceability"),
            "spotify_features":      r.get("spotify_like_features", {}),
        },
        "rule_based_prompt": prompt,
        "refinement_guidance": (
            "If the rule-based prompt tags feel generic or mismatched to the actual sound, "
            "replace them with more accurate Suno style tags. "
            "Prioritize: (1) precise genre sub-label over generic ones, "
            "(2) mood adjectives that match both the harmonic color AND the production energy, "
            "(3) instrument/production descriptors (e.g. 'Rhodes piano', 'modular synth', "
            "'acoustic guitar', 'distorted 808') when they're clearly audible in the mix. "
            "v6: Check spotify_features.valence+danceability+energy for precise mood calibration."
        ),
    }

    reasoning = {
        "genres_from_analysis": likely,
        "genre_scores":         genre_scores,
        "modal_flavor":         modal,
        "sub_bass_pct":         round(sub_pct, 1),
        "kick_per_bar":         kick_pb,
        "hihat_per_bar":        hihat_pb,
        "bpm":                  bpm,
        "is_edm_context":       is_edm,
        "is_hiphop_context":    is_hiphop,
        "is_dnb_context":       is_dnb,
        "is_orchestral":        is_orch,
        "has_build_drop":       has_drop,
        "crest_db":             crest,
        "h_ratio":              round(h_ratio, 3),
        "lyrics_language":      lyrics.get("language"),
        "lyric_mood_detected":  lyrics.get("lyric_mood"),
        "v6_mood_inputs":       {"valence": valence, "energy": energy, "danceability": dance, "mode": mode},
    }

    mode_str = "lyrics" if has_lyrics else "instrumental"
    return {
        "mode":            mode_str,
        "style_char_limit": STYLE_LIMIT,
        "prompt":          prompt,
        "char_count":      len(prompt),
        "within_limit":    len(prompt) <= STYLE_LIMIT,
        "note": ("Lyrics mode: paste this into Suno 'Style of Music' (<=120 chars). "
                 "Paste original lyrics into 'Lyrics' field (<=3000 chars)."
                 if mode_str == "lyrics" else
                 "Instrumental mode: paste this into Suno 'Style of Music' (<=200 chars)."),
        "llm_synthesis_hint": llm_hint,
        "reasoning":          reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer,)):                          return int(obj)
        if isinstance(obj, np.ndarray):                             return obj.tolist()
        return super().default(obj)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: analyze_audio.py <path>"})); sys.exit(1)
    try:
        print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, indent=2, cls=NumpyEncoder))
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}))
        sys.exit(1)
