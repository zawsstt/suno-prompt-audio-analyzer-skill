#!/usr/bin/env python3
"""
audio-analyzer v2: Professional audio analysis for music production reference
Usage: python3 analyze_audio.py <audio_file_path>
Outputs: JSON with comprehensive production-grade analysis

Memory-optimized: loads at 22050Hz mono for analysis; uses ffprobe for metadata.
"""

import sys, json, warnings
warnings.filterwarnings("ignore")

KEY_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
KS_MAJOR  = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
KS_MINOR  = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

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
    if M3>thr and p5>thr:  return f"{KEY_NAMES[root]}"
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
    """
    If the file is not natively supported by librosa/soundfile, convert it
    to a temporary 48kHz stereo WAV using ffmpeg.
    Supports: MP4, M4A, AAC, WMA, OPUS, WEBM, MOV, MKV, AVI, etc.
    Returns (wav_path, is_temp) — caller must clean up if is_temp=True.
    """
    import subprocess, tempfile, os
    ext = os.path.splitext(filepath)[1].lower()
    if ext in LIBROSA_NATIVE_EXTS:
        return filepath, False
    # Convert via ffmpeg → tmp WAV
    tmp = tempfile.mktemp(suffix=".wav", dir="/tmp")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", filepath,
         "-vn",                  # drop video track
         "-acodec", "pcm_s16le",
         "-ar", "48000",
         tmp],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {r.stderr.decode()[:300]}")
    return tmp, True


def analyze(filepath):
    import numpy as np, librosa, scipy.signal, os
    from scipy.ndimage import uniform_filter1d
    from collections import Counter
    result = {}

    # ── 0a. Format normalisation — convert non-WAV containers to WAV ─────────
    import subprocess
    wav_path, is_temp_wav = ensure_wav(filepath)

    # ── 0b. File metadata ─────────────────────────────────────────────────────
    probe = subprocess.run(
        ["ffprobe","-v","quiet","-print_format","json",
         "-show_format","-show_streams", filepath],
        capture_output=True, text=True)
    if probe.returncode == 0:
        meta = json.loads(probe.stdout)
        fmt = meta.get("format",{})
        astream = next((s for s in meta.get("streams",[]) if s.get("codec_type")=="audio"),{})
        result["file_info"] = {
            "format": fmt.get("format_long_name", fmt.get("format_name","?")),
            "duration_sec": round(float(fmt.get("duration",0)),2),
            "size_bytes": int(fmt.get("size",0)),
            "bitrate_kbps": round(int(fmt.get("bit_rate",0))/1000,1),
            "codec": astream.get("codec_name","?"),
            "native_sample_rate_hz": int(astream.get("sample_rate",0)),
            "channels": astream.get("channels",0),
            "channel_layout": astream.get("channel_layout","?"),
        }

    # ── 1. Load (22050 Hz mono for analysis; stereo separately) ─────────────
    SR = 22050
    y, sr = librosa.load(wav_path, sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Stereo info via ffprobe frames (avoid loading full stereo into RAM)
    # We approximate from the file metadata already captured
    ch = result.get("file_info",{}).get("channels",1)
    result["channel_analysis"] = {"type": "stereo" if ch >= 2 else "mono",
                                   "note": "Stereo correlation skipped for memory efficiency; use DAW for precise M/S analysis"}

    # ── 2. Loudness & dynamics ────────────────────────────────────────────────
    rms_v  = float(np.sqrt(np.mean(y**2)))
    peak_v = float(np.max(np.abs(y)))
    rms_db  = float(librosa.amplitude_to_db(np.array([rms_v+1e-9]))[0])
    peak_db = float(librosa.amplitude_to_db(np.array([peak_v+1e-9]))[0])
    lufs_approx = round(rms_db - 0.691, 2)
    crest_db = round(peak_db - rms_db, 2)
    hop_1s = sr
    rms_wins = [float(np.sqrt(np.mean(y[i:i+hop_1s]**2))) for i in range(0,len(y)-hop_1s,hop_1s)]
    rms_db_wins = librosa.amplitude_to_db(np.array(rms_wins)+1e-9)
    dr = round(float(np.percentile(rms_db_wins,95)-np.percentile(rms_db_wins,5)),2)
    result["loudness"] = {
        "rms_db": round(rms_db,2), "peak_db": round(peak_db,2),
        "lufs_approx": lufs_approx,
        "crest_factor_db": crest_db,
        "dynamic_range_db": dr,
        "headroom_db": round(-peak_db,2),
        "compression_hint": "heavily compressed" if crest_db<8 else ("moderate" if crest_db<14 else "wide dynamics"),
    }

    # ── 3. Frequency band energy ──────────────────────────────────────────────
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
        band_e[name] = {"energy_pct": round(e/total_e*100,2),
                        "db": round(float(librosa.amplitude_to_db(np.array([np.sqrt(e/(mask.sum()+1))]))[0]),1)}
    result["frequency_bands"] = band_e

    # ── 4. Rhythm & beat grid ─────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times)>4:
        ivs = np.diff(beat_times)
        cv = float(np.std(ivs)/np.mean(ivs))
        swing_hint = "straight" if cv<0.05 else ("light swing" if cv<0.12 else "heavy swing/rubato")
    else:
        cv=0.0; swing_hint="sparse"
    # Time sig via autocorrelation
    oe = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(oe, max_size=sr//2)
    tf = int(round(sr/(bpm/60)/512))
    beat4 = ac[tf*4] if tf*4<len(ac) else 0
    beat3 = ac[tf*3] if tf*3<len(ac) else 0
    time_sig = "4/4" if beat4>=beat3 else "3/4"

    result["rhythm"] = {
        "tempo_bpm": round(bpm,1),
        "beat_count": len(beat_times),
        "estimated_time_signature": time_sig,
        "swing_type": swing_hint,
        "beat_interval_cv": round(cv,4),
        "beat_times_first16": [round(float(t),3) for t in beat_times[:16]],
    }

    # ── 5. Drum pattern (kick/snare/hihat) ────────────────────────────────────
    _, y_perc = librosa.effects.hpss(y)
    def bp_filter(sig, lo, hi, fs):
        nyq = fs/2
        lo_n = max(lo/nyq, 0.001); hi_n = min(hi/nyq, 0.999)
        sos = scipy.signal.butter(4,[lo_n,hi_n],btype='band',output='sos')
        return scipy.signal.sosfilt(sos, sig)

    hop = 256
    bar_dur = 60.0/bpm*4
    end_s   = min(bar_dur*8, 20.0)
    end_smp = int(end_s*sr)

    def drum_onsets(sig, sr, hop, min_dist=0.08):
        frames = librosa.util.frame(sig, frame_length=hop*2, hop_length=hop)
        rms_arr = np.sqrt(np.mean(frames**2, axis=0))
        thr = np.percentile(rms_arr,80)
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
        "kick_per_bar":  round(len(kick_pat)/(end_s/bar_dur),2) if bar_dur>0 else 0,
        "snare_per_bar": round(len(snare_pat)/(end_s/bar_dur),2) if bar_dur>0 else 0,
        "hihat_per_bar": round(len(hihat_pat)/(end_s/bar_dur),2) if bar_dur>0 else 0,
    }

    # ── 6. Bass line (low-pass + pyin on first 90s) ───────────────────────────
    y90 = y[:int(90*sr)]
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
        "bass_sequence_start":  bass_seq,
    }
    del y_bass, y90

    # ── 7. Chord progression ──────────────────────────────────────────────────
    y_harm, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, bins_per_octave=36)
    beat_chroma = librosa.util.sync(chroma, beats, aggregate=np.median)
    chord_labels = [chord_label(beat_chroma[:,i]) for i in range(beat_chroma.shape[1])]
    chord_seq = [{"chord":k,"beats":v} for k,v in rle(chord_labels)[:32]]
    bigrams = [(chord_seq[i]["chord"], chord_seq[i+1]["chord"]) for i in range(len(chord_seq)-1)]
    common_trans = [{"from":a,"to":b,"count":c} for (a,b),c in Counter(bigrams).most_common(8)]
    # Unique chord set
    unique_chords = list(dict.fromkeys(c["chord"] for c in chord_seq))
    result["chord_progression"] = {
        "sequence": chord_seq[:24],
        "unique_chords_used": unique_chords[:12],
        "common_transitions": common_trans,
    }

    # ── 8. Key, mode, scale degrees ──────────────────────────────────────────
    chroma_mean = chroma.mean(axis=1)
    root_name, mode, ks_score = ks_key(chroma_mean)
    root_idx = KEY_NAMES.index(root_name)
    modal     = detect_mode(chroma_mean, root_idx)
    rel_idx   = (root_idx+3)%12 if mode=="minor" else (root_idx-3)%12
    relative  = f"{KEY_NAMES[rel_idx]} {'major' if mode=='minor' else 'minor'}"
    rotated   = np.roll(chroma_mean, -root_idx); rotated /= rotated.max()+1e-9
    degree_labels = ["1","b2","2","b3","3","4","b5","5","b6","6","b7","7"]
    strong_deg = [degree_labels[i] for i in range(12) if rotated[i]>0.5]

    result["tonality"] = {
        "key": f"{root_name} {mode}",
        "root": root_name, "mode": mode,
        "modal_flavor": modal,
        "ks_confidence": ks_score,
        "relative_key": relative,
        "strong_scale_degrees": strong_deg,
        "chroma_profile": {k:round(float(v),4) for k,v in zip(KEY_NAMES,chroma_mean)},
    }

    # ── 9. Key timeline (per 15s) ─────────────────────────────────────────────
    key_tl = []
    prev = None
    for i in range(int(np.ceil(duration/15))):
        s0=int(i*15*sr); s1=int(min((i+1)*15*sr,len(y)))
        if s1-s0 < sr: continue
        cm = librosa.feature.chroma_cqt(y=y[s0:s1],sr=sr).mean(axis=1)
        rn,rm,rs = ks_key(cm)
        ks = f"{rn} {rm}"
        t0s = f"{int(i*15//60):02d}:{int(i*15%60):02d}"
        entry = {"time":t0s,"key":ks,"conf":rs}
        if ks!=prev: entry["changed"]=True; prev=ks
        key_tl.append(entry)
    result["key_timeline"] = key_tl

    # ── 10. Melodic contour (harmonic channel, first 90s) ────────────────────
    y_harm90 = y_harm[:int(90*sr)]
    try:
        f0m, vm, _ = librosa.pyin(y_harm90, fmin=80, fmax=2000, sr=sr,
                                    frame_length=2048, hop_length=512)
        mel_notes = [f0_to_note(f) for f in f0m[vm]]
        mel_seq   = [{"note":k,"frames":v} for k,v in rle(mel_notes)[:32] if k!="?"]
        mel_top   = [{"note":k,"count":v} for k,v in Counter(n for n in mel_notes if n!="?").most_common(10)]
        valid_f0  = f0m[vm & ~np.isnan(f0m)]
        pr = {"lowest_note": f0_to_note(valid_f0.min()) if len(valid_f0)>0 else "?",
              "highest_note": f0_to_note(valid_f0.max()) if len(valid_f0)>0 else "?",
              "range_hz": f"{round(float(valid_f0.min()),1) if len(valid_f0)>0 else 0}–{round(float(valid_f0.max()),1) if len(valid_f0)>0 else 0}"}
    except Exception as e:
        mel_seq=[]; mel_top=[]; pr={"error":str(e)}
    del y_harm90
    result["melodic_contour"] = {
        "pitch_range": pr,
        "dominant_melody_notes": mel_top,
        "melody_sequence_start": mel_seq,
    }

    # ── 11. Energy / tension curve (per 5s) ──────────────────────────────────
    ecurve = []
    for i in range(int(np.ceil(duration/5))):
        s0=int(i*5*sr); s1=int(min((i+1)*5*sr,len(y)))
        if s1==s0: continue
        chunk=y[s0:s1]
        rv = float(np.sqrt(np.mean(chunk**2)))
        rd = float(librosa.amplitude_to_db(np.array([rv+1e-9]))[0])
        spec = np.abs(librosa.stft(chunk,n_fft=512))
        flux = float(np.mean(np.diff(spec,axis=1)**2)) if spec.shape[1]>1 else 0.0
        ecurve.append({"time":f"{int(i*5//60):02d}:{int(i*5%60):02d}",
                       "rms_db":round(rd,1),"spectral_flux":round(flux,4)})
    result["energy_curve"] = ecurve

    # ── 12. Structure (energy drop / peak detection) ──────────────────────────
    rms_env = librosa.feature.rms(y=y, frame_length=sr//2, hop_length=sr//4)[0]
    rms_sm  = uniform_filter1d(librosa.amplitude_to_db(rms_env+1e-9), size=8)
    drops=[]
    for i in range(4,len(rms_sm)-4):
        if rms_sm[i]<rms_sm[i-4]-3 and rms_sm[i]<rms_sm[i+4]-3:
            drops.append(round(float(librosa.frames_to_time(i,sr=sr,hop_length=sr//4)),1))
    pk_idx,_ = scipy.signal.find_peaks(rms_sm, prominence=3, distance=8)
    pk_times = [round(float(librosa.frames_to_time(p,sr=sr,hop_length=sr//4)),1) for p in pk_idx]
    result["structure"] = {
        "energy_drop_times_sec": drops[:10],
        "energy_peak_times_sec": pk_times[:10],
        "tip": "Drops = likely section breaks (verse↔chorus); Peaks = climax moments",
    }

    # ── 13. Spectral profile ──────────────────────────────────────────────────
    sc  = librosa.feature.spectral_centroid(y=y,sr=sr)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y,sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(y=y,sr=sr)[0]
    sf  = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc= librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20)
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

    # ── 14. Harmonic/Percussive ratio ─────────────────────────────────────────
    he=float(np.mean(y_harm**2)); pe=float(np.mean(y_perc**2)); te=he+pe+1e-12
    result["harmonic_percussive"] = {
        "harmonic_ratio":   round(he/te,4),
        "percussive_ratio": round(pe/te,4),
        "dominant": "harmonic" if he>pe else "percussive",
    }

    # ── 15. Production style assessment ──────────────────────────────────────
    sub_pct  = band_e["sub_bass_20_60hz"]["energy_pct"]
    air_pct  = band_e["air_10k_plus"]["energy_pct"]
    crest    = result["loudness"]["crest_factor_db"]

    genres=[]
    if 118<=bpm<=135 and sub_pct>2:    genres.append("House / Deep House")
    if 126<=bpm<=145 and avg_c>3000:   genres.append("Trance / Progressive")
    if bpm>=140 and pe/te>0.3:          genres.append("Drum & Bass / Breakbeat")
    if bpm<=90 and he/te>0.7:           genres.append("Ballad / Ambient / Neo-soul")
    if 85<=bpm<=115 and avg_c<2500:     genres.append("R&B / Soul / Lo-fi")
    if 100<=bpm<=130 and avg_c>2000:    genres.append("Pop / Indie Pop")
    if bpm>=130 and avg_c>4000:         genres.append("Hard Dance / Techno")
    if sub_pct>8:                        genres.append("Trap / Hip-hop (heavy sub)")
    if he/te>0.85 and bpm<100:          genres.append("Cinematic / Orchestral")
    if not genres: genres=["Electronic / Instrumental (mixed)"]

    result["production_style"] = {
        "likely_genres": genres,
        "mastering": "heavily limited" if crest<8 else ("radio-ready" if crest<14 else "wide dynamic range"),
        "daw_tips": [
            f"Set project BPM to {round(bpm,1)}, time signature {time_sig}",
            f"Program in {root_name} {mode} — scale: {modal}",
            f"Reference key: {relative} (for modulations)",
            f"Sub bass ({round(sub_pct,1)}% energy) — {'boost sub' if sub_pct<3 else 'sub already prominent'}",
            f"Mix brightness target: ~{round(avg_c)}Hz spectral centroid",
            f"Aim for ~{round(lufs_approx,1)} LUFS integrated loudness",
            f"Compression depth: crest factor {crest}dB → {'use heavy parallel compression' if crest>14 else 'moderate limiting'}",
        ],
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    result["summary"] = {
        "duration": f"{int(duration//60):02d}:{int(duration%60):02d}",
        "key_full": f"{root_name} {mode} ({modal})",
        "relative_key": relative,
        "tempo_bpm": round(bpm,1),
        "time_signature": time_sig,
        "swing": swing_hint,
        "lufs_approx": lufs_approx,
        "dynamic_range_db": dr,
        "brightness": result["spectral"]["brightness"],
        "harmonic_dominant": he>pe,
        "likely_genres": genres,
    }

    # ── Lyrics Analysis (faster-whisper) ─────────────────────────────────────
    result["lyrics"] = extract_lyrics(filepath)

    # ── Suno Prompt Generation ────────────────────────────────────────────────
    result["suno_prompt"] = generate_suno_prompt(result)

    # ── Cleanup temp WAV (if we converted from MP4/M4A/etc.) ─────────────────
    if is_temp_wav:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    return result


def extract_lyrics(path):
    """
    Transcribe vocals using faster-whisper (small model, CPU int8).
    Returns lyrics timeline, language, theme keywords, structure hints,
    and a cleaned full-text for Suno lyric scaffold.
    """
    import subprocess, re, collections

    # ── Step 1: convert to 16kHz mono WAV (whisper requirement) ──────────────
    tmp_wav = "/tmp/_whisper_input.wav"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp_wav],
        capture_output=True
    )
    if r.returncode != 0:
        return {"error": "ffmpeg conversion failed", "detail": r.stderr.decode()[:300]}

    # ── Step 2: transcribe ────────────────────────────────────────────────────
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

    # ── Step 3: filter noise / music placeholders ─────────────────────────────
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

    # ── Step 4: full lyrics text ──────────────────────────────────────────────
    full_text = " / ".join(s["text"] for s in lyric_segs)

    # ── Step 5: theme / keyword extraction (simple frequency) ────────────────
    stopwords = {"the","a","an","and","or","but","in","on","at","to","for",
                 "of","with","is","are","was","were","be","been","have","has",
                 "had","we","i","you","he","she","it","they","this","that",
                 "all","just","no","not","so","when","as","if","our","their",
                 "my","your","its","what","there","will","can","do","dont",
                 "im","its","its","set","one","like"}
    words = re.findall(r"[a-z']+", full_text.lower())
    word_freq = collections.Counter(w for w in words if w not in stopwords and len(w) > 2)
    top_keywords = [w for w, _ in word_freq.most_common(10)]

    # ── Step 6: structure detection ──────────────────────────────────────────
    # Find repeated lines → likely chorus
    line_texts = [s["text"].lower().strip() for s in lyric_segs]
    line_freq = collections.Counter(line_texts)
    repeated_lines = [l for l, c in line_freq.most_common(5) if c >= 2]

    # Estimate vocal density: lyric segments per minute
    total_dur_min = lyric_segs[-1]["start_sec"] / 60 if lyric_segs else 1
    vocal_density = round(len(lyric_segs) / max(total_dur_min, 0.1), 1)

    # ── Step 7: language → Suno tag ──────────────────────────────────────────
    lang_tag_map = {
        "en": "English lyrics", "zh": "Chinese lyrics", "ja": "Japanese lyrics",
        "ko": "Korean lyrics",  "es": "Spanish lyrics", "fr": "French lyrics",
        "de": "German lyrics",  "pt": "Portuguese lyrics",
    }
    lang_tag = lang_tag_map.get(info.language, f"{info.language} lyrics")

    # ── Step 8: mood from lyrics (simple keyword scan) ────────────────────────
    lyric_lower = full_text.lower()
    lyric_mood = None
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
    }


def generate_suno_prompt(r):
    """
    Synthesize analysis data into a Suno-compatible style prompt ≤200 characters.
    Returns dict with 'prompt', 'char_count', and 'reasoning'.
    """

    # ── 1. Genre slots ────────────────────────────────────────────────────────
    # Priority order: specific electronic genres beat generic "pop"
    genre_priority = [
        ("House / Deep House",           ["deep house"]),
        ("Tech House",                   ["tech house"]),
        ("Trance / Progressive",         ["trance"]),
        ("Drum & Bass / Breakbeat",      ["drum and bass"]),
        ("Hard Dance / Techno",          ["techno"]),
        ("Trap / Hip-hop (heavy sub)",   ["trap"]),
        ("Ballad / Ambient / Neo-soul",  ["neo-soul"]),
        ("R&B / Soul / Lo-fi",           ["lo-fi"]),
        ("Cinematic / Orchestral",       ["cinematic"]),
        ("Electronic / Instrumental (mixed)", ["electronic"]),
        ("Pop / Indie Pop",              ["indie pop"]),  # lowest priority
    ]
    likely = r.get("production_style", {}).get("likely_genres", [])
    genre_tags = []
    for key, tags in genre_priority:
        for g in likely:
            if key in g:
                for t in tags:
                    if t not in genre_tags:
                        genre_tags.append(t)
    if not genre_tags:
        genre_tags = ["electronic"]
    # Max 2 genres — prefer the two most specific
    genre_tags = genre_tags[:2]

    # ── 2. Modal mood ─────────────────────────────────────────────────────────
    modal = r.get("tonality", {}).get("modal_flavor", "")
    mode  = r.get("tonality", {}).get("mode", "minor")
    modal_mood_map = {
        "Phrygian":             ["dark", "brooding"],
        "Dorian":               ["melancholic", "moody"],
        "Aeolian (natural minor)": ["melancholic", "dark"],
        "Mixolydian":           ["uplifting", "groovy"],
        "Ionian (natural major)":  ["bright", "uplifting"],
    }
    mood_tags = modal_mood_map.get(modal, ["dark"] if mode == "minor" else ["uplifting"])
    mood_tags = mood_tags[:1]  # 1 mood word

    # ── 3. Bass character ─────────────────────────────────────────────────────
    sub_pct  = r.get("frequency_bands",{}).get("sub_bass_20_60hz",{}).get("energy_pct", 0)
    bass_pct = r.get("frequency_bands",{}).get("bass_60_250hz",{}).get("energy_pct", 0)
    bass_tags = []
    if sub_pct > 15:   bass_tags = ["heavy sub bass"]
    elif sub_pct > 8:  bass_tags = ["808 bass"]
    elif bass_pct > 40: bass_tags = ["deep bass"]
    elif bass_pct > 20: bass_tags = ["punchy bass"]

    # ── 4. Drum feel ──────────────────────────────────────────────────────────
    dp = r.get("drum_pattern", {})
    kick_pb  = dp.get("kick_per_bar", 0)
    hihat_pb = dp.get("hihat_per_bar", 0)
    drum_tags = []
    if kick_pb >= 3.5:      drum_tags = ["four-on-the-floor"]
    elif kick_pb >= 2:      drum_tags = ["punchy drums"]
    if hihat_pb >= 4:       drum_tags += ["trap hi-hats"]
    elif hihat_pb >= 2 and not drum_tags: drum_tags = ["electronic drums"]
    drum_tags = drum_tags[:2]

    # ── 5. Tempo descriptor ───────────────────────────────────────────────────
    bpm = r.get("summary", {}).get("tempo_bpm", 120)
    if bpm < 75:       tempo_tag = "downtempo"
    elif bpm < 100:    tempo_tag = "mid-tempo"
    elif bpm < 120:    tempo_tag = "uptempo"
    elif bpm < 140:    tempo_tag = "driving"
    else:              tempo_tag = "high-energy"

    # ── 6. Structural tag ─────────────────────────────────────────────────────
    ec = r.get("energy_curve", [])
    rms_vals = [e["rms_db"] for e in ec]
    has_drop = False
    if len(rms_vals) > 6:
        # Look for: a quiet valley anywhere in the track followed by a loud peak
        global_min_idx = rms_vals.index(min(rms_vals))
        global_max_idx = rms_vals.index(max(rms_vals))
        valley = rms_vals[global_min_idx]
        peak   = rms_vals[global_max_idx]
        avg    = sum(rms_vals) / len(rms_vals)
        # Drop = valley is ≥6dB below average AND a loud section follows it
        if valley < avg - 6 and global_max_idx > global_min_idx and peak > valley + 8:
            has_drop = True
    struct_tags = ["build and drop"] if has_drop else []

    # ── 7. Texture / mix ─────────────────────────────────────────────────────
    crest  = r.get("loudness", {}).get("crest_factor_db", 12)
    hp     = r.get("harmonic_percussive", {})
    h_ratio = hp.get("harmonic_ratio", 0.5)
    brightness = r.get("spectral", {}).get("brightness", "mid")

    texture_tags = []
    if h_ratio > 0.8:        texture_tags.append("atmospheric")
    elif h_ratio < 0.4:      texture_tags.append("percussive")
    if brightness == "dark":  texture_tags.append("dark mix")
    if crest > 14:            texture_tags.append("wide dynamics")
    elif crest < 8:           texture_tags.append("compressed")
    texture_tags = texture_tags[:1]

    # ── 8. Assemble with budget tracking ─────────────────────────────────────
    LIMIT = 200
    # Priority order: genre, mood, bass, drums, tempo, structure, texture
    all_slots = [
        genre_tags,
        mood_tags,
        bass_tags,
        drum_tags,
        [tempo_tag],
        struct_tags,
        texture_tags,
    ]

    selected = []
    for slot in all_slots:
        for tag in slot:
            candidate = ", ".join(selected + [tag])
            if len(candidate) <= LIMIT:
                selected.append(tag)

    prompt = ", ".join(selected)
    # Final safety truncation (should not be needed)
    while len(prompt) > LIMIT:
        selected.pop()
        prompt = ", ".join(selected)

    # ── 9. Lyrics-derived tags + dual-prompt mode ────────────────────────────
    lyrics = r.get("lyrics", {})
    has_lyrics = lyrics.get("has_lyrics", False)
    lyric_tags = []
    if has_lyrics:
        # Lyrics mode: do NOT add language tag to style prompt (it's in the Lyrics field)
        # Only add lyric mood if not already covered
        lyric_mood = lyrics.get("lyric_mood")
        if lyric_mood and lyric_mood not in mood_tags and lyric_mood not in genre_tags:
            lyric_tags.append(lyric_mood)
    else:
        lyric_tags.append("instrumental")

    # Insert lyric tags into all_slots
    all_slots = [
        genre_tags,
        mood_tags,
        bass_tags,
        drum_tags,
        [tempo_tag],
        struct_tags,
        lyric_tags,
        texture_tags,
    ]

    # ── Dual-prompt mode: lyrics → 120 char style limit; instrumental → 200 ──
    STYLE_LIMIT = 120 if has_lyrics else LIMIT  # LIMIT=200 defined earlier

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

    reasoning = {
        "genres_from_analysis": likely,
        "modal_flavor": modal,
        "sub_bass_pct": round(sub_pct, 1),
        "kick_per_bar": kick_pb,
        "hihat_per_bar": hihat_pb,
        "bpm": bpm,
        "has_build_drop": has_drop,
        "crest_db": crest,
        "h_ratio": round(h_ratio, 3),
        "lyrics_language": lyrics.get("language"),
        "lyric_mood_detected": lyrics.get("lyric_mood"),
    }

    mode = "lyrics" if has_lyrics else "instrumental"
    return {
        "mode": mode,
        "style_char_limit": STYLE_LIMIT,
        "prompt": prompt,
        "char_count": len(prompt),
        "within_limit": len(prompt) <= STYLE_LIMIT,
        "note": ("Lyrics mode: paste this into Suno 'Style of Music' (≤120 chars). "
                 "Paste original lyrics into 'Lyrics' field (≤3000 chars)."
                 if mode == "lyrics" else
                 "Instrumental mode: paste this into Suno 'Style of Music' (≤200 chars)."),
        "reasoning": reasoning,
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
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
