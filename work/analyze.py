#!/usr/bin/env python3
"""
Forensic analysis of the four Wailing Wailers source files.
Measures: bandwidth ceiling & codec signature, DC offset, hum, clipping,
L/R correlation & stereo content, noise floor, loudness, crest factor,
spectral tilt, and per-band energy distribution.
"""
import sys, os, json, glob
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal

SR_ANALYZE = 44100


def load(path):
    x, sr = sf.read(path, always_2d=True, dtype='float64')
    if sr != SR_ANALYZE:
        raise SystemExit(f'{path}: unexpected sr {sr}')
    return x, sr


def dbfs(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def welch_avg(x, sr, nperseg=8192):
    f, P = signal.welch(x, sr, nperseg=nperseg, noverlap=nperseg // 2,
                        window='hann', detrend=False, scaling='density')
    return f, P


def highfreq_ceiling(x, sr):
    """Find where the spectrum collapses: detect steep rolloff / hard cutoff."""
    f, P = welch_avg(x, sr, 8192)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    # median over time is already done by welch on whole signal
    # look at broad band levels in 1 kHz bins
    edges = np.arange(1000, 22051, 1000)
    bands = []
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        bands.append(float(np.mean(Pdb[m])) if m.any() else -np.inf)
    # cutoff: last band whose level is within 40 dB of the 1-2k band
    ref = bands[0]
    cutoff = None
    for i, b in enumerate(bands):
        if b > ref - 40:
            cutoff = edges[i + 1]
    return bands, cutoff


def hum_analysis(x, sr):
    """Detect mains hum and harmonics (50/60 Hz family)."""
    f, P = welch_avg(x, sr, 32768)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    out = {}
    for base in (50, 60):
        peaks = []
        for h in range(1, 9):
            tgt = base * h
            m = (f > tgt - 2) & (f < tgt + 2)
            if not m.any():
                continue
            peak = float(np.max(Pdb[m]))
            # local noise floor around it
            nm = ((f > tgt - 25) & (f < tgt - 6)) | ((f > tgt + 6) & (f < tgt + 25))
            floor = float(np.median(Pdb[nm])) if nm.any() else peak
            peaks.append((tgt, round(peak - floor, 2)))
        out[f'hum_{base}'] = peaks
    return out


def stereo_report(x):
    if x.shape[1] == 1:
        return {'channels': 1}
    L, R = x[:, 0], x[:, 1]
    mid = (L + R) / 2
    side = (L - R) / 2
    corr = float(np.corrcoef(L, R)[0, 1]) if np.std(L) > 0 and np.std(R) > 0 else 1.0
    side_db = 20 * np.log10(np.maximum(np.sqrt(np.mean(side ** 2)), 1e-12))
    mid_db = 20 * np.log10(np.maximum(np.sqrt(np.mean(mid ** 2)), 1e-12))
    # is it dual mono?
    diff = L - R
    ident = float(np.max(np.abs(diff)))
    return {
        'channels': 2,
        'l_r_correlation': round(corr, 6),
        'mid_rms_db': round(mid_db, 2),
        'side_rms_db': round(side_db, 2),
        'side_minus_mid_db': round(side_db - mid_db, 2),
        'max_abs_L_minus_R': float(ident),
        'dual_mono': bool(ident < 1e-6),
    }


def noise_floor(x, sr):
    """Estimate noise floor from quietest 5% of 100ms frames."""
    mono = x if x.ndim == 1 else (x.mean(axis=1) if x.shape[1] > 1 else x[:, 0])
    n = int(0.1 * sr)
    nf = len(mono) // n
    if nf < 5:
        return None
    fr = mono[:nf * n].reshape(nf, n)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    quiet = np.sort(db)[:max(1, int(0.05 * len(db)))]
    loud = np.sort(db)[-max(1, int(0.05 * len(db))):]

    # spectral shape of the noise floor (concatenate quietest frames)
    idx = np.argsort(db)[:max(1, int(0.05 * len(db)))]
    quiet_sig = fr[idx].reshape(-1)
    f, P = welch_avg(quiet_sig, sr, 4096)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    def band(a, b):
        m = (f >= a) & (f < b)
        return float(np.mean(Pdb[m])) if m.any() else float('nan')
    return {
        'floor_rms_db': round(float(np.mean(quiet)), 2),
        'loud_rms_db': round(float(np.mean(loud)), 2),
        'dynamic_range_db': round(float(np.mean(loud) - np.mean(quiet)), 2),
        'noise_spectrum_dbc_bands': {
            '20-100': round(band(20, 100), 1),
            '100-500': round(band(100, 500), 1),
            '500-2k': round(band(500, 2000), 1),
            '2k-6k': round(band(2000, 6000), 1),
            '6k-12k': round(band(6000, 12000), 1),
            '12k-20k': round(band(12000, 20000), 1),
        },
        'noise_tilt_low_minus_high_db': round(band(20, 500) - band(2000, 20000), 1),
    }


def clipping(x):
    """Detect flat-topped samples / digital clipping."""
    mono = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    peak = float(np.max(np.abs(x)))
    # count runs of >=3 identical consecutive samples at near peak
    d = np.diff(mono)
    flat = (d == 0).astype(np.int8)
    # runs
    runs = []
    c = 0
    for v in flat:
        if v:
            c += 1
        else:
            if c:
                runs.append(c)
            c = 0
    runs = np.array(runs) if runs else np.array([0])
    return {
        'peak_dbfs': round(float(dbfs(peak)), 3),
        'sample_peak': peak,
        'clipped_samples_est': int(np.sum(runs[runs >= 3], dtype=np.int64)),
        'pct_flat_run_samples': round(100.0 * float(np.sum(runs[runs >= 3], dtype=np.int64)) / float(len(mono)), 4),
    }


def band_energy(x, sr):
    mono = x if x.ndim == 1 else (x.mean(axis=1) if x.shape[1] > 1 else x[:, 0])
    f, P = welch_avg(mono, sr, 16384)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    bands = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
             (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 16000), (16000, 20000)]
    out = {}
    tot = float(np.sum(10 ** (Pdb / 10)))
    for a, b in bands:
        m = (f >= a) & (f < b)
        if m.any():
            out[f'{a}-{b}'] = round(float(np.mean(Pdb[m])), 2)
    return out


def spectral_flatness(x, sr):
    mono = x if x.ndim == 1 else (x.mean(axis=1) if x.shape[1] > 1 else x[:, 0])
    f, t, Z = signal.stft(mono, sr, nperseg=4096)
    mag = np.abs(Z)
    eps = 1e-12
    gm = np.exp(np.mean(np.log(mag + eps), axis=0))
    am = np.mean(mag, axis=0)
    return float(np.mean(gm / np.maximum(am, eps)))


def main():
    paths = sorted(glob.glob('/home/user/OldAlbum/work/src/*.wav'))
    allrep = {}
    for p in paths:
        name = os.path.basename(p)
        x, sr = load(p)
        mono = x if x.ndim == 1 else (x.mean(axis=1) if x.shape[1] > 1 else x[:, 0])
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(mono)
        bands, cutoff = highfreq_ceiling(mono, sr)
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(mono ** 2)))
        rep = {
            'file': name,
            'sr': sr,
            'duration_s': round(len(x) / sr, 2),
            'channels': x.shape[1],
            'integrated_lufs': round(float(lufs), 2),
            'peak_dbfs': round(float(dbfs(peak)), 3),
            'rms_dbfs': round(float(20 * np.log10(max(rms, 1e-12))), 2),
            'crest_db': round(float(dbfs(peak) - 20 * np.log10(max(rms, 1e-12))), 2),
            'dc_offset': [round(float(np.mean(x[:, c])), 6) for c in range(x.shape[1])],
            'spectral_flatness': round(spectral_flatness(mono, sr), 4),
            'hf_band_levels_1k_steps_db': [round(b, 1) for b in bands],
            'hf_cutoff_est_hz': (int(cutoff) if cutoff is not None else None),
            'stereo': stereo_report(x),
            'clipping': clipping(x),
            'noise': noise_floor(x, sr),
            'band_energy_db': band_energy(mono, sr),
            'hum': hum_analysis(mono, sr),
        }
        allrep[name] = rep
        print(json.dumps(rep, indent=2, ensure_ascii=False, default=float), flush=True)
        print('-' * 100, flush=True)
    with open('/home/user/OldAlbum/work/analysis.json', 'w') as f:
        json.dump(allrep, f, indent=2, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
