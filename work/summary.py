#!/usr/bin/env python3
"""Compact, reproducible before/after summary -> work/verify_summary.json + table."""
import os, sys, glob, json
import numpy as np
import soundfile as sf
sys.path.insert(0, '/home/user/OldAlbum/work')
import dsp

ROOT = '/home/user/OldAlbum'
SR = 44100


def load(p):
    x, sr = sf.read(p, always_2d=True, dtype='float64')
    return x, sr


def octave_balance(x, sr=SR):
    from scipy import signal
    f, P = signal.welch(x, sr, nperseg=16384)
    out = {}
    edges = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000,
             6300, 8000, 10000, 12500, 16000, 20000]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        out[str(a)] = round(10 * np.log10(max(float(P[m].sum()), 1e-30)), 1) if m.any() else None
    ref = np.mean([v for k, v in out.items() if 200 <= int(k) <= 800])
    return {k: round(v - ref, 2) for k, v in out.items() if v is not None}


def hf_reach(x, sr=SR, drop_db=-90.0):
    from scipy import signal
    f, P = signal.welch(x, sr, nperseg=16384)
    p = 10 * np.log10(np.maximum(P, 1e-30))
    p -= p.max()
    a = f[p > drop_db]
    return int(a.max()) if len(a) else 0


def noise_6k_16k(x, sr=SR):
    """Quiet-frame level of 6-16 kHz, normalised to -12 LUFS (comparable across masters)."""
    from scipy import signal
    g = 10 ** ((-12.0 - dsp.loudness(x)) / 20.0)
    y = x * g
    m = y.mean(axis=1)
    n = int(0.1 * sr)
    nb = len(m) // n
    fr = m[:nb * n].reshape(nb, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    idx = np.argsort(rms)[:max(1, int(0.10 * nb))]
    f, P = signal.welch(fr[idx].ravel(), sr, nperseg=8192)
    band = (f >= 6000) & (f < 16000)
    return round(10 * np.log10(max(float(P[band].sum()), 1e-30)), 1)


def measure(x):
    m = x.mean(axis=1)
    return {
        'lufs': round(float(dsp.loudness(x)), 2),
        'true_peak_dbtp': round(float(dsp.true_peak_db(x)), 2),
        'crest_db': round(float(dsp.peak_db(m) - dsp.rms_db(m)), 1),
        'rms_db': round(float(dsp.rms_db(m)), 1),
        'hf_reach_hz': hf_reach(m),
        'noise_6k_16k_db': noise_6k_16k(x),
        'stereo_corr': round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 5),
        'octave_balance_db': octave_balance(m),
    }


def main():
    rep = {}
    for src in sorted(glob.glob(f'{ROOT}/work/src/*.wav')):
        tag = os.path.basename(src)[:-4]
        out = f'{ROOT}/restored/{tag} (restored).wav'
        if not os.path.exists(out):
            continue
        xs, _ = load(src)
        xo, _ = load(out)
        rep[tag] = {'source': measure(xs), 'restored': measure(xo)}
        s, r = rep[tag]['source'], rep[tag]['restored']
        print(f"{tag:26} LUFS {s['lufs']:6.2f} -> {r['lufs']:6.2f} | "
              f"TP {s['true_peak_dbtp']:5.2f} -> {r['true_peak_dbtp']:5.2f} | "
              f"crest {s['crest_db']:5.1f} -> {r['crest_db']:5.1f} | "
              f"reach {s['hf_reach_hz']:5d} -> {r['hf_reach_hz']:5d} | "
              f"HFnoise {s['noise_6k_16k_db']:6.1f} -> {r['noise_6k_16k_db']:6.1f} | "
              f"corr {s['stereo_corr']:.4f} -> {r['stereo_corr']:.4f}")
    json.dump(rep, open(f'{ROOT}/work/verify_summary.json', 'w'), indent=2)
    print('\n-> work/verify_summary.json')


if __name__ == '__main__':
    main()
