#!/usr/bin/env python3
"""Deep per-song diagnostics with plots: spectra, spectrograms, noise, clicks, hum."""
import os, sys, glob, json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, ndimage

SR = 44100
OUT = '/home/user/OldAlbum/work/plots'


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-20))


def analyze(path, tag):
    x, sr = sf.read(path, always_2d=True, dtype='float64')
    mono = x.mean(axis=1)
    res = {}

    # ---- 1. spectrogram (log) + averaged spectrum
    f, t, Z = signal.stft(mono, sr, nperseg=4096, noverlap=3072)
    mag = 20 * np.log10(np.maximum(np.abs(Z), 1e-12))
    vmax = np.percentile(mag, 99.5)
    fig, ax = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    im = ax[0].pcolormesh(t, f, mag, vmin=vmax - 90, vmax=vmax, shading='auto', cmap='magma')
    ax[0].set_ylim(0, 20000); ax[0].set_ylabel('Hz'); ax[0].set_title(f'{tag} — spectrogram (0-20 kHz)')
    plt.colorbar(im, ax=ax[0], pad=0.01)

    # zoomed low band
    im2 = ax[1].pcolormesh(t, f, mag, vmin=vmax - 90, vmax=vmax, shading='auto', cmap='magma')
    ax[1].set_ylim(0, 2000); ax[1].set_ylabel('Hz'); ax[1].set_title('0-2 kHz (bass/hum/rumble)')
    ax[1].set_xlabel('s')

    # long-term average spectrum
    fw, P = signal.welch(mono, sr, nperseg=16384)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    ax[2].semilogx(fw[1:], Pdb[1:], lw=0.8, label='source')
    ax[2].set_xlim(20, 22050); ax[2].set_ylim(Pdb[1:].max() - 80, Pdb[1:].max() + 6)
    ax[2].set_xlabel('Hz'); ax[2].set_ylabel('dB'); ax[2].set_title('long-term average spectrum')
    ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/{tag}_spec.png', dpi=90); plt.close()

    # ---- 2. detect exact HF cutoff by looking at where spectral energy vanishes
    edges = np.arange(2000, 22000, 250)
    lev = []
    for i in range(len(edges) - 1):
        m = (fw >= edges[i]) & (fw < edges[i + 1])
        lev.append(Pdb[m].mean() if m.any() else -200)
    lev = np.array(lev)
    # cutoff = highest edge band still within 25 dB of 2-4k reference that is
    # followed by a steep drop
    ref = lev[:8].mean()
    above = edges[1:][lev > ref - 28]
    res['hf_bands_2k_to_22k'] = {int(edges[i]): round(float(lev[i]), 1) for i in range(len(lev))}
    res['hf_cutoff_hz'] = int(above.max()) if len(above) else None

    # ---- 3. subsonic / rumble
    sub = Pdb[(fw > 15) & (fw < 40)].mean()
    low = Pdb[(fw > 40) & (fw < 100)].mean()
    res['sub15_40_db'] = round(float(sub), 1)
    res['low40_100_db'] = round(float(low), 1)

    # ---- 4. precise hum (narrow bins) at 50 & 60 Hz families
    f2, P2 = signal.welch(mono, sr, nperseg=65536)
    P2db = 10 * np.log10(np.maximum(P2, 1e-20))
    hum = {}
    for base, label in ((50, 'eu'), (60, 'us')):
        tot = 0.0
        det = []
        for h in range(1, 11):
            tgt = base * h
            near = (f2 > tgt - 1.5) & (f2 < tgt + 1.5)
            if not near.any():
                continue
            pk = P2db[near].max()
            ring = ((f2 > tgt - 30) & (f2 < tgt - 5)) | ((f2 > tgt + 5) & (f2 < tgt + 30))
            fl = np.median(P2db[ring])
            det.append((tgt, round(float(pk - fl), 2)))
            tot += max(0.0, pk - fl)
        hum[label] = {'harmonics': det, 'sum_excess_db': round(float(tot), 1)}
    res['hum'] = hum

    # ---- 5. clicks / dropouts / discontinuities
    d = np.abs(np.diff(mono))
    med = np.median(d) + 1e-9
    spikes = d > 12 * med
    # group into events
    events = int(np.sum(np.diff(np.concatenate([[0], spikes.astype(np.int8), [0]])) == 1))
    res['diff_spikes_gt12x_median'] = events
    res['max_diff'] = float(d.max())
    # silence detection (dropouts)
    env = ndimage.uniform_filter1d(np.abs(mono), int(0.005 * sr))
    thr = 10 ** ((res.get('floor_db', -60)) / 20)
    quiet = env < np.percentile(env, 1) * 2
    res['pct_near_silence'] = round(100 * float(quiet.mean()), 3)

    # ---- 6. noise floor spectrum & character
    n = int(0.1 * sr)
    nf = len(mono) // n
    fr = mono[:nf * n].reshape(nf, n)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    idx = np.argsort(rms)[:max(1, int(0.04 * nf))]
    quiet_sig = fr[idx].reshape(-1)
    fq, Pq = signal.welch(quiet_sig, sr, nperseg=8192)
    Pqdb = 10 * np.log10(np.maximum(Pq, 1e-20))
    bands = {}
    for a, b in ((20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000),
                 (2000, 4000), (4000, 8000), (8000, 16000)):
        m = (fq >= a) & (fq < b)
        bands[f'{a}-{b}'] = round(float(Pqdb[m].mean()), 1)
    res['noise_floor_bands'] = bands
    res['noise_floor_rms_db'] = round(float(db(np.sqrt((quiet_sig ** 2).mean()))), 1)

    # plot noise floor vs signal
    fig, ax = plt.subplots(figsize=(14, 5))
    Pf, _ = signal.welch(mono, sr, nperseg=16384)
    Pfdb = 10 * np.log10(np.maximum(Pf, 1e-20))
    ax.semilogx(fw, Pfdb, label='signal', lw=0.8)
    ax.semilogx(fq, Pqdb, label='noise floor (quiet frames)', lw=0.9)
    ax.set_xlim(20, 22050); ax.set_xlabel('Hz'); ax.set_ylabel('dB'); ax.grid(alpha=0.3)
    ax.legend(); ax.set_title(f'{tag} — signal vs noise floor')
    plt.tight_layout(); plt.savefig(f'{OUT}/{tag}_noise.png', dpi=90); plt.close()

    # ---- 7. clipping detail
    peak = np.abs(x).max()
    thr = 0.999 * peak
    hits = (np.abs(mono) >= thr * 0.995).astype(np.int8)
    dd = np.diff(np.concatenate([[0], hits, [0]]))
    starts = np.flatnonzero(dd == 1)
    ends = np.flatnonzero(dd == -1)
    runs = (ends - starts) if len(starts) else np.array([0], dtype=np.int64)
    res['clip'] = {
        'peak': round(float(peak), 4),
        'flat_runs_ge3': int(np.sum(runs >= 3)),
        'flat_runs_ge10': int(np.sum(runs >= 10)),
        'max_run': int(runs.max()),
        'pct_samples_in_runs': round(100 * float(np.sum(runs[runs >= 3]) / len(mono)), 4),
    }

    # ---- 8. DC
    res['dc'] = [round(float(x[:, c].mean()), 6) for c in range(x.shape[1])]
    res['dc_db'] = round(float(db(x.mean(axis=0).mean())), 2)

    # ---- 9. inter-channel: is it真 mono + DC?
    res['channel_diff_rms_db'] = round(float(db(np.sqrt(((x[:, 0] - x[:, 1]) ** 2).mean()))), 1)
    res['channel_corr'] = round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 6)

    return res, (fw, Pdb, fq, Pqdb)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for p in sorted(glob.glob('/home/user/OldAlbum/work/src/*.wav')):
        tag = os.path.basename(p).replace('.wav', '')
        r, _ = analyze(p, tag)
        out[tag] = r
        print(json.dumps({tag: r}, indent=1))
    json.dump(out, open('/home/user/OldAlbum/work/diagnose.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
