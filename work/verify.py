#!/usr/bin/env python3
"""Stage 4: objective before/after verification + comparison plots."""
import os, glob, json, sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
sys.path.insert(0, '/home/user/OldAlbum/work')
import dsp

SR = 44100
ROOT = '/home/user/OldAlbum'
PLOT = f'{ROOT}/work/plots'


def load(p, mono=False):
    x, sr = sf.read(p, always_2d=True, dtype='float64')
    return (x.mean(axis=1) if mono else x), sr


def octave_balance(x, sr=SR):
    f, P = signal.welch(x, sr, nperseg=16384)
    out = {}
    edges = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000,
             6300, 8000, 10000, 12500, 16000, 20000]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        out[a] = 10 * np.log10(max(float(P[m].sum()), 1e-30)) if m.any() else np.nan
    ref = np.mean([v for k, v in out.items() if 200 <= k <= 800])
    return {k: round(float(v - ref), 2) for k, v in out.items()}


def noise_floor(x, sr=SR):
    m = x if x.ndim == 1 else x.mean(axis=1)
    n = int(0.1 * sr)
    nb = len(m) // n
    fr = m[:nb * n].reshape(nb, n)
    r = np.sqrt((fr ** 2).mean(axis=1))
    idx = np.argsort(r)[:max(1, int(0.04 * nb))]
    q = fr[idx].ravel()
    f, P = signal.welch(q, sr, nperseg=8192)
    pdb = 10 * np.log10(np.maximum(P, 1e-30))
    bands = {}
    for a, b in ((20, 100), (100, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 20000)):
        mm = (f >= a) & (f < b)
        bands[f'{a}-{b}'] = round(float(10 * np.log10(max(float(P[mm].sum()), 1e-30))), 1)
    return round(float(20 * np.log10(max(float(np.sqrt((q ** 2).mean())), 1e-12))), 1), bands


def hf_reach(x, sr=SR, ref_db=-90.0):
    f, P = signal.welch(x if x.ndim == 1 else x.mean(axis=1), sr, nperseg=16384)
    pdb = 10 * np.log10(np.maximum(P, 1e-30))
    pdb -= pdb.max()
    above = f[pdb > ref_db]
    return int(above.max()) if len(above) else 0


def band_snr(x, sr=SR):
    """
    Crude but useful: ratio between the programme's loudest and quietest
    spectral regions. A cleaner recording has more contrast (less constant hiss).
    """
    f, P = signal.welch(x if x.ndim == 1 else x.mean(axis=1), sr, nperseg=8192)
    pdb = 10 * np.log10(np.maximum(P, 1e-30))
    return round(float(pdb.max() - np.median(pdb[f > 1000])), 1)


def main():
    report = {}
    for src in sorted(glob.glob(f'{ROOT}/work/src/*.wav')):
        tag = os.path.basename(src)[:-4]
        out = f'{ROOT}/restored/{tag} (restored).wav'
        if not os.path.exists(out):
            print(f'!! missing {out}')
            continue
        xs, _ = load(src)
        xo, _ = load(out)
        ms = xs.mean(axis=1); mo = xo.mean(axis=1)

        r = {}
        r['source'] = {
            'lufs': round(float(dsp.loudness(ms)), 2),
            'true_peak': round(float(dsp.true_peak_db(xs)), 2),
            'crest_db': round(float(dsp.peak_db(ms) - dsp.rms_db(ms)), 2),
            'octave_db': octave_balance(ms),
            'hf_reach_hz': hf_reach(ms),
            'noise': noise_floor(xs),
            'band_snr_db': band_snr(ms),
            'stereo_corr': round(float(np.corrcoef(xs[:, 0], xs[:, 1])[0, 1]), 5),
            'duration_s': round(len(ms) / SR, 2),
        }
        r['restored'] = {
            'lufs': round(float(dsp.loudness(mo)), 2),
            'true_peak': round(float(dsp.true_peak_db(xo)), 2),
            'crest_db': round(float(dsp.peak_db(mo) - dsp.rms_db(mo)), 2),
            'octave_db': octave_balance(mo),
            'hf_reach_hz': hf_reach(mo),
            'noise': noise_floor(xo),
            'band_snr_db': band_snr(mo),
            'stereo_corr': round(float(np.corrcoef(xo[:, 0], xo[:, 1])[0, 1]), 5),
            'duration_s': round(len(mo) / SR, 2),
        }
        report[tag] = r

        # ---- comparison plots
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1], hspace=0.35, wspace=0.18)

        # spectrograms
        for col, (sig_, lab) in enumerate(((ms, 'SOURCE'), (mo, 'RESTORED'))):
            ax = fig.add_subplot(gs[0, col])
            f, t, Z = signal.stft(sig_, SR, nperseg=4096, noverlap=3072)
            M = 20 * np.log10(np.maximum(np.abs(Z), 1e-12))
            im = ax.pcolormesh(t, f, M, vmin=M.max() - 85, vmax=M.max(),
                               shading='auto', cmap='magma')
            ax.set_ylim(0, 20000); ax.set_title(f'{tag} — {lab}')
            ax.set_ylabel('Hz')
            plt.colorbar(im, ax=ax, pad=0.01)

        # long term spectra
        ax = fig.add_subplot(gs[1, :])
        fs_, Ps = signal.welch(ms, SR, nperseg=16384)
        fo_, Po = signal.welch(mo, SR, nperseg=16384)
        Psdb = 10 * np.log10(np.maximum(Ps, 1e-30)); Podb = 10 * np.log10(np.maximum(Po, 1e-30))
        Psdb -= Psdb.max(); Podb -= Podb.max()
        ax.semilogx(fs_[1:], Psdb[1:], label='source', lw=1.0, alpha=0.8)
        ax.semilogx(fo_[1:], Podb[1:], label='restored', lw=1.0, alpha=0.9)
        ax.set_xlim(20, 22050); ax.set_ylim(-95, 3); ax.grid(alpha=0.3)
        ax.set_xlabel('Hz'); ax.set_ylabel('dB (norm.)')
        ax.legend(); ax.set_title('long-term average spectrum: source vs restored')

        # noise floor comparison
        ax = fig.add_subplot(gs[2, 0])
        for sig_, lab, style in ((ms, 'source', '-'), (mo, 'restored', '-')):
            n = int(0.1 * SR); nb = len(sig_) // n
            fr = sig_[:nb * n].reshape(nb, n)
            rr = np.sqrt((fr ** 2).mean(axis=1))
            idx = np.argsort(rr)[:max(1, int(0.04 * nb))]
            fq, Pq = signal.welch(fr[idx].ravel(), SR, nperseg=8192)
            qdb = 10 * np.log10(np.maximum(Pq, 1e-30)); qdb -= qdb.max()
            ax.semilogx(fq[1:], qdb[1:], style, label=lab, lw=1.0)
        ax.set_xlim(20, 22050); ax.set_ylim(-60, 3); ax.grid(alpha=0.3)
        ax.set_xlabel('Hz'); ax.set_title('quietest-frame noise floor (norm.)'); ax.legend()

        # mono/stereo
        ax = fig.add_subplot(gs[2, 1])
        def sm(x):
            mid = x.mean(axis=1); side = (x[:, 0] - x[:, 1]) / 2
            fm, Pm = signal.welch(mid, SR, nperseg=8192)
            fsd, Psd = signal.welch(side, SR, nperseg=8192)
            return fm, 10 * np.log10(np.maximum(Psd / np.maximum(Pm, 1e-30), 1e-12))
        fm, ratio_s = sm(xs); fo, ratio_o = sm(xo)
        ax.semilogx(fm[1:], ratio_s[1:], label='source', lw=1.0, alpha=0.7)
        ax.semilogx(fo[1:], ratio_o[1:], label='restored', lw=1.0)
        ax.set_xlim(20, 22050); ax.set_ylim(-80, 0); ax.grid(alpha=0.3)
        ax.set_xlabel('Hz'); ax.set_ylabel('side/mid dB')
        ax.set_title('stereo image'); ax.legend()

        plt.savefig(f'{PLOT}/{tag}_compare.png', dpi=88, bbox_inches='tight')
        plt.close()
        print(f'{tag}: source {r["source"]["lufs"]} LUFS / reach {r["source"]["hf_reach_hz"]} Hz '
              f'/ floor {r["source"]["noise"][0]} dB  ->  restored {r["restored"]["lufs"]} LUFS / '
              f'reach {r["restored"]["hf_reach_hz"]} Hz / floor {r["restored"]["noise"][0]} dB', flush=True)

    json.dump(report, open(f'{ROOT}/work/verify.json', 'w'), indent=2)
    print('\nwritten work/verify.json')


if __name__ == '__main__':
    main()
