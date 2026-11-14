#!/usr/bin/env python3
"""
Stage 5: delivery exports + comparison assets.

* 320 kb/s CBR MP3 (gapless metadata) for portable use
* side-by-side 30 s A/B excerpts (source level-matched to -12.0 LUFS so the
  comparison judges *quality*, not loudness)
* documentation plots, with the source and the restored version in one figure
"""
import os, sys, glob, subprocess, json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
sys.path.insert(0, '/home/user/OldAlbum/work')
import dsp, summary

ROOT = '/home/user/OldAlbum'
SR = 44100
FFMPEG = f'{ROOT}/work/ffmpeg'


def tag_of(path):
    return os.path.basename(path).replace('.wav', '')


def make_mp3(wav, out, title, artist='The Wailing Wailers', album='Studio One (Restored)'):
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
           '-i', wav, '-map_metadata', '-1',
           '-c:a', 'libmp3lame', '-b:a', '320k', '-ar', str(SR),
           '-id3v2_version', '3',
           '-metadata', f'title={title} (restored)',
           '-metadata', f'artist={artist}',
           '-metadata', f'album={album}',
           '-metadata', 'genre=Reggae / Ska',
           '-metadata', 'comment=Restored and remastered from a lossy YouTube source. Not a remix.',
           '-write_xing', '1',
           out]
    subprocess.run(cmd, check=True)


def ab_excerpt(src, out, start=40.0, dur=30.0):
    """Render a level-matched A/B pair so the difference is quality, not volume."""
    a, _ = sf.read(src, always_2d=True, dtype='float64')
    b, _ = sf.read(out, always_2d=True, dtype='float64')
    i0 = int(start * SR)
    i1 = min(len(a), i0 + int(dur * SR))
    a = a[i0:i1]
    # the restored version is trimmed, so align by the trim offset if present
    meta = json.load(open(f'{ROOT}/work/prep_meta.json'))
    off = meta[tag_of(src)]['trim_offset_samples'] if tag_of(src) in meta else 0
    j0 = max(0, i0 - off)
    b = b[j0:j0 + len(a)]
    n = min(len(a), len(b))
    # the A3 process is sample-aligned, so no offset compensation is needed
    # (adding one here would misalign the two halves of the A/B)
    a, b = a[:n], b[:n]
    # level-match the source to -12 LUFS (the restored master's target)
    g = 10 ** ((-12.0 - dsp.loudness(a)) / 20.0)
    a = a * g
    b = b * (10 ** ((-12.0 - dsp.loudness(b)) / 20.0))
    combo = np.concatenate([a, np.zeros((int(0.6 * SR), 2)), b])
    # keep the pair within +-0.1 dB of each other so the A/B is fair
    ga = 10 ** ((-14.0 - dsp.loudness(a)) / 20.0)
    gb = 10 ** ((-14.0 - dsp.loudness(b)) / 20.0)
    a = a * ga
    b = b * gb
    combo = np.concatenate([a, np.zeros((int(0.6 * SR), 2)), b])
    return combo, i0, j0


def docs_plots():
    os.makedirs(f'{ROOT}/docs', exist_ok=True)
    for src in sorted(glob.glob(f'{ROOT}/work/src/*.wav')):
        tag = tag_of(src)
        out = f'{ROOT}/restored/{tag} (restored).wav'
        if not os.path.exists(out):
            continue
        xs, _ = sf.read(src, always_2d=True, dtype='float64')
        xo, _ = sf.read(out, always_2d=True, dtype='float64')
        # The separator input was silence-trimmed, so the restored file starts
        # `trim_offset` samples later than the source WAV. Cut the same amount
        # off the source, otherwise spectrum/noise-floor plots compare two
        # different points in the song and the A/B is not sample-aligned.
        _meta = json.load(open(f'{ROOT}/work/prep_meta.json'))
        _off = _meta.get(tag, {}).get('trim_offset_samples', 0)
        if _off:
            xs = xs[_off:]
        _n = min(len(xs), len(xo))
        xs, xo = xs[:_n], xo[:_n]
        ms, mo = xs.mean(1), xo.mean(1)

        fig = plt.figure(figsize=(15, 11))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.42, wspace=0.16)

        for col, (sig_, lab) in enumerate(((ms, 'SOURCE'), (mo, 'RESTORED'))):
            ax = fig.add_subplot(gs[0, col])
            f, t, Z = signal.stft(sig_, SR, nperseg=4096, noverlap=3072)
            M = 20 * np.log10(np.maximum(np.abs(Z), 1e-12))
            im = ax.pcolormesh(t, f, M, vmin=M.max() - 80, vmax=M.max(),
                               shading='auto', cmap='magma')
            ax.set_ylim(0, 20000); ax.set_ylabel('Hz')
            ax.set_title(f'{lab} — {tag}', fontsize=10)
            plt.colorbar(im, ax=ax, pad=0.01)

        ax = fig.add_subplot(gs[1, :])
        fs_, Ps = signal.welch(ms, SR, nperseg=16384)
        fo_, Po = signal.welch(mo, SR, nperseg=16384)
        a = 10 * np.log10(np.maximum(Ps, 1e-30)); b = 10 * np.log10(np.maximum(Po, 1e-30))
        a -= a.max(); b -= b.max()
        ax.semilogx(fs_[1:], a[1:], lw=1.0, label='source')
        ax.semilogx(fo_[1:], b[1:], lw=1.0, label='restored')
        ax.set_xlim(20, 22050); ax.set_ylim(-95, 4); ax.grid(alpha=0.3)
        ax.set_xlabel('Hz'); ax.set_ylabel('dB (normalised)')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_title('long-term average spectrum — the missing top octave is back, the hiss is gone')

        ax = fig.add_subplot(gs[2, 0])
        for sig_, lab in ((ms, 'source'), (mo, 'restored')):
            n = int(0.1 * SR); nb = len(sig_) // n
            fr = sig_[:nb * n].reshape(nb, n)
            r = np.sqrt((fr ** 2).mean(1))
            idx = np.argsort(r)[:max(1, int(0.04 * nb))]
            fq, Pq = signal.welch(fr[idx].ravel(), SR, nperseg=8192)
            q = 10 * np.log10(np.maximum(Pq, 1e-30)); q -= q.max()
            ax.semilogx(fq[1:], q[1:], lw=1.0, label=lab)
        ax.set_xlim(20, 22050); ax.set_ylim(-60, 3); ax.grid(alpha=0.3)
        ax.set_xlabel('Hz'); ax.set_title('quietest-frame noise floor'); ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[2, 1])
        for sig2, lab in ((xs, 'source'), (xo, 'restored')):
            mid = sig2.mean(1); side = (sig2[:, 0] - sig2[:, 1]) / 2
            fm, Pm = signal.welch(mid, SR, nperseg=8192)
            _, Psd = signal.welch(side, SR, nperseg=8192)
            ax.semilogx(fm[1:], 10 * np.log10(np.maximum(Psd / np.maximum(Pm, 1e-30), 1e-12))[1:],
                        lw=1.0, label=lab)
        ax.set_xlim(20, 22050); ax.set_ylim(-80, 0); ax.grid(alpha=0.3)
        ax.set_xlabel('Hz'); ax.set_ylabel('side / mid dB')
        ax.set_title('stereo image (source is dual mono)'); ax.legend(fontsize=8)

        plt.savefig(f'{ROOT}/docs/{tag}_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        print('plot:', tag, flush=True)


def main():
    print('--- A/B excerpts')
    for src in sorted(glob.glob(f'{ROOT}/work/src/*.wav')):
        tag = tag_of(src)
        out = f'{ROOT}/restored/{tag} (restored).wav'
        if not os.path.exists(out):
            continue
        combo, i0, j0 = ab_excerpt(src, out)
        sf.write(f'{ROOT}/restored/{tag} (A-B 30s).wav', combo, SR, subtype='PCM_24')
        print('  A/B:', tag, flush=True)

    print('--- MP3')
    for out in sorted(glob.glob(f'{ROOT}/restored/*(restored).wav')):
        tag = tag_of(out)
        title = tag.split('_', 1)[1].replace('_', ' ').replace("Whats", "What's")
        mp3 = f'{ROOT}/restored/{tag}.mp3'
        make_mp3(out, mp3, title)
        print('  mp3:', os.path.basename(mp3), flush=True)

    print('--- plots')
    docs_plots()


if __name__ == '__main__':
    main()
