#!/usr/bin/env python3
"""
Stage 4: systematic defect audit of a restored master against its source.

Every metric here exists because a specific defect of a 1960s transfer shows up
in it. The point is to *find* what is still wrong instead of guessing.
"""
import os, sys, json, glob
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, '/home/user/OldAlbum/work')
import dsp

SR = 44100
ROOT = '/home/user/OldAlbum'


def mono(x):
    x = np.asarray(x, np.float32)
    return x.mean(axis=1) if x.ndim > 1 else x


def bp(x, lo, hi, order=4):
    return signal.filtfilt(*signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], 'band'), x)


def frames(x, ms=50):
    n = int(ms * 1e-3 * SR)
    N = len(x) // n
    return np.sqrt(((x[:N * n] ** 2).reshape(N, n)).mean(axis=1) + 1e-30)


# ---------------------------------------------------------------- wow & flutter
def wow_flutter(x):
    """Measure transport drift on the most *stationary* sustained partial.

    Two earlier versions of this metric were wrong in opposite directions:

    * picking the dominant STFT peak reported 50-98 cents of "wow", but that is
      note changes and vibrato, not transport;
    * a magnitude peak is only accurate to half a bin (5.4 Hz = 38 cents at
      220 Hz), which is coarser than the defect itself.

    So the partial is now chosen by *stationarity*, not loudness: every strong
    spectral peak in 120-1500 Hz is band-passed and its Hilbert instantaneous
    frequency is scored by interquartile range. Only a line that actually holds a
    pitch can win, and the IF estimate is good to a couple of cents.

    Reading the result: `drift_cents` around 10-20 cents on sustained material is
    ordinary tape wow. If `f0` is None, the recording contains no pitch-stable
    note long enough to measure - which is itself the answer, and the reason this
    project does not apply a global de-wow (see RESTORATION.md).
    """
    import wowfix
    x = np.asarray(x, np.float64)[:int(90 * SR)]
    s_ = wowfix.find_stationary_partial(x, SR)
    if s_ is None:
        return {'ok': False, 'reason': 'no pitch-stable partial'}
    # An interquartile range above ~40 cents means the "winning" line is still
    # changing pitch - i.e. the song has no sustained note and a wow measurement
    # from it would be meaningless. Reporting a number here anyway is how a
    # measurement artefact turns into a "defect" that gets "fixed".
    if s_['iqr_cents'] > 40.0:
        return {'ok': False, 'reason': 'no sustained note (IQR %.0f cents on %0.1f Hz)'
                % (s_['iqr_cents'], s_['f0'])}
    c = wowfix.drift_curve(x, s_['f0'])
    if c is None:
        return {'ok': False, 'reason': 'partial too short'}
    return {'ok': True, 'f0_hz': round(s_['f0'], 2),
            'iqr_cents': s_['iqr_cents'],
            'transport_cents_pp': round(float(c['pp_cents']), 1),
            'raw_cents_pp': round(float(c['pp_cents_raw']), 1)}


def reverb_tail(x):
    """Estimate the amount of room/tape reverb from onset decay slopes.

    A dry studio recording decays by >40 dB within ~120 ms of a hit; a wet one
    (or a tape/acetate chain with echo) decays much more slowly. We measure the
    median -60 dB decay time of the HF envelope after strong onsets.
    """
    env = uniform_filter1d(np.abs(bp(x, 1500, 8000)), int(0.002 * SR))
    thr = np.percentile(env, 97)
    onsets = []
    i = 0
    while i < len(env) - 1:
        if env[i] > thr:
            onsets.append(i)
            i += int(0.15 * SR)
        else:
            i += 1
    rt = []
    for o in onsets:
        seg = env[o:o + int(0.6 * SR)]
        if len(seg) < 1000 or seg[0] <= 0:
            continue
        db = 20 * np.log10(seg / seg[:20].max() + 1e-12)
        below = np.flatnonzero(db < -25)
        if len(below) and below[0] > 5:
            rt.append(below[0] / SR * 1000)
    if not rt:
        return {'ok': False}
    return {'ok': True, 'median_decay_to_-25dB_ms': round(float(np.median(rt)), 1),
            'p90_ms': round(float(np.percentile(rt, 90)), 1), 'n': len(rt)}


# ------------------------------------------------------------- micro dynamics
def micro_dynamics(x):
    """Short-term crest variation: a squashed master has a constant crest."""
    fr = frames(x, 100)
    pk = np.array([np.abs(x[i * int(0.1 * SR):(i + 1) * int(0.1 * SR)]).max()
                   for i in range(len(fr))])
    crest = 20 * np.log10(pk / (fr + 1e-12) + 1e-9)
    return {'crest_mean_db': round(float(np.mean(crest)), 2),
            'crest_std_db': round(float(np.std(crest)), 2),
            'lra_db': round(float(np.percentile(fr, 95) - np.percentile(fr, 10)), 2)}


# -------------------------------------------------------------------- stereo
def stereo(x):
    """Stereo measurements, with the right interpretation.

    `hf_side_corr` (waveform correlation of the 3-20 kHz side with the mid) is
    *expected* to be near zero on a genuinely wide image - uncorrelated channels
    are what width means. The number that says whether the width is real is
    `hf_env_corr`: the correlation of the two *envelopes*. If the top end widens
    because the music is there, the envelopes track each other; if it widens
    because noise was added, they do not. The old report listed only the waveform
    figure, which reads as "the width is fake" even when it is not.
    """
    if x.ndim == 1:
        return {'ok': False}
    m = x.mean(1); s = (x[:, 0] - x[:, 1]) / 2
    b, a = signal.butter(4, 3000 / (SR / 2), 'high')
    sh, mh = signal.filtfilt(b, a, s), signal.filtfilt(b, a, m)
    out = {
        'corr': round(float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]), 4),
        'side_minus_mid_db': round(float(dsp.rms_db(s) - dsp.rms_db(m)), 2),
        'hf_side_corr': round(float(np.corrcoef(sh, mh)[0, 1]), 3),
    }
    if abs(dsp.rms_db(s) - dsp.rms_db(m)) > -35.0:      # only if there *is* width
        import scipy.signal as _sg
        env = lambda v: np.abs(_sg.hilbert(v[:int(min(len(v), 60 * SR))]))
        out['hf_env_corr'] = round(float(np.corrcoef(env(sh), env(mh))[0, 1]), 3)
    return out


# ------------------------------------------------------------- inter-sample
def intersample(x):
    tp = dsp.true_peak_db(x)
    pk = dsp.peak_db(mono(x))
    return {'true_peak_dbtp': round(float(tp), 2), 'sample_peak_db': round(float(pk), 2),
            'intersample_over_db': round(float(tp - pk), 2)}


# ---------------------------------------------------------------- noise floor
def noise_floor(x):
    fr = frames(mono(x), 100)
    q = np.argsort(fr)[:max(1, int(0.05 * len(fr)))]
    n = int(0.1 * SR)
    seg = np.concatenate([mono(x)[i * n:(i + 1) * n] for i in q])
    f, P = signal.welch(seg, SR, nperseg=8192)
    out = {}
    for lo, hi in ((40, 200), (200, 1000), (1000, 4000), (4000, 8000), (8000, 16000)):
        m = (f >= lo) & (f < hi)
        out[f'{lo}-{hi}'] = round(float(10 * np.log10(P[m].sum() + 1e-30)), 1)
    return out


def audit(tag):
    meta = json.load(open(f'{ROOT}/work/prep_meta.json'))
    off = meta[tag].get('trim_offset_samples', 0)
    s, _ = sf.read(f'{ROOT}/work/src/{tag}.wav', always_2d=True, dtype='float32')
    r, _ = sf.read(f'{ROOT}/restored/{tag} (restored).wav', always_2d=True, dtype='float32')
    s = s[off:]
    n = min(len(s), len(r))
    s, r = s[:n], r[:n]
    out = {'tag': tag,
           'source': {'wow': wow_flutter(mono(s)), 'reverb': reverb_tail(mono(s)),
                      'dyn': micro_dynamics(mono(s)), 'stereo': stereo(s),
                      'isp': intersample(s), 'noise': noise_floor(s)},
           'restored': {'wow': wow_flutter(mono(r)), 'reverb': reverb_tail(mono(r)),
                        'dyn': micro_dynamics(mono(r)), 'stereo': stereo(r),
                        'isp': intersample(r), 'noise': noise_floor(r)}}
    return out


if __name__ == '__main__':
    tags = sys.argv[1:] or ['1_I_Need_You', '2_Lonesome_Feeling', '3_Put_It_On',
                            '4_Whats_New_Pussycat']
    rep = {}
    for t in tags:
        rep[t] = audit(t)
        S, R = rep[t]['source'], rep[t]['restored']
        print(f'\n=== {t}')
        for k in ('wow', 'reverb', 'dyn', 'isp'):
            print(f'  {k:8} src {S[k]}')
            print(f'  {"":8} out {R[k]}')
        print(f'  stereo   src {S["stereo"]}  out {R["stereo"]}')
        print(f'  noise    src {S["noise"]}')
        print(f'  {"":8} out {R["noise"]}')
    json.dump(rep, open(f'{ROOT}/work/audit.json', 'w'), indent=1)
    print('\n-> work/audit.json')
