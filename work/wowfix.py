#!/usr/bin/env python3
"""
Transport-drift (wow / flutter) measurement and gentle correction.

Why this file exists
--------------------
A transfer from tape or a disc has a *transport* defect: the nominal speed varies
by a fraction of a percent at 0.3-3 Hz. On sustained material (organ, horns,
piano) that is audible as the whole chord "swimming" in and out of tune, and it
smears transients because every frequency is modulated at once. The measured
sizes in this project are small but real: +8.4 / -13.9 cents on the *Put It On*
drone, i.e. ordinary tape wow - **not** a semitone-scale fault. So the correction
here is deliberately gentle and bounded, and it is verified against its own
estimator before it is allowed to touch a master.

Why not a plain peak track
--------------------------
Picking the dominant STFT peak gives 50-98 cents of apparent "drift", but that is
note changes and vibrato, not transport: a bass line is a sequence of pitches. A
magnitude peak is also only accurate to about half a bin (5.4 Hz = 38 cents at
220 Hz), which is coarser than the defect. Both errors are fatal.

How it is measured here
-----------------------
1. Pick the **most stationary** sustained line: for every spectral peak in
   100-1500 Hz, band-pass +-30 Hz, take the Hilbert instantaneous frequency (IF)
   and score it by the *interquartile range* of IF over the whole song. The
   winning line is the one that actually holds a pitch - an organ/horn drone or
   a held piano note.
2. Smooth the IF with a short median (kills note onsets) and then a low-pass at
   2 Hz, which is the transport band. Anything faster (vibrato is 4-7 Hz) is kept
   out of the correction by design.
3. Integrate the remaining rate error into a position curve in samples, cap it at
   `max_cents` (default 15 - a gentle fix), remove any net drift, and warp with
   cubic interpolation.

The warp is a pure *time* warp, slow and C1-smooth: that is exactly the shape of
the defect and the only shape that can be undone without smearing attacks.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

SR = 44100


def _band(x, sr, f0, half=30.0, order=4):
    lo, hi = max(f0 - half, 20.0), min(f0 + half, sr / 2 - 100)
    b, a = signal.butter(order, [lo, hi], btype='band', fs=sr)
    return signal.filtfilt(b, a, x)


def _if_curve(x, sr):
    """Instantaneous frequency + envelope of an already band-limited signal."""
    z = signal.hilbert(x)
    env = np.abs(z)
    ph = np.unwrap(np.angle(z))
    if_ = np.diff(ph) * sr / (2 * np.pi)
    if_ = np.append(if_, if_[-1])
    return if_, env


def find_stationary_partial(x, sr=SR, lo=120.0, hi=1500.0, max_cand=6):
    """Locate the most pitch-stable sustained line; return (f0, iqr_cents)."""
    x = np.asarray(x, np.float64)
    x = x[:int(min(len(x), 90 * sr))]
    if x.ndim > 1:
        x = x.mean(axis=1)
    nper = min(len(x), 1 << 17)
    f, P = signal.welch(x, sr, nperseg=nper)
    band = (f >= lo) & (f <= hi)
    Pb = P.copy()
    Pb[~band] = 0
    peaks, props = signal.find_peaks(Pb, height=Pb.max() * 0.01, distance=8)
    order = np.argsort(props['peak_heights'])[::-1][:max_cand]
    best = (None, 1e9)
    for p in peaks[order]:
        f0 = float(f[p])
        try:
            bw = _band(x, sr, f0)
        except ValueError:
            continue
        if_ , env = _if_curve(bw, sr)
        ok = env > np.percentile(env, 40)
        if ok.sum() < sr:
            continue
        c = 1200 * np.log2(if_[ok] / f0)
        iqr = float(np.percentile(c, 85) - np.percentile(c, 15))
        # prefer lines that are both strong and stable
        score = iqr / (1.0 + np.log10(Pb[p] / Pb.max() + 1e-9))
        if score < best[1]:
            best = (f0, score, iqr)
    if best[0] is None:
        return None
    return {'f0': round(best[0], 2), 'iqr_cents': round(best[2], 2)}


def _extract_drift(cents_raw, iqr_cents, dt, max_cents=12.0, smooth_hz=2.0):
    """Extract the transport (0.2-2 Hz) part of a pitch track, with a decision gate.

    A time warp may only be applied to the part of the excursion that is *slower
    than the performance*. Vibrato and note changes live above ~1.5 Hz; transport
    wow lives below. So the track is band-limited to 0.2-2 Hz and the result is
    only accepted if the slow part dominates the excursion *and* the line was
    pitch-stable to begin with (`iqr_cents`). Both conditions are measured, not
    assumed - this is the difference between a de-wow and a detuner.

    Returns a dict, or None if the evidence does not support a correction.
    """
    ok = np.isfinite(cents_raw)
    if ok.sum() < 32:
        return None
    c = np.interp(np.arange(len(cents_raw)), np.flatnonzero(ok), cents_raw[ok])
    # The raw Hilbert IF of a real transfer is *full of spikes*: whenever the
    # isolated line momentarily loses to bleed or a note onset, the phase jumps and
    # the estimate leaves the band (measured on Lonesome Feeling: 22 % of samples
    # beyond +-100 cents, extremes at -376/+261). Filtering that directly turns the
    # spikes into a slow wander - which then looks exactly like wow and gets
    # "corrected". It has to be de-spiked first, with a median (a linear filter
    # cannot do this: its output follows the outliers it is meant to reject).
    c = median_filter(c, size=max(3, int(0.05 / dt)))
    # High-pass 0.2 Hz (tuning offset) and low-pass 2 Hz (vibrato) are applied at a
    # decimated rate. At 44.1 kHz a 0.2 Hz Butterworth has its pole at 0.99998 -
    # numerically indistinguishable from instability, and `filtfilt` blows it up
    # (measured: transport figures of +41711 cents). At 400 Hz the same filter is
    # trivial and exact. The pitch track has no content above a few Hz, so
    # decimating it costs nothing.
    dec = max(1, int(round((1.0 / dt) / 400.0)))
    cd = c[::dec]
    dt_d = dt * dec
    if len(cd) > 32:
        b, a = signal.butter(2, 0.2 / (0.5 / dt_d), 'high')
        cd = signal.filtfilt(b, a, cd)
        b, a = signal.butter(2, min(2.0, 0.45 / dt_d) / (0.5 / dt_d), 'low')
        cd = signal.filtfilt(b, a, cd)
        c = np.interp(np.linspace(0, 1, len(c)), np.linspace(0, 1, len(cd)), cd)
    n = max(3, int(1.0 / (smooth_hz * dt)))
    slow = uniform_filter1d(uniform_filter1d(c, n), n)
    total = float(np.percentile(c, 99) - np.percentile(c, 1))
    slow_pp = float(np.percentile(slow, 99) - np.percentile(slow, 1))
    frac = slow_pp / max(total, 1e-9)
    if iqr_cents > 25.0 or frac < 0.5 or slow_pp < 3.0:
        return None
    amps = np.clip(slow, -max_cents, max_cents)
    pos = np.cumsum(amps / 1200.0 * np.log(2) * 1.0)
    pos = uniform_filter1d(pos, n)
    pos -= np.linspace(pos[0], pos[-1], len(pos))      # net duration unchanged
    return {'pos': pos, 'cents': amps, 'slow_pp': slow_pp, 'fast_frac': 1.0 - frac,
            'corrected_pp': total - slow_pp}


def best_window(x, sr=SR, win_s=8.0, hop_s=4.0, lo=100.0, hi=1200.0):
    """Slide a window and return the most pitch-stable stretch.

    A whole song never holds one pitch for minutes, but an 8-second window often
    does. Looking only at the whole excerpt would report "no stable partial" for
    material that plainly has one (three of these four songs do, once you look at
    the right 8 seconds), and reporting that would hide a real defect.
    """
    x = np.asarray(x, np.float64)
    m = x.mean(axis=1) if x.ndim > 1 else x
    w, hop = int(win_s * sr), int(hop_s * sr)
    best = None
    for st in range(0, max(1, len(m) - w), hop):
        s_ = find_stationary_partial(m[st:st + w], sr, lo=lo, hi=hi)
        if s_ is None:
            continue
        if best is None or s_['iqr_cents'] < best['iqr_cents']:
            best = dict(s_, start_s=round(st / sr, 1), window_s=win_s)
    return best


def measure(x, sr=SR, f0=None, lo=100.0, hi=1200.0):
    """Report whether a de-wow is supported, and by how much. No processing."""
    x = np.asarray(x, np.float64)
    m = x.mean(axis=1) if x.ndim > 1 else x
    if f0 is None:
        s_ = best_window(m, sr, lo=lo, hi=hi)
        if s_ is None:
            return {'ok': False, 'reason': 'no pitch-stable partial'}
        f0, iqr = s_['f0'], s_['iqr_cents']
        st = int(s_['start_s'] * sr)
        m = m[st:st + int(s_['window_s'] * sr)]
    else:
        iqr = 0.0
    bw = _band(m, sr, f0)
    if_, env = _if_curve(bw, sr)
    ok = env > np.percentile(env, 40)
    cents = np.full(len(m), np.nan)
    c = 1200 * np.log2(np.maximum(if_[ok], 1e-9) / f0)
    cents[ok] = np.where(np.isfinite(c), c, np.nan)
    dt = 1.0 / sr
    res = _extract_drift(cents, iqr, dt)
    if res is None:
        return {'ok': False, 'reason': 'no coherent transport drift (IQR %.0f c, f0 %.1f Hz)'
                % (iqr, f0), 'f0_hz': round(f0, 1)}
    return {'ok': True, 'f0_hz': round(f0, 1), 'iqr_cents': round(iqr, 1),
            'transport_cents_pp': round(res['slow_pp'], 1),
            'vibrato_fraction': round(res['fast_frac'], 2)}


def correct(x, sr=SR, f0=None, lo=100.0, hi=1200.0, max_cents=12.0):
    """Measure and, only if the evidence supports it, apply the de-wow."""
    m = np.asarray(x, np.float64)
    mm = m.mean(axis=1) if m.ndim > 1 else m
    if f0 is None:
        s_ = best_window(mm, sr, lo=lo, hi=hi)
        if s_ is None:
            return np.asarray(x, np.float32), {'ok': False, 'reason': 'no pitch-stable partial'}
        f0, iqr = s_['f0'], s_['iqr_cents']
    else:
        iqr = 0.0
    bw = _band(mm, sr, f0)
    if_, env = _if_curve(bw, sr)
    ok = env > np.percentile(env, 40)
    cents = np.full(len(mm), np.nan)
    c = 1200 * np.log2(np.maximum(if_[ok], 1e-9) / f0)
    cents[ok] = np.where(np.isfinite(c), c, np.nan)
    res = _extract_drift(cents, iqr, 1.0 / sr, max_cents=max_cents)
    if res is not None and len(res['pos']) != len(mm):
        res['pos'] = np.interp(np.linspace(0, 1, len(mm)),
                               np.linspace(0, 1, len(res['pos'])), res['pos'])
    if res is None:
        return np.asarray(x, np.float32), {'ok': False, 'reason': 'transport drift not separable',
                                           'f0_hz': round(f0, 1), 'iqr_cents': round(iqr, 1)}
    y = apply_drift(x, res, sr)
    return y, {'ok': True, 'f0_hz': round(f0, 1), 'iqr_cents': round(iqr, 1),
               'transport_cents_pp': round(res['slow_pp'], 1),
               'vibrato_fraction': round(res['fast_frac'], 2)}


def apply_drift(x, curve, sr=SR):
    """Warp x by a position curve: y[n] = x[n + pos[n]] (cubic interpolation)."""
    if curve is None:
        return np.asarray(x, np.float32)
    x = np.asarray(x, np.float64)
    n = x.shape[0]
    pos = curve['pos'][:n] if len(curve['pos']) >= n else np.pad(curve['pos'], (0, n - len(curve['pos'])), mode='edge')
    grid = np.arange(n) + pos
    grid = np.clip(grid, 0, n - 1.001)
    if x.ndim == 1:
        return _cubic(x, grid).astype(np.float32)
    return np.stack([_cubic(x[:, c], grid) for c in range(x.shape[1])], axis=1).astype(np.float32)


def _cubic(y, xi):
    i0 = np.floor(xi).astype(np.int64)
    i0 = np.clip(i0, 1, len(y) - 3)
    t = xi - i0
    y0, y1, y2, y3 = y[i0 - 1], y[i0], y[i0 + 1], y[i0 + 2]
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3
    b = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3
    c = -0.5 * y0 + 0.5 * y2
    return ((a * t + b) * t + c) * t + y1


def process(x, sr=SR, max_cents=15.0, smooth_hz=2.0, strength=1.0, f0=None):
    """Measure and correct in one call. Returns (y, info)."""
    info = {'ok': False}
    if f0 is None:
        s = find_stationary_partial(x, sr)
        if s is None:
            return np.asarray(x, np.float32), info
        f0 = s['f0']
        info['iqr_cents'] = s['iqr_cents']
    c = drift_curve(x, f0, sr, max_cents=max_cents, smooth_hz=smooth_hz)
    if c is None:
        return np.asarray(x, np.float32), info
    c['pos'] = c['pos'] * strength
    y = apply_drift(x, c, sr)
    info.update({'ok': True, 'f0': f0, 'pp_cents_raw': round(c['pp_cents_raw'], 2),
                 'pp_cents_corrected': round(c['pp_cents'], 2)})
    return y, info
