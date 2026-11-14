#!/usr/bin/env python3
"""
Stage 3 DSP library: declick, denoise, dehum, decrackle, dynamic-EQ, harmonic
excitation, transient shaping, bus compression, limiter, true-peak safety.
All zero-phase / linear-phase so the four stem pairs stay sample-aligned.
"""
import numpy as np
from scipy import signal, ndimage


SR = 44100


# --------------------------------------------------------------------------- utils
def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-20))


def rms(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=axis))


def rms_db(x):
    return float(20 * np.log10(max(float(rms(x)), 1e-20)))


def peak_db(x):
    return float(20 * np.log10(max(float(np.max(np.abs(x))), 1e-20)))


def to_stereo(x):
    return x if x.ndim == 2 else np.stack([x, x], axis=-1)


def mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


# --------------------------------------------------------------------------- STFT helpers
def stft(x, n=4096, hop=1024):
    if x.ndim == 1:
        f, t, Z = signal.stft(x, SR, nperseg=n, noverlap=n - hop, window='hann',
                              padded=True, boundary='zeros')
        return f, t, Z[..., None]
    out = []
    for c in range(x.shape[1]):
        f, t, Z = signal.stft(x[:, c], SR, nperseg=n, noverlap=n - hop, window='hann',
                              padded=True, boundary='zeros')
        out.append(Z)
    return f, t, np.stack(out, axis=-1)


def istft(Z, n=4096, hop=1024, length=None):
    if Z.ndim == 2:
        Z = Z[..., None]
    chans = []
    for c in range(Z.shape[-1]):
        _, y = signal.istft(Z[..., c], SR, nperseg=n, noverlap=n - hop,
                            window='hann', boundary=True)
        chans.append(y[:length] if length else y)
    y = np.stack(chans, axis=-1)
    return y if y.shape[1] > 1 else y[:, 0]


# --------------------------------------------------------------------------- dehum
def remove_hum(x, base=50.0, harmonics=10, q=40.0, sr=SR, min_excess_db=6.0,
               require_comb=True):
    """
    Surgical hum removal.

    Real mains hum is a *comb*: several consecutive harmonics all stand above
    the local floor by a similar amount. A musical bass note is a single peak.
    We therefore only notch a harmonic if the fundamental or at least three
    consecutive neighbours show excess, which keeps bass lines untouched.
    """
    y = np.asarray(x, dtype=np.float64)
    xm = mono(y)
    f, P = signal.welch(xm, sr, nperseg=65536)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))

    excess = {}
    for h in range(1, harmonics + 1):
        f0 = base * h
        if f0 > min(16000.0, sr / 2 - 200):
            break
        near = (f > f0 - 1.2) & (f < f0 + 1.2)
        if not near.any():
            excess[h] = -99.0
            continue
        pk = float(Pdb[near].max())
        ring = ((f > f0 - 40) & (f < f0 - 3)) | ((f > f0 + 3) & (f < f0 + 40))
        if not ring.any():
            excess[h] = -99.0
            continue
        excess[h] = pk - float(np.median(Pdb[ring]))

    hs = sorted(excess)
    vals = np.array([excess[h] for h in hs])

    def comb_ok(i):
        # look at the three harmonics centred on i
        lo = max(0, i - 1)
        hi = min(len(hs), i + 2)
        window = vals[lo:hi]
        strong = np.sum(window >= min_excess_db * 0.7)
        return strong >= min(3, len(window)) or (i == 0 and vals[0] >= min_excess_db)

    removed = []
    for i, h in enumerate(hs):
        if excess[h] < min_excess_db:
            continue
        if require_comb and not comb_ok(i):
            continue
        f0 = base * h
        b, a = signal.iirnotch(f0, Q=q, fs=sr)
        y = signal.filtfilt(b, a, y, axis=0)
        removed.append({'hz': round(float(f0), 2), 'excess_db': round(float(excess[h]), 2)})
    return y.astype(np.float32), removed


# --------------------------------------------------------------------------- rumble / DC
def block_dc_subsonic(x, sr=SR, hp=28.0, lp=None):
    y = np.asarray(x, dtype=np.float64)
    if hp:
        b, a = signal.butter(4, hp / (sr / 2), btype='high')
        y = signal.filtfilt(b, a, y, axis=0)
    if lp:
        b, a = signal.butter(4, lp / (sr / 2), btype='low')
        y = signal.filtfilt(b, a, y, axis=0)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- declick
def declick(x, sr=SR, k=8.0, max_len_ms=3.0, passes=2):
    """
    Impulse/click removal: detect samples whose second difference is far above
    the local MAD, then repair with a cubic Hermite interpolation between the
    last good samples on either side. Zero-phase and sample-aligned.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    total = 0
    for _ in range(passes):
        xm = mono(y) if y.ndim == 2 else y
        d2 = np.diff(xm, n=2)
        mad = ndimage.median_filter(np.abs(d2), size=2048) + 1e-7
        # robust local sigma from MAD
        sigma = 1.4826 * mad
        hits = np.abs(d2) > k * sigma
        dd = np.diff(np.concatenate([[0], hits.astype(np.int8), [0]]))
        starts = np.flatnonzero(dd == 1)
        ends = np.flatnonzero(dd == -1)
        maxlen = int(max_len_ms * 1e-3 * sr)
        if len(starts) == 0:
            break
        for s, e in zip(starts, ends):
            if e - s > maxlen:
                continue
            total += e - s
            a = max(1, s - 3)
            b = min(len(xm) - 2, e + 3)
            n = b - a
            if n < 2:
                continue
            t = np.linspace(0, 1, b - a)
            if y.ndim == 2:
                for c in range(y.shape[1]):
                    y[a:b, c] = _hermite(y[a, c], y[b, c], t)
            else:
                y[a:b] = _hermite(y[a], y[b], t)
    return y.astype(np.float32), total


def _hermite(y0, y1, t):
    """Cubic shape-preserving interpolation between two boundary values."""
    h = y1 - y0
    return y0 + h * (3 * t ** 2 - 2 * t ** 3)


# --------------------------------------------------------------------------- crackle / surface noise
def decrackle(x, sr=SR, thresh_db=-42.0, mix=0.9):
    """
    Suppress sparse impulsive surface noise ('crackle') in the spectral domain:
    bins that stick out far above a time-smoothed local median are replaced by
    the median (LF) or the interpolated neighbours (HF).
    """
    n, hop = 2048, 512
    y = np.asarray(x, dtype=np.float64)
    out_c = []
    for c in range(y.shape[1] if y.ndim == 2 else 1):
        sig = y[:, c] if y.ndim == 2 else y
        f, t, Z = signal.stft(sig, sr, nperseg=n, noverlap=n - hop, window='hann')
        mag = np.abs(Z)
        phase = np.angle(Z)
        med = ndimage.median_filter(mag, size=(1, 9))
        ratio = mag / (med + 1e-9)
        # transient bins that spike hard in a single frame -> replace
        mask = ratio > (10 ** (thresh_db / 20) / 1.0) * 0.0 + 6.0
        mag2 = np.where(mask, med, mag)
        Z2 = mag2 * np.exp(1j * phase)
        _, res = signal.istft(Z2, sr, nperseg=n, noverlap=n - hop, window='hann',
                              boundary=True) if False else (None, None)
        out_c.append((Z2, len(sig)))
    # do the istft once per channel
    chans = []
    for c, (Z2, L) in enumerate(out_c):
        _, r = signal.istft(Z2, sr, nperseg=n, noverlap=n - hop, window='hann', boundary=True)
        r = r[:L]
        if mix:
            base = y[:, c] if y.ndim == 2 else y
            r = mix * r + (1 - mix) * base
        chans.append(r)
    res = np.stack(chans, axis=-1)
    return (res if res.shape[1] > 1 else res[:, 0]).astype(np.float32)


# --------------------------------------------------------------------------- denoise
def _min_stat_noise(mag, q=0.12, smooth_hz=5, smooth_t=7, block=16):
    """
    Per-bin noise estimate via minimum statistics.

    Cheap and memory-light: collapse the time axis into ~block-frame medians
    (a few hundred columns), take a low percentile across those, then smooth
    across frequency. Tracks the per-bin noise floor of tape hiss and AAC
    noise without the cost of a full percentile filter.
    """
    from scipy import ndimage
    nF, nT = mag.shape
    nb = max(1, nT // block)
    if nb >= 8:
        blocks = np.median(mag[:, :nb * block].reshape(nF, nb, block), axis=2)
        est = np.percentile(blocks, q * 100, axis=1, keepdims=True)
        est = np.repeat(est, nT, axis=1)
    else:
        est = np.repeat(np.percentile(mag, q * 100, axis=1, keepdims=True), nT, axis=1)
    est = ndimage.uniform_filter1d(est, size=smooth_hz, axis=0)
    est = ndimage.uniform_filter1d(est, size=max(3, smooth_t), axis=1)
    return np.maximum(est, 1e-12)


def denoise(x, sr=SR, n=2048, hop=512, reduction_db=10.0, over_sub=1.8,
            floor_db=-20.0, q=0.12, smooth_t=7, mix=1.0,
            freq_weight=((0, 0.0), (120, 0.0), (350, 0.45), (1500, 1.0))):
    """
    Spectral denoiser with minimum-statistics noise tracking.

    * noise floor per bin from the low percentile over time
    * power-domain over-subtraction with a soft floor (no musical noise)
    * gain smoothed across time+frequency, then mixed in by `mix`
    """
    from scipy import ndimage
    y = np.asarray(x, dtype=np.float64)
    ch_n = y.shape[1] if y.ndim == 2 else 1
    out = []
    for c in range(ch_n):
        sig = y[:, c] if y.ndim == 2 else y
        f, t, Z = signal.stft(sig, sr, nperseg=n, noverlap=n - hop, window='hann')
        mag = np.abs(Z)
        phase = np.angle(Z)

        noise = _min_stat_noise(mag, q=q, smooth_t=smooth_t)
        sig_pow = mag ** 2
        noise_pow = (noise * over_sub) ** 2
        g = np.maximum(sig_pow - noise_pow, 0.0) / (sig_pow + 1e-18)
        g = np.sqrt(np.maximum(g, 0.0))

        # requested amount of reduction, expressed as a gain floor.
        # Frequency-dependent: sustained musical content (bass lines, drones)
        # must never be mistaken for hiss, so below ~350 Hz the processor is
        # progressively disabled - tape/AAC hiss lives in the top end anyway.
        wf = np.array([p[0] for p in freq_weight], dtype=float)
        wv = np.array([p[1] for p in freq_weight], dtype=float)
        fw = np.interp(f, wf, wv)[:, None]
        floor = 10 ** (floor_db / 20.0)
        red_f = 10 ** ((-reduction_db * fw) / 20.0)
        red = red_f
        tgt = np.maximum(g, floor)
        gg = 1.0 - (1.0 - tgt) * ((1.0 - red) / (1.0 - floor + 1e-9))
        # blend back to unity where the weight is low
        gg = fw * gg + (1.0 - fw) * 1.0
        gg = np.clip(gg, floor, 1.0)
        gg = ndimage.uniform_filter(gg, size=(3, smooth_t))
        Z2 = (mag * gg) * np.exp(1j * phase)
        _, r = signal.istft(Z2, sr, nperseg=n, noverlap=n - hop, window='hann', boundary=True)
        r = r[:len(sig)]
        out.append(mix * r + (1 - mix) * sig if mix < 1.0 else r)
    res = np.stack(out, axis=-1)
    return (res if res.shape[1] > 1 else res[:, 0]).astype(np.float32)


# --------------------------------------------------------------------------- EQ
def _shelves_from_curve(freqs, gains_db, sr=SR):
    """Build a piecewise-smooth FIR magnitude response from (freq, dB) points."""
    f = np.asarray(freqs, float)
    g = np.asarray(gains_db, float)
    grid = np.linspace(0, sr / 2, 4097)
    lg = np.interp(np.log10(np.maximum(grid, 1.0)), np.log10(np.maximum(f, 1.0)), g)
    return grid, lg


def fir_from_curve(freqs, gains_db, sr=SR, taps=16385):
    """
    Linear-phase FIR whose magnitude follows the given dB curve.
    Odd length + windowing gives exactly linear phase (integer group delay),
    which we compensate so all stems stay aligned.
    """
    grid, lg = _shelves_from_curve(freqs, gains_db, sr)
    F = taps // 2
    mag_lin = 10 ** (np.interp(np.linspace(0, sr / 2, F + 1), grid, lg) / 20.0)
    h = np.fft.irfft(mag_lin, n=2 * F)      # length 2*F == taps
    h = np.roll(h, F)                       # zero-phase -> linear phase
    h *= np.hanning(len(h))
    h /= h.sum()
    return h


def apply_fir(x, h):
    """
    Apply a *linear-phase* FIR once. Because h is symmetric, convolution already
    introduces no phase error, and 'same' compensates the group delay exactly,
    so all stems stay sample-aligned. (Filtering twice would square the curve.)
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 2:
        out = np.stack([signal.fftconvolve(y[:, c], h, 'same') for c in range(y.shape[1])], axis=-1)
    else:
        out = signal.fftconvolve(y, h, 'same')
    return out.astype(np.float32)


def eq_curve(x, freqs, gains_db, sr=SR, taps=16385):
    h = fir_from_curve(freqs, gains_db, sr, taps)
    return apply_fir(x, h)


# --------------------------------------------------------------------------- dynamics / limiting
def compress(x, sr=SR, thresh_db=-18.0, ratio=2.0, attack_ms=25.0, release_ms=220.0,
             makeup_db=0.0, knee_db=6.0, detector='rms', lookahead_ms=5.0):
    """
    Feed-forward, stereo-linked compressor.

    detector='rms'  : block RMS in a 10 ms window (programme-loudness like)
    detector='peak' : fast peak envelope, for transient control

    Slow gain computation (cheap) + look-ahead so the attack never overshoots.
    """
    y = np.asarray(x, dtype=np.float64)
    m = mono(y)

    if detector == 'rms':
        w = max(16, int(0.010 * sr))
        # block RMS then hold/pad, then smooth with a short Hann
        nb = len(m) // w
        blocks = np.sqrt((m[:nb * w].reshape(nb, w) ** 2).mean(axis=1))
        blocks = np.repeat(blocks, w)
        if len(blocks) < len(m):
            blocks = np.concatenate([blocks, np.full(len(m) - len(blocks), blocks[-1] if len(blocks) else 0)])
        lvl_sig = blocks
    else:
        lvl_sig = np.abs(m)

    # envelope in dB with attack/release ballistics on the *slow* control rate
    step = max(1, int(sr * 0.001))          # 1 ms control rate
    nb = len(lvl_sig) // step
    ctrl = lvl_sig[:nb * step].reshape(nb, step).max(axis=1)
    ctrl_db = 20 * np.log10(np.maximum(ctrl, 1e-9))
    a_att = np.exp(-1.0 / max(1e-6, attack_ms * 1e-3 * sr / step))
    a_rel = np.exp(-1.0 / max(1e-6, release_ms * 1e-3 * sr / step))
    sm = np.empty_like(ctrl_db)
    v = ctrl_db[0]
    for i in range(len(ctrl_db)):
        c = a_att if ctrl_db[i] > v else a_rel
        v = c * v + (1 - c) * ctrl_db[i]
        sm[i] = v

    over = sm - thresh_db
    knee = max(1e-6, knee_db / 2)
    gr = np.zeros_like(over)
    below = over <= -knee
    above = over >= knee
    mid = ~(below | above)
    gr[mid] = (1 - 1 / ratio) * (over[mid] + knee) ** 2 / (4 * knee)
    gr[above] = (over[above]) * (1 - 1 / ratio)
    gr[below] = 0.0

    gain_db = -gr + makeup_db
    # upsample to sample rate (gain moves slowly, linear interp is transparent)
    idx = np.arange(len(m)) / step
    g = 10 ** (np.interp(idx, np.arange(len(gain_db)), gain_db, left=gain_db[0], right=gain_db[-1]) / 20.0)
    # look-ahead: apply the *minimum* gain of the next window so attacks are caught
    la = max(1, int(lookahead_ms * 1e-3 * sr))
    if la > 1:
        from scipy import ndimage
        g = ndimage.minimum_filter1d(g, size=la + 1)
        g = ndimage.uniform_filter1d(g, size=max(2, la // 2))
    out = y * (g[:, None] if y.ndim == 2 else g)
    return out.astype(np.float32), float(gr.max())


def limiter(x, sr=SR, ceiling_db=-1.0, lookahead_ms=4.0, release_ms=90.0):
    """
    Look-ahead brick-wall limiter with true-peak (4x oversampled) safety.
    """
    y = np.asarray(x, dtype=np.float64)
    ceil = 10 ** (ceiling_db / 20.0)
    n = len(y)
    la = max(1, int(lookahead_ms * 1e-3 * sr))

    # 4x oversampled true-peak envelope
    up = signal.resample_poly(y, 4, 1, axis=0)
    tp = np.abs(up).max(axis=1)
    tp = tp[:n * 4]
    # per-block max back to base rate (worst case in each 4-sample group)
    tp = tp[: (n) * 4].reshape(n, 4).max(axis=1) if len(tp) >= n * 4 else \
        np.interp(np.arange(n), np.linspace(0, n, len(tp)), tp)

    env = np.maximum(tp, 1e-9)
    env = ndimage.maximum_filter1d(env, size=la * 2 + 1)
    a_rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    sm = np.empty_like(env)
    v = env[0]
    for i in range(n):
        if env[i] > v:
            v = env[i]
        else:
            v = a_rel * v + (1 - a_rel) * env[i]
        sm[i] = v

    g = np.minimum(1.0, ceil / np.maximum(sm, 1e-9))
    # hold the gain for the look-ahead window, then smooth the attack
    g = ndimage.minimum_filter1d(g, size=la * 2 + 1)
    g = ndimage.uniform_filter1d(g, size=max(2, la // 2))
    out = y * (g[:, None] if y.ndim == 2 else g)
    # final hard safety (should be transparent)
    mx = np.max(np.abs(out))
    if mx > ceil:
        out *= ceil / mx
    return out.astype(np.float32), float(20 * np.log10(max(float(g.min()), 1e-9)))


# --------------------------------------------------------------------------- harmonic excitation
def spectral_repair(x, sr=SR, content_hz=6500.0, noise_cut_db=-14.0,
                    air_db=-31.0, excite_top=12000.0, tilt_db_per_oct=-6.0,
                    crossfade_oct=0.55):
    """
    Repair the missing top end in the only way that is honest.

    Diagnosis these transfers all share: real musical content stops around
    6-7 kHz, and everything above is tape hiss / AAC noise. Boosting that band
    therefore just amplifies noise. So:

      1. attenuate the noise-only band above `content_hz` (it contains no music)
      2. *synthesise* air by generating harmonics of the real 2.5-6 kHz content
      3. shape the generated spectrum with a natural -6 dB/octave tilt and
         cross-fade it in so the seam is inaudible
      4. the result is coherent with the music -> no hiss, no fizz, real air
    """
    from scipy import signal as sg
    y = np.asarray(x, dtype=np.float64)
    ch = y.shape[1] if y.ndim == 2 else 1
    outs = []
    for c in range(ch):
        s = y[:, c] if y.ndim == 2 else y

        # ---- 1. tame the noise-only band with a smooth shelf (linear phase-ish)
        nyq = sr / 2
        f_lo = content_hz * 2 ** (crossfade_oct / 2)
        f_hi = nyq - 200          # cover the whole noise-only region up to Nyquist
        stop = 10 ** (noise_cut_db / 20.0)
        b, a = sg.butter(4, [f_lo / nyq, f_hi / nyq], 'band')
        noise_band = sg.filtfilt(b, a, s)
        s = s - noise_band * (1.0 - stop)

        # ---- 2. synthesise coherent air from the real presence band.
        # Squaring/cubing a band-limited signal produces ONLY sum/difference
        # terms, i.e. genuine new harmonics - no copy of the original band.
        #   2.3-6.0 kHz squared  -> 4.6-12 kHz
        #   2.3-6.0 kHz cubed    -> 6.9-18 kHz
        b1, a1 = sg.butter(4, [2300 / nyq, min(content_hz, 6000.0) / nyq], 'band')
        pres = sg.filtfilt(b1, a1, s)
        pres = pres / (np.sqrt(np.mean(pres ** 2)) + 1e-12)
        gen = 0.30 * (pres ** 2) + 0.80 * (pres ** 3)
        gen = gen - gen.mean()
        # keep only the octaves we are replacing
        b2, a2 = sg.butter(4, (content_hz * 1.05) / nyq, 'high')
        gen = sg.filtfilt(b2, a2, gen)
        b3, a3 = sg.butter(4, min(excite_top, nyq - 400) / nyq, 'low')
        gen = sg.filtfilt(b3, a3, gen)
        # natural spectral tilt of the generated harmonics
        gen = tilt_filter(gen, sr, tilt_db_per_oct, ref_hz=content_hz * 1.2)
        gen = ndimage.gaussian_filter1d(gen, 4)     # decorrelate the 'fizz'
        outs.append((s, gen))

    if ch == 1:
        s, gen = outs[0]
    else:
        s = np.stack([o[0] for o in outs], axis=-1)
        gen = np.stack([o[1] for o in outs], axis=-1)
    # normalise the generated air to an explicit level relative to the programme,
    # so `air_db` is a direct, predictable control (-30 subtle, -24 obvious)
    prog = float(np.sqrt(np.mean(np.asarray(s) ** 2))) + 1e-12
    # measure the generated air where it actually lives (>7 kHz) so the
    # requested level is not diluted by the presence band
    from scipy import signal as _sg
    _b, _a = _sg.butter(4, 7000 / (sr / 2), 'high')
    g_air = np.asarray(gen)
    if g_air.ndim == 2:
        g_air = g_air.mean(axis=1)
    g_air = _sg.filtfilt(_b, _a, g_air)
    g_rms = float(np.sqrt(np.mean(g_air ** 2))) + 1e-12
    gen = gen * (prog * 10 ** (air_db / 20.0) / g_rms)
    return (s + gen).astype(np.float32)


def tilt_filter(x, sr=SR, tilt_db_per_oct=-6.0, ref_hz=7600.0, fmin=4000.0):
    """Apply a smooth dB-per-octave tilt above fmin (used to shape generated air)."""
    from scipy import signal as sg
    nyq = sr / 2
    freqs = [fmin * 0.8, fmin, ref_hz, ref_hz * 2, ref_hz * 4, nyq * 0.98]
    gains = []
    for f_ in freqs:
        octs = np.log2(max(f_, 1.0) / ref_hz)
        gains.append(float(np.clip(tilt_db_per_oct * octs, -30, 6)))
    return eq_curve(x, freqs, gains, sr, taps=4097)


# --------------------------------------------------------------------------- stereo
def haas_width(x, sr=SR, delay_ms=9.0, side_gain_db=-3.0, hp=350.0):
    """
    Turn a mono image into a natural, early-60s 'two-mic' stereo picture without
    the phase monster: broadband level stays neutral, only the side component of
    the >350 Hz region is delayed and attenuated.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = np.stack([y, y], axis=-1)
    mid = y.mean(axis=1)
    side = (y[:, 0] - y[:, 1]) / 2                      # ~0 for dual mono
    b, a = signal.butter(2, hp / (sr / 2), 'high')
    side_hp = signal.filtfilt(b, a, side)
    d = int(delay_ms * 1e-3 * sr)
    side_d = np.concatenate([np.zeros(d), side_hp])[:len(side_hp)]
    sg = 10 ** (side_gain_db / 20.0)
    out = np.stack([mid + sg * side_d, mid - sg * side_d], axis=-1)
    return out.astype(np.float32)


def mono_compat(x):
    """Keep mono sum equal to the original mono image (level-matched)."""
    y = np.asarray(x, dtype=np.float64)
    m = y.mean(axis=1)
    return m


# --------------------------------------------------------------------------- metering
def loudness(x, sr=SR):
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))


def true_peak_db(x, oversample=4):
    y = np.asarray(x, dtype=np.float64)
    up = signal.resample_poly(y, oversample, 1, axis=0)
    return float(20 * np.log10(max(float(np.max(np.abs(up))), 1e-20)))


# --------------------------------------------------------------------------- reference matching
# Restoration voicing.
#
# Rather than forcing every song onto a modern pop curve (which would mean
# hacking the bass out of a bass-led genre), we keep the *producer's* balance
# and correct only what the 1960s chain and the lossy encode demonstrably
# destroyed:
#   * a broad 200-500 Hz build-up from the acetate/tape transfer   -> -2.5 dB
#   * a 1.5-3 kHz scoop that swallows horns and vocal diction      -> +1.5 dB
#   * an early brickwall that kills everything above ~5 kHz        -> +5.5 dB
# The values are *corrections* added to the source's measured balance.
RESTORATION_CURVE_DB = {
    25: 0.0, 40: 0.0, 63: 0.0, 100: 0.0, 160: -0.5, 250: -1.8, 400: -2.5,
    630: -1.6, 1000: 0.0, 1600: 1.0, 2500: 1.1, 4000: 1.3, 6300: 0.8,
    8000: 0.8, 10000: 0.7, 12500: 0.6, 16000: 0.3,
}


def octave_balance(x, sr=SR, fmin=25.0, fmax=16000.0):
    """Third-octave-ish band levels, normalised to the 200-800 Hz region."""
    f, P = signal.welch(np.asarray(x, dtype=np.float64), sr, nperseg=16384)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    out = {}
    edges = [25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000,
             6300, 8000, 10000, 12500, 16000, 20000]
    for a, b in zip(edges[:-1], edges[1:]):
        if a > fmax:
            continue
        m = (f >= a) & (f < b)
        tot = float(P[m].sum()) if m.any() else 1e-30
        out[a] = 10 * np.log10(max(tot, 1e-30))
    ref_band = [v for k, v in out.items() if 200 <= k <= 800]
    ref = float(np.mean(ref_band)) if ref_band else 0.0
    return {k: round(v - ref, 2) for k, v in out.items()}


def match_target(x, sr=SR, curve=RESTORATION_CURVE_DB, strength=1.0, fmin=30.0,
                 fmax=16000.0, max_boost=6.0, max_cut=8.0):
    """
    Apply a fixed *restoration* voicing, scaled by `strength` and clipped.
    This is a corrective curve, not a re-voicing: the low end and the
    instrument balance of the original master are preserved.
    Returns (processed, applied_curve, before_balance, after_balance).
    """
    before = octave_balance(x, sr)
    keys = sorted(k for k in before
                  if fmin <= k <= fmax and k in curve and np.isfinite(before[k]))
    diff = {}
    for k in keys:
        d = float(np.clip(curve[k] * strength, -max_cut, max_boost))
        diff[k] = d
    # smooth the correction over neighbours so it stays musical
    ks = list(diff)
    vals = np.array([diff[k] for k in ks])
    sm = np.convolve(np.pad(vals, 1, mode='edge'), np.array([0.25, 0.5, 0.25]), mode='valid')
    curve = {k: float(v) for k, v in zip(ks, sm)}
    # extend flat to the extremes
    pts_f = [20] + ks + [20000]
    pts_g = [curve[ks[0]] * 0.5] + [curve[k] for k in ks] + [curve[ks[-1]] * 0.3]
    y = eq_curve(x, pts_f, pts_g, sr)
    after = octave_balance(y, sr)
    return y, curve, before, after


# --------------------------------------------------------------------------- small additive exciter (per stem)
def harmonic_excite(x, sr=SR, amount=0.22, band=(2600, 3400), order=(2, 3),
                    max_hz=15500.0):
    """
    Subtle additive exciter used *inside* the stems (not for restoring the
    missing top octaves - that is spectral_repair's job). Generates harmonics
    from a presence band so instruments cut through, at a low, musical level.
    """
    y = np.asarray(x, dtype=np.float64)
    ch = y.shape[1] if y.ndim == 2 else 1
    outs = []
    for c in range(ch):
        s = y[:, c] if y.ndim == 2 else y
        b, a = signal.butter(4, [band[0] / (sr / 2), band[1] / (sr / 2)], 'band')
        pres = signal.filtfilt(b, a, s)
        pres = pres / (np.sqrt(np.mean(pres ** 2)) + 1e-12)
        gen = np.zeros_like(pres)
        if 2 in order:
            gen += 0.7 * (pres ** 2)
        if 3 in order:
            gen += 0.5 * (pres ** 3)
        bh, ah = signal.butter(4, (band[1] * 1.3) / (sr / 2), 'high')
        gen = signal.filtfilt(bh, ah, gen - gen.mean())
        bl, al = signal.butter(4, min(max_hz, sr / 2 - 500) / (sr / 2), 'low')
        gen = signal.filtfilt(bl, al, gen)
        gen = ndimage.gaussian_filter1d(gen, 6)
        g_rms = float(np.sqrt(np.mean(gen ** 2))) + 1e-12
        outs.append(gen * (float(np.sqrt(np.mean(s ** 2))) * amount / g_rms))
    g = np.stack(outs, axis=-1)
    if g.shape[1] == 1:
        g = g[:, 0]
    return (y + g).astype(np.float32)
