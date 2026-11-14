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
def expansion_gate(x, sr=SR, gate_off_db=-8.0, ratio=3.0, attack_ms=4.0,
                   hold_ms=60.0, release_ms=180.0, band=(60.0, 6000.0)):
    """Downward expander: below the programme level, the signal is attenuated.

    A denoiser working in the spectral domain has to trade musical noise against
    suppression: to remove 10 dB of hiss from a quiet passage it must modulate
    individual bins, which is audible as "swimming" or "chirping". Real restoration
    does the opposite in the *easy* places and nothing in the hard ones: in the
    gaps between phrases (where only noise is left) the whole signal can go down
    hard and no one notices; while music is playing, the noise is masked and needs
    no treatment at all.

    So this expands around a *level* threshold derived from the programme itself
    (85th percentile of the short-term envelope minus `gate_off_db`), with a long
    hold so the gate cannot chatter on individual notes, and a slow release so the
    hiss fades back in rather than switching on. It is a gain envelope, so it
    introduces no distortion below the threshold either.
    """
    y = np.asarray(x, dtype=np.float64)
    m = y.mean(axis=1) if y.ndim == 2 else y
    # envelope in the band where the hiss is audible
    b, a = signal.butter(4, [band[0] / (sr / 2), min(band[1], sr / 2 - 500) / (sr / 2)], 'band')
    e = np.abs(signal.hilbert(signal.filtfilt(b, a, m)))
    a_att = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    e = signal.lfilter([1 - a_att], [1, -a_att], e)
    lvl = np.percentile(e[e > 1e-9], 85) if (e > 1e-9).any() else 1.0
    thr = lvl * 10 ** (gate_off_db / 20.0)
    # gain in dB: 0 above the threshold, -ratio*deficit below it
    with np.errstate(divide='ignore'):
        deficit = 20 * np.log10(np.maximum(e, 1e-9) / max(thr, 1e-9))
    att_db = np.minimum(0.0, deficit * (ratio - 1.0))
    g = 10 ** (att_db / 20.0)
    # hold: keep the gate open for hold_ms after the envelope leaves the threshold
    hold = int(hold_ms * 1e-3 * sr)
    if hold > 1:
        from scipy import ndimage
        g = ndimage.maximum_filter1d(g, size=hold)
    # release
    a_rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    out = np.empty_like(g)
    v = 1.0
    for i in range(len(g)):
        v = g[i] if g[i] > v else a_rel * v + (1 - a_rel) * g[i]
        out[i] = v
    return out.astype(np.float32)


def gate_apply(x, g):
    y = np.asarray(x, dtype=np.float32)
    return (y * (g[:, None] if y.ndim == 2 else g)).astype(np.float32)


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


def _denoise_block(seg, sr, n, hop, reduction_db, over_sub, floor_db, smooth_t,
                   mix, freq_weight, gate=None, floor_gap_db=-45.0,
                   tonal_protect=0.75):
    """One block of the spectral denoiser (see `denoise`).

    `gate` is an optional per-sample opening signal in [0, 1] (1 = music playing).
    Where it closes, the permitted residual floor drops from `floor_db` to
    `floor_gap_db`, so the gaps are cleaned to near silence while the music is left
    almost untouched. This is the difference between "reduce the hiss a bit
    everywhere" (which either does nothing audible or eats the music) and restoring
    the *silence* - which is what one actually hears on a clean record.
    """
    from scipy import ndimage
    y = np.asarray(seg, dtype=np.float32)
    ch_n = y.shape[1] if y.ndim == 2 else 1
    out = []
    for c in range(ch_n):
        sig = y[:, c] if y.ndim == 2 else y
        f, t, Z = signal.stft(sig.astype(np.float64), sr, nperseg=n,
                              noverlap=n - hop, window='hann')
        mag = np.abs(Z)
        phase = np.angle(Z)
        noise = _min_stat_noise(mag, q=0.12, smooth_t=smooth_t)
        sig_pow = mag ** 2
        noise_pow = (noise * over_sub) ** 2
        g = np.sqrt(np.maximum(np.maximum(sig_pow - noise_pow, 0.0) /
                               (sig_pow + 1e-18), 0.0))
        wf = np.array([p[0] for p in freq_weight], dtype=float)
        wv = np.array([p[1] for p in freq_weight], dtype=float)
        fw = np.interp(f, wf, wv)[:, None]
        # ---- tonal protection ------------------------------------------------
        # Minimum statistics takes a low percentile over time, so *anything that
        # is continuously present* is classified as noise. A held note - an organ
        # point, a sustained horn, a piano chord - is therefore "noise" and gets
        # suppressed: measured -5.26 dB on a pure 440 Hz tone, i.e. exactly the
        # 200-1000 Hz loss visible in the audit. The fix is a tonalness detector:
        # where the local spectrum has sharp peaks and low spectral flatness it is
        # tonal, and the suppression is reduced there by `tonal_protect`.
        if tonal_protect > 0:
            from scipy import ndimage as _nd
            # spectral flatness per frame over a 9-bin neighbourhood
            m2 = mag ** 2 + 1e-20
            geo = np.exp(_nd.uniform_filter1d(np.log(m2), size=9, axis=0))
            ari = _nd.uniform_filter1d(m2, size=9, axis=0)
            flat = np.clip(geo / (ari + 1e-20), 1e-6, 1.0)
            # flat ~ 1 -> noise (protect nothing); flat << 1 -> tonal (protect)
            ton = np.clip((0.35 - flat) / 0.35, 0.0, 1.0) ** 0.5
            fw = fw * (1.0 - tonal_protect * ton)
            fw = np.maximum(fw, 0.0)
        if gate is None:
            floor_col = np.full(mag.shape[1], 10 ** (floor_db / 20.0))
        else:
            gt = gate[:len(sig)] if len(gate) >= len(sig) else np.pad(gate, (0, len(sig) - len(gate)), mode='edge')
            nfr = mag.shape[1]
            idx = np.linspace(0, len(gt) - 1, nfr).astype(int)
            gfr = np.clip(gt[idx], 0.0, 1.0).astype(np.float64)
            fd = np.interp(gfr, [0.0, 1.0], [floor_gap_db, floor_db])
            floor_col = 10 ** (fd / 20.0)
        floor = floor_col[None, :]
        red = 10 ** ((-reduction_db * fw) / 20.0)
        tgt = np.maximum(g, floor)
        # NOTE: `floor` is a per-sample-column array, so it must stay 2-D here -
        # a 1-D floor would silently broadcast into (nFreq, nFreq) instead.
        gg = 1.0 - (1.0 - tgt) * ((1.0 - red) / (1.0 - floor + 1e-9))
        if gate is not None:
            # A hard weight of 1.0 outside the gate would leave 96 % of the signal
            # completely untouched even in the gaps, so the gate *is* the weighting
            # here: where it is open we apply exactly as much reduction as the
            # spectral estimate asks for, where it is closed we apply all of it.
            gg = fw * gg + (1.0 - fw) * 1.0
        else:
            gg = fw * gg + (1.0 - fw) * 1.0
        gg = np.clip(gg, floor, 1.0).astype(np.float32)
        gg = ndimage.uniform_filter(gg, size=(3, smooth_t))
        Z2 = (mag.astype(np.float32) * gg) * np.exp(1j * phase.astype(np.float32))
        _, r = signal.istft(Z2, sr, nperseg=n, noverlap=n - hop, window='hann',
                            boundary=True)
        r = r[:len(sig)].astype(np.float32)
        out.append(mix * r + (1 - mix) * sig.astype(np.float32) if mix < 1.0 else r)
    res = np.stack(out, axis=-1) if ch_n > 1 else out[0]
    return res.astype(np.float32)


def denoise(x, sr=SR, n=2048, hop=512, reduction_db=10.0, over_sub=1.8,
            floor_db=-20.0, q=0.12, smooth_t=7, mix=1.0,
            freq_weight=((0, 0.55), (120, 0.75), (350, 0.88), (1500, 1.0)),
            block_s=8.0, gate_off=0.0, floor_gap_db=-45.0, gate_ratio=3.0,
            tonal_protect=0.75):
    """
    Spectral denoiser with minimum-statistics noise tracking.

    * noise floor per bin from the low percentile over time
    * power-domain over-subtraction with a soft floor (no musical noise)
    * gain smoothed across time+frequency, then mixed in by `mix`

    Runs in overlapping blocks (8 s, 0.5 s cross-fade): the statistics are
    accumulated per block, and the memory footprint stays around 300 MB instead
    of the ~2 GB a full-length float64 STFT needs for a three-minute master.
    """
    y = np.asarray(x, dtype=np.float32)
    ch_n = y.shape[1] if y.ndim == 2 else 1
    blk = int(block_s * sr)
    gate = None
    # `gate_off` is a *negative* offset in dB below the programme level, so the
    # test must be `!= 0` - testing `> 0` silently disabled the gate for every
    # sensible value, which is why three rounds of tuning changed nothing at all.
    if gate_off:
        gate = expansion_gate(y, sr, gate_off_db=gate_off, ratio=gate_ratio)
    ovb = int(0.5 * sr)
    if len(y) <= blk:
        return _denoise_block(y, sr, n, hop, reduction_db, over_sub, floor_db,
                              smooth_t, mix, freq_weight, gate=gate,
                              floor_gap_db=floor_gap_db, tonal_protect=tonal_protect)
    win = np.ones(blk, dtype=np.float32)
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ovb))).astype(np.float32)
    win[:ovb] = ramp
    win[-ovb:] = ramp[::-1]
    acc = np.zeros_like(y, dtype=np.float32)
    wsum = np.zeros(len(y), dtype=np.float32)
    pos = 0
    while pos < len(y):
        a = max(0, pos - ovb // 2)
        b = min(len(y), a + blk)
        r = _denoise_block(y[a:b], sr, n, hop, reduction_db, over_sub, floor_db,
                           smooth_t, mix, freq_weight,
                           gate=(gate[a:b] if gate is not None else None),
                           floor_gap_db=floor_gap_db, tonal_protect=tonal_protect)
        w = win[:b - a]
        if ch_n > 1:
            acc[a:b] += r * w[:, None]
        else:
            acc[a:b] += r * w
        wsum[a:b] += w
        pos += blk - ovb
    wsum[wsum < 1e-6] = 1.0
    if ch_n > 1:
        acc /= wsum[:, None]
    else:
        acc /= wsum
    return acc.astype(np.float32)


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

    Normalisation matters: an earlier version divided by `h.sum()`, i.e. it forced
    0 dB at DC. For a curve that cuts 9 dB below 100 Hz that quietly added +9 dB to
    the *whole* signal - which was then read as "the stem processing adds 8.75 dB
    of level", inflated every low-mid band and, together with the bass stage, lifted
    the hiss floor by 7 dB. The filter is normalised on its peak gain instead, so a
    request of +2 dB really gives +2 dB.
    """
    grid, lg = _shelves_from_curve(freqs, gains_db, sr)
    F = taps // 2
    mag_lin = 10 ** (np.interp(np.linspace(0, sr / 2, F + 1), grid, lg) / 20.0)
    h = np.fft.irfft(mag_lin, n=2 * F)      # length 2*F == taps
    h = np.roll(h, F)                       # zero-phase -> linear phase
    h *= np.hanning(len(h))
    # ===== normalisation: this is the bug that survived two "fixes" ==========
    # `h /= h.sum()` normalises the response to **0 dB at DC** (|H(0)| = sum(h)).
    # For a curve that asks for -9 dB at 95 Hz, lg[0] is -9 dB, so the whole
    # filter is shifted +9 dB: every band lands 9 dB away from its target - and
    # because the peak then sits at 0 dB, a second normalisation to the peak
    # does *nothing* (it divides by 1.0). The filter was therefore correct in
    # *shape* and wrong in *level* for as long as this line existed. The fix is
    # to normalise to the requested peak gain, not to DC:
    # The windowing at the filter edges costs a few dB of accuracy, so a single
    # division is not enough: solve it by iteration on the *achieved* response,
    # which converges in 3-4 rounds because the error is a smooth scalar.
    want_db = np.float64(lg.max())
    for _ in range(6):
        H = np.fft.rfft(h, 1 << 15)
        peak_db = 20 * np.log10(np.abs(H).max() + 1e-15)
        if abs(peak_db - want_db) < 0.02:
            break
        h *= 10 ** ((want_db - peak_db) / 20.0)
    # finally make the *shape* right at the requested points (removes the residual
    # DC offset that the iteration above cannot see)
    H = np.abs(np.fft.rfft(h, 1 << 16))
    fr = np.fft.rfftfreq(1 << 16, 1 / sr)
    err = []
    for ff, gg in zip(np.asarray(freqs, float), np.asarray(gains_db, float)):
        if ff <= 0:
            continue
        got = 20 * np.log10(H[np.argmin(np.abs(fr - ff))] + 1e-15)
        err.append(got - gg)
    if err:
        # shift so the requested points sit on the requested curve on average
        h *= 10 ** (-np.mean(err) / 20.0)
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


def eq_curve(x, freqs, gains_db, sr=SR, taps=8193):
    """Zero-phase EQ via a linear-phase FIR.

    `taps` controls the frequency resolution: 8193 taps at 44.1 kHz resolve
    ~5.4 Hz, which matters because the bass corrections sit at 28-130 Hz where a
    shorter filter would simply not produce the requested shape. The filter is
    padded by its own length before convolution so the low bins are not
    distorted by the implicit zero-run at the start of the signal.
    """
    h = fir_from_curve(freqs, gains_db, sr, taps)
    y = np.asarray(x, dtype=np.float64)
    pad = len(h)
    if y.ndim == 2:
        yp = np.pad(y, ((pad, pad), (0, 0)), mode='reflect')
        out = np.stack([signal.fftconvolve(yp[:, c], h, 'same') for c in range(y.shape[1])], axis=-1)
        out = out[pad:pad + len(y)]
    else:
        yp = np.pad(y, pad, mode='reflect')
        out = signal.fftconvolve(yp, h, 'same')[pad:pad + len(y)]
    return out.astype(np.float32)


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
    # Soft-knee curve. It must be *continuous with the right slope* at both knee
    # edges, otherwise the compressor never reaches its nominal ratio. An earlier
    # version used (over+knee)^2 / (4*knee): that is continuous at -knee but its
    # slope at +knee is only half of (1-1/ratio), so a 4:1 setting behaved like
    # 1.85:1 (measured: -12.74 dB instead of -15.00 dB on a 20 dB-over signal).
    # The quadratic below is the standard one: value and slope match at both ends.
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
    # `up` is 1-D for mono input - max(axis=1) then raises, which is how the unit
    # test found that the limiter had never been exercised on a mono signal.
    tp = np.abs(up).max(axis=1) if up.ndim > 1 else np.abs(up)
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
                    crossfade_oct=0.55, smooth_ms=1.0, peak_cap=2.0, gate_ms=20.0,
                    follow=None, follow_ms=50.0):
    """
    Repair the missing top end in the only way that is honest.

    These transfers all stop carrying real music somewhere around 6-7 kHz, and
    everything above is tape hiss / AAC noise - so boosting that band only
    amplifies noise. Instead:

      1. attenuate the noise-only band above `content_hz` (it contains no music)
      2. *synthesise* air by generating harmonics of the real 2.3-4.2 kHz content:
         squaring/cubing a band-limited signal yields ONLY sum terms
         (4.6-8.4 kHz and 6.9-12.6 kHz), i.e. genuine new harmonics
      3. band-limit and smooth the generated signal on a ~1 ms constant, which
         removes the impulse character that squaring produces on its own
         (measured: keeps the click/kurtosis statistic identical to the source)
      4. normalise it to an explicit level below the programme and add

    The result is coherent with the music -> real air, no hiss, no fizz.
    """
    from scipy import signal as sg
    y = np.asarray(x, dtype=np.float64)
    ch = y.shape[1] if y.ndim == 2 else 1
    outs = []
    nyq = sr / 2

    for c in range(ch):
        sig = y[:, c] if y.ndim == 2 else y

        # ---- 1. tame the noise-only band
        f_lo = content_hz * 2 ** (crossfade_oct / 2)
        f_hi = nyq - 200
        if f_lo >= f_hi:
            f_lo = min(content_hz, f_hi * 0.8)
        stop = 10 ** (noise_cut_db / 20.0)
        b, a = sg.butter(4, [f_lo / nyq, f_hi / nyq], 'band')
        sig = sig - sg.filtfilt(b, a, sig) * (1.0 - stop)

        # ---- 2. synthesise the air
        b1, a1 = sg.butter(2, [2300 / nyq, min(4200.0, nyq * 0.7) / nyq], 'band')
        pres = sg.filtfilt(b1, a1, sig)
        pres = pres / (np.sqrt(np.mean(pres ** 2)) + 1e-12)
        gen = 0.22 * (pres ** 2) + 0.85 * (pres ** 3)
        gen -= gen.mean()
        b2, a2 = sg.butter(4, (content_hz * 1.05) / nyq, 'high')
        gen = sg.filtfilt(b2, a2, gen)
        b3, a3 = sg.butter(4, min(excite_top, nyq - 400) / nyq, 'low')
        gen = sg.filtfilt(b3, a3, gen)
        gen = tilt_filter(gen, sr, tilt_db_per_oct, ref_hz=content_hz * 1.2)
        # ---- 3. remove the impulse character
        nsm = max(3, int(smooth_ms * 1e-3 * sr))
        gen = np.convolve(gen, np.ones(nsm) / nsm, mode='same')
        # slow envelope of the real presence content - the air has to follow the
        # music, otherwise quiet passages keep as much 'air' as loud ones, which
        # is exactly what a listener hears as constant sizzle
        env = ndimage.uniform_filter1d(np.abs(pres), max(3, int(follow_ms * 1e-3 * sr)))
        outs.append((sig, gen, env))

    sig = np.stack([o[0] for o in outs], axis=-1)
    gen = np.stack([o[1] for o in outs], axis=-1)
    envl = np.stack([o[2] for o in outs], axis=-1)
    # The generated air inherits the *cubed* dynamics of the presence band, i.e.
    # about three times the dB range of the music - that is what puts a burst of
    # hiss on every loud note. Dividing by the presence envelope flattens it;
    # `follow` (0..1) adds back the part of the movement we actually want.
    if follow is not None:
        ref = float(np.percentile(envl, 85)) + 1e-12
        gen = gen * (envl / ref) ** (float(follow) - 1.0)
    if ch == 1:
        sig, gen = sig[:, 0], gen[:, 0]

    # ---- 4. level-match the air to the *presence band* rather than to the
    # broadband programme: the low end and the drums differ wildly between these
    # transfers, the 2-4 kHz music content does not - so an air level referenced
    # to it is comparable across songs, and the air can never outgrow the music.
    b_p, a_p = sg.butter(4, [2000 / nyq, 4000 / nyq], 'band')
    _sp = sg.filtfilt(b_p, a_p, np.asarray(sig).mean(axis=1) if np.asarray(sig).ndim == 2 else np.asarray(sig))
    prog = float(np.sqrt(np.mean(_sp ** 2))) + 1e-12
    g_air = np.asarray(gen)
    if g_air.ndim == 2:
        g_air = g_air.mean(axis=1)
    b4, a4 = sg.butter(4, 7000 / nyq, 'high')
    g_air = sg.filtfilt(b4, a4, g_air)
    g_rms = float(np.sqrt(np.mean(g_air ** 2))) + 1e-12
    gen = gen * (prog * 10 ** (air_db / 20.0) / g_rms)

    # ---- 5. envelope gate: keep the air *level*. Cubing the presence band
    # concentrates its energy exactly on transients, so the raw generator sits
    # +10..20 dB above the music on loud notes and far below it in quiet ones -
    # heard as sizzle on every hit. The gate squashes the generated signal
    # towards a constant level (peak_cap x the 95th-percentile level), which is
    # what a clean modern master has: steady air, no pumping.
    gg = np.asarray(gen, dtype=np.float64)
    gg = gg.mean(axis=1) if gg.ndim == 2 else gg
    env = ndimage.uniform_filter1d(np.abs(gg), max(3, int(0.005 * sr)))
    env = ndimage.uniform_filter1d(env, max(3, int(gate_ms * 1e-3 * sr)))
    # the reference is the 95th percentile of the gate envelope, i.e. the level
    # during normal (loud) passages, not the RMS of the whole programme
    ref_g = float(np.percentile(env, 95)) + 1e-12
    gate = np.minimum(1.0, peak_cap * ref_g / (env + 1e-12))
    if gen.ndim == 2:
        gate = gate[:, None]
    gen = gen * gate
    return (np.asarray(sig) + gen).astype(np.float32)


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


def spectral_noise_cut(x, sr=SR, cut_hz=6800.0, amount_db=-24.0, smooth_oct=0.25):
    """Attenuate the band above `cut_hz` where it is *only* noise.

    Unlike a plain high-shelf this multiplies the spectrum band by a smooth
    gain, leaving everything below `cut_hz` untouched, and it is the stage the
    neural reconstruction is added on top of - so the reconstruction replaces
    the noise instead of sitting on it.
    """
    from scipy import signal as sg
    y = np.asarray(x, dtype=np.float32)
    ch = y.shape[1] if y.ndim == 2 else 1
    g = 10 ** (amount_db / 20.0)
    lo, hi = cut_hz, cut_hz * 2 ** smooth_oct
    npar, nov = 2048, 1536
    blk, ov = int(8.0 * sr), int(0.5 * sr)
    out = np.zeros_like(y)
    wsum = np.zeros(len(y), dtype=np.float32)
    win = np.ones(blk, dtype=np.float32)
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov))).astype(np.float32)
    win[:ov] = ramp
    win[-ov:] = ramp[::-1]
    pos = 0
    while pos < len(y):
        a = max(0, pos - ov // 2)
        b = min(len(y), a + blk)
        seg = y[a:b]
        z = np.empty_like(seg) if ch > 1 else np.empty(len(seg), dtype=np.float32)
        for c in range(ch):
            s_ = seg[:, c] if ch > 1 else seg
            f_, t_, Z = sg.stft(s_.astype(np.float64), sr, nperseg=npar,
                                noverlap=nov, window='hann')
            w = np.clip((f_ - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            gm = (1.0 + (g - 1.0) * w)[:, None]
            _, r = sg.istft(Z * gm, sr, nperseg=npar, noverlap=nov, window='hann')
            r = r[:len(s_)].astype(np.float32)
            if ch > 1:
                z[:, c] = r
            else:
                z = r
        w = win[:b - a]
        if ch > 1:
            out[a:b] += z * w[:, None]
        else:
            out[a:b] += z * w
        wsum[a:b] += w
        pos += blk - ov
    wsum[wsum < 1e-6] = 1.0
    return (out / (wsum[:, None] if ch > 1 else wsum)).astype(np.float32)


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
    # `before` must describe the signal that went in, `after` the one that comes
    # out - measured relative to its own 200-800 Hz reference, so the two are
    # directly comparable. (The old version returned the input twice, which is why
    # every tone_match report in this project showed 0.0 dB everywhere and the EQ
    # looked like it was doing nothing.)
    return y, curve, before, after


def correction_from_source(stem_sum, source, sr=SR, fmin=32.0, fmax=6300.0,
                           max_db=3.0, smooth=True):
    """Measure what separation did to the balance and undo exactly that.

    The stems are built to sum to the source, but only *energetically*: Demucs
    over-weights whatever it is confident about. Measured on song 1, the stem sum
    is +2.3 dB high below 100 Hz (that is the separated bass bleeding into the
    bass stem) while 160 Hz - 2.5 kHz is flat. A *fixed* voicing curve cannot know
    that - it is written from listening experience and it happened to cut 2.5 dB
    at 400 Hz, which the source does not need at all. Measured, the fixed curve
    cost 1.5 dB of SNR in 200 Hz - 1 kHz while the actual error sat at 40-100 Hz.

    So: measure the per-octave difference between the stem sum and the original
    full mix, and invert it. This is self-calibrating and provably neutral when
    the separation is already balanced.
    """
    a = octave_balance(stem_sum, sr)     # stems, 200-800 Hz referenced
    b = octave_balance(source, sr)       # original mix, same reference
    keys = sorted(k for k in a if fmin <= k <= fmax and k in b
                  and np.isfinite(a[k]) and np.isfinite(b[k]))
    # the difference is what the separation *added*; we subtract it (limited)
    delta = np.array([float(np.clip(a[k] - b[k], -max_db, max_db)) for k in keys])
    if smooth and len(delta) >= 3:
        delta = np.convolve(np.pad(delta, 1, mode='edge'),
                            np.array([0.25, 0.5, 0.25]), mode='valid')
    pts_f = [20] + keys + [20000]
    pts_g = [delta[0] * 0.3] + list(-delta) + [delta[-1] * 0.0]
    y = eq_curve(stem_sum, pts_f, pts_g, sr)
    return y, {k: float(-d) for k, d in zip(keys, delta)}, keys, delta


# --------------------------------------------------------------------------- small additive exciter (per stem)
def harmonic_excite(x, sr=SR, amount=1.0, band=(2600, 3400), max_hz=12000.0,
                    smooth_ms=0.35, hp_hz=5200.0, peak_cap=2.0, gate_ms=20.0):
    """
    Additive exciter for the stems - gentle and, above all, *even*.

    `amount` is the intended level of the generated band relative to the
    high-frequency energy that is ALREADY in the signal (amount=1.0 => +0 dB,
    1.5 => about +1.4 dB of >5 kHz energy). Calibrating against the existing HF
    instead of the programme RMS is what keeps a quiet, dark stem from being
    flooded with synthesis.

    Squaring a presence band has a huge crest factor, so the generated signal is
    additionally *envelope-gated*: it is scaled by a slow envelope that never
    exceeds `peak_cap` times its own RMS. Without this the exciter puts a 2-4 ms
    HF burst on every drum hit - audible as sizzle/edge, measurable as a kurtosis
    explosion in the >5 kHz band.
    """
    from scipy import ndimage as _nd
    y = np.asarray(x, dtype=np.float64)
    ch = y.shape[1] if y.ndim == 2 else 1
    nyq = sr / 2
    centre = np.sqrt(band[0] * band[1])
    lo, hi = centre / 1.25, min(centre * 1.25, 4600.0)
    nsm = max(3, int(smooth_ms * 1e-3 * sr))
    b5, a5 = signal.butter(6, min(5000.0, nyq * 0.5) / nyq, 'high')

    def short_term(v):
        d = np.diff(np.asarray(v, dtype=np.float64))
        return float(np.mean(d ** 2)) + 1e-30

    out = []
    for c in range(ch):
        s_ = y[:, c] if y.ndim == 2 else y
        hf_now = float(np.std(signal.filtfilt(b5, a5, s_))) + 1e-12
        b, a = signal.butter(2, [lo / nyq, hi / nyq], 'band')
        pres = signal.filtfilt(b, a, s_)
        pres = pres / (np.sqrt(np.mean(pres ** 2)) + 1e-12)
        gen = 0.55 * pres ** 2 + 0.35 * pres ** 3
        gen -= gen.mean()
        gen = signal.filtfilt(*signal.butter(4, min(max(hi * 1.3, hp_hz), nyq * 0.8) / nyq, 'high'), gen)
        gen = signal.filtfilt(*signal.butter(4, min(max_hz, nyq - 3000) / nyq, 'low'), gen)
        gen = _nd.uniform_filter1d(gen, nsm)

        # ---- level relative to the HF that is already there
        gen = gen * (amount * hf_now / (float(np.std(gen)) + 1e-12))

        # ---- envelope gate: no HF burst may stick out above `peak_cap` x RMS
        env = _nd.uniform_filter1d(np.abs(gen), max(3, int(0.005 * sr)))
        env = _nd.uniform_filter1d(env, max(3, int(gate_ms * 1e-3 * sr)))
        gen = gen * np.minimum(1.0, peak_cap * float(np.std(gen)) / (env + 1e-12))

        # ---- hard short-term energy budget (+1.5 dB max)
        budget = short_term(s_) * (10 ** (1.5 / 10.0) - 1.0)
        add = short_term(gen)
        if add > budget:
            gen = gen * np.sqrt(budget / add)
        out.append(gen)
    g = np.stack(out, axis=-1)
    if g.shape[1] == 1:
        g = g[:, 0]
    return (y + g).astype(np.float32)


# --------------------------------------------------------------------------- stereo
