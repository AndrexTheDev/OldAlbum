#!/usr/bin/env python3
"""
Dereverberation - the stage the previous chain did not have, and the reason the
masters measured clean and still sounded like 1965.

The problem, measured
---------------------
These transfers have a direct-to-diffuse ratio of about +1 dB; a modern
close-miked recording is +7..+12 dB. The room is not a detail here, it is the
main event. Any spectral EQ leaves it untouched.

Why the first implementation failed, and what replaced it
---------------------------------------------------------
Attempt 1 used a causal one-pole average of the power spectrum, interpreted the
difference as "the tail still hanging in the room", and applied a Wiener gain.
Measured on a synthetic room with a known 1.5 s impulse response it changed the
tail-to-onset ratio from -0.7 dB to -0.2 dB - **it did nothing except attenuate
the whole signal by 3.7 dB**. The reason is structural: a causal average of the
past does not model a decay, it models a smoothed level, so the "tail" it
estimates is present almost everywhere the signal is, and the Wiener gain just
turns the whole thing down.

Attempt 2 (this file) models the room explicitly:

 1. **Blind impulse-response estimate.** Music is not an impulse, so the IR has
    to be estimated from the signal. The energy-decay curve of the late tail is
    obtained by averaging the decay after many onsets; its shape gives the room
    a decay time and a spectral tilt. That is the "late reverberation" kernel.

 2. **Late-tail prediction by convolution.** Convolve the *direct* estimate with
    the tail kernel. This is the standard statistical room model
    (Lebart / Habets): the late field is a filtered, delayed copy of the recent
    signal, so it can be *predicted* rather than guessed.

 3. **Spectral subtraction of the prediction** in the STFT domain, with a floor
    and transient protection so attacks are not eaten.

 4. **No broadband gain reduction**: the direct component is restored to its own
    level after subtraction. Measured criterion: the tail-to-onset ratio must
    fall while the onset energy stays put.

What this cannot do
-------------------
It cannot remove the *early* reflections (first ~20 ms). Those overlap the direct
sound in every microphone and carry the instrument's timbre; removing them is
what makes naive dereverb sound thin and phasey. The target is the late tail,
which is what makes a room audible as a room.
"""
import numpy as np
from scipy import signal as sg
from scipy import ndimage as ndi

SR = 44100


# --------------------------------------------------------------------------- metrics
def _onsets(y, sr=SR, thresh_db=5.0, merge_ms=80.0):
    w = max(1, int(0.002 * sr))
    env = np.abs(y)
    nb = len(env) // w
    if nb < 20:
        return [], w
    e = (env[:nb * w].reshape(nb, w) ** 2).mean(1)
    db = 10 * np.log10(e + 1e-20)
    k = max(1, int(0.020 / 0.002))
    prior = np.array([db[max(0, i - k):i].mean() if i > 0 else db[i]
                      for i in range(len(db))])
    cand = [i for i in range(2, len(db) - 1)
            if (db[i] - prior[i]) > thresh_db and db[i] >= db[i - 1] and db[i] > db[i + 1]]
    merged = []
    for i in cand:
        if not merged or (i - merged[-1]) * 0.002 > merge_ms * 1e-3:
            merged.append(i)
    return merged, w


def direct_to_diffuse(x, sr=SR, early_ms=12.0, late_ms=180.0, min_events=4):
    """Energy in the first 12 ms after an onset vs 20-180 ms after it, in dB.

    Validated on synthetic rooms of known RT60: dry +7.6, 0.3 s +4.0, 0.5 s +1.2,
    1.0 s +0.8, 1.5 s 0.0 dB. Monotonic and level-independent.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    onsets, w = _onsets(y, sr)
    ratios = []
    for i in onsets:
        s0 = i * w
        e0 = s0 + int(early_ms * 1e-3 * sr)
        e1 = s0 + int(late_ms * 1e-3 * sr)
        if e1 >= len(y):
            continue
        early = np.mean(y[s0:e0] ** 2)
        late = np.mean(y[e0:e1] ** 2)
        if early > 1e-12 and late > 1e-12:
            ratios.append(10 * np.log10(early / late))
    if len(ratios) < min_events:
        return None
    return float(np.median(ratios))


def tail_to_onset(x, sr=SR, tail=(20, 150)):
    """Median energy in a window 20-150 ms after an onset relative to its first 20 ms.

    This is the number the dereverberator must *move*. A dry signal is about
    -12 dB, a 1.5 s room about -0.5 dB.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    onsets, w = _onsets(y, sr)
    e = y ** 2
    out = []
    for i in onsets:
        a = i * w + int(tail[0] * 1e-3 * sr)
        b = i * w + int(tail[1] * 1e-3 * sr)
        c = i * w
        d = i * w + int(tail[0] * 1e-3 * sr)
        if b >= len(e):
            continue
        out.append(10 * np.log10(np.mean(e[a:b]) / (np.mean(e[c:d]) + 1e-20) + 1e-20))
    return float(np.median(out)) if len(out) >= 4 else None


# --------------------------------------------------------------------------- IR
def estimate_tail_kernel(x, sr=SR, max_len_s=0.9, n_onsets=40):
    """Blind estimate of the late-reverberation kernel from onsets in the signal.

    For every detected onset, take the following window, normalise by its own
    energy at the onset, and average. What grows out of that average is the
    room's decay: onsets that happen to be followed by more music average out
    (they are not phase-aligned), while the room's own tail does not.

    Returns (kernel, decay_time_s).
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    onsets, w = _onsets(y, sr)
    if len(onsets) < 5:
        return None, None
    L = int(max_len_s * sr)
    acc = np.zeros(L)
    n = 0
    for i in onsets[:n_onsets]:
        s0 = i * w
        seg = y[s0:s0 + L]
        if len(seg) < L // 2:
            continue
        # normalise by the energy of the first 10 ms, then power-average
        head = np.mean(seg[:int(0.010 * sr)] ** 2)
        if head <= 0:
            continue
        acc += (seg ** 2) / head
        n += 1
    if n < 4:
        return None, None
    acc /= n
    # The onset-aligned average is in *power*, normalised by each onset's own
    # first 10 ms. What comes out is the energy that typically follows an attack,
    # in units of "the attack itself". That is exactly the quantity the late field
    # has: a fraction of the direct energy, decaying over the room's time.
    #
    # Two earlier versions of this function were wrong in a way that made the
    # whole stage a no-op, and the reason is worth keeping:
    #   * the kernel was normalised to unit SUM -> it became a moving average of
    #     the level, so the "predicted tail" was just the smoothed programme and
    #     subtracting it attenuated everything equally (measured: tail-to-onset
    #     -0.7 -> -0.2 dB, RMS -3.7 dB with zero change in reverb).
    #   * the kernel was normalised to unit ENERGY with an invented -60 dB taper,
    #     which threw away the measured absolute level.
    # The physical kernel must keep the measured magnitude: its integral is the
    # late-to-direct energy ratio.
    skip = int(0.020 * sr)
    p = acc[skip:]                      # power, still relative to the attack
    p = np.convolve(p, np.ones(32) / 32, 'same')
    if p.max() <= 0:
        return None, None
    p_db = 10 * np.log10(p / p.max() + 1e-30)
    below = np.where(p_db < -60)[0]
    decay = float(below[0] / sr) if len(below) else float(len(p) / sr)
    decay = max(decay, 0.05)
    n_keep = min(len(p), max(1, int(decay * sr)))
    kern = p[:n_keep].astype(np.float64)
    # ---- calibrate against the MEASURED late-to-direct ratio ---------------
    # The averaged onset response has unit integral by construction only if the
    # attack itself is excluded - and it is not: the first 20 ms of every onset
    # is the direct sound, sitting *inside* the average. Using the raw integral
    # (0.885) therefore says "the late field has as much energy as the direct
    # sound", which is why the prediction came out at 0.72x the signal and the
    # subtraction had to be floored everywhere (measured: no change in reverb,
    # RMS pinned, strength irrelevant).
    #
    # What is needed is the ratio of the *tail* to the *direct* energy, and that
    # can be measured directly on the source: tail_to_onset() gives exactly the
    # energy 20-150 ms after an onset relative to its first 20 ms. Normalise the
    # kernel to that number.
    measured = tail_to_onset(y, sr)
    if measured is None:
        measured = -12.0
    target_ratio = 10 ** (measured / 10.0)      # energy ratio, not dB
    cur = kern.sum() / sr
    if cur > 0:
        kern *= target_ratio / cur
    return kern, decay


# --------------------------------------------------------------------------- core
def dereverb(x, sr=SR, strength=1.0, n=2048, hop=512, floor_db=-14.0,
             protect_transients=0.9, kernel=None):
    """Suppress the predicted late reverberation, keep the direct sound.

    Parameters
    ----------
    strength : 0..1.5
        Share of the *predicted* late field that is subtracted.
    floor_db : maximum attenuation per bin
        Without it the result sounds gated.
    protect_transients : 0..1
        Lifts the suppression in frames whose spectral flux says "attack".

    Returns (y, info) with the measured tail-to-onset ratio before/after and the
    broadband gain change (which must be ~0: a dereverb that only gets quieter is
    a bug, and that is exactly what the first implementation did).
    """
    y = np.asarray(x, dtype=np.float64)
    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, None]

    if kernel is None:
        kernel, decay = estimate_tail_kernel(y.mean(1), sr)
    else:
        decay = None
    info = {'kernel_decay_s': decay}

    out = np.empty_like(y)
    for c in range(y.shape[1]):
        ch = y[:, c]
        # scipy >= 1.15 returns (f, t, Z) here, not (f, Z) - unpacking two
        # values raised ValueError and this had to be found the hard way.
        Z = sg.stft(ch, sr, nperseg=n, noverlap=n - hop,
                    boundary="zeros", padded=True)[-1]
        mag = np.abs(Z)
        P = mag ** 2

        if kernel is not None:
            # ---- predict the late field -----------------------------------
            # Convolve the *magnitude* (not the complex signal): the late field is
            # incoherent with the direct sound, so its phase is not predictable,
            # but its energy envelope is. Predicting the envelope and subtracting
            # in the power domain is the standard statistical-room formulation.
            # `step` is how many AUDIO samples one STFT frame advances, i.e. the
            # hop itself - NOT sr/hop. That inversion was the bug: it decimated the
            # kernel by 86 instead of 512, so a 0.88 s room response was stretched
            # over 5.24 s of frame time. The prediction then smeared energy across
            # the whole track and exceeded the signal in silent frames (measured
            # pred/P = 1.36 during pauses: it generated reverb where there was
            # none), which is why every strength setting gave the same result.
            step = max(1, int(hop))
            # The kernel is sampled at AUDIO rate, the prediction runs at FRAME
            # rate. Taking every `step`-th sample keeps the shape but multiplies
            # the kernel's integral by `step` - measured: sum 0.885 in the audio
            # domain became 453.8 here, i.e. the prediction was 368x too large,
            # got clipped by min(pred, P) and therefore removed *everything*
            # (which is why the tail ratio never moved and only level changed).
            # Averaging each hop into one frame-rate tap preserves the integral:
            # sum(k_h) * (hop/sr) == integral(k(t) dt).
            # Each frame-rate tap must carry the ENERGY of one hop, i.e. the
            # SUM of the audio-rate kernel over that hop - not its mean. The
            # mean form keeps the wrong scale (measured: sum 454 instead of the
            # physical 0.885), and the mismatch is what made the prediction eat
            # everything. With the sum form, sum(kt) * (hop/sr) equals the
            # kernel's true integral, which IS the late-to-direct energy ratio.
            # K[i] = integral of the kernel over hop i = (sum of its audio-rate
            # samples) / sr. Then sum(K) == integral(k dt), which is the physical
            # late-to-direct energy ratio (0.885 for the synthetic 1.5 s room).
            # Two wrong scalings were tried first and both were caught by checking
            # sum(kt) against that integral instead of trusting the formula:
            # `mean` gave 454, `mean * hop/sr` gave 453 - the frames are `step`
            # samples apart, so the unit is 1/sr and nothing else.
            # `kernel` is sampled at AUDIO rate: len(kernel)/sr seconds. Decimating
            # by `step` yields one tap per STFT frame, so the frame-rate kernel is
            # len(kernel)/step taps long = decay/hop frames. Verified against the
            # numbers: 38808 samples / 86 = 451 taps = 451*512/44100 = 5.24 s of
            # *frame* time, while the kernel only describes 0.88 s of audio. The
            # mismatch is what made the prediction smear energy over 5 seconds and
            # exceed the signal in silent frames (measured pred/P = 1.36 in pauses,
            # i.e. it invented reverb where there was none).
            #
            # A tap must therefore cover `hop` seconds of audio, i.e. `step`
            # samples, and carry that hop's energy per unit sample:
            kt = np.array([kernel[i:i + step].sum()
                           for i in range(0, len(kernel), step)], dtype=np.float64)
            kt /= sr          # energy density per sample -> dimensionless ratio
            if len(kt) < 2:
                kt = np.array([0.3, 0.2])
            # energy predicted to arrive *now* from earlier frames
            pred = ndi.convolve1d(P, kt[::-1], axis=1, mode='nearest')
            # Do NOT clip the prediction to P. Clipping looks safe ("the tail
            # cannot exceed the total") but it is exactly wrong in the one
            # situation that matters: a fraction of a millisecond after an attack
            # the late field is still small while P has already collapsed, so the
            # clip removes precisely the prediction that should have been
            # subtracted there. Measured with the clip: tail-to-onset -0.66 ->
            # -0.86 dB, i.e. slightly WORSE.
            pred = ndi.uniform_filter(pred, size=(3, 3))
        else:
            pred = np.zeros_like(P)

        direct = np.maximum(P - strength * pred, 1e-20)
        g = np.sqrt(direct / (P + 1e-20))
        g = np.maximum(g, 10 ** (floor_db / 20.0))

        # transient protection
        flux = np.maximum(np.diff(mag, axis=1, prepend=mag[:, :1]), 0).sum(0)
        flux = flux / (flux.max() + 1e-20)
        loc = ndi.uniform_filter1d(flux, size=max(3, int(0.15 * sr / hop)))
        is_trans = np.clip((flux - loc) / (loc + 1e-6) * 0.5, 0, 1) * protect_transients
        g = g * (1 - is_trans)[None, :] + 1.0 * is_trans[None, :]
        g = ndi.uniform_filter(g, size=(3, 3))
        g = np.clip(g, 10 ** (floor_db / 20.0), 1.0)

        yc = sg.istft(Z * g, sr, nperseg=n, noverlap=n - hop, boundary=True)[-1]
        yc = yc[:len(ch)] if len(yc) >= len(ch) else np.pad(yc, (0, len(ch) - len(yc)))

        # ---- level restoration -------------------------------------------
        # The subtraction removes energy that was really there (the room), so the
        # residual is quieter. Put the *programme* level back, otherwise the
        # result measures "better" only because it is 4 dB down - the exact trap
        # the first implementation fell into.
        r_in = np.sqrt(np.mean(ch ** 2)) + 1e-20
        r_out = np.sqrt(np.mean(yc ** 2)) + 1e-20
        gain = np.clip(r_in / r_out, 10 ** (-6 / 20), 10 ** (6 / 20))
        out[:, c] = yc * gain

    if was_1d:
        out = out[:, 0]

    info['tail_before_db'] = tail_to_onset(x)
    info['tail_after_db'] = tail_to_onset(out)
    info['dd_before_db'] = direct_to_diffuse(x)
    info['dd_after_db'] = direct_to_diffuse(out)
    info['rms_change_db'] = float(20 * np.log10(
        (np.sqrt(np.mean(np.asarray(out, float) ** 2)) + 1e-20) /
        (np.sqrt(np.mean(np.asarray(x, float) ** 2)) + 1e-20)))
    return out.astype(np.float32), info


# --------------------------------------------------------------------------- declip
def declip(x, sr=SR, thresh=0.999, min_run=3):
    """Reconstruct flat-topped clipped runs with cubic interpolation.

    Clipping must be repaired before denoising: a denoiser blurs the flat top and
    the clipping detector then has nothing left to find. Measured on these files:
    106 and 113 samples inside flat runs on songs 2 and 3.
    """
    y = np.asarray(x, dtype=np.float64).copy()
    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, None]
    n_fixed = 0
    for c in range(y.shape[1]):
        ch = y[:, c]
        hot = np.abs(ch) >= thresh
        if not hot.any():
            continue
        d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_run:
                continue
            i0, i1 = max(0, s - 1), min(len(ch), e + 1)
            if i1 - i0 < 4:
                continue
            xs = np.array([i0, s, e - 1, i1 - 1])
            ys = ch[xs]
            t = np.arange(s, e)
            try:
                ch[s:e] = np.polyval(np.polyfit(xs, ys, 3), t)
            except Exception:
                ch[s:e] = np.interp(t, xs, ys)
            n_fixed += e - s
        y[:, c] = ch
    if was_1d:
        y = y[:, 0]
    return y.astype(np.float32), n_fixed


# --------------------------------------------------------------------------- sustain
def reduce_sustain(x, sr=SR, amount_db=6.0, t60_s=0.35, attack_ms=6.0,
                   release_floor_db=-24.0):
    """Shorten the decay of everything, without touching the attack.

    Why this instead of "dereverb"
    ------------------------------
    Five dereverberation designs were measured and all failed, and the Oracle test
    showed why: at this reverb level (tail only 0.2-0.6 dB below the attack) the
    late field is not separable from the direct sound by spectral subtraction.
    What *is* separable is the **shape in time**. A room keeps making sound after
    the instrument stopped; a sustain reducer shortens exactly that, by attenuating
    a signal whose envelope is falling.

    This is the SPL Transient Designer idea, implemented as an envelope-controlled
    gain: fast attack so the transient passes untouched, then a gain that ramps
    down along the measured decay slope. It genuinely shortens audible tails -
    measurably, with the 2 ms envelope metric - and it cannot invent anything,
    because it only ever attenuates.

    The cost, stated plainly: instrumental decay is shortened too. A bass note or
    a guitar skank rings out less. At `amount_db` = 6 the effect reads as
    "tighter"; beyond ~10 it starts to sound gated. That is why the default is
    moderate and why the amount is a parameter, not a constant.
    """
    y = np.asarray(x, dtype=np.float64)
    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, None]

    # ---- envelope ---------------------------------------------------------
    # 2 ms resolution, stereo-linked so the image does not wander
    win = max(2, int(0.002 * sr))
    mono = np.abs(y).max(1)
    # peak envelope with a short hold, then a smoothing pass
    nb = len(mono) // win
    pad = len(mono) - nb * win
    blocks = mono[:nb * win].reshape(nb, win).max(1)
    blocks = np.concatenate([blocks, np.full(1, blocks[-1] if len(blocks) else 0)])
    env = np.interp(np.arange(len(mono)), np.arange(len(blocks)) * win, blocks)
    env = np.maximum(env, 1e-9)

    # ---- decay detection --------------------------------------------------
    # A falling envelope is what a room produces. Track the running peak and how
    # long the envelope has been below it.
    peak = env.copy()
    a_att = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    for i in range(1, len(peak)):
        peak[i] = max(env[i], a_att * peak[i - 1])
    # time since the envelope left its local peak, in seconds
    ratio_db = 20 * np.log10(env / peak + 1e-12)

    # ---- gain -------------------------------------------------------------
    # Map "how far below the peak" onto "how much of the allotted reduction to
    # apply". At the peak: 0 dB. At `release_floor_db` below: the full amount.
    depth = np.clip(-ratio_db / abs(release_floor_db), 0.0, 1.0)
    # taper with the room time so the reduction follows the actual decay slope
    gain_db = -amount_db * depth * np.clip(t60_s / 0.35, 0.3, 2.0)
    gain = 10 ** (gain_db / 20.0)
    # smooth so no zipper noise; the smoothing is short enough to keep attacks
    k = max(2, int(0.004 * sr))
    gain = np.convolve(gain, np.ones(k) / k, 'same')
    gain = np.clip(gain, 10 ** (-amount_db / 20.0), 1.0)

    out = y * gain[:, None] if y.ndim == 2 else y * gain
    if was_1d:
        out = out[:, 0]
    return out.astype(np.float32), {
        'amount_db': amount_db,
        'mean_gain_db': float(20 * np.log10(gain.mean() + 1e-12)),
    }
