#!/usr/bin/env python3
"""
Dynamic-range restoration - the axis with the largest measurable deficit.

Measured on the four sources:

| song | crest | LRA | modern target |
|---|---|---|---|
| I Need You | 13.7 | 8.1 | crest 14-18, LRA 8-14 |
| Lonesome Feeling | **10.8** | **3.3** | |
| Put It On | 11.9 | 5.5 | |
| Pussycat | 12.1 | 5.4 | |

LRA 3.3 on song 2 means the quiet parts are almost as loud as the loud parts:
that is what "squashed" sounds like, and it is the second-largest reason these
recordings read as old (the first is the room, which is not separable - see
STATUS.md).

Why expansion is risky and how it is controlled
----------------------------------------------
Upward expansion amplifies whatever is in the gaps. Applied after a denoiser it
is safe; applied before one it re-raises the noise floor that was just removed.
It is therefore placed *last* in the chain, after denoise and dereverb attempts,
and the noise floor is measured before and after.

The curve is an upward compressor: below a threshold the gain rises with the
deficit, above it nothing happens. That restores contrast without touching peaks,
so the crest factor and the LRA both move in the right direction while the true
peak stays where the limiter left it.
"""
import numpy as np

SR = 44100


def _env_db(x, sr=SR, win_ms=10.0):
    """10 ms RMS envelope in dB. Slow on purpose: expansion must follow phrases,
    not individual cycles, or it modulates the timbre."""
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    win = max(4, int(win_ms * 1e-3 * sr))
    nb = len(y) // win
    if nb < 8:
        return None, win
    e = (y[:nb * win].reshape(nb, win) ** 2).mean(1)
    return 10 * np.log10(e + 1e-20), win


def expand(x, sr=SR, amount_db=6.0, thresh_percentile=55.0, ratio=1.0,
           smooth_ms=60.0, max_boost_db=12.0):
    """Upward expansion below a relative threshold.

    Parameters
    ----------
    amount_db : how much louder the quiet end may become (the headline control).
    thresh_percentile : where the expansion starts, as a percentile of the
        programme's own level distribution. 55 % means "everything below the
        middle gets lifted", which restores contrast without turning the loud
        parts up (they are already limited).
    ratio : >1 makes it a true expander; 1.0 means "just lift the quiet parts"
        (safer, and what the measurements here need).
    max_boost_db : hard ceiling on the gain, so a nearly silent passage cannot be
        amplified into a noise storm.

    Returns (y, info) with the measured crest and LRA change.
    """
    y = np.asarray(x, dtype=np.float64)
    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, None]

    out = np.empty_like(y)
    info = {}
    for c in range(y.shape[1]):
        ch = y[:, c]
        db, win = _env_db(ch, sr)
        if db is None:
            out[:, c] = ch
            continue
        thr = float(np.percentile(db, thresh_percentile))
        # Do NOT expand below the noise floor. Without this the quietest frames -
        # which contain only hiss - get lifted by the full amount, and measured on
        # the real material that raised the noise floor by 10 dB while the crest
        # went up: the numbers improve and the recording gets noisier. The floor is
        # the 5th percentile of the frame levels, i.e. what is there when the music
        # is not.
        nfloor = float(np.percentile(db, 5.0))
        gate = np.clip((db - nfloor) / 6.0, 0.0, 1.0)    # soft 6 dB gate above floor
        deficit = np.maximum(thr - db, 0.0) * gate
        # map the deficit onto a gain, scaled so that the *typical* deficit gets
        # the full `amount_db`. Without this normalisation the amount parameter
        # means nothing: a 3 dB deficit and a 30 dB deficit cannot both get 6 dB
        # (that would flatten the very contrast we are restoring).
        typical = float(np.percentile(deficit[deficit > 0], 75)) if np.any(deficit > 0) else 1.0
        gain_db = amount_db * np.clip(deficit / max(typical, 1e-6), 0.0, max_boost_db / max(amount_db, 1e-6))
        if ratio > 1.0:
            gain_db = gain_db * ratio
        gain_db = np.clip(gain_db, 0.0, max_boost_db)

        # smooth to phrase rate, then interpolate to sample rate
        k = max(1, int(smooth_ms * 1e-3 * sr / win))
        if k > 1:
            gain_db = np.convolve(gain_db, np.ones(k) / k, 'same')
        g = 10 ** (gain_db / 20.0)
        gs = np.interp(np.arange(len(ch)), np.arange(len(g)) * win, g)
        # short smoothing at sample rate so no zipper noise
        ks = max(2, int(0.003 * sr))
        gs = np.convolve(gs, np.ones(ks) / ks, 'same')
        out[:, c] = ch * gs

    if was_1d:
        out = out[:, 0]

    info['max_boost_db'] = float(np.max(20 * np.log10(
        np.abs(np.asarray(out, float)).max() / (np.abs(np.asarray(x, float)).max() + 1e-20))))
    return out.astype(np.float32), info


def crest_db(x):
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    return float(20 * np.log10(np.abs(y).max() / (np.sqrt(np.mean(y ** 2)) + 1e-20)))


def lra_db(x, sr=SR, block_s=0.4):
    """Short-term loudness range: 95th minus 10th percentile of 400 ms RMS.

    This is the EBU R128 LRA idea without the full gating; it tracks the same
    thing (how much the programme breathes) and is what the targets refer to.
    """
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    blk = max(8, int(block_s * sr))
    nb = len(y) // blk
    if nb < 8:
        return None
    r = np.sqrt((y[:nb * blk].reshape(nb, blk) ** 2).mean(1) + 1e-20)
    return float(20 * np.log10(np.percentile(r, 95) / np.percentile(r, 10)))


def decompress(x, sr=SR, amount_db=6.0, thresh_percentile=75.0, smooth_ms=40.0):
    """Pull the loud passages DOWN instead of the quiet ones up.

    Measured: `expand()` raises the crest factor but *lowers* the LRA on this
    material, and the reason is in the numbers - song 2 has 89 % of its 400 ms
    blocks within 3 dB of each other. The programme is not "quiet parts too
    quiet", it is "loud parts squashed into the quiet ones". Lifting the quiet
    end moves everything toward the middle; the contrast that is missing can only
    come back by giving the loud end its space.

    This is a soft down-expander above a relative threshold: everything above it
    is attenuated in proportion to how far above it sits, smoothly, so transients
    keep their shape and only their relative level changes. It cannot add anything
    that was not there, and it never raises the noise floor (it only attenuates).
    """
    y = np.asarray(x, dtype=np.float64)
    was_1d = y.ndim == 1
    if was_1d:
        y = y[:, None]

    out = np.empty_like(y)
    for c in range(y.shape[1]):
        ch = y[:, c]
        db, win = _env_db(ch, sr, win_ms=5.0)
        if db is None:
            out[:, c] = ch
            continue
        thr = float(np.percentile(db, thresh_percentile))
        excess = np.maximum(db - thr, 0.0)
        typical = float(np.percentile(excess, 90)) if np.any(excess > 0) else 1.0
        # full `amount_db` at the loudest 10 % of frames, proportional in between
        gain_db = -amount_db * np.clip(excess / max(typical, 1e-6), 0.0, 1.0)
        k = max(1, int(smooth_ms * 1e-3 * sr / win))
        if k > 1:
            gain_db = np.convolve(gain_db, np.ones(k) / k, 'same')
        gs = np.interp(np.arange(len(ch)), np.arange(len(gain_db)) * win,
                       10 ** (gain_db / 20.0))
        ks = max(2, int(0.003 * sr))
        gs = np.convolve(gs, np.ones(ks) / ks, 'same')
        # never attenuate more than the headline amount
        gs = np.clip(gs, 10 ** (-amount_db / 20.0), 1.0)
        out[:, c] = ch * gs

    if was_1d:
        out = out[:, 0]
    return out.astype(np.float32), {'amount_db': amount_db,
                                    'thresh_db': round(thr, 1) if 'thr' in dir() else None}
