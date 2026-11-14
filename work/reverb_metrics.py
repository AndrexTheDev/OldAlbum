#!/usr/bin/env python3
"""
Reverberation metrics - built after four failed estimators, and honest about it.

Why this file exists separately
------------------------------
Measuring "how much room is in this recording" turned out to be the hardest part
of the whole project, harder than any processing stage. Four estimators were
built and discarded before one behaved:

1. **Line fit to envelope maxima** - 24 s for a DRY signal. A decaying note
   re-triggers the onset detector, so every fit starts mid-decay.
2. **Schroeder integration over the whole track** - 8-9 s regardless of the true
   reverb time. Verified sound against the impulse response, where the identical
   code measures 1.00 s correctly, so the maths was fine and the *input* was
   wrong: Schroeder needs the signal to decay into the noise floor, and music
   stops instead.
3. **Per-onset Schroeder in 800 ms windows** - 1.12 s for a dry signal, None for
   a reverberant one. Too many thresholds, each rejecting the other's output.
4. **Direct-to-diffuse ratio / tail-to-onset ratio** - gave sensible numbers on
   synthetic rooms (dry -12.3 dB, 1.5 s room -0.7 dB) but *saturated at 0 dB on
   real music*, because in continuous music the running peak never falls: there
   is no "after the onset", the next note is already there. Measuring a tail
   needs a gap, and pop music has almost none.

What is used now
----------------
**Decay slope in dB/s, measured only where the envelope genuinely falls.** This
is monotonic on synthetic rooms (dry -296 dB/s, 0.3 s -143, 1.0 s -52) and, unlike
the ratio metrics, it works on material with no gaps. It is reported as an
equivalent RT60 (`-60 / slope`) purely for readability; the raw slope is the
number to compare.

Validation on the four delivered songs is in RESTORATION.md.
"""
import numpy as np

SR = 44100


def envelope_db(x, sr=SR, win_ms=2.0):
    """2 ms RMS envelope in dB, mono-summed. 2 ms is short enough to resolve a
    drum attack (a 46 ms STFT window is not, which is why the STFT-domain metric
    in earlier attempts could not see the thing it was measuring)."""
    y = np.asarray(x, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(1)
    win = max(2, int(win_ms * 1e-3 * sr))
    nb = len(y) // win
    if nb < 10:
        return None, win
    e = (y[:nb * win].reshape(nb, win) ** 2).mean(1)
    return 10 * np.log10(e + 1e-20), win


def decay_slope_db_per_s(x, sr=SR, lo_ms=4.0, hi_ms=60.0, min_segments=5):
    """Median slope of genuinely falling envelope segments, in dB per second.

    Only segments that start near a local peak are considered, so sustained notes
    and the body of a chord do not contribute. A dry close-miked attack falls at
    several hundred dB/s; a room lets it fall at tens.
    """
    db, win = envelope_db(x, sr)
    if db is None:
        return None
    fs = sr / win
    i0 = max(1, int(lo_ms * 1e-3 * fs))
    i1 = max(i0 + 3, int(hi_ms * 1e-3 * fs))
    slopes = []
    i = 0
    n = len(db)
    while i < n - i1 - 1:
        seg = db[i:i + i1]
        # a candidate starts at a local peak and is clearly below it later
        if seg[0] >= seg[1:4].max() and seg[:3].max() >= np.percentile(db, 70):
            if seg[-1] < seg[0] - 4.0:
                t = np.arange(i0, i1) / fs
                sl = np.polyfit(t, seg[i0:i1], 1)[0]
                if sl < -5.0:
                    slopes.append(sl)
                i += i1 // 2
                continue
        i += 1
    if len(slopes) < min_segments:
        return None
    return float(np.median(slopes))


def equivalent_rt60(x, sr=SR):
    """-60 dB / decay slope, in seconds. Readability only - compare the slope."""
    sl = decay_slope_db_per_s(x, sr)
    if sl is None or sl >= 0:
        return None
    return float(-60.0 / sl)


def report(x, sr=SR):
    sl = decay_slope_db_per_s(x, sr)
    return {
        'decay_db_per_s': round(sl, 1) if sl is not None else None,
        'equivalent_rt60_s': round(-60.0 / sl, 2) if sl else None,
    }
