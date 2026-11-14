#!/usr/bin/env python3
"""
Stereo image generation for mono sources - without the phase damage.

The problem with the obvious method
-----------------------------------
A mono source can be widened by delaying one channel ("Haas"), or by subtracting
two different delays of the same signal into the side channel. Both are comb
filters: subtracting two delays that differ by tau gives a response with nulls
every 1/tau Hz. The version that was in the chain before used 8.7 ms and 13.1 ms,
i.e. tau = 4.4 ms, so it had a null every 227 Hz and, measured over 1-16 kHz, a
**28.3 dB peak-to-trough ripple**. A comb of that depth is not "width", it is a
resonance: the top end sounds like it is coming through a tin can, and it is the
opposite of the natural, close-miked sound this project is aiming for. It was also
completely incoherent with the music (correlation of its 3-20 kHz side with the
mid: -0.006), which is why it read as artificial.

The method used here
--------------------
A cascade of **Schroeder allpasses** with mutually prime delays. An allpass has,
by definition, a perfectly flat magnitude response - so it *cannot* ring or comb -
while its phase is scattered by up to a few milliseconds. Applied to a band of the
music, it produces a signal with:

* the same long-term spectrum as the source (flat, no resonances),
* the same short-term envelope / the same transients (it is the same material,
  time-smeared by only a few ms),
* near-zero waveform correlation with the original (which is exactly what "wide"
  means - the two channels differ, the mono sum does not).

The side component is then added as +/- so it cancels *exactly* in mono. Nothing
here changes the mono sum, adds noise, or invents frequencies: the side is made of
the programme itself. That is the same trick a 1970s "duct tape" double-tracking
job achieves, minus the tape.
"""
import numpy as np
from scipy import signal as sg

SR = 44100

# mutually prime delays in ms and their allpass gains - chosen so the impulse
# response is dense but still short (a few tens of ms), i.e. it smears a hi-hat
# but never turns into reverb
_STAGES = ((1.13, 0.55), (2.09, 0.53), (3.71, 0.51), (6.23, 0.49), (10.31, 0.47))


def _allpass(x, delay, g):
    """Schroeder allpass: flat magnitude, scattered phase."""
    d = max(1, int(round(delay)))
    b = np.zeros(d + 1)
    b[0] = -g
    b[d] = 1.0
    a = np.zeros(d + 1)
    a[0] = 1.0
    a[d] = -g          # y[n] = -g*x[n] + x[n-D] + g*y[n-D]  =>  |H| == 1
    return sg.lfilter(b, a, x)


def decorrelate(x, sr=SR, hp_hz=250.0, stages=_STAGES, normalize=True):
    """Phase-scatter `x` with an allpass cascade (flat magnitude, same envelope)."""
    y = np.asarray(x, dtype=np.float64)
    if hp_hz and hp_hz > 0:
        b, a = sg.butter(2, hp_hz / (sr / 2), 'high')
        y = sg.filtfilt(b, a, y)
    for ms, g in stages:
        y = _allpass(y, ms * 1e-3 * sr, g)
    if normalize:
        r0 = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12
        r1 = np.sqrt(np.mean(y ** 2)) + 1e-12
        y *= r0 / r1
    return y


def build_side(mid_sources, sr=SR, target_corr=0.93, hp_hz=900.0, mix_db=0.0):
    """Build a side signal from the programme whose level hits a target correlation.

    For an uncorrelated side of RMS ratio k against the mid, the inter-channel
    correlation is (1-k^2)/(1+k^2). Inverting that gives the exact k needed for a
    requested correlation, so the width is *specified* rather than guessed.
    """
    m = sum(mid_sources) if isinstance(mid_sources, (list, tuple)) else mid_sources
    m = np.asarray(m, dtype=np.float64)
    side = np.zeros_like(m)
    for s in (mid_sources if isinstance(mid_sources, (list, tuple)) else [mid_sources]):
        side += decorrelate(np.asarray(s, dtype=np.float64), sr, hp_hz=hp_hz)
    rms = lambda v: np.sqrt(np.mean(v ** 2)) + 1e-12
    k = np.sqrt(max(1e-6, (1.0 - target_corr) / (1.0 + target_corr)))
    side *= (k * rms(m) * (10 ** (mix_db / 20.0))) / rms(side)
    return side.astype(np.float32)


def analyse(st, sr=SR):
    """Measurements that describe a stereo image in perceptual terms."""
    y = np.asarray(st, dtype=np.float64)
    l, r = y[:, 0], y[:, 1]
    m, s = (l + r) / 2, (l - r) / 2
    out = {'corr': round(float(np.corrcoef(l, r)[0, 1]), 4),
           'side_minus_mid_db': round(float(20 * np.log10((np.std(s) + 1e-12) / (np.std(m) + 1e-12))), 2)}
    # envelope correlation above 3 kHz: does the width *follow the music*?
    bb, aa = sg.butter(4, [3000 / (sr / 2), min(20000, sr / 2 - 100) / (sr / 2)], 'band')
    sh, mh = sg.filtfilt(bb, aa, s), sg.filtfilt(bb, aa, m)
    env = lambda v: np.abs(sg.hilbert(v[:int(min(len(v), 60 * sr))]))
    out['hf_env_corr'] = round(float(np.corrcoef(env(sh), env(mh))[0, 1]), 4)
    out['hf_wave_corr'] = round(float(np.corrcoef(sh, mh)[0, 1]), 4)
    # comb detection: the side spectrum must look like the mid spectrum
    f, Ps = sg.welch(sh, sr, nperseg=8192)
    f, Pm = sg.welch(mh, sr, nperseg=8192)
    i = (f > 1000) & (f < 16000)
    d = 10 * np.log10(Ps[i] / (Pm[i] + 1e-30) + 1e-20)
    out['side_spec_ripple_db'] = round(float(np.percentile(d, 95) - np.percentile(d, 5)), 1)
    return out
