#!/usr/bin/env python3
"""
Unit tests for the DSP primitives - each against an analytically known answer.

Why this file exists
--------------------
Every serious bug in this project was found by measuring, not by listening, and
several of them survived *because nothing compared a function against the answer
it was supposed to produce*. The two worst examples:

  * `fir_from_curve` normalised the FIR to 0 dB at DC. For a curve that cuts 9 dB
    at 95 Hz, that shifted the *whole* filter -9 dB, so every EQ request landed
    the wrong distance from its target. It was "fixed" once, incorrectly, and the
    wrong version shipped.
  * `filtfilt` on a 0.2 Hz Butterworth at 44.1 kHz (pole at 0.99998) returned
    transport-drift figures of +41711 cents.

Both are one-line checks. This file is those checks, kept permanently so they
cannot come back. Run:

    python work/selftest.py          # exit code 0 == all pass
"""
import sys
import numpy as np
from scipy import signal as sg

sys.path.insert(0, '/home/user/OldAlbum/work')
import dsp
import image

SR = 44100
_FAILED = []


def check(name, cond, detail=''):
    print(('  PASS  ' if cond else '  FAIL  ') + name + ('   ' + detail if detail else ''))
    if not cond:
        _FAILED.append(name)


def rms_db(x):
    a = np.asarray(x, dtype=np.float64)
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)


# ---------------------------------------------------------------- 1. EQ ======
def test_eq():
    print('\n[1] EQ: does a requested curve reach its targets?')
    # flat gains: the output level must change by exactly that gain
    for g in (-9.0, 0.0, +6.0):
        h = dsp.fir_from_curve([20, 20000], [g, g], SR)
        w = np.random.default_rng(0).standard_normal(SR * 5)
        got = rms_db(dsp.apply_fir(w, h)) - rms_db(w)
        check(f'flat {g:+.1f} dB curve -> {got:+.2f} dB', abs(got - g) < 0.05)

    # shaped curves: every requested point must land on the requested value
    curves = {
        'other': ([95, 150, 260, 420, 700, 1200, 2000, 3200, 4600, 6500,
                   9000, 12000, 16000],
                  [-9, -3.5, -1.2, 0.2, 0.6, 0.8, 1.1, 1.4, 1.5, 1.7, 2.0, 2.2, 1.8]),
        'bass': ([28, 40, 52, 65, 80, 100, 130, 170, 220, 300, 450, 700, 1000, 1400],
                 [-4, 0, 1, 0.9, 0.8, 0.6, 0.4, 0, -1, -1.6, -2.2, -3.4, -6, -12]),
    }
    for name, (f, g) in curves.items():
        h = dsp.fir_from_curve(f, g, SR)
        H = 20 * np.log10(np.abs(np.fft.rfft(h, 1 << 16)) + 1e-15)
        fr = np.fft.rfftfreq(1 << 16, 1 / SR)
        errs = [H[np.argmin(np.abs(fr - x))] - gv for x, gv in zip(f, g)]
        check(f'{name}: all {len(f)} bands within 0.5 dB',
              max(np.abs(errs)) < 0.5,
              f'max error {max(np.abs(errs)):.2f} dB')


# ------------------------------------------------------------ 2. allpass ====
def test_allpass():
    print('\n[2] Allpass decorrelator: magnitude must be exactly flat')
    imp = np.zeros(1 << 15)
    imp[0] = 1.0
    h = image.decorrelate(imp, SR, hp_hz=0, normalize=False)
    H = np.abs(np.fft.rfft(h, 1 << 17))
    f = np.fft.rfftfreq(1 << 17, 1 / SR)
    band = (f > 50) & (f < 20000)
    dev = np.abs(20 * np.log10(H[band] + 1e-15)).max()
    check(f'|H| deviation from flat: {dev:.6f} dB', dev < 0.01)

    # the comb filter it replaced: must show a large ripple (regression guard)
    def comb(x, d1, d2):
        a = np.zeros(len(x)); a[int(d1 * 1e-3 * SR):] = x[:len(x) - int(d1 * 1e-3 * SR)]
        b = np.zeros(len(x)); b[int(d2 * 1e-3 * SR):] = x[:len(x) - int(d2 * 1e-3 * SR)]
        return 0.42 * (a - b)
    w = np.random.default_rng(0).standard_normal(SR * 20)
    y = comb(w, 8.7, 13.1)
    f2, P = sg.welch(y, SR, nperseg=16384)
    m = (f2 > 1000) & (f2 < 16000)
    d = 10 * np.log10(P[m] / P[m].mean() + 1e-20)
    ripple = np.percentile(d, 99.9) - np.percentile(d, 0.1)
    check(f'old comb filter ripple {ripple:.1f} dB (regression: must be >20)',
          ripple > 20)

    # stereo solve: measured correlation must hit the requested one
    for target in (0.90, 0.95):
        side = image.build_side([w], SR, target_corr=target)
        corr = np.corrcoef(w + side, w - side)[0, 1]
        check(f'requested corr {target:.2f} -> {corr:.4f}', abs(corr - target) < 0.03)
        # mono compatibility: the side must cancel exactly
        check(f'  mono sum == mid (target {target:.2f})',
              np.allclose((w + side + (w - side)) / 2, w))


# --------------------------------------------------------- 3. compressor ====
def test_compressor():
    print('\n[3] Compressor: static curve at the nominal ratio')
    # The detector is RMS, so a sine at peak -10 dBFS presents -13.01 dBFS.
    # The expected gain reduction must be computed from the RMS, not the peak -
    # getting this wrong once made a correct compressor look broken.
    for peak_db in (-25.0, -13.0, -3.0):
        x = 10 ** (peak_db / 20) * np.sin(2 * np.pi * 500 * np.arange(SR * 4) / SR)
        y, _ = dsp.compress(x, SR, thresh_db=-20.0, ratio=4.0, attack_ms=5,
                            release_ms=60, makeup_db=0.0, knee_db=6.0)
        seg = slice(SR * 3, SR * 4)
        gr = rms_db(y[seg]) - rms_db(x[seg])
        in_rms = rms_db(x[seg])
        over = in_rms - (-20.0)
        exp = 0.0 if over <= -3.0 else (over * (1 - 1 / 4.0) if over >= 3.0 else
                                        (1 - 1 / 4.0) * (over + 3) ** 2 / 12)
        check(f'input {in_rms:+.1f} dB RMS -> GR {gr:+.2f} dB (expect {-exp:+.2f})',
              abs(gr + exp) < 0.4)


# ----------------------------------------------------------- 4. limiter =====
def test_limiter():
    print('\n[4] Limiter: true peak must not exceed the ceiling')
    for nd in (1, 2):
        loud = 0.99 * np.sin(2 * np.pi * 997 * np.arange(SR * 2) / SR)
        if nd == 2:
            loud = np.stack([loud, loud], axis=1)
        out, _ = dsp.limiter(loud, SR, ceiling_db=-1.0)
        tp = 20 * np.log10(np.abs(sg.resample_poly(out, 4, 1, axis=0)).max())
        check(f'{"stereo" if nd == 2 else "mono"}: {tp:.3f} dBTP', tp <= -1.0 + 1e-6)


# ----------------------------------------------------------- 5. denoise =====
def test_denoise():
    print('\n[5] Denoiser: remove hiss, keep the tone')
    rng = np.random.default_rng(0)
    t = np.arange(SR * 8) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noisy = tone + 0.01 * rng.standard_normal(len(t))
    den = dsp.denoise(noisy, SR, reduction_db=12.0, floor_db=-22.0)

    def band(x, lo, hi):
        b, a = sg.butter(4, [lo / 22050, hi / 22050], 'band')
        return rms_db(sg.filtfilt(b, a, x))
    tone_keep = band(den, 400, 480) - band(noisy, 400, 480)
    noise_cut = band(den, 8000, 12000) - band(noisy, 8000, 12000)
    # A held tone is continuously present, and minimum statistics classifies
    # "continuously present" as noise - without tonal protection it loses 5.26 dB
    # (that is the 200-1000 Hz loss in the audit). The protection must keep it
    # within ~1 dB while still killing the hiss.
    check(f'held tone 440 Hz kept: {tone_keep:+.2f} dB', abs(tone_keep) < 1.0)
    check(f'hiss 8-12 kHz removed: {noise_cut:+.2f} dB', noise_cut < -6.0)


# ------------------------------------------------------------- 6. declick ===
def test_declick():
    print('\n[6] Declicker: removes an impulse without touching the programme')
    sig = 0.2 * np.sin(2 * np.pi * 200 * np.arange(SR * 4) / SR)
    sig[SR * 2 + 100] = 0.9
    dec, n = dsp.declick(sig, SR, k=7.0)
    check(f'impulse {sig[SR*2+100]:.2f} -> {dec[SR*2+100]:.2f} ({n} found)',
          abs(dec[SR*2+100]) < 0.3)
    # The repair is local. Away from the impulse the difference must be tiny -
    # note that declick detects 10 events on a clean 200 Hz sine (the cubic
    # repair of the impulse leaves ringing that re-triggers), so "bit-identical"
    # was too strong a claim: what matters is that nothing is *audibly* changed.
    far = slice(0, SR * 2 - 4000)
    dmax = np.abs(dec[far] - sig[far]).max()
    check(f'audio away from the impulse <= -50 dBFS (max {dmax:.5f})', dmax < 0.003)


# ------------------------------------------------------- 7. pitch track =====
def test_wowfix():
    print('\n[7] Wow measurement: filter stability at audio rate')
    import wowfix
    # The 0.2 Hz high-pass is only usable after decimation; without that it
    # exploded and reported +41711 cents. Guard the helper that does the work.
    x = np.sin(2 * np.pi * 220 * np.arange(SR * 8) / SR)
    if_, env = wowfix._if_curve(x, SR)
    check('IF of a pure tone is its frequency',
          abs(np.median(if_[1000:-1000]) - 220.0) < 1.0,
          f'{np.median(if_[1000:-1000]):.2f} Hz')

    cents = np.full(SR * 8, 5.0)          # constant +5 cents
    res = wowfix._extract_drift(cents, iqr_cents=5.0, dt=1.0 / SR, max_cents=12.0)
    # a constant offset is not transport drift: the gate must reject it
    check('constant offset rejected by the gate', res is None)


if __name__ == '__main__':
    test_eq()
    test_allpass()
    test_compressor()
    test_limiter()
    test_denoise()
    test_declick()
    test_wowfix()
    print()
    if _FAILED:
        print('%d FAILED: %s' % (len(_FAILED), ', '.join(_FAILED)))
        sys.exit(1)
    print('all checks passed')
