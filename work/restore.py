#!/usr/bin/env python3
"""
Stage 3: restore + remix + master one song from its separated stems.

Design rule: this is *restoration*, not remixing. The 1960s balance that the
producer chose is preserved; we only correct what is objectively broken
(noise, hum, mud, missing top end, level) and give stems the treatment they
individually need. Stem gains are therefore clamped to a few dB.

Per stem : subsonic/DC -> dehum -> declick -> decrackle -> denoise
           -> corrective EQ -> harmonic excitation -> dynamics
Bus      : balance -> glue -> tone -> excitement -> imaging
Master   : true-peak limiter -> loudness normalise
"""
import os, sys, json, time
import numpy as np
import soundfile as sf
from scipy import signal as sg
sys.path.insert(0, '/home/user/OldAlbum/work')
import dsp
import image
import bwe
import bwe2
import torch
from dsp import (db, rms_db, peak_db, declick, decrackle, denoise, remove_hum,
                 block_dc_subsonic, eq_curve, compress, limiter, harmonic_excite,
                 haas_width, loudness, true_peak_db)

SR = 44100
ROOT = '/home/user/OldAlbum'
STEMS = ('vocals', 'drums', 'bass', 'other')

# ---------------------------------------------------------------- per-song recipes
# 'balance_db' = intended level change of each stem relative to its separated
#                 level (CLAMPED to +-5 dB so the original blend survives)
RECIPES = {
    # 'tilt'      : overall voicing strength (matches the modern 200-800 Hz
    #               referenced curve; see RESTORATION.md)
    # 'balance_db': level of each stem, clamped to +-5 dB
    # 'dn_bus'    : suppression of the single denoiser that runs on the bus
    # 'par'       : share of the parallel compressed copy on the bus
    '1_I_Need_You': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 5.0, 'bass': -3.0, 'other': 0.5},
        tilt=1.0, max_boost=6.0, max_cut=8.0,
        dn_bus=11.0, par=0.16,
        air_from_hz=5700.0, air_noise_cut_db=-12.0, air_master=0.3,
        target_lufs=-12.0, ceiling=-1.0,
    ),
    '2_Lonesome_Feeling': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 3.0, 'bass': -3.0, 'other': 0.5},
        tilt=1.0, max_boost=6.0, max_cut=8.0,
        dn_bus=11.0, par=0.16,
        air_from_hz=5700.0, air_noise_cut_db=-12.0, air_master=0.3,
        target_lufs=-12.0, ceiling=-1.0,
    ),
    '3_Put_It_On': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 2.5, 'bass': -3.0, 'other': 0.5},
        tilt=1.0, max_boost=6.0, max_cut=8.0,
        dn_bus=11.0, par=0.16,
        air_from_hz=5700.0, air_noise_cut_db=-12.0, air_master=0.3,
        target_lufs=-12.0, ceiling=-1.0,
    ),
    '4_Whats_New_Pussycat': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 2.0, 'bass': -3.0, 'other': 1.0},
        tilt=1.0, max_boost=6.0, max_cut=8.0,
        dn_bus=10.0, par=0.16,
        air_from_hz=5700.0, air_noise_cut_db=0.0, air_master=0.3,
        target_lufs=-12.0, ceiling=-1.0,
    ),
}

# stem-specific minimum-phase corrections, in Hz: each stem gets only the
# filtering it truly needs. Deliberately short: every extra filter is another
# chance to smear a transient or to lift a noise floor.
STEM_CHAIN = {
    #             high-pass  low-pass  extra EQ (freqs, gains)
    'vocals': dict(hp=100.0, lp=17500.0,
                   eq=([260, 420, 700, 1100, 2000, 3000, 4200, 5600, 7200, 9500, 12000, 16000],
                       [-2.0, -1.2, 0.4, 0.6, 1.3, 2.0, 1.9, 1.4, 1.2, 1.4, 1.2, 0.9])),
    'drums': dict(hp=35.0, lp=17500.0,
                  eq=([40, 60, 85, 120, 180, 300, 600, 1200, 2500, 4000, 6000, 9000, 13000, 16000],
                      [1.0, 2.2, 1.4, -0.6, -1.2, -0.6, 0.4, 0.9, 1.4, 1.5, 1.7, 1.9, 2.0, 1.6])),
    'bass': dict(hp=30.0, lp=1400.0,
                 eq=([28, 40, 52, 65, 80, 100, 130, 170, 220, 300, 450, 700, 1000, 1400],
                     [-4.0, 0.0, 1.0, 0.9, 0.8, 0.6, 0.4, 0.0, -1.0, -1.6, -2.2, -3.4, -6.0, -12.0])),
    'other': dict(hp=95.0, lp=17500.0,
                  eq=([150, 260, 420, 700, 1200, 2000, 3200, 4600, 6500, 9000, 12000, 16000],
                      [-1.0, -0.6, 0.2, 0.6, 0.8, 1.1, 1.4, 1.5, 1.7, 2.0, 2.2, 1.8])),
}


# ---------------------------------------------------------------- small helpers
def load_mono(path):
    x, sr = sf.read(path, always_2d=False, dtype='float32')
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def norm_db(x, target_db):
    return (x * (10 ** ((target_db - rms_db(x)) / 20.0))).astype(np.float32)


def env_follow(x, sr, att_ms, rel_ms):
    a_att = np.exp(-1.0 / (att_ms * 1e-3 * sr))
    a_rel = np.exp(-1.0 / (rel_ms * 1e-3 * sr))
    e = np.abs(x)
    out = np.empty_like(e)
    v = 0.0
    for i in range(len(e)):
        c = a_att if e[i] > v else a_rel
        v = c * v + (1 - c) * e[i]
        out[i] = v
    return out


def hf_control(x, sr=SR, freq=7000.0, thresh_db=-30.0, ratio=2.5):
    """Static high-shelf 'air limiter': keeps restored top end from turning into
    a constant fizz by compressing only the >freq band."""
    b, a = sg.butter(4, freq / (sr / 2), 'high')
    hi = sg.filtfilt(b, a, x, axis=0).astype(np.float32)
    lo = (x - hi).astype(np.float32)
    e = env_follow(hi, sr, 8.0, 120.0)
    lvl = 20 * np.log10(np.maximum(e, 1e-9))
    gr = np.clip((lvl - thresh_db) * (1 - 1 / ratio), 0, 8)
    return (lo + hi * (10 ** (-gr / 20.0))).astype(np.float32)


def deess(x, sr=SR, freq=5700.0, thresh_db=-27.0, ratio=3.5):
    b, a = sg.butter(4, freq / (sr / 2), 'high')
    hi = sg.filtfilt(b, a, x, axis=0).astype(np.float32)
    lo = (x - hi).astype(np.float32)
    e = env_follow(hi, sr, 4.0, 70.0)
    lvl = 20 * np.log10(np.maximum(e, 1e-9))
    gr = np.clip((lvl - thresh_db) * (1 - 1 / ratio), 0, 10)
    return (lo + hi * (10 ** (-gr / 20.0))).astype(np.float32)


def transient_shape(x, sr=SR, attack_db=2.0, sustain_db=-1.0):
    """Attack emphasis via envelope slope (sample-aligned, no phase issues)."""
    from scipy import ndimage
    env = ndimage.uniform_filter1d(np.abs(x), max(3, int(0.004 * sr)))
    d = np.gradient(env)
    d = ndimage.uniform_filter1d(d, max(3, int(0.006 * sr)))
    n = d / (np.percentile(np.abs(d), 99.5) + 1e-9)
    n = np.clip(n, -1, 1)
    g = 10 ** ((attack_db * np.clip(n, 0, 1) + sustain_db * np.clip(-n, 0, 1)) / 20.0)
    return (x * g).astype(np.float32)


def soft_comp(x, sr=SR, thresh_offset=-10.0, ratio=2.0, attack_ms=12.0,
              release_ms=180.0, makeup='auto'):
    """
    Compressor whose threshold sits `thresh_offset` dB below the stem's own RMS,
    with automatic makeup gain so the stem keeps its original level (dynamics
    only, no loudness change - that is the master's job).
    """
    thr = rms_db(x) + thresh_offset
    r_in = rms_db(x)
    y, gr = compress(x, sr, thresh_db=thr, ratio=ratio, attack_ms=attack_ms,
                     release_ms=release_ms, makeup_db=0.0, knee_db=8)
    if makeup == 'auto':
        y = (y * (10 ** ((r_in - rms_db(y)) / 20.0))).astype(np.float32)
    return y, gr


def parallel_comp(x, sr=SR, thresh_offset=-10.0, ratio=2.0, attack_ms=18.0,
                  release_ms=200.0, mix=0.25):
    """Parallel ('New York') compression: dry + a compressed copy, level-matched.

    Unlike `soft_comp` this never applies make-up gain to the *whole* signal, so
    quiet passages keep their level (and their noise floor) while the loud parts
    get support from the compressed copy. `mix` is the share of that copy.
    """
    thr = rms_db(x) + thresh_offset
    par, gr = compress(x, sr, thresh_db=thr, ratio=ratio, attack_ms=attack_ms,
                       release_ms=release_ms, makeup_db=0.0, knee_db=10)
    par = norm_db(par, rms_db(x) - 6.0)
    return (x * (1.0 - mix) + par * mix).astype(np.float32), gr


def band_edges(x, sr=SR):
    f, P = sg.welch(x, sr, nperseg=8192)
    Pdb = 10 * np.log10(np.maximum(P, 1e-20))
    out = {}
    for a, b in ((20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
                 (1000, 2000), (2000, 4000), (4000, 8000), (8000, 12000),
                 (12000, 16000), (16000, 20000)):
        m = (f >= a) & (f < b)
        out[f'{a}-{b}'] = round(float(10 * np.log10(max(float(P[m].sum()), 1e-30))), 2) if m.any() else None
    return out



# ---------------------------------------------------------------- diagnostics
import os as _os
_DEBUG = _os.environ.get('RESTORE_DEBUG') == '1'
_KLOG = []


def k5(x, label=''):
    """Kurtosis of the difference above 5 kHz - sensitive to spike/edge artifacts."""
    if not _DEBUG:
        return
    from scipy import signal as _s
    a = np.asarray(x, dtype=np.float64)
    if a.ndim > 1:
        a = a.mean(axis=1)
    b1, a1 = _s.butter(6, 5000.0 / (SR / 2), 'high')
    hi = _s.filtfilt(b1, a1, a)
    d = np.diff(hi); d -= d.mean(); v = d.var()
    k = float((d ** 4).mean() / v ** 2) if v > 0 else 0.0
    _KLOG.append((label, round(k, 2)))
    # 6-16 kHz level in the quietest 10% of 100 ms frames, normalised to -12 LUFS
    st_ = np.stack([a, a], 1)
    g = 10 ** ((-12.0 - loudness(st_)) / 20.0)
    m = a * g
    nb = int(0.1 * SR)
    N = len(m) // nb
    fr = m[:N * nb].reshape(N, nb)
    rmsv = np.sqrt((fr ** 2).mean(axis=1))
    idx = np.argsort(rmsv)[:max(1, int(0.10 * N))]
    f_, P_ = _s.welch(fr[idx].ravel(), SR, nperseg=8192)
    bm = (f_ >= 6000) & (f_ < 16000)
    n6 = round(10 * np.log10(max(float(P_[bm].sum()), 1e-30)), 1)
    f2_, P2_ = _s.welch(m, SR, nperseg=8192)
    hi_ = (f2_ >= 15000) & (f2_ < 20000)
    ref_ = (f2_ >= 200) & (f2_ < 800)
    h15 = round(10 * np.log10(max(float(P2_[hi_].sum()), 1e-30) / max(float(P2_[ref_].sum()), 1e-30)), 1)
    print('  k5 %-22s %8.2f   n6 %6.1f   h15 %6.1f' % (label, k, n6, h15), flush=True)


_STAGE_DIR = _os.environ.get('RESTORE_STAGE_DIR')


def stage(x, label):
    """Write an intermediate stage to disk if RESTORE_STAGE_DIR is set."""
    if not _STAGE_DIR:
        return
    import soundfile as _sf
    _os.makedirs(_STAGE_DIR, exist_ok=True)
    a = np.asarray(x, dtype=np.float32)
    _sf.write(_os.path.join(_STAGE_DIR, label + '.wav'), a, SR, subtype='FLOAT')


def wb_declick(x, sr=SR, freq=6000.0, amount_db=3.0, thresh_db=9.0):
    """Wideband-aware declicker for the *residual* edge in the top octave.

    The neural reconstruction is time-warped: a harmonic at 9 kHz must travel
    faster than the transient that produced it, so a hard drum hit leaves a
    short wideband discontinuity in the synthesised band. This is a *targeted*
    repair - it only ever touches the reconstructed band above `freq` (never the
    real signal below it), and only where the wideband energy is 6+ dB above the
    narrowband expectation of the same moment.
    """
    from scipy import signal as sg
    y = np.asarray(x, dtype=np.float32)
    hi = sg.filtfilt(*sg.butter(4, freq / (sr / 2), 'high'), y)
    lo = y - hi
    # narrowband (band-limited) expectation of the HF: median over ~1.5 ms
    from scipy.ndimage import median_filter, uniform_filter1d
    env_w = uniform_filter1d(np.abs(hi), max(3, int(0.004 * sr)))
    env_n = median_filter(np.abs(hi), size=max(3, int(0.003 * sr)))
    excess = 20 * np.log10((env_w + 1e-9) / (env_n + 1e-9))
    # only genuine outliers are touched, and only by `amount_db`: anything
    # stronger would dull the cymbal transients instead of cleaning them
    over = np.clip((excess - thresh_db) / 12.0, 0.0, 1.0)
    g = 1.0 - over * (1.0 - 10 ** (-amount_db / 20.0))
    g = uniform_filter1d(g, max(3, int(0.001 * sr)))
    return (lo + hi * g).astype(np.float32)


# ---------------------------------------------------------------- neural air (BWE)
_BWE_MODELS = {}
_BWE_M, _BWE_E = bwe.band_matrix()


def neural_air(x, nm, sr=SR, amount=1.0, hp_hz=6300.0):
    """Reconstruct the top octave of one stem with the trained per-instrument
    network (work/bwe.py) and mix it in at `amount`.

    Unlike an exciter this is not a fixed DSP law: the network was trained on
    this album's own 4-8 kHz -> 8-16 kHz relationship, so it produces the upper
    harmonics that these particular instruments and microphones had. Verified:
    the synthesised band correlates 0.93 (drums) / 0.88 (vocals) with the *real*
    8-16 kHz content of the source, and the >5 kHz kurtosis drops instead of
    rising (20.5 -> 19.8 on the drum stem), i.e. it adds music, not spikes.
    """
    if nm not in _BWE_MODELS:
        _BWE_MODELS[nm] = bwe.load_instrument(nm)
    net, norm = _BWE_MODELS[nm]
    y = bwe.reconstruct_long(np.asarray(x, np.float64), net, norm, _BWE_M, _BWE_E)
    b, a = sg.butter(6, hp_hz / (sr / 2), 'high')
    y = sg.filtfilt(b, a, y).astype(np.float32)
    return (np.asarray(x, dtype=np.float32) + y * amount).astype(np.float32)


def neural_air_stereo(st, nm, amount=1.0, hp_hz=6800.0, sr=SR):
    """Neural top-octave reconstruction for a stereo bus.

    Runs on the mid channel only (all four sources are dual mono, so the side
    channel carries nothing above 6 kHz that is not already in the mid) and adds
    the result to both channels - i.e. the reconstruction stays mono-compatible.
    """
    if nm not in _BWE_MODELS:
        _BWE_MODELS[nm] = bwe.load_instrument(nm)
    net, norm = _BWE_MODELS[nm]
    y = np.asarray(st, dtype=np.float32)
    mid = y.mean(axis=1)
    r = bwe.reconstruct_long(mid.astype(np.float64), net, norm, _BWE_M, _BWE_E)
    b, a = sg.butter(6, hp_hz / (sr / 2), 'high')
    r = sg.filtfilt(b, a, r).astype(np.float32) * amount
    return (y + r[:, None]).astype(np.float32)



# ------------------------------------------------------- bwe2 (validated BWE)
_BWE2 = {}
_TARGET_8_12_DB = -26.0     # wanted level of 12-16 kHz vs. the 300-3 kHz reference


def _band_level_db(x, lo, hi, sr=SR):
    """Level of one band relative to the 300 Hz - 3 kHz reference, median frame."""
    m = np.asarray(x, dtype=np.float64)
    m = m.mean(1) if m.ndim > 1 else m
    n = 8192
    k = len(m) // n * n
    if k < n:
        return None
    fr = m[:k].reshape(-1, n)
    W = np.abs(np.fft.rfft(fr * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    ref = np.median(W[:, (f > 300) & (f < 3000)].sum(1))
    return float(10 * np.log10(np.median(W[:, (f >= lo) & (f < hi)].sum(1)) / ref + 1e-30))


def air_bwe2(st, sr=SR, strength_db=None, target_db=_TARGET_8_12_DB,
             lo=12.0, hi=16.0):
    """Reconstruct the top two octaves with the recurrent adversarial network.

    Replaces the per-instrument band-matrix model (`bwe.neural_air_stereo`), which
    scored R2 -0.19..+0.42 across songs and learned the average band level rather
    than its structure. `bwe2` scores +0.41..+0.61 on the same split - see BWE.md.

    The strength is chosen per song by measurement, not by ear: the network's lift
    maps 1:1 onto both octaves (BWE.md, table), so the setting is simply the
    distance from where the band currently sits to `target_db`. That keeps the
    four masters at a consistent brightness instead of applying one number that
    happens to suit song 1 and over-brightens song 4, whose top octave is 13 dB
    higher to begin with.

    Applied to the mid channel only and added to both, so the result stays
    mono-compatible by construction.
    """
    if not _BWE2:
        ck = torch.load(f'{ROOT}/work/models/bwe2.pt', weights_only=False)
        net = bwe2.BWE()
        net.load_state_dict(ck['state'])
        net.eval()
        _BWE2['net'] = net
        _BWE2['stats'] = ck['stats']
    y = np.asarray(st, dtype=np.float32)
    mid = y.mean(axis=1)
    cur = _band_level_db(mid, int(lo * 1000), int(hi * 1000), sr)
    if strength_db is None:
        # Measured calibration (BWE.md): the network's lift is exactly 1 dB per dB
        # of `strength_db` on all four sources, so the setting is simply the gap
        # to the target - no iteration needed. The 22 dB cap is deliberate: it
        # keeps the quietest top octave (song 1, -51.7 dB) from being boosted far
        # enough to expose codec artefacts that sit just above its music.
        if cur is None:
            strength_db = 12.0
        else:
            strength_db = float(np.clip(target_db - cur, 8.0, 22.0))
    # ---- closed loop, because the open-loop calibration is not enough ---------
    # The 1 dB-per-dB law was measured on the *raw* signal, but this stage sits
    # after separation, the separation correction, the voicing curve and the
    # denoiser, and those shift the band reference. Measured open-loop, the four
    # finished masters came out at -20.0 / -21.6 / -28.0 / -31.3 dB against a
    # -26 dB target: right on average, 11 dB apart. So: apply, measure the actual
    # 12-16 kHz level of the staged result, and correct once. One correction is
    # enough because the law is exactly 1:1, and the loop stops when the residual
    # is under 0.5 dB.
    add = None
    hist = []
    for _ in range(6):
        r = bwe2.extend(mid.astype(np.float32), _BWE2['net'], _BWE2['stats'],
                        strength_db=strength_db)
        add = (np.asarray(r, dtype=np.float32) - mid.astype(np.float32))
        got = _band_level_db((mid + add), int(lo * 1000), int(hi * 1000), sr)
        hist.append(round(got, 1) if got is not None else None)
        if got is None:
            break
        resid = target_db - got
        if abs(resid) < 0.5:
            break
        strength_db = float(np.clip(strength_db + resid, 4.0, 38.0))
    return (y + add[:, None]).astype(np.float32), {
        'strength_db': round(strength_db, 2),
        'before_db': None if cur is None else round(cur, 1),
        'after_db': hist[-1], 'iterations': len(hist)}


# ---------------------------------------------------------------- main
def restore_song(tag, out_dir, rec=None, verbose=True):
    rec = rec or RECIPES[tag]
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    info = {'tag': tag}

    raw = {n: load_mono(f'{ROOT}/work/stems/{tag}/{n}.wav')[0] for n in STEMS}
    n = len(raw['vocals'])
    src, _ = load_mono(f'{ROOT}/work/prep/{tag}.wav')
    src = src[:n] if len(src) >= n else np.pad(src, (0, n - len(src)))
    # `src` is the original mix and stays untouched (it is also the A/B reference);
    # `src_mono` is its mono sum, used further down to measure what the separation
    # did to the octave balance.
    src_mono = src.mean(axis=1) if src.ndim == 2 else src
    info['duration_s'] = round(n / SR, 3)
    info['source'] = {'lufs': round(loudness(src), 2), 'rms_db': round(rms_db(src), 2),
                      'bands': band_edges(src)}
    info['stems_in_rms_db'] = {k: round(rms_db(v), 2) for k, v in raw.items()}

    # =========================================================== PER STEM
    # The chain here is deliberately *short*. Each stage below is in because it
    # was measured to help; nothing is in because it is traditional.
    #
    # What is NOT here any more, and why (all measured, see RESTORATION.md):
    #  * per-stem spectral denoising: three of the four stems are pure bleed of
    #    the same noise, so cleaning them separately costs signal three times.
    #    One denoiser on the bus removes the same noise once.
    #  * per-stem insert compression with automatic make-up: it raises everything
    #    below the threshold, i.e. it pumped the gaps back up (+5 dB on the bass
    #    floor) and ate the programme's own dynamics.
    #  * harmonic excitation / "air" per stem: measured to add sibilance-like
    #    sidebands, not harmonics; the top octave is reconstructed once, on the
    #    bus, by the trained network.
    stem = {}
    proc = {}
    for k in STEMS:
        x = raw[k]
        x, nclick = declick(x, SR, k=7.0)
        x, hum = remove_hum(x, base=rec.get('hum', 50.0))
        ch = STEM_CHAIN[k]
        x = block_dc_subsonic(x, SR, hp=ch['hp'], lp=ch['lp'])
        x = decrackle(x, SR, thresh_db=-26.0, mix=0.5)
        x = eq_curve(x, ch['eq'][0], ch['eq'][1])
        x = wb_declick(x, SR, freq=6000.0)
        stem[k] = x
        proc[k] = {'declicked': int(nclick), 'hum_notches': hum}
        stage(x, 'S1_' + k)

    # =========================================================== BALANCE
    # Only the two edges of the spectrum are nudged; the mid balance the producer
    # chose stays. A wider change would be a remix, not a restoration.
    for k in stem:
        g = float(np.clip(rec['balance_db'][k], -5.0, 5.0))
        stem[k] = (stem[k] * (10 ** (g / 20.0))).astype(np.float32)
    info['stems_out_rms_db'] = {k: round(rms_db(v), 2) for k, v in stem.items()}

    # Sum to the original programme level; this is also the point where the
    # separated stems become one signal again, so all bus processing below sees
    # exactly the material the listener will hear.
    core = sum(stem.values())
    gain = rms_db(src) - rms_db(core)
    core = (core * (10 ** (gain / 20.0))).astype(np.float32)
    info['balance_gain_db'] = round(gain, 2)
    stage(core, 'B1_sum')

    # =========================================================== BUS: noise
    # One denoiser for the whole programme. Minimum statistics over the full mix
    # can tell "hiss in a gap" from "cymbals in a hit"; on a single stem it cannot,
    # and the four independent estimates add up to more hiss than they remove.
    core = denoise(core, SR, reduction_db=rec.get('dn_bus', 11.0), floor_db=-22.0)
    stage(core, 'B2_denoise')

    # =========================================================== BUS: voicing
    # Two steps, and the order matters:
    #  (a) undo what the *separation* did to the balance, measured against the
    #      original mix. This is not a taste decision - it is a measurement, and
    #      it is provably neutral when the separation was already balanced.
    #  (b) apply the intended restoration voicing on top of that.
    # Before this, step (a) did not exist and the fixed voicing curve of step (b)
    # was doing a job it could not do: it cut 2.5 dB at 400 Hz, where the source
    # needs nothing (+0.1 dB), and left the +2.3 dB of bass overhang from Demucs
    # untouched. Measured cost: 1.5 dB of SNR in 200 Hz - 1 kHz.
    core, sep_curve, sep_keys, sep_delta = dsp.correction_from_source(
        core, src_mono, SR, fmin=32.0, fmax=6300.0, max_db=rec.get('sep_fix_db', 3.0))
    info['separation_correction_db'] = {int(k): round(v, 2) for k, v in sep_curve.items()}
    stage(core, 'B3a_sep_fix')

    core, curve, before, after = dsp.match_target(
        core, SR, strength=rec['tilt'], fmin=30.0, fmax=6300.0,
        max_boost=rec['max_boost'], max_cut=rec['max_cut'])
    info['tone_match'] = {'correction_db': {int(k): round(v, 2) for k, v in curve.items()},
                          'before_octave_db': {int(k): v for k, v in before.items()},
                          'after_octave_db': {int(k): v for k, v in after.items()}}
    stage(core, 'B3_voicing')

    # =========================================================== BUS: dynamics
    # Parallel compression only: the dry signal keeps its transient and its
    # breathing, the compressed copy underneath supplies the density. Measured
    # programme spread is reported below and must not collapse.
    m_core = core
    par, gr = compress(core, SR, thresh_db=rms_db(m_core) - 22.0, ratio=2.5,
                       attack_ms=10, release_ms=180, makeup_db=0.0, knee_db=10)
    par = norm_db(par, rms_db(core) - 7.0)
    pmix = float(rec.get('par', 0.16))
    core = (core * (1 - pmix) + par * pmix).astype(np.float32)
    info['parallel_pct'] = round(pmix * 100, 1)
    stage(core, 'B4_parallel')

    # =========================================================== BUS: glue
    # A very gentle slow compressor for cohesion. Threshold far below the peaks so
    # it only shapes phrases, not hits.
    core, gr_glue = compress(core, SR, thresh_db=rms_db(m_core) - 14.0, ratio=1.35,
                             attack_ms=45, release_ms=350, makeup_db=0.0, knee_db=14)
    core = norm_db(core, rms_db(m_core))
    stage(core, 'B5_glue')

    # =========================================================== STEREO
    # Sources are dual mono. The side component is built from the programme with
    # an allpass decorrelator (flat magnitude -> no comb), level-solved to a
    # requested correlation, and added as +/- so the mono sum is untouched.
    target_corr = float(np.clip(rec.get('stereo_corr', 0.93), 0.85, 0.995))
    side = image.build_side([core], SR, target_corr=target_corr, hp_hz=900.0)
    st = np.stack([core + side, core - side], axis=-1).astype(np.float32)
    info['stereo_target_corr'] = round(target_corr, 4)
    info['premaster_corr'] = round(float(np.corrcoef(st[:, 0], st[:, 1])[0, 1]), 4)
    _s, _m = (st[:, 0] - st[:, 1]) / 2, (st[:, 0] + st[:, 1]) / 2
    info['premaster_side_minus_mid_db'] = round(rms_db(_s) - rms_db(_m), 1)
    stage(st, 'B6_stereo')

    # =========================================================== TOP OCTAVE
    # The top octave is gone in the source (the 12.2-15.8 kHz band sits 35-45 dB
    # below the mid reference). The trained network reconstructs it; the noise-only
    # remainder above `air_cut_hz` is attenuated first so the network replaces hiss
    # instead of stacking on top of it.
    # The denoise cut and the reconstruction have to OVERLAP, not abut.
    #
    # Measured on the first render of this stage: the noise cut ramped from 8.0 to
    # 9.5 kHz while the network started adding at exactly 8.0 kHz, so 6.6-7.5 kHz
    # came out 2.4-2.9 dB low and 8 kHz jumped +24.5 dB in a single bin. In the
    # spectrogram that is a black bar at 7.2-8 kHz followed by a hard edge - the
    # "hole then ledge" the listener would hear as a lisp with a shelf on it.
    #
    # Now the cut starts a full octave lower (4 kHz) and ramps gently to 9.5 kHz,
    # so the top of the cymbal and the sibilance of the vocal are being faded out
    # while the network is fading in over the same region.
    # The noise cut must NOT eat the region the network is about to fill. It ran
    # at -12 dB from 8 kHz upwards, and the reconstruction target is -26 dB
    # relative - so the cut was removing about half of what the band needed and
    # the closed loop compensated by asking for more gain everywhere, which put a
    # -6 dB notch at 6.6-7 kHz next to a +24 dB shelf at 8 kHz.
    #
    # The cut is there for one reason: the source's 8-16 kHz region is 17-26 dB
    # below where it belongs and consists mostly of codec noise. The network
    # replaces that content anyway. -5 dB is enough to take the noise down without
    # fighting the reconstruction, and the reconstruction sets the resulting level.
    # The cut is applied ONLY above the reconstruction band. Any cut inside
    # 6.8-16 kHz is a notch that the reconstruction then has to climb out of, and
    # the spectrogram showed exactly that: a dark bar at 7-8 kHz in every render,
    # no matter where the crossover was placed or how wide the ramp was made.
    # The band's own level is set by the continuous gain curve in bwe2 alone, so
    # there is nothing to carve out first.
    st = dsp.spectral_noise_cut(st, SR, cut_hz=16000.0, smooth_oct=0.5,
                                amount_db=rec.get('air_noise_cut_db', -6.0))
    st, air_info = air_bwe2(st, SR, target_db=rec.get('air_target_db', -26.0))
    info['air'] = air_info
    stage(st, 'B7_air')

    # =========================================================== LIMIT + NORMALISE
    # The subsonic high-pass happens *before* the loudness loop: doing it after
    # left song 2 at -1.76 dBTP while the others sat at -1.00, because the filter
    # removes a little energy the limiter had already accounted for. Before the
    # loop, the loop's final gain applies to the signal that is actually written.
    _b, _a = sg.butter(6, 24.0 / (SR / 2), 'high')
    st = sg.filtfilt(_b, _a, st, axis=0).astype(np.float32)
    st -= st.mean(axis=0)

    # Loudness and true peak have to be met *together*, and they fight: raising the
    # level to reach -12 LUFS drives the peaks into the limiter, which adds density
    # and therefore raises LUFS again, which then has to be pulled back down. That
    # is a fixed point, not a single gain. The old loop only ever gained *up* to the
    # loudness target, so a programme whose peaks sat 2.4 dB lower (song 2) was
    # delivered at -2.38 dBTP instead of -1.00 - the numbers looked "safe" but the
    # four masters were not uniform, which is the one thing this delivery promises.
    # Order matters: gain to the loudness target, limit at the ceiling, then pull
    # the (now slightly louder) result back to the target with a plain gain. That
    # last step can only lower peaks, so it can never break the ceiling - whereas
    # correcting *before* the limiter let the limiter add density and drift the
    # loudness (song 1 landed at -12.40 LUFS that way).
    grs = []
    for _ in range(4):
        cur = loudness(st)
        if abs(cur - rec['target_lufs']) > 0.02:
            st = (st * (10 ** ((rec['target_lufs'] - cur) / 20.0))).astype(np.float32)
        # 0.08 dB of margin below the ceiling: the limiter works on the sample
        # peak, and 8x oversampling later reveals a little inter-sample overshoot
        # above it. Without the margin song 1 shipped at -0.90 dBTP against a
        # -1.00 spec even though its sample peak was at -1.006 dBFS.
        st, gr = limiter(st, SR, ceiling_db=rec['ceiling'] - 0.08, lookahead_ms=5.0,
                         release_ms=170)
        grs.append(gr)
        cur = loudness(st)
        if cur > rec['target_lufs']:
            st = (st * (10 ** ((rec['target_lufs'] - cur) / 20.0))).astype(np.float32)
        if abs(loudness(st) - rec['target_lufs']) < 0.02:
            break

    # ---- final catch-up: hit BOTH targets, not just loudness -----------------
    # The loop above pulls the loudness *down* after limiting without re-limiting,
    # which is safe. But when the programme comes out below target it has to be
    # gained *up*, and then the true peak rises with it. Song 2 did exactly that
    # and shipped at -1.17 dBTP instead of -1.00. So: bound the gain increase by
    # the headroom left to the ceiling, and re-limit afterwards if anything is
    # still over. Two rounds are enough because the limiter only ever lowers peaks,
    # so the second round can only need a tiny correction.
    for _ in range(3):
        cur_l = loudness(st)
        cur_tp = true_peak_db(st)
        need = rec['target_lufs'] - cur_l
        head = rec['ceiling'] - cur_tp
        if abs(need) < 0.02 and cur_tp <= rec['ceiling'] + 1e-3:
            break
        gain = min(need, head) if need > 0 else need
        if abs(gain) > 1e-4:
            st = (st * (10 ** (gain / 20.0))).astype(np.float32)
        if true_peak_db(st) > rec['ceiling'] + 1e-3:
            st, _ = limiter(st, SR, ceiling_db=rec['ceiling'] - 0.08, lookahead_ms=4.0,
                            release_ms=150)
        # a limit can push loudness up: only ever pull down from here
        over = loudness(st) - rec['target_lufs']
        if over > 0.02:
            st = (st * (10 ** (-over / 20.0))).astype(np.float32)

    info['master'] = {
        'lufs': round(loudness(st), 2),
        'true_peak_dbtp': round(true_peak_db(st), 2),
        'rms_db': round(rms_db(st), 2),
        'crest_db': round(peak_db(st) - rms_db(st), 2),
        'glue_gr_db': round(gr_glue, 2),
        'parallel_gr_db': round(gr, 2),
        'limiter_gr_db': round(min(grs), 2),
        'bands': band_edges(st.mean(axis=1)),
        'stereo_corr': round(float(np.corrcoef(st[:, 0], st[:, 1])[0, 1]), 4),
        'image': image.analyse(st, SR),
        'mono_sum_rms_db': round(rms_db(st.mean(axis=1)), 2),
    }
    info['stem_chain'] = proc
    info['elapsed_s'] = round(time.time() - t0, 1)

    base = f'{out_dir}/{tag} (restored)'
    sf.write(base + '.wav', st, SR, subtype='PCM_24')
    rng = np.random.default_rng(7)
    q = 1.0 / 32768.0
    dth = (rng.random(st.shape) - rng.random(st.shape)) * q
    sf.write(base + ' 16bit.wav', np.clip(st + dth, -1.0, 1.0), SR, subtype='PCM_16')
    info['output'] = base + '.wav'
    if verbose:
        print(json.dumps(info, indent=1, default=str), flush=True)
    return info, st


if __name__ == '__main__':
    tags = sys.argv[1:] or list(RECIPES)
    report_path = f'{ROOT}/work/report.json'
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}
    for tag in tags:
        print(f'--- restoring {tag}', flush=True)
        info, _ = restore_song(tag, f'{ROOT}/restored')
        report[tag] = info
        json.dump(report, open(report_path, 'w'), indent=2, default=str)
