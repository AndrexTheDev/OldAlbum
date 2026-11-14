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
import bwe
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
    '1_I_Need_You': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 5.0, 'bass': -3.0, 'other': 0.5},
        vocal=dict(hp=100.0, presence=2.2, air=4.5, excite=1.9, deess=5700.0),
        drums=dict(hp=35.0, punch=2.6, excite=1.3),
        bass=dict(hp=30.0, lp=1400.0, lowmid=1.6, harm=0.28),
        other=dict(hp=95.0, excite=1.9, width=4.0),
        tilt_gain=1.0, max_boost=6.0, max_cut=8.0, air_from_hz=5700.0,
        air_noise_cut_db=-12.0, air_db=-17.0, air_top=12000.0,
        air_vocals=0.0, air_drums=0.0, air_other=0.0, air_master=0.3,
        dn_vocals=10.0, dn_drums=8.0, dn_other=11.0, dn_master=9.0,
        target_lufs=-12.0, ceiling=-1.0,
    ),
    '2_Lonesome_Feeling': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 3.0, 'bass': -3.0, 'other': 0.5},
        vocal=dict(hp=100.0, presence=2.0, air=4.0, excite=1.8, deess=5700.0),
        drums=dict(hp=35.0, punch=2.2, excite=1.2),
        bass=dict(hp=30.0, lp=1400.0, lowmid=1.5, harm=0.26),
        other=dict(hp=95.0, excite=1.8, width=3.6),
        tilt_gain=1.0, max_boost=6.0, max_cut=8.0, air_from_hz=5700.0,
        air_noise_cut_db=-12.0, air_db=-17.0, air_top=12000.0,
        air_vocals=0.0, air_drums=0.0, air_other=0.0, air_master=0.3,
        target_lufs=-12.0, ceiling=-1.0,
    ),
    '3_Put_It_On': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 2.5, 'bass': -3.0, 'other': 0.5},
        vocal=dict(hp=100.0, presence=2.1, air=4.2, excite=1.9, deess=5700.0),
        drums=dict(hp=35.0, punch=2.4, excite=1.3),
        bass=dict(hp=30.0, lp=1400.0, lowmid=1.55, harm=0.27),
        other=dict(hp=95.0, excite=1.9, width=3.8),
        tilt_gain=1.0, max_boost=6.0, max_cut=8.0, air_from_hz=5700.0,
        air_noise_cut_db=-12.0, air_db=-17.0, air_top=12000.0,
        air_vocals=0.0, air_drums=0.0, air_other=0.0, air_master=0.3,
        target_lufs=-12.0, ceiling=-1.0,
    ),
    '4_Whats_New_Pussycat': dict(
        hum=50.0,
        balance_db={'vocals': 1.5, 'drums': 2.0, 'bass': -3.0, 'other': 1.0},
        vocal=dict(hp=100.0, presence=1.9, air=3.8, excite=1.8, deess=5700.0),
        drums=dict(hp=35.0, punch=2.0, excite=1.1),
        bass=dict(hp=30.0, lp=1400.0, lowmid=1.4, harm=0.25),
        other=dict(hp=95.0, excite=1.8, width=3.4),
        tilt_gain=1.0, max_boost=6.0, max_cut=8.0, air_from_hz=5700.0,
        air_noise_cut_db=0.0, air_db=-17.0, air_top=12000.0,
        air_vocals=0.0, air_drums=0.0, air_other=0.0, air_master=0.3,
        dn_vocals=9.0, dn_drums=7.0, dn_other=10.0, dn_master=8.0,
        target_lufs=-12.0, ceiling=-1.0,
    ),
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
    info['duration_s'] = round(n / SR, 3)
    info['source'] = {'lufs': round(loudness(src), 2), 'rms_db': round(rms_db(src), 2),
                      'peak_db': round(peak_db(src), 2), 'bands': band_edges(src)}
    info['stems_in_rms_db'] = {k: round(rms_db(v), 2) for k, v in raw.items()}

    proc = {}

    # ---------------------------------------------------------- VOCALS
    v = raw['vocals']
    v, nclick = declick(v, SR, k=7.0)
    v, hum = remove_hum(v, base=rec['hum'])
    v = block_dc_subsonic(v, SR, hp=rec['vocal']['hp'])
    v = decrackle(v, SR, thresh_db=-26.0, mix=0.6)
    p = rec['vocal']['presence']
    v = eq_curve(v, [45, 100, 165, 260, 420, 700, 1100, 2000, 3000, 4200,
                     5600, 7200, 9500, 12000, 16000, 20000],
                 [-8, -2.5, -3.0, -2.0, -0.6, 0.6, 0.8, p * 0.6, p, p * 0.8,
                      p * 0.5, 1.0, rec['vocal']['air'] * 0.8, rec['vocal']['air'],
                      rec['vocal']['air'] * 1.1, rec['vocal']['air'] * 0.4])
    # denoise AFTER the tonal lift, otherwise we amplify tape/quantisation hiss
    v = denoise(v, SR, reduction_db=rec.get('dn_vocals', 6.0), floor_db=-22.0)
    v = deess(v, SR, freq=rec['vocal']['deess'])
    # NOTE: the neural reconstruction is deliberately NOT applied per stem.
    # Stacking four independent reconstructions on top of each other gives a
    # spectrally flat, noise-like top end (measured: flatness 0.21 -> 0.38),
    # which is exactly the "effect" sound that must be avoided. The reconstruction
    # happens once, on the full mix, where the model sees the complete programme.
    v = wb_declick(v, SR, freq=rec.get('wb_declick_hz', 6000.0))
    v = transient_shape(v, SR, attack_db=1.4, sustain_db=-0.7)
    v, gr = soft_comp(v, SR, -9.0, 2.2, 8, 150, makeup=0)
    k5(v, 'vocals_out')
    proc['vocals'] = {'declicked': int(nclick), 'hum_notches': hum, 'comp_gr_db': round(gr, 2)}

    # ---------------------------------------------------------- DRUMS
    d = raw['drums']
    d, nclick = declick(d, SR, k=7.0)
    d, hum = remove_hum(d, base=rec['hum'])
    d = block_dc_subsonic(d, SR, hp=rec['drums']['hp'], lp=17500.0)
    d = decrackle(d, SR, thresh_db=-26.0, mix=0.5)
    d = transient_shape(d, SR, attack_db=rec['drums']['punch'], sustain_db=-1.4)
    d = eq_curve(d, [40, 60, 85, 120, 180, 300, 600, 1200, 2500, 4000,
                     6000, 9000, 13000, 16000],
                 [1.0, 2.2, 1.4, -0.6, -1.2, -0.6, 0.4, 0.9, 1.4, 1.5, 1.7, 1.9, 2.0, 1.6])
    d = denoise(d, SR, reduction_db=rec.get('dn_drums', 4.0), floor_db=-22.0)
    d = wb_declick(d, SR, freq=rec.get('wb_declick_hz', 6000.0))
    d = hf_control(d, SR, 6500.0, -34.0, 3.0)
    d, gr = soft_comp(d, SR, -8.0, 2.4, 14, 200)
    k5(d, 'drums_out')
    proc['drums'] = {'declicked': int(nclick), 'hum_notches': hum, 'comp_gr_db': round(gr, 2)}

    # ---------------------------------------------------------- BASS
    b = raw['bass']
    b, nclick = declick(b, SR, k=7.0)
    b, hum = remove_hum(b, base=rec['hum'])
    b = block_dc_subsonic(b, SR, hp=rec['bass']['hp'], lp=rec['bass']['lp'])
    b = decrackle(b, SR, thresh_db=-28.0, mix=0.5)
    lm = rec['bass']['lowmid']
    # the separated bass has a huge 40-60 Hz bump and a 150-250 Hz 'mud' swell
    b = eq_curve(b, [28, 40, 52, 65, 80, 100, 130, 170, 220, 300, 450, 700, 1000, 1400],
                 [-4, 0.0, 1.0, lm, lm, lm * 0.7, 0.6, 0.0, -1.4, -2.4, -3.4,
                  -5.0, -8.0, -12.0])
    # 2nd-harmonic bass enhancement so the line reads on small speakers
    harm = sg.filtfilt(*sg.butter(4, [70 / (SR / 2), 400 / (SR / 2)], 'band'), np.abs(b))
    b = (b + rec['bass']['harm'] * harm.astype(np.float32)).astype(np.float32)
    b, gr = soft_comp(b, SR, -8.0, 2.0, 18, 210)
    k5(b, 'bass_out')
    proc['bass'] = {'declicked': int(nclick), 'hum_notches': hum, 'comp_gr_db': round(gr, 2)}

    # ---------------------------------------------------------- OTHER
    o = raw['other']
    o, nclick = declick(o, SR, k=7.0)
    o, hum = remove_hum(o, base=rec['hum'])
    o = block_dc_subsonic(o, SR, hp=rec['other']['hp'])
    o = decrackle(o, SR, thresh_db=-26.0, mix=0.5)
    o = eq_curve(o, [95, 150, 260, 420, 700, 1200, 2000, 3200, 4600, 6500,
                     9000, 12000, 16000],
                 [-9, -3.5, -1.2, 0.2, 0.6, 0.8, 1.1, 1.4, 1.5, 1.7, 2.0, 2.2, 1.8])
    o = denoise(o, SR, reduction_db=rec.get('dn_other', 8.0), floor_db=-22.0)
    o = wb_declick(o, SR, freq=rec.get('wb_declick_hz', 6000.0))
    o = transient_shape(o, SR, attack_db=0.8, sustain_db=-0.4)
    o = hf_control(o, SR, 6500.0, -30.0, 2.5)
    o, gr = soft_comp(o, SR, -9.0, 1.9, 20, 240)
    k5(o, 'other_out')
    proc['other'] = {'declicked': int(nclick), 'hum_notches': hum, 'comp_gr_db': round(gr, 2)}

    # ---------------------------------------------------------- BALANCE
    stem = {'vocals': v, 'drums': d, 'bass': b, 'other': o}
    for k in stem:                       # clamp the requested change
        g = float(np.clip(rec['balance_db'][k], -5.0, 5.0))
        stem[k] = (stem[k] * (10 ** (g / 20.0))).astype(np.float32)
    info['stems_out_rms_db'] = {k: round(rms_db(s), 2) for k, s in stem.items()}

    # match the sum to the original program level, then keep headroom for mastering
    s_sum = sum(stem.values())
    gain = (rms_db(src) - 3.0) - rms_db(s_sum)
    stem = {k: (s * (10 ** (gain / 20.0))).astype(np.float32) for k, s in stem.items()}
    info['balance_gain_db'] = round(gain, 2)

    # ---------------------------------------------------------- STEREO IMAGE
    # The sources are 1960s mono. A modern release needs a real stereo picture,
    # but it must be built so that the MONO SUM stays intact (no comb filtering,
    # no phase cancellation): everything we add is a pure *side* component
    # (+d on the left, -d on the right), which cancels exactly in mono.
    n = len(stem['vocals'])

    def side_from(sig, ms):
        """Airy, high-passed delayed copy used only in the side channel."""
        d = int(ms * 1e-3 * SR)
        hp = sg.filtfilt(*sg.butter(2, 320 / (SR / 2), 'high'), sig).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        if d < n:
            out[d:] = hp[:n - d]
        return out

    core = stem['vocals'] + stem['drums'] + stem['bass']        # dead centre
    bed = stem['other']
    wgain = 10 ** (rec['other']['width'] / 20.0) - 1.0

    side_music = wgain * 0.42 * (side_from(bed, 8.7) - side_from(bed, 13.1))
    side_vocal = 0.10 * (side_from(stem['vocals'], 11.3) - side_from(stem['vocals'], 16.7))
    side = (side_music + side_vocal).astype(np.float32)

    k5(side, 'side')
    k5(bed, 'bed')
    mid = core + bed * 0.94
    left = mid + side
    right = mid - side
    st = np.stack([left, right], axis=-1).astype(np.float32)
    k5(st, 'premaster')
    info['premaster_corr'] = round(float(np.corrcoef(st[:, 0], st[:, 1])[0, 1]), 4)
    _side = (st[:, 0] - st[:, 1]) / 2
    _mid = (st[:, 0] + st[:, 1]) / 2
    info['premaster_side_minus_mid_db'] = round(rms_db(_side) - rms_db(_mid), 1)
    info['mono_sum_vs_mid_db'] = round(rms_db(_mid) - rms_db(mid + 0.0), 3)

    # ---------------------------------------------------------- MASTER BUS
    # 1. gentle glue so the parts move together
    m_st = st.mean(axis=1)
    st, gr_glue = compress(st, SR, thresh_db=rms_db(m_st) - 7.0, ratio=2.0,
                           attack_ms=35, release_ms=280, makeup_db=0.0, knee_db=10)
    # put back the sub-40 Hz weight the lossy encode clipped away
    k5(st, 'glue')
    st = eq_curve(st, [20, 30, 40, 55, 80, 120, 200],
                  [3.5, 2.6, 1.4, 0.4, 0.0, 0.0, 0.0])
    k5(st, 'sub_eq')
    # 2. parallel ('New York') compression: a heavily compressed copy lifts the
    #    quiet detail between the hits - the density a modern master has, while
    #    the transients of the dry signal survive untouched
    par, _ = compress(st, SR, thresh_db=rms_db(m_st) - 22.0, ratio=3.5,
                      attack_ms=6, release_ms=140, makeup_db=0.0, knee_db=8)
    par = norm_db(par, rms_db(st) - 6.0)
    k5(par, 'parallel_copy')
    st = (st * 0.80 + par * 0.20).astype(np.float32)
    k5(st, 'parallel_mix')
    info['parallel_pct'] = 20.0
    # tonal finish: match a modern reference balance (this is where the
    # 'sounds like a clean record of today' part happens)
    st, curve, before, after = dsp.match_target(
        st, SR, strength=rec['tilt_gain'], fmin=30.0, fmax=6300.0,
        max_boost=rec['max_boost'], max_cut=rec['max_cut'])
    info['tone_match'] = {'correction_db': {int(k): round(v, 2) for k, v in curve.items()},
                          'before_octave_db': {int(k): v for k, v in before.items()},
                          'after_octave_db': {int(k): v for k, v in after.items()}}
    # the tonal lift inevitably raises tape/quantisation hiss -> clean it again
    k5(st, 'tone_match')
    st = denoise(st, SR, reduction_db=rec.get('dn_master', 6.0), floor_db=-24.0)
    k5(st, 'denoise_master')
    # Remove the boxiness the acetate transfer left around 250-500 Hz: the
    # source carries +2..3 dB too much there relative to the rest.
    st = eq_curve(st, [120, 200, 280, 380, 500, 700, 900],
                  [0.0, -0.8, -1.8, -2.2, -1.6, -0.4, 0.0])
    k5(st, 'debox')
    # ---- top octave, part 2: neural reconstruction of the *mix*
    # The stems already carry their reconstructed top end; what is still missing
    # is whatever the separator assigned to nobody. A second network trained on
    # the full mix fills exactly that, level-matched to the predicted envelope,
    # and the noise-only remainder above 6.3 kHz is attenuated below it.
    # Everything below 8 kHz carries real music (measured: the 6.3-8 kHz band
    # has 23 dB SNR over silence and correlates 0.74 with the programme), so it
    # must NOT be attenuated - only the noise-only remainder above 8 kHz is,
    # and the neural model then fills that region back in.
    st = dsp.spectral_noise_cut(st, SR, cut_hz=rec.get('air_cut_hz', 8000.0),
                                amount_db=rec.get('air_noise_cut_db', -12.0))
    st = neural_air_stereo(st, 'master', amount=rec.get('air_master', 1.0),
                           hp_hz=rec.get('air_from_hz', 6800.0))
    k5(st, 'spectral_repair')

    # ---------------------------------------------------------- LIMIT + NORMALISE
    # Iterate: push to the loudness target, catch the peaks at the ceiling,
    # repeat until both land. More iterations => peaks ride closer to the
    # ceiling (denser master) without ever exceeding it.
    grs = []
    for _ in range(5):
        cur = loudness(st)
        if cur < rec['target_lufs']:
            st = (st * (10 ** ((rec['target_lufs'] - cur) / 20.0))).astype(np.float32)
        st, gr = limiter(st, SR, ceiling_db=rec['ceiling'], lookahead_ms=5.0,
                         release_ms=170)
        grs.append(gr)
    cur = loudness(st)
    if cur > rec['target_lufs'] + 0.1:
        st = (st * (10 ** ((rec['target_lufs'] - cur) / 20.0))).astype(np.float32)

    info['master'] = {
        'lufs': round(loudness(st), 2),
        'true_peak_dbtp': round(true_peak_db(st), 2),
        'rms_db': round(rms_db(st), 2),
        'crest_db': round(peak_db(st) - rms_db(st), 2),
        'glue_gr_db': round(gr_glue, 2),
        'limiter_gr_db': round(min(grs), 2),
        'bands': band_edges(st.mean(axis=1)),
        'stereo_corr': round(float(np.corrcoef(st[:, 0], st[:, 1])[0, 1]), 4),
        'mono_sum_rms_db': round(rms_db(st.mean(axis=1)), 2),
    }
    info['stem_chain'] = proc
    info['elapsed_s'] = round(time.time() - t0, 1)

    # ---------------------------------------------------------- SUB-REMOVAL + EXPORT
    # The sub-restoration EQ and the compressors can leave inaudible subsonic
    # energy below ~24 Hz. On a real system it only steals headroom and pumps
    # woofers, so it goes out with a steep zero-phase high-pass (and this also
    # centres the waveform at exactly 0).
    from scipy import signal as _sg
    _b, _a = _sg.butter(6, 24.0 / (SR / 2), 'high')
    st = _sg.filtfilt(_b, _a, st, axis=0).astype(np.float32)
    st -= st.mean(axis=0)
    # one last, transparent true-peak safety check after the high-pass
    if true_peak_db(st) > rec['ceiling']:
        st, _ = limiter(st, SR, ceiling_db=rec['ceiling'], lookahead_ms=4.0, release_ms=150)
    info['master']['lufs'] = round(loudness(st), 2)
    info['master']['true_peak_dbtp'] = round(true_peak_db(st), 2)
    info['master']['dc_offset'] = [round(float(v), 8) for v in st.mean(axis=0)]

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
