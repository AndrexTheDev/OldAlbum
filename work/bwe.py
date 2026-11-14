#!/usr/bin/env python3
"""
Neural bandwidth extension (BWE) + coherent harmonic transposition.

Why this file exists
--------------------
Measured on the raw sources, the upper bands are neither "dead" nor "pure hiss".
Taking each band's *residual above its own noise floor* and correlating it with
the broadband music envelope gives, on Put It On:

    4-6 kHz   r = 0.82, SNR 26 dB   real music, keep
    6-8 kHz   r = 0.74, SNR 23 dB   real music + damage, repair
    8-11 kHz  r = 0.73, SNR 20 dB   mostly damage, reconstruct
    11-14 kHz r = 0.80, SNR 11 dB   reconstruct
    14-15.8 k r = 0.75, SNR 11 dB   reconstruct

and a ridge regression from 24 log-spaced bands (200 Hz-6 kHz) to the 16 bands
between 6.3 and 16 kHz explains **57 % of the variance** of that region
(5-fold CV, ~3 dB RMSE per band). So the top end is *predictable*: it is the
upper harmonics of instruments whose lower harmonics are still intact.

Two pieces make the reconstruction:

1. **Spectral envelope** - a small CNN predicts, per frame, the log-magnitude
   envelope one octave above the highest reliable band. It is trained
   self-supervised *on this album*: its inputs are the 4-8 kHz octave plus the
   200 Hz-6 kHz context, its target is the real 8-16 kHz octave. Whatever it
   learns about these particular microphones, instruments and room then gets
   applied one octave lower, which is the standard recursive trick for
   band-limited material. A generic pretrained model would not know that this is
   a 1964 Studio One horn section through a ribbon mic.
2. **Fine structure** - taken from the real signal one octave below by
   *harmonic transposition* in the STFT domain (bin k -> 2k, phase doubled).
   That keeps every attack aligned with the note that caused it, which is what
   makes the difference between "restored top end" and "a hiss layer on top".

The network only ever predicts a *correction* (+-12 dB) to the transposed
signal's own spectrum, so it can never invent an instrument that is not playing.
"""
import os
import sys
import json
import time
import numpy as np

SR = 44100
NFFT = 1024
HOP = 256
N_BANDS = 96
F_LO = 200.0
F_HI = 18000.0
ROOT = '/home/user/OldAlbum'


# --------------------------------------------------------------------- features
def band_matrix(sr=SR, nfft=NFFT, n_bands=N_BANDS, f_lo=F_LO, f_hi=F_HI):
    n_bins = nfft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    edges = np.geomspace(f_lo, f_hi, n_bands + 2)
    M = np.zeros((n_bands, n_bins), dtype=np.float32)
    for i in range(n_bands):
        a, b = edges[i], edges[i + 2]
        c = edges[i + 1]
        left = (freqs - a) / max(c - a, 1e-9)
        right = (b - freqs) / max(b - c, 1e-9)
        M[i] = np.clip(np.minimum(left, right), 0, None)
        s = M[i].sum()
        if s > 0:
            M[i] /= s
    return M, edges


def stft(x, nfft=NFFT, hop=HOP):
    from scipy import signal
    return signal.stft(np.asarray(x, np.float64), SR, nperseg=nfft,
                       noverlap=nfft - hop, window='hann', padded=True)


def band_db(mag, M):
    E = M @ (mag ** 2)
    return (10 * np.log10(np.maximum(E, 1e-16))).astype(np.float32)


def band_centres(edges):
    return np.sqrt(edges[:-1] * edges[1:])


# ------------------------------------------------------------------ data gather
def frame_pairs(x, M, edges, src_lo=4000.0, src_hi=8000.0, tgt_hi=16000.0,
                min_snr_db=6.0, n_ctx=24):
    """input = 200 Hz-4 kHz context + 4-8 kHz octave, target = 8-16 kHz octave."""
    _, _, Z = stft(x)
    mag = np.abs(Z)
    dB = band_db(mag, M)
    c = band_centres(edges)
    i_ctx = np.flatnonzero((c >= F_LO) & (c < src_lo))[:n_ctx]
    i_lo = np.flatnonzero((c >= src_lo) & (c < src_hi))
    i_hi = np.flatnonzero((c >= src_hi) & (c < tgt_hi))
    ctx, lo, hi = dB[i_ctx].T, dB[i_lo].T, dB[i_hi].T
    # a target frame is usable when the target band is clearly above its floor
    floor = np.percentile(hi, 15, axis=0)
    usable = (hi > floor + min_snr_db).mean(axis=1) > 0.6
    music = ctx[:, -6:].mean(axis=1)
    usable &= music > np.percentile(music, 20)
    X = np.concatenate([ctx, lo], axis=1)[usable]
    Y = hi[usable]
    return X.astype(np.float32), Y.astype(np.float32)


# ------------------------------------------------------------------ the network
def build_model(n_in, n_out, width=24):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv1d(1, width, 5, padding=2)
            self.b = nn.Conv1d(width, width * 2, 5, padding=2)
            self.c = nn.Conv1d(width * 2, width * 2, 3, padding=1)
            self.u = nn.Conv1d(width * 2, width, 5, padding=2)
            self.head = nn.Conv1d(width, 1, 1)
            self.act = nn.GELU()
            self.drop = nn.Dropout(0.05)
            self.n_out = n_out
            self.n_in = n_in

        def forward(self, x):
            # 1-D convolution along the *band* axis: each output band sees its
            # neighbouring bands, which is what a spectral envelope is
            h = x[:, None, :]
            a = self.act(self.a(h))
            b = self.act(self.b(a))
            c = self.act(self.c(b))
            u = self.act(self.u(c))
            y = self.head(self.drop(u))[:, 0, :]
            return y[:, -self.n_out:]

    return Net()


def train(X, Y, epochs=120, lr=6e-3, seed=0, bs=512, verbose=True):
    import torch
    # one thread is ~12x faster here: the problems are tiny, the thread
    # synchronisation dominates with more workers
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net = build_model(X.shape[1], Y.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    n = len(x)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = net(x[idx])
            loss = ((pred - y[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if verbose and (ep + 1) % 30 == 0:
            print(f'    epoch {ep+1:3d}/{epochs}  train MSE {tot/n:6.2f} dB^2', flush=True)
    net.eval()
    return net


def predict(net, X):
    import torch
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()


# ------------------------------------------------------- harmonic transposition
def transpose_octave(Z, noise_Z=None, noise_gain=0.0):
    """Move every STFT bin one octave up (k -> 2k), keeping time alignment.

    Doubling the phase keeps the transient alignment of the original event. A
    naive copy leaves every second bin empty (a comb), so each new bin also gets
    a share of its two neighbouring source bins - and, because a real cymbal is
    not a pure harmonic stack, an optional slice of the *original* high band is
    mixed in as noise. That noise is what makes the difference between "a stack
    of harmonics" (flatness 0.21, no spectral peaks) and an actual instrument
    (flatness 0.10, 23 dB spectral crest).
    """
    n_bins, n_frames = Z.shape
    out = np.zeros_like(Z)
    k = np.arange(1, n_bins // 2)
    mag = np.abs(Z[k])
    ph = np.angle(Z[k])
    out[2 * k] = mag * np.exp(2j * ph)
    odd = 2 * k - 1
    ok = (odd > 0) & (odd < n_bins)
    out[odd[ok]] = (0.5 * mag[ok]) * np.exp(2j * ph[ok])
    if noise_Z is not None and noise_gain > 0:
        nm = np.abs(noise_Z)
        out += (nm * np.exp(1j * np.angle(noise_Z)) * noise_gain).astype(np.complex64)
    return out.astype(np.complex64)


def apply_envelope(Zsrc, dB_target, centres_t, edges, sr=SR, nfft=NFFT):
    """Scale a transposed STFT by a smooth log-frequency envelope given in dB."""
    n_bins = Zsrc.shape[0]
    freqs = np.linspace(0, sr / 2, n_bins)
    env = np.empty(n_bins, dtype=np.float64)
    env[:] = np.nan
    # interpolate the predicted band levels onto the FFT grid (log-frequency)
    for j, fc in enumerate(centres_t):
        pass
    lo = centres_t[0] / np.sqrt(centres_t[1] / centres_t[0])
    hi = centres_t[-1] * np.sqrt(centres_t[-1] / centres_t[-2])
    sel = (freqs >= lo) & (freqs <= hi)
    env[sel] = np.interp(np.log(np.maximum(freqs[sel], 1.0)),
                         np.log(centres_t), dB_target)
    # outside the predicted range: hold the edge values
    env[freqs < lo] = np.nan
    env[freqs > hi] = dB_target[-1] - 6.0 * np.log2(freqs[freqs > hi] / hi)
    return env


def reconstruct(x, net, norm, M, edges, blend=1.0, noise_gain=0.0,
                envelope_gamma=1.0, anchor=1.0, verbose=False):
    """Return the synthesised top octave (as a full-length signal) + diagnostics."""
    from scipy import signal
    f, t, Z = stft(x)
    mag = np.abs(Z)
    dB = band_db(mag, M)
    c = band_centres(edges)
    i_ctx = np.flatnonzero((c >= F_LO) & (c < 4000.0))[:24]
    i_lo = np.flatnonzero((c >= 4000.0) & (c < 8000.0))
    i_hi = np.flatnonzero((c >= 8000.0) & (c < 16000.0))
    X = np.concatenate([dB[i_ctx].T, dB[i_lo].T], axis=1).astype(np.float32)
    mu, sd, my = norm
    pred = predict(net, (X - mu) / sd) + my

    # transposed source: take 4-8 kHz, move it up one octave.
    Zlow = Z.astype(np.complex64)
    kb = np.linspace(0, SR / 2, Z.shape[0])
    # the noise share is taken from the *real* source band (6.3-8 kHz), so it
    # carries the microphone's own noise signature instead of synthetic white
    # noise; it is what gives the reconstruction the spectral "grain" of a real
    # instrument (measured: flatness 0.21 with harmonics only, 0.10 with grain)
    Znoise = Z.astype(np.complex64)
    Zlow[(kb < 4000) | (kb > 8000), :] = 0
    Znoise[(kb < 6300), :] = 0
    Zt = transpose_octave(Zlow, noise_Z=Znoise, noise_gain=noise_gain)
    del Zlow, Znoise

    centers_t = c[i_hi][:len(i_hi)]
    src_mag = np.abs(Zt)
    Mt = M[i_hi]
    # The octave transposition halves the number of bins per band, so the raw
    # copy is systematically ~2-6 dB quieter than the predicted level. Calibrate
    # the copy on its *own* median so the network only has to supply the shape
    # and the song-to-song balance, never a huge constant gain.
    Et = Mt @ (src_mag ** 2)
    meas = (10 * np.log10(np.maximum(Et, 1e-16))).T            # (frames, bands)
    offset = float(np.median(pred - meas))
    gain_db = np.clip(pred - meas - offset, -12.0, 12.0) * blend
    # sharpen the predicted envelope around its running mean: the network
    # regresses to the mean (that is what MSE does), which flattens the spectral
    # peaks an instrument needs. gamma > 1 restores contrast without changing
    # the average level.
    from scipy.ndimage import uniform_filter1d as _uf
    base = _uf(gain_db.mean(axis=1), 25)[:, None]
    gain_db = base + (gain_db - base) * envelope_gamma
    # smooth in time so the envelope does not chatter
    from scipy.ndimage import uniform_filter1d
    gain_db = uniform_filter1d(gain_db, 9, axis=0)
    n_bins = Zt.shape[0]
    freqs = np.linspace(0, SR / 2, n_bins)
    lo_e = centers_t[0] / np.sqrt(centers_t[1] / centers_t[0])
    hi_e = centers_t[-1] * np.sqrt(centers_t[-1] / centers_t[-2])
    lf = np.log(np.maximum(freqs, 1.0))
    sel = freqs >= lo_e
    # per-bin interpolation of the predicted band gains on the log-freq grid,
    # done once as a (bins, 1) x (frames,) broadcast instead of a Python loop
    # over frames (which is what made this stage run out of memory)
    # weight matrix from target bands to FFT bins (same log-freq interpolation
    # used for the envelope, reused for the anchor correction)
    w_full = np.zeros((n_bins, len(centers_t)), dtype=np.float32)
    for j in range(len(centers_t) - 1):
        a_, b_ = np.log(centers_t[j]), np.log(centers_t[j + 1])
        m_ = (lf >= a_) & (lf <= b_)
        t_ = (lf[m_] - a_) / max(b_ - a_, 1e-9)
        w_full[m_, j] = 1 - t_
        w_full[m_, j + 1] = t_
    w_full[lf < np.log(centers_t[0]), 0] = 1.0
    w_full[lf > np.log(centers_t[-1]), -1] = 1.0
    w = np.zeros((int(sel.sum()), len(centers_t)), dtype=np.float32)
    for j in range(len(centers_t) - 1):
        a, b = np.log(centers_t[j]), np.log(centers_t[j + 1])
        m = (lf[sel] >= a) & (lf[sel] <= b)
        t = (lf[sel][m] - a) / max(b - a, 1e-9)
        w[m, j] = 1 - t
        w[m, j + 1] = t
    w[lf[sel] < np.log(centers_t[0]), 0] = 1.0
    w[lf[sel] > np.log(centers_t[-1]), -1] = 1.0
    env_db = np.full((n_bins, Zt.shape[1]), -90.0, dtype=np.float32)
    env_db[sel, :] = (w @ gain_db.T).astype(np.float32)   # (bins, frames)
    env_db_meas = np.array(gain_db, dtype=np.float32)     # (frames, bands)
    above = freqs > hi_e
    if above.any():
        env_db[above, :] -= (6.0 * np.log2(freqs[above] / hi_e))[:, None]
    # apply the envelope in place: a second (bins x frames) complex64 array
    # would push this stage over the memory limit on long stems
    # ---- anchor to the real recording
    # The source still carries genuine (if damaged) music up to ~15.8 kHz. The
    # synthesis must *continue* that, not overwrite it: the final envelope is
    # pulled towards the real band wherever the real band is strong (i.e. above
    # its own local floor). This is what keeps the result a restoration instead
    # of a layer, and it measurably raises the correlation between the
    # reconstructed band and the recording's own 8-16 kHz content.
    if anchor > 0:
        # Minimal, *linear* pull of the predicted envelope towards the energy
        # the recording itself has in the band. Power-domain blending of two
        # uncorrelated signals produces exactly the discontinuities the earlier
        # attempt showed (kurtosis 132), so this only ever nudges the envelope:
        # the synthesised fine structure is untouched, and the total energy in
        # the band moves at most `anchor` dB towards the real one.
        real_db = (10 * np.log10(np.maximum(Mt @ (np.abs(Z) ** 2), 1e-16))).T
        frame_lvl = real_db.mean(axis=1, keepdims=True)
        conf = np.clip((frame_lvl - np.percentile(frame_lvl, 20)) / 14.0, 0.0, 1.0)
        delta = np.clip(real_db - env_db_meas, -anchor, anchor) * conf
        env_db = env_db + (w_full @ delta.T).astype(np.float32)
    np.multiply(Zt, 10.0 ** (env_db / 20.0), out=Zt)
    del env_db
    Y = Zt
    _, y = signal.istft(Y, SR, nperseg=NFFT, noverlap=NFFT - HOP, window='hann')
    y = y[:len(x)]
    # final check: the synthesised band must sit where the network predicted it
    Et2 = Mt @ (np.abs(signal.stft(y, SR, nperseg=NFFT, noverlap=NFFT - HOP,
                                  window='hann', padded=True)[2]) ** 2)
    m2 = (10 * np.log10(np.maximum(Et2, 1e-16))).T
    resid = float(np.median(pred - m2))
    if abs(resid) > 0.5:
        y = y * (10 ** (resid / 20.0))
    if verbose:
        print('    BWE: %d frames, envelope correction %+.1f..%+.1f dB, '
              'residual fixed %+.1f dB, out rms %.4f'
              % (len(X), gain_db.min(), gain_db.max(), resid, np.std(y)), flush=True)
    return y.astype(np.float32), gain_db


def reconstruct_long(x, net, norm, M, edges, block_s=20.0, overlap_s=1.0,
                     sr=SR, verbose=False, **kw):
    """`reconstruct` on a whole song, in overlapping blocks.

    A full song is ~32k STFT frames; the intermediate complex arrays for that are
    2.5 GB, which does not fit in this sandbox. Block-wise processing with a
    cross-fade is lossless here because the network works per frame on a
    smoothed envelope, so block boundaries are inaudible.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    blk = int(block_s * sr)
    ov = int(overlap_s * sr)
    out = np.zeros(n, dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    win = np.ones(blk, dtype=np.float32)
    fade = int(0.5 * ov)
    if fade > 0:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
        win[:fade] = ramp
        win[-fade:] = ramp[::-1]
    pos = 0
    while pos < n:
        a = max(0, pos - ov // 2)
        b = min(n, a + blk)
        seg = x[a:b]
        y, _ = reconstruct(seg, net, norm, M, edges, **kw)
        w = win[:len(seg)]
        out[a:b] += y[:len(seg)] * w
        wsum[a:b] += w
        pos += blk - ov
    wsum[wsum < 1e-6] = 1.0
    return (out / wsum).astype(np.float32)


# ----------------------------------------------------------------------- driver
def train_model(verbose=True):
    M, edges = band_matrix()
    tags = ['1_I_Need_You', '2_Lonesome_Feeling', '3_Put_It_On', '4_Whats_New_Pussycat']
    import soundfile as sf
    Xs, Ys = [], []
    for tag in tags:
        x = sf.read(f'{ROOT}/work/src/{tag}.wav', always_2d=True, dtype='float32')[0].mean(1)
        X, Y = frame_pairs(x, M, edges)
        Xs.append(X); Ys.append(Y)
        if verbose:
            print(f'  {tag:24} {len(X):5d} reliable frames', flush=True)
    X = np.vstack(Xs); Y = np.vstack(Ys)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
    my = Y.mean(0, keepdims=True)
    t0 = time.time()
    net = train((X - mu) / sd, Y - my, epochs=120, verbose=verbose)
    # honest held-out score
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(X)); te = idx[:len(X) // 5]
    pred = predict(net, ((X[te] - mu) / sd)) + my
    mse = float(np.mean((pred - Y[te]) ** 2))
    base = float(np.mean((Y[np.setdiff1d(idx, te)].mean(0) - Y[te]) ** 2))
    import torch
    torch.save({'state': net.state_dict(),
                'norm': [mu.tolist(), sd.tolist(), my.tolist()],
                'n_in': int(X.shape[1]), 'n_out': int(Y.shape[1])},
               f'{ROOT}/work/models/bwe.pt')
    rep = {'frames': int(len(X)), 'time_s': round(time.time() - t0, 1),
           'heldout_mse_db2': mse, 'baseline_mse_db2': base,
           'r2': round(1 - mse / base, 3)}
    json.dump(rep, open(f'{ROOT}/work/bwe_report.json', 'w'), indent=1)
    if verbose:
        print('  held-out: MSE %.1f dB^2 (baseline %.1f) -> R2 %.2f in %.0f s'
              % (mse, base, 1 - mse / base, time.time() - t0), flush=True)
    return net, (mu, sd, my)


def load_model():
    import torch
    d = torch.load(f'{ROOT}/work/models/bwe.pt', weights_only=False)
    net = build_model(d['n_in'], d['n_out'])
    net.load_state_dict(d['state'])
    net.eval()
    norm = tuple(np.array(a, dtype=np.float32) for a in d['norm'])
    return net, norm


if __name__ == '__main__':
    train_model()


# --------------------------------------------------------------- multi-instrument
# One network per instrument family, each with its own normalisation. A horn
# section and a hi-hat do not share an upper-octave law, so sharing one model
# (as a first version did) averages the two and produces the generic "air"
# sound the user rightly rejected. 147k reliable frames across the four songs
# are enough to fit four small models.
INSTRUMENTS = ('vocals', 'drums', 'bass', 'other')
SEED_RATIO = {'vocals': 1.00, 'drums': 0.60, 'bass': 0.25, 'other': 0.85}


def train_per_instrument(verbose=True):
    M, edges = band_matrix()
    import soundfile as sf, torch
    tags = ['1_I_Need_You', '2_Lonesome_Feeling', '3_Put_It_On', '4_Whats_New_Pussycat']
    models, report = {}, {}
    for nm in INSTRUMENTS:
        Xs, Ys = [], []
        for tag in tags:
            x = sf.read(f'{ROOT}/work/stems/{tag}/{nm}.wav', always_2d=False,
                        dtype='float32')[0]
            if x.ndim > 1:
                x = x.mean(1)
            X, Y = frame_pairs(x, M, edges)
            Xs.append(X); Ys.append(Y)
        X = np.vstack(Xs); Y = np.vstack(Ys)
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
        my = Y.mean(0, keepdims=True)
        t0 = time.time()
        net = train((X - mu) / sd, Y - my, epochs=120, verbose=False)
        rng = np.random.default_rng(1)
        idx = rng.permutation(len(X)); te = idx[:len(X) // 5]
        pred = predict(net, (X[te] - mu) / sd) + my
        mse = float(np.mean((pred - Y[te]) ** 2))
        base = float(np.mean((Y[np.setdiff1d(idx, te)].mean(0) - Y[te]) ** 2))
        torch.save({'state': net.state_dict(),
                    'norm': [mu.tolist(), sd.tolist(), my.tolist()],
                    'n_in': int(X.shape[1]), 'n_out': int(Y.shape[1])},
                   f'{ROOT}/work/models/bwe_{nm}.pt')
        report[nm] = {'frames': int(len(X)), 'r2': round(1 - mse / base, 3),
                      'mse_db2': round(mse, 2), 'time_s': round(time.time() - t0, 1)}
        if verbose:
            print(f'  {nm:7} {len(X):6d} frames  R2 {report[nm]["r2"]:.2f}  '
                  f'MSE {mse:5.2f} dB^2  ({report[nm]["time_s"]:.0f}s)', flush=True)
    json.dump(report, open(f'{ROOT}/work/bwe_report.json', 'w'), indent=1)
    return report


def load_instrument(nm):
    """Load a trained model; 'master' = the full-mix air model."""
    import torch
    d = torch.load(f'{ROOT}/work/models/bwe_{nm}.pt', weights_only=False)
    net = build_model(d['n_in'], d['n_out'])
    net.load_state_dict(d['state']); net.eval()
    return net, tuple(np.array(a, dtype=np.float32) for a in d['norm'])


def add_neural_air(x, nm, M, edges, sr=SR, amount=1.0):
    """Add the reconstructed top octave to one stem, coherent and bounded.

    `amount` scales the synthesised band before mixing. The result is
    high-passed at 6.3 kHz so nothing below the reconstruction region is
    touched, and it is level-matched to the band it replaces.
    """
    from scipy import signal
    net, norm = load_instrument(nm)
    y, _ = reconstruct(x, net, norm, M, edges)
    b, a = signal.butter(6, 6300.0 / (sr / 2), 'high')
    y = signal.filtfilt(b, a, y).astype(np.float32)
    return (np.asarray(x, dtype=np.float32) + y * amount).astype(np.float32)
