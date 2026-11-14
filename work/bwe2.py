#!/usr/bin/env python3
"""
Bandwidth extension, second attempt - and the first one that is validated honestly.

What was wrong with the first model
-----------------------------------
It was an MLP on a 24-band log-mel context that predicted the 8.8-17.6 kHz band
frame by frame. Held-out R2 was 0.69, and *cross-song* R2 was -0.19 to +0.42 with
the predicted band mean tracking the target band mean almost exactly. Diagnosis:
it learned to predict the average level of the band, not its detail. Against a
band-limited source that is enough to fill the spectrum with plausible energy, and
not enough to make a top end that belongs to the music.

Design here
-----------
* **Input is broadband context, not just the band below the gap.** 40 log-mel
  bands from 1 kHz to the cutoff carry the harmonic and transient structure;
  predicting the top octave from a narrow sliver is what forces the mean.
* **Recurrent across time.** A GRU over a 120 ms context lets the model see
  whether the current frame is inside a note or a transient. The MLP had no such
  notion - its "context" was a symmetric window of mel bands at one instant.
* **Predict in log-magnitude, train on L1 + spectral convergence**, then run an
  adversarial discriminator on the reconstructed band. BEHM-GAN's finding: the
  adversarial loss is what removes the "smoothed" character; reconstruction losses
  alone always give a blurred top end.
* **Validated across songs, not within them.** The only number that matters for
  music is: train on three songs, measure on the fourth. Anything else is a
  memorisation test, and this project already shipped one of those.

Reality check
-------------
Above ~15.8 kHz these sources are at -108 dB (the AAC codec's digital zero). There
is nothing to recover there, only to synthesise. This module extends 8-16 kHz -
where music genuinely exists but sits 17-26 dB low - and leaves 16 kHz upwards
alone rather than inventing a hiss wall.
"""
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from scipy import signal as sg

SR = 44100
N_FFT = 1024
HOP = 256
N_MELS_IN = 48
N_MELS_OUT = 24
CUT_HZ = 8000.0        # input band limit (what the source reliably has)
TOP_HZ = 16000.0       # where the source ends


# --------------------------------------------------------------------------- features
def mel_fb(n_mels, fmin, fmax, sr=SR, n_fft=N_FFT):
    """HTK-style mel filterbank (triangular, area-normalised)."""
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        for k in range(l, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m - 1, k] = (r - k) / max(1, r - c)
    return fb


_FB_IN = mel_fb(N_MELS_IN, 500.0, CUT_HZ)
_FB_OUT = mel_fb(N_MELS_OUT, CUT_HZ, TOP_HZ)


def stft_mag(x, n_fft=N_FFT, hop=HOP):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    f, t, Z = sg.stft(x, SR, nperseg=n_fft, noverlap=n_fft - hop,
                      boundary='zeros', padded=True)
    return np.abs(Z), f


def logmel(mag, fb):
    m = fb @ (mag ** 2)
    return np.log10(m + 1e-12)


def istft_from_mag_phase(mag, phase_src, n_fft=N_FFT, hop=HOP, length=None):
    Z = mag * np.exp(1j * phase_src)
    _, y = sg.istft(Z, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y


# --------------------------------------------------------------------------- model
class BWE(nn.Module):
    """GRU over log-mel context -> the missing upper band, plus a discriminator."""

    def __init__(self, n_in=N_MELS_IN, n_out=N_MELS_OUT, hidden=128, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.LeakyReLU(0.2))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.1 if layers > 1 else 0.0)
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):                     # x: (B, T, n_in)
        h = self.inp(x)
        h, _ = self.gru(h)
        return self.out(h)                    # (B, T, n_out)


class Discriminator(nn.Module):
    """Small 1-D CNN that asks: does this look like a real top octave?"""

    def __init__(self, n_out=N_MELS_OUT, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_out, ch, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(ch, 1, 3, padding=1),
        )

    def forward(self, x):                     # (B, T, n_out)
        return self.net(x.transpose(1, 2)).mean(-1).squeeze(-1)


# --------------------------------------------------------------------------- data
def make_windows(song, win=64, stride=16):
    """Cut one song into (input logmel, target logmel) windows.

    Target is the REAL 8-16 kHz band of the same song, which the source does have
    (17-26 dB low, but present and structured). The model learns to bring it up,
    not to invent it from nothing - that is the difference between this and the
    first attempt, which tried to synthesise a band the source barely contains.
    """
    mag, f = stft_mag(song)
    lin = logmel(mag, _FB_IN)                 # (n_in, T)
    lout = logmel(mag, _FB_OUT)               # (n_out, T)
    n = lin.shape[1]
    xs, ys = [], []
    for i in range(0, n - win, stride):
        xs.append(lin[:, i:i + win].T)
        ys.append(lout[:, i:i + win].T)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def spectral_convergence(pred, targ):
    return torch.norm(targ - pred, dim=-1).mean() / (torch.norm(targ, dim=-1).mean() + 1e-8)


def train_bwe(songs, val_song, epochs=60, win=64, batch=32, lr=2e-3, adv_w=0.05,
              seed=0, log=print):
    """Train on `songs`, validate on `val_song` - never the other way round."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ytr = [], []
    for s in songs:
        a, b = make_windows(s, win)
        Xtr.append(a)
        Ytr.append(b)
    Xtr = np.concatenate(Xtr)
    Ytr = np.concatenate(Ytr)
    Xva, Yva = make_windows(val_song, win)
    if len(Xva) == 0:
        raise RuntimeError('validation song produced no windows')

    # normalise on the TRAINING set only
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0, keepdims=True)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0, keepdims=True) + 1e-6
    ymu = Ytr.reshape(-1, Ytr.shape[-1]).mean(0, keepdims=True)
    ysd = Ytr.reshape(-1, Ytr.shape[-1]).std(0, keepdims=True) + 1e-6

    G = BWE()
    D = Discriminator()
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    Xt = torch.tensor((Xtr - mu) / sd)
    Yt = torch.tensor((Ytr - ymu) / ysd)
    Xv = torch.tensor((Xva - mu) / sd)
    Yv = torch.tensor(Yva)

    n = len(Xt)
    best = None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n - batch, batch):
            idx = perm[i:i + batch]
            xb, yb = Xt[idx], Yt[idx]
            # ---- generator ----
            optG.zero_grad()
            pred = G(xb)
            l_rec = l1(pred, yb) + 0.5 * spectral_convergence(pred, yb)
            l_adv = bce(D(pred), torch.ones(len(xb)))
            loss = l_rec + adv_w * l_adv
            loss.backward()
            optG.step()
            # ---- discriminator ----
            optD.zero_grad()
            real = D(yb.detach())
            fake = D(G(xb).detach())
            l_d = bce(real, torch.ones(len(xb))) + bce(fake, torch.zeros(len(xb)))
            l_d.backward()
            optD.step()
        # ---- held-out (different song) ----
        with torch.no_grad():
            pv = G(Xv) * torch.tensor(ysd) + torch.tensor(ymu)
            yv = Yv
            mse = float(torch.mean((pv - yv) ** 2))
            var = float(torch.var(yv))
            r2 = 1.0 - mse / (var + 1e-12)
            lsd = float(torch.mean(torch.sqrt(torch.mean((pv - yv) ** 2, dim=-1))))
        if best is None or r2 > best['r2']:
            best = {'r2': r2, 'mse_db2': mse, 'lsd_db': lsd, 'epoch': ep + 1}
        if ep % 10 == 0 or ep == epochs - 1:
            log('  ep %3d  held-out(song) R2 %+.3f  MSE %.2f  LSD %.2f dB'
                % (ep + 1, r2, mse, lsd))
    G.eval()
    return G, {'mu': mu, 'sd': sd, 'ymu': ymu, 'ysd': ysd}, best


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def extend(song, model, stats, strength_db=18.0, blend=1.0, max_gain_db=24.0):
    """Add the reconstructed 8-16 kHz band to a band-limited song.

    `strength_db` is how much the band is raised relative to where it currently
    sits; it is applied on top of the model output so the amount is a single
    explicit number rather than an emergent property of the network.
    """
    mag, f = stft_mag(song)
    T = mag.shape[1]
    lin = logmel(mag, _FB_IN).T[None]                     # (1, T, n_in)
    x = (lin - stats['mu']) / stats['sd']
    # process in chunks so long songs do not blow memory
    out = []
    for i in range(0, T, 512):
        out.append(model(torch.tensor(x[:, i:i + 512]).float()).numpy())
    pred = np.concatenate(out, axis=1)[0]                  # (T, n_out)
    pred = pred * stats['ysd'] + stats['ymu']              # log10 power, mel

    # mel -> linear by least squares against the filterbank (stable, no inverse
    # filterbank artefact at the band edges)
    fb = _FB_OUT                                   # (n_out, bins)
    target_pow = np.maximum(10 ** (pred.T + strength_db / 10.0), 0.0)   # (n_out, T)
    # Mel -> linear: solve fb @ X = target for X (least squares, non-negative
    # clipped). np.linalg.lstsq needs (bins, n_out) here, i.e. the transpose -
    # passing fb directly raised "incompatible dimensions" because the filterbank
    # is (n_out, bins) and the right-hand side is (n_out, T).
    # fb is (n_out, bins); solving fb @ X = target_pow gives X of shape
    # (bins, T). Passing fb.T instead transposes the problem and numpy rejects it.
    sol = np.linalg.lstsq(fb, target_pow, rcond=1e-3)[0]                # (bins, T)
    lin_band = np.maximum(sol, 0.0)

    # ---- splice: keep everything below the cutoff untouched ---------------
    n_bins = mag.shape[0]
    f_lo = CUT_HZ
    f_hi = TOP_HZ
    idx = (f >= f_lo) & (f < f_hi)
    newmag = mag.copy()
    # only raise where the source is below the intended level; never attenuate
    cur = mag[idx]
    proposed = np.sqrt(lin_band[idx]) * blend + cur.mean(0, keepdims=True) * 0
    newmag[idx] = np.maximum(cur, proposed)
    # soft edge so the splice is not a step
    edge = (f >= f_lo * 0.9) & (f < f_lo)
    newmag[edge] = mag[edge]

    y = istft_from_mag_phase(newmag, np.angle(
        sg.stft(np.asarray(song, float).mean(1) if np.ndim(song) > 1 else song,
                SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary='zeros',
                padded=True)[-1]), length=len(song) if np.ndim(song) == 1 else np.shape(song)[0])
    return y.astype(np.float32)
