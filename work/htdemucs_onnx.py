#!/usr/bin/env python3
"""
Faithful numpy port of bakkot/demucs-js (HTDemucs v4, ONNX) inference.

The ONNX graph exposes only the two "runnable" branches; the STFT / iSTFT and
the hybrid-spectrogram plumbing are re-implemented here exactly as in
dsp.js + apply.js so that the output matches the reference implementation.
"""
import numpy as np

NFFT = 4096
HOP = 1024
SEGMENT = 7.8
SR = 44100
TRAIN_LEN = int(np.floor(SEGMENT * SR))  # 343980
SOURCES = ['drums', 'bass', 'other', 'vocals']


# ----------------------------------------------------------------------------- helpers
def pad_reflect(x, left, right):
    """np.pad reflect, but matching the JS fallback for small arrays."""
    if left == 0 and right == 0:
        return x
    n = x.shape[-1]
    maxpad = max(left, right)
    if n <= maxpad:
        extra = maxpad - n + 1
        er = min(right, extra)
        el = extra - er
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(el, er)], mode='constant')
        left -= el
        right -= er
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode='reflect')


def hann_periodic(n):
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


_WIN = hann_periodic(NFFT)


def _frames(x, frame_len, hop):
    """View (..., L) -> (..., nframes, frame_len) strided frames."""
    n = x.shape[-1]
    nframes = (n - frame_len) // hop + 1
    idx = np.arange(nframes) * hop
    view = np.lib.stride_tricks.sliding_window_view(x, frame_len, axis=-1)
    # sliding_window_view moves the window axis to the end
    return view[..., idx, :]


def stft(x, nfft=NFFT, hop=HOP, center=True):
    """torch.stft(normalized=True, periodic hann, reflect padded) -> (..., F, T)."""
    if center:
        x = pad_reflect(x, nfft // 2, nfft // 2)
    fr = _frames(x, nfft, hop) * _WIN[:nfft]
    fr = fr / np.sqrt(nfft)
    spec = np.fft.rfft(fr, n=nfft, axis=-1)          # (..., T, F)
    return np.ascontiguousarray(np.swapaxes(spec, -1, -2))   # (..., F, T)


def istft(Z, nfft=None, hop=HOP, length=None, center=True):
    """
    torch.istft with sqrt-hann norm. Memory-lean: overlaps frame by frame
    instead of materialising a padded spectrum.
    Z: complex (..., F, T) with F = nfft/2+1
    """
    F, T = Z.shape[-2], Z.shape[-1]
    if nfft is None:
        nfft = 2 * F - 2
    assert 2 * F - 2 == nfft, (F, nfft)
    if length is None:
        length = nfft + (T - 1) * hop - (nfft if center else 0)
    win = hann_periodic(nfft)
    lead = Z.shape[:-2]
    N = int(np.prod(lead)) if lead else 1
    Zf = Z.reshape(N, F, T)
    out = np.zeros((N, length), dtype=np.float32)
    wsum = np.zeros(length, dtype=np.float64)
    norm = np.sqrt(nfft)
    pad = nfft // 2 if center else 0
    win_sq = (win ** 2).astype(np.float32)
    for t in range(T):
        frame = np.fft.irfft(Zf[:, :, t], n=nfft, axis=-1).astype(np.float32) * win.astype(np.float32)
        frame *= norm
        start = t * hop - pad
        lo = max(0, start)
        hi = min(length, start + nfft)
        if hi <= lo:
            continue
        a, b = lo - start, hi - start
        out[:, lo:hi] += frame[:, a:b]
        if t == 0:
            wsum[lo:hi] = win_sq[a:b]
        elif wsum[lo:hi].max() == 0:
            wsum[lo:hi] = win_sq[a:b]
        else:
            wsum[lo:hi] += win_sq[a:b]
    ok = wsum > 1e-8
    out[:, ok] /= wsum[ok].astype(np.float32)
    return out.reshape(lead + (length,))


def magnitude(z):
    """complex (B,C,F,T) -> (B, 2C, F, T) interleaved real/imag."""
    return np.concatenate([z.real, z.imag], axis=1).astype(np.float32)


def mask_from_out(out):
    """(B,S,2C,F,T) interleaved -> complex (B,S,C,F,T)."""
    re = out[:, :, 0::2]
    im = out[:, :, 1::2]
    return re + 1j * im


# ----------------------------------------------------------------------------- spec/ispec
def spec(x):
    """htdemucs.hdemucs._spec: (B, C, L) -> complex (B, C, 2048, le)."""
    hl = 1024
    le = int(np.ceil(x.shape[-1] / hl))
    pad = (hl // 2) * 3
    pr = pad + le * hl - x.shape[-1]
    xp = pad_reflect(x, pad, pr)
    z = stft(xp, 4096, hl)                    # (B, C, 2049, T)
    z = z[..., :-1, :]                        # drop last freq bin
    z = z[..., 2:2 + le]
    return np.asarray(z, dtype=np.complex64)


def ispec(z, length):
    """
    htdemucs.hdemucs._ispec, memory-lean.
    z: complex (..., 2048, T) -> (..., length) float32
    Adds the +1 zero frequency bin and the 2/2 time padding implicitly.
    """
    hl = 1024
    nF, T = z.shape[-2], z.shape[-1]
    nfft = 2 * (nF + 1) - 2          # 2049 bins -> 4096
    pad = (hl // 2) * 3
    le = hl * int(np.ceil(length / hl)) + 2 * pad
    lead = z.shape[:-2]
    N = int(np.prod(lead)) if lead else 1
    Zf = z.reshape(N, nF, T)
    out = np.zeros((N, le), dtype=np.float32)
    wsum = np.zeros(le, dtype=np.float64)
    win = hann_periodic(nfft).astype(np.float32)
    norm = np.sqrt(nfft)
    win_sq = (win ** 2)
    # padded time index: 2 zeros, then T frames, then 2 zeros
    for t in range(2, 2 + T):
        spec_t = np.zeros((N, nF + 1), dtype=np.complex64)
        spec_t[:, :nF] = Zf[:, :, t - 2]
        frame = np.fft.irfft(spec_t, n=nfft, axis=-1).astype(np.float32) * win * norm
        start = t * hl - (nfft // 2)     # istft(center=True)
        lo = max(0, start)
        hi = min(le, start + nfft)
        if hi <= lo:
            continue
        a, b = lo - start, hi - start
        out[:, lo:hi] += frame[:, a:b]
        wsum[lo:hi] += win_sq[a:b]
    ok = wsum > 1e-8
    out[:, ok] /= wsum[ok].astype(np.float32)
    x = out.reshape(lead + (le,))
    return np.ascontiguousarray(x[..., pad:pad + length])


# ----------------------------------------------------------------------------- model wrapper
class HTDemucs:
    def __init__(self, onnx_path, threads=2):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = False    # freed activation buffers must go back to the OS
        so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
        ins = self.sess.get_inputs()
        self.in_mix = ins[0].name
        self.in_mag = ins[1].name
        outs = self.sess.get_outputs()
        self.out_mask = outs[0].name
        self.out_time = outs[1].name

    def forward(self, mix, magspec):
        r = self.sess.run([self.out_mask, self.out_time],
                          {self.in_mix: mix.astype(np.float32),
                           self.in_mag: magspec.astype(np.float32)})
        return r[0], r[1]

    def infer_chunk(self, chunk):
        """chunk: (2, TRAIN_LEN) float32 -> stems (4, 2, TRAIN_LEN)."""
        mix = chunk[None]                       # (1,2,L)
        z = spec(mix)
        mag = magnitude(z)
        out_mask, out_time = self.forward(mix, mag)
        zout = mask_from_out(out_mask)           # (1,4,2,2048,T)
        time_branch = ispec(zout, TRAIN_LEN)     # (1,4,2,L)
        out = out_time + time_branch
        return out[0].astype(np.float32)


def _trim():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def separate(model, src, out_dir, overlap=0.5, progress=None, names=SOURCES):
    """
    Stream a wav file through htdemucs, writing one wav per stem.

    src     : path to a 2-channel 44.1 kHz wav
    out_dir : directory for the stem wavs (written as float32 wavs)
    Memory-lean: only one 7.8 s chunk is resident at a time. NOTE: we must NOT
    gc.collect()/malloc_trim() inside the loop - that frees the ONNX arena
    buffers the next chunk reuses, doubling peak RSS.
    """
    import os, soundfile as sf
    info = sf.info(src)
    n, ch = info.frames, info.channels
    assert info.samplerate == SR, info.samplerate
    seg = TRAIN_LEN
    stride = int(np.floor((1 - overlap) * seg))

    w = np.zeros(seg, dtype=np.float32)
    half = seg // 2 + 1
    w[:half] = np.arange(half) + 1
    w[half:] = seg - np.arange(half, seg)
    w /= w.max()

    acc = np.zeros((len(names), ch, n), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)

    offsets = []
    off = 0
    while True:
        offsets.append(off)
        if off + stride >= n:
            break
        off += stride
    total = len(offsets)

    fh = sf.SoundFile(src, 'r')
    try:
        for i, off in enumerate(offsets):
            c_len = min(seg, n - off)
            delta = seg - c_len
            start = off - delta // 2
            end = start + seg
            cs, ce = max(0, start), min(n, end)
            chunk = np.zeros((ch, seg), dtype=np.float32)
            fh.seek(cs)
            data = fh.read(ce - cs, dtype='float32', always_2d=True)
            chunk[:, cs - start:cs - start + (ce - cs)] = data.T
            del data
            stems = model.infer_chunk(chunk)
            del chunk
            lo = cs - start
            hi = lo + (ce - cs)
            wt = w[lo:hi]
            acc[:, :, cs:ce] += stems[:, :, lo:hi] * wt[None, None, :]
            wsum[cs:ce] += wt
            del stems
            if progress:
                progress(i + 1, total)
    finally:
        fh.close()

    wsum[wsum == 0] = 1.0
    acc /= wsum[None, None, :]
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for k, name in enumerate(names):
        path = os.path.join(out_dir, f'{name}.wav')
        sf.write(path, acc[k].T, SR, subtype='FLOAT')
        paths[name] = path
    return paths
