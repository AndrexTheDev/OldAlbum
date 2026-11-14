#!/usr/bin/env python3
"""
Stage 1: prepare + stem-separate all four source files.

Prep is deliberately minimal (DC removal + leading/trailing digital silence trim)
so the separator sees a clean, unbiased mix. Everything else happens per stem.
"""
import os, sys, glob, json, time
import numpy as np
import soundfile as sf
sys.path.insert(0, '/home/user/OldAlbum/work')

SR = 44100
ROOT = '/home/user/OldAlbum'
SRC = f'{ROOT}/work/src'
PREP = f'{ROOT}/work/prep'
STEMS = f'{ROOT}/work/stems'


def trim_silence(x, sr, thresh_db=-72.0, pad=0.05):
    env = np.abs(x).max(axis=1)
    thr = 10 ** (thresh_db / 20.0)
    nz = np.flatnonzero(env > thr)
    if len(nz) == 0:
        return x, 0
    a = max(0, nz[0] - int(pad * sr))
    b = min(len(x), nz[-1] + int(pad * sr))
    return x[a:b], a


def dc_block(x, sr, fc=12.0):
    """2nd-order Butterworth high-pass, zero-phase, to kill DC + sub-rumble."""
    from scipy import signal
    b, a = signal.butter(2, fc / (sr / 2), btype='high')
    return signal.filtfilt(b, a, x, axis=0)


def main():
    os.makedirs(PREP, exist_ok=True)
    os.makedirs(STEMS, exist_ok=True)
    from htdemucs_onnx import HTDemucs, separate

    meta = {}
    t0 = time.time()
    for p in sorted(glob.glob(f'{SRC}/*.wav')):
        tag = os.path.basename(p)[:-4]
        print(f'\n=== {tag} ===', flush=True)
        x, sr = sf.read(p, always_2d=True, dtype='float32')
        assert sr == SR

        dc_orig = x.mean(axis=0).copy()
        x, off = trim_silence(x, sr)
        x = dc_block(x, sr).astype(np.float32)

        # source is dual-mono: separate a single mono image (halves memory,
        # removes AAC joint-stereo artefacts from the mask estimation)
        mono = x.mean(axis=1)
        prep = np.stack([mono, mono], axis=1)
        prep_path = f'{PREP}/{tag}.wav'
        sf.write(prep_path, prep, SR, subtype='FLOAT')

        meta[tag] = {
            'trim_offset_samples': int(off),
            'trim_offset_s': round(off / SR, 4),
            'dc_offset_removed': [round(float(v), 7) for v in dc_orig],
            'duration_s': round(len(x) / SR, 3),
            'samples': int(len(x)),
        }
        print(f'  trimmed {off} samples ({off/SR:.3f}s), dc={dc_orig}, dur={len(x)/SR:.2f}s', flush=True)

        # stem separation on the mono image
        stem_dir = f'{STEMS}/{tag}'
        if os.path.exists(f'{stem_dir}/vocals.wav'):
            print('  stems exist, skipping', flush=True)
            continue
        t = time.time()

        def prog(a, b, _t=tag):
            print(f'    chunk {a}/{b} ({time.time()-t:.0f}s)', flush=True)

        separate(HTDemucs(f'{ROOT}/work/models/htdemucs.onnx'), prep_path, stem_dir,
                 overlap=0.5, progress=prog)
        print(f'  separation done in {time.time()-t:.0f}s', flush=True)

    json.dump(meta, open(f'{ROOT}/work/prep_meta.json', 'w'), indent=2)
    print(f'\nALL DONE in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
