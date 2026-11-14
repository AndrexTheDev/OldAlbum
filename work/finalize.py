#!/usr/bin/env python3
"""
Stage 6: package the deliverables.

* restored/  : MP3 320 (portable) + 16-bit WAV (legacy/CD)
* masters/   : 24-bit masters as FLAC (lossless, keeps full quality)
* docs/      : diagnostic + comparison plots, reports
* The stem set is 968 MB and the intermediate work files ~600 MB, so they stay
  out of git (they are regenerable with work/run_separation.py).
"""
import os, sys, glob, subprocess, shutil, json

ROOT = '/home/user/OldAlbum'
FF = f'{ROOT}/work/ffmpeg'


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(f'{ROOT}/masters', exist_ok=True)
    for wav in sorted(glob.glob(f'{ROOT}/restored/*(restored).wav')):
        tag = os.path.basename(wav).replace(' (restored).wav', '')
        flac = f'{ROOT}/masters/{tag} (restored master 24bit).flac'
        run([FF, '-hide_banner', '-loglevel', 'error', '-y', '-i', wav,
             '-c:a', 'flac', '-compression_level', '8', '-sample_fmt', 's32',
             '-map_metadata', '-1',
             '-metadata', f'title={tag} (restored)',
             '-metadata', 'artist=The Wailing Wailers',
             '-metadata', 'album=Studio One (Restored)',
             flac])
        print(f'  flac: {os.path.basename(flac)}  '
              f'{os.path.getsize(flac)/1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
