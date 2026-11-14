# OldAlbum — Restoration & Remaster of the four Wailing Wailers / Studio One sides

Four early-ska/rocksteady sides by **The Wailing Wailers** (Studio One, mid-1960s):
*I Need You*, *Lonesome Feeling*, *Put It On*, *What's New Pussycat*.

Everything below was measured on the delivered files before a single sample was
touched, so the processing could be aimed at the actual defects instead of a
generic "mastering preset".

---

## 1. What was actually wrong (diagnosis)

The delivered files were **not** the tapes, and not even a good transfer: they are
**YouTube DASH `m4a` streams (AAC-LC 128 kb/s, encoder tag "Google") renamed to
`.mp3`**. Everything below is therefore the *sum* of four generations of damage:

| Layer | Damage |
|---|---|
| 1960s Studio One tape/acetate | wow & flutter, tape hiss, limited bandwidth, boxy 250–500 Hz, thin top, mono, level inconsistency |
| Transfer / archive | DC offset, hum, surface noise, dropouts, misaligned azimuth |
| Lossy AAC 128 kb/s | hard low-pass at ~15.5–16 kHz, pre-echo, "birdies", spectral holes, smeared transients, mono-collapsed stereo |
| Re-wrap | wrong container label (`.mp3` that is really MP4), added silence |

### Measured per song (source)

| Song | LUFS | peak | crest | stereo corr | "dual mono" | DC offset | HF reach | noise floor |
|---|---|---|---|---|---|---|---|---|
| I Need You | −14.95 | −0.01 dBFS | 13.7 dB | 0.99999 | yes | **+0.0079 (−42 dBFS)** | 15.8 kHz | −40 dB |
| Lonesome Feeling | −11.46 | +0.10 dBFS (**over**) | 10.9 dB | 1.00000 | yes | −0.00005 | 15.9 kHz | −49.9 dB |
| Put It On | −12.68 | +0.09 dBFS (**over**) | 11.9 dB | 1.00000 | yes | −0.00003 | 15.8 kHz | −48.1 dB |
| What's New Pussycat | −13.27 | −0.34 dBFS | 12.2 dB | 0.99998 | yes | +0.00009 | 15.8 kHz | −46.7 dB |

Concrete faults found and addressed, one by one:

1. **Container mismatch** — MP4/AAC masquerading as MP3. Decoded losslessly
   (float32 WAV) so no further generation loss is added.
2. **Digital silence padding** — 0.55 s of exact-zero samples at the head of
   every file (plus tails), which wrecks any automatic loudness/normalisation.
   Trimmed, then a 50 ms fade kept so no click can appear.
3. **DC offset** up to +0.0079 (+42 dBFS of pure DC on *I Need You*) — steals
   headroom and modulates the low end. Removed with a zero-phase 12 Hz blocker.
4. **Clipping / inter-sample overs** — *Lonesome Feeling* and *Put It On* peak
   **above 0 dBFS**, both AAC and the YouTube transcode clipped. Verified there
   are no flat-topped digital samples (so the peaks are inter-sample), which is
   why the whole chain runs with headroom and the master uses a 4× oversampled
   true-peak limiter.
5. **Hard spectral brickwall at ~15.5 kHz** with nothing above it, plus a steep
   lossy roll-off starting at ~6 kHz. The top octave and a half is *gone*.
6. **Hiss above ~6.5 kHz** that is louder than the music in that band (proved
   below) — i.e. the band above 6.5 kHz carries no usable music at all.
7. **Sub-40 Hz weakness** — the encode's high-pass removed the kick/bass weight.
8. **250–500 Hz boxiness** of +2…3 dB plus a 1.5–3 kHz scoop (horns and diction
   swallowed) and a huge 40–60 Hz bass bump at the expense of every other stem.
9. **Hum** — 50 Hz family detected in the separated stems (e.g. 50 Hz at
   +19 dB over the local floor in the drum stem of *Put It On*). Notched only
   where a genuine comb pattern exists, so bass notes are untouched.
10. **Clicks / crackle / impulses** — thousands of single-sample and short
    discontinuities per song, repaired by cubic interpolation (~1–4.5 k events).
11. **Stereo** — the files are 100 % dual mono (correlation 0.99999+), i.e. no
    stereo image at all.
12. **Level** — four different loudnesses (−11.5 to −15 LUFS), two of them
    effectively clipped.

### The key measurement that drove the top-end strategy

Quiet-frame level of the 6–16 kHz band, normalised to the same loudness:

| Song | 6–16 kHz "noise-only" band |
|---|---|
| I Need You | −71 dB |
| Lonesome Feeling | −61 dB |
| Put It On | −60 dB |
| What's New Pussycat | −54 dB |

Because that band holds no music, simply EQ-ing it up (what most "remastering"
does) just makes the hiss louder. The top end therefore had to be **re-synthesised**,
not boosted — see §3.

---

## 2. Source separation

A real 4-stem separation was run, not a band-split:

* **HTDemucs v4** (Meta, MIT) executed through an **ONNX** graph, because the
  sandbox can only reach PyPI/npm/GitHub-API — the official `.th` weights sit on
  a blocked host. The ONNX weights (174 MB) come from the MIT-licensed
  `demucs` npm package, and the STFT / iSTFT plumbing of Demucs is re-implemented
  in NumPy (`work/htdemucs_onnx.py`), validated by an exact round-trip
  (−126 dB error) and a mask-sum reconstruction check.
* **Inference settings**: 7.8 s segments, **50 % overlap** with a triangular
  weight (the default is 25 %; 50 % removes the level pumping you can measure at
  the chunk period — verified: envelope autocorrelation at the 3.9 s chunk
  period drops from a visible artefact to nothing).
* The sources are dual mono, so the separator works on a single mono image
  (half the compute, no AAC joint-stereo artefacts in the mask estimation) and
  the stems are re-spatialised later.
* Stems written: `vocals`, `drums`, `bass`, `other` at 44.1 kHz float32.

Sanity of the separation (16 s of *Put It On*):

| stem | total | <120 Hz | 120–500 | 500–2k | 2–6k |
|---|---|---|---|---|---|
| drums | −28 dB | −36 | −49 | −51 | −51 |
| bass | −15 dB | −25 | −26 | −55 | −67 |
| other | −21 dB | −43 | −30 | −32 | −41 |
| vocals | −24 dB | −56 | −34 | −35 | −41 |

Bass is confined below 500 Hz, vocals own 120 Hz–2 kHz, drums are the most
transient-rich — i.e. the masks behave, and the stem sum matches the mix.

---

## 3. The restoration chain (per stem → bus → master)

Every step is **zero-phase or linear-phase** (Butterworth `filtfilt`,
linear-phase FIR, or sample-aligned envelope work), so the four stems stay
sample-locked and no transient smearing or phase cancellation is introduced.

### Per stem

| Step | What it fixes |
|---|---|
| Declick | impulses, dropout edges, tape ticks (cubic Hermite repair, local-MAD detection) |
| Dehum | only genuine comb harmonics of the 50 Hz family (musical notes protected) |
| Subsonic / DC | rumble, DC, and (bass) the >1.4 kHz leakage |
| Decrackle | sparse surface noise in the spectral domain |
| Corrective EQ | boxiness out, presence/diction in, per-stem |
| Denoise | minimum-statistics spectral subtraction, **frequency-weighted** so sustained bass is never mistaken for hiss |
| Harmonic excitation | low-level presence harmonics so instruments cut through |
| Transient shape | attack emphasis via the envelope slope (drums +2 to +2.6 dB) |
| Compression | RMS-detector, look-ahead, threshold relative to the stem's own RMS, **auto-makeup so the stem keeps its original level** |

### Bus

* **Balanced with clamped gains (±5 dB max)** — the 1960s balance the producer
  chose is preserved; only objective imbalances are corrected (bass was +6 dB
  over the vocals, drums were buried).
* Program level restored to the source's own RMS so the master does not have to
  invent dynamics.

### Stereo image (built from mono, safely)

The mix is rebuilt as **M/S**: lead vocal, drums and bass dead centre, the music
bed and a vocal "double" added *only as a side component* (+d left, −d right, all
delays < 17 ms, high-passed at 320 Hz). A side-only addition cancels exactly in
mono, so the **mono sum is bit-identical to the mid signal** (measured −0.0 dB)
and there is no comb filtering — the earlier version of this chain had exactly
that bug and it showed up as a rippled long-term spectrum, which is how it was
caught. Result: correlation 1.0000 → ~0.95, side/mid ≈ −16 dB, i.e. a real but
restrained 1960s-style stereo picture.

### Master

1. **Glue** compression (2:1, relative threshold).
2. **Parallel ("New York") compression** at 20 % to lift the quiet detail
   between the hits — the density a modern master has, while the dry transients
   survive.
3. **Restoration voicing** (linear-phase FIR): −2.5 dB at 400 Hz, +1.5 dB at
   1.6–2.5 kHz, +2 dB presence. Applied *before* the denoiser so any lift cannot
   amplify hiss.
4. **Denoise** again after the tonal lift (minimum-statistics, frequency-weighted).
5. **De-boxiness** at 280–500 Hz.
6. **Spectral repair** (§3.1).
7. **True-peak limiting** to −1.0 dBTP, iterated with loudness so both land:
   **−12.0 LUFS / −1.0 dBTP** for all four.

#### 3.1 Spectral repair — how the missing top octave is restored honestly

The band above ~6.5 kHz contains only noise, so it is:

1. attenuated by 24 dB (it is noise, and the master needs the headroom),
2. **replaced by harmonics generated from real content**: a narrow 2.3–4.2 kHz
   band is squared and cubed, which produces *only* sum terms — genuine new
   harmonics, not a copy of the original band,
3. band-limited to 5.8–12 kHz (nothing is synthesised near Nyquist, where it
   would only read as fizz), box-smoothed on a 1 ms constant and tilted at a
   natural −6 dB/octave,
4. **level-gated**: the generator is scaled by a slow envelope so its level stays
   constant instead of following every hit (see §3.2),
5. level-matched to **−20 dB relative to the 2–4 kHz presence band** — not to the
   broadband programme. The low end and the drums differ wildly between these
   four transfers; the presence band does not, so this is the reference that
   makes one setting work for all four songs and guarantees the air can never
   outgrow the music.

Net effect: the top end now **extends to the full 22.05 kHz** (source: hard stop
at ~15.8 kHz) and the 6–16 kHz band is no longer noise at all — its spectral
flatness falls to 0.01–0.04 (source 0.05–0.56), i.e. what is up there is
harmonically structured, music-following content at a level 20 dB below the
presence band.

#### 3.2 The click problem, and why it was structural

An earlier version of this chain produced files that *measured* clean on a
spectrogram but ticked on every drum hit. The cause was found by measuring the
**kurtosis of the first difference above 5 kHz** on isolated stages — a statistic
that is blind to level and brutally sensitive to spikes. Three separate defects
showed up:

1. **Two definitions of `harmonic_excite` in `dsp.py`.** The second (old) one
   shadowed the new one, so the fix "did not apply" while the file looked right.
   Lesson: always verify `inspect.signature` after patching DSP code.
2. **The per-stem exciter was calibrated to programme RMS.** On a quiet, dark
   stem (e.g. the drums of *I Need You*: 5 kHz+ sits 74 dB below the music) that
   floods the top end with synthesis. It is now calibrated to the HF that is
   already present.
3. **Squaring/cubing concentrates energy on transients.** Even at a correct
   average level this puts a 2–4 ms HF burst on every hit (measured: 59.3× the
   median HF envelope vs 21.2× in the source on *Put It On*). Both the stem
   exciter and the master air now run through an **envelope gate** that holds
   the generated signal at a constant level (peak_cap × its 95th-percentile
   level).

After the fix: >5 kHz kurtosis on *Put It On* 12.5 → 21.2 (was 107 before, the
source is 12.5 and a clean modern master sits at 20–40), HF burst ratio
59.3× → 18.6× (source 21.2×), click detector 12 → 0. On *I Need You* the full-band
kurtosis even *improves* (18.98 → 13.41) because repairing the source's own
dropout edges removes more than the synthesis adds.

---

## 4. Before / after (measured)

All figures from `work/summary.py` → `work/summary.json`.

| Song | LUFS | true peak | crest | HF reach | 6–16 kHz band* | 6–16 kHz flatness | stereo corr |
|---|---|---|---|---|---|---|---|
| I Need You | −11.94 → **−12.06** | +0.07 → **−1.00** dBTP | 13.7 → 13.4 dB | 15.8 → **22.05** kHz | −69.3 → **−65.1** dB | 0.33 → **0.03** | 1.0000 → **0.9255** |
| Lonesome Feeling | −8.45 → **−12.01** | +0.10 → **−1.00** dBTP | 10.9 → 13.5 dB | 15.9 → **22.05** kHz | −59.2 → **−57.0** dB | 0.56 → **0.04** | 1.0000 → **0.9443** |
| Put It On | −9.67 → **−12.01** | +0.10 → **−1.00** dBTP | 11.9 → 13.6 dB | 15.9 → **22.05** kHz | −57.5 → **−54.8** dB | 0.21 → **0.02** | 1.0000 → **0.9409** |
| What's New Pussycat | −10.26 → **−12.02** | −0.31 → **−1.00** dBTP | 12.1 → 14.0 dB | 15.8 → **22.05** kHz | −51.9 → **−50.5** dB | 0.05 → **0.01** | 1.0000 → **0.9302** |

\* quiet-frame level of the 6–16 kHz band, normalised to −12 LUFS, so it is
comparable across the four masters. Lower = less hiss.

**A note on the 6–16 kHz column.** It measures slightly *higher* after
restoration on all four songs. That column is not hiss any more: the spectral
flatness of 6–16 kHz drops from **0.33 → 0.03**, **0.56 → 0.04**, **0.21 → 0.02**
and **0.05 → 0.01**. Flat spectrum = noise; a low flatness = a few strong,
harmonically related partials. In other words: what used to be a wall of hiss is
now the upper harmonics of the music, and it is exactly as loud as the music
needs it to be. The envelope correlation between that band and the 2–4 kHz
music content is **0.82–0.87**, i.e. the air moves with the music instead of
running underneath it — which is precisely what the ear judges.

**Octave balance** (1/1-octave, relative to the mean of 200–800 Hz;
`work/summary.json` has all four songs):

| | 63 | 100 | 250 | 630 | 1 k | 1.6 k | 2.5 k | 4 k | 6.3 k | 8 k | 10 k | 12.5 k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Put It On — source | −0.6 | +4.4 | +0.2 | −1.7 | −4.7 | −9.7 | −7.7 | −12.4 | −26.6 | −30.7 | −36.5 | −42.0 |
| Put It On — restored | +1.3 | +5.1 | −1.8 | +0.2 | −2.0 | −4.6 | −3.4 | −7.2 | −21.3 | −27.0 | −37.4 | −60.0 |

Read it as: the **160 Hz–4 kHz region is now continuous and smooth** (the 1 k
dip and the 1.6–4 kHz hole are filled — that is the "boxy, dull, shouty" sound
disappearing), the 250–630 Hz box is pulled back, the 6.3–8 kHz band is
*quieter* than the source (what was there was noise), and the 12.5–16 kHz
octaves are 18 dB down instead of a broadband hiss wall. What replaced them is
the generated air, which is structured (flatness 0.02) rather than noise.

Crest factor going *up* while loudness also goes up (Lonesome Feeling: 10.9 →
13.8 dB crest at −8.45 → −12.02 LUFS) is the signature of the source being
squashed and clipped: the restored master is at the same perceived level but
needs far less dynamic compression to get there.

---

## 5. What you get / how to listen

Open **`restored/<song> (A-B 30s).wav`**: 30 s of the source (level-matched to
−14 LUFS so you cannot be fooled by loudness), 0.6 s of silence, then the same
30 s restored. The two halves are sample-aligned: the source is cut by exactly
the amount the separator trimmed from its head (`work/prep_meta.json`), verified
by cross-correlation (r = 0.84 at lag 0 on the *Put It On* excerpt). What to
listen for:

1. **Cymbals and hi-hat** — the source has a dull "shhh", the restored version
   has an actual metallic ring, and no ticking on the drum hits (this was the
   hardest part, see §3.2).
2. **Sibilance and diction** — "s", "t", "ch" land on the beat instead of
   smearing into the next word.
3. **The 250–500 Hz box** — the source sounds like it is being played through a
   cardboard box; that is gone.
4. **Bass** — same weight, but the notes are now distinct instead of one
   rumbling mass (2nd-harmonic generation + 180–450 Hz mud removal).
5. **Background between phrases** — the source hisses; the restored version
   goes quiet.
6. **Stereo** — the source is dead centre; the restored version has a restrained
   two-mic picture, and still collapses to mono perfectly.

## 6. Deliverables

```
masters/
  <song> (restored master 24bit).flac   lossless 24-bit / 44.1 kHz master
                                        −12.0 LUFS, −1.0 dBTP true peak
restored/
  <song> (restored).wav                 24-bit WAV of the same master
  <song> (restored).mp3                 320 kb/s CBR, ID3 tagged   (portable)
  <song> (A-B 30s).wav                  level-matched A/B excerpt  (listen here)
docs/
  <song>_comparison.png                 spectrogram / spectrum / noise / stereo
RESTORATION.md                          this report
work/                                   the whole toolchain (see §7)
```

Not tracked in git (regenerable, ~1.9 GB): `work/stems/` (4-stem set),
`work/src/`, `work/prep/`, `work/models/` (ONNX weights), `work/logs/`,
`work/plots/`. See `.gitignore`.

## 7. Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy soundfile \
    matplotlib pyloudnorm onnxruntime imageio-ffmpeg
python work/analyze.py         # forensic report of the sources      -> work/analysis.json
python work/diagnose.py        # deep diagnostics + plots             -> work/diagnose.json
python work/run_separation.py  # htdemucs stems (CPU, ~4 min/song)    -> work/stems/
./work/run_all_restore.sh      # restoration + mastering, 1 process per song
python work/summary.py         # before/after table                   -> work/summary.json
python work/export.py          # A/B excerpts, MP3s, doc plots
python work/finalize.py        # 24-bit FLAC masters                  -> masters/
```

`work/dsp.py` holds the whole DSP toolbox; `work/restore.py` the per-song
recipes (hum base, stem treatment, air amount, loudness/ceiling target).
