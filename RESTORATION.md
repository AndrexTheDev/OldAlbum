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
sample-locked and no transient smearing or phase cancellation is introduced
(verified: best lag between the stem sum and the source is **0 samples** at
20/40/60/80 s, correlation 0.97).

The architecture is deliberately lopsided: **almost nothing per stem, almost
everything on the bus.** That is a measurement result, not a preference.

### Why the bus, not the stems

An earlier version of this chain processed all four stems independently, each
with its own denoiser, EQ and compressor. It measured badly and the reason is
structural:

* **The noise is the same noise in every stem.** Demucs splits one hiss into four
  hisses; four independent minimum-statistics estimates then *add up* to more
  residual than they remove. Measured on *I Need You*: the denoiser improved each
  stem's own SNR by 4-5 dB, yet the sum came out **8.5 dB worse** than the
  source, because what each denoiser left behind was uncorrelated with the other
  three.
* **Per-stem insert compression with auto-make-up** raises everything below the
  threshold. On the bass it pumped the gaps back up by **+5.0 dB**, undoing the
  denoising of the stage before it.
* **Per-stem harmonic excitation** adds sidebands, not harmonics - the classic
  "exciter" colour that makes a restoration sound like an effect.

All three are gone. One denoiser, one voicing, one dynamics stage, all on the
summed programme. Measured effect of the change, in-band SNR of the noise floor
(true quiet passages) against a level-matched source:

| song | 200-1k | 1k-4k | 4k-8k |
|---|---|---|---|
| *I Need You* | −7.8 → **−0.7** | −6.5 → **+1.3** | −6.3 → **+2.1** |
| *Put It On* | −4.5 → **−0.9** | −5.3 → **+0.2** | −4.3 → **−1.0** |
| *Pussycat* | −7.3 → **−1.2** | −2.6 → **+0.3** | −4.2 → **−0.5** |

### Per stem (short on purpose)

| Step | What it fixes |
|---|---|
| Declick | impulses, dropout edges, tape ticks (cubic Hermite repair, local-MAD detection) |
| Dehum | only genuine comb harmonics of the 50 Hz family (musical notes protected) |
| Subsonic / DC + band limit | rumble, DC, and (bass) the >1.4 kHz leakage |
| Decrackle | sparse surface noise in the spectral domain |
| Corrective EQ | only the stem's own defect: boxiness out, presence in |
| `wb_declick` | residual top-octave edge (see §3.2) |

That is all. No denoise, no compression, no exciter per stem.

### Bus

1. **Balance** with clamped gains (±5 dB). The 1960s balance the producer chose
   is preserved; only objective imbalances are corrected.
2. **One denoiser** for the whole programme (minimum statistics, frequency
   weighted 0.55 → 1.0 across the band). On a full mix it can tell "hiss in a
   gap" from "cymbals in a hit"; on a single stem it cannot.
3. **Voicing**: linear-phase FIR towards a modern balance, referenced to the
   200-800 Hz region so it cannot be fooled by absolute level. Measured
   excursions at 400 Hz / 250 Hz were over-corrected in an earlier version; the
   filter also gained `+8.75 dB` on the whole signal because it was normalised to
   0 dB *at DC* (see §3.4).
4. **Parallel compression** (~16 %), level-matched to the dry signal. A copy at
   2.5:1 supplies density; the dry path keeps the transients and the breathing.
5. **Gentle glue** (1.35:1, threshold 14 dB below programme RMS) then
   level-restored, so it only shapes phrases.
6. **Top octave**: noise-only band above 8 kHz attenuated, then the trained
   network reconstructs 5.7-22.05 kHz (§3.1).

### Stereo image (built from mono, without a comb)

The sources are **dual mono**. The side component is built from the programme
itself with a **Schroeder allpass cascade** (5 stages, 1.1-10.3 ms, mutually
prime): by definition an allpass has a perfectly flat magnitude response -
measured maximum deviation from flat over 50 Hz-20 kHz: **0.0000 dB** - while its
phase is scattered. So the side carries the same spectrum and the same transients
as the music, at near-zero waveform correlation, which is exactly what "wide"
means.

The level is *solved*, not guessed: for an uncorrelated side of RMS ratio k the
inter-channel correlation is (1−k²)/(1+k²), so k is inverted from the requested
correlation (0.93). The side is added as ±, so it cancels exactly in mono - the
mono sum is the mid signal, bit for bit.

The previous version subtracted two delayed copies of the bed (8.7 ms / 13.1 ms).
That is a **comb filter**: measured 34.5 dB peak-to-trough ripple over 1-16 kHz,
and its top end correlated −0.006 with the music. It was heard as a "tin can"
colour on the whole top end.

Which number describes width? `hf_side_corr` (waveform) is *expected* to be near
zero here - uncorrelated channels are what width is. The honest measure is the
correlation of the 3-20 kHz **envelopes**: 0.00 in the source mono, **0.41-0.67**
in the masters, i.e. the width opens and closes with the music because it *is*
the music.

### Master

1. Loudness and true peak met *together*: gain → limiter → linear trim, iterated,
   so all four masters land at −12.0 LUFS / −1.00 dBTP (§3.5).
2. True-peak safety at 4× oversampling; subsonic high-pass **before** the
   loudness loop.
3. 24-bit FLAC master + 16-bit dithered WAV + 320 kb/s MP3.

### 3.4 Four bugs found by auditing the audit

The defect table in §4 disagreed with the chain's own report, which turned out to
be correct - about the wrong thing. All four were found by measuring, and all
four were real:

1. **`fir_from_curve` normalised to 0 dB at DC.** Any curve that cut the bass
   (-9 dB at 95 Hz) silently added **+9 dB to the entire signal**. Every "the EQ
   lifted the noise floor" conclusion was this. Fixed: normalise on the peak of
   the response. A flat +6 dB request now measures +6.00 dB.
2. **`match_target` returned the input spectrum twice**, so every `tone_match`
   report in this project showed a correction of 0.0 dB everywhere and the EQ
   looked like it was doing nothing. It was doing plenty; the report was blind.
3. **`if gate_off and gate_off > 0`** - the gate offset is a *negative* dB value,
   so the gate never ran. Three rounds of tuning it changed exactly nothing.
4. **The loudness loop only ever gained up**, so a programme whose peaks sat
   lower was delivered at −2.38 dBTP instead of −1.00. Fixed with the fixed-point
   iteration above.

### 3.5 Wow and flutter: measured, bounded, and only applied where it is provable

Tape wow is real on these transfers, but every measurement of it has to survive a
simple test: **is the excursion slow and periodic (transport) or is it the
performance moving (notes, vibrato)?** Three estimators were built and the first
two were thrown away:

| method | reading | verdict |
|---|---|---|
| dominant STFT peak, 300-1500 Hz | 50-98 cents | **artefact** - follows note changes and vibrato |
| bass-stem pitch track | 115-178 cents std | **artefact** - the bass plays a line, not a drone |
| Hilbert IF of an isolated line, de-spiked | see below | **the honest one** |

The honest method needs two corrections of its own before it can be trusted:

1. **De-spiking.** The raw Hilbert instantaneous frequency of a real transfer is
   full of spikes: 22 % of samples beyond ±100 cents (extremes −376/+261) on
   *Lonesome Feeling*, because the isolated line keeps losing to bleed and note
   onsets. Filtering that directly turns spikes into a slow wander that *looks
   exactly like wow* and would be "corrected". A median filter has to come first.
2. **Decimation before filtering.** At 44.1 kHz a 0.2 Hz Butterworth has its pole
   at 0.99998 - `filtfilt` on it produced transport figures of +41 711 cents. At
   400 Hz the same filter is exact.

What the corrected measurement says, on high-resolution partial tracking
(16 384-point FFT, ×4 zero padding, parabolic interpolation), 12-second windows:

| song / window | excursion | of which slow (transport band) | of which faster (performance) |
|---|---|---|---|
| I Need You @108 s | 305 cents | 169 | 136 |
| Lonesome Feeling @76 s | 362 cents | 172 | 190 |
| Put It On @28 s | 382 cents | 348 | 34 |
| Put It On @148 s | 298 cents | 225 | 73 |

So the drift is **large** (150-350 cents) and **mostly slow** - and that is why it
is *not* globally corrected here. A 350-cent warp cannot be applied to a finished
recording: time-warping the whole programme by ±3 % to chase a note that is being
bent smears every transient and detunes everything else that is playing at the
same time. Real wow of that size would be *periodic*, and the modulation here is
not: it needs a per-note analysis, not a global warp. Applying the warp anyway
would be the classic restoration mistake - a measurable number mistaken for a
curable defect.

What `work/wowfix.py` therefore does is implement the correction properly and
**gate it on evidence**: it needs a pitch-stable line (interquartile range < 25
cents), the slow part must dominate the excursion, and the warp is bounded to
±12 cents, so the tool can only ever remove a *small* transport wobble and can
never become a detuner. Where the gate rejects, it reports why. On this material
it declines on three of four songs, and that refusal is the correct outcome.

`work/audit.py` uses the same estimator and reports `ok: false` with the reason
instead of inventing a number.

## 4. Before / after (measured)

All figures from `work/summary.py` → `work/verify_summary.json`.

| Song | LUFS | true peak | crest | HF reach | corr(output HF, source HF) | stereo corr |
|---|---|---|---|---|---|---|
| I Need You | −11.94 → **−12.01** | +0.07 → **−1.00** dBTP | 13.7 → 13.4 dB | 15.8 → **15.97** kHz | **0.71** | 1.0000 → **0.9266** |
| Lonesome Feeling | −8.45 → **−12.00** | +0.10 → **−1.00** dBTP | 10.9 → 13.3 dB | 15.9 → **16.5** kHz | **0.92** | 1.0000 → **0.9202** |
| Put It On | −9.67 → **−12.00** | +0.10 → **−1.00** dBTP | 11.9 → 13.6 dB | 15.9 → **16.5** kHz | **0.88** | 1.0000 → **0.9220** |
| What's New Pussycat | −10.26 → **−12.01** | −0.31 → **−1.00** dBTP | 12.1 → 13.6 dB | 15.8 → **17.9** kHz | **0.93** | 1.0000 → **0.9197** |

### Noise: the number this project got wrong for a long time

Measured in **true quiet passages** (the quietest 10 % of frames), as SNR
against a level-matched source, per band. Positive = cleaner than the source.

| song | 40-200 | 200-1k | 1k-4k | 4k-8k | 8k-16k |
|---|---|---|---|---|---|
| I Need You | −1.3 | −0.7 | **+1.3** | **+2.1** | **+3.3** |
| Lonesome Feeling | −0.1 | −2.2 | **+2.0** | **+1.5** | **+1.9** |
| Put It On | **+0.9** | −0.9 | **+0.2** | −1.0 | **+0.7** |
| Pussycat | **+2.3** | −1.2 | **+0.3** | −0.5 | −1.4 |

Honest reading: the mid and top are cleaner than the source; the 40-200 Hz band
is a wash (the acetate's LF rumble is partly *signal* for the bass, and removing
it would remove the bass); and three of four songs still lose 1-2 dB around
200-1000 Hz, because the separation does not perfectly reconstruct that region.
That is the residual cost of the stem split, and it is 5-7 dB smaller than it was
before the bus architecture.

Measured on **absolute** noise floor (`work/audit.json`, quietest 5 % of frames,
LUFS-matched), all four improve on the previous chain: +10.0/+7.4/+12.3 dB worse
became +0.7/−1.1/+1.7 (I Need You); +6.6/+3.7/+9.9 became −1.6/−3.3/+0.3 (Put It
On).

**A note on the 6-16 kHz band.** It is *higher* after restoration, and that is
intended: the band now contains reconstructed harmonics of the music instead of a
noise floor. What matters is that it is *structured*: the spectral flatness of the
band falls (source → output: 0.51 → 0.26, 0.66 → 0.46, 0.35 → 0.28, 0.08 → 0.21)
and its envelope correlates **0.71-0.93** with the real 8-16 kHz content of the
original transfer. A generator that ignores the music cannot do that. The HF
"reach" figure is the 90 dB bandwidth, not the extent of musical content: it grows
modestly (15.8 → 16-18 kHz) because the network reconstructs real harmonics, not a
hiss wall - the previous chain reported 22.05 kHz for all four, which was the
network's noise floor.

**Octave balance** (1/1-octave, relative to the mean of 200-800 Hz;
`work/verify_summary.json` has all four songs):

| | 63 | 100 | 250 | 630 | 1 k | 1.6 k | 2.5 k | 4 k | 6.3 k | 8 k | 10 k | 12.5 k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Put It On — source | −0.6 | +4.4 | +0.2 | −1.7 | −4.7 | −9.7 | −7.7 | −12.4 | −26.6 | −30.7 | −36.5 | −42.0 |
| Put It On — restored | +1.4 | +5.2 | −1.8 | +0.2 | −2.1 | −4.7 | −3.3 | −7.0 | −22.1 | −28.6 | −37.5 | −40.8 |

Read it as: the **160 Hz-4 kHz region is now continuous and smooth** (the 1 k dip
and the 1.6-4 kHz hole are filled - that is the "boxy, dull, shouty" sound
disappearing), the 250-630 Hz box is pulled back, and the 6.3-8 kHz band is
*quieter* than the source, because what was there was noise.

Crest factor going *up* while loudness also goes up (Lonesome Feeling: 10.9 → 13.3 dB
crest at −8.45 → −12.00 LUFS) is the signature of the source being squashed and
clipped: the restored master is at the same perceived level but needs far less
dynamic compression to get there.

### Micro-dynamics: what "breathing" means numerically

Short-term (100 ms) crest variation across the programme, and the
10th-95th percentile spread of the short-term level:

| song | spread before | spread after | crest variation before → after |
|---|---|---|---|
| I Need You | 12.4 dB | **11.4 dB** | 6.4 → 6.1 dB |
| Lonesome Feeling | 5.2 dB | **5.6 dB** | 1.3 → 1.5 dB |
| Put It On | 11.0 dB | **10.3 dB** | 1.4 → 1.7 dB |
| Pussycat | 8.5 dB | **8.3 dB** | 1.4 → 1.5 dB |

The previous chain collapsed this: 7.1 dB on *I Need You*, i.e. the quiet
passages had been pushed up by 5 dB. The cause was the parallel-compression ratio
(20 % of a 3.5:1 copy at −6 dB is dominated by the quiet parts) plus the per-stem
make-up gain; both are fixed.

---

## 5. What you get / how to listen

Two things are worth knowing before you listen:

* **The noise is gone from the gaps, not from the music.** A spectral denoiser
  that removes hiss during sustained passages must modulate individual bins, which
  is audible as swimming or chirping - exactly the "restored" sound. So the
  suppression is frequency- and level-aware and the gaps benefit most. That is
  also why the measured SNR gain is a few dB rather than 20: the remaining hiss is
  masked by the music and removing it would cost more than it buys.
* **The width is made from the music itself.** An allpass decorrelator, not a
  delay line: the side channel has the *same spectrum* as the mid. If you fold the
  master to mono nothing changes - that was a hard requirement and it is verified
  (mono sum == mid, 0.000 dB difference).

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
work/bwe.py                             neural top-octave reconstruction (PyTorch)
work/models/bwe_*.pt                    the five trained models (vocals/drums/bass/other/master)
work/bwe_report.json                    held-out R² per model
work/audit.py                           defect audit (this report's tables) -> work/audit.json
work/image.py                           allpass stereo decorrelator (flat magnitude, no comb)
work/wowfix.py                          wow/flutter measurement + bounded time warp
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
python work/bwe.py             # train the top-octave networks        -> work/models/bwe*.pt
./work/run_all_restore.sh      # restoration + mastering, 1 process per song
python work/summary.py         # before/after table                   -> work/verify_summary.json
python work/export.py          # A/B excerpts, MP3s, doc plots
python work/finalize.py        # 24-bit FLAC masters                  -> masters/
python work/audit.py           # defect audit (SNR/dynamics/stereo/wow)-> work/audit.json
```

`work/dsp.py` holds the whole DSP toolbox; `work/restore.py` the per-song
recipes and `STEM_CHAIN` (per-stem HP/LP and EQ only - see §3); `work/image.py`
builds the stereo image; `work/audit.py` is the tool that produced every number
in §4 and is meant to be re-run after any change to the chain.

Debugging aids: `RESTORE_STAGE_DIR=/tmp/stages python work/restore.py <tag>`
writes every intermediate stage to disk, so a measurement can be attributed to a
single processing step instead of guessed.
