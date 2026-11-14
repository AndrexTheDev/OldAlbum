# Verification log — every stage checked against a known answer

This file exists because the first version of this project claimed numbers that
did not survive re-measurement. Every claim below was produced by code in
`work/`, and the checks are permanent (`work/selftest.py`, exit code 0).

Run: `.venv/bin/python work/selftest.py` → **all checks passed**.

---

## 1. Was the right tool used at every stage?

| stage | what the problem needed | what is in the chain | verdict |
|---|---|---|---|
| separation | 4 stems from a mono 1960s mix | **Demucs v4 htdemucs** (ONNX, 174 MB) | correct tool. Measured back: stems sum to the source at 0 samples lag, corr 0.97 |
| separation error | undo what the split did to the balance | `dsp.correction_from_source` (new) | **was missing** — see §4 |
| hiss | statistical noise reduction that understands music | spectral subtraction, minimum statistics, tonal protection (new) | correct class of algorithm; **two bugs fixed** |
| rumble / hum | high-pass + comb-aware notch | 30 Hz HPF (zero-phase), 50/100/150 Hz notch only where the comb is real | correct |
| clicks | impulse detection + interpolation | median/MAD detector, cubic repair | correct |
| missing top end | *re-synthesis*, not a shelf boost | PyTorch MLP (24-band log-mel → 8.8-17.6 kHz) + transposer | correct approach, **honest score is 0.69 not 0.75**, does not transfer across songs — §3.6 of RESTORATION.md |
| boxy midrange | measured correction | measured separation fix + fixed voicing curve | **was correcting the wrong band** — see §4 |
| dynamics | gentle glue + parallel density | 1.35:1 glue, 2.5:1 parallel at 16 % | correct, and "parallel" is not a make-up-gain trick: dry path untouched |
| stereo | mono → stereo without comb filtering | Schroeder allpass cascade, solved side level | correct: allpass magnitude deviation **0.000000 dB** |
| loudness / peaks | −12.0 LUFS, −1.0 dBTP | fixed-point gain → limit → pull back | correct, spec met on all four |

---

## 2. Unit tests (the ones that caught the bugs)

Every primitive is checked against an analytically known answer, not against
"looks plausible":

| check | expected | measured |
|---|---|---|
| flat +6 dB EQ request | +6.00 dB | **+6.00 dB** |
| flat −9 dB EQ request | −9.00 dB | **−9.00 dB** |
| *other* stem curve, 13 bands | ≤ 0.5 dB error each | **max 0.15 dB, mean 0.00 dB** |
| *bass* stem curve, 14 bands | ≤ 0.5 dB error each | **max 0.47 dB** |
| allpass magnitude flatness 50 Hz-20 kHz | 0 dB | **0.000000 dB** |
| old comb filter (regression guard) | must show ripple | **34.5 dB ripple** |
| stereo solve, target corr 0.90 | 0.90 | **0.9155** |
| stereo solve, target corr 0.95 | 0.95 | **0.9579** |
| mono sum == mid | exact | **exact** |
| compressor 4:1, −28 dB RMS input | 0.00 dB GR | **−0.00 dB** |
| compressor 4:1, −16 dB RMS input | −2.99 dB GR | **−2.99 dB** |
| compressor 4:1, −6 dB RMS input | −10.49 dB GR | **−10.49 dB** |
| limiter, mono, ceiling −1.0 | ≤ −1.000 dBTP | **−1.000 dBTP** |
| limiter, stereo, ceiling −1.0 | ≤ −1.000 dBTP | **−1.000 dBTP** |
| denoiser, held 440 Hz tone | kept within 1 dB | **−0.43 dB** |
| denoiser, 8-12 kHz hiss | removed | **−8.19 dB** |
| declicker, 0.9 impulse | repaired | **0.90 → 0.05** |
| declicker, rest of programme | unchanged | **max 0.00000** |
| IF of a pure 220 Hz tone | 220.00 Hz | **220.00 Hz** |

Two of these tests initially failed, and **both were real defects, not test bugs**:

* `fir_from_curve` returned **0.00 dB for a +6 dB request**. The function had been
  "fixed" once already; the replacement line normalised to 0 dB *at DC*, which
  shifts the whole filter by the curve's DC value. Peak normalisation then divided
  by 1.0 and did nothing.
* the denoiser removed **−5.26 dB** from a held tone: minimum statistics classify
  anything continuously present as noise.

Two other initial "failures" were **my own test bugs**, and are recorded so the
distinction is visible:

* the compressor test used a sine at −10 dBFS **peak** and called it −10 dB. The
  detector is RMS, so the correct answer was −5.24 dB, which is what the code
  produced. The test was wrong; the compressor was right.
* the declicker test demanded bit-identical output over the untouched region. The
  repair leaves ringing that re-triggers detection, so a one-sample tolerance was
  unrealistic; it is now "≤ −50 dBFS difference".

---

## 3. Delivered masters, re-measured from the files

| master | LUFS | dBTP | crest | stereo corr | HF reach | duration |
|---|---|---|---|---|---|---|
| I Need You | −12.01 | −1.00 | 13.0 | 0.9599 | 18174 Hz | 185.2 s |
| Lonesome Feeling | −12.00 | **−1.17** | 13.2 | 0.9667 | 18174 Hz | 169.0 s |
| Put It On | −12.02 | −1.00 | 13.3 | 0.9617 | 18260 Hz | 186.7 s |
| What's New Pussycat | −12.01 | −1.00 | 13.4 | 0.9591 | 18949 Hz | 177.8 s |

Lonesome Feeling lands at −1.17 dBTP **not −1.00**: that is *under* the ceiling,
i.e. the spec "≤ −1.0 dBTP" is met with 0.17 dB of unused headroom. Its crest is
the lowest of the four, so at −12.00 LUFS its true peak cannot reach −1.00 without
raising loudness. Chasing the last 0.17 dB would mean exceeding either the
loudness or the peak target; leaving it is the correct choice.

FLAC masters verified **bit-identical** to the rendered WAVs (24-bit PCM, 44.1 kHz).

### The 320 kb/s MP3s were violating the peak spec

Found on the last pass, by decoding the delivered files instead of trusting the
encoder. The masters sit at −1.00 dBTP; the 320 kb/s MP3s decoded to
**−0.95 / −1.17 / −0.97 / −0.98 dBTP** — three of four *over* the ceiling, because
the transform coder adds pre-echo and inter-sample overshoot that was not in the
WAV. The portable copies did not meet the spec the lossless ones met.

First attempted fix, a static −0.4 dB trim, was **wrong**: it brought the peaks to
−1.33 dBTP but dropped the MP3s to −12.41 LUFS, breaking the album's loudness
consistency. Measured, the codec only overshoots by **+0.07 dB**, so the encoder
input is now limited at `ceiling − 0.07 dBTP`. The limiter touches only the top
0.07 dB of the loudest peaks; the decoded files now measure:

| MP3 | LUFS | true peak | spec |
|---|---|---|---|
| I Need You | −12.01 | −1.03 | OK |
| Lonesome Feeling | −12.00 | −1.17 | OK |
| Put It On | −12.02 | −1.02 | OK |
| What's New Pussycat | −12.01 | −1.03 | OK |

Verified by decoding each file, not by assuming the conversion was transparent.

---

## 4. What the re-verification changed in the output

Five changes were made *because* a measurement contradicted a claim, and the
masters were re-rendered afterwards:

1. **EQ level fix** — every band had been off by the curve's DC value (up to 9 dB).
2. **Tonal protection in the denoiser** — held notes lost 5.26 dB; now 0.43 dB.
3. **Measured separation correction** (`correction_from_source`) — Demucs
   over-weights below 100 Hz by +2.33/+2.32/+1.88 dB while the *fixed* voicing
   curve was cutting 2.5 dB at 400 Hz where the source needs +0.11 dB. After the
   measured correction the residual is +0.06/+0.36/+0.95 dB at 40/63/100 Hz and
   ≤ 0.15 dB elsewhere.
4. **Honest BWE report** — grouped block split; 0.493/0.810/0.468/0.622/0.690
   instead of training error.
5. **Master loop can no longer trade loudness against peaks** without re-limiting.

---

## 5. What is *not* verified

Stated plainly, because a document that hides this is worthless:

* **Nobody has listened to these masters through a reference system.** Every
  number here is a measurement. A file that measures clean and sounds wrong is
  still a failure.
* **The top-end reconstruction is band-accurate, not event-accurate** on songs 3
  and 4 (8-16 kHz envelope correlation 0.190 / 0.234). It adds a plausible,
  structured top end; it does not restore what was recorded.
* **The 200 Hz - 1 kHz band is not cleaner than the source** (−1.9 to +0.6 dB).
  Three earlier versions of RESTORATION.md claimed +1 to +3 dB there; that claim
  was produced by the two bugs above.
* **Wow/flutter is measured but mostly not corrected** (§3.5 of RESTORATION.md):
  the transport drift on two songs is below what can be separated from the
  performance, and correcting it would be inventing.
* **"Sounds like it was recorded yesterday" is not a measurable target.** What is
  measurable is: hiss floor down 4.5-5.8 dB in the top octave, no stem build-up,
  mono-compatible width, −12 LUFS/−1 dBTP, and no audible artefacts *as measured*
  (no clipping, no inter-sample overs, crest preserved at 13+ dB).
