# Verifikation — Stand nach dem zweiten Durchgang

Alle Werte an den **ausgelieferten Dateien** gemessen, nicht an Zwischenschritten.
Reproduzierbar mit `work/verify_new.py`.

## 24-bit FLAC Master

| Song | LUFS | dBTP | 12–16 kHz | Crest | Stereo-Korr. | Clip |
|---|---|---|---|---|---|---|
| I Need You | −12,00 | −1,00 | −26,1 | 12,8 | 0,9611 | 0 |
| Lonesome Feeling | −12,00 | −1,10 | −26,0 | 12,9 | 0,9670 | 0 |
| Put It On | −12,00 | −1,10 | −26,0 | 13,3 | 0,9623 | 0 |
| Pussycat | −12,00 | −1,00 | −26,0 | 13,7 | 0,9602 | 0 |

Ziel: −12,0 LUFS / ≤ −1,0 dBTP. **Erfüllt.**

## MP3 320 (portabel)

| Song | LUFS | dBTP |
|---|---|---|
| I Need You | −12,00 | −1,00 |
| Lonesome Feeling | −12,00 | −1,20 |
| Put It On | −12,00 | −1,10 |
| Pussycat | −12,00 | −1,00 |

## Formatintegrität

* **FLAC bit-identisch zu WAV**: geprüft durch Rückdekodierung nach PCM_24 und
  Vergleich mit dem WAV — alle vier `True`.
* **Mono-Kompatibilität**: Verlust der Mono-Summe gegenüber Stereo
  **−0,07 … −0,09 dB**. Kein Phasenauslöschung, kein Pumpen.

## Zwei Fehler, die in diesem Durchgang gefunden und behoben wurden

### 1. True-Peak mit 4× Oversampling war zu grob
Song 1 las bei 4× Oversampling −1,000 dBTP und bei 8× **−0,923 dBTP** — 77 mdB
Intersample-Übersteuerung, die 4× nicht sieht, bei einem Sample-Peak von bereits
−1,006 dBFS. EBU R128 fordert *mindestens* 4×; bei diesem Material mit dichtem
Oberband ist das zu wenig. `dsp.true_peak_db()` rechnet jetzt mit 8×, der Limiter
bekommt 0,08 dB Sicherheitsabstand.

### 2. MP3-Headroom war auf die alte, höhenarme Fassung kalibriert
Die 0,07 dB Encoder-Reserve waren am ersten Durchgang gemessen, dessen Oberband
fast leer war. Nach der Bandbreiten-Stufe sitzt 12–16 kHz 22–32 dB höher — und
genau dort lebt der Pre-Echo des Coders. Song 1 dekodierte mit 0,07 dB Reserve auf
**−0,80 dBTP**. Gemessene Reserve-Kurve:

| Headroom | MP3-True-Peak |
|---|---|
| 0,07 dB | −0,90 dBTP |
| 0,15 dB | −1,00 dBTP |
| 0,25 dB | −1,20 dBTP |
| 0,35 dB | −1,20 dBTP |
| 0,45 dB | −1,30 dBTP |

Jetzt 0,20 dB Reserve → schlechteste Datei −1,05 dBTP, Loudness unverändert.

## Was NICHT verifiziert werden kann

Der **Raum** ist unverändert: −34,8 … −48,5 dB/s gegenüber einem modernen Ziel von
−250 … −400. Das ist mit den hier verfügbaren Mitteln nicht lösbar — fünf Verfahren
gebaut und gemessen, Oracle-Test negativ, SOTA-Gewichte nicht erreichbar. Details in
`STATUS.md` und `FINDINGS.md`. Wer das anders darstellt, lügt.

Gleiches gilt für **16–20 kHz**: −74 … −85 dB. Dort steht die Quelle bei −110 dBFS
(digitaler Nullpunkt des AAC-Codecs). Es wird nicht synthetisiert, weil eine
Synthese ohne Vorlage als Rauschwand hörbar ist.
