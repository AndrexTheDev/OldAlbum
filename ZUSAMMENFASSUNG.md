# Zusammenfassung — zweiter Durchgang

## Was Sie kritisiert haben, und was ich dagegen getan habe

Sie hatten recht. Der alte Höhenbereich wurde von einem Modell erzeugt, das
nachweislich **nur die mittlere Pegelhöhe** des oberen Bandes gelernt hatte:
Cross-Song R² **−0,19 … +0,42**, und der vorhergesagte Bandmittelwert folgte dem
Zielmittelwert fast exakt. Gegen eine bandbegrenzte Quelle reicht das, um das
Spektrum mit plausibler Energie zu füllen — und nicht, um einen Höhenklang zu
erzeugen, der zur Musik gehört.

## Was jetzt anders ist

**Neues Modell** (`work/bwe2.py`), drei Änderungen an den Ursachen:

* Eingabe ist **breitbandiger Kontext** (48 Mel-Bänder, 500–8000 Hz) über 120 ms,
  nicht 24 Bänder zu einem Zeitpunkt. Das Modell kann unterscheiden, ob ein Frame
  in einer Note oder in einem Anschlag liegt.
* **GRU mit Zeitgedächtnis** statt MLP ohne Gedächtnis.
* **Adversariales Training** (CNN-Diskriminator) zusätzlich zu L1 und spektraler
  Konvergenz. Rekonstruktionsverluste allein ergeben immer ein verwaschenes
  oberes Band, weil der Erwartungswert der beste L1-Schätzer ist.

**Cross-Song-Validierung** — Training auf drei Songs, Messung auf dem vierten:

| Held-out-Song | alt | **neu** | LSD |
|---|---|---|---|
| I Need You | −0,19 | **+0,727** | 0,34 dB |
| Put It On | −0,13 | **+0,650** | 0,42 dB |
| Lonesome Feeling | −0,19 | +0,405 | 0,60 dB |
| Pussycat | −0,006 | +0,612 | 0,70 dB |

## Ist der Klang jetzt besser?

Ehrliche Antwort: **der Bandpegel ist gleich, die Qualität des Bandes ist besser.**

| Song | 12–16 kHz alt → neu | Flachheit alt → neu |
|---|---|---|
| I Need You | −26,1 → −26,1 | −3,8 → **−5,3** |
| Lonesome Feeling | −26,0 → −25,6 | −3,1 → **−5,0** |
| Put It On | −26,0 → −26,0 | −3,3 → **−5,5** |
| Pussycat | −26,0 → −26,0 | −4,3 → **−6,4** |

Die **spektrale Flachheit** ist der Unterschied, den man hört. Eine Rauschwand
hat 0 dB, Musik ist deutlich negativ. Die Quelle selbst liegt bei −5,8 dB. Der
neue Höhenbereich (−5,0 … −6,4) trifft die Charakteristik des Originals; der alte
(−3,1 … −4,3) war rauschiger — er hat den Pegel mit Rauschen erreicht, der neue
mit Struktur.

Und gemessen direkt nach der Rekonstruktion, vor dem Mastering:

| | Kopplung Höhen ↔ Gesamtenergie |
|---|---|
| Quelle | 0,311 |
| nach Rekonstruktion | **0,326** |
| nach Mastering | 0,212 |

Der rekonstruierte Höhenbereich folgt den Anschlägen **besser als die Quelle
selbst**. Der Abfall auf 0,212 entsteht erst im Limiter — Loudness-Druck glättet
zwangsläufig. Das ist eine Eigenschaft der Aussteuerung, nicht der Restaurierung.

## Was ich nicht gelöst habe — ohne Ausrede

**Der Raum bleibt.** −34,8 … −48,5 dB/s gegen ein modernes Ziel von −250 … −400.
Ich habe fünf Verfahren gebaut und gemessen; das entscheidende war der Oracle-Test
mit perfekter Kenntnis des trockenen Signals: die spektrale Subtraktion bewegt den
Nachhall **in die falsche Richtung** (+4,0 dB statt −0,7). Der Grund ist
strukturell — das STFT-Fenster (46 ms) ist länger als der Bereich, in dem der Hall
vom Direktsignal getrennt wäre. Gelöste Modelle dafür liegen auf Hosts, die aus
dieser Umgebung nicht erreichbar sind.

Das ist die Grenze. Auf diesen Bändern ist der Hall **so laut wie der Anschlag** —
das ist keine Einstellung, das ist die Aufnahme: ein Mikrofon, ein großer Raum,
kein Nahbesprechungsabstand. Studio One 1965 war genau das.

**16–20 kHz wird nicht synthetisiert.** Dort steht die Quelle bei −110 dBFS, dem
digitalen Nullpunkt des AAC-Codecs. Es ist nichts zu rekonstruieren, weil nie
etwas da war. Eine Synthese ohne Vorlage wird als Rauschwand gehört.

**Die Dynamik war nie das Problem.** Ich hatte die Aufnahmen für stark komprimiert
gehalten, weil die 400-ms-LRA bei Song 2 nur 3,3 dB betrug. Die Verteilung zeigt
warum: der Median-Block liegt bei −10,5 dBFS und 89 % aller Blöcke innerhalb von
3 dB — eine normale Verteilung in einem *lauten* Programm. Die
Standardabweichung der Blockpegel ist 8–10 dB, genau der moderne Bereich. Die
niedrige LRA war ein Artefakt der Stille am Anfang und Ende. Zwei
Expansionsverfahren sind gebaut und gemessen — und **werden nicht eingesetzt**,
weil sie nichts verbessern.

## Die gelieferten Dateien

`masters/` — 24-bit FLAC, verlustfrei:

| Song | LUFS | dBTP | Crest | Korr |
|---|---|---|---|---|
| I Need You | −12,00 | −1,00 | 12,8 | 0,962 |
| Lonesome Feeling | −12,00 | −1,10 | 13,1 | 0,967 |
| Put It On | −12,00 | −1,10 | 13,3 | 0,962 |
| Pussycat | −12,00 | −1,00 | 14,2 | 0,963 |

`restored/` — MP3 320 (−12,00 LUFS, ≤ −1,00 dBTP) und A/B-Vergleiche.
`docs/` — Spektrogramme und Spektren je Song.

Alle geprüft mit `work/verify_new.py`: FLAC bit-identisch zum WAV, Mono-Summe
verliert nur 0,07–0,09 dB, null geclippte Samples.

## Die Stellschraube, falls Sie es anders wollen

`work/restore.py` → `target_db` in `air_bwe2()`, Standard −26.

* `−20` = deutlich mehr Höhen
* `−32` = zurückhaltender

Gemessen: **exakt 1 dB Bandanstieg pro dB Einstellung**, linear über 14–30 dB,
auf allen vier Songs. Der Regelkreis sucht die Einstellung pro Song selbst.
