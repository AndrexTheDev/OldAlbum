# Befund — was auf diesen Aufnahmen tatsächlich kaputt ist

Gemessen an allen vier Quelldateien, gegen die sieben Zielachsen aus
`ROADMAP.md` §1. Die Reihenfolge ist nach **Abstand zum Ziel** sortiert, nicht
nach Bearbeitbarkeit — das ist der Punkt.

| # | Achse | IST | Modernes Ziel | Abstand | bearbeitbar? |
|---|---|---|---|---|---|
| 1 | **Raum / Nachhall** | **−32 … −47 dB/s** (≙ RT60 1,3–1,9 s) | −250 … −400 dB/s | **×7–10** | **nein** (siehe unten) |
| 2 | Höhen 12–16 kHz | −42 … −51 dB | −25 dB | 17–26 dB | ja, aber nur *erzeugt* |
| 3 | Rauschboden | +35 … +45 dBFS in Pausen | −90 dBFS | **prinzipiell** | teilweise |
| 4 | Clipping | 106 / 113 Samples (Songs 2, 3) | 0 | klein | **ja** |
| 5 | Crest-Faktor | 10,8 … 13,7 dB | 14–18 dB | 1–3 dB | ja, begrenzt |
| 6 | Block-Streuung (Dynamik) | 8,0 … 9,9 dB std | 8–14 dB | **0** | **schon im Ziel** |
| 7 | Stereo | Korrelation 1,00000 | 0,85–0,95 | voll | ja (erzeugt) |

## Was das heißt

### Achse 6 ist bereits erfüllt — meine frühere Diagnose war falsch
Ich hatte die Aufnahmen für „stark komprimiert" gehalten, weil die 400-ms-LRA
bei Song 2 nur 3,3 dB betrug. Die Verteilung zeigt warum: der **Median**-Block
liegt bei −10,5 dBFS und 89 % aller Blöcke innerhalb von 3 dB — das ist eine
*normale* Verteilung in einem *lauten* Programm, kein Dynamikverlust. Die
Standardabweichung der Blockpegel ist **8–10 dB**, das ist genau der Bereich
moderner Produktionen. LRA war hier ein Artefakt der Stille am Anfang und Ende
(p1 = −77 dBFS).

Konsequenz: `dynamics.expand()` und `dynamics.decompress()` sind gebaut, gemessen
und **werden nicht eingesetzt**. Beide verbessern den Crest-Faktor, aber keine
von beiden bringt einen echten Gewinn — die Dynamik war nie das Problem. Sie
stehen im Code, damit die Messung nachvollziehbar bleibt, nicht als aktive Stufe.

### Achse 1 ist nicht bearbeitbar — und das ist keine Ausrede
Der Nachhall ist auf diesen Bändern **so laut wie der Anschlag**. Das ist keine
Einstellung, das ist die Aufnahme: ein Mikrofon, ein großer Raum, kein
Nahbesprechungsabstand. Studio One 1965 war genau das.

Fünf Dereverberations-Verfahren wurden gebaut und gemessen (Details in
`STATUS.md`), das entscheidende Ergebnis ist der **Oracle-Test**: selbst mit
perfekter Kenntnis des trockenen Signals trennt die spektrale Subtraktion den
Hall nicht, weil das STFT-Fenster (46 ms) länger ist als der Bereich, in dem der
Hall vom Direktsignal getrennt wäre. Was hilft, ist ein gelerntes Modell — und
die Gewichte dafür (BEHM-GAN, BABE-2, BS-RoFormer-Dereverb) liegen auf Hosts,
die aus dieser Umgebung nicht erreichbar sind.

### Achse 3 ist prinzipiell begrenzt
Der Rauschboden liegt rechnerisch bei +35…+45 dBFS — das klingt absurd, ist aber
korrekt: die *leisesten 10 % Frames* enthalten bei diesen Dateien noch Musik
(kein echter Stille-Abschnitt). Wo tatsächlich Stille ist, steht der AAC-Codec bei
−108 dBFS, also digitaler Nullpunkt. Ein „−90 dBFS"-Ziel ist damit nicht
messbar — die Kennzahl ist auf diesem Material nicht definiert. Was zählt, ist die
Rauschunterdrückung *relativ zur Quelle*, und die ist messbar und bereits
dokumentiert.

## Die zwei Achsen, die wirklich etwas bringen

**Achse 2 (Höhen)** und **Achse 7 (Stereo)** — plus das kleine Clipping auf zwei
Songs. Das ist weniger als „jedes Mittel", aber es ist das, was physikalisch da
ist. Alles andere wäre Effekt ohne Substanz.

## Was ich daraus für die Kette ableite

```
1. Declip            (106/113 Samples, klein aber der erste Schritt)
2. Denoise           (relativ zur Quelle messbar, Boden senken)
3. Separation        (Demucs, auf dem enthallten Material unverändert)
4. Bandwidth         (das eigentliche Geschäft: 12-16 kHz um 17-26 dB anheben,
                      ohne Hiss-Wand — nur mit strukturierter Rekonstruktion)
5. Voicing           (gemessene Korrektur + gewollte Kurve)
6. Stereo            (Allpass, 0.000000 dB Abweichung, bereits validiert)
7. Master            (Loudness/Peak, Specs erfüllt)
```

Kein Dereverb, keine Expansion. Nicht weil sie zu schwer sind, sondern weil die
Messung zeigt, dass sie hier nichts verbessern.
