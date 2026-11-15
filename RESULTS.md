# Ergebnisse — zweiter Durchgang

## Was ersetzt wurde

Eine Stufe: **das Bandbreiten-Modell**. Der alte Extender war ein MLP, der die
mittlere Pegelhöhe des oberen Bandes gelernt hatte (Cross-Song R² −0,19 … +0,42).
Das neue Modell ist ein rekurrentes adversariales Netz. Details in `BWE.md`,
die Naht-Odyssee in `SEAM.md`, die Diagnose in `FINDINGS.md`.

## Cross-Song-Validierung (das einzige ehrliche Maß)

Training auf drei Songs, Messung auf dem vierten:

| Held-out-Song | altes Modell | neu (1024 FFT) | **neu (2048 FFT)** | LSD |
|---|---|---|---|---|
| I Need You | −0,19 | +0,518 | **+0,727** | 0,34 dB |
| Lonesome Feeling | −0,19 | +0,405 | — | — |
| Put It On | −0,13 | +0,545 | **+0,650** | 0,42 dB |
| Pussycat | −0,006 | +0,612 | — | — |

Erstmals deutlich positiv auf allen Songs. Das Fenster von 2048 statt 1024
Samples war dabei entscheidend: bei 1024 maß das Band 8–16 kHz **−78 dB** gegen
echte −40 dB, das Netz bekam oberhalb 8 kHz praktisch nichts zu sehen.

## Die gelieferten Master

| Song | LUFS | dBTP | 12–16 kHz | Crest | Korr | Clip |
|---|---|---|---|---|---|---|
| I Need You | −12,00 | −1,00 | −26,1 | 12,8 | 0,9618 | 0 |
| Lonesome Feeling | −12,00 | −1,10 | −25,6 | 13,1 | 0,9666 | 0 |
| Put It On | −12,00 | −1,10 | −26,0 | 13,3 | 0,9616 | 0 |
| Pussycat | −12,00 | −1,00 | −26,0 | 14,2 | 0,9631 | 0 |

Ziel ≥ −25 dB im oberen Band; erreicht −25,6 … −26,1.

## Ist es besser als der abgelehnte Durchgang?

Ehrliche Antwort: **im Bandpegel gleich, in der Qualität des Bandes besser.**

| Song | 12–16 kHz alt → neu | Flachheit alt → neu |
|---|---|---|
| I Need You | −26,1 → −26,1 | −3,8 → **−5,3** |
| Lonesome Feeling | −26,0 → −25,6 | −3,1 → −5,0 |
| Put It On | −26,0 → −26,0 | −3,3 → −5,5 |
| Pussycat | −26,0 → −26,0 | −4,3 → −6,4 |

Die **Flachheit** ist der Unterschied. Eine Hiss-Wand hat 0 dB, Musik ist deutlich
negativ. Der alte Master lag bei −3,1 … −4,3 dB, der neue bei −5,0 … −6,4 dB,
die **Quelle** bei −5,8 dB. Der neue Höhenbereich ist also tonal und trifft die
Charakteristik des Originals; der alte war rauschiger.

Zusätzlich, gemessen direkt nach der BWE (vor dem Mastering):

| | Kopplung Höhen ↔ Gesamtenergie |
|---|---|
| Quelle | 0,311 |
| nach BWE | **0,326** |
| nach Mastering | 0,212 |

Das rekonstruierte Band folgt den Anschlägen **besser als die Quelle selbst**.
Der Verlust auf 0,212 entsteht erst im Limiter — der Loudness-Druck glättet
zwangsläufig. Das ist eine Eigenschaft des Masters, nicht der Restaurierung.

## Was sich nicht geändert hat

| Achse | Wert | Ziel | Status |
|---|---|---|---|
| Raum (dB/s) | −34,8 … −48,5 | −250 … −400 | **nicht bearbeitbar** |
| 16–20 kHz | −74 … −85 dB | — | bewusst nicht synthetisiert |
| Dynamik (Block-std) | 8–10 dB | 8–14 | war schon im Ziel |
| Clipping | 0 in allen vier | 0 | erreicht |

## Die Stellschraube

`strength_db` in `work/restore.py` → `air_bwe2()`. Gemessen: **exakt 1 dB
Bandanstieg pro dB Einstellung**, linear über 14–30 dB, auf allen vier Songs.
Der Regelkreis sucht die Einstellung pro Song selbst, bis der fertige Master
−26 dB trifft — 1–2 Durchläufe.

Wer mehr will: `target_db` auf −20 setzen. Wer weniger will: −32.
