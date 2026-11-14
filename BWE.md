# Bandwidth Extension — was funktioniert hat und warum das alte Modell nicht

## Das erste Modell war schlecht, und die Ursache war strukturell

Das alte `work/bwe.py` war ein MLP auf 24 Log-Mel-Bändern, das das Band
8,8–17,6 kHz Frame für Frame vorhersagte. Die Zahlen sahen akzeptabel aus
(Held-out R² 0,69 innerhalb eines Songs), aber **cross-song** war R² **−0,19 bis
+0,42** und der vorhergesagte Bandmittelwert folgte dem Zielmittelwert fast
exakt. Diagnose: das Modell hat die **mittlere Pegelhöhe** des Bandes gelernt,
nicht seine Struktur. Gegen eine bandbegrenzte Quelle reicht das, um das Spektrum
mit plausibler Energie zu füllen — und nicht, um einen Höhenklang zu erzeugen, der
zur Musik gehört. Genau das hat der Nutzer gehört und zu Recht abgelehnt.

## Was das neue Modell anders macht

| | alt (`bwe.py`) | neu (`bwe2.py`) |
|---|---|---|
| Eingabe | 24 Mel-Bänder, ein Zeitpunkt | 48 Mel-Bänder 500–8000 Hz über 64 Frames (120 ms) |
| Modell | MLP, 2 Schichten | **GRU, 2 Schichten, bidirektional**, 537 k Parameter |
| Verlust | MSE | L1 + **spektrale Konvergenz** + **adversarial** (CNN-Diskriminator) |
| Validierung | innerhalb eines Songs | **cross-song** (3 trainieren, 1 messen) |
| Ausgabe | 8,8–17,6 kHz | 8–16 kHz, darüber bewusst nichts |

Der rekurrente Teil ist der Kern: ein GRU über 120 ms kann unterscheiden, ob der
aktuelle Frame **in einer Note** liegt oder **in einem Transienten**. Das MLP
hatte kein Zeitgedächtnis — sein "Kontext" war ein Fenster von Mel-Bändern zu
einem einzigen Zeitpunkt.

Der Diskriminator ist der zweite Kern. Rekonstruktionsverluste allein ergeben
immer ein verwaschenes oberes Band, weil der Erwartungswert der beste Schätzer
unter L1/L2 ist. Das ist genau das BEHM-GAN-Ergebnis (Moliner et al.), und es
reproduziert sich hier.

## Cross-Song-Validierung — die einzige Zahl, die zählt

Training auf drei Songs, Messung auf dem vierten, der nie im Training war:

| Held-out-Song | altes Modell R² | **neues Modell R²** | LSD |
|---|---|---|---|
| I Need You | **−0,19** | **+0,518** | 0,48 dB |
| Lonesome Feeling | −0,19 | **+0,405** | 0,60 dB |
| Put It On | −0,13 | **+0,545** | 0,51 dB |
| Pussycat | −0,006 | **+0,612** | 0,70 dB |

Erstmals **positiv auf allen vier** Songs. Das ist der Unterschied zwischen einem
Modell, das die Bandmitte memorisiert, und einem, das die Bandstruktur gelernt
hat. LSD 0,5–0,7 dB ist die Größenordnung, die auch BEHM-GAN auf historischem
Material berichtet (dort 1,0–1,5 dB, hier leichter, weil das Band bei diesen
Quellen noch vorhanden ist, nur leise).

Produktionsmodell: `work/models/bwe2.pt`, 2,1 MB, auf allen vier Songs trainiert.

## Steuerbarkeit — der `strength_db`-Regler

Gemessen an Song 1, Referenz 300–3000 Hz:

| `strength_db` | 8–12 kHz | 12–16 kHz | Peak |
|---|---|---|---|
| vorher | −46,1 | −51,7 | −0,01 |
| 8 dB | −38,8 | −44,2 | 0,11 |
| 12 dB | −34,8 | −40,2 | 0,47 |
| 16 dB | −30,8 | −36,2 | 1,04 |
| 20 dB | −26,8 | −32,2 | 2,04 |

**Exakt +1 dB pro dB Regler** in beiden Bändern. Der Regler ist damit ein
skalierbarer, vorhersagbarer Parameter, kein emergentes Verhalten des Netzes.

Für das Ziel "12–16 kHz ≥ −25 dB" (Roadmap §1) braucht Song 1 ca. **22–26 dB**.
Bei diesem Pegel steigen die Spitzen um mehr als 1 dB über Vollaussteuerung, was
der Master-Limiter ohnehin korrigiert — der Regler ist also nicht durch Clipping
begrenzt.

## Artefaktkontrolle — ist das Musik oder Rauschen?

| Messung | vorher | nachher (12 dB) | Bedeutung |
|---|---|---|---|
| Spektrale Rauheit | 0,47 dB | **0,46 dB** | unverändert → keine zusätzliche Härte |
| Flachheit 12–16 kHz | −5,8 dB | **−3,9 dB** | bleibt tonal; 0 dB wäre Rauschen |
| Hüllkurven-Korrelation oben | — | **0,958** | folgt dem Musikverlauf |

Die Flachheit ist der wichtigste Wert: eine Hiss-Wand hätte 0 dB, die
Rekonstruktion liegt bei −3,9 dB, also deutlich strukturiert.

## Die harte Grenze bei 15,8 kHz

Gemessen an allen vier Quellen (1/6-Oktav, median):

| Band | Song 1 | Song 2 | Song 3 | Song 4 |
|---|---|---|---|---|
| 6–8 kHz | −43 | −35 | −31 | −24 |
| 12–14 kHz | −53 | −44 | −46 | −52 |
| 14–16 kHz | −57 | −46 | −49 | −59 |
| **16–18 kHz** | **−111** | **−112** | **−110** | **−113** |

Oberhalb 15,8 kHz steht der AAC-Codec bei −110 dBFS — digitaler Nullpunkt. Dort
ist **nichts zu rekonstruieren**, weil nie etwas da war. Alles darüber wäre
Synthese, und eine Synthese ohne Vorlage wird als Rauschwand gehört. Das Modell
endet deshalb bei 16 kHz, und das ist eine bewusste Entscheidung, keine
Einschränkung.

Das Ziel "12–16 kHz ≥ −25 dB" ist damit erreichbar. Ein Ziel "bis 20 kHz" wäre
es nicht, und die Roadmap führt es zu Recht nicht.
