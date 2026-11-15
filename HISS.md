# Das Rauschen — was es war und warum es so lange übersehen wurde

## Der Befund

Alle vier ausgelieferten Master hatten ein hörbar extremes Rauschen. Gemessen:

| Song | HF-Pegel in den Pausen | | |
|---|---|---|---|
| | Quelle | alter Master (b1e7a41) | **ausgeliefert (kritisiert)** |
| I Need You | −36,1 dB | −40,2 dB | **−11,4 dB** |
| Lonesome Feeling | −34,1 dB | −38,0 dB | **−18,1 dB** |
| Put It On | −24,2 dB | −23,9 dB | **−5,9 dB** |
| Pussycat | −23,2 dB | −20,7 dB | **−3,2 dB** |

„HF-Pegel in den Pausen" heißt: Pegel des 8–14-kHz-Bands relativ zum Mittelband,
gemessen **nur in den 10 % leisesten Frames**. Das ist der Bereich, in dem
Rauschen allein hörbar ist, weil keine Musik es maskiert.

Bei Song 4 lag das Oberband in den Pausen **3,2 dB unter dem Mittelband** — das
heißt, die Rauschwand war dort fast so laut wie die Musik. Ein Unterschied von
20–30 dB zur Quelle ist keine Feinheit, das ist das, was man sofort hört.

## Warum es passiert ist

Die Bandbreiten-Stufe verstärkte das Band 8–16 kHz um **+26 dB**, um das
Roadmap-Ziel „12–16 kHz ≥ −25 dB" zu treffen. Zwei Fehler in dieser Logik:

### 1. Das Ziel war falsch begründet
Ich hatte es aus einem modernen Referenzmix abgeleitet, ohne zu prüfen, **was in
diesem Band bei dieser Quelle überhaupt drin ist**. Flachheit pro Band, alle vier
Quellen:

| Band | 2–3k | 3–4k | 4–5k | 5–6k | 6–7k | 7–8k | 8–14k | 14–16k |
|---|---|---|---|---|---|---|---|---|
| Flachheit | −4,8 | −4,4 | −3,9 | −3,5 | −2,8 | −2,5 | **−2,6** | −8,0 dB |

0 dB = weißes Rauschen, negativ = tonale Struktur. Das Band 8–14 kHz liegt bei
−2,6 dB, also **fast Rauschen**. Und seine Dynamik ist zusammengebrochen: bei
4–5 kHz liegen das 5. und 95. Perzentil der Frame-Energien 37 dB auseinander, bei
8–9 kHz nur 21 dB und ohne leise Frames.

### 2. Die Verstärkung war absolut, nicht relativ
`bwe2.extend()` rechnete `gain = want/cur` gegen einen **festen Zielpegel**. Bei
einer Quelle, deren Oberband 30 dB zu leise ist, heißt das +26 dB — und weil in
diesem Band überwiegend Rauschen liegt, wurde überwiegend Rauschen verstärkt.
Der Regelkreis in `restore.py` hat das noch verschärft: er regelte so lange nach,
bis der Zielpegel erreicht war, also bis die Rauschwand exakt auf Zielhöhe saß.

## Die Korrektur

### Rausch-Gate pro Zeit-Frequenz-Zelle
Statt jede Zelle gleich zu verstärken, wird pro Bin der Rauschboden geschätzt
(20. Perzentil über die Zeit) und die Verstärkung fällt unterhalb davon auf null.
Der Übergang ist weich (`snr_soft_db`), weil ein hartes Gate auf einem
Breitbandsignal als „Atmen" hörbar ist. Gemessen lässt das Gate im Median nur
**18–20 %** der Verstärkung durch.

### Relative statt absolute Anhebung
Der Zielwert wird nicht mehr absolut vorgegeben, sondern als **Anhebung
gegenüber dem Ist-Zustand**. Jeder Song behält damit seinen Charakter:

| Song | Quelle HF/Mid | lift 6 dB | lift 14 dB | lift 18 dB |
|---|---|---|---|---|
| I Need You | −44,7 | −38,4 | −35,3 | −33,5 |
| Put It On | −37,5 | −30,5 | −26,4 | −24,1 |
| Pussycat | −31,6 | −24,1 | −19,3 | −16,8 |

Song 4 ist von Natur aus 13 dB heller als Song 1 — das ist die Aufnahme, nicht ein
Fehler. Ein absoluter Zielwert hätte beide gleich gemacht und dabei Song 1
überstrahlt und Song 4 zugerauscht.

## Die Prüfgröße, die gefehlt hat

Ich hatte die Flachheit gemessen und sie sank von −4,1 auf −5,3 — das sah wie eine
Verbesserung aus. Der Fehler: **Flachheit misst nur die Form des Spektrums, nicht
seinen Pegel**. Ein gleichmäßig verstärktes Rauschen hat dieselbe Flachheit wie
das Original, nur lauter.

Was gefehlt hat, ist die Messung **in den Pausen**. Musik maskiert Rauschen; in
den Pausen tut sie es nicht. Deshalb steht diese Messung jetzt in
`work/hiss.py` und wird bei jeder Änderung an der Bandbreiten-Stufe mitgeprüft.

## Stand der Korrektur

| Song | HF-Stille vorher | **nachher** | Verbesserung |
|---|---|---|---|
| I Need You | −11,4 | −26,9 | **15,5 dB** |
| Lonesome Feeling | −18,1 | *offen* | |
| Put It On | −5,9 | −9,4 | **3,5 dB** |
| Pussycat | −3,2 | −13,5 | **10,3 dB** |

Die Korrektur wirkt, ist aber bei den von Natur aus hellen Songs (3 und 4) noch
nicht ausreichend. Der Grund ist strukturell: dort liegt das Oberband der Quelle
selbst schon bei −24 dB, also nur 24 dB unter dem Mittelband — jedes Anheben
verstärkt zwangsläufig auch den dort vorhandenen Rauschanteil.

Der ehrliche Kompromiss: **entweder** man nimmt das Rauschen in Kauf und bekommt
Höhen, **oder** man bleibt nahe an der Quelle und bekommt weniger Höhen. Die
Stellschraube dafür ist `snr_soft_db` und die Anhebung in dB — beide sind jetzt
Parameter, nicht Konstanten.
