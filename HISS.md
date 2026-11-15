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

---

# Die Korrektur, zweiter Anlauf — und der Preis

## Was eingebaut wurde

1. **Per-Bin-Rausch-Gate** in `bwe2.extend()`: Rauschboden pro Frequenzbin als
   20. Perzentil über die Zeit, Verstärkung fällt darunter auf null, weicher
   Übergang (`snr_soft_db`).
2. **Relative statt absolute Anhebung**: `lift_db` statt `target_db`. Jeder Song
   behält seinen Charakter.
3. **Die separate Rauschunterdrückung entfernt.** `spectral_noise_cut` lief mit
   −12 dB ab 6,8 kHz und zwang den Regelkreis, doppelt so viel Verstärkung zu
   verlangen — wodurch das Gate mehr Rauschen durchließ. Gemessen an Song 2:
   −21,2 dB Rauschen mit dem Schnitt, −24,0 dB ohne. Ein Mechanismus, eine
   Aufgabe.

## Der Preis, ehrlich benannt

Gemessen an Song 1 und 2, jeweils Pegel im Oberband gegen Rauschen in den Pausen:

| Anhebung | Höhengewinn | Rauschen über Quelle |
|---|---|---|
| 4 dB | +5,6 … 6,4 dB | +6,2 … 8,4 dB |
| 7 dB | +6,7 … 8,1 | +7,7 … 10,6 |
| 14 dB | +9,5 … 12,2 | +9,2 … 13,7 |

**Das Verhältnis ist fast 1:1.** Jeder dB Höhen kostet einen dB Rauschen. Das ist
kein Implementierungsfehler, sondern die Physik dieser Quelle: das Oberband
besteht überwiegend aus Rauschen, also verstärkt jede lineare Anhebung beides
gleich.

Zwei Versuche, das zu umgehen, sind gemessen und gescheitert:

* **Weiche Wiener-Maske im Oberband** (spektrale Subtraktion des Rauschbodens):
  wirkungslos. Rauschpegel −22,2 dB bei Faktor 1, 2 und 4 — identisch. Derselbe
  Grund wie beim gescheiterten Dereverb: das STFT-Fenster trennt Rauschen und
  Signal bei diesem Material nicht.
* **Stärkeres Gate**: 96 % der Bandenergie liegen über dem 20.-Perzentil-Boden.
  Ein höherer Faktor schneidet die Musik mit.

## Der eingestellte Kompromiss: `lift_db = 8`

| Song | Rauschen Quelle | **ausgeliefert** | **jetzt** | Verbesserung | Höhengewinn |
|---|---|---|---|---|---|
| I Need You | −36,1 | −11,4 | **−30,7** | **19,3 dB** | +5,0 dB |
| Lonesome Feeling | −34,1 | −18,1 | **−25,8** | **7,8 dB** | +6,7 dB |
| Put It On | −24,2 | −5,9 | **−13,1** | **7,2 dB** | +5,3 dB |
| Pussycat | −23,2 | −3,2 | **−14,3** | **11,1 dB** | +8,7 dB |

Song 1 liegt mit −30,7 dB nur noch 5,4 dB über der Quelle. Die Songs 3 und 4
bleiben 9–11 dB über ihrer Quelle — dort ist das Oberband der Quelle selbst schon
laut (−24 dB), und jedes Anheben verstärkt zwangsläufig mit.

Wer mehr Höhen will, stellt `lift_db` höher (10–14). Wer weniger Rauschen will,
niedriger (4–6). Beide Richtungen sind kalibriert und gemessen.

---

# Die letzte Rauschquelle: der Separator selbst

Nach dem Gate und der relativen Anhebung blieb ein Restschleier bei 15,8–20 kHz.
Die Ursache war nicht die Bandbreiten-Stufe:

| | > 16 kHz (rel. 300–3000 Hz) |
|---|---|
| Quelle | **−107,0 dB** |
| Summe der vier Demucs-Stems | **−62,0 dB** |
| — vocals | −68,2 |
| — drums | −47,9 |
| — bass | **−42,7** |
| — other | −59,0 |

Die Quelle steht oberhalb 15,8 kHz bei digitalem Nullpunkt. Die Separation
erzeugt dort **+45 dB** — sie gibt nicht Stille aus, sondern ihre eigene
Rechenrückstände, und der Bass-Stem mit −42,7 dB ist der schlimmste. Alles
Nachgelagerte hielt das für Signal: die Höhenstufe hob es weitere 25 dB an, der
Limiter zog es nach vorn.

**Behebung:** ein Brickwall-Tiefpass 16. Ordnung bei 15,2 kHz, direkt an der
Stelle, wo die Stems summiert werden — vor allem anderen. Er entfernt den
Artefakt und sonst nichts, weil die Quelle darüber nichts enthält.

Ergebnis, > 16 kHz:

| Song | Quelle | vorher | **jetzt** |
|---|---|---|---|
| I Need You | −107,0 | −50,5 | **−126,0** |
| Lonesome Feeling | −107,7 | — | **−120,4** |
| Put It On | −105,8 | — | **−122,5** |
| Pussycat | −108,5 | — | **−125,6** |

Der Master ist dort jetzt **leiser als die Quelle**. Das ist richtig so: es gibt
nichts zu hören, und was nicht da ist, soll auch nicht verstärkt werden.

## Endstand, alle vier Songs

| Song | Rauschen in Pausen: Quelle | ausgeliefert | **jetzt** | Höhengewinn 8–14 kHz |
|---|---|---|---|---|
| I Need You | −36,1 | −11,4 | **−30,4** | +5,0 dB |
| Lonesome Feeling | −34,1 | −18,1 | **−25,8** | +6,8 dB |
| Put It On | −24,2 | −5,9 | **−13,2** | +5,3 dB |
| Pussycat | −23,2 | −3,2 | **−14,3** | +8,7 dB |

Song 1 liegt 5,7 dB über der Quelle, Song 3 und 4 etwa 10 dB — dort ist das
Oberband der Quelle selbst schon bei −24 dB, jedes Anheben verstärkt
zwangsläufig mit. Das ist die Untergrenze dieses Materials, nicht ein Mangel der
Umsetzung.
