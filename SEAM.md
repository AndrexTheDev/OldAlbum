# Die Naht bei 8 kHz — vier Fehlversuche und was es wirklich war

Die Bandbreiten-Stufe muss an das vorhandene Signal anschließen. Das klingt
trivial und war der schwierigste Teil dieses Durchgangs. Vier Versionen, jede
gemessen, jede falsch:

## 1. `max(current, predicted)` — harte Kante
Das erste Verfahren setzte oberhalb 8 kHz einfach den größeren von beiden Werten.
Ergebnis: **+24,5 dB Sprung** in einem einzigen Bin, im Spektrogramm ein schwarzer
Balken bei 7,2–8 kHz gefolgt von einer Kante.

## 2. Kreuzblendung, volle Oktave — Rampe an falscher Stelle
Kreuzblendung über eine Oktave (Start bei 4 kHz). Ergebnis: der Sprung sank auf
6,1 dB, aber **5–8 kHz wurden um bis zu +27 dB angehoben** — das ist eine
Präsenzanhebung, keine Höhenrekonstruktion.

## 3. Kubische Rampe, Vierteloktave — Kerbe blieb
Übergang auf eine Vierteloktave verengt, kubische Rampe. Ergebnis: bei
6,6–7 kHz **−6,1 dB** (Kerbe) und trotzdem 10,6 dB Sprung. Das Verschieben des
Crossovers verschob nur das Problem.

## 4. Die Ursache: die Rauschunterdrückung schnitt in das Rekonstruktionsband
`dsp.spectral_noise_cut` lief mit **−12 dB ab 8 kHz** — genau dort, wo die
Rekonstruktion anfängt. Der Regelkreis glich das aus, indem er überall mehr
Verstärkung verlangte, und das erzeugte die Kerbe vor der Kante.

Der Schnitt hat nur einen Zweck: das Codec-Rauschen im Bereich 8–16 kHz
abzusenken, das 17–26 dB unter dem richtigen Pegel liegt. Die Rekonstruktion
**ersetzt** diesen Inhalt ohnehin. Also:

```python
st = dsp.spectral_noise_cut(st, SR, cut_hz=16000.0, smooth_oct=0.5,
                            amount_db=-6.0)      # nur noch > 16 kHz
```

## Das Ergebnis

Verstärkung pro Frequenzband, Song 1, Median über die Zeit:

| Band | Gewinn |
|---|---|
| 5,0–6,0 kHz | +0,0 dB |
| 6,0–6,6 kHz | +0,3 dB |
| 6,6–7,0 kHz | +3,8 dB |
| 7,0–7,2 kHz | +7,1 dB |
| 7,2–7,5 kHz | +11,4 dB |
| 7,5–7,8 kHz | +16,6 dB |
| 7,8–8,0 kHz | +22,2 dB |
| 8,0–8,3 kHz | +23,2 dB |
| 8,3–8,7 kHz | +21,6 dB |
| 12–16 kHz | +24,4 dB |

**Monoton, ohne Kerbe, größter Sprung 5,6 dB** zwischen Nachbarbandmitten. Und
Song 1 braucht nur noch 26,9 dB Verstärkung statt 31,8 — der Regelkreis
konvergiert in 2 Iterationen statt 6.

## Was das Spektrogramm zeigt und was nicht

Im Spektrogramm bleibt ein dunkleres Band bei 6,5–8 kHz sichtbar. Das ist
**kein Artefakt**: die Quelle selbst fällt dort von +11,5 dB (4–5 kHz) auf
−19,3 dB (7,8–8 kHz) — ein Abfall von 30 dB über weniger als eine Oktave, wie er
für eine 1965er Bandaufnahme typisch ist. Der Master folgt diesem Abfall
(−8,7 dB bei 6–6,6 kHz) und steigt dann durch die Rekonstruktion wieder an.
Gemessen mit Mittelwert **und** Median, beide monoton.
