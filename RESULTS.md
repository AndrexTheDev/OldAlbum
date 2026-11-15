# Ergebnisse — zweiter Durchgang

## Was gegen den abgelehnten ersten Durchgang geändert wurde

Eine Stufe, aber die entscheidende: **das Bandbreiten-Modell ist ersetzt**. Der
alte Extender war ein MLP, der die mittlere Pegelhöhe des oberen Bandes gelernt
hatte (Cross-Song R² −0,19 … +0,42). Das neue Modell ist ein rekurrentes
adversariales Netz (Cross-Song R² **+0,41 … +0,61**). Details in `BWE.md`, die
Beweisführung in `FINDINGS.md`.

## Die Zahlen — vorher (abgelehnt) / nachher (neu)

Oberes Band relativ zum 300–3000-Hz-Band, Median-Frame, gemessen am **fertigen
Master** (nicht am Zwischenschritt):

| Song | 12–16 kHz abgelehnt | **12–16 kHz neu** | Gewinn |
|---|---|---|---|
| I Need You | −58,2 dB | **−26,1 dB** | **+32,1 dB** |
| Lonesome Feeling | −50,2 dB | **−26,0 dB** | **+24,2 dB** |
| Put It On | −51,7 dB | **−26,0 dB** | **+25,7 dB** |
| Pussycat | −47,9 dB | **−26,0 dB** | **+21,9 dB** |

Ziel aus `ROADMAP.md` §1 war ≥ −25 dB; erreicht sind −26,0 dB mit Absicht
1 dB Puffer. **Alle vier Songs liegen jetzt auf demselben Wert** statt 10 dB
auseinander — das ist der Regelkreis, nicht der Zufall.

| Song | 8–12 kHz | 16–20 kHz | Crest | Stereo-Korr. |
|---|---|---|---|---|
| I Need You | −21,1 | −85,4 | 12,9 | 0,961 |
| Lonesome Feeling | −21,8 | −81,2 | 12,9 | 0,967 |
| Put It On | −18,7 | −81,3 | 13,4 | 0,962 |
| Pussycat | −10,0 | −73,7 | 13,8 | 0,960 |

## Warum das kein Rauschregler ist

Der entscheidende Test ist die **spektrale Flachheit** des oberen Bandes. Eine
Hiss-Wand hat 0 dB (jedes Bin gleich viel Energie), Musik ist deutlich negativ.

| Song | Flachheit 12–16 kHz | Bedeutung |
|---|---|---|
| I Need You | −3,8 dB | tonal |
| Lonesome Feeling | −3,1 dB | tonal |
| Put It On | −3,3 dB | tonal |
| Pussycat | −4,3 dB | tonal |

Zum Vergleich: die abgelehnten Master lagen bei −3,6 … −5,0 dB, also **ähnlich
tonal** — der neue Master ist nicht flacher, nur lauter. Und die spektrale
Rauheit bleibt unverändert (0,47 → 0,46 dB), es kommt also keine Härte dazu.

## Was sich *nicht* geändert hat, und warum das so bleibt

| Achse | Wert | Ziel | Status |
|---|---|---|---|
| Raum (dB/s) | −34,8 … −48,5 | −250 … −400 | **nicht bearbeitbar** |
| 16–20 kHz | −74 … −85 dB | — | bewusst nicht synthetisiert |
| Dynamik (Block-std) | 8–10 dB | 8–14 | **war schon im Ziel** |
| Clipping | 0 | 0 | erreicht |

Der Raum bleibt. Das ist die dokumentierte Grenze: fünf Verfahren gebaut und
gemessen, der Oracle-Test zeigt, dass spektrale Subtraktion bei diesem
Hallpegel nicht trennt, und die gelernten Modelle liegen auf nicht erreichbaren
Hosts. Alles darüber steht in `STATUS.md` und `FINDINGS.md`.

## Die Stellschraube

`strength_db` in `work/restore.py` → `air_bwe2()`. Gemessen: **exakt 1 dB
Bandanstieg pro dB Einstellung**, auf allen vier Songs (Tabelle in `BWE.md`).
Der Regelkreis sucht die Einstellung pro Song selbst, bis der fertige Master
−26,0 dB trifft, und braucht dafür zwei Durchläufe.

Wer mehr will: `target_db` auf −20 setzen. Wer weniger will: −32. Der Bereich
4–30 dB ist kalibriert und frei von Clipping-Artefakten.
