# Status — ehrlich

## Was seit dem letzten Stand passiert ist

1. **Recherche und Roadmap** (`ROADMAP.md`): State of the Art 2025/26,
   Reihenfolge der Stufen nach MSR-Challenge und SonicMaster, Zielwerte.
2. **Forensik des Materials** (`work/baseline_1965.json`): was die Dateien
   wirklich sind.
3. **Declip und Dereverb gebaut** (`work/dereverb.py`).
4. **Dereverb funktioniert nicht.** Das ist der wichtigste Punkt hier.

## Der Befund, der alles ändert

| Messung | Wert |
|---|---|
| Tail-Energie 30–80 ms nach dem Anschlag, relativ zum Anschlag | **−0,2 bis −0,6 dB** |
| Direkt/Diffus-Verhältnis | **+0,2 bis +0,8 dB** |
| Ein moderner, trocken abgenommener Mix | −20 bis −25 dB bzw. +7 bis +12 dB |

Der Nachhall ist auf diesen Aufnahmen **so laut wie der Anschlag selbst**. Das ist
kein Detail, das ist die Signatur der Aufnahme: ein Mikrofon, großer Raum, kein
Nahbesprechungsabstand. Und es erklärt, warum jede spektrale Verbesserung
wirkungslos blieb — man kann einen Klang nicht enthallten, indem man seine Höhen
anhebt.

## Was ich versucht habe, und warum es gescheitert ist

Fünf Konstruktionen, alle gemessen, alle verworfen:

| # | Ansatz | Ergebnis |
|---|---|---|
| 1 | Kausaler Einpol-Mittelwert des Spektrums als "Tail", Wiener-Gain | Tail-Verhältnis −0,7 → −0,2 dB. Nur Pegel um 3,7 dB gesenkt. **Wirkungslos.** |
| 2 | Blinder IR-Kernel + Energiesubtraktion, Kernel auf Summe 1 normiert | Kernel wurde zum gleitenden Mittelwert des Pegels. **Wirkungslos.** |
| 3 | Kernel mit korrekter Magnitude | Dezimierung um 86 statt 512 → 0,88-s-Raum auf 5,24 s gedehnt, Prädiktion 1,36× größer als das Signal in Pausen. **Erzeugte Hall, wo keiner war.** |
| 4 | Schrittweite korrigiert (hop=512) | RMS neutral, Tail unverändert. **Wirkungslos.** |
| 5 | **Oracle-Test**: perfekte Kenntnis des trockenen Signals | Tail-Verhältnis verschlechtert sich auf +4,0 dB. |

Der Oracle-Test ist der entscheidende: **wenn ein Verfahren mit perfekter
Kenntnis der Wahrheit den Nachhall nicht trennt, kann kein blindes Verfahren es.**
Das heißt nicht, dass die Aufgabe unlösbar ist — es heißt, dass *mein Messaufbau*
ihn nicht trennt: die STFT-Fensterlänge (2048 Samples = 46 ms) ist länger als der
Bereich, den ich messen will (30–80 ms nach dem Anschlag), also verschmiert die
Transformation genau die Information, die ich brauche.

## Was das für die Aufgabe bedeutet

Ein Ein-Kanal-Dereverb dieser Stärke (Nachhall ≈ Direktsignal) ist nach allem, was
die Literatur zeigt, **nicht** mit klassischer Spektralsubtraktion zu lösen. Die
Systeme, die es können (MSR-Challenge 2025: BS-RoFormer mit Dereverb-Modul), sind
gelernt und ihre Gewichte liegen auf HuggingFace — **von hier aus nicht
erreichbar** (siehe `ROADMAP.md` §3, Whitelist).

Damit stehen drei Wege offen, und ich will nicht ohne deine Entscheidung
weiterlaufen:

**A. Ohne Dereverb weiter.** Die anderen Stufen (Declip, Denoise, BWE, Dynamik,
Stereo, Master) sind gebaut und messbar. Der Nachhall bleibt, und ich schreibe
das klar in die Dokumentation, statt es zu verschweigen.

**B. Dereverb als gelerntes Modell selbst trainieren.** Möglich, aber: ich habe
nur vier Songs Material. Ein Modell darauf zu trainieren, das auf genau diesen
vier Songs Hall entfernt, ist Overfitting — es würde messbar besser und
vermutlich schlechter klingen. Ich kann es mit synthetisch verhallten Versionen
der eigenen Songs versuchen (Self-Supervised, wie BEHM-GAN es macht). Das ist
ehrlich, aber der Trainingsdatensatz ist winzig und das Ergebnis unsicher.

**C. Du gibst mir Zugang.** Ein Host in der Whitelist (HuggingFace oder
`research.spa.aalto.fi` für die BABE-2-Checkpoints, oder die
BEHM-GAN-Release-Assets) würde das Problem in einer Stunde lösen, weil das
richtige Modell für genau diese Ära existiert und getestet ist.

## Was ich nicht tun werde

Weiter Code zu schreiben, der "Dereverb" heißt und den Nachhall nicht entfernt.
Das habe ich fünfmal getan; ein sechstes Mal wäre keine Arbeit mehr, sondern
Beschäftigung. Die bisherigen Master sind gegenüber dem letzten Stand unverändert.
