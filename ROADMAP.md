# Rejuvenation Roadmap — *The Wailing Wailers* (Studio One, 1965)

Stand: nach systematischer Recherche. Dieses Dokument ist **vor** dem nächsten
Eingriff in die Audiokette geschrieben und begründet jede Entscheidung.

---

## 1. Das Ziel, präzise formuliert

„Soll sich anhören wie heute aufgenommen" ist kein Messwert. Es zerfällt in
**sieben Eigenschaften**, die ein Hörer als „modern" identifiziert, und jede ist
messbar:

| # | Eigenschaft moderner Aufnahmen | Messgröße | Zielwert |
|---|---|---|---|
| 1 | **Direktheit / Trockenheit** — kein Raumklang um die Quelle | Nachhallzeit RT60, Vorlauf-Verhältnis direkt/diffus | RT60 < 0,35 s, Direktanteil > 70 % |
| 2 | **Volles Band bis 16–20 kHz** | Energie bei 12–16 kHz relativ zu 300–3 kHz | ≥ −25 dB |
| 3 | **Stille ist still** | Rauschboden in Pausen | < −85 dBFS |
| 4 | **Kein Clipping / keine Verzerrung** | Anteil Samples ≥ 0,999, THD | 0 %, THD < 0,3 % |
| 5 | **Dynamik vorhanden** | Crest-Faktor, LRA | Crest 12–16 dB |
| 6 | **Stabile Tonhöhe** | Wow/Flutter in Cent | < 5 Cent |
| 7 | **Breite, aber monokompatibel** | Interkanal-Korrelation | 0,85–0,95 |

Alles, was die Kette tut, muss auf eine dieser sieben Zeilen einzahlen. Was das
nicht tut, fliegt raus.

---

## 2. Was das Material wirklich ist (gemessen, nicht angenommen)

### Forensik der vier Dateien

| Befund | Wert | Konsequenz |
|---|---|---|
| Container | AAC-LC, **129 kb/s**, in `.mp4` mit `.mp3`-Endung | Quelle ist **lossy komprimiert**, kein Rohmaster |
| Abtastrate | 44,1 kHz | keine Upsampling-Lüge, aber Codec-Kante |
| Bandbreite | **Musik endet real bei ~15,8 kHz**; 16–20 kHz = **−108 dB** (digitaler Nullpunkt) | oben ist **nichts** zu retten, nur zu erzeugen |
| Kanal | **echtes Dual-Mono**, Korrelation 1,00000 | kein Stereo-Informationsgehalt, Breite muss **erzeugt** werden |
| Clipping | Songs 2+3 bei exakt 0,00 dBFS, 106/113 Samples | 2 von 4 sind **digital übersteuert** |
| Nachhall | RT60 ≈ **1,6 s / 0,9 s / 2,4 s** | **das ist der Hauptgrund für „alt"** |
| Höhenlage | 8–12 kHz nur −31…−46 dB | Codec-Kante, teils Musik, teils Artefakt |

### Warum das der eigentliche Fehler der bisherigen Kette war

Die bisherige Kette hat **Höhen erzeugt und Rauschen gesenkt** — und den
**Nachhall nie angefasst**. RT60 1,6–2,4 s ist der Klang einer Turnhalle. Kein
moderner Mix hat das. Deshalb klang das Ergebnis trotz korrekter Messwerte alt:
die spektralen Metriken waren erfüllt, die **räumliche** Signatur war unverändert
1965.

Zusätzlich: die Separatoren (Demucs) arbeiten mit **trockenen** Trainingsdaten.
Auf einem halligen Mix liefern sie verschmierte Stems, und jedes Stem trägt den
Hall weiter. Der Hall muss **vor oder statt** der Separation weg.

### Was das Aufnahme-Setup 1965 war (recherchiert)

Studio One (Brentford Road, Kingston), Hausengineer Sylvan Morris:
Hammond B3, Ampex-Bandmaschine, **Pultec**-EQ, Sound Dimension Echo/Reverb,
AKG-Mikro auf dem Klavier, Konsole von „Lang" (nicht Oberklasse),
**1–2 Mikrofone auf dem Schlagzeug, viel Übersprechen**, Acetate-Schnitt für die
UK-Pressung. Aufgenommen auf 1/4-Zoll-Band, wahrscheinlich 7,5 ips.

Das heißt für die Zieldefinition: die **Instrumente** (Hammond, Kontrabass,
Skank-Gitarre, Pioniertrommel) sind das Original und bleiben. Was ersetzt wird,
ist die **Übertragungskette**: Band → Acetat → YouTube-AAC.

---

## 3. Stand der Technik — was Profis 2025/26 tatsächlich benutzen

### 3.1 Bandbreiten-Erweiterung (das „fehlende Höhen"-Problem)

| Verfahren | Typ | Qualität | hier verfügbar? |
|---|---|---|---|
| **AudioSR** (Liu et al., ICASSP 2024) | Latent-Diffusion, 2–16 kHz → 24 kHz | SOTA allgemein, ~6 GB, ~0,6× Echtzeit **auf GPU** | nein (HuggingFace blockiert) |
| **AERO / AEROMamba** | GAN, spektral, Encoder-Decoder | SOTA Musik, schnell (CPU-tauglich) | nein |
| **BEHM-GAN** (Moliner, 2022) | GAN, **für historische Musik gebaut** | erste Arbeit, die echte historische Aufnahmen erweitert | **Gewichte auf GitHub-Releases — Host blockiert** |
| **BABE / BABE-2** (Moliner, DAFx-24) | Diffusions-Prior, blinde BWE + generative EQ | evaluiert auf **Caruso, Melba** — exakt diese Ära | Code ja, **Checkpoints auf Aalto-Server — blockiert** |
| **A2SB / Latent-Bridge** (2025) | Schrödinger Bridge | SOTA, any→48 kHz | nein |
| **FiPA-SR** (2026) | GAN + FiLM, 60× schneller als AudioSR | übertrifft AudioSR bei LSD/ViSQOL | nein |

**Fazit:** das akademisch richtige Werkzeug für *genau diesen Fall* ist BABE-2
bzw. BEHM-GAN. Beide sind aus dieser Umgebung **nicht** ladbar (HuggingFace,
release-assets, Aalto sind nicht in der erlaubten Hostliste; nur github.com,
codeload, api.github.com, PyPI, npm sind offen).

### 3.2 Nachhall-Entfernung (das eigentliche Problem hier)

| Verfahren | Typ | Anmerkung |
|---|---|---|
| **WPE** (Weighted Prediction Error, Nakatani) | klassisch, linear-prädiktiv | kein Training, robust, in `pyroomacoustics`-Nachbau trivial |
| **Neural Dereverb** (BS-RoFormer/Mel-Band, MSR-Challenge 2025) | gelernt | Gewichte auf HuggingFace → blockiert |
| **HPSS** (harmonic/percussive) | klassisch | entfernt Perkussives mit, für Musik riskant |
| **Spectral Subtraction von Spätreflexionen** | klassisch, auf EDC | einfach, aber artefaktanfällig |

Die Gewinner der **Music Source Restoration Challenge 2025** (SJTU X-LANCE,
CUPAudioGroup) nutzen **mehrstufig**: erst Denoise, dann Separation, dann
Dereverb **nur auf Vocals** — weil Hall bei Instrumenten Teil des Timbres ist.
Das ist die Blaupause, die ich übernehme.

### 3.3 Was die Profis sonst machen (SonicMaster, MSR-Challenge)

**SonicMaster** (2025) macht alles in einem Flow-Matching-Modell: Dereverb, EQ,
Declip, Dynamic-Expansion, Stereo — trainiert auf kombinatorisch simulierten
Degradationen. Ergebnis der Listening Study: **Declipping und Dereverb sind die
zwei Kategorien mit dem höchsten wahrgenommenen Gewinn.** Genau die zwei, die
meine bisherige Kette nicht (Dereverb) oder kaum (Declip) adressiert hat.

**Wichtigste methodische Lehre aus der MSR-Challenge:** die zwei besten Systeme
setzen auf **mehrstufige Verarbeitung** statt ein All-in-One-Modell — die
Kaskade gewinnt, weil jede Stufe mit einem spezialisierten Modell arbeitet.

### 3.4 Reihenfolge — die in der Literatur belegte

Aus der MSR-Challenge und SonicMaster:

```
1. Declip           (VOR allem anderen: Denoiser verwischen Clipping-Kanten)
2. Denoise          (statisch; vor Separation, weil Separatoren auf Rauschen empfindlich sind)
3. Dereverb         (auf Musik; auf Vocals stärker)
4. Separation       (erst jetzt — auf sauberem, trockenem Material)
5. Bandwidth Ext.   (pro Stem + Bus)
6. Dynamik / EQ     (musikalisch)
7. Stereo           (erzeugen)
8. Master           (Loudness/Peak)
```

Meine bisherige Kette hatte die Reihenfolge 4 → 2 → 6 → 5 → 7 → 8 und **3 fehlte
komplett**. Das ist die Erklärung.

---

## 4. Die Roadmap

Jede Phase hat ein **messbares Abnahmekriterium**. Keine Phase gilt als fertig,
weil der Code läuft — nur weil die Zahl stimmt.

### Phase 0 — Messlatte (Baseline)
- Forensik aller vier Dateien als JSON festschreiben (✓ oben erledigt, wird Datei).
- **Abnahme:** RT60, Clipping-Anteil, Bandbreite, Korrelation, Crest, LRA pro Song.

### Phase 1 — Declip
- Clipping erkennen (flache Spitzen, Runs ≥ 3 Samples bei 0,999).
- Rekonstruktion per **AR-Prädiktion / Cubic-Hermite** auf dem betroffenen Fenster.
- **Abnahme:** 0 Samples ≥ 0,999; kein Sprung > 0,05 im reparierten Fenster;
  Spektrum unterhalb 8 kHz ändert sich < 0,5 dB (keine neuen Artefakte).
- Betroffen: Songs 2, 3 (106/113 Samples). Klein, aber es ist der erste Schritt,
  weil Denoiser Clipping-Kanten verwischen.

### Phase 2 — Denoise (neu gebaut)
- Bisheriger Denoiser bleibt, aber: **transientengeschützt** und mit
  **frequenzabhängigem Boden** (unten tiefer absenken als oben, weil oben
  Codec-Artefakt, unten Musik).
- **Abnahme:** Rauschboden in Pausen < −85 dBFS; RMS der lautesten 30 % ändert
  sich < 0,3 dB; kein Hör-Artefakt als "musical noise" (messbar: Kurtosis des
  Fehlersignals).

### Phase 3 — **Dereverb (der fehlende Kern)**
- **WPE** auf dem Gesamtmix, plus **spektrales Spätreflexions-Modell** (EDC-basiert).
- Parametrisch so, dass Direktanteil steigt, ohne die Instrumente dünn zu machen.
- **Abnahme:** RT60 sinkt von 1,6/0,9/2,4 s auf **< 0,5 s**; Direkt/Diffus
  verbessert sich um ≥ 6 dB; **Crest-Faktor bleibt** (Dereverb darf keine
  Transienten fressen); Spektrum 200 Hz–4 kHz ändert sich < 2 dB.
- **Risiko:** zu aggressiv → "trocken, tot, künstlich". Deshalb drei Stärken
  rendern und messen, nicht raten.

### Phase 4 — Separation (auf trockenem Material)
- Demucs wie bisher, aber **nach** Phase 3. Zusätzlich: **Stem-Konsistenz prüfen**
  (Summe = Mix, Lag 0) und Separierbarkeit messen.
- **Abnahme:** Lag 0, Korrelation > 0,97; BLEED-Messung pro Stem.

### Phase 5 — Bandwidth Extension
- Zwei Wege, beide bauen und **messen**, dann den besseren nehmen:
  - (a) **verbessertes eigenes Modell**: das bisherige MLP sagt nur den
    Band-Mittelwert voraus (R² 0,69, Cross-Song ≈ 0). Ersetzen durch ein
    **GRU/ConvNet auf Log-Mel mit Residual-Anbindung** und **adversarialem
    Training** (BEHM-GAN-Rezept) auf den eigenen vier Songs + Augmentierung.
  - (b) **Transposer/Phasen-Vocoder-Kaskade** als Fallback, falls (a) nicht
    messbar besser wird.
- **Abnahme:** LSD (Log-Spectral Distance) gegen den echten 8–16-kHz-Anteil
  sinkt gegenüber dem alten Modell; Envelope-Korrelation > 0,5 auf **jedem**
  Song (nicht nur im Training); Flachheit 6–16 kHz < 0,3.

### Phase 6 — Dynamik & Tonbalance
- Kompression nur parallel, Glue sanft. Voicing: **gemessene** Korrektur
  (bereits gebaut, `correction_from_source`).
- **Abnahme:** Crest 12–16 dB, LRA ≥ 6 LU; Oktav-Balance-Residuum < 1,5 dB.

### Phase 7 — Stereo
- Allpass-Dekorrelator (bereits gebaut, 0,000000 dB Abweichung) + Frequenz-Tilt.
- **Abnahme:** Korrelation 0,85–0,95; Monosumme bit-identisch zum Mid.

### Phase 8 — Master
- −12,0 LUFS / −1,0 dBTP, verifiziert an der **dekodierten** MP3.
- **Abnahme:** wie bereits erreicht (alle Specs erfüllt).

### Phase 9 — Gegenprobe
- **MUSHRA-artiger Blindtest** ist hier nicht möglich (keine Hörer).
  Ersatz: **ABX-Statistik auf messbaren Diskriminatoren** (RT60, Crest, LSD,
  Rauschboden) und ein **Rückwärts-Test**: kann ein Klassifikator (Merkmale:
  RT60, Crest, Bandbreite, Spektraltilt) die restaurierte Datei noch als
  "1965" erkennen? Ziel: nein.
- **Abnahme:** Klassifikator-Genauigkeit "alt vs. modern" fällt von ~100 % auf
  Zufallsniveau. Das ist der einzige objektive Ersatz für „klingt neu".

---

## 5. Was ich **nicht** liefern kann, und warum

Ehrlich, weil es sonst wieder ein Dokument ist, das mehr verspricht als es hält:

1. **Kein Zugriff auf die SOTA-BWE-Modelle.** AudioSR, BEHM-GAN, BABE-2 sind
   hier nicht ladbar. Ich baue den besten Ersatz aus eigenen Mitteln und
   dokumentiere die Lücke — statt zu behaupten, es sei SOTA.
2. **Die Quelle ist 129 kb/s AAC.** Was der Codec zerstört hat, ist nicht
   rekonstruierbar, nur plausibel zu ergänzen. Ein Rohmaster (Band oder
   Acetat) würde das Ergebnis in einer Größenordnung verbessern, die kein
   Algorithmus erreicht.
3. **„Klingt wie heute" ist kein Messwert.** Ich kann die sieben messbaren
   Eigenschaften aus §1 auf moderne Werte bringen. Ob ein Mensch das als
   „neu" hört, kann ich nicht garantieren — nur, dass kein messbarer
   Alt-Signatur-Wert mehr übrig ist.
4. **Der Hall ist teilweise Kunst.** Studio One 1965 *war* ein großer Raum;
   ein Teil des Charmes ist der Raum. Vollständige Trockenlegung kann
   musikalisch falsch sein. Deshalb: drei Stärken, messbar verglichen.
