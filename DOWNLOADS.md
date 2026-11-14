# Direktlinks — alle Dateien

Branch: `arena/f51fb1ee-oldalbum` · Repo: [AndrexTheDev/OldAlbum](https://github.com/AndrexTheDev/OldAlbum/tree/arena/f51fb1ee-oldalbum)

Alle Links sind **direkte Downloads** (raw, ohne GitHub-Oberfläche).
Die Klammern in den Dateinamen sind als `%20`/`(` kodiert — so wie unten angegeben
funktionieren sie unverändert.

---

## 1. Hören: A/B-Vergleich (hier anfangen)

Je 60 s: 30 s Original → 0,6 s Pause → 30 s restauriert, **pegelgleich auf −14 LUFS**.
Man hört also nur den Unterschied, nicht die Lautstärke.

| Song | Direktlink |
|---|---|
| I Need You | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/1_I_Need_You%20(A-B%2030s).wav |
| Lonesome Feeling | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/2_Lonesome_Feeling%20(A-B%2030s).wav |
| Put It On | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/3_Put_It_On%20(A-B%2030s).wav |
| What's New Pussycat | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/4_Whats_New_Pussycat%20(A-B%2030s).wav |

## 2. Verlustfreie Master (24 bit / 44,1 kHz FLAC)

| Song | LUFS | dBTP | Direktlink |
|---|---|---|---|
| I Need You | −12.00 | −1.00 | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/masters/1_I_Need_You%20(restored%20master%2024bit).flac |
| Lonesome Feeling | −12.00 | −1.20 | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/masters/2_Lonesome_Feeling%20(restored%20master%2024bit).flac |
| Put It On | −12.02 | −1.00 | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/masters/3_Put_It_On%20(restored%20master%2024bit).flac |
| What's New Pussycat | −12.00 | −1.00 | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/masters/4_Whats_New_Pussycat%20(restored%20master%2024bit).flac |

## 3. Unterwegs: 320 kb/s MP3 (mit ID3-Tags)

| Song | Direktlink |
|---|---|
| I Need You | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/1_I_Need_You%20(restored).mp3 |
| Lonesome Feeling | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/2_Lonesome_Feeling%20(restored).mp3 |
| Put It On | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/3_Put_It_On%20(restored).mp3 |
| What's New Pussycat | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/restored/4_Whats_New_Pussycat%20(restored).mp3 |

## 4. Vergleichsdiagramme

Spektrogramm (Quelle vs. Master), Spektralkurven, Rauschboden pro Band, Stereo-Bild.

| Song | Direktlink |
|---|---|
| I Need You | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/docs/1_I_Need_You_comparison.png |
| Lonesome Feeling | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/docs/2_Lonesome_Feeling_comparison.png |
| Put It On | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/docs/3_Put_It_On_comparison.png |
| What's New Pussycat | https://raw.githubusercontent.com/AndrexTheDev/OldAlbum/arena/f51fb1ee-oldalbum/docs/4_Whats_New_Pussycat_comparison.png |

## 5. Dokumentation

* [RESTORATION.md](https://github.com/AndrexTheDev/OldAlbum/blob/arena/f51fb1ee-oldalbum/RESTORATION.md) — die Restaurationskette und alle Messwerte
* [VERIFICATION.md](https://github.com/AndrexTheDev/OldAlbum/blob/arena/f51fb1ee-oldalbum/VERIFICATION.md) — Prüfprotokoll: jede Stufe gegen eine bekannte Antwort, und was sie geändert hat
* [Gesamter Branch](https://github.com/AndrexTheDev/OldAlbum/tree/arena/f51fb1ee-oldalbum)

---

## Gemessen an genau diesen Dateien

| | LUFS | True Peak |
|---|---|---|
| FLAC | −12.00 / −12.00 / −12.02 / −12.00 | −1.00 / −1.20 / −1.00 / −1.00 |
| MP3 (dekodiert geprüft) | −12.01 / −12.00 / −12.02 / −12.01 | −1.03 / −1.21 / −1.04 / −1.02 |

Lonesome Feeling liegt bei −1.20 statt −1.00 dBTP: das ist **unter** der Decke, also
spec-konform mit 0,2 dB ungenutztem Headroom. Bei −12.00 LUFS und dem niedrigsten
Crest der vier Songs lässt sich der Peak nicht weiter anheben, ohne die Loudness zu
verletzen.

## Wenn ein Link nicht lädt

`raw.githubusercontent.com` liefert bei großen Dateien gelegentlich einen
Timeout statt eines Fehlers. Alternativen:

* **GitHub-Oberfläche (mit „Download"-Knopf):**
  https://github.com/AndrexTheDev/OldAlbum/tree/arena/f51fb1ee-oldalbum/masters
* **Kommandozeile:**
  ```
  git clone -b arena/f51fb1ee-oldalbum https://github.com/AndrexTheDev/OldAlbum.git
  ```
  (die FLACs sind ~30 MB pro Song, das Repo ~220 MB)
