# PDF Analyse Tool

Dieses Programm ermöglicht es, gezielt Fragen zu einem PDF-Dokument zu stellen und erhält darauf präzise, KI-generierte Antworten. Es nutzt eine Vektordatenbank, um relevante Textabschnitte aus dem PDF zu finden, und kombiniert diese mit einem Sprachmodell für die Antwortgenerierung.

1. Muss vector.py ausgeführt werden, um die Datenbank zu generieren
2. Muss main.py ausgeführt werden.

## Stärken

- Beantwortet spezifische, inhaltsbezogene Fragen sehr präzise.
- Findet relevante Textstellen im PDF automatisch.
- Einfache, interaktive Bedienung über die Konsole.

## Einschränkungen

- Allgemeine oder globale Fragen (z.B. „Fasse das gesamte PDF zusammen“) werden nur auf Basis einzelner Textabschnitte beantwortet, nicht mit Überblick über das ganze Dokument.
- Tabellen, Bilder und komplexe Layouts werden nicht berücksichtigt.
- Die Qualität der Antwort hängt von der Qualität der Textextraktion und Chunk-Bildung ab.