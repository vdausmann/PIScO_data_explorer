# PIScO Data Explorer – Schnelleinstieg

## Installation

1. Repository klonen oder herunterladen:
   ```bash
   git clone https://github.com/vdausmann/PIScO_data_explorer.git
   cd PIScO_data_explorer
   ```

2. Python-Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

3. **Cached Metadaten entpacken**: Das ZIP mit den Profile-Metadaten in dieses Verzeichnis entpacken (oder den Pfad notieren).

## App starten

```bash
python plotting_app.py
```

Im Terminal wird die URL angezeigt, z.B. `http://127.0.0.1:8050` – dort öffnen.

## Workflow

1. **TSV-Datei hochladen**: EcoTaxa-Export oder Annotationsdatei in die App laden
2. **Cache-Verzeichnis angeben** (optional): Pfad zum entpackten Metadaten-ZIP unter "Cache Root"
3. **Profil auswählen**: Aus der (gekachten) Profilliste ein Profil wählen
4. **Visualisieren**: Daten nach Tiefe binnen und ESD-Spektrum analysieren

Das war's – keine Netzwerkverbindung zu externen Servern notwendig.

## Hinweis

Die erste gültige Profil-Auswahl lädt die Bildvolumina aus den gecachten CSV-Dateien. Falls diese nicht vorhanden sind, versucht die App, sie aus den Bilddateien zu berechnen (kann länger dauern).
