# PIScO Data Explorer

Standalone Dash app for plotting PIScO EcoTaxa TSV data, with portable/offline volume cache support.

## Included files

- `plotting_app.py` — main Dash app
- `cache_image_metadata.py` — builds per-profile metadata cache CSVs
- `requirements.txt` — minimal Python dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run app

```bash
python plotting_app.py
```

Open: `http://127.0.0.1:8050`

## Build portable cache

```bash
python cache_image_metadata.py /path/to/PISCO-Profiles \
  --cache-root /path/to/portable_cache \
  --cruise M181 --overwrite
```

## Use cache in app

1. Start app.
2. Set **Portable Cache Root (CSV only)** to your cache folder.
3. Upload TSV and select matching profile in **Match to Server Profile (for volumes)**.

This upload + cache flow works fully offline once cache and TSV are available locally.
