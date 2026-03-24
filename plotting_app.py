import re
import base64
import io
import traceback
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

# ─── App init ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="PIScO Data Explorer")

# Server-side cache: maps cache_key → DataFrame
# Raw 1 M-row DataFrames stay here; only small summaries are ever sent to the browser.
_data_cache: dict = {}

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_ROOT = "/mnt/filer"
DEFAULT_CACHE_ROOT = "/mnt/filer/M181/M181_Profiles_cached"
MICRONS_PER_PIXEL = 22.5
DEPTH_MM = 500

# ─── Utility functions ────────────────────────────────────────────────────────

def extract_pressure_from_filename(filename: str):
    """Return pressure in bar from a PIScO PNG filename, or None."""
    match = re.search(r'(\d+\.\d+)bar', filename)
    return float(match.group(1)) if match else None


def extract_timestamp_from_filename(filename: str):
    """Return image timestamp token like YYYYMMDD-HHMMSSff from filename, or None."""
    match = re.search(r'(\d{8}-\d{8})', filename)
    return match.group(1) if match else None


def parse_timestamp_to_iso(timestamp_token: str):
    """Convert YYYYMMDD-HHMMSSff token to ISO datetime string, or None."""
    if not timestamp_token:
        return None
    try:
        dt = datetime.strptime(timestamp_token, "%Y%m%d-%H%M%S%f")
        return dt.isoformat()
    except ValueError:
        return None


def parse_timestamp_to_unix_s(timestamp_token: str):
    """Convert YYYYMMDD-HHMMSSff token to Unix epoch seconds, or None."""
    if not timestamp_token:
        return None
    try:
        dt = datetime.strptime(timestamp_token, "%Y%m%d-%H%M%S%f")
        return dt.timestamp()
    except ValueError:
        return None


def get_mask_radius(profile_path: Path):
    """Read mask_radius (pixels) from {profile}_Results/settings.csv."""
    profile_name = profile_path.name
    settings_file = profile_path / f"{profile_name}_Results" / "settings.csv"
    print(f"[SETTINGS] Looking for: {settings_file}  exists={settings_file.exists()}")
    if not settings_file.exists():
        return None
    try:
        df = pd.read_csv(settings_file)
        if "Field Name" in df.columns and "Value" in df.columns:
            row = df[df["Field Name"] == "mask_radius"]
            if not row.empty:
                val = float(row["Value"].iloc[0])
                print(f"[SETTINGS] mask_radius = {val}")
                return val
    except Exception as e:
        print(f"[SETTINGS] Error: {e}")
    return None


def calculate_volume(mask_radius_pixels: float) -> float:
    """Return sampled volume per image in litres."""
    radius_mm = (mask_radius_pixels * MICRONS_PER_PIXEL) / 1000
    return np.pi * (radius_mm ** 2) * DEPTH_MM / 1_000_000


def _safe_name(name: str) -> str:
    """Filesystem-safe cruise/profile key part."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name or "").strip())


def _cache_candidates(cache_root: str, profile_name: str, cruise: str = None) -> list:
    """Return candidate CSV cache paths in lookup priority order."""
    if not cache_root or not profile_name:
        return []
    root = Path(cache_root).expanduser()
    profile_key = _safe_name(profile_name)
    candidates = []
    if cruise:
        candidates.append(root / _safe_name(cruise) / profile_key / "image_metadata.csv")
    candidates.append(root / "_global" / profile_key / "image_metadata.csv")
    return candidates


def _resolve_cache_csv(cache_root: str, profile_name: str, cruise: str = None) -> Path:
    """Resolve first matching cache CSV path for a profile, if any."""
    for candidate in _cache_candidates(cache_root, profile_name, cruise):
        if candidate.exists():
            return candidate
    root = Path(cache_root or "").expanduser()
    profile_key = _safe_name(profile_name)
    if root.exists():
        matches = sorted(root.glob(f"*/{profile_key}/image_metadata.csv"))
        if matches:
            return matches[0]
    return None


def _volumes_from_metadata_df(df_meta: pd.DataFrame, profile_path: Path = None) -> dict:
    """Convert image metadata rows into {pressure_dbar: sampled_volume_L}."""
    if df_meta.empty or "pressure_dbar" not in df_meta.columns:
        return None

    vol_per_image = None
    if "vol_per_image_L" in df_meta.columns:
        series = pd.to_numeric(df_meta["vol_per_image_L"], errors="coerce").dropna()
        if not series.empty and float(series.iloc[0]) > 0:
            vol_per_image = float(series.iloc[0])

    if vol_per_image is None and "mask_radius_pixels" in df_meta.columns:
        series = pd.to_numeric(df_meta["mask_radius_pixels"], errors="coerce").dropna()
        if not series.empty and float(series.iloc[0]) > 0:
            vol_per_image = calculate_volume(float(series.iloc[0]))

    if vol_per_image is None and profile_path is not None:
        mask_radius = get_mask_radius(profile_path)
        if mask_radius is not None:
            vol_per_image = calculate_volume(mask_radius)

    if vol_per_image is None:
        return None

    pressure_counts = pd.to_numeric(df_meta["pressure_dbar"], errors="coerce").dropna().value_counts().to_dict()
    if not pressure_counts:
        return None
    return {float(p): c * vol_per_image for p, c in pressure_counts.items()}


def create_image_metadata_csv(
    profile_path: Path,
    force_regenerate: bool = False,
    output_csv_path: Path = None,
    cruise: str = None,
):
    """
    Scan PNG files in profile and save metadata to a CSV cache.
    Returns path to the created/updated CSV file.

    CSV format:
            image_filename | image_timestamp | image_datetime_iso | image_unix_s | pressure_bar | pressure_dbar | mask_radius_pixels | vol_per_image_L
    """
    profile_name = profile_path.name
    if output_csv_path is None:
        results_dir = profile_path / f"{profile_name}_Results"
        results_dir.mkdir(exist_ok=True)
        csv_path = results_dir / "image_metadata.csv"
    else:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force_regenerate:
        print(f"[CACHE] Using existing: {csv_path}")
        return csv_path

    mask_radius = get_mask_radius(profile_path)
    vol_per_image = calculate_volume(mask_radius) if mask_radius is not None else np.nan

    png_dir = profile_path / "PNG"
    if not png_dir.exists():
        png_dir = next((p for p in profile_path.glob("**/PNG") if p.is_dir()), None)

    if not png_dir or not png_dir.exists():
        print(f"[CACHE] No PNG directory found for {profile_name}")
        return None

    png_files = list(png_dir.glob("*.png"))

    if not png_files:
        print(f"[CACHE] No PNG files found in {png_dir}")
        return None

    print(f"[CACHE] Scanning {len(png_files)} PNGs for {profile_name}...")
    records = []
    for img_path in png_files:
        p_bar = extract_pressure_from_filename(img_path.name)
        ts = extract_timestamp_from_filename(img_path.name)
        ts_iso = parse_timestamp_to_iso(ts)
        ts_unix = parse_timestamp_to_unix_s(ts)
        if p_bar is not None:
            p_dbar = round(p_bar * 10, 1)
            records.append({
                "image_filename": img_path.name,
                "image_timestamp": ts,
                "image_datetime_iso": ts_iso,
                "image_unix_s": ts_unix,
                "pressure_bar": p_bar,
                "pressure_dbar": p_dbar,
                "mask_radius_pixels": mask_radius,
                "vol_per_image_L": vol_per_image,
                "profile": profile_name,
                "cruise": cruise or "",
            })

    if not records:
        print("[CACHE] No valid pressure data extracted from filenames")
        return None

    df_meta = pd.DataFrame(records)
    df_meta.to_csv(csv_path, index=False)
    print(f"[CACHE] Created: {csv_path} ({len(records)} images)")
    return csv_path


def load_image_volumes_from_csv(csv_path: Path, profile_path: Path = None) -> dict:
    """
    Load image volume data from cached CSV.
    Returns {pressure_dbar: sampled_volume_L} or None if CSV unavailable/unusable.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    try:
        df_meta = pd.read_csv(csv_path)
        volume_dict = _volumes_from_metadata_df(df_meta, profile_path=profile_path)
        if not volume_dict:
            print(f"[CACHE] CSV lacks usable volume metadata: {csv_path}")
            return None

        print(f"[CACHE] Loaded {len(df_meta)} images from {csv_path} "
              f"({len(volume_dict)} pressure bins)")
        return volume_dict

    except Exception as e:
        print(f"[CACHE] Error reading CSV {csv_path}: {e}")
        return None


def load_image_volumes_from_cache_root(cache_root: str, profile_name: str, cruise: str = None) -> dict:
    """
    Load image volume data from portable cache-root CSVs.
    Returns {pressure_dbar: sampled_volume_L} or None if no cache entry exists.
    """
    csv_path = _resolve_cache_csv(cache_root, profile_name, cruise)
    if not csv_path:
        return None
    return load_image_volumes_from_csv(csv_path, profile_path=None)


def get_cached_profiles(cache_root: str, cruise: str = None) -> list:
    """List profile names available from portable cache-root CSVs."""
    root = Path(cache_root or "").expanduser()
    if not root.exists():
        return []

    profile_names = set()
    if cruise:
        cruise_dir = root / _safe_name(cruise)
        if cruise_dir.exists():
            for csv_file in cruise_dir.glob("*/image_metadata.csv"):
                profile_names.add(csv_file.parent.name)

    global_dir = root / "_global"
    if global_dir.exists():
        for csv_file in global_dir.glob("*/image_metadata.csv"):
            profile_names.add(csv_file.parent.name)

    if not cruise:
        for csv_file in root.glob("*/*/image_metadata.csv"):
            profile_names.add(csv_file.parent.name)

    return sorted(profile_names)


def calculate_profile_volumes(profile_path: Path, cache_root: str = DEFAULT_CACHE_ROOT, cruise: str = None):
    """
    Build {pressure_dbar: sampled_volume_L} using this order:
      1) portable cache-root CSV
      2) local profile CSV
      3) PNG scan fallback
    Returns None if essential data is missing.
    """
    profile_name = profile_path.name

    volumes = load_image_volumes_from_cache_root(cache_root, profile_name, cruise)
    if volumes is not None:
        print(f"[VOLUME_CALC] Used portable cache for {profile_name}")
        return volumes

    local_csv = profile_path / f"{profile_name}_Results" / "image_metadata.csv"
    volumes = load_image_volumes_from_csv(local_csv, profile_path=profile_path)
    if volumes is not None:
        print(f"[VOLUME_CALC] Used local CSV cache for {profile_name}")
        return volumes

    print(f"[VOLUME_CALC] No CSV cache, scanning filesystem for {profile_name}")
    mask_radius = get_mask_radius(profile_path)
    if mask_radius is None:
        print(f"[VOLUME_CALC] Cannot read mask_radius for {profile_name}")
        return None

    vol_per_image = calculate_volume(mask_radius)
    print(f"[VOLUME_CALC] {profile_name}: vol_per_image = {vol_per_image:.4f} L")

    png_dir = profile_path / "PNG"
    if png_dir.exists():
        png_files = list(png_dir.glob("*.png"))
    else:
        png_files = list(profile_path.glob("**/PNG/*.png"))

    if not png_files:
        print(f"[VOLUME_CALC] No PNG files found for {profile_name}")
        return None

    print(f"[VOLUME_CALC] Counting {len(png_files)} PNGs for {profile_name} ...")
    pressure_counts: dict = {}
    for img in png_files:
        p_bar = extract_pressure_from_filename(img.name)
        if p_bar is not None:
            p_dbar = round(p_bar * 10, 1)
            pressure_counts[p_dbar] = pressure_counts.get(p_dbar, 0) + 1

    if not pressure_counts:
        print(f"[VOLUME_CALC] No valid pressures extracted for {profile_name}")
        return None

    volume_dict = {p: c * vol_per_image for p, c in pressure_counts.items()}
    print(f"[VOLUME_CALC] {profile_name}: {len(volume_dict)} pressure bins")

    print("[VOLUME_CALC] Caching metadata locally and in portable cache...")
    create_image_metadata_csv(profile_path, force_regenerate=False, cruise=cruise)
    if cache_root:
        for candidate in _cache_candidates(cache_root, profile_name, cruise):
            create_image_metadata_csv(
                profile_path,
                force_regenerate=False,
                output_csv_path=candidate,
                cruise=cruise,
            )

    return volume_dict


def get_cruise_list(root_path: str) -> list:
    """Return sorted subdirectory names under root_path (one per cruise)."""
    p = Path(root_path)
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def detect_profiles_dir(root_path: str, cruise: str):
    """Auto-detect the PISCO-Profiles subdirectory inside a cruise folder."""
    cruise_dir = Path(root_path) / cruise
    if not cruise_dir.exists():
        return None
    for d in sorted(cruise_dir.iterdir()):
        if d.is_dir() and ("PISCO-Profiles" in d.name or "Profiles" in d.name):
            return str(d)
    return str(cruise_dir)  # fallback: cruise dir itself


def get_available_profiles(base_dir: str) -> list:
    """Return sorted list of profiles that have an EcoTaxa TSV under base_dir."""
    p = Path(base_dir)
    if not p.exists():
        return []
    profiles = []
    for d in p.iterdir():
        if not d.is_dir():
            continue
        tsv = d / f"{d.name}_Results" / "EcoTaxa" / f"{d.name}_ecotaxa.tsv"
        if tsv.exists():
            profiles.append(d.name)
    return sorted(profiles)


def batch_create_image_metadata(
    profiles_dir: str,
    overwrite: bool = False,
    cache_root: str = None,
    cruise: str = None,
) -> dict:
    """
    Batch-generate image metadata CSVs for all profiles in a directory.
    Useful for pre-caching before running the Dash app.
    
    Returns: {profile_name: (success: bool, message: str)}
    """
    profiles_dir_path = Path(profiles_dir)
    if not profiles_dir_path.exists():
        return {"error": f"Profiles directory not found: {profiles_dir}"}
    
    results = {}
    profile_dirs = sorted([d for d in profiles_dir_path.iterdir() if d.is_dir()])
    
    for profile_path in profile_dirs:
        profile_name = profile_path.name
        try:
            local_csv = create_image_metadata_csv(
                profile_path,
                force_regenerate=overwrite,
                cruise=cruise,
            )

            portable_written = 0
            if cache_root:
                for candidate in _cache_candidates(cache_root, profile_name, cruise):
                    created = create_image_metadata_csv(
                        profile_path,
                        force_regenerate=overwrite,
                        output_csv_path=candidate,
                        cruise=cruise,
                    )
                    if created:
                        portable_written += 1

            if local_csv:
                msg = f"Created {local_csv.name}"
                if cache_root:
                    msg += f" + portable cache ({portable_written} file(s))"
                results[profile_name] = (True, msg)
            else:
                results[profile_name] = (False, "No PNG files or PNG directory found")
        except Exception as e:
            results[profile_name] = (False, f"Error: {e}")
    
    return results


def bin_dataframe(df: pd.DataFrame, volume_dicts: dict, edges: list) -> pd.DataFrame:
    """
    Aggregate the raw DataFrame into depth bins.
    Returns a small summary DataFrame ready for plotting.
    """
    labels = [f"{edges[i]}-{edges[i+1]}m" for i in range(len(edges) - 1)]

    df = df.copy()
    df["depth_bin"] = pd.cut(
        df["object_pressure"], bins=edges, labels=labels,
        include_lowest=True, right=False,
    )

    # Sum volumes per bin across all profiles.
    # Note: JSON round-trip via dcc.Store converts float keys → strings; cast back to float.
    total_vol_per_bin = {l: 0.0 for l in labels}
    for profile_name, vdict in volume_dicts.items():
        pressures = np.array([float(p) for p in vdict.keys()])
        vols = np.array(list(vdict.values()), dtype=float)
        bin_indices = pd.cut(pressures, bins=edges, labels=False,
                             include_lowest=True, right=False)
        for bin_idx, vol in zip(bin_indices, vols):
            if not pd.isna(bin_idx):
                total_vol_per_bin[labels[int(bin_idx)]] += vol

    # Aggregate biology
    agg_spec = {"count": ("object_annotation_category", "size"),
                "biovolume_mm3": ("biovolume_mm3", "sum")}

    if "object_esd" in df.columns:
        agg_spec["esd_mean_um"]   = ("object_esd", "mean")
        agg_spec["esd_median_um"] = ("object_esd", "median")
        agg_spec["esd_std_um"]    = ("object_esd", "std")
    if "object_pressure" in df.columns:
        agg_spec["pressure_mean_dbar"] = ("object_pressure", "mean")
    if "profile" in df.columns:
        agg_spec["n_profiles"] = ("profile", "nunique")

    binned = (
        df.groupby(["depth_bin", "object_annotation_category"], observed=True)
        .agg(**agg_spec)
        .reset_index()
    )

    binned["depth_bin_str"] = binned["depth_bin"].astype(str)
    binned["sampled_volume_L"] = binned["depth_bin_str"].map(total_vol_per_bin).fillna(0)

    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    label_to_mid = dict(zip(labels, midpoints))
    binned["depth_bin_midpoint"] = binned["depth_bin"].map(label_to_mid)

    mask = binned["sampled_volume_L"] > 0
    binned["concentration_per_L"] = np.where(mask, binned["count"] / binned["sampled_volume_L"], 0.0)
    binned["biovolume_mm3_per_L"] = np.where(mask, binned["biovolume_mm3"] / binned["sampled_volume_L"], 0.0)

    return binned


def esd_spectrum_dataframe(
    df: pd.DataFrame,
    volume_dicts: dict,
    depth_edges: list,
    esd_min: float = 50.0,
    esd_max: float = 10000.0,
    n_bins: int = 25,
) -> pd.DataFrame:
    """
    Compute biovolume / L vs ESD (µm) spectrum with one row per
    (depth_bin, esd_bin_center).  Uses log-spaced ESD bins.
    """
    required = {"object_esd", "object_pressure"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    depth_labels = [
        f"{depth_edges[i]:.0f}–{depth_edges[i+1]:.0f} dbar"
        for i in range(len(depth_edges) - 1)
    ]

    df = df.copy()
    df["depth_bin"] = pd.cut(
        df["object_pressure"], bins=depth_edges, labels=depth_labels,
        include_lowest=True, right=False,
    )
    df = df.dropna(subset=["depth_bin"])

    # --- Total sampled volume per depth bin (same logic as bin_dataframe) ---
    total_vol_per_bin = {lbl: 0.0 for lbl in depth_labels}
    for _profile, vdict in volume_dicts.items():
        pressures  = np.array([float(p) for p in vdict.keys()])
        vols       = np.array(list(vdict.values()), dtype=float)
        bin_idx    = pd.cut(pressures, bins=depth_edges, labels=False,
                            include_lowest=True, right=False)
        for idx, vol in zip(bin_idx, vols):
            if not pd.isna(idx):
                total_vol_per_bin[depth_labels[int(idx)]] += vol

    # --- Log-spaced ESD bins ---
    esd_edges  = np.logspace(np.log10(max(esd_min, 1e-6)),
                              np.log10(max(esd_max, esd_min + 1)), n_bins + 1)
    esd_centers = np.sqrt(esd_edges[:-1] * esd_edges[1:])   # geometric mean
    esd_widths  = esd_edges[1:] - esd_edges[:-1]             # bin width in µm
    esd_idx_labels = list(range(n_bins))

    df["esd_bin_idx"] = pd.cut(
        df["object_esd"], bins=esd_edges, labels=esd_idx_labels,
        include_lowest=True,
    )
    df = df.dropna(subset=["esd_bin_idx"])
    df["esd_bin_idx"] = df["esd_bin_idx"].astype(int)
    df["esd_center_um"] = df["esd_bin_idx"].map(lambda i: esd_centers[i])

    if "biovolume_mm3" not in df.columns:
        esd_mm = df["object_esd"] / 1000.0
        df["biovolume_mm3"] = (np.pi / 6.0) * esd_mm ** 3

    grouped = (
        df.groupby(["depth_bin", "esd_center_um"], observed=True)
        .agg(
            count=("biovolume_mm3", "size"),
            biovolume_mm3_total=("biovolume_mm3", "sum"),
        )
        .reset_index()
    )
    grouped["total_vol_L"] = grouped["depth_bin"].astype(str).map(total_vol_per_bin).fillna(0)
    grouped = grouped[grouped["total_vol_L"] > 0].copy()
    grouped["biovolume_mm3_per_L"] = grouped["biovolume_mm3_total"] / grouped["total_vol_L"]
    # Normalize by ESD bin width (mm³ L⁻¹ µm⁻¹)
    center_to_width = {esd_centers[i]: esd_widths[i] for i in range(n_bins)}
    grouped["esd_bin_width_um"] = grouped["esd_center_um"].map(center_to_width)
    grouped["biovolume_mm3_per_L_per_um"] = (
        grouped["biovolume_mm3_per_L"] / grouped["esd_bin_width_um"]
    )
    # Abundance (count) per litre and size-normalised abundance
    grouped["count_per_L"] = grouped["count"] / grouped["total_vol_L"]
    grouped["count_per_L_per_um"] = grouped["count_per_L"] / grouped["esd_bin_width_um"]
    return grouped


# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div([
    html.H1("PIScO Data Explorer",
            style={"textAlign": "center", "fontFamily": "sans-serif"}),

    # Stores navigation state and data metadata
    dcc.Store(id="nav-store", data={"root": DEFAULT_ROOT, "cache_root": DEFAULT_CACHE_ROOT, "profiles_dir": ""}),
    dcc.Store(id="data-store"),
    dcc.Store(id="bin-edges-store"),  # holds computed edge list

    html.Div([
        # ── Sidebar ──────────────────────────────────────────────────────────
        html.Div([
            html.H3("Controls"),

            html.Label("Root Path:"),
            html.Div([
                dcc.Input(
                    id="root-path-input", type="text", value=DEFAULT_ROOT,
                    style={"width": "73%", "marginRight": "4%"},
                    debounce=False,
                ),
                html.Button("Scan", id="scan-btn", n_clicks=0,
                            style={"width": "23%"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="root-status",
                     style={"color": "grey", "fontSize": "12px", "marginTop": "3px"}),

            html.Br(),

            html.Label("Portable Cache Root (CSV only):"),
            dcc.Input(
                id="cache-root-input", type="text", value=DEFAULT_CACHE_ROOT,
                style={"width": "100%"}, debounce=False,
            ),
            html.Div(id="cache-status",
                     style={"color": "grey", "fontSize": "12px", "marginTop": "3px"}),

            html.Br(),

            html.Label("Cruise:"),
            dcc.Dropdown(id="cruise-dropdown", placeholder="Scan root path first..."),
            html.Div(id="cruise-status",
                     style={"color": "grey", "fontSize": "12px", "marginTop": "3px"}),

            html.Br(),

            html.Label("Select Profile(s):"),
            dcc.Dropdown(
                id="profile-dropdown",
                options=[],
                multi=True,
                placeholder="Select a cruise first...",
            ),
            html.Div(id="load-status",
                     style={"color": "blue", "fontStyle": "italic", "marginTop": "5px"}),

            html.Hr(),

            html.Label("Or Upload TSV File:"),
            dcc.Upload(
                id="upload-component",
                children=html.Div(["Drag & drop or ", html.A("click to select")]),
                style={
                    "width": "100%", "height": "60px", "lineHeight": "60px",
                    "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
                    "textAlign": "center", "marginBottom": "10px",
                    "fontFamily": "sans-serif", "fontSize": "12px",
                    "backgroundColor": "#f0f0f0",
                },
                multiple=False,
            ),
            html.Label("Match to Server Profile (for volumes):"),
            dcc.Dropdown(
                id="upload-profile-dropdown",
                options=[],
                placeholder="Select a cruise first...",
            ),
            html.Div(id="upload-status",
                     style={"color": "green", "fontStyle": "italic", "marginTop": "5px"}),

            html.Hr(),

            html.Label("Depth Binning (dbar ≈ m):"),
            dcc.RadioItems(
                id="bin-mode",
                options=[
                    {"label": " Presets",  "value": "preset"},
                    {"label": " Segments", "value": "segments"},
                    {"label": " Manual",   "value": "manual"},
                ],
                value="segments",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
                style={"marginBottom": "6px"},
            ),
            # ── Preset picker ────────────────────────────────────────────────
            html.Div(id="bin-preset-div", style={"display": "none"}, children=[
                dcc.Dropdown(
                    id="bin-preset",
                    options=[
                        {"label": "Coarse  — 4 bins (0/200/1k/4k/10k)", "value": "coarse"},
                        {"label": "Medium  — 8 bins (every 50/200/500/1k)", "value": "medium"},
                        {"label": "Fine    — 10/50/100 dbar segments",     "value": "fine"},
                        {"label": "Very fine — 5/25/50 dbar segments",     "value": "veryfine"},
                        {"label": "1 dbar uniform (0–1000)",               "value": "1dbar"},
                        {"label": "10 dbar uniform (0–5000)",              "value": "10dbar"},
                    ],
                    value="fine",
                    clearable=False,
                ),
            ]),
            # ── Segment notation ─────────────────────────────────────────────
            html.Div(id="bin-segments-div", children=[
                html.Div("Format: start:stop:step, …",
                         style={"fontSize": "11px", "color": "#888", "marginBottom": "3px"}),
                dcc.Input(
                    id="bin-segments-input", type="text",
                    value="0:100:10, 100:1000:50, 1000:5000:100",
                    debounce=True,
                    style={"width": "100%"},
                ),
            ]),
            # ── Manual comma list ────────────────────────────────────────────
            html.Div(id="bin-manual-div", style={"display": "none"}, children=[
                dcc.Input(
                    id="bin-edges-input", type="text",
                    value="0, 200, 1000, 4000, 10000",
                    debounce=True,
                    style={"width": "100%"},
                ),
            ]),
            html.Div(id="bin-preview",
                     style={"fontSize": "11px", "color": "#555", "marginTop": "4px"}),

            html.Br(), html.Br(),

            html.Label("Plot Type:"),
            dcc.RadioItems(
                id="plot-type",
                options=[
                    {"label": " Depth Profile",  "value": "depth"},
                    {"label": " ESD Spectrum",    "value": "esd"},
                ],
                value="depth",
                labelStyle={"display": "inline-block", "marginRight": "12px"},
            ),

            html.Br(),

            # ── Depth profile controls ────────────────────────────────────────
            html.Div(id="depth-controls", children=[
                html.Br(),

                html.Label("X-Axis:"),
                dcc.Dropdown(id="x-axis-dropdown", clearable=False),
                html.Label("Y-Axis:"),
                dcc.Dropdown(id="y-axis-dropdown", clearable=False),

                html.Br(),

                html.Label("Color / Group By:"),
                dcc.Dropdown(
                    id="color-dropdown",
                    options=[{"label": "Taxonomic Category",
                              "value": "object_annotation_category"}],
                    value="object_annotation_category",
                    clearable=True,
                ),
            ]),

            # ── ESD spectrum controls ─────────────────────────────────────────
            html.Div(id="esd-controls", style={"display": "none"}, children=[
                html.Label("ESD range (µm):"),
                html.Div([
                    dcc.Input(id="esd-min", type="number", value=50,  placeholder="min",
                              style={"width": "45%", "marginRight": "5%"}),
                    dcc.Input(id="esd-max", type="number", value=10000, placeholder="max",
                              style={"width": "45%"}),
                ], style={"display": "flex"}),
                html.Br(),
                html.Label("Number of ESD bins:"),
                dcc.Slider(
                    id="esd-nbins", min=5, max=60, step=5, value=25,
                    marks={5: "5", 20: "20", 40: "40", 60: "60"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                html.Label("Y-axis metric:"),
                dcc.RadioItems(
                    id="esd-yaxis",
                    options=[
                        {"label": " Biovolume (mm³ L⁻¹)",            "value": "biovolume_mm3_per_L"},
                        {"label": " Biovolume norm. (mm³ L⁻¹ µm⁻¹)", "value": "biovolume_mm3_per_L_per_um"},
                        {"label": " Abundance (# L⁻¹)",              "value": "count_per_L"},
                        {"label": " Abundance norm. (# L⁻¹ µm⁻¹)",  "value": "count_per_L_per_um"},
                    ],
                    value="biovolume_mm3_per_L",
                    labelStyle={"display": "block"},
                ),
            ]),

            html.Br(),

            html.Label("Plot Options:"),
            dcc.Checklist(
                id="plot-options",
                options=[
                    {"label": " Connect points", "value": "lines"},
                    {"label": " Log scale X",    "value": "log_x"},
                    {"label": " Log scale Y",    "value": "log_y"},
                ],
                value=["lines"],
                labelStyle={"display": "block"},
            ),

            html.Br(),

            html.Label("Filter Taxonomic Classes:"),
            dcc.Dropdown(
                id="taxon-filter",
                multi=True,
                placeholder="All classes shown (select to filter)...",
                clearable=True,
            ),

            html.Br(),

            dcc.Checklist(
                id="validated-only",
                options=[{"label": " Validated annotations only", "value": "validated"}],
                value=[],
                labelStyle={"display": "block"},
            ),
        ], style={
            "width": "28%", "flexShrink": "0",
            "padding": "20px", "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #ddd",
            "overflowY": "auto",
        }),

        # ── Main plot + table (right column) ─────────────────────────────────
        html.Div([
            dcc.Loading(
                id="loading-plot",
                type="circle",
                children=dcc.Graph(
                    id="main-plot",
                    style={"height": "87vh"},
                    responsive=True,
                ),
            ),

            # ── Data table ───────────────────────────────────────────────────
            html.H3("Data Preview", style={"marginTop": "20px"}),
            dcc.Loading(
                id="loading-table",
                type="dot",
                children=html.Div(
                    dash_table.DataTable(
                        id="data-table",
                        page_action="none",
                        style_table={"overflowX": "auto", "overflowY": "auto",
                                     "maxHeight": "500px", "minWidth": "100%"},
                        style_cell={"textAlign": "left", "padding": "5px",
                                    "fontFamily": "sans-serif", "whiteSpace": "normal"},
                        style_header={"backgroundColor": "lightgrey", "fontWeight": "bold",
                                      "position": "sticky", "top": 0, "zIndex": 1},
                        fixed_rows={"headers": True},
                    ),
                    style={"overflowX": "auto", "width": "100%"},
                ),
            ),
        ], style={"flex": "1", "padding": "20px", "overflow": "hidden"}),
    ], style={"display": "flex", "alignItems": "stretch"}),
])

# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    [Output("cruise-dropdown", "options"),
     Output("cruise-dropdown", "value"),
     Output("root-status", "children"),
     Output("cache-status", "children")],
    [Input("scan-btn", "n_clicks")],
    [State("root-path-input", "value"),
     State("cache-root-input", "value")],
    prevent_initial_call=True,
)
def scan_root(n_clicks, root_path, cache_root):
    """Populate cruise list by scanning subdirectories of root path."""
    root_path = (root_path or DEFAULT_ROOT).strip()
    cache_root = (cache_root or DEFAULT_CACHE_ROOT).strip()
    cache_dir = Path(cache_root).expanduser()
    cache_count = len(get_cached_profiles(cache_root, cruise=None)) if cache_dir.exists() else 0
    cache_msg = (f"Portable cache: {cache_count} profile(s) at {cache_dir}"
                 if cache_dir.exists() else
                 f"Portable cache will be created at {cache_dir}")

    cruises = get_cruise_list(root_path)
    if not cruises:
        return [], None, f"❌ No directories found at '{root_path}'", cache_msg
    opts = [{"label": c, "value": c} for c in cruises]
    auto = cruises[0] if len(cruises) == 1 else None
    return opts, auto, f"Found {len(cruises)} cruise(s)", cache_msg


@app.callback(
    [Output("nav-store", "data"),
     Output("profile-dropdown", "options"),
     Output("profile-dropdown", "value"),
     Output("cruise-status", "children")],
    [Input("cruise-dropdown", "value"),
     Input("cache-root-input", "value")],
    [State("root-path-input", "value")],
    prevent_initial_call=True,
)
def on_cruise_selected(cruise, cache_root, root_path):
    """Detect profiles dir for the chosen cruise and populate profile dropdowns."""
    cache_root = (cache_root or DEFAULT_CACHE_ROOT).strip()
    if not cruise or not root_path:
        return {"root": root_path or DEFAULT_ROOT, "cache_root": cache_root, "profiles_dir": ""}, [], None, ""
    root_path = root_path.strip()
    profiles_dir = detect_profiles_dir(root_path, cruise)
    if not profiles_dir:
        return ({"root": root_path, "cache_root": cache_root, "cruise": cruise, "profiles_dir": ""},
                [], None, f"❌ Could not find a profiles folder in {root_path}/{cruise}")
    profiles = get_available_profiles(profiles_dir)
    opts = [{"label": p, "value": p} for p in profiles]
    nav = {"root": root_path, "cache_root": cache_root, "cruise": cruise, "profiles_dir": profiles_dir}
    msg = f"✅ {len(profiles)} profiles found in …/{cruise}/{Path(profiles_dir).name}/"
    return nav, opts, None, msg


@app.callback(
    Output("upload-profile-dropdown", "options"),
    [Input("cache-root-input", "value"),
     Input("cruise-dropdown", "value")],
    [State("root-path-input", "value")],
)
def update_upload_profile_options(cache_root, cruise, root_path):
    """Populate upload profile choices from portable cache and (if available) server profiles."""
    cache_root = (cache_root or DEFAULT_CACHE_ROOT).strip()
    cached_profiles = set(get_cached_profiles(cache_root, cruise=cruise))

    server_profiles = set()
    root_path = (root_path or "").strip()
    if cruise and root_path:
        profiles_dir = detect_profiles_dir(root_path, cruise)
        if profiles_dir:
            server_profiles = set(get_available_profiles(profiles_dir))

    merged = sorted(cached_profiles.union(server_profiles))
    return [{"label": p, "value": p} for p in merged]


@app.callback(
    [Output("data-store", "data"),
     Output("load-status", "children"),
     Output("upload-status", "children")],
    [Input("profile-dropdown", "value"),
     Input("upload-component", "contents"),
     Input("upload-profile-dropdown", "value")],
    [State("upload-component", "filename"),
     State("nav-store", "data")],
    prevent_initial_call=True,
)
def load_data(selected_profiles, upload_contents, upload_profile, upload_filename, nav_data):
    """
    Loads TSV(s), caches the raw DataFrame server-side, and returns only a small
    metadata dict to dcc.Store — the browser never handles millions of rows.
    """
    global _data_cache
    nav_data = nav_data or {}
    profiles_dir = nav_data.get("profiles_dir", "")
    cache_root = (nav_data.get("cache_root") or DEFAULT_CACHE_ROOT).strip()
    cruise = nav_data.get("cruise", None)

    # ── Case 1: file upload ───────────────────────────────────────────────────
    if upload_contents and upload_profile:
        print(f"[LOAD] Processing upload for profile: {upload_profile}")
        try:
            _, content_string = upload_contents.split(",", 1)
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\t")

            if "object_pressure" not in df.columns:
                return None, "", "❌ Uploaded TSV missing 'object_pressure' column"
            if "object_esd" not in df.columns:
                df["object_esd"] = np.nan

            df["profile"] = upload_profile
            df["biovolume_mm3"] = (np.pi / 6) * ((df["object_esd"] / 1000) ** 3)

            cache_key = f"upload|{upload_profile}"
            _data_cache[cache_key] = df

            vols = load_image_volumes_from_cache_root(cache_root, upload_profile, cruise=cruise)
            if vols is None and profiles_dir:
                vols = calculate_profile_volumes(
                    Path(profiles_dir) / upload_profile,
                    cache_root=cache_root,
                    cruise=cruise,
                )
            volume_dicts = {upload_profile: vols} if vols else {}
            msg = (f"✅ Uploaded {len(df):,} rows, matched with "
                   f"{upload_profile} ({len(vols) if vols else 0} pressure bins)"
                   if vols else
                   f"⚠️ Uploaded {len(df):,} rows — could not calculate volumes")

            store = {"cache_key": cache_key, "volume_dicts": volume_dicts, "n_rows": len(df)}
            return store, "", msg

        except Exception as e:
            traceback.print_exc()
            return None, "", f"❌ Upload error: {e}"

    # ── Case 2: server profiles ───────────────────────────────────────────────
    if selected_profiles:
        if not profiles_dir:
            return None, "❌ No profiles directory set — please select a cruise first.", ""
        print(f"[LOAD] Loading {len(selected_profiles)} profile(s)")
        try:
            base_dir = Path(profiles_dir)
            all_dfs = []
            volume_dicts = {}

            for name in selected_profiles:
                profile_dir = base_dir / name
                tsv = profile_dir / f"{name}_Results" / "EcoTaxa" / f"{name}_ecotaxa.tsv"
                if not tsv.exists():
                    print(f"[LOAD] TSV not found: {tsv}")
                    continue

                print(f"[LOAD] Reading {tsv}")
                df = pd.read_csv(tsv, sep="\t", skiprows=[1])

                if "object_pressure" not in df.columns:
                    print(f"[LOAD] Missing object_pressure in {name}")
                    continue
                if "object_esd" not in df.columns:
                    df["object_esd"] = np.nan

                df["profile"] = name
                df["biovolume_mm3"] = (np.pi / 6) * ((df["object_esd"] / 1000) ** 3)
                all_dfs.append(df)
                print(f"[LOAD] {name}: {len(df):,} rows")

                vols = calculate_profile_volumes(profile_dir, cache_root=cache_root, cruise=cruise)
                if vols:
                    volume_dicts[name] = vols

            if not all_dfs:
                return None, "No valid profiles found.", ""

            combined = pd.concat(all_dfs, ignore_index=True)
            cache_key = "|".join(sorted(selected_profiles))
            _data_cache[cache_key] = combined
            print(f"[LOAD] Cached {len(combined):,} rows under key '{cache_key}'")

            n_missing = len(selected_profiles) - len(volume_dicts)
            status = f"✅ {len(combined):,} rows from {len(all_dfs)} profile(s)."
            if n_missing:
                status += f" ⚠️ Volumes missing for {n_missing} profile(s)."

            store = {"cache_key": cache_key, "volume_dicts": volume_dicts, "n_rows": len(combined)}
            return store, status, ""

        except Exception as e:
            traceback.print_exc()
            return None, f"❌ Error: {e}", ""

    return None, "No profiles selected.", ""


# ── Bin-edge builder ────────────────────────────────────────────────────

def _segments_to_edges(text: str) -> list:
    """Parse 'start:stop:step, ...' into a sorted unique edge list."""
    edges = set()
    for seg in text.split(","):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad segment '{seg}' — expected start:stop:step")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        if step <= 0 or start >= stop:
            raise ValueError(f"Invalid range in '{seg}'")
        val = start
        while val <= stop + 1e-9:
            edges.add(round(val, 6))
            val += step
    return sorted(edges)


PRESETS = {
    "coarse":   [0, 200, 1000, 4000, 10000],
    "medium":   "0:50:50, 50:200:50, 200:1000:200, 1000:4000:500, 4000:10000:1000",
    "fine":     "0:100:10, 100:1000:50, 1000:5000:100",
    "veryfine": "0:100:5, 100:500:25, 500:2000:50",
    "1dbar":    "0:1000:1",
    "10dbar":   "0:5000:10",
}


@app.callback(
    [Output("bin-preset-div",   "style"),
     Output("bin-segments-div", "style"),
     Output("bin-manual-div",   "style"),
     Output("bin-edges-store",  "data"),
     Output("bin-preview",      "children")],
    [Input("bin-mode",           "value"),
     Input("bin-preset",         "value"),
     Input("bin-segments-input", "value"),
     Input("bin-edges-input",    "value")],
)
def update_bin_edges(mode, preset, segments_str, manual_str):
    show = {"display": "block"}
    hide = {"display": "none"}
    preset_style   = show if mode == "preset"   else hide
    segments_style = show if mode == "segments" else hide
    manual_style   = show if mode == "manual"   else hide

    try:
        if mode == "preset":
            spec = PRESETS.get(preset or "fine", "0:5000:100")
            edges = spec if isinstance(spec, list) else _segments_to_edges(spec)
        elif mode == "segments":
            edges = _segments_to_edges(segments_str or "0:5000:100")
        else:  # manual
            edges = sorted(set(float(x.strip()) for x in (manual_str or "0,5000").split(",")))

        n = len(edges) - 1
        preview = f"→ {n} bin{'s' if n != 1 else ''}, {edges[0]:.0f}–{edges[-1]:.0f} dbar"
        return preset_style, segments_style, manual_style, edges, preview

    except Exception as e:
        return preset_style, segments_style, manual_style, None, f"❌ {e}"


@app.callback(
    [Output("depth-controls", "style"),
     Output("esd-controls",   "style")],
    [Input("plot-type", "value")],
)
def toggle_plot_controls(plot_type):
    """Show only the relevant control panel for the selected plot type."""
    show = {"display": "block"}
    hide = {"display": "none"}
    if plot_type == "esd":
        return hide, show
    return show, hide


@app.callback(
    [Output("taxon-filter", "options"),
     Output("taxon-filter", "value")],
    [Input("data-store", "data")],
)
def update_taxon_options(store_data):
    """Populate the taxon filter dropdown from the loaded data."""
    if not store_data:
        return [], []
    df = _data_cache.get(store_data.get("cache_key"), pd.DataFrame())
    if df.empty or "object_annotation_category" not in df.columns:
        return [], []
    cats = sorted(df["object_annotation_category"].dropna().unique().tolist())
    opts = [{"label": c, "value": c} for c in cats]
    return opts, cats   # all selected by default


@app.callback(
    [Output("x-axis-dropdown", "options"),
     Output("x-axis-dropdown", "value"),
     Output("y-axis-dropdown", "options"),
     Output("y-axis-dropdown", "value")],
    [Input("data-store", "data")],
)
def update_dropdowns(store_data):
    if not store_data:
        return [], None, [], None

    cols = [
        "depth_bin_str", "depth_bin_midpoint", "object_annotation_category",
        "count", "biovolume_mm3", "sampled_volume_L",
        "concentration_per_L", "biovolume_mm3_per_L",
        "esd_mean_um", "esd_median_um", "esd_std_um",
        "pressure_mean_dbar", "n_profiles",
    ]
    return cols, "concentration_per_L", cols, "depth_bin_midpoint"


@app.callback(
    [Output("main-plot", "figure"),
     Output("data-table", "data"),
     Output("data-table", "columns")],
    [Input("data-store", "data"),
     Input("bin-edges-store", "data"),
     Input("plot-type", "value"),
     # depth-profile inputs
     Input("x-axis-dropdown", "value"),
     Input("y-axis-dropdown", "value"),
     Input("color-dropdown", "value"),
     Input("plot-options", "value"),
     Input("taxon-filter", "value"),
     Input("validated-only", "value"),
     # ESD spectrum inputs
     Input("esd-min", "value"),
     Input("esd-max", "value"),
     Input("esd-nbins", "value"),
     Input("esd-yaxis", "value")],
)
def update_dashboard(store_data, edges, plot_type,
                     x_axis, y_axis, color_col, plot_options, taxon_filter,
                     validated_only, esd_min, esd_max, esd_nbins, esd_yaxis):
    plot_options = plot_options or []
    empty_fig = px.scatter(title="No Data Selected")
    if not store_data:
        return empty_fig, [], []

    cache_key    = store_data.get("cache_key")
    volume_dicts = store_data.get("volume_dicts", {})

    df = _data_cache.get(cache_key)
    if df is None or df.empty:
        print(f"[PLOT] Cache miss for key '{cache_key}'")
        return px.scatter(title="Data not in server cache — please reload."), [], []

    # ── Apply validated-only filter ───────────────────────────────────────────
    if validated_only and "validated" in (validated_only or []):
        if "object_annotation_status" in df.columns:
            df = df[df["object_annotation_status"] == "validated"]
        else:
            print("[FILTER] 'object_annotation_status' column not found — skipping validated filter")

    # ── Apply taxon filter ────────────────────────────────────────────────────
    if taxon_filter and "object_annotation_category" in df.columns:
        df = df[df["object_annotation_category"].isin(taxon_filter)]

    print(f"[PLOT] {len(df):,} rows after taxon filter, plot_type={plot_type}")

    if not edges or len(edges) < 2:
        edges = [0, 100, 1000, 5000]

    # ══════════════════════════════════════════════════════════════════════════
    # ESD SPECTRUM branch
    # ══════════════════════════════════════════════════════════════════════════
    if plot_type == "esd":
        esd_min   = float(esd_min   or 50)
        esd_max   = float(esd_max   or 10000)
        esd_nbins = int(esd_nbins   or 25)

        try:
            spec_df = esd_spectrum_dataframe(df, volume_dicts, edges,
                                             esd_min, esd_max, esd_nbins)
        except Exception as e:
            traceback.print_exc()
            return px.scatter(title=f"ESD spectrum error: {e}"), [], []

        if spec_df.empty:
            return px.scatter(title="No data for ESD spectrum (check bins / data)"), [], []

        # Determine a nice colour sequence (one colour per depth bin)
        depth_bins = spec_df["depth_bin"].unique().tolist()
        colors = px.colors.qualitative.Plotly
        mode = "lines+markers" if "lines" in plot_options else "lines"

        esd_yaxis = esd_yaxis or "biovolume_mm3_per_L"
        y_col = esd_yaxis
        _y_meta = {
            "biovolume_mm3_per_L":        ("Biovolume (mm³ L⁻¹)",            "Biovolume/L"),
            "biovolume_mm3_per_L_per_um": ("Biovolume norm. (mm³ L⁻¹ µm⁻¹)", "Norm. biovolume"),
            "count_per_L":                ("Abundance (# L⁻¹)",               "Abundance/L"),
            "count_per_L_per_um":         ("Abundance norm. (# L⁻¹ µm⁻¹)",   "Norm. abundance"),
        }
        y_label, y_hover_label = _y_meta.get(y_col, (y_col, y_col))

        fig = go.Figure()
        for i, dbin in enumerate(depth_bins):
            sub = spec_df[spec_df["depth_bin"] == dbin].sort_values("esd_center_um")
            total_count = int(sub["count"].sum())
            fig.add_trace(go.Scatter(
                x=sub["esd_center_um"],
                y=sub[y_col],
                name=f"{dbin}  (n={total_count:,})",
                mode=mode,
                line=dict(color=colors[i % len(colors)]),
                marker=dict(color=colors[i % len(colors)], size=5),
                customdata=sub[["count", "biovolume_mm3_total", "total_vol_L"]].values,
                hovertemplate=(
                    f"ESD: %{{x:.1f}} µm<br>"
                    f"{y_hover_label}: %{{y:.4e}}<br>"
                    "Count: %{customdata[0]:.0f}<br>"
                    "Total biovolume: %{customdata[1]:.4e} mm³<br>"
                    "Sampled vol: %{customdata[2]:.2f} L"
                    "<extra>%{fullData.name}</extra>"
                ),
            ))

        fig.update_layout(
            template="plotly_white",
            xaxis_title="ESD (µm)",
            yaxis_title=y_label,
            xaxis_type="log" if "log_x" in plot_options else "linear",
            yaxis_type="log" if "log_y" in plot_options else "linear",
            legend_title_text="Depth bin",
            autosize=True,
        )

        table_data = spec_df.to_dict("records")
        table_cols = [{"name": c, "id": c} for c in spec_df.columns]
        return fig, table_data, table_cols

    # ══════════════════════════════════════════════════════════════════════════
    # DEPTH PROFILE branch (binned)
    # ══════════════════════════════════════════════════════════════════════════
    try:
        plot_df = bin_dataframe(df, volume_dicts, edges)
    except Exception as e:
        traceback.print_exc()
        return px.scatter(title=f"Binning error: {e}"), [], []
    if plot_df.empty:
        return px.scatter(title="No data after binning (check bin edges)"), [], []

    cols = list(plot_df.columns)
    if x_axis not in cols:
        x_axis = cols[0]
    if y_axis not in cols:
        y_axis = cols[0]

    mode = "lines+markers" if "lines" in plot_options else "markers"

    fig = px.scatter(
        plot_df, x=x_axis, y=y_axis,
        color=color_col if color_col and color_col in plot_df.columns else None,
        hover_data=cols,
        template="plotly_white",
    )
    fig.update_traces(mode=mode)
    fig.update_layout(autosize=True)

    invert_y = y_axis and ("depth" in y_axis.lower() or "pressure" in y_axis.lower())
    fig.update_yaxes(
        autorange="reversed" if invert_y else True,
        type="log" if "log_y" in plot_options else "linear",
    )
    if "log_x" in plot_options:
        fig.update_xaxes(type="log")

    print(f"[PLOT] Done — {len(plot_df)} points rendered")

    table_data = plot_df.to_dict("records")
    table_cols = [{"name": c, "id": c} for c in cols]
    return fig, table_data, table_cols


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
