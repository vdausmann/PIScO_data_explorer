#!/usr/bin/env python3
"""
Utility to batch pre-cache image metadata CSVs for all profiles.

This script scans PNG files in each profile and creates image_metadata.csv
files in the {profile}_Results/ directories. This makes the plotting app
more robust by reducing dependency on network access during visualization.

Usage:
    python cache_image_metadata.py /path/to/profiles/directory
    python cache_image_metadata.py /path/to/profiles/directory --overwrite
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import re
import pandas as pd
import numpy as np


MICRONS_PER_PIXEL = 22.5
DEPTH_MM = 500


def extract_pressure_from_filename(filename: str) -> Optional[float]:
    """Return pressure in bar from a PIScO PNG filename, or None."""
    match = re.search(r'(\d+\.\d+)bar', filename)
    return float(match.group(1)) if match else None


def extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """Return image timestamp token like YYYYMMDD-HHMMSSff from filename, or None."""
    match = re.search(r'(\d{8}-\d{8})', filename)
    return match.group(1) if match else None


def parse_timestamp_to_iso(timestamp_token: Optional[str]) -> Optional[str]:
    """Convert YYYYMMDD-HHMMSSff token to ISO datetime string, or None."""
    if not timestamp_token:
        return None
    try:
        dt = datetime.strptime(timestamp_token, "%Y%m%d-%H%M%S%f")
        return dt.isoformat()
    except ValueError:
        return None


def parse_timestamp_to_unix_s(timestamp_token: Optional[str]) -> Optional[float]:
    """Convert YYYYMMDD-HHMMSSff token to Unix epoch seconds, or None."""
    if not timestamp_token:
        return None
    try:
        dt = datetime.strptime(timestamp_token, "%Y%m%d-%H%M%S%f")
        return dt.timestamp()
    except ValueError:
        return None


def get_mask_radius(profile_path: Path) -> Optional[float]:
    """Read mask_radius (pixels) from {profile}_Results/settings.csv."""
    profile_name = profile_path.name
    settings_file = profile_path / f"{profile_name}_Results" / "settings.csv"
    if not settings_file.exists():
        return None
    try:
        df = pd.read_csv(settings_file)
        if "Field Name" in df.columns and "Value" in df.columns:
            row = df[df["Field Name"] == "mask_radius"]
            if not row.empty:
                val = float(row["Value"].iloc[0])
                return val
    except Exception as e:
        print(f"  ⚠️  Error reading mask_radius: {e}")
    return None


def create_image_metadata_csv(profile_path: Path, force_regenerate: bool = False) -> Optional[Path]:
    """
    Scan PNG files in profile and save metadata to a CSV cache.
    Returns path to the created/updated CSV file, or None on failure.
    
    CSV format:
            image_filename | image_timestamp | image_datetime_iso | image_unix_s | pressure_bar | pressure_dbar
    """
    profile_name = profile_path.name
    results_dir = profile_path / f"{profile_name}_Results"
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / "image_metadata.csv"
    
    # Check if cache is already valid
    if csv_path.exists() and not force_regenerate:
        print(f"  ℹ️  Skipping (cache already exists)")
        return csv_path
    
    # Find PNG directory
    png_dir = profile_path / "PNG"
    if not png_dir.exists():
        png_dir = next((p for p in profile_path.glob("**/PNG") if p.is_dir()), None)
    
    if not png_dir or not png_dir.exists():
        print(f"  ❌ No PNG directory found")
        return None
    
    # Scan all PNG files
    png_files = list(png_dir.glob("*.png"))
    if not png_files:
        print(f"  ❌ No PNG files found in {png_dir.name}")
        return None
    
    print(f"  📊 Scanning {len(png_files)} PNG files...")
    
    # Extract metadata from filenames
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
            })
    
    if not records:
        print(f"  ❌ No valid pressure data extracted from filenames")
        return None
    
    # Write to CSV
    df_meta = pd.DataFrame(records)
    df_meta.to_csv(csv_path, index=False)
    print(f"  ✅ Created: {csv_path.name} ({len(records)} images)")
    return csv_path


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name or "").strip())


def create_portable_cache_csv(
    profile_path: Path,
    cache_root: str,
    cruise: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Create transportable cache CSV at {cache_root}/{cruise|_global}/{profile}/image_metadata.csv."""
    profile_name = profile_path.name
    cruise_key = _safe_name(cruise) if cruise else "_global"
    target = Path(cache_root).expanduser() / cruise_key / _safe_name(profile_name) / "image_metadata.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        return target

    local_csv = create_image_metadata_csv(profile_path, force_regenerate=overwrite)
    if not local_csv:
        return None

    try:
        df = pd.read_csv(local_csv)
        mask_radius = get_mask_radius(profile_path)
        vol_per_image = (np.pi * ((mask_radius * MICRONS_PER_PIXEL) / 1000) ** 2 * DEPTH_MM / 1_000_000
                         if mask_radius is not None else np.nan)
        df["mask_radius_pixels"] = mask_radius
        df["vol_per_image_L"] = vol_per_image
        df["profile"] = profile_name
        df["cruise"] = cruise or ""
        df.to_csv(target, index=False)
        return target
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache image metadata CSVs for all profiles to make the "
                    "plotting app more robust and faster."
    )
    parser.add_argument(
        "profiles_dir",
        help="Path to directory containing profile folders (e.g., PISCO-Profiles/)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate CSVs even if they already exist"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Optional portable cache root to export transportable CSVs"
    )
    parser.add_argument(
        "--cruise",
        default=None,
        help="Optional cruise name used in portable cache folder structure"
    )
    
    args = parser.parse_args()
    
    profiles_dir_path = Path(args.profiles_dir).resolve()
    if not profiles_dir_path.exists():
        print(f"❌ Error: Profiles directory not found: {profiles_dir_path}")
        sys.exit(1)
    
    # Find all profile directories (containing {name}_Results/)
    profile_dirs = sorted([
        d for d in profiles_dir_path.iterdir()
        if d.is_dir() and (d / f"{d.name}_Results").exists()
    ])
    
    if not profile_dirs:
        print(f"⚠️  No profiles found in: {profiles_dir_path}")
        print(f"    (looking for directories containing {{name}}_Results/)")
        sys.exit(1)
    
    print(f"\n📁 Processing {len(profile_dirs)} profiles in {profiles_dir_path.name}/ ...\n")
    
    results = {}
    for i, profile_path in enumerate(profile_dirs, 1):
        profile_name = profile_path.name
        print(f"[{i}/{len(profile_dirs)}] {profile_name}")
        
        try:
            csv_path = create_image_metadata_csv(profile_path, force_regenerate=args.overwrite)
            if csv_path:
                if args.cache_root:
                    portable = create_portable_cache_csv(
                        profile_path,
                        cache_root=args.cache_root,
                        cruise=args.cruise,
                        overwrite=args.overwrite,
                    )
                    if portable:
                        results[profile_name] = ("success", f"CSV + portable cache ({portable})")
                    else:
                        results[profile_name] = ("warning", "Local CSV ok, portable cache failed")
                else:
                    results[profile_name] = ("success", "CSV created/exists")
            else:
                results[profile_name] = ("warning", "No PNG data")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[profile_name] = ("error", str(e))
    
    # Summary
    print(f"\n{'='*60}")
    success = sum(1 for s, _ in results.values() if s == "success")
    warnings = sum(1 for s, _ in results.values() if s == "warning")
    errors = sum(1 for s, _ in results.values() if s == "error")
    
    print(f"Summary: {success} ✅ | {warnings} ⚠️  | {errors} ❌")
    
    if errors > 0:
        print(f"\nFailed profiles:")
        for name, (status, msg) in results.items():
            if status == "error":
                print(f"  • {name}: {msg}")
    
    print(f"\n💡 Tip: Run the plotting app with: python plotting_app.py")
    print(f"   It will now load volumes from the cached CSV files.")
    print()
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
