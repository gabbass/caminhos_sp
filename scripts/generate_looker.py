#!/usr/bin/env python
"""
generate_looker.py
===================

This helper script takes the outputs produced by the existing ``run_pipeline.py``
pipeline (found in the ``data_out`` directory) and assembles a new set of
Looker‑friendly data files.  The original pipeline exports separate CSVs
partitioned by status (e.g. ``linhas_existente.csv``, ``linhas_em_projeto.csv``
and so on) and does not include latitude/longitude in a single field.

The purpose of this script is to normalise those outputs into a single file
per domain and to add ``latlon`` or ``latlon_start``/``latlon_end``
columns where appropriate.  A new ``looker`` folder will be created under
``data_out`` containing these consolidated files.  You can safely run this
script after executing ``run_pipeline.py`` — it will leave the original
exports untouched and build new files in the ``looker`` folder.

Key features:

* **Latitude/longitude concatenation:** For stations, line points and
  terminals the script creates a ``latlon`` column in the format
  ``"-23.550520,-46.633308"`` (six decimal places).  This meets the user
  requirement for a single geographic field compatible with Looker Studio,
  PowerBI and QGIS.
* **Line start/end coordinates:** Rail lines do not have explicit
  geographic coordinates in the original tables.  The script derives
  ``latlon_start`` and ``latlon_end`` fields by taking the first and last
  point from the line points table for each ``linha_id``.
* **Unified tables:** All status‑specific CSVs (e.g. ``linhas_existente.csv``,
  ``linhas_em_projeto.csv``) are concatenated into a single file per domain
  with a ``status`` column.  This makes it simpler to build visualisations
  that slice by status without requiring multiple datasets.
* **File copying:** The bus lines and district tables are copied directly
  into the ``looker`` folder.  If the district GeoJSON exists it is
  copied too.

Usage::

    python generate_looker.py --base-dir . --out-dir data_out

The default values assume that you run the script from the repository root
and that the ``run_pipeline.py`` output lives in ``data_out``.  Adjust
``--base-dir`` and ``--out-dir`` if necessary.

"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


def add_latlon(df: pd.DataFrame, lat_col: str, lon_col: str, new_col: str = "latlon") -> pd.DataFrame:
    """Add a ``latlon`` column to ``df`` by combining ``lat_col`` and ``lon_col``.

    Missing or NaN values result in ``None`` entries.  Values are formatted
    with six decimal places, which strikes a balance between precision and
    file size.
    """
    if lat_col in df.columns and lon_col in df.columns:
        df = df.copy()
        df[new_col] = [
            f"{lat:.6f},{lon:.6f}" if pd.notnull(lat) and pd.notnull(lon) else None
            for lat, lon in zip(df[lat_col], df[lon_col])
        ]
    return df


def concat_by_prefix(directory: Path, prefix: str) -> pd.DataFrame:
    """Concatenate all CSVs in ``directory`` whose filenames start with ``prefix``.

    This helper looks for files matching the glob pattern ``f"{prefix}_*.csv"``.
    It returns an empty DataFrame if no files are found.
    """
    pattern = str(directory / f"{prefix}_*.csv")
    files = glob.glob(pattern)
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_looker(base_dir: str = ".", out_dir: str = "data_out") -> None:
    """Entry point for the Looker generator.

    Parameters
    ----------
    base_dir : str
        The directory to search for ``data_out``.  Typically the repository root.
    out_dir : str
        The name of the output directory produced by ``run_pipeline.py``.  Defaults
        to ``data_out``.
    """
    base = Path(base_dir)
    out = base / out_dir
    lines_dir = out / "lines"
    stations_dir = out / "stations"
    line_points_dir = out / "line_points"
    terminals_dir = out / "terminals"
    bus_dir = out / "bus"
    districts_dir = out / "districts"

    looker_dir = out / "looker"
    looker_dir.mkdir(parents=True, exist_ok=True)

    # Process rail lines
    linhas = concat_by_prefix(lines_dir, "linhas")
    if not linhas.empty:
        # Compute start and end lat/lon from line points
        line_points = concat_by_prefix(line_points_dir, "linha_pontos")
        if not line_points.empty and {"linha_id", "latitude", "longitude", "ordem_ponto"}.issubset(line_points.columns):
            # Ensure numeric sorting
            try:
                line_points_sorted = line_points.dropna(subset=["latitude", "longitude"]).sort_values(
                    by=["linha_id", "ordem_ponto"]
                )
            except Exception:
                # If ordem_ponto isn't numeric, sort lexicographically
                line_points_sorted = line_points.dropna(subset=["latitude", "longitude"]).sort_values(
                    by=["linha_id", "ordem_ponto"], key=lambda col: col.astype(str)
                )
            start_pts = (
                line_points_sorted.groupby("linha_id").first().reset_index()[["linha_id", "latitude", "longitude"]]
            )
            end_pts = (
                line_points_sorted.groupby("linha_id").last().reset_index()[["linha_id", "latitude", "longitude"]]
            )
            linhas = linhas.merge(
                start_pts.rename(columns={"latitude": "latitude_start", "longitude": "longitude_start"}),
                on="linha_id",
                how="left",
            )
            linhas = linhas.merge(
                end_pts.rename(columns={"latitude": "latitude_end", "longitude": "longitude_end"}),
                on="linha_id",
                how="left",
            )
            # Compute concatenated latlon fields
            linhas["latlon_start"] = [
                f"{lat:.6f},{lon:.6f}" if pd.notnull(lat) and pd.notnull(lon) else None
                for lat, lon in zip(linhas["latitude_start"], linhas["longitude_start"])
            ]
            linhas["latlon_end"] = [
                f"{lat:.6f},{lon:.6f}" if pd.notnull(lat) and pd.notnull(lon) else None
                for lat, lon in zip(linhas["latitude_end"], linhas["longitude_end"])
            ]
        # Write consolidated lines table
        linhas.to_csv(looker_dir / "linhas.csv", index=False, encoding="utf-8")

    # Process stations
    estacoes = concat_by_prefix(stations_dir, "estacoes")
    if not estacoes.empty:
        estacoes = add_latlon(estacoes, "latitude", "longitude", "latlon")
        estacoes.to_csv(looker_dir / "estacoes.csv", index=False, encoding="utf-8")

    # Process line points
    line_points = concat_by_prefix(line_points_dir, "linha_pontos")
    if not line_points.empty:
        line_points = add_latlon(line_points, "latitude", "longitude", "latlon")
        line_points.to_csv(looker_dir / "linha_pontos.csv", index=False, encoding="utf-8")

    # Process terminals
    terminais = concat_by_prefix(terminals_dir, "terminais")
    if not terminais.empty:
        terminais = add_latlon(terminais, "latitude", "longitude", "latlon")
        terminais.to_csv(looker_dir / "terminais.csv", index=False, encoding="utf-8")

    # Copy bus lines
    bus_file = bus_dir / "linhas_onibus.csv"
    if bus_file.exists():
        shutil.copy2(bus_file, looker_dir / "linhas_onibus.csv")

    # Copy districts
    dist_file = districts_dir / "distritos_od.csv"
    if dist_file.exists():
        shutil.copy2(dist_file, looker_dir / "distritos_od.csv")
    geojson_file = districts_dir / "distritos_od.geojson"
    if geojson_file.exists():
        shutil.copy2(geojson_file, looker_dir / "distritos_od.geojson")

    # Attempt to copy Looker fact tables from existing looker scripts (if present)
    # These files are optional because they are generated by the ``scripts/looker``
    # suite rather than ``run_pipeline``.  We copy them if they exist so that
    # downstream tools can consume everything from a single location.
    for subdir_name in ["od", "trilhos", "onibus", "geo"]:
        candidate = out / "looker" / subdir_name
        if candidate.exists() and candidate.is_dir():
            for f in candidate.glob("*.csv"):
                shutil.copy2(f, looker_dir / f.name)
            for f in candidate.glob("*.geojson"):
                shutil.copy2(f, looker_dir / f.name)

    print(f"Looker‑friendly data has been written to {looker_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Looker‑friendly datasets from run_pipeline outputs.")
    parser.add_argument("--base-dir", default=".", help="Repository base directory (defaults to current directory)")
    parser.add_argument("--out-dir", default="data_out", help="Folder containing run_pipeline outputs (defaults to 'data_out')")
    args = parser.parse_args()
    generate_looker(args.base_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())