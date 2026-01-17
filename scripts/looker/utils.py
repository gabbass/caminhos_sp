"""Utilities for building Looker Studio outputs."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - handled by calling scripts
    gpd = None


LOGGER_NAME = "looker_builder"


@dataclass
class BuildReport:
    generated_files: list[tuple[str, int]]
    warnings: list[str]

    def add_file(self, path: Path, rows: int) -> None:
        self.generated_files.append((str(path), rows))

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def find_candidate_files(
    base_dir: Path,
    keywords: Iterable[str],
    extensions: Iterable[str],
) -> list[Path]:
    base_dir = base_dir.resolve()
    keywords_lower = [kw.lower() for kw in keywords]
    exts_lower = [ext.lower().lstrip(".") for ext in extensions]
    matches: list[Path] = []
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in exts_lower:
            continue
        name = path.name.lower()
        if all(kw in name for kw in keywords_lower):
            matches.append(path)
    return matches


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")
        for col in frame.columns
    ]
    return frame


def parse_year_month(frame: pd.DataFrame) -> pd.Series:
    if "ano_mes" in frame.columns:
        series = frame["ano_mes"].astype(str)
        return series.str.slice(0, 7)
    if "mes" in frame.columns and "ano" in frame.columns:
        return frame["ano"].astype(int).astype(str).str.zfill(4) + "-" + frame[
            "mes"
        ].astype(int).astype(str).str.zfill(2)
    if "data" in frame.columns:
        parsed = pd.to_datetime(frame["data"], errors="coerce")
        return parsed.dt.strftime("%Y-%m")
    if "ano" in frame.columns:
        return frame["ano"].astype(int).astype(str)
    return pd.Series([None] * len(frame))


def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[^\d,.-]", "", regex=True)
        cleaned = cleaned.str.replace(".", "", regex=False)
        cleaned = cleaned.str.replace(",", ".", regex=False)
    else:
        cleaned = series
    return pd.to_numeric(cleaned, errors="coerce")


def select_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        if cand in frame.columns:
            return cand
    return None


def safe_json_dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def read_geodata(path: Path) -> "gpd.GeoDataFrame":
    if gpd is None:
        raise ImportError("geopandas is required to read geodata")
    suffix = path.suffix.lower()
    if suffix in {".zip", ".kmz"}:
        return gpd.read_file(f"zip://{path}")
    return gpd.read_file(path)


def ensure_wgs84(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    return gdf


def representative_lat_lon(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    gdf = gdf.copy()
    rep_points = gdf.geometry.representative_point()
    gdf["lat"] = rep_points.y
    gdf["lon"] = rep_points.x
    gdf["latlon"] = (
        gdf["lat"].map(lambda val: f"{val:.6f}")
        + ","
        + gdf["lon"].map(lambda val: f"{val:.6f}")
    )
    return gdf


def validate_year_month(series: pd.Series) -> pd.Series:
    pattern = re.compile(r"^\d{4}(-\d{2})?$")
    return series.where(series.astype(str).str.match(pattern), None)


def drop_duplicate_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.drop_duplicates(subset=keys, keep="first")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return frame


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_identifier(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def filter_non_null(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for col in columns:
        mask &= frame[col].notna()
    return frame.loc[mask]

