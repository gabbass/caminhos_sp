"""Build dimension table for OD districts and simplified GeoJSON."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

from .utils import (
    BuildReport,
    clean_identifier,
    ensure_directory,
    ensure_wgs84,
    filter_non_null,
    find_candidate_files,
    normalize_columns,
    read_geodata,
    representative_lat_lon,
    setup_logging,
)


def score_candidate(path: Path, keywords: Iterable[str]) -> int:
    name = path.name.lower()
    return sum(1 for kw in keywords if kw in name)


def select_best_candidate(base_dir: Path) -> Path | None:
    keywords = ["origemdestino", "od", "distrito", "zona"]
    extensions = ["shp", "geojson", "json", "gpkg", "kml", "kmz", "zip"]
    candidates = find_candidate_files(base_dir, keywords=["od"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=keywords, extensions=extensions)
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda path: score_candidate(path, keywords), reverse=True)
    return ranked[0]


def extract_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = normalize_columns(gdf)
    id_candidates = [
        "id_distrito_od",
        "id_distrito",
        "cd_distrito",
        "codigo_distrito",
        "codigo",
        "id",
        "zona",
        "zona_od",
        "zona_distrito",
    ]
    name_candidates = [
        "nm_distrito_od",
        "nm_distrito",
        "nome_distrito",
        "distrito",
        "nome",
    ]
    municipio_candidates = ["nm_municipio", "municipio", "nome_municipio"]
    uf_candidates = ["uf", "sigla_uf", "estado"]

    id_col = next((col for col in id_candidates if col in gdf.columns), None)
    name_col = next((col for col in name_candidates if col in gdf.columns), None)
    municipio_col = next((col for col in municipio_candidates if col in gdf.columns), None)
    uf_col = next((col for col in uf_candidates if col in gdf.columns), None)

    if id_col is None or name_col is None:
        raise ValueError("Não foi possível identificar colunas de id/nome do distrito OD.")

    gdf = gdf.rename(
        columns={
            id_col: "id_distrito_od",
            name_col: "nm_distrito_od",
        }
    )
    if municipio_col:
        gdf = gdf.rename(columns={municipio_col: "nm_municipio"})
    else:
        gdf["nm_municipio"] = ""
    if uf_col:
        gdf = gdf.rename(columns={uf_col: "uf"})
    else:
        gdf["uf"] = ""
    return gdf


def build_dim_distritos(
    base_dir: Path,
    out_dir: Path,
    simplify_tolerance: float,
    report: BuildReport,
    sep: str = ",",
) -> None:
    logger = setup_logging()
    candidate = select_best_candidate(base_dir)
    if candidate is None:
        message = "Insumo de distritos OD não encontrado."
        logger.warning(message)
        report.warn(message)
        return

    logger.info("Lendo geografia OD de %s", candidate)
    gdf = read_geodata(candidate)
    gdf = ensure_wgs84(gdf)
    gdf = extract_columns(gdf)
    gdf["id_distrito_od"] = clean_identifier(gdf["id_distrito_od"])
    gdf["nm_distrito_od"] = gdf["nm_distrito_od"].astype(str).str.strip()
    gdf["nm_municipio"] = gdf["nm_municipio"].fillna("").astype(str).str.strip()
    gdf["uf"] = gdf["uf"].fillna("").astype(str).str.strip()
    if not gdf["nm_municipio"].any():
        gdf["nm_municipio"] = "São Paulo"
    if not gdf["uf"].any():
        gdf["uf"] = "SP"

    gdf = filter_non_null(gdf, ["id_distrito_od"])
    gdf = gdf.drop_duplicates(subset=["id_distrito_od"])

    gdf = representative_lat_lon(gdf)
    if gdf[["lat", "lon"]].isna().any().any():
        raise ValueError("Lat/Lon inválidos ao calcular representative_point.")

    dim_cols = [
        "id_distrito_od",
        "nm_distrito_od",
        "nm_municipio",
        "uf",
        "lat",
        "lon",
        "latlon",
    ]
    dim_df = pd.DataFrame(gdf[dim_cols])
    ensure_directory(out_dir)
    dim_path = out_dir / "dim_distritos_od.csv"
    dim_df.to_csv(dim_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(dim_path, len(dim_df))
    logger.info("Gerado %s (%s linhas)", dim_path, len(dim_df))

    geo_out_dir = out_dir.parent / "geo"
    ensure_directory(geo_out_dir)
    geo_gdf = gdf[
        ["id_distrito_od", "nm_distrito_od", "nm_municipio", "uf", "geometry"]
    ].copy()
    geo_gdf = ensure_wgs84(geo_gdf)
    geo_gdf["geometry"] = geo_gdf.geometry.simplify(
        simplify_tolerance, preserve_topology=True
    )
    geojson_path = geo_out_dir / "distritos_od_simplificado.geojson"
    geo_gdf.to_file(geojson_path, driver="GeoJSON")
    report.add_file(geojson_path, len(geo_gdf))
    logger.info("Gerado %s (%s features)", geojson_path, len(geo_gdf))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OD district dimension.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", default="data_out/looker/geo")
    parser.add_argument("--geo-simplify-tolerance", type=float, default=0.0005)
    parser.add_argument("--sep", default=",")
    args = parser.parse_args()

    report = BuildReport(generated_files=[], warnings=[])
    build_dim_distritos(
        Path(args.base_dir),
        Path(args.out_dir),
        args.geo_simplify_tolerance,
        report,
        args.sep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
