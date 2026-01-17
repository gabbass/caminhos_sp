"""Build rail demand outputs and station-district bridge."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .utils import (
    BuildReport,
    clean_identifier,
    coerce_numeric,
    ensure_directory,
    ensure_wgs84,
    filter_non_null,
    find_candidate_files,
    normalize_columns,
    parse_year_month,
    read_geodata,
    setup_logging,
    validate_year_month,
)


def score_candidate(path: Path, keywords: Iterable[str]) -> int:
    name = path.name.lower()
    return sum(1 for kw in keywords if kw in name)


def select_demanda_candidate(base_dir: Path) -> Path | None:
    keywords = ["demanda", "estacao", "embarque", "mes", "trilho", "metro", "cptm"]
    extensions = ["csv", "xlsx", "xls", "parquet"]
    candidates = find_candidate_files(base_dir, keywords=["demanda"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=["estacao"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=keywords, extensions=extensions)
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda path: score_candidate(path, keywords), reverse=True)
    return ranked[0]


def select_geo_candidate(base_dir: Path) -> Path | None:
    geojson = base_dir / "data_out" / "looker" / "geo" / "distritos_od_simplificado.geojson"
    if geojson.exists():
        return geojson
    keywords = ["od", "distrito", "zona"]
    extensions = ["shp", "geojson", "json", "gpkg", "kml", "kmz", "zip"]
    candidates = find_candidate_files(base_dir, keywords=["od"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=keywords, extensions=extensions)
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda path: score_candidate(path, keywords), reverse=True)
    return ranked[0]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def extract_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_columns(frame)
    id_candidates = ["id_estacao", "codigo_estacao", "cod_estacao", "estacao_id", "id"]
    name_candidates = ["nm_estacao", "nome_estacao", "estacao", "nome"]
    linha_candidates = ["nm_linha", "linha", "linha_nome"]
    operador_candidates = ["operador", "operadora"]
    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lon", "longitude", "long"]
    embarques_candidates = ["embarques", "passageiros", "demanda", "qtd", "volume"]

    id_col = next((col for col in id_candidates if col in frame.columns), None)
    name_col = next((col for col in name_candidates if col in frame.columns), None)
    embarques_col = next(
        (col for col in embarques_candidates if col in frame.columns), None
    )

    if id_col is None or name_col is None or embarques_col is None:
        raise ValueError("Colunas de estação/embarques não identificadas.")

    frame = frame.rename(
        columns={
            id_col: "id_estacao",
            name_col: "nm_estacao",
            embarques_col: "embarques",
        }
    )
    if linha_candidates:
        linha_col = next((col for col in linha_candidates if col in frame.columns), None)
        if linha_col:
            frame = frame.rename(columns={linha_col: "nm_linha"})
    if operador_candidates:
        operador_col = next(
            (col for col in operador_candidates if col in frame.columns), None
        )
        if operador_col:
            frame = frame.rename(columns={operador_col: "operador"})
    lat_col = next((col for col in lat_candidates if col in frame.columns), None)
    lon_col = next((col for col in lon_candidates if col in frame.columns), None)
    if lat_col:
        frame = frame.rename(columns={lat_col: "lat"})
    if lon_col:
        frame = frame.rename(columns={lon_col: "lon"})
    return frame


def build_trilhos(
    base_dir: Path,
    out_dir: Path,
    nearest_radius_m: float,
    report: BuildReport,
    sep: str = ",",
) -> None:
    logger = setup_logging()
    candidate = select_demanda_candidate(base_dir)
    if candidate is None:
        message = "Insumo de demanda de trilhos não encontrado."
        logger.warning(message)
        report.warn(message)
        return

    logger.info("Lendo demanda de trilhos de %s", candidate)
    frame = read_table(candidate)
    frame = extract_columns(frame)
    frame["id_estacao"] = clean_identifier(frame["id_estacao"])
    frame["nm_estacao"] = frame["nm_estacao"].astype(str).str.strip()
    frame["embarques"] = coerce_numeric(frame["embarques"]).fillna(0).astype(float)
    frame["ano_mes"] = parse_year_month(frame)
    frame["ano_mes"] = validate_year_month(frame["ano_mes"])
    frame = filter_non_null(frame, ["id_estacao"])
    frame = frame[frame["embarques"] >= 0]

    dim_cols = ["id_estacao", "nm_estacao"]
    if "nm_linha" in frame.columns:
        dim_cols.append("nm_linha")
    if "operador" in frame.columns:
        dim_cols.append("operador")
    if "lat" in frame.columns and "lon" in frame.columns:
        dim_cols.extend(["lat", "lon"])

    dim_df = frame[dim_cols].drop_duplicates(subset=["id_estacao"])
    ensure_directory(out_dir)

    dim_path = out_dir / "dim_estacoes.csv"
    dim_df.to_csv(dim_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(dim_path, len(dim_df))
    logger.info("Gerado %s (%s linhas)", dim_path, len(dim_df))

    group_keys = ["id_estacao", "ano_mes"]
    if "operador" in frame.columns:
        group_keys.append("operador")
    if "nm_linha" in frame.columns:
        group_keys.append("nm_linha")
    fact_df = frame[group_keys + ["embarques"]]
    fact_df = fact_df.groupby(group_keys)["embarques"].sum().reset_index()
    fact_df["embarques"] = fact_df["embarques"].round().astype(int)

    fact_path = out_dir / "fato_demanda_estacao_mes.csv"
    fact_df.to_csv(fact_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(fact_path, len(fact_df))
    logger.info("Gerado %s (%s linhas)", fact_path, len(fact_df))

    if "lat" not in frame.columns or "lon" not in frame.columns:
        message = "Lat/Lon de estação ausentes; bridge estacao-distrito não gerada."
        logger.warning(message)
        report.warn(message)
        return

    geo_candidate = select_geo_candidate(base_dir)
    if geo_candidate is None:
        message = "Geografia de distritos OD não encontrada para bridge."
        logger.warning(message)
        report.warn(message)
        return

    logger.info("Lendo geografia OD para bridge de %s", geo_candidate)
    distritos = read_geodata(geo_candidate)
    distritos = ensure_wgs84(distritos)
    distritos = normalize_columns(distritos)
    id_col = next(
        (col for col in ["id_distrito_od", "id_distrito", "cd_distrito", "id"] if col in distritos.columns),
        None,
    )
    if id_col is None:
        message = "Coluna de id de distrito não identificada para bridge."
        logger.warning(message)
        report.warn(message)
        return

    distritos = distritos.rename(columns={id_col: "id_distrito_od"})
    distritos["id_distrito_od"] = clean_identifier(distritos["id_distrito_od"])
    distritos = distritos[["id_distrito_od", "geometry"]]

    stations = frame[["id_estacao", "lat", "lon"]].drop_duplicates(subset=["id_estacao"])
    stations = stations.dropna(subset=["lat", "lon"])
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=[Point(xy) for xy in zip(stations["lon"], stations["lat"])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(stations_gdf, distritos, how="left", predicate="within")
    missing = joined[joined["id_distrito_od"].isna()]

    if not missing.empty:
        logger.warning("Estacoes fora do poligono; aplicando nearest com raio %sm", nearest_radius_m)
        stations_proj = missing.to_crs(3857)
        distritos_proj = distritos.to_crs(3857)
        nearest = gpd.sjoin_nearest(
            stations_proj,
            distritos_proj,
            how="left",
            max_distance=nearest_radius_m,
        )
        nearest = nearest.to_crs(4326)
        joined.loc[missing.index, "id_distrito_od"] = nearest["id_distrito_od"].values

    unresolved = joined[joined["id_distrito_od"].isna()]
    if not unresolved.empty:
        message = (
            "Estacoes sem distrito apos tentativa nearest: "
            + ", ".join(unresolved["id_estacao"].astype(str).tolist())
        )
        logger.warning(message)
        report.warn(message)

    bridge = joined[["id_estacao", "id_distrito_od"]].dropna()
    bridge = bridge.drop_duplicates(subset=["id_estacao"])
    bridge_path = out_dir / "bridge_estacao_distrito_od.csv"
    bridge.to_csv(bridge_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(bridge_path, len(bridge))
    logger.info("Gerado %s (%s linhas)", bridge_path, len(bridge))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rail demand outputs.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", default="data_out/looker/trilhos")
    parser.add_argument("--nearest-radius-m", type=float, default=200)
    parser.add_argument("--sep", default=",")
    args = parser.parse_args()

    report = BuildReport(generated_files=[], warnings=[])
    build_trilhos(
        Path(args.base_dir),
        Path(args.out_dir),
        args.nearest_radius_m,
        report,
        args.sep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
