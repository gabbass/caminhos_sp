"""Pipeline para padronizar dados de transporte para BI e GIS."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data_raw"
DEFAULT_OUT = ROOT / "data_out"

NS = {"kml": "http://www.opengis.net/kml/2.2"}


@dataclass
class PolygonRecord:
    nome: str
    coordenadas: list[list[tuple[float, float]]]


STATUS_MAP = {
    "existente": "existente",
    "atual": "existente",
    "operacao": "existente",
    "operação": "existente",
    "em operação": "existente",
    "em operacao": "existente",
    "em_operacao": "existente",
    "construcao": "em_construcao",
    "construção": "em_construcao",
    "em construcao": "em_construcao",
    "em construção": "em_construcao",
    "em_construcao": "em_construcao",
    "projeto": "em_projeto",
    "em projeto": "em_projeto",
    "em_projeto": "em_projeto",
    "proposta": "proposto",
    "proposto": "proposto",
}

STATUS_LABELS = {
    "existente": "existente",
    "em_projeto": "em projeto",
    "em_construcao": "em construção",
    "proposto": "proposta",
}

STATUS_CANONICAL_ALIASES = {
    "existente": "existente",
    "em_projeto": "em_projeto",
    "em projeto": "em_projeto",
    "em_construcao": "em_construcao",
    "em construcao": "em_construcao",
    "em construção": "em_construcao",
    "proposta": "proposto",
    "proposto": "proposto",
}

def normalize_text(value: str) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def to_snake(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake(col) for col in df.columns]
    return df


def sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitiza colunas textuais substituindo \r e \n por espaço.

    Política: mantém o conteúdo textual original, apenas removendo quebras de linha
    internas para evitar linhas quebradas no CSV. Não altera colunas não textuais.
    """
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.replace(r"[\r\n]+", " ", regex=True)
        )
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df = sanitize_text_columns(df)
    df.to_csv(
        path,
        index=False,
        line_terminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
    )


def infer_status(*values: str) -> str:
    combined = " ".join(normalize_text(value) for value in values if value)
    for key, status in STATUS_MAP.items():
        if key in combined:
            return status
    return "desconhecido"


def to_final_status(value: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("Status desconhecido (valor ausente).")
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError("Status desconhecido (valor vazio).")
    canonical = STATUS_CANONICAL_ALIASES.get(normalized, normalized)
    if canonical not in STATUS_LABELS:
        raise ValueError(f"Status desconhecido para revisão: {value!r}")
    return STATUS_LABELS[canonical]


def finalize_statuses(df: pd.DataFrame, context: str) -> pd.DataFrame:
    df = df.copy()
    try:
        df["status"] = df["status"].apply(to_final_status)
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc
    return df

def read_excel_sheets(path: Path, sheets: Iterable[str]) -> pd.DataFrame:
    frames = []
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet)
        df = normalize_columns(df)
        df["origem_aba"] = to_snake(sheet)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_status_map(raw_dir: Path) -> pd.DataFrame:
    status_path = raw_dir / "OD2023_Distritos_Infra_Transporte.xlsx"
    df = pd.read_excel(status_path, sheet_name="Status_Linhas_Plano")
    df = normalize_columns(df)
    df["status_assumido"] = df["status_assumido"].astype(str)
    df["status_norm"] = df["status_assumido"].apply(infer_status)
    df["linha_norm"] = df["linha"].astype(str).map(to_snake)
    df["sistema_norm"] = df["sistema"].astype(str).map(to_snake)
    df["subsistema_norm"] = df["subsistema"].astype(str).map(to_snake)
    return df


def normalize_status_join_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["linha_norm"] = df.get("linha", "").astype(str).map(to_snake)
    df["sistema_norm"] = df.get("sistema", "").astype(str).map(to_snake)
    df["subsistema_norm"] = df.get("subsistema", "").astype(str).map(to_snake)
    return df


def apply_status_map_with_fallback(
    df: pd.DataFrame,
    status_map: pd.DataFrame,
    fallback_fields: list[str],
) -> pd.DataFrame:
    df = normalize_status_join_fields(df)
    merged = df.merge(
        status_map[["linha_norm", "sistema_norm", "subsistema_norm", "status_norm"]],
        how="left",
        on=["linha_norm", "sistema_norm", "subsistema_norm"],
    )
    merged["status"] = merged["status_norm"].fillna(
        merged.apply(lambda row: infer_status(*(row.get(field) for field in fallback_fields)), axis=1)
    )
    merged = merged.drop(columns=["status_norm"], errors="ignore")
    return merged


def apply_status_map(df: pd.DataFrame, status_map: pd.DataFrame) -> pd.DataFrame:
    return apply_status_map_with_fallback(df, status_map, ["fase", "etapa", "linha"])


def build_lines(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    path = raw_dir / "01_Propostas_Plano_Corpus_Ducen_v5_4_ATUALIZADO_GERAL.xlsx"
    df = read_excel_sheets(path, ["Dados", "Metropolitano", "Regional"])
    df = df.rename(columns={
        "quantidade_de_vias": "qtde_vias",
        "numero_de_estacoes": "qtde_estacoes",
        "comprimento_km": "comprimento_km",
    })
    status_map = load_status_map(raw_dir)
    df = apply_status_map(df, status_map)
    df["linha_id"] = df.get("linha_limpo", df.get("linha", "")).astype(str).map(to_snake)
    df["operador"] = df.get("subsistema", "").astype(str)
    df["cidade"] = "São Paulo"
    columns = [
        "linha_id",
        "linha",
        "sistema",
        "subsistema",
        "operador",
        "fase",
        "etapa",
        "status",
        "qtde_vias",
        "qtde_estacoes",
        "comprimento_km",
        "origem_aba",
        "cidade",
    ]
    df = df[[col for col in columns if col in df.columns]]
    return df


def build_stations(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    path = raw_dir / "02_Estacoes_Plano_Corpus_Ducen_v5_4_ATUALIZADO_GERAL.xlsx"
    df = read_excel_sheets(path, ["Dados", "Metropolitano", "Regional"])
    status_map = load_status_map(raw_dir)
    df = apply_status_map_with_fallback(
        df,
        status_map,
        ["fase", "etapa", "nome_da_estacao"],
    )
    df["estacao_id"] = df.get("nome_da_estacao_limpo", df.get("nome_da_estacao", "")).astype(str).map(to_snake)
    df["operador"] = df.get("subsistema", "").astype(str)
    df["cidade"] = "São Paulo"
    columns = [
        "estacao_id",
        "nome_da_estacao",
        "sistema",
        "subsistema",
        "operador",
        "linha",
        "fase",
        "etapa",
        "status",
        "servico_expresso",
        "latitude",
        "longitude",
        "origem_aba",
        "cidade",
    ]
    df = df[[col for col in columns if col in df.columns]]
    return df


def build_line_points(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    path = raw_dir / "03_Coordenadas_Plano_Corpus_Ducen_v5_4_ATUALIZADO_GERAL.xlsx"
    df = read_excel_sheets(path, ["Dados", "Metropolitano", "Regional"])
    status_map = load_status_map(raw_dir)
    df = apply_status_map_with_fallback(df, status_map, ["fase", "etapa", "linha"])
    df["linha_id"] = df.get("linha", "").astype(str).map(to_snake)
    df["operador"] = df.get("subsistema", "").astype(str)
    df["cidade"] = "São Paulo"
    columns = [
        "linha_id",
        "linha",
        "sistema",
        "subsistema",
        "operador",
        "fase",
        "etapa",
        "status",
        "placemark",
        "segmento",
        "ordem_ponto",
        "latitude",
        "longitude",
        "origem_aba",
        "cidade",
    ]
    df = df[[col for col in columns if col in df.columns]]
    return df


def read_gtfs_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def build_bus_lines(raw_dir: Path) -> pd.DataFrame:
    routes = read_gtfs_csv(raw_dir / "routes.txt")
    routes = normalize_columns(routes)
    routes["route_id"] = routes["route_id"].astype(str)

    dim_path = raw_dir / "Relacionamentos_Linhas_Demanda_AreaOperacional_CORES_1a9_TP_TS.xlsx"
    area = pd.read_excel(dim_path, sheet_name="Dim_Area_Operacional")
    area = normalize_columns(area)
    area["route_id"] = area["route_id"].astype(str)

    df = routes.merge(area, how="left", on="route_id")
    df["sistema"] = "onibus"
    df["subsistema"] = "municipal"
    df["status"] = "existente"
    df["cidade"] = "São Paulo"
    columns = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "agency_id",
        "sistema",
        "subsistema",
        "status",
        "grupo_mode",
        "lote_mode",
        "empresa_mode",
        "consorcio_operador",
        "area_operacional_num_1a9",
        "nome",
        "area_operacional_hex_cor",
        "cidade",
    ]
    df = df[[col for col in columns if col in df.columns]]
    return df


def is_terminal(name: str) -> bool:
    if not name:
        return False
    name_norm = normalize_text(name)
    return bool(re.search(r"\bterm", name_norm))


def build_terminals(raw_dir: Path) -> pd.DataFrame:
    stops = read_gtfs_csv(raw_dir / "stops.txt")
    stops = normalize_columns(stops)
    terminals = stops[stops["stop_name"].map(is_terminal)].copy()
    terminals["terminal_id"] = terminals["stop_id"].astype(str)
    terminals["sistema"] = "onibus"
    terminals["subsistema"] = "municipal"
    terminals["status"] = "existente"
    terminals["cidade"] = "São Paulo"
    columns = [
        "terminal_id",
        "stop_name",
        "stop_desc",
        "stop_lat",
        "stop_lon",
        "sistema",
        "subsistema",
        "status",
        "cidade",
    ]
    terminals = terminals[[col for col in columns if col in terminals.columns]]
    terminals = terminals.rename(columns={"stop_lat": "latitude", "stop_lon": "longitude"})
    return terminals


def parse_kml_coordinates(coord_text: str) -> list[tuple[float, float]]:
    coords = []
    for chunk in coord_text.strip().split():
        lon, lat, *_ = chunk.split(",")
        coords.append((float(lon), float(lat)))
    return coords


def parse_kmz_districts(kmz_path: Path) -> list[PolygonRecord]:
    polygons: list[PolygonRecord] = []
    with zipfile.ZipFile(kmz_path) as zf:
        kml_name = next((name for name in zf.namelist() if name.endswith(".kml")), None)
        if not kml_name:
            return polygons
        root = ET.fromstring(zf.read(kml_name))
    for placemark in root.findall(".//kml:Placemark", NS):
        name_el = placemark.find("kml:name", NS)
        if name_el is None:
            continue
        name = name_el.text or ""
        polygons_coords: list[list[tuple[float, float]]] = []
        for coords_el in placemark.findall(".//kml:Polygon//kml:coordinates", NS):
            coords_text = coords_el.text or ""
            polygons_coords.append(parse_kml_coordinates(coords_text))
        if polygons_coords:
            polygons.append(PolygonRecord(nome=name, coordenadas=polygons_coords))
    return polygons


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        ):
            inside = not inside
    return inside


def assign_districts(df: pd.DataFrame, polygons: list[PolygonRecord]) -> pd.DataFrame:
    if not polygons:
        df["distrito"] = None
        return df
    records = []
    for _, row in df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            records.append(None)
            continue
        point = (float(lon), float(lat))
        found = None
        for poly in polygons:
            if any(point_in_polygon(point, coords) for coords in poly.coordenadas):
                found = poly.nome
                break
        records.append(found)
    df = df.copy()
    df["distrito"] = records
    return df


def build_districts(raw_dir: Path) -> tuple[pd.DataFrame, list[PolygonRecord]]:
    od_path = raw_dir / "OD2023_Distritos_Infra_Transporte.xlsx"
    df = pd.read_excel(od_path, sheet_name="Distritos_OD2023")
    df = normalize_columns(df)
    df["cidade"] = "São Paulo"
    kmz_path = raw_dir / "Dados GEOSAMPA" / "LL_WGS84_KMZ_distrito.kmz"
    polygons = parse_kmz_districts(kmz_path)
    df["distrito_norm"] = df["nomedistrito"].astype(str).map(to_snake)
    return df, polygons


def polygons_to_geojson(polygons: list[PolygonRecord]) -> dict:
    features = []
    for poly in polygons:
        geometry = {
            "type": "MultiPolygon" if len(poly.coordenadas) > 1 else "Polygon",
            "coordinates": [],
        }
        if len(poly.coordenadas) > 1:
            geometry["coordinates"] = [
                [[(lon, lat) for lon, lat in coords]] for coords in poly.coordenadas
            ]
        else:
            geometry["coordinates"] = [[(lon, lat) for lon, lat in poly.coordenadas[0]]]
        features.append(
            {
                "type": "Feature",
                "properties": {"distrito": poly.nome, "cidade": "São Paulo"},
                "geometry": geometry,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def polygons_to_wkt(polygons: list[PolygonRecord]) -> dict[str, str]:
    wkt_map = {}
    for poly in polygons:
        if len(poly.coordenadas) > 1:
            parts = []
            for coords in poly.coordenadas:
                ring = ", ".join(f"{lon} {lat}" for lon, lat in coords)
                parts.append(f"(({ring}))")
            wkt = f"MULTIPOLYGON({', '.join(parts)})"
        else:
            ring = ", ".join(f"{lon} {lat}" for lon, lat in poly.coordenadas[0])
            wkt = f"POLYGON(({ring}))"
        wkt_map[poly.nome] = wkt
    return wkt_map


def export_by_status(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    df = finalize_statuses(df, f"export_by_status({prefix})")
    for status, group in df.groupby("status"):
        safe_status = to_snake(status)
        if not safe_status:
            safe_status = "desconhecido"
        output_path = out_dir / f"{prefix}_{safe_status}.csv"
        write_csv(group, output_path)


def save_geojson(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "lines": out_dir / "lines",
        "stations": out_dir / "stations",
        "terminals": out_dir / "terminals",
        "line_points": out_dir / "line_points",
        "bus": out_dir / "bus",
        "districts": out_dir / "districts",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de dados Caminhos SP")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dirs = ensure_dirs(args.out_dir)

    linhas = build_lines(args.raw_dir, args.out_dir)
    estacoes = build_stations(args.raw_dir, args.out_dir)
    line_points = build_line_points(args.raw_dir, args.out_dir)
    bus_lines = build_bus_lines(args.raw_dir)
    terminais = build_terminals(args.raw_dir)
    distritos, polygons = build_districts(args.raw_dir)

    if polygons:
        estacoes = assign_districts(estacoes, polygons)
        terminais = assign_districts(terminais, polygons)
        line_points = assign_districts(line_points, polygons)

    export_by_status(linhas, out_dirs["lines"], "linhas")
    export_by_status(estacoes, out_dirs["stations"], "estacoes")
    export_by_status(line_points, out_dirs["line_points"], "linha_pontos")
    export_by_status(terminais, out_dirs["terminals"], "terminais")

    write_csv(bus_lines, out_dirs["bus"] / "linhas_onibus.csv")

    wkt_map = polygons_to_wkt(polygons)
    distritos["geometria_wkt"] = distritos["nomedistrito"].map(wkt_map)
    write_csv(distritos, out_dirs["districts"] / "distritos_od.csv")

    if polygons:
        geojson = polygons_to_geojson(polygons)
        save_geojson(geojson, out_dirs["districts"] / "distritos_od.geojson")


if __name__ == "__main__":
    main()
