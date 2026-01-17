"""Build OD fact tables (origins, destinations, matrix)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import (
    BuildReport,
    clean_identifier,
    coerce_numeric,
    ensure_directory,
    filter_non_null,
    find_candidate_files,
    normalize_columns,
    parse_year_month,
    setup_logging,
    validate_year_month,
)


def score_candidate(path: Path, keywords: Iterable[str]) -> int:
    name = path.name.lower()
    return sum(1 for kw in keywords if kw in name)


def select_best_candidate(base_dir: Path) -> Path | None:
    keywords = ["origemdestino", "od", "matriz", "fluxo", "distrito", "zona"]
    extensions = ["csv", "xlsx", "xls", "parquet"]
    candidates = find_candidate_files(base_dir, keywords=["od"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=["origem"], extensions=extensions)
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
    origem_candidates = ["id_origem", "origem", "id_zona_origem", "zona_origem", "zona_o"]
    destino_candidates = [
        "id_destino",
        "destino",
        "id_zona_destino",
        "zona_destino",
        "zona_d",
    ]
    viagens_candidates = ["viagens", "volume", "fluxo", "qtd_viagens", "demanda"]
    fator_candidates = ["fator_expansao", "fator", "expansao", "expansion"]

    origem_col = next((col for col in origem_candidates if col in frame.columns), None)
    destino_col = next((col for col in destino_candidates if col in frame.columns), None)
    viagens_col = next((col for col in viagens_candidates if col in frame.columns), None)
    fator_col = next((col for col in fator_candidates if col in frame.columns), None)

    if origem_col is None or destino_col is None or viagens_col is None:
        raise ValueError("Colunas de origem/destino/viagens não identificadas para OD.")

    frame = frame.rename(
        columns={
            origem_col: "id_origem_distrito_od",
            destino_col: "id_destino_distrito_od",
            viagens_col: "viagens",
        }
    )
    if fator_col:
        frame = frame.rename(columns={fator_col: "fator_expansao"})
    return frame


def build_od_fatos(
    base_dir: Path, out_dir: Path, report: BuildReport, sep: str = ","
) -> None:
    logger = setup_logging()
    candidate = select_best_candidate(base_dir)
    if candidate is None:
        message = "Insumo de matriz OD não encontrado."
        logger.warning(message)
        report.warn(message)
        return

    logger.info("Lendo matriz OD de %s", candidate)
    frame = read_table(candidate)
    frame = extract_columns(frame)
    frame["id_origem_distrito_od"] = clean_identifier(frame["id_origem_distrito_od"])
    frame["id_destino_distrito_od"] = clean_identifier(frame["id_destino_distrito_od"])
    frame["viagens"] = coerce_numeric(frame["viagens"]).fillna(0).astype(float)

    if "fator_expansao" in frame.columns:
        fator = coerce_numeric(frame["fator_expansao"]).fillna(1)
        frame["viagens"] = frame["viagens"] * fator

    frame["ano_mes"] = parse_year_month(frame)
    frame["ano_mes"] = validate_year_month(frame["ano_mes"])

    frame = filter_non_null(frame, ["id_origem_distrito_od", "id_destino_distrito_od"])
    frame = frame[frame["viagens"] >= 0]

    if frame["ano_mes"].isna().all():
        frame["ano_mes"] = frame.get("ano", pd.Series([""] * len(frame))).astype(str)

    ensure_directory(out_dir)
    fluxo = (
        frame.groupby(["id_origem_distrito_od", "id_destino_distrito_od", "ano_mes"], dropna=False)[
            "viagens"
        ]
        .sum()
        .reset_index()
    )
    fluxo["viagens"] = fluxo["viagens"].round().astype(int)

    fluxo_path = out_dir / "fato_od_fluxo_distrito.csv"
    fluxo.to_csv(fluxo_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(fluxo_path, len(fluxo))
    logger.info("Gerado %s (%s linhas)", fluxo_path, len(fluxo))

    origens = (
        fluxo.groupby(["id_origem_distrito_od", "ano_mes"])["viagens"]
        .sum()
        .reset_index()
        .rename(
            columns={"id_origem_distrito_od": "id_distrito_od", "viagens": "viagens_origem"}
        )
    )
    origens_path = out_dir / "fato_od_origens_distrito.csv"
    origens.to_csv(origens_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(origens_path, len(origens))
    logger.info("Gerado %s (%s linhas)", origens_path, len(origens))

    destinos = (
        fluxo.groupby(["id_destino_distrito_od", "ano_mes"])["viagens"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "id_destino_distrito_od": "id_distrito_od",
                "viagens": "viagens_destino",
            }
        )
    )
    destinos_path = out_dir / "fato_od_destinos_distrito.csv"
    destinos.to_csv(destinos_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(destinos_path, len(destinos))
    logger.info("Gerado %s (%s linhas)", destinos_path, len(destinos))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OD fact tables.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", default="data_out/looker/od")
    parser.add_argument("--sep", default=",")
    args = parser.parse_args()

    report = BuildReport(generated_files=[], warnings=[])
    build_od_fatos(Path(args.base_dir), Path(args.out_dir), report, args.sep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
