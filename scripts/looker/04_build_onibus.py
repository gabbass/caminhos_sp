"""Build bus demand outputs for Looker Studio."""
from __future__ import annotations

import argparse
import re
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


def select_candidates(base_dir: Path) -> list[Path]:
    keywords = ["sptrans", "onibus", "linha", "passageiros", "demanda"]
    extensions = ["csv", "xlsx", "xls", "parquet"]
    candidates = find_candidate_files(base_dir, keywords=["onibus"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=["sptrans"], extensions=extensions)
    if not candidates:
        candidates = find_candidate_files(base_dir, keywords=keywords, extensions=extensions)
    if not candidates:
        return []
    return sorted(candidates, key=lambda path: score_candidate(path, keywords), reverse=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_table_head(path: Path, nrows: int = 5) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=nrows)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path).head(nrows)
    return pd.read_csv(path, nrows=nrows)


def normalize_column_name(name: str | None) -> str | None:
    if name is None:
        return None
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def has_required_columns(
    frame: pd.DataFrame,
    col_id_linha_onibus: str | None = None,
    col_passageiros_total: str | None = None,
) -> bool:
    frame = normalize_columns(frame)
    id_candidates = ["id_linha_onibus", "id_linha", "linha_id", "linha"]
    passageiros_candidates = ["passageiros", "demanda", "qtd_passageiros", "volume"]

    id_override = normalize_column_name(col_id_linha_onibus)
    passageiros_override = normalize_column_name(col_passageiros_total)

    if id_override is not None and id_override not in frame.columns:
        return False
    if passageiros_override is not None and passageiros_override not in frame.columns:
        return False

    id_col = (
        id_override
        if id_override is not None
        else next((col for col in id_candidates if col in frame.columns), None)
    )
    passageiros_col = (
        passageiros_override
        if passageiros_override is not None
        else next((col for col in passageiros_candidates if col in frame.columns), None)
    )
    return id_col is not None and passageiros_col is not None


def extract_columns(
    frame: pd.DataFrame,
    col_id_linha_onibus: str | None = None,
    col_passageiros_total: str | None = None,
) -> pd.DataFrame:
    frame = normalize_columns(frame)
    id_candidates = ["id_linha_onibus", "id_linha", "linha_id", "linha"]
    name_candidates = ["nm_linha", "nome_linha"]
    sentido_candidates = ["sentido", "direcao"]
    tipo_candidates = ["tipo", "categoria"]
    empresa_candidates = ["empresa", "operadora"]
    lote_candidates = ["lote", "area"]
    passageiros_candidates = ["passageiros", "demanda", "qtd_passageiros", "volume"]

    id_override = normalize_column_name(col_id_linha_onibus)
    passageiros_override = normalize_column_name(col_passageiros_total)

    if id_override is not None and id_override not in frame.columns:
        raise ValueError(
            "Coluna informada --col-id-linha-onibus não encontrada: "
            f"{col_id_linha_onibus}."
        )
    if passageiros_override is not None and passageiros_override not in frame.columns:
        raise ValueError(
            "Coluna informada --col-passageiros-total não encontrada: "
            f"{col_passageiros_total}."
        )

    id_col = (
        id_override
        if id_override is not None
        else next((col for col in id_candidates if col in frame.columns), None)
    )
    passageiros_col = (
        passageiros_override
        if passageiros_override is not None
        else next((col for col in passageiros_candidates if col in frame.columns), None)
    )

    if id_col is None or passageiros_col is None:
        raise ValueError("Colunas de linha/passageiros não identificadas.")

    frame = frame.rename(
        columns={id_col: "id_linha_onibus", passageiros_col: "passageiros_total"}
    )
    for col, name in [
        (name_candidates, "nm_linha"),
        (sentido_candidates, "sentido"),
        (tipo_candidates, "tipo"),
        (empresa_candidates, "empresa"),
        (lote_candidates, "lote"),
    ]:
        selected = next((cand for cand in col if cand in frame.columns), None)
        if selected:
            frame = frame.rename(columns={selected: name})
    return frame


def build_onibus(
    base_dir: Path,
    out_dir: Path,
    report: BuildReport,
    sep: str = ",",
    col_id_linha_onibus: str | None = None,
    col_passageiros_total: str | None = None,
) -> None:
    logger = setup_logging()
    candidates = select_candidates(base_dir)
    if not candidates:
        message = "Insumo de demanda de ônibus não encontrado."
        logger.warning(message)
        report.warn(message)
        return

    frame = None
    selected_candidate = None
    for candidate in candidates:
        logger.info("Validando colunas mínimas de %s", candidate)
        head = read_table_head(candidate)
        if has_required_columns(head, col_id_linha_onibus, col_passageiros_total):
            selected_candidate = candidate
            break
        message = f"Candidato ignorado por falta de colunas mínimas: {candidate}"
        logger.warning(message)
        report.warn(message)

    if selected_candidate is None:
        message = "Nenhum candidato válido encontrado para demanda de ônibus."
        logger.warning(message)
        report.warn(message)
        return

    logger.info("Lendo demanda de ônibus de %s", selected_candidate)
    frame = read_table(selected_candidate)
    try:
        frame = extract_columns(frame, col_id_linha_onibus, col_passageiros_total)
    except ValueError as exc:
        message = str(exc)
        logger.warning(message)
        report.warn(message)
        return
    frame["id_linha_onibus"] = clean_identifier(frame["id_linha_onibus"])
    frame["passageiros_total"] = (
        coerce_numeric(frame["passageiros_total"]).fillna(0).astype(float)
    )
    frame["ano_mes"] = parse_year_month(frame)
    frame["ano_mes"] = validate_year_month(frame["ano_mes"])
    frame = filter_non_null(frame, ["id_linha_onibus"])
    frame = frame[frame["passageiros_total"] >= 0]

    group_keys = ["id_linha_onibus", "ano_mes"]
    if "empresa" in frame.columns:
        group_keys.append("empresa")
    if "lote" in frame.columns:
        group_keys.append("lote")

    fact_df = frame[group_keys + ["passageiros_total"]]
    fact_df = fact_df.groupby(group_keys)["passageiros_total"].sum().reset_index()
    fact_df["passageiros_total"] = fact_df["passageiros_total"].round().astype(int)

    ensure_directory(out_dir)
    fact_path = out_dir / "fato_onibus_linha_mes.csv"
    fact_df.to_csv(fact_path, index=False, encoding="utf-8", sep=sep)
    report.add_file(fact_path, len(fact_df))
    logger.info("Gerado %s (%s linhas)", fact_path, len(fact_df))

    dim_cols = ["id_linha_onibus"]
    for col in ["nm_linha", "sentido", "tipo"]:
        if col in frame.columns:
            dim_cols.append(col)
    dim_df = frame[dim_cols].drop_duplicates(subset=["id_linha_onibus"])
    if len(dim_df) > 0 and len(dim_cols) > 1:
        dim_path = out_dir / "dim_linhas_onibus.csv"
        dim_df.to_csv(dim_path, index=False, encoding="utf-8", sep=sep)
        report.add_file(dim_path, len(dim_df))
        logger.info("Gerado %s (%s linhas)", dim_path, len(dim_df))
    else:
        message = "Dimensão de linhas de ônibus não gerada por falta de colunas."
        logger.warning(message)
        report.warn(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bus demand outputs.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", default="data_out/looker/onibus")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--col-id-linha-onibus")
    parser.add_argument("--col-passageiros-total")
    args = parser.parse_args()

    report = BuildReport(generated_files=[], warnings=[])
    build_onibus(
        Path(args.base_dir),
        Path(args.out_dir),
        report,
        args.sep,
        args.col_id_linha_onibus,
        args.col_passageiros_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
