"""Orchestrate Looker Studio export builds."""
from __future__ import annotations

import argparse
from pathlib import Path

from .01_build_dim_distritos_od import build_dim_distritos
from .02_build_od_fatos import build_od_fatos
from .03_build_trilhos import build_trilhos
from .04_build_onibus import build_onibus
from .utils import BuildReport, ensure_directory, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all Looker outputs.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--out-dir", default="data_out/looker")
    parser.add_argument("--geo-simplify-tolerance", type=float, default=0.0005)
    parser.add_argument("--nearest-radius-m", type=float, default=200)
    parser.add_argument("--sep", default=",")
    args = parser.parse_args()

    logger = setup_logging()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    ensure_directory(out_dir / "geo")
    ensure_directory(out_dir / "od")
    ensure_directory(out_dir / "trilhos")
    ensure_directory(out_dir / "onibus")

    report = BuildReport(generated_files=[], warnings=[])

    build_dim_distritos(
        base_dir,
        out_dir / "geo",
        args.geo_simplify_tolerance,
        report,
        args.sep,
    )
    build_od_fatos(base_dir, out_dir / "od", report, args.sep)
    build_trilhos(
        base_dir,
        out_dir / "trilhos",
        args.nearest_radius_m,
        report,
        args.sep,
    )
    build_onibus(base_dir, out_dir / "onibus", report, args.sep)

    logger.info("Relatório final Looker Studio:")
    print("\nRELATORIO LOOKER STUDIO")
    print("=======================")
    if report.generated_files:
        print("Arquivos gerados:")
        for path, rows in report.generated_files:
            print(f"- {path} ({rows} linhas)")
    else:
        print("Nenhum arquivo gerado.")
    if report.warnings:
        print("\nAvisos:")
        for warning in report.warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
