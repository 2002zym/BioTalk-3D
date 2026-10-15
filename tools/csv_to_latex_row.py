#!/usr/bin/env python3
"""Convert BioTalk result CSV rows to LaTeX table rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


DEFAULT_COLUMNS = [
    "bs_mae_51d",
    "bs_mse_51d",
    "mouth_mae_51d",
    "upper_face_mae_51d",
    "MVE",
    "LVE",
    "UFVE",
    "FDD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV row to LaTeX row.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--columns", type=str, default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--precision", type=int, default=4)
    return parser.parse_args()


def _format_value(row: Dict[str, str], key: str, precision: int) -> str:
    raw = row.get(key, "")
    if raw == "":
        return "-"
    try:
        return f"{float(raw):.{precision}f}"
    except ValueError:
        return raw


def main() -> None:
    args = parse_args()
    cols: List[str] = [c.strip() for c in args.columns.split(",") if c.strip()]
    rows = list(csv.DictReader(args.csv.open("r", encoding="utf-8")))
    matched = [
        r for r in rows
        if r.get("protocol") == args.protocol and r.get("baseline") == args.baseline
    ]
    if not matched:
        raise RuntimeError(
            f"No matched row for protocol={args.protocol}, baseline={args.baseline}"
        )
    row = matched[-1]
    values = [_format_value(row, key, args.precision) for key in cols]
    latex = f"{args.protocol} & {args.baseline} & " + " & ".join(values) + r" \\"
    print(latex)


if __name__ == "__main__":
    main()
