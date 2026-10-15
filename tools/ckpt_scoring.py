#!/usr/bin/env python3
"""Shared checkpoint scoring helpers for BioTalk experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_METRICS: Sequence[str] = ("MVE", "LVE", "UFVE", "FDD")
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "MVE": 0.20,
    "LVE": 0.15,
    "UFVE": 0.35,
    "FDD": 0.30,
}


def normalize_weights(metrics: Sequence[str],
                      weights: Mapping[str, float] | None) -> Dict[str, float]:
    m = [str(x) for x in metrics]
    if not m:
        raise ValueError("metrics cannot be empty")
    if weights is None:
        weights = DEFAULT_WEIGHTS
    out: Dict[str, float] = {}
    for key in m:
        if key not in weights:
            raise ValueError(f"missing weight for metric: {key}")
        out[key] = float(weights[key])
    s = sum(out.values())
    if s <= 0:
        raise ValueError("weights sum must be > 0")
    # Normalize for stable cross-script behavior.
    return {k: v / s for k, v in out.items()}


def _to_float_row(row: Mapping[str, object],
                  metrics: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in metrics:
        if metric not in row:
            raise ValueError(f"row missing metric: {metric}")
        out[metric] = float(row[metric])  # type: ignore[arg-type]
    return out


def score_rows_by_rank(rows: Iterable[Mapping[str, object]],
                       metrics: Sequence[str] = DEFAULT_METRICS,
                       weights: Mapping[str, float] | None = None
                       ) -> List[Dict[str, object]]:
    metrics = [str(x) for x in metrics]
    w = normalize_weights(metrics, weights)
    scored: List[Dict[str, object]] = [dict(r) for r in rows]
    if not scored:
        return []

    for idx, row in enumerate(scored):
        vals = _to_float_row(row, metrics)
        for k, v in vals.items():
            row[k] = v
        row["_row_index"] = idx

    for metric in metrics:
        order = sorted(range(len(scored)),
                       key=lambda i: float(scored[i][metric]))
        for rank, ridx in enumerate(order, start=1):
            scored[ridx][f"rank_{metric}"] = float(rank)

    for row in scored:
        rank_score = 0.0
        for metric in metrics:
            rank_score += w[metric] * float(row[f"rank_{metric}"])
        row["rank_score"] = rank_score

    scored.sort(
        key=lambda r: (float(r["rank_score"]),
                       *(float(r[m]) for m in metrics),
                       int(r["_row_index"])))
    return scored


def pick_best_by_rank(rows: Iterable[Mapping[str, object]],
                      metrics: Sequence[str] = DEFAULT_METRICS,
                      weights: Mapping[str, float] | None = None
                      ) -> Dict[str, object]:
    scored = score_rows_by_rank(rows, metrics=metrics, weights=weights)
    if not scored:
        raise ValueError("rows is empty")
    return scored[0]

