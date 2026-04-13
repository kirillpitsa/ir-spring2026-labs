from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def ensure_str_ids(df: pd.DataFrame, cols: Iterable[str] = ("query_id", "doc_id")) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def build_run_for_single_query(
    query_id: str,
    doc_ids,
    scores,
    run_name: str,
    top_k: int | None = None,
):
    scores = np.asarray(scores, dtype=float)
    top_k = len(scores) if top_k is None else min(top_k, len(scores))

    if top_k == 0:
        return []

    if top_k == len(scores):
        ranked_idx = np.argsort(-scores)
    else:
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked_idx = top_idx[np.argsort(-scores[top_idx])]

    rows = []
    for rank, idx in enumerate(ranked_idx, start=1):
        rows.append(
            {
                "query_id": str(query_id),
                "Q0": "Q0",
                "doc_id": str(doc_ids[idx]),
                "rank": rank,
                "score": float(scores[idx]),
                "run_name": run_name,
            }
        )
    return rows


def make_ranked_run(
    scored_df: pd.DataFrame,
    score_col: str = "score",
    run_name: str = "run",
    top_k: int | None = None,
):
    run_df = ensure_str_ids(scored_df[["query_id", "doc_id", score_col]].copy())
    run_df[score_col] = run_df[score_col].astype(float)

    run_df = run_df.sort_values(
        ["query_id", score_col, "doc_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    run_df["rank"] = run_df.groupby("query_id").cumcount() + 1

    if top_k is not None:
        run_df = run_df[run_df["rank"] <= top_k].reset_index(drop=True)

    if score_col != "score":
        run_df = run_df.rename(columns={score_col: "score"})

    run_df["Q0"] = "Q0"
    run_df["run_name"] = run_name
    return run_df[["query_id", "Q0", "doc_id", "rank", "score", "run_name"]]


def top_k_per_query(df: pd.DataFrame, k: int, rank_col: str = "rank"):
    ordered = df.sort_values(["query_id", rank_col], ascending=[True, True])
    return ordered.groupby("query_id", as_index=False).head(k).reset_index(drop=True)
