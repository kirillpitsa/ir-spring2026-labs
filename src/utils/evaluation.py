from __future__ import annotations

import numpy as np
import pandas as pd
import ir_measures
from ir_measures import AP, P, nDCG

from .ranking import ensure_str_ids

DEFAULT_MEASURES = {
    "P@1": P(rel=1, judged_only=False) @ 1,
    "P@10": P(rel=1, judged_only=False) @ 10,
    "P@20": P(rel=1, judged_only=False) @ 20,
    "MAP@20": AP(rel=1, judged_only=False) @ 20,
    "nDCG@20": nDCG(
        dcg="log2",
        gains={0: 0, 1: 1, 2: 2},
        judged_only=False,
    ) @ 20,
}


def _prepare_qrels(qrels_df: pd.DataFrame):
    qrels = ensure_str_ids(qrels_df[["query_id", "doc_id", "relevance"]].copy())
    qrels["relevance"] = qrels["relevance"].astype(int)
    return qrels


def _prepare_run(run_df: pd.DataFrame, score_col: str = "score"):
    run = ensure_str_ids(run_df[["query_id", "doc_id", score_col]].copy())
    run[score_col] = run[score_col].astype(float)

    if score_col != "score":
        run = run.rename(columns={score_col: "score"})

    return run


def evaluate_measure(run_df: pd.DataFrame, qrels_df: pd.DataFrame, measure):
    scores = ir_measures.calc_aggregate(
        [measure],
        _prepare_qrels(qrels_df),
        _prepare_run(run_df),
    )
    return float(scores[measure])


def evaluate_run(
    run_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    measures: dict | None = None,
):
    measures = measures or DEFAULT_MEASURES
    scores = ir_measures.calc_aggregate(
        list(measures.values()),
        _prepare_qrels(qrels_df),
        _prepare_run(run_df),
    )
    return {name: float(scores[measure]) for name, measure in measures.items()}


def normalize_minmax_by_query(
    df: pd.DataFrame,
    score_col: str = "score",
    output_col: str = "score_minmax",
):
    result = df.copy()
    min_scores = result.groupby("query_id")[score_col].transform("min")
    max_scores = result.groupby("query_id")[score_col].transform("max")
    denom = max_scores - min_scores

    result[output_col] = np.where(
        denom > 0,
        (result[score_col] - min_scores) / denom,
        0.0,
    )
    return result
