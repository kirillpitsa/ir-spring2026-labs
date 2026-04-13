from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.evaluation import DEFAULT_MEASURES, evaluate_measure, normalize_minmax_by_query
from ..utils.ranking import make_ranked_run


def apply_mixture(
    candidates_df: pd.DataFrame,
    alpha: float,
    bm25_col: str = "score",
    neural_col: str = "dense_score",
    bm25_norm_col: str = "bm25_norm",
    top_k: int = 20,
    run_name: str = "mixture",
):
    mixed = normalize_minmax_by_query(
        candidates_df,
        score_col=bm25_col,
        output_col=bm25_norm_col,
    )
    mixed["mixture_score"] = alpha * mixed[bm25_norm_col] + (1.0 - alpha) * mixed[neural_col]

    run_df = make_ranked_run(
        mixed,
        score_col="mixture_score",
        run_name=run_name,
        top_k=top_k,
    )
    return mixed, run_df


def tune_alpha(
    candidates_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    alphas=None,
    metric_name: str = "nDCG@20",
    bm25_col: str = "score",
    neural_col: str = "dense_score",
    top_k: int = 20,
):
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 21)

    # Normalize BM25 once — it doesn't depend on alpha
    normed = normalize_minmax_by_query(candidates_df, score_col=bm25_col, output_col="_bm25_norm")
    metric = DEFAULT_MEASURES[metric_name]
    rows = []

    for alpha in alphas:
        normed["_mix"] = alpha * normed["_bm25_norm"] + (1.0 - alpha) * normed[neural_col]
        run_df = make_ranked_run(normed, score_col="_mix", run_name=f"tune_{alpha:.2f}", top_k=top_k)
        rows.append({"alpha": float(alpha), metric_name: evaluate_measure(run_df, qrels_df, measure=metric)})

    tuning_df = pd.DataFrame(rows).sort_values(metric_name, ascending=False).reset_index(drop=True)
    return float(tuning_df.iloc[0]["alpha"]), tuning_df
