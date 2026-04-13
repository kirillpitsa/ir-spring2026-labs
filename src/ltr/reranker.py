from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from ..utils.evaluation import normalize_minmax_by_query
from ..utils.ranking import make_ranked_run

FEATURE_COLS = (
    "bm25_norm", "dense_score",
    "tf_sum", "tf_max", "idf_sum", "idf_max",
    "doc_len", "query_len", "matched_terms_ratio", "min_span",
)


def _prepare_ltr_pool(
    scored_df: pd.DataFrame,
    qrels_df: pd.DataFrame | None,
    feature_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, Pool]:
    df = normalize_minmax_by_query(scored_df, score_col="score", output_col="bm25_norm")

    if qrels_df is not None:
        df = df.merge(qrels_df[["query_id", "doc_id", "relevance"]], on=["query_id", "doc_id"], how="left")
        df["relevance"] = df["relevance"].fillna(0).astype(int)

    df = df.sort_values("query_id").reset_index(drop=True)
    query_encoder = {q: i for i, q in enumerate(df["query_id"].unique())}
    group_id = df["query_id"].map(query_encoder).to_numpy(dtype=np.int64)

    X = df[list(feature_cols)].to_numpy(dtype=np.float32)
    y = df["relevance"].to_numpy(dtype=np.float32) if qrels_df is not None else None
    return df, Pool(X, y, group_id=group_id)


def train_catboost_ltr(
    scored_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
    loss: str = "YetiRank",
    depth: int = 6,
    iterations: int = 200,
) -> CatBoostRanker:
    _, pool = _prepare_ltr_pool(scored_df, qrels_df, feature_cols)
    model = CatBoostRanker(loss_function=loss, depth=depth, iterations=iterations, verbose=False)
    model.fit(pool)
    return model


def apply_catboost_ltr(
    model: CatBoostRanker,
    scored_df: pd.DataFrame,
    feature_cols: tuple[str, ...] = FEATURE_COLS,
    top_k: int = 20,
    run_name: str = "catboost_ltr",
) -> pd.DataFrame:
    df, pool = _prepare_ltr_pool(scored_df, qrels_df=None, feature_cols=feature_cols)
    df = df.copy()
    df["ltr_score"] = model.predict(pool)
    return make_ranked_run(df, score_col="ltr_score", run_name=run_name, top_k=top_k)
