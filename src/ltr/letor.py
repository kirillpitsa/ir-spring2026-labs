from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from catboost.utils import eval_metric


def parse_letor_file(path: Path):
    X, y, group_id = [], [], []
    with path.open() as source:
        for line in source:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            y.append(float(parts[0]))
            group_id.append(int(parts[1].split(":", 1)[1]))
            X.append([float(token.split(":", 1)[1]) for token in parts[2:]])
    return {
        "X": np.array(X, dtype=np.float32),
        "y": np.array(y, dtype=np.float32),
        "group_id": np.array(group_id, dtype=np.int64),
    }


def _evaluate_ranking(y_true: np.ndarray, scores: np.ndarray, group_id: np.ndarray, top_k: int = 10):
    y_binary = (y_true > 0).astype(int)
    return {
        f"Precision@{top_k}": eval_metric(y_binary, scores, f"PrecisionAt:top={top_k}", group_id=group_id)[0],
        f"MAP@{top_k}": eval_metric(y_binary, scores, f"MAP:top={top_k}", group_id=group_id)[0],
        f"NDCG@{top_k}": eval_metric(y_true, scores, f"NDCG:top={top_k}", group_id=group_id)[0],
    }


def evaluate_saved_letor_models(
    data_root: Path,
    models_dir: Path,
    losses=None,
    folds=None,
    top_k: int = 10,
):
    if losses is None:
        losses = ["RMSE", "QueryRMSE", "YetiRank"]
    if folds is None:
        folds = [1, 2, 3, 4, 5]

    rows = []
    for fold in folds:
        test_data = parse_letor_file(Path(data_root) / f"Fold{fold}" / "test.txt")
        test_pool = Pool(test_data["X"], test_data["y"], group_id=test_data["group_id"])
        for loss_name in losses:
            model = CatBoostRanker()
            model.load_model(Path(models_dir) / f"{loss_name.lower()}_fold_{fold}.cbm")
            scores = model.predict(test_pool)
            rows.append({
                "loss": loss_name,
                "fold": fold,
                **_evaluate_ranking(test_data["y"], scores, test_data["group_id"], top_k=top_k),
            })

    results_df = pd.DataFrame(rows)
    summary_df = (
        results_df
        .groupby("loss", as_index=False)
        .agg(**{
            f"Precision@{top_k}": (f"Precision@{top_k}", "mean"),
            f"MAP@{top_k}": (f"MAP@{top_k}", "mean"),
            f"NDCG@{top_k}": (f"NDCG@{top_k}", "mean"),
        })
        .sort_values(f"NDCG@{top_k}", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, summary_df
