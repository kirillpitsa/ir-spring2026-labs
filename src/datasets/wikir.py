from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.ranking import ensure_str_ids


def load_wikir_split(data_root: Path, split: str = "test"):
    wikir_root = Path(data_root) / "wikIR1k"

    docs = pd.read_csv(wikir_root / "documents.csv", dtype={"id_right": str})
    docs = ensure_str_ids(
        docs.rename(columns={"id_right": "doc_id", "text_right": "text"}), cols=("doc_id",)
    )

    queries = pd.read_csv(wikir_root / split / "queries.csv", dtype={"id_left": str})
    queries = ensure_str_ids(
        queries.rename(columns={"id_left": "query_id", "text_left": "text"}), cols=("query_id",)
    )

    qrels = pd.read_csv(
        wikir_root / split / "qrels",
        sep="\t",
        header=None,
        names=["query_id", "unused", "doc_id", "relevance"],
        dtype={0: str, 2: str, 3: int},
    )
    qrels = ensure_str_ids(qrels[["query_id", "doc_id", "relevance"]].copy())
    qrels["relevance"] = qrels["relevance"].astype(int)

    return docs, queries, qrels


def load_trec_run(path: Path, run_name: str | None = None):
    run_df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "run_name"],
        dtype={0: str, 2: str},
    )
    run_df = ensure_str_ids(run_df)
    run_df["score"] = run_df["score"].astype(float)
    run_df["rank"] = run_df["rank"].astype(int) + 1

    if run_name is not None:
        run_df["run_name"] = run_name

    return run_df


def load_wikir_bm25_run(data_root: Path, split: str = "test"):
    wikir_root = Path(data_root) / "wikIR1k"
    return load_trec_run(wikir_root / split / "BM25.res", run_name=f"bm25_{split}")
