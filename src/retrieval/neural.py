from __future__ import annotations

from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.ranking import build_run_for_single_query


def get_best_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _embed_with_cache(
    model, texts, ids, model_name, prefix, cache_dir, is_query, batch_size,
):
    cache_path: Path | None = None
    if cache_dir is not None:
        ids_hash = md5(",".join(str(i) for i in ids).encode()).hexdigest()[:12]
        short_name = model_name.split("/")[-1]
        cache_path = Path(cache_dir) / f"{prefix}__{short_name}__{ids_hash}.npy"
        if cache_path.exists():
            return np.load(str(cache_path))

    encode_fn = model.encode_query if is_query else model.encode_document
    embeddings = np.asarray(
        encode_fn(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True),
        dtype=np.float32,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)

    return embeddings


def run_dense_retrieval(
    model_name: str,
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    top_k: int = 20,
    batch_size: int = 64,
    device: str | None = None,
    run_name: str | None = None,
    cache_dir: Path | None = None,
):
    import torch
    from sentence_transformers import SentenceTransformer, util

    device = device or get_best_device()
    model = SentenceTransformer(model_name, device=device)

    doc_ids = docs_df["doc_id"].tolist()
    query_ids = queries_df["query_id"].tolist()

    corpus_embeddings = _embed_with_cache(
        model, docs_df["text"].tolist(), doc_ids,
        model_name=model_name, prefix="corpus", cache_dir=cache_dir,
        is_query=False, batch_size=batch_size,
    )
    query_embeddings = _embed_with_cache(
        model, queries_df["text"].tolist(), query_ids,
        model_name=model_name, prefix="queries", cache_dir=cache_dir,
        is_query=True, batch_size=batch_size,
    )

    hits = util.semantic_search(
        torch.from_numpy(query_embeddings),
        torch.from_numpy(corpus_embeddings),
        top_k=top_k,
        score_function=util.dot_score,
    )

    effective_run_name = run_name or model_name
    rows = []
    for query_id, query_hits in zip(query_ids, hits):
        scores = [hit["score"] for hit in query_hits]
        hit_doc_ids = [doc_ids[hit["corpus_id"]] for hit in query_hits]
        rows.extend(build_run_for_single_query(
            query_id=query_id, doc_ids=hit_doc_ids, scores=scores,
            run_name=effective_run_name, top_k=top_k,
        ))

    return pd.DataFrame(rows)


def score_candidate_pairs_biencoder(
    model_name: str,
    queries_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    batch_size: int = 64,
    device: str | None = None,
    output_col: str = "dense_score",
    cache_dir: Path | None = None,
):
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or get_best_device()
    model = SentenceTransformer(model_name, device=device)

    query_subset = (
        candidates_df[["query_id"]].drop_duplicates()
        .merge(queries_df[["query_id", "text"]], on="query_id", how="left")
    )
    doc_subset = (
        candidates_df[["doc_id"]].drop_duplicates()
        .merge(docs_df[["doc_id", "text"]], on="doc_id", how="left")
    )

    query_ids = query_subset["query_id"].tolist()
    doc_ids = doc_subset["doc_id"].tolist()

    query_embeddings = _embed_with_cache(
        model, query_subset["text"].tolist(), query_ids,
        model_name=model_name, prefix="queries", cache_dir=cache_dir,
        is_query=True, batch_size=batch_size,
    )
    doc_embeddings = _embed_with_cache(
        model, doc_subset["text"].tolist(), doc_ids,
        model_name=model_name, prefix="docs", cache_dir=cache_dir,
        is_query=False, batch_size=batch_size,
    )

    query_lookup = {qid: i for i, qid in enumerate(query_ids)}
    doc_lookup = {did: i for i, did in enumerate(doc_ids)}

    q_emb = torch.from_numpy(query_embeddings)
    d_emb = torch.from_numpy(doc_embeddings)
    q_idx = torch.tensor(candidates_df["query_id"].map(query_lookup).to_numpy())
    d_idx = torch.tensor(candidates_df["doc_id"].map(doc_lookup).to_numpy())
    scores = (q_emb[q_idx] * d_emb[d_idx]).sum(dim=1).numpy()

    result = candidates_df.copy()
    result[output_col] = scores.astype(float)
    return result


def score_candidate_pairs_cross_encoder(
    model_name: str,
    queries_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    batch_size: int = 32,
    device: str | None = None,
    output_col: str = "rerank_score",
    max_length: int = 512,
):
    from sentence_transformers import CrossEncoder

    device = device or get_best_device()
    model = CrossEncoder(model_name, device=device, max_length=max_length)

    merged = (
        candidates_df[["query_id", "doc_id"]]
        .merge(queries_df[["query_id", "text"]].rename(columns={"text": "query_text"}), on="query_id", how="left")
        .merge(docs_df[["doc_id", "text"]].rename(columns={"text": "doc_text"}), on="doc_id", how="left")
    )

    pairs = list(zip(merged["query_text"], merged["doc_text"]))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    result = candidates_df.copy()
    result[output_col] = np.asarray(scores, dtype=float)
    return result
