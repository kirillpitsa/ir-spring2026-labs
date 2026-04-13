from __future__ import annotations

import pandas as pd
from rank_bm25 import BM25Okapi

_ZERO_PAIR_FEATURES = {
    "tf_sum": 0.0, "tf_max": 0.0,
    "idf_sum": 0.0, "idf_max": 0.0,
    "matched_terms_ratio": 0.0, "min_span": 0,
}


def _compute_pair_features(query_tokens, query_set, doc_toks, doc_freqs, idf_dict) -> dict:
    tf_vals = [doc_freqs.get(t, 0) for t in query_tokens]
    idf_vals = [idf_dict.get(t, 0.0) for t in query_tokens]
    matched = {t for t in query_set if doc_freqs.get(t, 0) > 0}
    ratio = len(matched) / len(query_set) if query_set else 0.0

    if len(matched) <= 1:
        span = 0
    else:
        positions = [i for i, t in enumerate(doc_toks) if t in matched]
        span = (positions[-1] - positions[0]) if positions else 0

    return {
        "tf_sum": float(sum(tf_vals)),
        "tf_max": float(max(tf_vals, default=0)),
        "idf_sum": float(sum(idf_vals)),
        "idf_max": float(max(idf_vals, default=0.0)),
        "matched_terms_ratio": ratio,
        "min_span": span,
    }


def extract_lexical_features(
    scored_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add lexical features to scored_df.

    Added columns: tf_sum, tf_max, idf_sum, idf_max, doc_len, query_len,
    matched_terms_ratio, min_span.
    """
    doc_ids = docs_df["doc_id"].tolist()
    tokenized = [text.split() for text in docs_df["text"]]
    bm25 = BM25Okapi(tokenized)

    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    query_id_to_tokens = dict(zip(queries_df["query_id"], queries_df["text"].str.split()))

    doc_len_ser = pd.Series(
        [len(t) for t in tokenized], index=docs_df["doc_id"], name="doc_len"
    )
    query_len_ser = pd.Series(
        queries_df["text"].str.split().str.len().tolist(),
        index=queries_df["query_id"],
        name="query_len",
    )

    pair_rows: list[dict] = []
    for query_id, group in scored_df.groupby("query_id", sort=False):
        qtoks = query_id_to_tokens[query_id]
        qset = set(qtoks)

        for doc_id in group["doc_id"]:
            idx = doc_id_to_idx.get(doc_id)
            if idx is None:
                features = dict(_ZERO_PAIR_FEATURES)
            else:
                features = _compute_pair_features(
                    qtoks, qset, tokenized[idx], bm25.doc_freqs[idx], bm25.idf,
                )
            pair_rows.append({"query_id": query_id, "doc_id": doc_id, **features})

    pair_feat_df = pd.DataFrame(pair_rows)

    result = scored_df.copy()
    result["doc_len"] = result["doc_id"].map(doc_len_ser)
    result["query_len"] = result["query_id"].map(query_len_ser)
    result = result.merge(pair_feat_df, on=["query_id", "doc_id"], how="left")
    return result
