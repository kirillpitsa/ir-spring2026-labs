from __future__ import annotations

import pandas as pd
from rank_bm25 import BM25Okapi

from ..datasets.mirage import combine_title_and_chunk, iter_mirage_pool_docs, make_mirage_doc_id
from ..utils.ranking import build_run_for_single_query


def _score_query(model: BM25Okapi, doc_ids, query_id, query_text, run_name, top_k):
    scores = model.get_scores(query_text.split())
    return build_run_for_single_query(
        query_id=query_id,
        doc_ids=doc_ids,
        scores=scores,
        run_name=run_name,
        top_k=top_k,
    )


def run_bm25(
    docs_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    top_k: int = 1000,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
    run_name: str = "bm25",
):
    doc_ids = docs_df["doc_id"].tolist()
    model = BM25Okapi(docs_df["text"].tolist(), tokenizer=str.split, k1=k1, b=b, epsilon=epsilon)

    rows = []
    for query_id, query_text in queries_df[["query_id", "text"]].itertuples(index=False):
        rows.extend(_score_query(model, doc_ids, query_id, query_text, run_name, top_k))

    return pd.DataFrame(rows)


def build_mirage_bm25_run(
    dataset,
    top_k: int = 100,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
    run_name: str = "bm25_mirage",
):
    rows = []
    for example in dataset:
        doc_texts, doc_ids = [], []
        for doc in iter_mirage_pool_docs(example):
            title = doc.get("doc_name")
            chunk = doc.get("doc_chunk")
            full_text = combine_title_and_chunk(title, chunk)
            if not full_text:
                continue
            doc_texts.append(full_text)
            doc_ids.append(make_mirage_doc_id(title, chunk))

        model = BM25Okapi(doc_texts, tokenizer=str.split, k1=k1, b=b, epsilon=epsilon)
        rows.extend(_score_query(
            model, doc_ids, str(example["query_id"]), example["query"], run_name, top_k,
        ))

    return pd.DataFrame(rows)
