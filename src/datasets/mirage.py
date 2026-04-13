from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset, load_from_disk
from sklearn.model_selection import StratifiedGroupKFold

MIRAGE_DATASET_NAME = "nlpai-lab/mirage"


def combine_title_and_chunk(title: str | None, chunk: str | None) -> str:
    title = (title or "").strip()
    chunk = (chunk or "").strip()
    if not chunk:
        return ""
    if title and not chunk.startswith(title):
        return f"{title}. {chunk}"
    return chunk


def make_mirage_doc_id(title: str | None, chunk: str | None) -> str:
    return "mirage_" + md5(combine_title_and_chunk(title, chunk).encode()).hexdigest()


def iter_mirage_pool_docs(example: dict):
    doc_pool = example["doc_pool"]
    if isinstance(doc_pool, dict):
        for idx in range(len(doc_pool["doc_chunk"])):
            yield {key: doc_pool[key][idx] for key in doc_pool}
    else:
        yield from doc_pool


def _make_grouped_split(dataset, test_size: float = 0.1, random_state: int = 42):
    rows = []
    for idx, example in enumerate(dataset):
        oracle = example.get("oracle") or {}
        group_key = oracle.get("mapped_id") or example.get("doc_url") or str(example["query_id"])
        rows.append({"row_idx": idx, "source": str(example["source"]), "group_key": str(group_key)})

    df = pd.DataFrame(rows)
    splitter = StratifiedGroupKFold(
        n_splits=int(round(1.0 / test_size)), shuffle=True, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, y=df["source"], groups=df["group_key"]))
    return df.iloc[train_idx]["row_idx"].tolist(), df.iloc[test_idx]["row_idx"].tolist()


def get_local_mirage_paths(data_root: Path):
    data_root = Path(data_root)
    return {"cache_dir": data_root / "mirage_cache", "split_dir": data_root / "mirage_grouped_split"}


def prepare_local_mirage_split(
    data_root: Path,
    dataset_name: str = MIRAGE_DATASET_NAME,
    test_size: float = 0.1,
    random_state: int = 42,
):
    paths = get_local_mirage_paths(data_root)
    cache_dir, split_dir = paths["cache_dir"], paths["split_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train", cache_dir=str(cache_dir))
    train_idx, test_idx = _make_grouped_split(dataset, test_size=test_size, random_state=random_state)

    DatasetDict({
        "train": dataset.select(train_idx),
        "test": dataset.select(test_idx),
    }).save_to_disk(str(split_dir / "hf"))

    metadata = {
        "dataset_name": dataset_name,
        "test_size": test_size,
        "random_state": random_state,
        "num_train": len(train_idx),
        "num_test": len(test_idx),
    }
    (split_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_local_mirage_split(data_root: Path, split: str = "test"):
    split_dir = get_local_mirage_paths(data_root)["split_dir"]
    return load_from_disk(str(split_dir / "hf"))[split]


def build_mirage_collection(dataset):
    query_rows, doc_rows, qrel_rows = [], [], []
    seen_doc_ids: set[str] = set()

    for example in dataset:
        query_id = str(example["query_id"])
        query_rows.append({"query_id": query_id, "text": example["query"], "source": example.get("source", "")})

        for doc in iter_mirage_pool_docs(example):
            title = doc.get("doc_name")
            chunk = doc.get("doc_chunk")
            full_text = combine_title_and_chunk(title, chunk)
            if not full_text:
                continue
            doc_id = make_mirage_doc_id(title, chunk)
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                doc_rows.append({"doc_id": doc_id, "title": (title or "").strip(), "text": full_text})
            qrel_rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": int(doc.get("support", 0))})

    docs = pd.DataFrame(doc_rows).reset_index(drop=True)
    queries = pd.DataFrame(query_rows).drop_duplicates("query_id").reset_index(drop=True)
    qrels = pd.DataFrame(qrel_rows).drop_duplicates().reset_index(drop=True)
    return docs, queries, qrels
