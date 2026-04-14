from __future__ import annotations

import logging
import os

import chromadb
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.preprocessing import normalize

from churn_retrieval.config import ModelConfig
from churn_retrieval.constants import NUMERIC_COLUMNS
from churn_retrieval.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def build_openai_embeddings(model_name: str) -> OpenAIEmbeddings:
    kwargs = {
        "model": model_name,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def build_fused_embeddings(df, text_embeddings: np.ndarray, config: ModelConfig) -> np.ndarray:
    numeric_embeddings = df[NUMERIC_COLUMNS].to_numpy(dtype=float)
    weighted = np.hstack(
        [
            text_embeddings * config.text_weight,
            numeric_embeddings * config.numeric_weight,
        ]
    )
    return normalize(weighted)


def create_collection(config: ModelConfig):
    ensure_dir(config.chroma_dir)
    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    try:
        client.delete_collection(name=config.collection_name)
    except Exception:
        pass
    return client.create_collection(name=config.collection_name)


def add_embeddings_in_batches(collection, train_df, train_embeddings: np.ndarray, batch_size: int) -> None:
    total_docs = len(train_df)
    total_batches = (total_docs + batch_size - 1) // batch_size
    logger.info(
        "begin storing train embeddings: total_docs=%s batch_size=%s total_batches=%s",
        total_docs,
        batch_size,
        total_batches,
    )

    metadatas = [{"key_id": value} for value in train_df["key_label"].tolist()]
    documents = train_df["text_feature"].tolist()
    ids = [f"doc_{index}" for index in range(total_docs)]
    embeddings = train_embeddings.tolist()

    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        collection.add(
            ids=ids[start_idx:end_idx],
            embeddings=embeddings[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],
        )
        logger.info("stored batch %s:%s", start_idx, end_idx)


def predict_by_retrieval(collection, valid_df, valid_embeddings: np.ndarray, config: ModelConfig):
    results: list[list[object]] = []
    n_results = max(1, min(config.top_k, collection.count()))

    for row_index, (_, row) in enumerate(valid_df.iterrows()):
        query_result = collection.query(
            query_embeddings=[valid_embeddings[row_index].tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        metadatas = query_result.get("metadatas", [[]])[0]
        votes = [int(metadata["key_id"].split(":")[1]) for metadata in metadatas]
        average_vote = sum(votes) / max(len(votes), 1)
        prediction = int(average_vote >= config.vote_threshold)
        results.append([row["key_label"], prediction, sum(votes)])

    return results
