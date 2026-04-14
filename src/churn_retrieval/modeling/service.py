from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from churn_retrieval.config import ModelConfig
from churn_retrieval.modeling.feature_builder import prepare_model_input
from churn_retrieval.modeling.retrieval import (
    add_embeddings_in_batches,
    build_fused_embeddings,
    build_openai_embeddings,
    create_collection,
    predict_by_retrieval,
)
from churn_retrieval.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def run_prediction(config: ModelConfig) -> Path:
    ensure_dir(config.predictions_dir)

    cleaned_csv_path = config.processed_data_dir / "cleaned_data.csv"
    df = prepare_model_input(cleaned_csv_path)

    train_df = df[df["dataset"] == "train"].reset_index(drop=True)
    valid_df = df[df["dataset"] == "valid"].reset_index(drop=True)
    merged_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    embeddings_model = build_openai_embeddings(config.embedding_model)
    text_embeddings = np.array(embeddings_model.embed_documents(merged_df["text_feature"].tolist()))
    fused_embeddings = build_fused_embeddings(merged_df, text_embeddings, config)

    train_size = len(train_df)
    train_embeddings = fused_embeddings[:train_size]
    valid_embeddings = fused_embeddings[train_size:]

    collection = create_collection(config)
    add_embeddings_in_batches(collection, train_df, train_embeddings, config.batch_size)
    results = predict_by_retrieval(collection, valid_df, valid_embeddings, config)

    prediction_df = pd.DataFrame(results, columns=["key_label", "pred_label", "total_score"])
    prediction_df["VIN"] = prediction_df["key_label"].apply(lambda value: value.split(":")[0])
    prediction_df["true_label"] = prediction_df["key_label"].apply(lambda value: int(value.split(":")[1]))

    output_path = config.predictions_dir / "prediction.csv"
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("saved predictions -> %s", output_path)
    return output_path
