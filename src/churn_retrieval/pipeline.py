from __future__ import annotations

import logging
from pathlib import Path

from churn_retrieval.config import AppConfig, load_config
from churn_retrieval.logging import setup_logging
from churn_retrieval.utils.env import load_project_env
from churn_retrieval.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def prepare_runtime(app_config: AppConfig) -> None:
    ensure_dir(app_config.paths.processed_data_dir)
    ensure_dir(app_config.paths.predictions_dir)
    ensure_dir(app_config.paths.metrics_dir)
    ensure_dir(app_config.paths.logs_dir)
    ensure_dir(app_config.paths.chroma_dir)
    setup_logging(app_config.paths.logs_dir)
    load_project_env(app_config.paths.project_root)


def run_pipeline(config_path: str | Path = "configs/default.toml") -> dict[str, object]:
    from churn_retrieval.evaluation.service import evaluate_predictions
    from churn_retrieval.modeling.service import run_prediction
    from churn_retrieval.preprocessing.service import run_preprocessing

    app_config = load_config(config_path)
    prepare_runtime(app_config)

    logger.info("starting preprocessing")
    cleaned_path = run_preprocessing(app_config.preprocess)

    logger.info("starting prediction")
    prediction_path = run_prediction(app_config.model)

    logger.info("starting evaluation")
    metrics = evaluate_predictions(app_config.evaluation)

    logger.info("pipeline completed")
    return {
        "cleaned_data": cleaned_path,
        "prediction": prediction_path,
        "metrics": metrics,
    }
