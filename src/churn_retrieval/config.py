from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tomllib

from churn_retrieval.utils.io import resolve_path


@dataclass(slots=True)
class PathsConfig:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    logs_dir: Path
    chroma_dir: Path


@dataclass(slots=True)
class PreprocessConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    label_lookback_months: int = 3
    churn_inactive_years: int = 3
    valid_size: float = 0.1
    random_state: int = 42
    today: pd.Timestamp | None = None


@dataclass(slots=True)
class ModelConfig:
    processed_data_dir: Path
    predictions_dir: Path
    chroma_dir: Path
    embedding_model: str = "text-embedding-3-small"
    text_weight: float = 0.3
    numeric_weight: float = 0.7
    top_k: int = 10
    vote_threshold: float = 0.4
    batch_size: int = 5000
    collection_name: str = "multimodal"


@dataclass(slots=True)
class EvaluationConfig:
    prediction_path: Path
    metrics_dir: Path
    metrics_filename: str = "metrics.csv"
    summary_filename: str = "metrics.json"
    roc_filename: str = "roc_curve.png"


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    evaluation: EvaluationConfig


def _load_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def load_config(config_path: str | Path = "configs/default.toml") -> AppConfig:
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent
    data = _load_toml(config_file)

    paths_section = data["paths"]
    preprocess_section = data["preprocess"]
    model_section = data["model"]
    evaluation_section = data["evaluation"]

    paths = PathsConfig(
        project_root=project_root,
        raw_data_dir=resolve_path(paths_section["raw_data_dir"], project_root),
        processed_data_dir=resolve_path(paths_section["processed_data_dir"], project_root),
        predictions_dir=resolve_path(paths_section["predictions_dir"], project_root),
        metrics_dir=resolve_path(paths_section["metrics_dir"], project_root),
        logs_dir=resolve_path(paths_section["logs_dir"], project_root),
        chroma_dir=resolve_path(paths_section["chroma_dir"], project_root),
    )

    today_value = preprocess_section.get("today", "")
    today = None if not today_value else pd.Timestamp(today_value)

    preprocess = PreprocessConfig(
        raw_data_dir=paths.raw_data_dir,
        processed_data_dir=paths.processed_data_dir,
        label_lookback_months=preprocess_section.get("label_lookback_months", 3),
        churn_inactive_years=preprocess_section.get("churn_inactive_years", 3),
        valid_size=preprocess_section.get("valid_size", 0.1),
        random_state=preprocess_section.get("random_state", 42),
        today=today,
    )

    model = ModelConfig(
        processed_data_dir=paths.processed_data_dir,
        predictions_dir=paths.predictions_dir,
        chroma_dir=paths.chroma_dir,
        embedding_model=model_section.get("embedding_model", "text-embedding-3-small"),
        text_weight=model_section.get("text_weight", 0.3),
        numeric_weight=model_section.get("numeric_weight", 0.7),
        top_k=model_section.get("top_k", 10),
        vote_threshold=model_section.get("vote_threshold", 0.4),
        batch_size=model_section.get("batch_size", 5000),
        collection_name=model_section.get("collection_name", "multimodal"),
    )

    prediction_path = paths.predictions_dir / evaluation_section.get("prediction_filename", "prediction.csv")
    evaluation = EvaluationConfig(
        prediction_path=prediction_path,
        metrics_dir=paths.metrics_dir,
        metrics_filename=evaluation_section.get("metrics_filename", "metrics.csv"),
        summary_filename=evaluation_section.get("summary_filename", "metrics.json"),
        roc_filename=evaluation_section.get("roc_filename", "roc_curve.png"),
    )

    return AppConfig(paths=paths, preprocess=preprocess, model=model, evaluation=evaluation)


import pandas as pd  # noqa: E402  Keep local dependency clear for dataclass type hints.
