from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from churn_retrieval.config import EvaluationConfig
from churn_retrieval.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def evaluate_predictions(config: EvaluationConfig) -> dict[str, float]:
    ensure_dir(config.metrics_dir)
    prediction_df = pd.read_csv(config.prediction_path)
    for column in ["true_label", "pred_label"]:
        prediction_df[column] = prediction_df[column].astype(float).astype(int)

    y_true = prediction_df["true_label"].values
    y_pred = prediction_df["pred_label"].values

    metrics = {
        "auc": float(roc_auc_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    metrics_csv_path = config.metrics_dir / config.metrics_filename
    metrics_json_path = config.metrics_dir / config.summary_filename
    roc_path = config.metrics_dir / config.roc_filename

    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    logger.info("saved metrics csv -> %s", metrics_csv_path)
    logger.info("saved metrics json -> %s", metrics_json_path)
    logger.info("saved roc curve -> %s", roc_path)
    return metrics
