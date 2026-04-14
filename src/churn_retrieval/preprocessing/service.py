from __future__ import annotations

import logging
from pathlib import Path

from churn_retrieval.config import PreprocessConfig
from churn_retrieval.preprocessing.cleaning import build_customer_base_info, filter_active_repair_records
from churn_retrieval.preprocessing.datasets import load_member_data, load_repair_data, load_vehicle_data
from churn_retrieval.preprocessing.features import build_feature_table
from churn_retrieval.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def run_preprocessing(config: PreprocessConfig) -> Path:
    ensure_dir(config.processed_data_dir)

    vehicle_df = load_vehicle_data(config.raw_data_dir)
    member_df = load_member_data(config.raw_data_dir)
    repair_df = load_repair_data(config.raw_data_dir)

    customer_df, purchase_df = build_customer_base_info(vehicle_df, member_df, repair_df)
    active_repair_df = filter_active_repair_records(repair_df, customer_df)
    feature_df = build_feature_table(active_repair_df, customer_df, purchase_df, config)

    output_path = config.processed_data_dir / "cleaned_data.csv"
    feature_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("saved cleaned data -> %s", output_path)
    return output_path
