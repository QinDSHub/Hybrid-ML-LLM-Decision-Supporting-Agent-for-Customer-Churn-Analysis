from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from churn_retrieval.constants import NUMERIC_COLUMNS, TEXT_COLUMNS

logger = logging.getLogger(__name__)


def auto_scale(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    for col in num_cols:
        col_data = df[col].values.reshape(-1, 1)
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        skew = pd.Series(col_data.flatten()).skew()
        if iqr > 0 and (np.max(col_data) - np.min(col_data)) / iqr > 10:
            scaler = RobustScaler()
        elif abs(skew) > 1:
            scaler = PowerTransformer(method="yeo-johnson")
        else:
            scaler = StandardScaler()
        scaled[col] = scaler.fit_transform(col_data)
    return scaled


def car_level_to_text(value: str) -> str:
    if value == "family_1":
        return "高档车"
    if value == "family_2":
        return "中档车"
    return "低档车"


def prepare_model_input(cleaned_csv_path: str | pd.PathLike) -> pd.DataFrame:
    df = pd.read_csv(cleaned_csv_path)
    required_columns = ["VIN", "dataset", "churn_label"] + NUMERIC_COLUMNS + TEXT_COLUMNS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned data: {missing}")

    df = df[required_columns].copy()
    logger.info("model input loaded: shape=%s", df.shape)

    df["member_level"] = df["member_level"].astype(str).apply(lambda value: f"会员卡：{value}")
    df["owner_type"] = df["owner_type"].astype(str).apply(lambda value: f"用户性质：{value}")
    df["car_mode"] = df["car_mode"].astype(str).apply(lambda value: f"汽车型号：{value}")
    df["car_level"] = df["car_level"].astype(str).apply(car_level_to_text)
    df["last_repair_type"] = df["last_repair_type"].astype(str).apply(lambda value: f"上次进店类型：{value}")
    df["all_repair_types"] = df["all_repair_types"].astype(str).apply(lambda value: f"历史进店类型：{value}")

    df = auto_scale(df, NUMERIC_COLUMNS)
    label_map = {1: "用户标签：流失", 0: "用户标签：未流失"}
    df["label_txt"] = df["churn_label"].map(label_map)
    df["key_label"] = df.apply(lambda row: f"{row['VIN']}:{int(row['churn_label'])}", axis=1)
    df["text_feature"] = df[TEXT_COLUMNS].astype(str).agg("，".join, axis=1)
    return df
