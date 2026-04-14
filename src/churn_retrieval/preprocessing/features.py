from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from churn_retrieval.config import PreprocessConfig

logger = logging.getLogger(__name__)


def build_feature_table(
    repair_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    purchase_df: pd.DataFrame,
    config: PreprocessConfig,
) -> pd.DataFrame:
    repair_df = repair_df.copy()
    repair_df["date"] = pd.to_datetime(repair_df["date"])
    today = config.today or pd.Timestamp.today().normalize()

    data_split_date = repair_df["date"].max() - pd.DateOffset(months=config.label_lookback_months)
    history_df = repair_df[repair_df["date"] <= data_split_date].copy()
    history_df = history_df.sort_values(["VIN", "date"], ascending=[True, False]).reset_index(drop=True)

    feat_recent = history_df.groupby("VIN", as_index=False).first()
    feat_recent["last_till_now_days"] = (today - pd.to_datetime(feat_recent["date"])).dt.days
    feat_recent = feat_recent.rename(
        columns={"date": "last_date", "mile": "last_mile", "repair_type": "last_repair_type"}
    )

    history_df["relative_last_date"] = history_df.groupby("VIN")["date"].shift(-1)
    history_df["relative_last_mile"] = history_df.groupby("VIN")["mile"].shift(-1)
    history_df = history_df.merge(purchase_df[["VIN", "purchase_date"]].drop_duplicates(), on="VIN", how="left")
    history_df["purchase_date"] = pd.to_datetime(history_df["purchase_date"])
    history_df.loc[history_df["relative_last_date"].isna(), "relative_last_date"] = history_df["purchase_date"]
    history_df.loc[history_df["relative_last_mile"].isna(), "relative_last_mile"] = 0

    history_df["day_diff"] = (
        pd.to_datetime(history_df["date"]) - pd.to_datetime(history_df["relative_last_date"])
    ).dt.days
    history_df["mile_diff"] = history_df["mile"] - history_df["relative_last_mile"]
    history_df["day_speed"] = history_df["mile_diff"] / history_df["day_diff"]
    history_df = history_df[history_df["day_diff"] > 0].copy()

    fill_df = history_df[history_df["mile_diff"] >= 0].copy()
    vin_fill_df = fill_df.groupby("VIN", as_index=False)["day_speed"].median().rename(
        columns={"day_speed": "median_day_speed"}
    )
    history_df = history_df.merge(vin_fill_df, on="VIN", how="left")

    abnormal_mask = history_df["mile_diff"] < 0
    history_df.loc[abnormal_mask, "day_speed"] = history_df.loc[abnormal_mask, "median_day_speed"]
    history_df.loc[abnormal_mask, "mile_diff"] = (
        history_df.loc[abnormal_mask, "day_diff"] * history_df.loc[abnormal_mask, "median_day_speed"]
    )
    history_df.loc[abnormal_mask, "relative_last_mile"] = (
        history_df.loc[abnormal_mask, "mile"] - history_df.loc[abnormal_mask, "mile_diff"]
    )
    history_df = history_df.drop(columns=["median_day_speed"])

    churn_cutoff = pd.to_datetime(data_split_date) - pd.DateOffset(years=config.churn_inactive_years)
    max_service_df = history_df.groupby("VIN", as_index=False)["date"].max().rename(columns={"date": "max_date"})
    active_vins = max_service_df[max_service_df["max_date"] >= churn_cutoff]["VIN"]
    history_df = history_df[history_df["VIN"].isin(active_vins)].copy()

    history_df = history_df.sort_values(["VIN", "date"], ascending=[True, False]).copy()
    history_df["rk"] = history_df.groupby("VIN").cumcount() + 1

    feat_first = history_df[history_df["rk"] == 1][["VIN", "day_diff", "mile_diff"]].drop_duplicates().rename(
        columns={
            "day_diff": "first_to_purchase_day_diff",
            "mile_diff": "first_to_purchase_mile_diff",
        }
    )
    feat_second = history_df[history_df["rk"] == 2][["VIN", "day_diff", "mile_diff"]].drop_duplicates().rename(
        columns={
            "day_diff": "second_to_first_day_diff",
            "mile_diff": "second_to_first_mile_diff",
        }
    )

    feat_stats = (
        history_df.groupby("VIN")
        .agg(
            day_diff_median=("day_diff", "median"),
            day_diff_std=("day_diff", "std"),
            day_diff_mean=("day_diff", "mean"),
            mile_diff_median=("mile_diff", "median"),
            mile_diff_std=("mile_diff", "std"),
            mile_diff_mean=("mile_diff", "mean"),
            day_speed_median=("day_speed", "median"),
            day_speed_std=("day_speed", "std"),
            day_speed_mean=("day_speed", "mean"),
        )
        .reset_index()
    )
    feat_stats["day_cv"] = feat_stats["day_diff_std"] / feat_stats["day_diff_mean"]
    feat_stats["mile_cv"] = feat_stats["mile_diff_std"] / feat_stats["mile_diff_mean"]
    feat_stats["day_speed_cv"] = feat_stats["day_speed_std"] / feat_stats["day_speed_mean"]

    for column in ["day_cv", "mile_cv", "day_speed_cv"]:
        feat_stats[column] = feat_stats[column].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    for column in ["day_diff_std", "mile_diff_std", "day_speed_std"]:
        feat_stats[column] = feat_stats[column].fillna(0).astype(float)

    feat_stats = feat_stats[feat_stats["mile_diff_median"].notna()].copy()
    feat_count = history_df.groupby("VIN", as_index=False).agg(all_times=("date", "count"))
    feat_types = history_df.groupby("VIN")["repair_type"].agg(";".join).rename("all_repair_types").reset_index()
    feat_types["all_repair_types"] = feat_types["all_repair_types"].apply(_deduplicate_semicolon_values)

    feature_df = (
        feat_recent.merge(feat_first, on="VIN", how="inner")
        .merge(feat_second, on="VIN", how="inner")
        .merge(feat_stats, on="VIN", how="inner")
        .merge(feat_count, on="VIN", how="inner")
        .merge(feat_types, on="VIN", how="inner")
        .merge(customer_df, on="VIN", how="inner")
    )

    feature_df["car_age"] = (today - pd.to_datetime(feature_df["purchase_date"])).dt.days
    feature_df["car_age"] = feature_df["car_age"].apply(lambda value: math.ceil(value / 365))

    feature_df["relative_next_instore_date"] = pd.to_datetime(feature_df["last_date"]) + pd.to_timedelta(
        feature_df["day_diff_median"], unit="D"
    )
    feature_df["max_relative_next_instore_date"] = (
        pd.to_datetime(feature_df["relative_next_instore_date"]) + pd.DateOffset(months=config.label_lookback_months)
    )
    feature_df["churn_label"] = (
        feature_df["max_relative_next_instore_date"] <= data_split_date
    ).astype(int)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.valid_size,
        random_state=config.random_state,
    )
    train_index, valid_index = next(splitter.split(feature_df, feature_df["churn_label"]))

    train_df = feature_df.iloc[train_index].reset_index(drop=True)
    train_df["dataset"] = "train"
    valid_df = feature_df.iloc[valid_index].reset_index(drop=True)
    valid_df["dataset"] = "valid"

    final_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    logger.info("feature table prepared: %s rows", len(final_df))
    return final_df


def _deduplicate_semicolon_values(text: str) -> str:
    values = [value.strip() for value in str(text).split(";") if value.strip()]
    return ";".join(sorted(set(values)))
