from __future__ import annotations

import logging

import pandas as pd
try:
    from natsort import natsorted
except ImportError:
    def natsorted(values):
        return sorted(values)


from churn_retrieval.constants import INTERNAL_REPAIR_PATTERN, PASSIVE_REPAIR_PATTERN

logger = logging.getLogger(__name__)


def normalize_repair_type(cell: str) -> str:
    normalized: list[str] = []
    for item in str(cell).split(";"):
        value = item.strip()
        if not value:
            continue
        if "首" in value:
            value = "首次保养"
        elif "普修" in value:
            value = "普通维修"
        normalized.append(value)
    return ";".join(natsorted(set(normalized)))


def build_customer_base_info(vehicle_df: pd.DataFrame, member_df: pd.DataFrame, repair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vehicle_df = vehicle_df[["VIN", "车主性质", "车型", "family_name"]].copy()
    vehicle_df.columns = ["VIN", "owner_type", "car_mode", "car_level"]

    member_df = member_df[["VIN", "会员等级"]].drop_duplicates().copy()
    member_df = member_df[member_df["会员等级"].notna()].reset_index(drop=True)
    member_df.columns = ["VIN", "member_level"]

    customer_df = vehicle_df.merge(member_df, on="VIN", how="outer").drop_duplicates().reset_index(drop=True)
    customer_df["member_level"] = customer_df["member_level"].fillna("无")
    customer_df["owner_type"] = customer_df["owner_type"].fillna("个人")
    customer_df = customer_df.dropna().copy()

    purchase_df = repair_df[["VIN", "purchase_date"]].drop_duplicates().dropna().reset_index(drop=True)
    customer_df = customer_df.merge(purchase_df, on="VIN", how="inner")

    logger.info("customer base info prepared: %s rows", len(customer_df))
    return customer_df, purchase_df


def filter_active_repair_records(repair_df: pd.DataFrame, customer_df: pd.DataFrame) -> pd.DataFrame:
    repair_df = repair_df[["VIN", "修理日期", "公里数", "修理类型"]].copy()
    repair_df.columns = ["VIN", "date", "mile", "repair_type"]
    repair_df = repair_df[repair_df["VIN"].isin(customer_df["VIN"].values)].copy()
    repair_df = repair_df.dropna().sort_values(["VIN", "date"]).reset_index(drop=True)
    repair_df["date"] = pd.to_datetime(repair_df["date"]).dt.normalize()

    internal_vins = repair_df[
        repair_df["repair_type"].str.contains(INTERNAL_REPAIR_PATTERN, na=False)
    ][["VIN"]].drop_duplicates()

    active_df = repair_df[~repair_df["VIN"].isin(internal_vins["VIN"].values)].copy()
    active_df = active_df[
        ~active_df["repair_type"].str.contains(PASSIVE_REPAIR_PATTERN, na=False)
    ].copy()

    mile_df = active_df.groupby(["VIN", "date"], as_index=False).agg(mile=("mile", "mean"))
    repair_type_df = (
        active_df[["VIN", "date", "repair_type"]]
        .drop_duplicates()
        .groupby(["VIN", "date"])["repair_type"]
        .agg(lambda values: ";".join(values))
        .reset_index()
    )
    repair_type_df["repair_type"] = repair_type_df["repair_type"].apply(normalize_repair_type)

    cleaned = mile_df.merge(repair_type_df, on=["VIN", "date"], how="outer")
    logger.info("active repair records prepared: %s rows", len(cleaned))
    return cleaned
