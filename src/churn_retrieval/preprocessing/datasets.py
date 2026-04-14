from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_retrieval.constants import MEMBER_FILE, REPAIR_FILE, VEHICLE_FILE


def load_vehicle_data(raw_data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / VEHICLE_FILE)


def load_member_data(raw_data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / MEMBER_FILE)


def load_repair_data(raw_data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_data_dir / REPAIR_FILE)
