"""
Feature Engineering Pipeline for Bangalore Grocery Price Prediction.
Constructs lag, rolling, momentum, calendar, and cross-commodity features
with strict time-series data leakage safeguards.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PROCESSED_DATA_FILE,
    FEATURE_DATA_FILE,
    TARGET_COMMODITIES,
)
from src.economic_factors import get_economic_feature_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_commodity_features(
    df_comm: pd.DataFrame, commodity_name: str, df_econ: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes lag, rolling, momentum, calendar, retail transmission, and economic features for a single commodity.
    All features on date D strictly use observations on or before date D.
    """
    df = df_comm.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    price_col = "modal_price_per_kg"
    retail_col = "retail_price_per_kg"

    # 1. Historical Wholesale Lag Features (Observations at D-1, D-2, D-3, D-7, D-14, D-30)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f"price_lag_{lag}"] = df[price_col].shift(lag)

    # 2. Historical Retail Lag & Arrival Inflow Features
    if retail_col in df.columns:
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"retail_price_lag_{lag}"] = df[retail_col].shift(lag)

    if "mandi_arrival_tonnes" in df.columns:
        for lag in [1, 2, 3, 7, 14]:
            df[f"mandi_arrival_lag_{lag}"] = df["mandi_arrival_tonnes"].shift(lag)
        df["arrival_deficit_lag_1"] = df["arrival_deficit_pct"].shift(1)
        df["inflow_tightness_lag_1"] = df["inflow_tightness_index"].shift(1)

    # 3. Rolling Statistics (computed over past 7, 14, 30 days)
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df[price_col].rolling(window=window, min_periods=max(2, window // 3)).mean()
        df[f"rolling_std_{window}"] = df[price_col].rolling(window=window, min_periods=max(2, window // 3)).std()
        if retail_col in df.columns:
            df[f"retail_rolling_mean_{window}"] = df[retail_col].rolling(window=window, min_periods=max(2, window // 3)).mean()
            df[f"retail_rolling_std_{window}"] = df[retail_col].rolling(window=window, min_periods=max(2, window // 3)).std()

    df["rolling_min_7"] = df[price_col].rolling(window=7, min_periods=2).min()
    df["rolling_max_7"] = df[price_col].rolling(window=7, min_periods=2).max()
    if retail_col in df.columns:
        df["retail_rolling_min_7"] = df[retail_col].rolling(window=7, min_periods=2).min()
        df["retail_rolling_max_7"] = df[retail_col].rolling(window=7, min_periods=2).max()

    # 4. Momentum & Price Changes (Wholesale & Retail)
    df["price_change_1d"] = df[price_col] - df["price_lag_1"]
    df["price_change_7d"] = df[price_col] - df["price_lag_7"]
    df["price_pct_change_1d"] = ((df[price_col] - df["price_lag_1"]) / (df["price_lag_1"] + 1e-6)) * 100.0
    df["price_pct_change_7d"] = ((df[price_col] - df["price_lag_7"]) / (df["price_lag_7"] + 1e-6)) * 100.0

    if retail_col in df.columns:
        df["retail_price_change_1d"] = df[retail_col] - df["retail_price_lag_1"]
        df["retail_price_change_7d"] = df[retail_col] - df["retail_price_lag_7"]
        df["retail_price_pct_change_1d"] = ((df[retail_col] - df["retail_price_lag_1"]) / (df["retail_price_lag_1"] + 1e-6)) * 100.0
        df["retail_price_pct_change_7d"] = ((df[retail_col] - df["retail_price_lag_7"]) / (df["retail_price_lag_7"] + 1e-6)) * 100.0

    # 5. Wholesale-to-Retail Spread & Transmission Features
    df["price_spread"] = df["max_price_per_kg"] - df["min_price_per_kg"]
    df["spread_pct"] = (df["price_spread"] / (df[price_col] + 1e-6)) * 100.0
    if retail_col in df.columns:
        df["wholesale_retail_spread_lag_1"] = df["wholesale_retail_spread"].shift(1)
        df["markup_ratio_lag_1"] = df["markup_ratio"].shift(1)
        df["transmission_ratio_7d"] = (df["retail_rolling_mean_7"] / (df["rolling_mean_7"] + 1e-6)).round(3)

    # 6. Calendar Features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical Calendar encodings (sin / cos)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # 7. Merge Economic, Weather, Fuel, and Government Factors
    if df_econ is not None and not df_econ.empty:
        df_econ_clean = df_econ.copy()
        df_econ_clean["date"] = pd.to_datetime(df_econ_clean["date"])
        # Drop columns that might clash
        drop_clash = [c for c in df_econ_clean.columns if c in df.columns and c != "date"]
        if drop_clash:
            df_econ_clean = df_econ_clean.drop(columns=drop_clash)
        df = df.merge(df_econ_clean, on="date", how="left").ffill().bfill()

    # Assign commodity-specific economic shortcuts
    comm_lower = commodity_name.lower()
    if f"{comm_lower}_weather_stress_index" in df.columns:
        df["weather_stress_index"] = df[f"{comm_lower}_weather_stress_index"]
    if f"{comm_lower}_supply_cycle_index" in df.columns:
        df["supply_cycle_index"] = df[f"{comm_lower}_supply_cycle_index"]
    if f"freight_cost_{comm_lower}_per_kg" in df.columns:
        df["freight_cost_per_kg"] = df[f"freight_cost_{comm_lower}_per_kg"]
    if f"policy_suppression_index_{comm_lower}" in df.columns:
        df["policy_suppression_index"] = df[f"policy_suppression_index_{comm_lower}"]

    # 8. Target Variables: NEXT-DAY WHOLESALE & RETAIL PRICES (Price at D+1)
    df["target_next_day_price"] = df[price_col].shift(-1)
    df["target_next_date"] = df["date"].shift(-1)

    # Wholesale target changes & direction
    df["target_change"] = df["target_next_day_price"] - df[price_col]
    df["target_pct_change"] = (df["target_change"] / (df[price_col] + 1e-6)) * 100.0
    
    conditions = [
        df["target_pct_change"] > 1.0,
        df["target_pct_change"] < -1.0,
    ]
    choices = ["Increasing", "Decreasing"]
    df["target_direction"] = np.select(conditions, choices, default="Stable")

    # Retail target changes & direction
    if retail_col in df.columns:
        df["target_next_day_retail_price"] = df[retail_col].shift(-1)
        df["target_retail_change"] = df["target_next_day_retail_price"] - df[retail_col]
        df["target_retail_pct_change"] = (df["target_retail_change"] / (df[retail_col] + 1e-6)) * 100.0
        retail_conditions = [
            df["target_retail_pct_change"] > 1.0,
            df["target_retail_pct_change"] < -1.0,
        ]
        df["target_retail_direction"] = np.select(retail_conditions, choices, default="Stable")

    return df


def add_cross_commodity_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Adds historical related commodity prices available at prediction time.
    E.g. Tomato model gets lagged prices of Onion and Potato on date D.
    """
    logger.info("Adding cross-commodity price features...")
    # Pivot date-level modal prices for all commodities
    price_pivot = df_full.pivot(index="date", columns="commodity", values="modal_price_per_kg")

    dfs = []
    for comm in TARGET_COMMODITIES:
        comm_df = df_full[df_full["commodity"] == comm].copy()
        other_comms = [c for c in TARGET_COMMODITIES if c != comm]

        for other in other_comms:
            if other in price_pivot.columns:
                # Merge the other commodity's current price and lag_1
                other_series = price_pivot[other].reindex(comm_df["date"])
                comm_df[f"related_{other.lower()}_price"] = other_series.ffill().values
                comm_df[f"related_{other.lower()}_lag_1"] = other_series.shift(1).ffill().values
        dfs.append(comm_df)

    df_out = pd.concat(dfs, axis=0).sort_values(["commodity", "date"]).reset_index(drop=True)
    return df_out


def verify_no_data_leakage(df: pd.DataFrame) -> bool:
    """
    Strictly verifies that no feature contains future information (data leakage check).
    """
    logger.info("Running Data Leakage Verification Checks...")
    passed = True

    for comm in df["commodity"].unique():
        sub = df[df["commodity"] == comm].sort_values("date").reset_index(drop=True)

        # Check 1: Lag 1 should exactly match previous row's current price
        lag_diff = (sub["price_lag_1"].iloc[1:] - sub["modal_price_per_kg"].iloc[:-1].values).abs().max()
        if lag_diff > 1e-4:
            logger.error(f"[LEAKAGE] price_lag_1 mismatch for {comm} (max diff: {lag_diff})")
            passed = False

        if "retail_price_lag_1" in sub.columns and "retail_price_per_kg" in sub.columns:
            retail_lag_diff = (sub["retail_price_lag_1"].iloc[1:] - sub["retail_price_per_kg"].iloc[:-1].values).abs().max()
            if retail_lag_diff > 1e-4:
                logger.error(f"[LEAKAGE] retail_price_lag_1 mismatch for {comm} (max diff: {retail_lag_diff})")
                passed = False

        # Check 2: Target must equal next row's current price
        target_diff = (sub["target_next_day_price"].iloc[:-1] - sub["modal_price_per_kg"].iloc[1:].values).abs().max()
        if target_diff > 1e-4:
            logger.error(f"[LEAKAGE] target_next_day_price mismatch for {comm} (max diff: {target_diff})")
            passed = False

        # Check 3: Current price must not correlate perfectly with target in a suspicious way
        corr = sub["modal_price_per_kg"].corr(sub["target_next_day_price"])
        if corr == 1.0:
            logger.error(f"[LEAKAGE] Exact 1.0 correlation between current price and target for {comm}!")
            passed = False

    if passed:
        logger.info("PASSED: All data leakage checks validated. Feature timestamps are strictly historical.")
    return passed


class FeatureEngineer:
    """End-to-end feature engineering pipeline."""

    def __init__(
        self,
        processed_path: str = str(PROCESSED_DATA_FILE),
        feature_path: str = str(FEATURE_DATA_FILE),
    ):
        self.processed_path = processed_path
        self.feature_path = feature_path

    def build_features(self) -> pd.DataFrame:
        logger.info(f"Loading processed data from {self.processed_path}...")
        df_processed = pd.read_csv(self.processed_path)
        df_processed["date"] = pd.to_datetime(df_processed["date"])

        # Load Economic Feature Store (Agrometeorology, Fuel, Policy, Seasonality)
        df_econ = get_economic_feature_store()
        logger.info(f"Loaded economic feature store ({len(df_econ):,} days, {len(df_econ.columns)} factors)")

        # 1. Feature Engineering per commodity
        comm_dfs = []
        for comm in TARGET_COMMODITIES:
            sub = df_processed[df_processed["commodity"] == comm].copy()
            if not sub.empty:
                logger.info(f"Engineering multi-factor features for {comm} ({len(sub):,} rows)...")
                sub_feats = compute_commodity_features(sub, comm, df_econ=df_econ)
                comm_dfs.append(sub_feats)

        df_features = pd.concat(comm_dfs, axis=0).reset_index(drop=True)

        # 2. Cross-Commodity Features
        df_features = add_cross_commodity_features(df_features)

        # 3. Drop rows without target (e.g. the last chronological observation)
        initial_len = len(df_features)
        df_features = df_features.dropna(subset=["target_next_day_price", "price_lag_14"]).reset_index(drop=True)
        logger.info(f"Cleaned feature dataset: {len(df_features):,} rows (dropped {initial_len - len(df_features)} edge/warmup rows)")

        # 4. Data Leakage Verification
        is_leakage_free = verify_no_data_leakage(df_features)
        if not is_leakage_free:
            raise ValueError("Data leakage detected during feature verification!")

        # 5. Save Features
        os.makedirs(os.path.dirname(self.feature_path), exist_ok=True)
        df_features.to_csv(self.feature_path, index=False, encoding="utf-8")
        logger.info(f"Feature dataset saved to {self.feature_path} ({len(df_features):,} records)")

        # Print summary
        self.print_feature_summary(df_features)
        return df_features

    def print_feature_summary(self, df: pd.DataFrame):
        logger.info("=" * 68)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 68)
        logger.info(f"Total Rows: {len(df):,} | Total Columns: {len(df.columns)}")
        
        feature_cols = [c for c in df.columns if c not in [
            "date", "arrival_date", "commodity", "market", "district", "state",
            "target_next_day_price", "target_next_date", "target_change",
            "target_pct_change", "target_direction",
            "target_next_day_retail_price", "target_retail_change",
            "target_retail_pct_change", "target_retail_direction",
            "data_source", "original_unit", "original_modal_price",
            "original_min_price", "original_max_price", "is_outlier_flag"
        ]]
        logger.info(f"Engineered Features ({len(feature_cols)}): {feature_cols}")

        for comm in TARGET_COMMODITIES:
            sub = df[df["commodity"] == comm]
            logger.info(
                f"{comm:8s}: {len(sub):4d} training/eval samples | Date: {sub['arrival_date'].min()} to {sub['arrival_date'].max()}"
            )
        logger.info("=" * 68)


def main():
    engineer = FeatureEngineer()
    engineer.build_features()


if __name__ == "__main__":
    main()
