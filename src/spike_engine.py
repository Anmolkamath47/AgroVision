"""
Sudden Price Spike, Crash, and Multi-Horizon Feature Engineering Engine.
Generates forward-looking price targets, multi-horizon price paths (1d, 3d, 7d, 14d),
and exogenous shock features (agrometeorology, deluge thresholds, diesel velocity, harvest gaps).
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PROCESSED_DATA_FILE,
    FEATURE_DATA_FILE,
    TARGET_COMMODITIES,
)
from src.features import FeatureEngineer, compute_commodity_features, add_cross_commodity_features
from src.economic_factors import get_economic_feature_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Output dataset path for spike models
SPIKE_DATASET_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "bangalore_grocery_spike_features.csv",
)


class SpikeFeatureEngine:
    """
    Engineers multi-horizon forward targets (1d, 3d, 7d, 14d), sudden spike/crash labels,
    and extreme shock trigger features for agricultural commodities in Bangalore.
    """

    def __init__(self, spike_threshold_pct: float = 20.0, spike_threshold_abs: float = 8.0):
        self.spike_threshold_pct = spike_threshold_pct
        self.spike_threshold_abs = spike_threshold_abs

    def build_spike_dataset(self) -> pd.DataFrame:
        """
        Loads the multi-factor feature dataset and computes:
        1. Multi-horizon future prices: t+1, t+3, t+7, t+14
        2. Sudden spike flags: 3-day, 7-day, 14-day surges
        3. Sudden crash flags: 3-day, 7-day, 14-day drops
        4. Peak forward price (maximum over next 14 days)
        5. Days to peak price
        """
        logger.info("Building Multi-Horizon & Sudden Spike Training Dataset...")
        
        # Ensure base features exist
        if not os.path.exists(FEATURE_DATA_FILE):
            feat_eng = FeatureEngineer()
            df_base = feat_eng.build_features()
        else:
            df_base = pd.read_csv(FEATURE_DATA_FILE)

        df_base["date"] = pd.to_datetime(df_base["date"])
        dfs = []

        for comm in TARGET_COMMODITIES:
            sub = df_base[df_base["commodity"] == comm].copy().sort_values("date").reset_index(drop=True)
            if sub.empty:
                continue

            # Multi-horizon future retail prices
            sub["target_retail_1d"] = sub["retail_price_per_kg"].shift(-1)
            sub["target_retail_3d"] = sub["retail_price_per_kg"].shift(-3)
            sub["target_retail_7d"] = sub["retail_price_per_kg"].shift(-7)
            sub["target_retail_14d"] = sub["retail_price_per_kg"].shift(-14)

            # Multi-horizon future wholesale prices
            sub["target_modal_1d"] = sub["modal_price_per_kg"].shift(-1)
            sub["target_modal_3d"] = sub["modal_price_per_kg"].shift(-3)
            sub["target_modal_7d"] = sub["modal_price_per_kg"].shift(-7)
            sub["target_modal_14d"] = sub["modal_price_per_kg"].shift(-14)

            # Percentage changes across horizons
            sub["pct_change_ahead_3d"] = (sub["target_retail_3d"] - sub["retail_price_per_kg"]) / sub["retail_price_per_kg"] * 100.0
            sub["pct_change_ahead_7d"] = (sub["target_retail_7d"] - sub["retail_price_per_kg"]) / sub["retail_price_per_kg"] * 100.0
            sub["pct_change_ahead_14d"] = (sub["target_retail_14d"] - sub["retail_price_per_kg"]) / sub["retail_price_per_kg"] * 100.0

            # Absolute changes across horizons
            sub["abs_change_ahead_3d"] = sub["target_retail_3d"] - sub["retail_price_per_kg"]
            sub["abs_change_ahead_7d"] = sub["target_retail_7d"] - sub["retail_price_per_kg"]
            sub["abs_change_ahead_14d"] = sub["target_retail_14d"] - sub["retail_price_per_kg"]

            # Sudden Spike Flags (Surge >= +20% or >= +Rs.8/kg within 7 days, or >= +15% / +Rs.6 in 3 days)
            sub["is_spike_3d"] = ((sub["pct_change_ahead_3d"] >= 15.0) | (sub["abs_change_ahead_3d"] >= 6.0)).astype(int)
            sub["is_spike_7d"] = ((sub["pct_change_ahead_7d"] >= self.spike_threshold_pct) | (sub["abs_change_ahead_7d"] >= self.spike_threshold_abs)).astype(int)
            sub["is_spike_14d"] = ((sub["pct_change_ahead_14d"] >= 25.0) | (sub["abs_change_ahead_14d"] >= 10.0)).astype(int)

            # Sudden Crash Flags (Drop <= -15% or <= -Rs.6/kg)
            sub["is_crash_3d"] = ((sub["pct_change_ahead_3d"] <= -12.0) | (sub["abs_change_ahead_3d"] <= -5.0)).astype(int)
            sub["is_crash_7d"] = ((sub["pct_change_ahead_7d"] <= -15.0) | (sub["abs_change_ahead_7d"] <= -7.0)).astype(int)
            sub["is_crash_14d"] = ((sub["pct_change_ahead_14d"] <= -20.0) | (sub["abs_change_ahead_14d"] <= -9.0)).astype(int)

            # Forward 14-day rolling peak & trough
            # Reverse rolling max to look forward 14 days
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=14)
            sub["target_peak_price_14d"] = sub["retail_price_per_kg"].rolling(window=indexer, min_periods=1).max()
            sub["target_trough_price_14d"] = sub["retail_price_per_kg"].rolling(window=indexer, min_periods=1).min()
            sub["target_max_surge_14d"] = np.maximum(0.0, sub["target_peak_price_14d"] - sub["retail_price_per_kg"])

            # Shock Trigger Multipliers & Agrometeorology Acceleration
            if "kolar_rain_3d_sum" in sub.columns:
                sub["kolar_rain_surge_intensity"] = sub["kolar_rain_3d_sum"] / (sub["kolar_rain_7d_sum"].replace(0, 0.1) + 1.0)
            if "nashik_rain_7d_sum" in sub.columns:
                sub["nashik_rain_surge_intensity"] = sub["nashik_rain_3d_sum"] / (sub["nashik_rain_7d_sum"].replace(0, 0.1) + 1.0)

            dfs.append(sub)

        df_spike = pd.concat(dfs, axis=0).reset_index(drop=True)
        os.makedirs(os.path.dirname(SPIKE_DATASET_FILE), exist_ok=True)
        df_spike.to_csv(SPIKE_DATASET_FILE, index=False, encoding="utf-8")
        
        logger.info(f"Spike training dataset created ({len(df_spike):,} records, {df_spike.shape[1]} columns).")
        logger.info(f"Saved to: {SPIKE_DATASET_FILE}")
        return df_spike


def get_spike_dataset() -> pd.DataFrame:
    """Loads or creates the multi-horizon spike feature dataset."""
    if os.path.exists(SPIKE_DATASET_FILE):
        df = pd.read_csv(SPIKE_DATASET_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    engine = SpikeFeatureEngine()
    return engine.build_spike_dataset()


if __name__ == "__main__":
    engine = SpikeFeatureEngine()
    df = engine.build_spike_dataset()
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].dropna(subset=["target_retail_7d"])
        spk_7d = sub["is_spike_7d"].sum()
        crsh_7d = sub["is_crash_7d"].sum()
        print(f"[{comm}] Total: {len(sub)} samples | 7d Spikes: {spk_7d} ({spk_7d/len(sub)*100:.1f}%) | 7d Crashes: {crsh_7d} ({crsh_7d/len(sub)*100:.1f}%)")
