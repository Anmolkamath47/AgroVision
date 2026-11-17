"""
Baseline Forecasting Models for Bangalore Grocery Price Prediction.
Implements:
1. Persistence Baseline: Tomorrow's price = Today's price
2. 7-Day Moving Average Baseline: Tomorrow's price = Mean of past 7 days
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_DATA_FILE,
    TARGET_COMMODITIES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_current: np.ndarray) -> Dict[str, float]:
    """
    Computes standard time-series forecasting metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - sMAPE (Symmetric Mean Absolute Percentage Error)
    - Directional Accuracy (% correct prediction of price increase/decrease/stable)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_current = np.asarray(y_current)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # MAPE (avoid div by zero)
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100.0)
    
    # sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-6
    smape = float(np.mean(np.abs(y_pred - y_true) / denominator) * 100.0)

    # Directional Accuracy: Did predicted change match actual direction from today's price?
    actual_dir = np.sign(y_true - y_current)
    pred_dir = np.sign(y_pred - y_current)
    directional_acc = float(np.mean(actual_dir == pred_dir) * 100.0)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
        "Directional_Accuracy": round(directional_acc, 2),
    }


def evaluate_baselines_on_test_set(
    test_df: pd.DataFrame, commodity_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates Persistence and 7-Day Moving Average baselines on the test set.
    """
    sub = test_df[test_df["commodity"] == commodity_name].copy()
    if sub.empty:
        logger.warning(f"No test samples for {commodity_name}")
        return {}

    y_true = sub["target_next_day_price"].values
    y_current = sub["modal_price_per_kg"].values
    y_persistence = y_current
    y_ma7 = sub["rolling_mean_7"].fillna(sub["modal_price_per_kg"]).values

    persistence_metrics = calculate_metrics(y_true, y_persistence, y_current)
    ma7_metrics = calculate_metrics(y_true, y_ma7, y_current)

    return {
        "Persistence": persistence_metrics,
        "7_Day_Moving_Average": ma7_metrics,
    }


def main():
    logger.info("Loading feature dataset for Baseline evaluation...")
    df = pd.read_csv(FEATURE_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    logger.info("=" * 68)
    logger.info("BASELINE MODELS EVALUATION REPORT (Full Dataset & Test Set)")
    logger.info("=" * 68)

    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].sort_values("date").reset_index(drop=True)
        # Chronological Split (Last 20% as Test)
        split_idx = int(len(sub) * 0.8)
        test_sub = sub.iloc[split_idx:]

        results = evaluate_baselines_on_test_set(test_sub, comm)
        logger.info(f"\nCommodity: {comm.upper()} (Test Set: {len(test_sub)} days | {test_sub['date'].min().strftime('%Y-%m-%d')} to {test_sub['date'].max().strftime('%Y-%m-%d')})")
        
        p = results["Persistence"]
        logger.info(f"  [Baseline 1 - Persistence]:")
        logger.info(f"    * MAE                 : Rs.{p['MAE']:.2f}/kg")
        logger.info(f"    * RMSE                : Rs.{p['RMSE']:.2f}/kg")
        logger.info(f"    * sMAPE               : {p['sMAPE']:.2f}%")
        logger.info(f"    * Directional Accuracy: {p['Directional_Accuracy']:.2f}%")

        m = results["7_Day_Moving_Average"]
        logger.info(f"  [Baseline 2 - 7-Day Moving Average]:")
        logger.info(f"    * MAE                 : Rs.{m['MAE']:.2f}/kg")
        logger.info(f"    * RMSE                : Rs.{m['RMSE']:.2f}/kg")
        logger.info(f"    * sMAPE               : {m['sMAPE']:.2f}%")
        logger.info(f"    * Directional Accuracy: {m['Directional_Accuracy']:.2f}%")

    logger.info("=" * 68)


if __name__ == "__main__":
    main()
