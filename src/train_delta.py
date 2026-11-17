"""
Residual / Delta Forecasting Pipeline.
Trains XGBoost to predict the price change (Delta = Next_Day_Price - Current_Price)
and reconstructs: Predicted_Next_Day_Price = Current_Price + Predicted_Delta.
This provides structural robustness against price regime shifts.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_DATA_FILE,
    MODELS_DIR,
    TARGET_COMMODITIES,
)
from src.baseline import calculate_metrics
from src.train import chronological_split, get_feature_columns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train_and_eval_delta_models():
    """Trains Wholesale and Retail Delta-predicting XGBoost models and evaluates on test set."""
    df = pd.read_csv(FEATURE_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])

    logger.info("=" * 68)
    logger.info("TRAINING DELTA (RESIDUAL) FORECASTING MODELS: WHOLESALE & RETAIL")
    logger.info("=" * 68)

    delta_results = {}

    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].copy()
        train_df, val_df, test_df = chronological_split(sub)

        feat_cols = get_feature_columns(train_df, "full")
        
        # 1. Wholesale Delta Model
        y_train_ws_delta = train_df["target_next_day_price"] - train_df["modal_price_per_kg"]
        y_val_ws_delta = val_df["target_next_day_price"] - val_df["modal_price_per_kg"]
        y_test_ws_true = test_df["target_next_day_price"].values
        y_test_ws_current = test_df["modal_price_per_kg"].values

        model_ws = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=42,
            early_stopping_rounds=30,
            n_jobs=-1,
        )

        model_ws.fit(
            train_df[feat_cols],
            y_train_ws_delta,
            eval_set=[(train_df[feat_cols], y_train_ws_delta), (val_df[feat_cols], y_val_ws_delta)],
            verbose=False,
        )

        pred_ws_delta = model_ws.predict(test_df[feat_cols])
        y_pred_ws_price = np.maximum(y_test_ws_current + pred_ws_delta, 0.5)
        metrics_ws = calculate_metrics(y_test_ws_true, y_pred_ws_price, y_test_ws_current)

        # Save Wholesale Delta models
        delta_file_ws = MODELS_DIR / f"{comm.lower()}_delta_xgboost.json"
        delta_file_ws_named = MODELS_DIR / f"{comm.lower()}_wholesale_delta_xgboost.json"
        model_ws.save_model(str(delta_file_ws))
        model_ws.save_model(str(delta_file_ws_named))

        # 2. Retail Delta Model (if retail targets present)
        has_retail = "target_next_day_retail_price" in train_df.columns
        if has_retail:
            y_train_ret_delta = train_df["target_next_day_retail_price"] - train_df["retail_price_per_kg"]
            y_val_ret_delta = val_df["target_next_day_retail_price"] - val_df["retail_price_per_kg"]
            y_test_ret_true = test_df["target_next_day_retail_price"].values
            y_test_ret_current = test_df["retail_price_per_kg"].values

            model_ret = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                early_stopping_rounds=30,
                n_jobs=-1,
            )

            model_ret.fit(
                train_df[feat_cols],
                y_train_ret_delta,
                eval_set=[(train_df[feat_cols], y_train_ret_delta), (val_df[feat_cols], y_val_ret_delta)],
                verbose=False,
            )

            pred_ret_delta = model_ret.predict(test_df[feat_cols])
            y_pred_ret_price = np.maximum(y_test_ret_current + pred_ret_delta, 1.0)
            metrics_ret = calculate_metrics(y_test_ret_true, y_pred_ret_price, y_test_ret_current)

            delta_file_ret = MODELS_DIR / f"{comm.lower()}_retail_delta_xgboost.json"
            model_ret.save_model(str(delta_file_ret))
        else:
            metrics_ret = {}

        delta_results[comm] = {
            "wholesale": metrics_ws,
            "retail": metrics_ret,
        }

        logger.info(
            f"[{comm.upper()} WHOLESALE DELTA] MAE: Rs.{metrics_ws['MAE']:.2f}/kg | "
            f"RMSE: Rs.{metrics_ws['RMSE']:.2f}/kg | sMAPE: {metrics_ws['sMAPE']:.2f}% | "
            f"DirAcc: {metrics_ws['Directional_Accuracy']:.2f}%"
        )
        if has_retail:
            logger.info(
                f"[{comm.upper()} RETAIL DELTA]    MAE: Rs.{metrics_ret['MAE']:.2f}/kg | "
                f"RMSE: Rs.{metrics_ret['RMSE']:.2f}/kg | sMAPE: {metrics_ret['sMAPE']:.2f}% | "
                f"DirAcc: {metrics_ret['Directional_Accuracy']:.2f}%"
            )

    with open(MODELS_DIR / "delta_models_metrics.json", "w", encoding="utf-8") as f:
        json.dump(delta_results, f, indent=2)

    logger.info("=" * 68)
    return delta_results


if __name__ == "__main__":
    train_and_eval_delta_models()
