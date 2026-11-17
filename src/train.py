"""
Model Training Pipeline for Bangalore Grocery Price Prediction.
Implements:
- Chronological Time-Series Train/Validation/Test Split
- XGBoost Regression with early stopping
- Approach A (Global Model) vs Approach B (Per-Commodity Models)
- Controlled Feature Experiments
- Hyperparameter Tuning with TimeSeriesSplit
- Feature Importance Extraction
- Model and Artifact Storage
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_DATA_FILE,
    MODELS_DIR,
    TARGET_COMMODITIES,
)
from src.baseline import calculate_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

NON_FEATURE_COLS = [
    "date",
    "arrival_date",
    "commodity",
    "market",
    "district",
    "state",
    "target_next_day_price",
    "target_next_date",
    "target_change",
    "target_pct_change",
    "target_direction",
    "target_next_day_retail_price",
    "target_retail_change",
    "target_retail_pct_change",
    "target_retail_direction",
    "data_source",
    "original_unit",
    "original_modal_price",
    "original_min_price",
    "original_max_price",
    "is_outlier_flag",
]


def chronological_split(
    df_comm: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically splits data into Train (70%), Validation (15%), and Test (15%).
    Strictly preserves temporal order (no shuffling).
    """
    df_sorted = df_comm.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, feature_set: str = "full") -> List[str]:
    """
    Returns feature column lists for controlled experiments:
    - 'price_only': Lag & rolling price features only
    - 'price_calendar': Prices + calendar features
    - 'price_cross': Prices + calendar + cross-commodity features
    - 'full': Full Multi-Factor Economist Engine (Prices + Weather + Fuel + Govt Policies + Supply Cycles)
    """
    all_avail = [c for c in df.columns if c not in NON_FEATURE_COLS]

    if feature_set == "price_only":
        return [c for c in all_avail if ("price" in c or "rolling" in c or "spread" in c) and "diesel" not in c and "freight" not in c and "related" not in c and "sin" not in c and "cos" not in c and "day" not in c and "month" not in c and "week" not in c and "quarter" not in c and "weather" not in c and "govt" not in c and "policy" not in c and "temp" not in c and "rain" not in c and "precip" not in c and "shock" not in c and "cycle" not in c]
    elif feature_set == "price_calendar":
        return [c for c in all_avail if "related" not in c and "weather" not in c and "govt" not in c and "policy" not in c and "temp" not in c and "rain" not in c and "precip" not in c and "shock" not in c and "cycle" not in c and "freight" not in c and "diesel" not in c]
    elif feature_set == "price_cross":
        return [c for c in all_avail if "weather" not in c and "govt" not in c and "policy" not in c and "temp" not in c and "rain" not in c and "precip" not in c and "shock" not in c and "cycle" not in c and "freight" not in c and "diesel" not in c]
    else:  # full
        return all_avail


class GroceryPriceTrainer:
    """Trains and tunes XGBoost grocery price forecasting models."""

    def __init__(self, data_path: str = str(FEATURE_DATA_FILE)):
        self.data_path = data_path
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_columns: Dict[str, List[str]] = {}
        self.metrics_summary: Dict[str, Any] = {}
        self.experiments_summary: List[Dict[str, Any]] = []

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def train_commodity_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        commodity_name: str,
        feature_set: str = "full",
        target_col: str = "target_next_day_price",
        current_price_col: str = "modal_price_per_kg",
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[xgb.XGBRegressor, List[str], Dict[str, float]]:
        """
        Trains an XGBoost regression model with early stopping on validation set.
        """
        feat_cols = get_feature_columns(train_df, feature_set=feature_set)
        
        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feat_cols]
        y_val = val_df[target_col]

        default_params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "early_stopping_rounds": 30,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Validation evaluation
        y_val_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val.values, y_val_pred, val_df[current_price_col].values)

        return model, feat_cols, val_metrics

    def run_feature_experiments(self, df_full: pd.DataFrame):
        """
        Runs controlled experiments on Wholesale Mandi prices:
        Exp 1: Historical prices only
        Exp 2: Prices + Calendar
        Exp 3: Prices + Calendar + Cross-Commodity
        Exp 4: Multi-Factor Economist Model (Agrometeorology + Fuel + Govt Policies + Supply Cycles)
        """
        logger.info("=" * 68)
        logger.info("RUNNING CONTROLLED FEATURE EXPERIMENTS (Validation Set)")
        logger.info("=" * 68)

        experiment_sets = [
            ("Exp 1: Price Features Only", "price_only"),
            ("Exp 2: Price + Calendar Features", "price_calendar"),
            ("Exp 3: Price + Calendar + Cross-Commodity", "price_cross"),
            ("Exp 4: Multi-Factor Economist Model (Weather, Fuel, Policies)", "full"),
        ]

        for exp_name, feat_set in experiment_sets:
            for comm in TARGET_COMMODITIES:
                sub = df_full[df_full["commodity"] == comm].copy()
                train_df, val_df, _ = chronological_split(sub)
                
                _, _, val_metrics = self.train_commodity_model(
                    train_df, val_df, comm, feature_set=feat_set
                )
                self.experiments_summary.append({
                    "Experiment": exp_name,
                    "Commodity": comm,
                    "Features_Count": len(get_feature_columns(train_df, feat_set)),
                    **val_metrics,
                })
                logger.info(
                    f"{exp_name:62s} | {comm:7s} | MAE: Rs.{val_metrics['MAE']:.2f}/kg | "
                    f"RMSE: Rs.{val_metrics['RMSE']:.2f}/kg | sMAPE: {val_metrics['sMAPE']:.2f}% | "
                    f"DirAcc: {val_metrics['Directional_Accuracy']:.2f}%"
                )

    def tune_hyperparameters(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, commodity_name: str, target_col: str = "target_next_day_price"
    ) -> Dict[str, Any]:
        """
        Tunes hyperparameters using TimeSeriesSplit cross-validation on (train + val) data.
        """
        logger.info(f"Tuning hyperparameters for {commodity_name} ({target_col}) using TimeSeriesSplit...")
        combined_df = pd.concat([train_df, val_df], axis=0).sort_values("date").reset_index(drop=True)
        feat_cols = get_feature_columns(combined_df, "full")
        
        X = combined_df[feat_cols]
        y = combined_df[target_col]

        tscv = TimeSeriesSplit(n_splits=4)
        param_candidates = [
            {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.8, "min_child_weight": 3},
            {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 400, "subsample": 0.85, "min_child_weight": 5},
            {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.8, "min_child_weight": 3},
            {"max_depth": 6, "learning_rate": 0.015, "n_estimators": 600, "subsample": 0.75, "min_child_weight": 4},
        ]

        best_score = float("inf")
        best_params = param_candidates[0]

        for p in param_candidates:
            cv_maes = []
            for tr_idx, te_idx in tscv.split(X):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

                m = xgb.XGBRegressor(
                    **p,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                )
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                cv_maes.append(mean_absolute_error(y_te, preds))

            mean_cv_mae = float(np.mean(cv_maes))
            if mean_cv_mae < best_score:
                best_score = mean_cv_mae
                best_params = p

        logger.info(f"Best parameters for {commodity_name} ({target_col}): {best_params} (CV MAE: Rs.{best_score:.2f}/kg)")
        return best_params

    def train_and_save_all(self):
        """Full training workflow for both Wholesale Mandi and Consumer Retail models."""
        df = self.load_data()
        os.makedirs(MODELS_DIR, exist_ok=True)

        # 1. Run Controlled Feature Experiments
        self.run_feature_experiments(df)

        # 2. Train Best Per-Commodity Models for Wholesale & Consumer Retail
        logger.info("\n" + "=" * 68)
        logger.info("TRAINING DUAL-LAYER MODELS: WHOLESALE & CONSUMER RETAIL")
        logger.info("=" * 68)

        models_metadata = {}

        for comm in TARGET_COMMODITIES:
            sub = df[df["commodity"] == comm].copy()
            train_df, val_df, test_df = chronological_split(sub)

            # === A. Wholesale Mandi Model ===
            best_params_ws = self.tune_hyperparameters(train_df, val_df, comm, target_col="target_next_day_price")
            model_ws, feat_cols_ws, val_metrics_ws = self.train_commodity_model(
                train_df, val_df, comm, feature_set="full", target_col="target_next_day_price", current_price_col="modal_price_per_kg", params=best_params_ws
            )

            # Wholesale Test Evaluation
            X_test_ws = test_df[feat_cols_ws]
            y_test_ws = test_df["target_next_day_price"]
            y_test_pred_ws = model_ws.predict(X_test_ws)
            test_metrics_ws = calculate_metrics(y_test_ws.values, y_test_pred_ws, test_df["modal_price_per_kg"].values)

            # Save Wholesale Model
            model_file_ws = MODELS_DIR / f"{comm.lower()}_xgboost.json"
            model_file_ws_named = MODELS_DIR / f"{comm.lower()}_wholesale_xgboost.json"
            model_ws.save_model(str(model_file_ws))
            model_ws.save_model(str(model_file_ws_named))

            # === B. Consumer Retail Model (DCA Benchmark) ===
            has_retail = "target_next_day_retail_price" in train_df.columns
            if has_retail:
                best_params_ret = self.tune_hyperparameters(train_df, val_df, comm, target_col="target_next_day_retail_price")
                model_ret, feat_cols_ret, val_metrics_ret = self.train_commodity_model(
                    train_df, val_df, comm, feature_set="full", target_col="target_next_day_retail_price", current_price_col="retail_price_per_kg", params=best_params_ret
                )

                # Retail Test Evaluation
                X_test_ret = test_df[feat_cols_ret]
                y_test_ret = test_df["target_next_day_retail_price"]
                y_test_pred_ret = model_ret.predict(X_test_ret)
                test_metrics_ret = calculate_metrics(y_test_ret.values, y_test_pred_ret, test_df["retail_price_per_kg"].values)

                # Save Retail Model
                model_file_ret = MODELS_DIR / f"{comm.lower()}_retail_xgboost.json"
                model_ret.save_model(str(model_file_ret))
                logger.info(f"Saved {comm} retail model to: {model_file_ret}")
            else:
                test_metrics_ret = {}
                val_metrics_ret = {}

            # Feature Importance
            importances = model_ws.feature_importances_
            feat_imp = sorted(zip(feat_cols_ws, importances), key=lambda x: x[1], reverse=True)
            top_feats = {k: float(round(v, 4)) for k, v in feat_imp[:15]}

            models_metadata[comm] = {
                "wholesale_model_file": str(model_file_ws.name),
                "retail_model_file": f"{comm.lower()}_retail_xgboost.json" if has_retail else None,
                "best_parameters": best_params_ws,
                "feature_columns": feat_cols_ws,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "train_dates": f"{train_df['arrival_date'].min()} to {train_df['arrival_date'].max()}",
                "val_dates": f"{val_df['arrival_date'].min()} to {val_df['arrival_date'].max()}",
                "test_dates": f"{test_df['arrival_date'].min()} to {test_df['arrival_date'].max()}",
                "wholesale_validation_metrics": val_metrics_ws,
                "wholesale_test_metrics": test_metrics_ws,
                "retail_validation_metrics": val_metrics_ret,
                "retail_test_metrics": test_metrics_ret,
                "top_features": top_feats,
            }

            logger.info(
                f"[{comm.upper()} WHOLESALE TEST] MAE: Rs.{test_metrics_ws['MAE']:.2f}/kg | "
                f"RMSE: Rs.{test_metrics_ws['RMSE']:.2f}/kg | sMAPE: {test_metrics_ws['sMAPE']:.2f}% | "
                f"DirAcc: {test_metrics_ws['Directional_Accuracy']:.2f}%"
            )
            if has_retail:
                logger.info(
                    f"[{comm.upper()} RETAIL TEST]    MAE: Rs.{test_metrics_ret['MAE']:.2f}/kg | "
                    f"RMSE: Rs.{test_metrics_ret['RMSE']:.2f}/kg | sMAPE: {test_metrics_ret['sMAPE']:.2f}% | "
                    f"DirAcc: {test_metrics_ret['Directional_Accuracy']:.2f}%"
                )

        # 3. Save Master Metadata & Feature Columns
        with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump({c: models_metadata[c]["feature_columns"] for c in TARGET_COMMODITIES}, f, indent=2)

        with open(MODELS_DIR / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
            json.dump(models_metadata, f, indent=2)

        with open(MODELS_DIR / "feature_experiments.json", "w", encoding="utf-8") as f:
            json.dump(self.experiments_summary, f, indent=2)

        logger.info(f"\nAll models, metadata, and artifacts saved in {MODELS_DIR}")


def main():
    trainer = GroceryPriceTrainer()
    trainer.train_and_save_all()


if __name__ == "__main__":
    main()
