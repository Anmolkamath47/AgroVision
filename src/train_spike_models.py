"""
Model Training & Tuning Pipeline for Sudden Price Spike & Multi-Horizon Early Warning System.
Trains:
1. Sudden Spike Probability Classifier (XGBClassifier)
2. Sudden Crash Probability Classifier (XGBClassifier)
3. Multi-Horizon Future Price Path Regressors (t+3d, t+7d, t+14d)
4. Peak Magnitude Ceiling Regressor (t+14d maximum price)
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    brier_score_loss,
)
from sklearn.model_selection import TimeSeriesSplit

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TARGET_COMMODITIES, MODELS_DIR
from src.spike_engine import get_spike_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SPIKE_MODELS_DIR = os.path.join(MODELS_DIR, "spike")
os.makedirs(SPIKE_MODELS_DIR, exist_ok=True)


class SpikeModelTrainer:
    """Trains and tunes sudden spike classifiers, crash classifiers, and multi-horizon regressors."""

    def __init__(self, test_ratio: float = 0.15):
        self.test_ratio = test_ratio
        self.models_dir = SPIKE_MODELS_DIR

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Excludes metadata, identifiers, and forward-looking target columns."""
        target_prefixes = [
            "target_", "is_spike", "is_crash", "pct_change_ahead", "abs_change_ahead"
        ]
        non_features = [
            "date", "arrival_date", "commodity", "market", "district", "state",
            "data_source", "is_outlier_flag", "original_unit"
        ]
        feat_cols = []
        for c in df.columns:
            if c in non_features:
                continue
            if any(c.startswith(tp) for tp in target_prefixes):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                feat_cols.append(c)
        return feat_cols

    def train_spike_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        commodity: str,
        target_type: str = "spike",
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """Trains a calibrated probability classifier for sudden spike/crash detection."""
        # Calculate positive class weighting to handle class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = max(1.0, float(neg_count) / max(1.0, float(pos_count)))

        clf = xgb.XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        probs_test = clf.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= 0.50).astype(int)

        auc = roc_auc_score(y_test, probs_test) if len(np.unique(y_test)) > 1 else 0.5
        prec = precision_score(y_test, preds_test, zero_division=0)
        rec = recall_score(y_test, preds_test, zero_division=0)
        f1 = f1_score(y_test, preds_test, zero_division=0)
        brier = brier_score_loss(y_test, probs_test)

        metrics = {
            "roc_auc": round(float(auc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "brier_score": round(float(brier), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
            "scale_pos_weight": round(scale_pos_weight, 2),
        }

        # Save model
        filename = f"{commodity.lower()}_{target_type}_classifier.json"
        clf.save_model(os.path.join(self.models_dir, filename))
        return clf, metrics

    def train_horizon_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        commodity: str,
        horizon_name: str,
    ) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
        """Trains direct multi-horizon regressor (e.g. 3-day, 7-day, 14-day ahead, or peak ceiling)."""
        reg = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.80,
            random_state=42,
            eval_metric="mae",
        )
        reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = reg.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        smape = np.mean(200.0 * np.abs(preds - y_test) / (np.abs(preds) + np.abs(y_test) + 1e-6))

        metrics = {
            "mae": round(float(mae), 3),
            "rmse": round(float(rmse), 3),
            "smape": round(float(smape), 2),
        }

        filename = f"{commodity.lower()}_{horizon_name}_regressor.json"
        reg.save_model(os.path.join(self.models_dir, filename))
        return reg, metrics

    def train_all_commodities(self) -> Dict[str, Any]:
        """Trains spike classifiers and multi-horizon regressors for all commodities."""
        df = get_spike_dataset()
        feat_cols = self.get_feature_columns(df)
        all_metrics = {}

        # Save feature column list
        feat_path = os.path.join(self.models_dir, "spike_feature_columns.json")
        with open(feat_path, "w", encoding="utf-8") as f:
            json.dump(feat_cols, f, indent=2)

        for comm in TARGET_COMMODITIES:
            logger.info("=" * 72)
            logger.info(f"TRAINING SUDDEN SPIKE & MULTI-HORIZON MODELS: {comm.upper()}")
            logger.info("=" * 72)

            sub = df[df["commodity"] == comm].sort_values("date").reset_index(drop=True)
            valid_sub = sub.dropna(subset=["target_retail_14d"]).reset_index(drop=True)

            n_samples = len(valid_sub)
            n_train = int(n_samples * (1.0 - self.test_ratio))

            train_df = valid_sub.iloc[:n_train]
            test_df = valid_sub.iloc[n_train:]

            X_train = train_df[feat_cols].ffill(axis=1).fillna(0)
            X_test = test_df[feat_cols].ffill(axis=1).fillna(0)

            comm_metrics = {
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "features_count": len(feat_cols),
            }

            # 1. Train 7-Day Sudden Spike Classifier
            y_spike_train = train_df["is_spike_7d"]
            y_spike_test = test_df["is_spike_7d"]
            _, spike_m = self.train_spike_classifier(
                X_train, y_spike_train, X_test, y_spike_test, comm, target_type="spike"
            )
            comm_metrics["spike_classifier_7d"] = spike_m
            logger.info(f"[{comm}] 7-Day Spike Classifier: ROC-AUC={spike_m['roc_auc']}, Recall={spike_m['recall']*100:.1f}%, F1={spike_m['f1_score']}")

            # 2. Train 7-Day Sudden Crash Classifier
            y_crash_train = train_df["is_crash_7d"]
            y_crash_test = test_df["is_crash_7d"]
            _, crash_m = self.train_spike_classifier(
                X_train, y_crash_train, X_test, y_crash_test, comm, target_type="crash"
            )
            comm_metrics["crash_classifier_7d"] = crash_m
            logger.info(f"[{comm}] 7-Day Crash Classifier: ROC-AUC={crash_m['roc_auc']}, Recall={crash_m['recall']*100:.1f}%, F1={crash_m['f1_score']}")

            # 3. Train Multi-Horizon Price Regressors (3d, 7d, 14d)
            for h in ["3d", "7d", "14d"]:
                target_col = f"target_retail_{h}"
                _, reg_m = self.train_horizon_regressor(
                    X_train, train_df[target_col], X_test, test_df[target_col], comm, horizon_name=f"retail_{h}"
                )
                comm_metrics[f"regressor_{h}"] = reg_m
                logger.info(f"[{comm}] {h} Forward Regressor: MAE=Rs.{reg_m['mae']}/kg, sMAPE={reg_m['smape']}%")

            # 4. Train 14-Day Peak Magnitude Ceiling Regressor
            _, peak_m = self.train_horizon_regressor(
                X_train, train_df["target_peak_price_14d"], X_test, test_df["target_peak_price_14d"], comm, horizon_name="peak_14d"
            )
            comm_metrics["peak_regressor_14d"] = peak_m
            logger.info(f"[{comm}] 14-Day Peak Price Ceiling Regressor: MAE=Rs.{peak_m['mae']}/kg, sMAPE={peak_m['smape']}%")

            all_metrics[comm] = comm_metrics

        # Save all metrics
        metrics_file = os.path.join(self.models_dir, "spike_models_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)

        logger.info("=" * 72)
        logger.info(f"ALL SPIKE & MULTI-HORIZON MODELS TRAINED AND SAVED TO: {self.models_dir}")
        logger.info("=" * 72)
        return all_metrics


if __name__ == "__main__":
    trainer = SpikeModelTrainer()
    trainer.train_all_commodities()
