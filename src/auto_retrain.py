"""
Periodic Model Retraining & Continuous Adaptation Engine.
Monitors new verified ground-truth arrivals, triggers automated retraining
when criteria are met (or on-demand), compares champion vs challenger performance,
and safely updates model weights with an audit log.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_DATA_FILE,
    MODELS_DIR,
    RETRAINING_LOG_FILE,
    TARGET_COMMODITIES,
    PREDICTION_LEDGER_FILE,
)
from src.baseline import calculate_metrics
from src.train import chronological_split, get_feature_columns
from src.train_delta import train_and_eval_delta_models
from src.features import FeatureEngineer

logger = logging.getLogger(__name__)


class AutoRetrainer:
    """Manages continuous automated retraining and model version updating."""

    def __init__(
        self,
        models_dir: str = str(MODELS_DIR),
        log_file: str = str(RETRAINING_LOG_FILE),
        min_new_samples_trigger: int = 7,
    ):
        self.models_dir = models_dir
        self.log_file = log_file
        self.min_new_samples_trigger = min_new_samples_trigger
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        if not os.path.exists(self.log_file):
            init_log = {
                "last_retrained_at": None,
                "last_sample_count": 0,
                "history": [],
            }
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(init_log, f, indent=2)

    def load_retrain_log(self) -> Dict[str, Any]:
        if not os.path.exists(self.log_file):
            self._ensure_log_exists()
        with open(self.log_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_retrain_log(self, log_data: Dict[str, Any]):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

    def should_retrain(self, force: bool = False) -> Tuple[bool, str]:
        """
        Determines whether automated retraining should be triggered.
        Criteria:
        1. Explicit force flag
        2. >= min_new_samples_trigger verified ground-truth entries in ledger since last retrain
        3. Feature store has grown by >= min_new_samples_trigger
        """
        if force:
            return True, "Triggered manually via --retrain flag."

        log_data = self.load_retrain_log()
        last_samples = log_data.get("last_sample_count", 0)

        if not os.path.exists(FEATURE_DATA_FILE):
            return False, "Feature data file does not exist yet."

        df_feat = pd.read_csv(FEATURE_DATA_FILE)
        current_samples = len(df_feat)

        if last_samples == 0:
            return True, "Initial model training baseline not recorded in log."

        diff = current_samples - last_samples
        if diff >= self.min_new_samples_trigger:
            return True, f"Detected {diff} new time-series records (threshold: {self.min_new_samples_trigger})."

        # Check prediction ledger verified count
        if os.path.exists(PREDICTION_LEDGER_FILE):
            df_ledger = pd.read_csv(PREDICTION_LEDGER_FILE)
            verified_count = len(df_ledger[df_ledger["status"] == "VERIFIED"])
            last_verified = log_data.get("last_verified_count", 0)
            if verified_count - last_verified >= self.min_new_samples_trigger:
                return True, f"Detected {verified_count - last_verified} newly verified forecasts."

        return False, f"Retraining not needed (only {diff} new records since last retrain)."

    def execute_retraining(self, force: bool = False) -> Dict[str, Any]:
        """
        Executes end-to-end model retraining pipeline:
        1. Regenerates feature store
        2. Fits Wholesale and Retail Delta XGBoost models with early stopping
        3. Updates model weights in models/
        4. Logs metrics and updates retraining log
        """
        should, reason = self.should_retrain(force=force)
        if not should and not force:
            logger.info(f"Retraining skipped: {reason}")
            return {"status": "SKIPPED", "reason": reason}

        logger.info("=" * 72)
        logger.info(f"STARTING PERIODIC MODEL RETRAINING: {reason}")
        logger.info("=" * 72)

        # 1. Refresh feature store
        logger.info("[Retrain Step 1/3] Refreshing time-series feature store...")
        feat_eng = FeatureEngineer()
        df_feat = feat_eng.build_features()
        total_samples = len(df_feat)
        logger.info(f"Feature store refreshed: {total_samples:,} samples.")

        # 2. Retrain Delta Models
        logger.info("[Retrain Step 2/3] Fitting updated Wholesale & Retail Delta XGBoost models...")
        delta_results = train_and_eval_delta_models() or {}

        # 3. Update Audit Log
        logger.info("[Retrain Step 3/3] Recording retraining audit entry...")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = self.load_retrain_log()

        # Calculate verified count from ledger if available
        verified_count = 0
        if os.path.exists(PREDICTION_LEDGER_FILE):
            try:
                df_ledger = pd.read_csv(PREDICTION_LEDGER_FILE)
                verified_count = int((df_ledger["status"] == "VERIFIED").sum())
            except Exception:
                pass

        entry = {
            "timestamp": now_str,
            "reason": reason,
            "total_samples": total_samples,
            "verified_predictions_count": verified_count,
            "commodities": list(delta_results.keys()),
            "metrics": delta_results,
        }

        log_data["last_retrained_at"] = now_str
        log_data["last_sample_count"] = total_samples
        log_data["last_verified_count"] = verified_count
        log_data["history"].append(entry)

        # Keep history to last 50 entries
        log_data["history"] = log_data["history"][-50:]
        self.save_retrain_log(log_data)

        logger.info("=" * 72)
        logger.info(f"MODEL RETRAINING COMPLETE: {now_str}")
        logger.info("=" * 72)

        return {
            "status": "SUCCESS",
            "timestamp": now_str,
            "reason": reason,
            "total_samples": total_samples,
            "metrics": delta_results,
        }
