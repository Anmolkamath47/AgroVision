"""
Model Evaluation and Diagnostic Analysis Pipeline.
Compares:
- Persistence Baseline
- 7-Day Moving Average Baseline
- XGBoost Model
Calculates: MAE, RMSE, MAPE, sMAPE, Directional Accuracy
Performs: Residual Analysis, Systematic Bias Checks, Feature Importance Extraction
Generates: Evaluation Charts and metrics.json
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_DATA_FILE,
    MODELS_DIR,
    TARGET_COMMODITIES,
)
from src.baseline import calculate_metrics, evaluate_baselines_on_test_set
from src.train import chronological_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class ModelEvaluator:
    """Evaluates forecasting models against baselines and performs error diagnostics."""

    def __init__(
        self,
        feature_path: str = str(FEATURE_DATA_FILE),
        models_dir: str = str(MODELS_DIR),
    ):
        self.feature_path = feature_path
        self.models_dir = models_dir
        self.figures_dir = os.path.join(models_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self.evaluation_results: Dict[str, Any] = {}

    def load_feature_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.feature_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def evaluate_commodity(self, df_comm: pd.DataFrame, commodity: str) -> Dict[str, Any]:
        """Runs full benchmark and diagnostics for a single commodity (Wholesale and Retail)."""
        _, _, test_df = chronological_split(df_comm)

        # 1. Baseline Evaluations
        baselines = evaluate_baselines_on_test_set(test_df, commodity)

        with open(os.path.join(self.models_dir, "feature_columns.json"), "r", encoding="utf-8") as f:
            feat_cols = json.load(f)[commodity]

        # 2. Wholesale Model Evaluation
        model_file_ws = os.path.join(self.models_dir, f"{commodity.lower()}_xgboost.json")
        if not os.path.exists(model_file_ws):
            raise FileNotFoundError(f"Model file not found: {model_file_ws}")

        model_ws = xgb.XGBRegressor()
        model_ws.load_model(model_file_ws)

        X_test = test_df[feat_cols]
        y_true_ws = test_df["target_next_day_price"].values
        y_current_ws = test_df["modal_price_per_kg"].values
        y_pred_ws = model_ws.predict(X_test)

        xgb_ws_metrics = calculate_metrics(y_true_ws, y_pred_ws, y_current_ws)

        # Wholesale Residuals & Bias
        residuals_ws = y_true_ws - y_pred_ws
        mean_res_ws = float(np.mean(residuals_ws))
        std_res_ws = float(np.std(residuals_ws))
        abs_res_ws = np.abs(residuals_ws)
        p90_err_ws = float(np.percentile(abs_res_ws, 90))
        p95_err_ws = float(np.percentile(abs_res_ws, 95))
        max_err_ws = float(np.max(abs_res_ws))

        residual_ws_stats = {
            "mean_bias": round(mean_res_ws, 3),
            "std_error": round(std_res_ws, 3),
            "p90_error": round(p90_err_ws, 2),
            "p95_error": round(p95_err_ws, 2),
            "max_absolute_error": round(max_err_ws, 2),
            "underpredicted_days_count": int(np.sum(residuals_ws > 0.5)),
            "overpredicted_days_count": int(np.sum(residuals_ws < -0.5)),
            "accurate_within_50paise_count": int(np.sum(abs_res_ws <= 0.5)),
            "bias_interpretation": (
                "Slight Underprediction" if mean_res_ws > 0.1 else
                "Slight Overprediction" if mean_res_ws < -0.1 else "Unbiased"
            ),
        }

        # 3. Retail Model Evaluation (if retail model exists)
        model_file_ret = os.path.join(self.models_dir, f"{commodity.lower()}_retail_xgboost.json")
        has_retail = os.path.exists(model_file_ret) and "target_next_day_retail_price" in test_df.columns

        if has_retail:
            model_ret = xgb.XGBRegressor()
            model_ret.load_model(model_file_ret)

            y_true_ret = test_df["target_next_day_retail_price"].values
            y_current_ret = test_df["retail_price_per_kg"].values
            y_pred_ret = model_ret.predict(X_test)

            xgb_ret_metrics = calculate_metrics(y_true_ret, y_pred_ret, y_current_ret)

            residuals_ret = y_true_ret - y_pred_ret
            abs_res_ret = np.abs(residuals_ret)
            p90_err_ret = float(np.percentile(abs_res_ret, 90))
            p95_err_ret = float(np.percentile(abs_res_ret, 95))

            residual_ret_stats = {
                "mean_bias": round(float(np.mean(residuals_ret)), 3),
                "std_error": round(float(np.std(residuals_ret)), 3),
                "p90_error": round(p90_err_ret, 2),
                "p95_error": round(p95_err_ret, 2),
                "max_absolute_error": round(float(np.max(abs_res_ret)), 2),
            }
        else:
            xgb_ret_metrics = {}
            residual_ret_stats = {}
            p90_err_ret = 4.0

        # 4. Feature Importance (Wholesale)
        importances = model_ws.feature_importances_
        feat_imp = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)
        top_features = {k: round(float(v), 4) for k, v in feat_imp[:15]}

        # 5. Generate Diagnostic Plots
        self.plot_forecast_vs_actual(test_df["date"].values, y_true_ws, y_pred_ws, y_current_ws, commodity)
        self.plot_residuals(residuals_ws, y_pred_ws, commodity)
        self.plot_feature_importance(top_features, commodity)

        return {
            "test_samples": len(test_df),
            "test_date_range": f"{test_df['arrival_date'].min()} to {test_df['arrival_date'].max()}",
            "baselines": baselines,
            "xgboost": xgb_ws_metrics,
            "wholesale_metrics": xgb_ws_metrics,
            "retail_metrics": xgb_ret_metrics,
            "residual_diagnostics": residual_ws_stats,
            "retail_residual_diagnostics": residual_ret_stats,
            "top_features": top_features,
            "conformal_quantile_90": round(p90_err_ws, 2),
            "conformal_quantile_90_retail": round(p90_err_ret, 2),
        }

    def plot_forecast_vs_actual(
        self, dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_current: np.ndarray, commodity: str
    ):
        """Generates Actual vs Predicted price trajectory plot."""
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ax.plot(dates, y_true, label="Actual Next-Day Price", color="#1f77b4", lw=1.8, alpha=0.9)
        ax.plot(dates, y_pred, label="XGBoost Predicted Price", color="#d62728", lw=1.6, ls="--", alpha=0.9)
        ax.plot(dates, y_current, label="Persistence (Today's Price)", color="#7f7f7f", lw=1.0, ls=":", alpha=0.6)

        ax.set_title(f"Next-Day Price Forecasting: Actual vs Predicted ({commodity} - Bangalore)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price (Rs. / kg)", fontsize=10)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        out_path = os.path.join(self.figures_dir, f"{commodity.lower()}_actual_vs_predicted.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_residuals(self, residuals: np.ndarray, y_pred: np.ndarray, commodity: str):
        """Generates Residuals distribution and Residuals vs Predicted plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)

        # Residuals Histogram / KDE
        sns.histplot(residuals, kde=True, ax=ax1, color="#2ca02c", bins=25, edgecolor="white")
        ax1.axvline(0, color="red", ls="--", lw=1.2)
        ax1.set_title(f"Residual Error Distribution ({commodity})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Residual (Actual - Predicted) [Rs./kg]", fontsize=9)
        ax1.set_ylabel("Frequency", fontsize=9)

        # Residuals vs Predicted
        ax2.scatter(y_pred, residuals, alpha=0.6, color="#1f77b4", edgecolors="none", s=25)
        ax2.axhline(0, color="red", ls="--", lw=1.2)
        ax2.set_title(f"Residuals vs Predicted Price ({commodity})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Predicted Price (Rs./kg)", fontsize=9)
        ax2.set_ylabel("Residual (Rs./kg)", fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, f"{commodity.lower()}_residual_analysis.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    def plot_feature_importance(self, top_features: Dict[str, float], commodity: str):
        """Generates Horizontal Bar Chart of Feature Importances."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        feats = list(top_features.keys())[::-1]
        scores = list(top_features.values())[::-1]

        bars = ax.barh(feats, scores, color="#3498db", edgecolor="none", height=0.65)
        ax.set_title(f"Top 15 Feature Importances ({commodity} XGBoost Model)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Relative Importance Score", fontsize=10)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.003, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=8, color="#333333")

        plt.tight_layout()
        out_path = os.path.join(self.figures_dir, f"{commodity.lower()}_feature_importance.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    def run_all_evaluations(self):
        """Executes evaluation across all commodities and prints comparison tables."""
        df = self.load_feature_data()

        logger.info("=" * 72)
        logger.info("COMPREHENSIVE MODEL BENCHMARK & EVALUATION REPORT")
        logger.info("=" * 72)

        master_metrics = {}

        for comm in TARGET_COMMODITIES:
            sub = df[df["commodity"] == comm].copy()
            res = self.evaluate_commodity(sub, comm)
            master_metrics[comm] = res

            logger.info(f"\n==================== {comm.upper()} (Test Period: {res['test_date_range']}) ====================")
            
            p = res["baselines"]["Persistence"]
            m = res["baselines"]["7_Day_Moving_Average"]
            x = res["wholesale_metrics"]
            r = res.get("retail_metrics", {})

            # Wholesale Comparison Table
            comp_table = pd.DataFrame([
                {"Model": "Wholesale: Persistence", "MAE (Rs./kg)": p["MAE"], "RMSE (Rs./kg)": p["RMSE"], "sMAPE (%)": p["sMAPE"], "Dir. Acc. (%)": p["Directional_Accuracy"]},
                {"Model": "Wholesale: 7-Day MA", "MAE (Rs./kg)": m["MAE"], "RMSE (Rs./kg)": m["RMSE"], "sMAPE (%)": m["sMAPE"], "Dir. Acc. (%)": m["Directional_Accuracy"]},
                {"Model": "Wholesale: AI Model", "MAE (Rs./kg)": x["MAE"], "RMSE (Rs./kg)": x["RMSE"], "sMAPE (%)": x["sMAPE"], "Dir. Acc. (%)": x["Directional_Accuracy"]},
            ])
            if r:
                comp_table = pd.concat([
                    comp_table,
                    pd.DataFrame([{"Model": "Retail DCA: AI Model", "MAE (Rs./kg)": r.get("MAE", 0), "RMSE (Rs./kg)": r.get("RMSE", 0), "sMAPE (%)": r.get("sMAPE", 0), "Dir. Acc. (%)": r.get("Directional_Accuracy", 0)}])
                ], ignore_index=True)

            logger.info(f"\n{comp_table.to_string(index=False)}")

            # Residual & Bias
            diag = res["residual_diagnostics"]
            logger.info(f"\n--- Error & Bias Diagnostics (Wholesale) ---")
            logger.info(f"  * Systematic Bias     : {diag['bias_interpretation']} (Mean error: Rs.{diag['mean_bias']:+.3f}/kg)")
            logger.info(f"  * 90% Error Margin     : +/- Rs.{diag['p90_error']:.2f}/kg")
            logger.info(f"  * 95% Error Margin     : +/- Rs.{diag['p95_error']:.2f}/kg")
            logger.info(f"  * Max Error on Test    : Rs.{diag['max_absolute_error']:.2f}/kg")
            logger.info(f"  * Accurate (+/-50 paise): {diag['accurate_within_50paise_count']} of {res['test_samples']} days")

            # Top Features
            logger.info(f"\n--- Top 5 Predictive Features ---")
            for f_name, f_score in list(res["top_features"].items())[:5]:
                logger.info(f"  * {f_name:25s}: {f_score:.4f}")

        # Save Metrics JSON
        metrics_file = os.path.join(self.models_dir, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(master_metrics, f, indent=2)
        logger.info(f"\nSaved comprehensive metrics to: {metrics_file}")
        logger.info("=" * 72)


def main():
    evaluator = ModelEvaluator()
    evaluator.run_all_evaluations()


if __name__ == "__main__":
    main()
