"""
Continuous Forecasting, Tracking, Backtesting & Retraining Orchestrator.
Executes the automated end-to-end daily MLOps cycle:
1. Ingests today's live actual Mandi & weather data
2. Verifies yesterday's pending predictions against actuals
3. Computes rolling MAE, RMSE, MAPE, Directional Accuracy & Tolerance Hit Rates
4. Generates tomorrow's new price forecasts and logs to permanent ledger
5. Checks automated periodic model retraining triggers
6. Updates interactive HTML dashboard and high-resolution charts
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PREDICTION_LEDGER_FILE,
    DASHBOARD_HTML_FILE,
    TARGET_COMMODITIES,
)
from src.tracker import PredictionTracker
from src.auto_retrain import AutoRetrainer
from src.dashboard import DashboardGenerator
from src.daily_update import sync_live_mandi_data
from src.predict import GroceryPricePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_continuous_cycle(
    sync_live: bool = True,
    force_retrain: bool = False,
    generate_dashboard: bool = True,
) -> Dict[str, Any]:
    """
    Executes a single complete daily continuous forecasting and tracking cycle.
    """
    cycle_start = datetime.now()
    today_str = cycle_start.strftime("%Y-%m-%d")
    tomorrow_str = (cycle_start + timedelta(days=1)).strftime("%Y-%m-%d")

    print("\n" + "=" * 80)
    print(f"   AI AGRICULTURAL ECONOMIST: CONTINUOUS FORECASTING & TRACKING CYCLE")
    print(f"               DATE: {today_str}  |  TARGET: {tomorrow_str}")
    print("=" * 80)

    tracker = PredictionTracker()
    retrainer = AutoRetrainer()
    dash_gen = DashboardGenerator()

    # Step 1: Sync Live Data
    if sync_live:
        logger.info("[Step 1/5] Syncing live daily APMC Mandi prices & producing belt weather...")
        try:
            sync_live_mandi_data(timeout=15, refresh_weather=True)
        except Exception as e:
            logger.warning(f"Live data sync note: {e}")
    else:
        logger.info("[Step 1/5] Skipping live network sync (--offline mode).")

    # Step 2: Verify Pending Predictions
    logger.info("[Step 2/5] Verifying pending past predictions against ground-truth actuals...")
    ver_res = tracker.verify_predictions()
    logger.info(f"Ground-Truth Verification: {ver_res['verified_count']} verified, {ver_res['remaining_pending']} pending.")

    # Step 3: Check & Execute Periodic Retraining
    logger.info("[Step 3/5] Checking automated periodic model retraining triggers...")
    retrain_res = retrainer.execute_retraining(force=force_retrain)
    if retrain_res.get("status") == "SUCCESS":
        logger.info(f"Automated retraining executed: {retrain_res['reason']}")
    else:
        logger.info(f"Retraining status: {retrain_res.get('reason', 'Skipped')}")

    # Step 4: Generate Tomorrow's Forecast & Log to Permanent Ledger
    logger.info("[Step 4/5] Generating tomorrow's price forecasts and updating ledger...")
    predictor = GroceryPricePredictor(auto_sync=False)
    tomorrow_forecasts = {}

    for comm in TARGET_COMMODITIES:
        try:
            res = predictor.predict_commodity(comm)
            tomorrow_forecasts[comm] = res
            ws = res["wholesale"]
            ret = res["retail_benchmark"]
            spike = res.get("spike_radar", {})
            p_driver = res["economic_attribution"]["primary_driver"]["name"]

            tracker.log_prediction(
                date_predicted=today_str,
                target_date=tomorrow_str,
                commodity=comm,
                market="Bangalore",
                current_wholesale=ws["current_modal_price_per_kg"],
                predicted_wholesale=ws["predicted_next_day_price_per_kg"],
                current_retail=ret["current_retail_price_per_kg"],
                predicted_retail=ret["predicted_next_day_price_per_kg"],
                predicted_trend=ret["trend"],
                spike_risk_pct=spike.get("spike_probability_pct", 0.0),
                primary_driver=p_driver,
            )
        except Exception as e:
            logger.error(f"Error predicting {comm}: {e}")

    # Step 5: Generate Visualizations & Interactive Dashboard
    if generate_dashboard:
        logger.info("[Step 5/5] Generating interactive HTML dashboard and performance plots...")
        try:
            dash_gen.generate_all()
        except Exception as e:
            logger.warning(f"Dashboard generation note: {e}")

    # Step 6: Print Consolidated Scorecard & Summary
    metrics = tracker.calculate_metrics()
    print_cycle_summary(today_str, tomorrow_str, tomorrow_forecasts, metrics, ver_res)

    return {
        "today": today_str,
        "target_date": tomorrow_str,
        "verification": ver_res,
        "retraining": retrain_res,
        "metrics": metrics,
        "dashboard_path": str(DASHBOARD_HTML_FILE),
    }


def print_cycle_summary(
    today_str: str,
    tomorrow_str: str,
    tomorrow_forecasts: Dict[str, Any],
    metrics: Dict[str, Any],
    ver_res: Dict[str, Any],
):
    """Prints a clean, informative executive scorecard to console."""
    print("\n" + "=" * 80)
    print("                CONTINUOUS TRACKING SCORECARD & ACCURACY REPORT")
    print("=" * 80)

    ret_m = metrics.get("overall_retail", {})
    sum_m = metrics.get("summary", {})

    print(f"  • Total Forecasts Tracked   : {sum_m.get('total_predictions', 0):,} (Verified: {sum_m.get('total_verified', 0):,} | Pending: {sum_m.get('pending_verifications', 0):,})")
    print(f"  • Overall Directional Acc.  : {ret_m.get('directional_accuracy_pct', 0.0):.1f}%  (Trend Up/Down Hit Rate)")
    print(f"  • Retail Mean Abs. Error    : Rs. {ret_m.get('mae_rs_per_kg', 0.0):.2f} / kg  (MAPE: {ret_m.get('mape_pct', 0.0):.1f}%)")
    print(f"  • High-Precision Hit Rate   : {ret_m.get('hit_rate_within_5pct', 0.0):.1f}% (Within ±5%) | {ret_m.get('hit_rate_within_10pct', 0.0):.1f}% (Within ±10%)")
    print("-" * 80)
    print(f"  PER-COMMODITY PERFORMANCE SCORECARD:")
    print(f"  {'Commodity':<10s} | {'Verified':<10s} | {'MAE (₹/kg)':<12s} | {'MAPE (%)':<10s} | {'Directional Acc':<16s} | {'±5% Hit Rate':<14s}")
    print("  " + "-" * 76)
    for comm, comm_m in metrics.get("by_commodity", {}).items():
        print(f"  {comm:<10s} | {comm_m['count']:<10d} | Rs.{comm_m['mae_retail_rs']:<8.2f} | {comm_m['mape_retail_pct']:<8.1f}% | {comm_m['directional_accuracy_pct']:<14.1f}% | {comm_m['hit_rate_5pct']:<12.1f}%")
    print("-" * 80)

    if tomorrow_forecasts:
        predictor = GroceryPricePredictor(auto_sync=False)
        predictor.print_all_summary_table(tomorrow_forecasts)

        # Save latest forecast text and markdown reports
        report_txt_file = os.path.join(os.path.dirname(DASHBOARD_HTML_FILE), "latest_daily_forecast.txt")
        report_md_file = os.path.join(os.path.dirname(DASHBOARD_HTML_FILE), "latest_daily_forecast.md")
        
        # Build clean report content
        report_lines = [
            predictor.format_summary_table(tomorrow_forecasts),
            "",
            "================================================================================",
            "                  BANGALORE DAILY COMMODITY PRICE PREDICTION",
            "================================================================================",
            f"  Generated On: {gen_now}  |  Forecast Target: {tomorrow_str}",
            "--------------------------------------------------------------------------------",
            "",
        ]

        for comm, res in tomorrow_forecasts.items():
            emoji = {"Tomato": "🍅", "Onion": "🧅", "Potato": "🥔"}.get(comm, "🥬")
            ret = res["retail_benchmark"]
            spike = res.get("spike_radar", {})
            econ = res.get("economic_attribution", {})
            today_p = ret["current_retail_price_per_kg"]
            pred_p = ret["predicted_next_day_price_per_kg"]

            p_spike = spike.get("spike_probability_pct", 0.0)
            p_crash = spike.get("crash_probability_pct", 0.0)
            peak_p = spike.get("expected_peak_ceiling_price", pred_p)
            max_surge = spike.get("expected_max_surge_pct", 0.0)
            traj = spike.get("price_trajectory", {
                "today_d0": today_p,
                "tomorrow_d1": pred_p,
                "day_3_ahead": pred_p,
                "day_7_ahead": pred_p,
                "day_14_ahead": pred_p,
            })
            min_path = min(traj.get("day_3_ahead", pred_p), traj.get("day_7_ahead", pred_p), traj.get("day_14_ahead", pred_p))

            if p_spike >= 50.0 and max_surge >= 25.0:
                status_str = f"[CRITICAL SPIKE WARNING] (Spike Peak: ₹{peak_p:.2f} / kg, +{max_surge:.1f}% from current)"
            elif p_crash >= 45.0 or min_path <= today_p - 4.0:
                status_str = f"[PRICE DROP EXPECTED] (Easing to ₹{min_path:.2f} / kg over 14 days)"
            elif p_spike >= 35.0:
                status_str = f"[ELEVATED SPIKE RISK] (Trending towards ₹{traj.get('day_7_ahead', pred_p):.2f} / kg)"
            else:
                status_str = "[STABLE MARKET]"

            causes = []
            for t in spike.get("shock_triggers", []):
                if t.get("type") in ["SURGE_TRIGGER", "COOLING_PRESSURE"]:
                    causes.append(t["message"])

            for p_key in ["govt_policy", "harvest_seasonality", "fuel_logistics", "weather_agri", "market_momentum"]:
                if p_key in econ.get("factors", {}):
                    for d in econ["factors"][p_key].get("details", []):
                        if not any(d.lower()[:18] in c.lower() for c in causes):
                            causes.append(d)

            if not causes:
                causes.append(f"Normal weather across {comm} producing belts and steady cold-storage dispatches from producing corridors.")

            adv = spike.get("consumer_action_advice", econ.get("procurement_advisory", {}).get("consumer", "Normal daily buying recommended."))

            report_lines.append(f"{emoji} {comm}")
            report_lines.append("")
            report_lines.append(f"• Today's Price: ₹{today_p:.2f} / kg → Tomorrow's Price: ₹{pred_p:.2f} / kg")
            report_lines.append(f"• Early Warning Status: {status_str}")
            report_lines.append(f"• 14-Day Forward Trajectory:")
            report_lines.append(f"  • Today (D0): ₹{traj['today_d0']:.2f} / kg")
            report_lines.append(f"  • Tomorrow (D+1): ₹{traj['tomorrow_d1']:.2f} / kg")
            report_lines.append(f"  • Day 3 (D+3): ₹{traj['day_3_ahead']:.2f} / kg")
            report_lines.append(f"  • Day 7 (D+7): ₹{traj['day_7_ahead']:.2f} / kg")
            report_lines.append(f"  • Day 14 (D+14): ₹{traj['day_14_ahead']:.2f} / kg")

            if len(causes) == 1:
                report_lines.append(f"• Identified Cause: {causes[0]}")
            else:
                report_lines.append(f"• Identified Causes:")
                for c in causes[:3]:
                    report_lines.append(f"  • {c}")

            report_lines.append(f"• Shopping Advice: {adv}")
            report_lines.append("")

        report_txt = "\n".join(report_lines)
        with open(report_txt_file, "w", encoding="utf-8") as f:
            f.write(report_txt)
        with open(report_md_file, "w", encoding="utf-8") as f:
            f.write(f"```\n{report_txt}\n```")

    print(f"\n  [OK] Interactive Dashboard : file:///{str(DASHBOARD_HTML_FILE).replace(os.sep, '/')}")
    print(f"  [OK] Daily Forecast File   : file:///{os.path.join(os.path.dirname(DASHBOARD_HTML_FILE), 'latest_daily_forecast.txt').replace(os.sep, '/')}")
    print("=" * 80 + "\n")


def print_status_only():
    """Prints the current tracking scorecard from the ledger without running a cycle."""
    tracker = PredictionTracker()
    metrics = tracker.calculate_metrics()
    print_cycle_summary(
        today_str=datetime.now().strftime("%Y-%m-%d"),
        tomorrow_str=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        tomorrow_forecasts={},
        metrics=metrics,
        ver_res={"verified_count": 0, "remaining_pending": metrics["summary"]["pending_verifications"]},
    )


def run_daemon(interval_hours: float = 24.0, sync_live: bool = True):
    """
    Runs the continuous pipeline continuously as an automated background daily service.
    """
    logger.info(f"Starting Continuous Forecasting Daemon (Interval: {interval_hours} hours)...")
    try:
        while True:
            logger.info("Executing scheduled continuous forecasting cycle...")
            run_continuous_cycle(sync_live=sync_live)
            logger.info(f"Cycle finished. Sleeping for {interval_hours} hours until next run...")
            time.sleep(interval_hours * 3600)
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="AI Agricultural Economist: Continuous Forecasting, Tracking & Retraining System")
    parser.add_argument("--once", action="store_true", help="Run 1 complete daily forecasting, verification, and dashboard cycle")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous background daemon with periodic interval")
    parser.add_argument("--interval-hours", type=float, default=24.0, help="Interval in hours for daemon mode (default: 24.0)")
    parser.add_argument("--backtest", action="store_true", help="Seed ledger with historical walk-forward backtesting predictions")
    parser.add_argument("--days", type=int, default=60, help="Number of historical days for backtest seeding (default: 60)")
    parser.add_argument("--retrain", action="store_true", help="Force immediate model retraining on all verified data")
    parser.add_argument("--dashboard", action="store_true", help="Generate/refresh interactive HTML dashboard and figures")
    parser.add_argument("--status", action="store_true", help="Print current tracking scorecard and accuracy metrics")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode without fetching live external data")
    args = parser.parse_args()

    if args.status:
        print_status_only()
    elif args.backtest:
        tracker = PredictionTracker()
        tracker.seed_historical_backtest(days=args.days)
        dash_gen = DashboardGenerator()
        dash_gen.generate_all()
        print_status_only()
    elif args.retrain:
        retrainer = AutoRetrainer()
        retrainer.execute_retraining(force=True)
        dash_gen = DashboardGenerator()
        dash_gen.generate_all()
    elif args.dashboard:
        dash_gen = DashboardGenerator()
        dash_gen.generate_all()
        print(f"\n[OK] Dashboard generated at: file:///{str(DASHBOARD_HTML_FILE).replace(os.sep, '/')}\n")
    elif args.daemon:
        run_daemon(interval_hours=args.interval_hours, sync_live=not args.offline)
    else:
        # Default action: run once
        run_continuous_cycle(sync_live=not args.offline)


if __name__ == "__main__":
    main()
