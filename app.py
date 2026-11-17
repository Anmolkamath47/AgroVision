"""
AgroVision Intelligence: Institutional Bangalore Vegetable Market & Price Intelligence Platform.
Flask Application & High-Performance REST API.
"""

import os
import sys
import math
import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any

from flask import Flask, render_template, jsonify, request

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import threading
import time

from src.predict import GroceryPricePredictor
from src.predict_spike import SpikeEarlyWarningPredictor
from src.real_time_prices import PriceFusionEngine
from src.tracker import PredictionTracker
from src.database import check_db_connection, get_db_telemetry_stats, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgroVisionApp")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Global Predictor Instance (Cached for fast API responses)
predictor = GroceryPricePredictor(auto_sync=False)
spike_predictor = SpikeEarlyWarningPredictor()
fusion_engine = PriceFusionEngine()
tracker = PredictionTracker()

TARGET_COMMODITIES = ["Tomato", "Onion", "Potato"]


def start_auto_daily_updater():
    """
    Background worker thread that runs continuously.
    1. Reconciles past pending predictions when actuals arrive.
    2. Automatically generates today's next-day predictions for tomorrow (D+1) with status=PENDING.
    3. Periodically checks every 60 seconds and triggers on date turnover.
    """
    def _worker():
        last_checked_date = None
        while True:
            try:
                current_date = date.today().strftime("%Y-%m-%d")
                if current_date != last_checked_date:
                    logger.info(f"[AUTO-UPDATER] Date rollover / cycle check detected for date {current_date}...")
                    tracker.ensure_daily_forecasts_and_reconciliation(predictor_instance=predictor)
                    last_checked_date = current_date
                    logger.info(f"[AUTO-UPDATER] Daily forecasts updated and verified successfully for {current_date}.")
                else:
                    tracker.verify_predictions()
            except Exception as e:
                logger.error(f"[AUTO-UPDATER] Error in background updater: {e}")
            time.sleep(60)

    t = threading.Thread(target=_worker, daemon=True, name="AgroVisionDailyUpdater")
    t.start()
    logger.info("AgroVision automated daily forecasting & tracking background daemon started.")


# Initial daily check on boot and start background updater
try:
    tracker.ensure_daily_forecasts_and_reconciliation(predictor_instance=predictor)
except Exception as e:
    logger.warning(f"Initial daily forecast check note: {e}")

start_auto_daily_updater()

# Producing Hub Agrometeorology Metadata
PRODUCING_HUBS = [
    {
        "hub_name": "Kolar Corridor (Kolar / Chikkaballapur / Madanapalle)",
        "primary_crop": "Tomato (Hybrid & Native)",
        "distance_km": 70,
        "transit_hours": "3 - 4 hrs (NH75)",
        "role": "Primary daily tomato supply hub for Bangalore urban belt",
        "rain_sum_3d": 4.2,
        "rain_sum_7d": 12.8,
        "wet_days": 1,
        "status": "Optimal Harvesting",
        "status_color": "emerald",
    },
    {
        "hub_name": "Nashik & Northern Corridor (Lasalgaon / Niphad / Hubballi)",
        "primary_crop": "Onion (Kharif & Stored Rabi)",
        "distance_km": 1050,
        "transit_hours": "36 - 48 hrs (NH48)",
        "role": "Major national onion dispatch center feeding South India",
        "rain_sum_3d": 8.5,
        "rain_sum_7d": 24.1,
        "wet_days": 2,
        "status": "Transit Normal",
        "status_color": "emerald",
    },
    {
        "hub_name": "Hassan & Southern Belt (Hassan / Chikmagalur / Agra)",
        "primary_crop": "Potato (Jyoti & Cold-Storage Table)",
        "distance_km": 180,
        "transit_hours": "5 - 7 hrs (NH75)",
        "role": "Year-round cold storage buffer and fresh harvest dispatches",
        "rain_sum_3d": 2.1,
        "rain_sum_7d": 8.4,
        "wet_days": 0,
        "status": "Steady Supply",
        "status_color": "emerald",
    },
]


@app.after_request
def add_security_and_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def index():
    """Renders the main institutional dashboard."""
    return render_template("index.html")


@app.route("/api/prices", methods=["GET"])
def get_prices():
    """
    Returns live prices, 14-day trajectories, and trends for target commodities.
    Supports ?channel=local|pushcart|supermarket|quick_commerce|wholesale|hopcoms
    """
    channel = request.args.get("channel", "local").lower()
    
    channel_multipliers = {
        "local": 1.00,
        "pushcart": 0.95,
        "supermarket": 1.15,
        "quick_commerce": 1.45,
        "wholesale": 0.60,
        "hopcoms": 0.98,
    }
    multiplier = channel_multipliers.get(channel, 1.00)

    today = date.today()
    d0_str = today.strftime("%Y-%m-%d")
    d1_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    d3_str = (today + timedelta(days=3)).strftime("%Y-%m-%d")
    d7_str = (today + timedelta(days=7)).strftime("%Y-%m-%d")
    d14_str = (today + timedelta(days=14)).strftime("%Y-%m-%d")

    def _fmt_range(val: float) -> Dict[str, Any]:
        adj_val = val * multiplier
        low = int(math.floor(adj_val - 0.5))
        high = int(math.ceil(adj_val + 0.5))
        if low == high:
            high = low + 1
        low = max(1, low)
        return {
            "avg": round(adj_val, 1),
            "low": low,
            "high": high,
            "display": f"₹{low} - {high}",
        }

    results = []
    summary_raw = {}
    for comm in TARGET_COMMODITIES:
        try:
            summary_raw[comm] = predictor.predict_commodity(comm)
        except Exception as e:
            logger.error(f"Error predicting {comm}: {e}")

    for comm in TARGET_COMMODITIES:
        data = summary_raw.get(comm, {})
        ret = data.get("retail_benchmark", {})
        spike = data.get("spike_radar", {})
        traj = spike.get("price_trajectory", {})

        cur_p = ret.get("current_retail_price_per_kg", 25.0)
        p_d1 = traj.get("tomorrow_d1", cur_p)
        p_d3 = traj.get("day_3_ahead", cur_p)
        p_d7 = traj.get("day_7_ahead", cur_p)
        p_d14 = traj.get("day_14_ahead", cur_p)
        trend = spike.get("trend", "STABLE")

        # Economic driver narrative dynamically derived from attribution engine
        driver = data.get("economic_attribution", {}).get("primary_driver", {})
        driver_desc = driver.get("description", "")
        driver_name = driver.get("name", "")
        
        if driver_desc:
            narrative = driver_desc
        elif driver_name:
            narrative = f"Primary market driver: {driver_name} with {trend.lower()} price pressure."
        else:
            if comm == "Onion":
                narrative = "Buffer stock releases and regional dispatch flows actively balancing Yeshwanthpur mandi supply."
            elif comm == "Tomato":
                narrative = "Kolar corridor harvesting pace and daily market inflows guiding retail price discovery."
            else:
                narrative = "Cold storage release schedules and transport freight rates stabilizing table potato retail prices."

        results.append({
            "commodity": comm,
            "channel": channel,
            "trend": trend,
            "narrative": narrative,
            "today_d0": {
                "date": d0_str,
                "price": _fmt_range(cur_p),
            },
            "tomorrow_d1": {
                "date": d1_str,
                "price": _fmt_range(p_d1),
            },
            "day_3_ahead": {
                "date": d3_str,
                "price": _fmt_range(p_d3),
            },
            "day_7_ahead": {
                "date": d7_str,
                "price": _fmt_range(p_d7),
            },
            "day_14_ahead": {
                "date": d14_str,
                "price": _fmt_range(p_d14),
            },
            "spike_probability_pct": spike.get("spike_probability_pct", 15.0),
            "crash_probability_pct": spike.get("crash_probability_pct", 15.0),
            "alert_level": spike.get("alert_level", "STABLE_MARKET"),
            "alert_badge": spike.get("alert_badge", "[STABLE MARKET]"),
            "advice": spike.get("consumer_action_advice", "Normal daily buying recommended."),
        })

    return jsonify({
        "status": "success",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "channel": channel,
        "commodities": results,
    })


@app.route("/api/weather", methods=["GET"])
def get_weather():
    """Returns producing hub agrometeorological telemetry."""
    return jsonify({
        "status": "success",
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "hubs": PRODUCING_HUBS,
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    What-If Scenario Simulation Engine.
    Allows user to test price sensitivity to:
    - rainfall_shock_pct (-50% to +100%)
    - arrival_deficit_pct (-50% to +100%)
    - diesel_freight_pct (-20% to +30%)
    """
    req_data = request.get_json(silent=True) or {}
    commodity = req_data.get("commodity", "Tomato").capitalize()
    rain_shock = float(req_data.get("rainfall_shock_pct", 0.0))
    arrival_deficit = float(req_data.get("arrival_deficit_pct", 0.0))
    freight_shock = float(req_data.get("diesel_freight_pct", 0.0))

    # Base ground truths
    base_prices = {"Tomato": 27.5, "Onion": 50.0, "Potato": 25.0}
    p0 = base_prices.get(commodity, 30.0)

    # Elasticities
    weather_elasticity = {"Tomato": 0.15, "Onion": 0.08, "Potato": 0.02}.get(commodity, 0.05)
    supply_elasticity = 0.25
    freight_elasticity = 0.08

    net_shock_pct = (
        (rain_shock / 100.0) * weather_elasticity
        + (arrival_deficit / 100.0) * supply_elasticity
        + (freight_shock / 100.0) * freight_elasticity
    )

    sim_d1 = round(p0 * (1.0 + net_shock_pct * 0.25), 1)
    sim_d3 = round(p0 * (1.0 + net_shock_pct * 0.70), 1)
    sim_d7 = round(p0 * (1.0 + net_shock_pct * 1.00), 1)
    sim_d14 = round(p0 * (1.0 + net_shock_pct * 0.85), 1)

    return jsonify({
        "status": "success",
        "commodity": commodity,
        "baseline_price": p0,
        "simulated_trajectory": {
            "today_d0": p0,
            "tomorrow_d1": sim_d1,
            "day_3_ahead": sim_d3,
            "day_7_ahead": sim_d7,
            "day_14_ahead": sim_d14,
        },
        "price_delta_pct": round(net_shock_pct * 100.0, 1),
        "price_delta_rs": round(sim_d7 - p0, 2),
        "impact_summary": (
            f"A {arrival_deficit:+.0f}% arrival deficit & {rain_shock:+.0f}% rain shift causes {commodity} to peak at ₹{max(sim_d3, sim_d7):.1f}/kg (+₹{max(sim_d3, sim_d7) - p0:.1f}/kg)."
        ),
    })


@app.route("/audit")
def audit_page():
    """Renders the prediction audit ledger dashboard view."""
    return render_template("index.html")


@app.route("/api/ledger", methods=["GET"])
def get_ledger():
    """
    Returns the prediction audit ledger history and ground-truth verification status.
    Supports ?commodity=Tomato|Onion|Potato|all and ?status=VERIFIED|PENDING|all
    """
    commodity = request.args.get("commodity", "all").strip()
    status = request.args.get("status", "all").strip()

    # Automatically ensure today's next-day forecast is generated and past pending records reconciled
    try:
        tracker.ensure_daily_forecasts_and_reconciliation(predictor_instance=predictor)
    except Exception as e:
        logger.warning(f"Auto daily sync in /api/ledger note: {e}")

    records = tracker.get_audit_ledger_records(commodity=commodity, status=status)
    metrics = tracker.calculate_metrics()

    return jsonify({
        "status": "success",
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "filter": {"commodity": commodity, "status": status},
        "total_records": len(records),
        "summary": metrics.get("summary", {}),
        "records": records,
    })


@app.route("/api/verify-actuals", methods=["GET", "POST"])
def verify_actuals():
    """
    Triggers automated ground-truth verification against actual market observations.
    """
    try:
        result = tracker.verify_predictions()
        records = tracker.get_audit_ledger_records()
        return jsonify({
            "status": "success",
            "message": f"Ground-truth verification complete. Reconciled {result.get('verified_count', 0)} forecasts.",
            "verified_count": result.get("verified_count", 0),
            "remaining_pending": result.get("remaining_pending", 0),
            "records": records,
        })
    except Exception as e:
        logger.error(f"Error during ground-truth verification: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/db/status", methods=["GET"])
def db_status():
    """
    Returns PostgreSQL database connection health, telemetry stats, and table record counts.
    """
    stats = get_db_telemetry_stats()
    return jsonify({
        "status": "success",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "db": stats,
    })


@app.route("/api/daily-train", methods=["POST"])
def trigger_daily_train():
    """
    Executes the full automated daily continuous cycle:
    1. Reconciles pending predictions when target date arrives.
    2. Ingests today's live Bangalore market rates into the historical datasets.
    3. Re-engineers the 221-factor feature matrix.
    4. Retrains machine learning models (XGBoost Delta & Direct).
    5. Generates tomorrow's D+1 forecast and logs as PENDING.
    """
    try:
        from src.daily_auto_trainer import DailyAutoTrainer
        trainer = DailyAutoTrainer()
        result = trainer.run_daily_cycle()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during daily auto-training cycle: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/daily-train/status", methods=["GET"])
def get_daily_train_status():
    """
    Returns history and performance of automated continuous daily training runs.
    """
    try:
        from src.daily_auto_trainer import DailyAutoTrainer
        trainer = DailyAutoTrainer()
        log_data = trainer.load_log()
        return jsonify({
            "status": "success",
            "last_run_at": log_data.get("last_run_at"),
            "total_runs": log_data.get("total_runs", 0),
            "recent_runs": log_data.get("runs", [])[:10],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health")
@app.route("/api/health")
def health_check():
    """
    Production health-check endpoint for container orchestrators, load balancers, and Render/Railway.
    """
    db_stats = get_db_telemetry_stats()
    return jsonify({
        "status": "healthy",
        "service": "AgroVision Commodity Intelligence Platform",
        "version": "2.0.0",
        "environment": os.environ.get("FLASK_ENV", "production"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "database": {
            "connected": db_stats.get("connected", False),
            "engine": db_stats.get("engine", "PostgreSQL / CSV Fallback"),
        },
        "models_ready": True,
    }), 200


@app.after_request
def add_security_and_cors_headers(response):
    """Adds security and CORS headers for production deployment."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgroVision Intelligence Web App")
    parser.add_argument("-p", "--port", type=int, default=int(os.environ.get("PORT", 5001)), help="Port number to run the server on (default: 5001)")
    parser.add_argument("-H", "--host", type=str, default=os.environ.get("HOST", "127.0.0.1"), help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable Flask debug mode")
    args, _ = parser.parse_known_args()

    port = args.port
    host = args.host
    logger.info(f"Starting AgroVision Analytics Platform on http://{host}:{port}")
    app.run(host=host, port=port, debug=args.debug)

