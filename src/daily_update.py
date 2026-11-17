"""
Daily Multi-Factor Live Ingestion and Automated Dataset Synchronization Pipeline.
Synchronizes:
1. Real-time AGMARKNET / data.gov.in APMC Mandi price records
2. Live agrometeorology & precipitation feeds for key producing hubs (Kolar, Nashik, Hassan, Madanapalle, Bangalore)
3. Diesel fuel prices & transit logistics indices
4. Government buffer releases, export curbs, and market interventions
5. Automated feature store regeneration & daily forecast generation
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    FEATURE_DATA_FILE,
    TARGET_COMMODITIES,
    TARGET_DISTRICT_KEYWORDS,
)
from src.collect_data import DataGovCollector, build_unified_raw_dataset
from src.clean_data import DataCleaner
from src.economic_factors import EconomicFactorEngine, get_economic_feature_store
from src.features import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sync_live_mandi_data(timeout: int = 15, refresh_weather: bool = True) -> Dict[str, Any]:
    """
    Executes full multi-factor daily synchronization:
    - Queries data.gov.in AGMARKNET live feed for today's market arrivals in Karnataka.
    - Synchronizes live agrometeorological observations for Kolar, Nashik, Hassan, Madanapalle, Bangalore.
    - Updates fuel logistics and government policy indicators.
    - Cleans datasets and regenerates the full feature store.
    """
    logger.info("=" * 72)
    logger.info("STARTING DAILY MULTI-FACTOR MARKET & AGRI SYNCHRONIZATION")
    logger.info("=" * 72)

    # 1. Sync Live Mandi Prices from data.gov.in
    logger.info("[Step 1/3] Querying data.gov.in Live AGMARKNET Mandi Price API...")
    collector = DataGovCollector()
    live_records = []
    try:
        live_records = collector.fetch_current_karnataka_records()
    except Exception as e:
        logger.warning(f"Unable to fetch live records from data.gov.in: {e}")

    new_added = 0
    if live_records:
        df_live = build_unified_raw_dataset({}, live_records)
        if not df_live.empty:
            logger.info(f"Extracted {len(df_live)} live Bangalore market record(s) from data.gov.in.")
            
            if os.path.exists(RAW_DATA_FILE):
                df_raw = pd.read_csv(RAW_DATA_FILE)
            else:
                df_raw = pd.DataFrame()

            initial_count = len(df_raw)
            df_combined = pd.concat([df_raw, df_live], ignore_index=True)
            df_combined["parsed_date"] = pd.to_datetime(df_combined["arrival_date"], errors="coerce")
            df_combined = df_combined.drop_duplicates(subset=["parsed_date", "commodity", "market"], keep="last")
            df_combined = df_combined.sort_values(["commodity", "parsed_date"]).drop(columns=["parsed_date"])

            new_added = len(df_combined) - initial_count
            df_combined.to_csv(RAW_DATA_FILE, index=False, encoding="utf-8")
            logger.info(f"Raw dataset updated: {len(df_combined):,} total records ({new_added} new entries).")

            # Clean and preprocess
            cleaner = DataCleaner(raw_path=str(RAW_DATA_FILE), processed_path=str(PROCESSED_DATA_FILE))
            df_proc = cleaner.clean_and_preprocess(df_combined)
            cleaner.save_processed_data(df_proc)
        else:
            logger.info("No matching Bangalore commodity records in today's live feed yet.")
    else:
        logger.info("Live data.gov.in feed check completed.")

    # 2. Sync Agrometeorology & Regional Weather for Producing Hubs
    logger.info("[Step 2/3] Synchronizing Source-Region Weather & Agrometeorology (Kolar, Nashik, Hassan)...")
    try:
        econ_engine = EconomicFactorEngine(force_refresh_weather=refresh_weather)
        df_econ = econ_engine.build_economic_feature_store()
        logger.info(f"Source-region weather and logistics store synchronized ({len(df_econ):,} days).")
    except Exception as e:
        logger.warning(f"Note on weather store sync: {e}")
        df_econ = get_economic_feature_store()

    # 3. Regenerate Feature Store
    logger.info("[Step 3/3] Regenerating Multi-Factor Time-Series Feature Store...")
    try:
        feat_eng = FeatureEngineer()
        df_feat = feat_eng.build_features()
        logger.info(f"Feature store refreshed ({len(df_feat):,} samples).")
    except Exception as e:
        logger.error(f"Error regenerating feature store: {e}")

    # Build market and economic snapshot
    df_proc = pd.read_csv(PROCESSED_DATA_FILE)
    latest_snapshot = {}
    for comm in TARGET_COMMODITIES:
        sub = df_proc[df_proc["commodity"] == comm]
        if not sub.empty:
            latest_snapshot[comm] = {
                "latest_date": sub["arrival_date"].max(),
                "latest_modal_price_per_kg": float(sub.iloc[-1]["modal_price_per_kg"]),
                "latest_retail_price_per_kg": float(sub.iloc[-1]["retail_price_per_kg"]),
            }

    # Latest weather and economic metrics
    latest_econ_row = df_econ.iloc[-1].to_dict() if not df_econ.empty else {}
    date_val = str(latest_econ_row.get("date", ""))[:10]

    result = {
        "status": "success",
        "sync_timestamp": datetime.now().isoformat(),
        "new_mandi_records": new_added,
        "total_processed_records": len(df_proc),
        "latest_market_prices": latest_snapshot,
        "source_region_agri_status": {
            "date": date_val,
            "kolar_tomato_belt": {
                "rain_3d_mm": float(round(float(latest_econ_row.get("kolar_rain_3d_sum", 0.0)), 1)),
                "temp_max_c": float(round(float(latest_econ_row.get("kolar_temp_max", 0.0)), 1)),
                "stress_index": float(round(float(latest_econ_row.get("tomato_weather_stress_index", 0.0)), 2)),
            },
            "nashik_onion_belt": {
                "rain_7d_mm": float(round(float(latest_econ_row.get("nashik_rain_7d_sum", 0.0)), 1)),
                "temp_max_c": float(round(float(latest_econ_row.get("nashik_temp_max", 0.0)), 1)),
                "stress_index": float(round(float(latest_econ_row.get("onion_weather_stress_index", 0.0)), 2)),
            },
            "hassan_potato_belt": {
                "rain_7d_mm": float(round(float(latest_econ_row.get("hassan_rain_7d_sum", 0.0)), 1)),
                "temp_min_c": float(round(float(latest_econ_row.get("hassan_temp_min", 0.0)), 1)),
                "stress_index": float(round(float(latest_econ_row.get("potato_weather_stress_index", 0.0)), 2)),
            },
            "diesel_price_inr_per_l": float(round(float(latest_econ_row.get("diesel_price_per_litre", 88.5)), 2)),
        },
    }

    logger.info("=" * 72)
    logger.info("DAILY MULTI-FACTOR SYNCHRONIZATION COMPLETE")
    logger.info("=" * 72)
    return result


def record_daily_observation(
    commodity: str,
    modal_price_per_kg: float,
    date_str: Optional[str] = None,
    min_price_per_kg: Optional[float] = None,
    max_price_per_kg: Optional[float] = None,
    market: str = "Bangalore",
) -> Dict[str, Any]:
    """
    Manually inserts or updates a daily market observation for days when government APIs
    are delayed (e.g. Sundays / Mandi strike days).
    """
    comm_clean = commodity.strip().capitalize()
    if comm_clean not in TARGET_COMMODITIES:
        raise ValueError(f"Commodity '{commodity}' not supported. Supported: {TARGET_COMMODITIES}")

    obs_date = date_str or date.today().strftime("%Y-%m-%d")
    modal_quintal = modal_price_per_kg * 100.0
    min_quintal = (min_price_per_kg or modal_price_per_kg * 0.9) * 100.0
    max_quintal = (max_price_per_kg or modal_price_per_kg * 1.1) * 100.0

    new_row = pd.DataFrame([{
        "arrival_date": obs_date,
        "state": "Karnataka",
        "district": "Bangalore",
        "market": market,
        "commodity": comm_clean,
        "variety": "Local/Standard",
        "grade": "FAQ",
        "min_price": min_quintal,
        "max_price": max_quintal,
        "modal_price": modal_quintal,
        "data_source": "MANUAL_DAILY_FEED",
    }])

    if os.path.exists(RAW_DATA_FILE):
        df_raw = pd.read_csv(RAW_DATA_FILE)
    else:
        df_raw = pd.DataFrame()

    df_combined = pd.concat([df_raw, new_row], ignore_index=True)
    df_combined["parsed_date"] = pd.to_datetime(df_combined["arrival_date"], errors="coerce")
    df_combined = df_combined.drop_duplicates(subset=["parsed_date", "commodity", "market"], keep="last")
    df_combined = df_combined.sort_values(["commodity", "parsed_date"]).drop(columns=["parsed_date"])
    df_combined.to_csv(RAW_DATA_FILE, index=False, encoding="utf-8")

    cleaner = DataCleaner(raw_path=str(RAW_DATA_FILE), processed_path=str(PROCESSED_DATA_FILE))
    df_proc = cleaner.clean_and_preprocess(df_combined)
    cleaner.save_processed_data(df_proc)

    # Refresh feature store
    feat_eng = FeatureEngineer()
    feat_eng.build_features()

    logger.info(f"Recorded daily observation for {comm_clean} on {obs_date}: ₹{modal_price_per_kg:.2f}/kg.")
    return {
        "status": "success",
        "commodity": comm_clean,
        "date": obs_date,
        "modal_price_per_kg": modal_price_per_kg,
    }


def main():
    parser = argparse.ArgumentParser(description="Synchronize Live Daily Bangalore Mandi & Agrometeorological Data")
    parser.add_argument("--sync", action="store_true", help="Fetch and merge today's live Mandi feed and regional weather")
    parser.add_argument("--commodity", type=str, default=None, help="Commodity name for manual observation")
    parser.add_argument("--price", type=float, default=None, help="Current Mandi price in Rs./kg for manual observation")
    parser.add_argument("--date", type=str, default=None, help="Observation date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()

    if args.sync or (not args.commodity and not args.price):
        res = sync_live_mandi_data()
        print(json.dumps(res, indent=2))
    elif args.commodity and args.price is not None:
        res = record_daily_observation(args.commodity, args.price, date_str=args.date)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
