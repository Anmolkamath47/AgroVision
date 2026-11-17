#!/usr/bin/env python
"""
AgroVision Database Migration Utility: Flat-File / CSV -> PostgreSQL.
Initializes schema and migrates historical prices, economic indicators, and prediction ledgers.

Usage:
    python scripts/migrate_to_postgres.py --all
    python scripts/migrate_to_postgres.py --init-schema
    python scripts/migrate_to_postgres.py --ledger
    python scripts/migrate_to_postgres.py --processed
    python scripts/migrate_to_postgres.py --economic
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import text

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    PREDICTION_LEDGER_FILE,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    DATA_DIR,
)
from src.database import (
    get_engine,
    init_db,
    check_db_connection,
    PredictionLedgerModel,
    ProcessedPriceModel,
    EconomicFactorModel,
    RawMandiArrivalModel,
    get_db_session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("AgroVisionMigration")


def migrate_prediction_ledger(force: bool = False) -> int:
    """Migrates prediction_ledger.csv into PostgreSQL prediction_ledger table."""
    if not os.path.exists(PREDICTION_LEDGER_FILE):
        logger.warning(f"Ledger file not found at {PREDICTION_LEDGER_FILE}")
        return 0

    df = pd.read_csv(PREDICTION_LEDGER_FILE)
    if df.empty:
        logger.info("Ledger CSV is empty, nothing to migrate.")
        return 0

    logger.info(f"Ingesting {len(df)} rows from prediction_ledger.csv...")
    migrated_count = 0

    with get_db_session() as session:
        for _, row in df.iterrows():
            pred_id = str(row.get("prediction_id", "")).strip()
            if not pred_id:
                continue

            existing = session.query(PredictionLedgerModel).filter_by(prediction_id=pred_id).first()
            if existing and not force:
                continue

            record = {
                "prediction_id": pred_id,
                "date_predicted": pd.to_datetime(row.get("date_predicted")).date() if pd.notna(row.get("date_predicted")) else None,
                "target_date": pd.to_datetime(row.get("target_date")).date() if pd.notna(row.get("target_date")) else None,
                "commodity": str(row.get("commodity", "")),
                "market": str(row.get("market", "Bangalore")),
                "current_wholesale_price": float(row["current_wholesale_price"]) if pd.notna(row.get("current_wholesale_price")) else None,
                "predicted_wholesale_price": float(row["predicted_wholesale_price"]) if pd.notna(row.get("predicted_wholesale_price")) else 0.0,
                "predicted_wholesale_delta": float(row["predicted_wholesale_delta"]) if pd.notna(row.get("predicted_wholesale_delta")) else None,
                "actual_wholesale_price": float(row["actual_wholesale_price"]) if pd.notna(row.get("actual_wholesale_price")) else None,
                "wholesale_error_rs": float(row["wholesale_error_rs"]) if pd.notna(row.get("wholesale_error_rs")) else None,
                "wholesale_pct_error": float(row["wholesale_pct_error"]) if pd.notna(row.get("wholesale_pct_error")) else None,
                "current_retail_price": float(row["current_retail_price"]) if pd.notna(row.get("current_retail_price")) else None,
                "predicted_retail_price": float(row["predicted_retail_price"]) if pd.notna(row.get("predicted_retail_price")) else 0.0,
                "predicted_retail_delta": float(row["predicted_retail_delta"]) if pd.notna(row.get("predicted_retail_delta")) else None,
                "actual_retail_price": float(row["actual_retail_price"]) if pd.notna(row.get("actual_retail_price")) else None,
                "actual_retail_delta": float(row["actual_retail_delta"]) if pd.notna(row.get("actual_retail_delta")) else None,
                "retail_error_rs": float(row["retail_error_rs"]) if pd.notna(row.get("retail_error_rs")) else None,
                "retail_pct_error": float(row["retail_pct_error"]) if pd.notna(row.get("retail_pct_error")) else None,
                "predicted_trend": str(row.get("predicted_trend", "")) if pd.notna(row.get("predicted_trend")) else None,
                "actual_trend": str(row.get("actual_trend", "")) if pd.notna(row.get("actual_trend")) else None,
                "spike_risk_pct": float(row.get("spike_risk_pct", 0.0)) if pd.notna(row.get("spike_risk_pct")) else 0.0,
                "primary_driver": str(row.get("primary_driver", "")) if pd.notna(row.get("primary_driver")) else None,
                "is_direction_correct": bool(row["is_direction_correct"]) if pd.notna(row.get("is_direction_correct")) else None,
                "is_within_5pct": bool(row["is_within_5pct"]) if pd.notna(row.get("is_within_5pct")) else None,
                "is_within_10pct": bool(row["is_within_10pct"]) if pd.notna(row.get("is_within_10pct")) else None,
                "is_within_2rs": bool(row["is_within_2rs"]) if pd.notna(row.get("is_within_2rs")) else None,
                "status": str(row.get("status", "PENDING")).upper(),
            }

            if existing and force:
                for k, v in record.items():
                    setattr(existing, k, v)
            else:
                session.add(PredictionLedgerModel(**record))
            migrated_count += 1

    logger.info(f"Successfully migrated {migrated_count} prediction ledger records.")
    return migrated_count


def migrate_processed_prices(batch_size: int = 1000) -> int:
    """Migrates bangalore_grocery_prices_processed.csv into PostgreSQL processed_prices."""
    if not os.path.exists(PROCESSED_DATA_FILE):
        logger.warning(f"Processed price file not found at {PROCESSED_DATA_FILE}")
        return 0

    df = pd.read_csv(PROCESSED_DATA_FILE)
    if df.empty:
        return 0

    logger.info(f"Ingesting {len(df)} rows from processed prices dataset...")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    # Rename or map columns
    col_map = {
        "wholesale_modal_price": "wholesale_price",
        "retail_modal_price": "retail_price",
        "modal_price": "wholesale_price",
        "modal_price_per_kg": "wholesale_price",
        "retail_price_per_kg": "retail_price",
        "mandi_arrival_tonnes": "arrivals_tonnes",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    migrated_count = 0
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE processed_prices RESTART IDENTITY CASCADE;"))
        conn.commit()

    # Fast bulk insertion using pandas to_sql with if_exists='append'
    # Select target columns
    db_cols = ["date", "commodity", "market"]
    if "wholesale_price" in df.columns:
        db_cols.append("wholesale_price")
    elif "modal_price" in df.columns:
        df["wholesale_price"] = df["modal_price"]
        db_cols.append("wholesale_price")
    if "retail_price" in df.columns:
        db_cols.append("retail_price")
    if "arrivals_tonnes" in df.columns:
        db_cols.append("arrivals_tonnes")

    subset = df[db_cols].drop_duplicates(subset=["date", "commodity", "market"]).copy()

    try:
        subset.to_sql("processed_prices", con=engine, if_exists="append", index=False, chunksize=batch_size)
        migrated_count = len(subset)
        logger.info(f"Successfully migrated {migrated_count} processed price records.")
    except Exception as e:
        logger.error(f"Bulk insert failed (possible duplicates): {e}")
        # Insert record-by-record fallback
        with get_db_session() as session:
            for _, row in subset.iterrows():
                try:
                    p_obj = ProcessedPriceModel(
                        date=row["date"],
                        commodity=row["commodity"],
                        market=row.get("market", "Bangalore"),
                        wholesale_price=float(row.get("wholesale_price", 0.0)),
                        retail_price=float(row["retail_price"]) if pd.notna(row.get("retail_price")) else None,
                        arrivals_tonnes=float(row["arrivals_tonnes"]) if pd.notna(row.get("arrivals_tonnes")) else None,
                    )
                    session.merge(p_obj)
                    migrated_count += 1
                except Exception:
                    pass

    return migrated_count


def migrate_economic_factors() -> int:
    """Migrates bangalore_economic_factors.csv into PostgreSQL economic_factors."""
    factor_path = DATA_DIR / "processed" / "bangalore_economic_factors.csv"
    if not os.path.exists(factor_path):
        logger.warning(f"Economic factors file not found at {factor_path}")
        return 0

    df = pd.read_csv(factor_path)
    if df.empty:
        return 0

    logger.info(f"Ingesting {len(df)} rows from economic factors dataset...")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    migrated_count = 0

    with get_db_session() as session:
        for _, row in df.iterrows():
            d = row["date"]
            existing = session.query(EconomicFactorModel).filter_by(date=d).first()
            if existing:
                continue

            rec = EconomicFactorModel(
                date=d,
                kolar_rain_1d=float(row.get("kolar_rain_1d", 0.0)) if pd.notna(row.get("kolar_rain_1d")) else 0.0,
                kolar_rain_3d=float(row.get("kolar_rain_3d", 0.0)) if pd.notna(row.get("kolar_rain_3d")) else 0.0,
                kolar_rain_7d=float(row.get("kolar_rain_7d", 0.0)) if pd.notna(row.get("kolar_rain_7d")) else 0.0,
                nashik_rain_1d=float(row.get("nashik_rain_1d", 0.0)) if pd.notna(row.get("nashik_rain_1d")) else 0.0,
                nashik_rain_3d=float(row.get("nashik_rain_3d", 0.0)) if pd.notna(row.get("nashik_rain_3d")) else 0.0,
                nashik_rain_7d=float(row.get("nashik_rain_7d", 0.0)) if pd.notna(row.get("nashik_rain_7d")) else 0.0,
                karnataka_diesel_price=float(row.get("karnataka_diesel_price")) if pd.notna(row.get("karnataka_diesel_price")) else None,
                freight_cost_index=float(row.get("freight_cost_index")) if pd.notna(row.get("freight_cost_index")) else None,
                buffer_stock_release_flag=bool(row.get("buffer_stock_release_flag", False)),
            )
            session.add(rec)
            migrated_count += 1

    logger.info(f"Successfully migrated {migrated_count} economic factor records.")
    return migrated_count


def main():
    parser = argparse.ArgumentParser(description="AgroVision CSV to PostgreSQL Migration Utility")
    parser.add_argument("--all", action="store_true", help="Initialize schema and migrate all datasets")
    parser.add_argument("--init-schema", action="store_true", help="Create database tables")
    parser.add_argument("--ledger", action="store_true", help="Migrate prediction ledger")
    parser.add_argument("--processed", action="store_true", help="Migrate processed prices")
    parser.add_argument("--economic", action="store_true", help="Migrate economic factors")
    parser.add_argument("--force", action="store_true", help="Overwrite existing records if present")

    args = parser.parse_args()

    print("=" * 70)
    print(" AgroVision PostgreSQL Migration Utility")
    print("=" * 70)

    db_status = check_db_connection()
    if not db_status["connected"]:
        print("\n[ERROR] Could not connect to PostgreSQL Database:")
        print(f"  Host:     {db_status['host']}:{db_status['port']}")
        print(f"  Database: {db_status['database']}")
        print(f"  Error:    {db_status.get('error')}")
        print("\nPlease ensure PostgreSQL is running and credentials are configured in .env")
        sys.exit(1)

    print(f"\n[OK] Connected to PostgreSQL at {db_status['host']}:{db_status['port']} (Database: {db_status['database']})")

    # If no flags provided, default to --all
    if not (args.all or args.init_schema or args.ledger or args.processed or args.economic):
        args.all = True

    if args.init_schema or args.all:
        print("\n[1/4] Initializing Schema / Creating Tables...")
        success = init_db()
        if not success:
            print("Failed to initialize schema.")
            sys.exit(1)
        print("Schema initialized successfully.")

    if args.ledger or args.all:
        print("\n[2/4] Migrating Prediction Ledger...")
        c = migrate_prediction_ledger(force=args.force)
        print(f"Ledger migration completed: {c} records.")

    if args.processed or args.all:
        print("\n[3/4] Migrating Processed Prices...")
        c = migrate_processed_prices()
        print(f"Processed prices migration completed: {c} records.")

    if args.economic or args.all:
        print("\n[4/4] Migrating Economic Factors...")
        c = migrate_economic_factors()
        print(f"Economic factors migration completed: {c} records.")

    print("\n" + "=" * 70)
    print(" Migration Complete! PostgreSQL database is synchronized.")
    print("=" * 70)


if __name__ == "__main__":
    main()
