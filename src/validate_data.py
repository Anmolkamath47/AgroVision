"""
Data Validation Module for Bangalore Grocery Price Prediction.
Validates data integrity, completeness, and anomalies in raw market data.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_DATA_FILE,
    TARGET_COMMODITIES,
    TARGET_STATE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "arrival_date",
    "state",
    "district",
    "market",
    "commodity",
    "min_price",
    "max_price",
    "modal_price",
]


class DataValidator:
    """Validates raw grocery price dataset against quality standards."""

    def __init__(self, data_path: str = str(RAW_DATA_FILE)):
        self.data_path = data_path
        self.report: Dict[str, Any] = {}

    def run_validation(self) -> Dict[str, Any]:
        """Runs full validation pipeline on the raw dataset."""
        logger.info(f"Loading dataset for validation from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Raw data file not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df):,} records with columns: {list(df.columns)}")

        # 1. Schema & Required Columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        self.report["schema_valid"] = len(missing_cols) == 0
        self.report["missing_required_columns"] = missing_cols

        # 2. Missing Value Analysis
        null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
        null_pcts = (df[REQUIRED_COLUMNS].isnull().sum() / len(df) * 100).round(2).to_dict()
        self.report["missing_values"] = {
            col: {"count": int(null_counts[col]), "percentage": float(null_pcts[col])}
            for col in REQUIRED_COLUMNS
        }

        # 3. Duplicate Records
        exact_dups = int(df.duplicated().sum())
        business_dups = int(df.duplicated(subset=["arrival_date", "commodity", "market"]).sum())
        self.report["exact_duplicates"] = exact_dups
        self.report["business_key_duplicates"] = business_dups

        # 4. Date Validation
        df["parsed_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
        invalid_dates = int(df["parsed_date"].isna().sum())
        date_min = df["parsed_date"].min().strftime("%Y-%m-%d") if df["parsed_date"].notna().any() else None
        date_max = df["parsed_date"].max().strftime("%Y-%m-%d") if df["parsed_date"].notna().any() else None
        self.report["invalid_dates_count"] = invalid_dates
        self.report["date_range"] = {"earliest": date_min, "latest": date_max}

        # 5. Price Validation
        numeric_prices = {}
        for p_col in ["min_price", "max_price", "modal_price"]:
            col_series = pd.to_numeric(df[p_col], errors="coerce")
            neg_count = int((col_series < 0).sum())
            zero_count = int((col_series == 0).sum())
            nan_count = int(col_series.isna().sum())
            numeric_prices[p_col] = {
                "negative_count": neg_count,
                "zero_count": zero_count,
                "nan_count": nan_count,
                "min": float(col_series.min()) if col_series.notna().any() else None,
                "max": float(col_series.max()) if col_series.notna().any() else None,
                "mean": float(col_series.mean()) if col_series.notna().any() else None,
            }

        # Check Price Hierarchy (min_price <= modal_price <= max_price)
        p_min = pd.to_numeric(df["min_price"], errors="coerce")
        p_mod = pd.to_numeric(df["modal_price"], errors="coerce")
        p_max = pd.to_numeric(df["max_price"], errors="coerce")
        hierarchy_violations = int(((p_min > p_mod) | (p_mod > p_max)).sum())
        numeric_prices["hierarchy_violations_count"] = hierarchy_violations
        self.report["price_validation"] = numeric_prices

        # 6. Commodity & Market Consistency
        commodities_present = df["commodity"].unique().tolist()
        markets_present = df["market"].unique().tolist()
        self.report["commodities_present"] = commodities_present
        self.report["markets_present"] = markets_present
        self.report["target_commodities_found"] = [
            tc for tc in TARGET_COMMODITIES if any(tc.lower() in str(c).lower() for c in commodities_present)
        ]

        # 7. Commodity-Wise Breakdown
        comm_breakdown = {}
        for tc in TARGET_COMMODITIES:
            sub = df[df["commodity"].astype(str).str.contains(tc, case=False, na=False)].copy()
            if not sub.empty:
                sub_dates = pd.to_datetime(sub["arrival_date"], errors="coerce").dropna()
                comm_breakdown[tc] = {
                    "total_records": len(sub),
                    "unique_dates": int(sub_dates.nunique()),
                    "earliest_date": sub_dates.min().strftime("%Y-%m-%d"),
                    "latest_date": sub_dates.max().strftime("%Y-%m-%d"),
                    "avg_modal_price_quintal": round(float(pd.to_numeric(sub["modal_price"], errors="coerce").mean()), 2),
                }
            else:
                comm_breakdown[tc] = {"total_records": 0}
        self.report["commodity_breakdown"] = comm_breakdown

        # Overall Quality Status
        has_critical_error = (
            not self.report["schema_valid"]
            or invalid_dates > 0
            or numeric_prices["modal_price"]["negative_count"] > 0
        )
        self.report["status"] = "FAIL" if has_critical_error else "PASS"

        self.print_report()
        return self.report

    def print_report(self):
        """Prints a human-readable validation summary."""
        logger.info("=" * 68)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 68)
        logger.info(f"Overall Validation Status: {self.report['status']}")
        logger.info(f"Schema Valid: {self.report['schema_valid']} (Missing required cols: {self.report['missing_required_columns']})")
        logger.info(f"Date Range: {self.report['date_range']['earliest']} to {self.report['date_range']['latest']}")
        logger.info(f"Invalid Dates: {self.report['invalid_dates_count']}")
        logger.info(f"Exact Duplicate Rows: {self.report['exact_duplicates']}")
        logger.info(f"Business Key Duplicates: {self.report['business_key_duplicates']}")

        logger.info("\n--- Price Validation Checks ---")
        pv = self.report["price_validation"]
        logger.info(f"Negative Modal Prices: {pv['modal_price']['negative_count']}")
        logger.info(f"Zero Modal Prices    : {pv['modal_price']['zero_count']}")
        logger.info(f"Hierarchy Violations : {pv['hierarchy_violations_count']} (where min > modal or modal > max)")
        logger.info(f"Modal Price Range    : Rs.{pv['modal_price']['min']:.0f} - Rs.{pv['modal_price']['max']:.0f}/quintal")

        logger.info("\n--- Target Commodity Coverage ---")
        for comm, stats in self.report["commodity_breakdown"].items():
            if stats["total_records"] > 0:
                logger.info(
                    f"  * {comm:8s}: {stats['total_records']:,} records | {stats['unique_dates']:,} days | "
                    f"Dates: {stats['earliest_date']} to {stats['latest_date']} | "
                    f"Avg Modal: Rs.{stats['avg_modal_price_quintal']:.1f}/quintal"
                )
            else:
                logger.warning(f"  * {comm:8s}: 0 records")
        logger.info("=" * 68)


def main():
    validator = DataValidator()
    report = validator.run_validation()
    
    # Save validation report
    report_file = RAW_DATA_FILE.parent / "validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved validation report to: {report_file}")


if __name__ == "__main__":
    main()
