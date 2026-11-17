"""
Multi-Hub Agrometeorology & Economic Factors Engine for Bangalore Grocery Price Forecasting.
Maps and integrates exact producing origins supplying Bangalore APMC:
1. Tomato Corridors: Kolar, Chikkaballapur, Madanapalle (AP), Dharmapuri (TN)
2. Onion Corridors: Nashik / Lasalgaon (MH), Chitradurga, Hubballi-Dharwad
3. Potato Corridors: Hassan / Channarayapatna, Agra / Farrukhabad (UP)
4. Consumption APMC: Bangalore (Yeshwanthpur)
5. Multi-Route Diesel Freight Logistics & Distance-Weighted Transport Pass-Through Costs
6. Government Interventions, Buffer Stocks & Subsidies (NAFED/NCCF Bharat Onion, Export Bans, Operation Greens)
7. Agricultural Harvest Arrival Cycles & Lean Deficit Indices
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import urllib.request
import urllib.parse
import ssl
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COMMODITIES,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# File Paths
WEATHER_CACHE_FILE = RAW_DATA_DIR / "source_region_weather.csv"
ECONOMIC_DATA_FILE = PROCESSED_DATA_DIR / "bangalore_economic_factors.csv"

# Comprehensive Geographic Coordinates of All Primary Producing Hubs supplying Bangalore APMC
SOURCE_REGIONS = {
    # --- TOMATO CORRIDORS ---
    "kolar": {
        "name": "Kolar & Malur (Primary Tomato Hub)",
        "lat": 13.1367,
        "lon": 78.1291,
        "state": "Karnataka",
        "distance_km": 70,
        "primary_for": ["Tomato"],
        "corridor_weight": 0.40,
    },
    "chikkaballapur": {
        "name": "Chikkaballapur & Chintamani (Tomato Tract)",
        "lat": 13.4355,
        "lon": 77.7315,
        "state": "Karnataka",
        "distance_km": 60,
        "primary_for": ["Tomato"],
        "corridor_weight": 0.25,
    },
    "madanapalle": {
        "name": "Madanapalle & Punganur (Andhra Border Hub)",
        "lat": 13.5500,
        "lon": 78.5000,
        "state": "Andhra Pradesh",
        "distance_km": 130,
        "primary_for": ["Tomato"],
        "corridor_weight": 0.25,
    },
    "dharmapuri": {
        "name": "Dharmapuri & Krishnagiri (Tamil Nadu Belt)",
        "lat": 12.1211,
        "lon": 78.1582,
        "state": "Tamil Nadu",
        "distance_km": 100,
        "primary_for": ["Tomato"],
        "corridor_weight": 0.10,
    },
    # --- ONION CORRIDORS ---
    "nashik": {
        "name": "Nashik / Lasalgaon / Pimpalgaon (National Onion Hub)",
        "lat": 19.9975,
        "lon": 73.7898,
        "state": "Maharashtra",
        "distance_km": 1050,
        "primary_for": ["Onion"],
        "corridor_weight": 0.65,
    },
    "chitradurga": {
        "name": "Chitradurga & Challakere (Karnataka Kharif Onion)",
        "lat": 14.2251,
        "lon": 76.3980,
        "state": "Karnataka",
        "distance_km": 200,
        "primary_for": ["Onion"],
        "corridor_weight": 0.20,
    },
    "hubballi": {
        "name": "Hubballi-Dharwad & Gadag (North Karnataka Onion)",
        "lat": 15.3647,
        "lon": 75.1240,
        "state": "Karnataka",
        "distance_km": 410,
        "primary_for": ["Onion"],
        "corridor_weight": 0.15,
    },
    # --- POTATO CORRIDORS ---
    "hassan": {
        "name": "Hassan & Channarayapatna (Karnataka Kharif Potato)",
        "lat": 13.0033,
        "lon": 76.1004,
        "state": "Karnataka",
        "distance_km": 180,
        "primary_for": ["Potato"],
        "corridor_weight": 0.65,
    },
    "agra": {
        "name": "Agra & Farrukhabad (UP Cold Storage Reserve)",
        "lat": 27.1767,
        "lon": 78.0081,
        "state": "Uttar Pradesh",
        "distance_km": 1950,
        "primary_for": ["Potato"],
        "corridor_weight": 0.35,
    },
    # --- CONSUMPTION MANDI ---
    "bangalore": {
        "name": "Bangalore APMC (Yeshwanthpur Consumption Hub)",
        "lat": 12.9716,
        "lon": 77.5946,
        "state": "Karnataka",
        "distance_km": 0,
        "primary_for": ["Tomato", "Onion", "Potato"],
        "corridor_weight": 1.00,
    },
}


def _create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_open_meteo_weather(
    lat: float,
    lon: float,
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetches daily weather metrics (rainfall, max/min temp) from Open-Meteo APIs.
    Combines Archive API (historical) and Forecast API (recent/current).
    """
    if end_date is None:
        end_date = (date.today() + timedelta(days=2)).strftime("%Y-%m-%d")

    ssl_ctx = _create_ssl_context()
    dfs = []

    # 1. Historical Archive API (supports dates up to ~5 days ago)
    archive_end_date = min(end_date, (date.today() - timedelta(days=5)).strftime("%Y-%m-%d"))
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    archive_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": archive_end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
        "timezone": "Asia/Kolkata",
    }
    encoded_archive = f"{archive_url}?{urllib.parse.urlencode(archive_params)}"

    try:
        req = urllib.request.Request(encoded_archive, headers={"User-Agent": "BangaloreAgriPriceForecaster/2.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "daily" in data and "time" in data["daily"]:
                df_arch = pd.DataFrame({
                    "date": pd.to_datetime(data["daily"]["time"]),
                    "temp_max": data["daily"]["temperature_2m_max"],
                    "temp_min": data["daily"]["temperature_2m_min"],
                    "precipitation_mm": data["daily"]["precipitation_sum"],
                    "rain_mm": data["daily"]["rain_sum"],
                })
                dfs.append(df_arch)
    except Exception as e:
        logger.warning(f"Archive API call note ({lat}, {lon}): {e}")

    # 2. Live Forecast API
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 14,
        "forecast_days": 3,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
        "timezone": "Asia/Kolkata",
    }
    encoded_fc = f"{forecast_url}?{urllib.parse.urlencode(forecast_params)}"

    try:
        req = urllib.request.Request(encoded_fc, headers={"User-Agent": "BangaloreAgriPriceForecaster/2.0"})
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if "daily" in data and "time" in data["daily"]:
                df_fc = pd.DataFrame({
                    "date": pd.to_datetime(data["daily"]["time"]),
                    "temp_max": data["daily"]["temperature_2m_max"],
                    "temp_min": data["daily"]["temperature_2m_min"],
                    "precipitation_mm": data["daily"]["precipitation_sum"],
                    "rain_mm": data["daily"]["rain_sum"],
                })
                dfs.append(df_fc)
    except Exception as e:
        logger.warning(f"Forecast API call note ({lat}, {lon}): {e}")

    if not dfs:
        logger.warning(f"Generating synthetic agrometeorological fallback for ({lat}, {lon}).")
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame({
            "date": dates,
            "temp_max": 30.0 + 3.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25),
            "temp_min": 19.0 + 2.0 * np.sin(2 * np.pi * dates.dayofyear / 365.25),
            "precipitation_mm": 2.0 + 3.0 * np.maximum(0, np.sin(2 * np.pi * (dates.dayofyear - 150) / 365.25)),
            "rain_mm": 2.0 + 3.0 * np.maximum(0, np.sin(2 * np.pi * (dates.dayofyear - 150) / 365.25)),
        })

    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df_combined


class EconomicFactorEngine:
    """
    Fetches, builds, and maintains the Unified 10-Hub Economic & Agrometeorological Store.
    """

    def __init__(self, force_refresh_weather: bool = False):
        self.force_refresh_weather = force_refresh_weather

    def fetch_all_source_region_weather(self) -> pd.DataFrame:
        """Fetches daily weather across all 10 producing and consumption hubs."""
        if os.path.exists(WEATHER_CACHE_FILE) and not self.force_refresh_weather:
            try:
                df_cached = pd.read_csv(WEATHER_CACHE_FILE)
                df_cached["date"] = pd.to_datetime(df_cached["date"])
                max_cached_date = df_cached["date"].max().strftime("%Y-%m-%d")
                today_str = date.today().strftime("%Y-%m-%d")
                
                # If cached within last 2 days, use cache
                if max_cached_date >= (date.today() - timedelta(days=2)).strftime("%Y-%m-%d"):
                    # Check if all 10 hubs are in cache
                    required_hubs = set(SOURCE_REGIONS.keys())
                    cached_hubs = {c.split("_")[0] for c in df_cached.columns if "_" in c}
                    if required_hubs.issubset(cached_hubs):
                        logger.info(f"Loaded source-region weather dataset from cache ({len(df_cached):,} days).")
                        return df_cached
            except Exception as e:
                logger.debug(f"Cache check note: {e}")

        logger.info("Fetching agrometeorological time series for all 10 key agricultural supply corridors...")
        region_dfs = {}
        for reg_key, reg_info in SOURCE_REGIONS.items():
            logger.info(f"  * Fetching weather for {reg_info['name']} ({reg_info['lat']}, {reg_info['lon']})...")
            df_reg = fetch_open_meteo_weather(reg_info["lat"], reg_info["lon"])
            prefix = reg_key.lower()
            df_reg = df_reg.rename(columns={
                "temp_max": f"{prefix}_temp_max",
                "temp_min": f"{prefix}_temp_min",
                "precipitation_mm": f"{prefix}_precip_mm",
                "rain_mm": f"{prefix}_rain_mm",
            })
            region_dfs[reg_key] = df_reg
            time.sleep(0.4)  # Rate limiting courtesy

        # Merge on date
        merged = region_dfs["kolar"]
        for reg_key in list(SOURCE_REGIONS.keys())[1:]:
            merged = pd.merge(merged, region_dfs[reg_key], on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged = merged.ffill().bfill()

        # Cache weather dataset
        os.makedirs(os.path.dirname(WEATHER_CACHE_FILE), exist_ok=True)
        merged.to_csv(WEATHER_CACHE_FILE, index=False, encoding="utf-8")
        logger.info(f"Regional weather dataset cached to {WEATHER_CACHE_FILE} ({len(merged):,} days, {len(SOURCE_REGIONS)} hubs).")
        return merged

    def generate_fuel_price_series(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Models realistic Karnataka diesel retail price series & 30-day velocity."""
        df_fuel = pd.DataFrame({"date": dates})
        days_from_start = (dates - dates[0]).days.values
        
        # Base realistic diesel price trajectory in Karnataka (Rs.86 to Rs.89.50/L)
        base_diesel = 86.5 + (days_from_start / len(dates)) * 2.8
        crude_fluctuation = 1.2 * np.sin(2 * np.pi * days_from_start / 180.0) + 0.8 * np.cos(2 * np.pi * days_from_start / 90.0)
        diesel_series = np.clip(base_diesel + crude_fluctuation, 84.5, 93.0)

        df_fuel["diesel_price_per_litre"] = np.round(diesel_series, 2)
        df_fuel["diesel_price_lag_7"] = df_fuel["diesel_price_per_litre"].shift(7).bfill()
        df_fuel["diesel_price_pct_change_30d"] = (
            (df_fuel["diesel_price_per_litre"] - df_fuel["diesel_price_per_litre"].shift(30).bfill())
            / df_fuel["diesel_price_per_litre"].shift(30).bfill() * 100.0
        ).round(2)

        return df_fuel

    def generate_government_policy_indicators(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Models central & state government interventions, buffer releases, and export curbs."""
        df_govt = pd.DataFrame({"date": dates})

        # 1. Onion Export Ban (Active Dec 8, 2023 to May 4, 2024; Export Duty from Aug 2023)
        df_govt["govt_export_ban_onion"] = (
            (dates >= "2023-12-08") & (dates <= "2024-05-04")
        ).astype(int)

        df_govt["govt_export_duty_onion"] = (
            ((dates >= "2023-08-19") & (dates <= "2023-12-07")) |
            ((dates >= "2024-05-05") & (dates <= "2024-09-13"))
        ).astype(int)

        # 2. NAFED / NCCF Buffer Stock Open Market Releases (Bharat Onion @ Rs.25/kg)
        df_govt["govt_buffer_release_onion"] = (
            ((dates >= "2023-09-01") & (dates <= "2023-12-15")) |
            ((dates >= "2024-09-01") & (dates <= "2024-12-20")) |
            ((dates >= "2025-08-15") & (dates <= "2025-11-30")) |
            ((dates >= "2026-08-01") & (dates <= "2026-08-26"))
        ).astype(int)

        # 3. Tomato Market Stabilization Intervention
        df_govt["govt_market_intervention_tomato"] = (
            ((dates >= "2023-07-10") & (dates <= "2023-08-25")) |
            ((dates >= "2024-06-20") & (dates <= "2024-07-31"))
        ).astype(int)

        # 4. Operation Greens (TOP) Transport & Storage Subsidies
        df_govt["govt_operation_greens_top"] = 1  # Continuous standard scheme

        # Synthetic Policy Pressure Indices
        df_govt["policy_suppression_index_onion"] = (
            df_govt["govt_export_ban_onion"] * 0.45 +
            df_govt["govt_export_duty_onion"] * 0.25 +
            df_govt["govt_buffer_release_onion"] * 0.30
        )
        df_govt["policy_suppression_index_tomato"] = df_govt["govt_market_intervention_tomato"] * 0.70
        df_govt["policy_suppression_index_potato"] = 0.15

        return df_govt

    def generate_crop_seasonality_cycles(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Models Kharif, Late Kharif, and Rabi production cycles and lean arrival gaps."""
        df_season = pd.DataFrame({"date": dates})
        doy = dates.dayofyear.values

        # Tomato: Heavy lean peak in June-July and Nov; Winter Rabi peak in Jan-March
        df_season["tomato_supply_cycle_index"] = np.round(
            0.65 * np.sin(2 * np.pi * (doy - 45) / 365.25) + 0.35 * np.cos(4 * np.pi * (doy - 20) / 365.25),
            3,
        )

        # Onion: Rabi flush in April-June (+supply); Lean pre-Kharif gap in Sept-Nov (-supply/tightness)
        df_season["onion_supply_cycle_index"] = np.round(
            0.80 * np.sin(2 * np.pi * (doy - 120) / 365.25),
            3,
        )

        # Potato: Agra/UP harvest in Feb-April (+supply); Southern monsoon Hassan crop in July-Oct
        df_season["potato_supply_cycle_index"] = np.round(
            0.60 * np.sin(2 * np.pi * (doy - 75) / 365.25) + 0.40 * np.sin(2 * np.pi * (doy - 240) / 365.25),
            3,
        )

        return df_season

    def generate_cpi_food_inflation(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Models Consumer Food Price Index (CFPI) monthly trajectory & YoY momentum."""
        df_cpi = pd.DataFrame({"date": dates})
        days_from_start = (dates - dates[0]).days.values
        # Base realistic food inflation trajectory in India (140 to 195 base index)
        base_cfpi = 142.0 + (days_from_start / len(dates)) * 48.0
        seasonal_cfpi = 3.5 * np.sin(2 * np.pi * (dates.dayofyear - 180) / 365.25)
        df_cpi["cpi_food_price_index"] = np.round(base_cfpi + seasonal_cfpi, 2)
        df_cpi["cpi_food_inflation_yoy_pct"] = (
            (df_cpi["cpi_food_price_index"] - df_cpi["cpi_food_price_index"].shift(365).bfill())
            / df_cpi["cpi_food_price_index"].shift(365).bfill() * 100.0
        ).round(2)
        return df_cpi

    def generate_festival_demand_indices(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Models Indian cultural festival surges & wedding season consumption demand multipliers."""
        df_demand = pd.DataFrame({"date": dates})
        doy = dates.dayofyear.values
        month = dates.month.values
        day = dates.day.values

        # 1. Major Festival Demand Peaks (Sankranti: Jan, Ugadi: Apr, Ganesh: Sep, Dussehra/Diwali: Oct-Nov)
        # Sankranti/Pongal (Jan 12-16)
        sankranti_surge = np.where((month == 1) & (day >= 12) & (day <= 16), 0.35, 0.0)
        # Ugadi / Summer festivals (April 5-15)
        ugadi_surge = np.where((month == 4) & (day >= 5) & (day <= 15), 0.30, 0.0)
        # Ganesh Chaturthi / Onam (Sept 5-15)
        ganesh_surge = np.where((month == 9) & (day >= 5) & (day <= 15), 0.40, 0.0)
        # Dussehra / Navratri / Diwali Peak (Oct 10 - Nov 15)
        diwali_surge = np.where((month == 10) & (day >= 10) | ((month == 11) & (day <= 15)), 0.60, 0.0)
        # Winter Wedding Season (Nov 16 - Dec 15 & Jan 15 - Feb 28)
        wedding_surge = np.where(
            ((month == 11) & (day >= 16)) | ((month == 12) & (day <= 15)) | ((month == 1) & (day >= 15)) | (month == 2),
            0.25,
            0.0,
        )

        composite_festival_demand = np.clip(
            sankranti_surge + ugadi_surge + ganesh_surge + diwali_surge + wedding_surge,
            0.0,
            1.0,
        )

        df_demand["festival_demand_surge_index"] = np.round(composite_festival_demand, 3)
        df_demand["is_peak_festival_week"] = (composite_festival_demand >= 0.35).astype(int)
        return df_demand

    def build_economic_feature_store(self) -> pd.DataFrame:
        """
        Synthesizes all 10-hub weather features, corridor-weighted stress metrics,
        freight logistics, policy interventions, CPI food inflation, and crop seasonality into a master dataframe.
        """
        logger.info("Building Master Multi-Hub Economic & Agrometeorological Feature Store...")
        df_weather = self.fetch_all_source_region_weather()
        dates = pd.DatetimeIndex(df_weather["date"])

        # 1. Agrometeorological Shock Indicators per Hub
        for hub in SOURCE_REGIONS.keys():
            precip_col = f"{hub}_precip_mm"
            temp_max_col = f"{hub}_temp_max"
            temp_min_col = f"{hub}_temp_min"

            if precip_col in df_weather.columns:
                df_weather[f"{hub}_rain_3d_sum"] = df_weather[precip_col].rolling(3, min_periods=1).sum().round(1)
                df_weather[f"{hub}_rain_7d_sum"] = df_weather[precip_col].rolling(7, min_periods=1).sum().round(1)
                df_weather[f"{hub}_rain_14d_sum"] = df_weather[precip_col].rolling(14, min_periods=1).sum().round(1)
                df_weather[f"{hub}_heavy_rain_shock"] = (df_weather[precip_col] >= 25.0).astype(int)
                df_weather[f"{hub}_deluge_shock"] = (df_weather[f"{hub}_rain_3d_sum"] >= 50.0).astype(int)

                # Consecutive wet harvesting days (rain >= 10mm)
                is_wet = (df_weather[precip_col] >= 10.0).astype(int)
                wet_runs = is_wet.groupby((~is_wet.astype(bool)).cumsum()).cumsum()
                df_weather[f"{hub}_consecutive_wet_days"] = wet_runs

            if temp_max_col in df_weather.columns:
                df_weather[f"{hub}_heatwave_shock"] = (df_weather[temp_max_col] >= 36.0).astype(int)

        # Unseasonal rains in Maharashtra/Karnataka onion corridors (Oct-Nov & March)
        month_arr = dates.month.values
        if "nashik_rain_7d_sum" in df_weather.columns:
            unseasonal_months = np.isin(month_arr, [3, 10, 11]).astype(int)
            df_weather["nashik_unseasonal_rain_shock"] = (
                (df_weather["nashik_rain_7d_sum"] > 25.0) & (unseasonal_months == 1)
            ).astype(int)

        # Late-Blight Risk in Hassan & Agra Potato Belts
        if "hassan_precip_mm" in df_weather.columns and "hassan_temp_min" in df_weather.columns:
            df_weather["hassan_blight_risk_index"] = np.where(
                (df_weather["hassan_rain_7d_sum"] > 30.0) & (df_weather["hassan_temp_min"] <= 20.0),
                1.0,
                0.0,
            )
        if "agra_precip_mm" in df_weather.columns:
            df_weather["agra_rain_7d_sum"] = df_weather["agra_precip_mm"].rolling(7, min_periods=1).sum().round(1)

        # 2. Corridor-Weighted Composite Stress Indices
        # Tomato Corridor (Kolar 40%, Chikkaballapur 25%, Madanapalle 25%, Dharmapuri 10%)
        df_weather["tomato_corridor_rain_3d"] = (
            0.40 * df_weather.get("kolar_rain_3d_sum", 0) +
            0.25 * df_weather.get("chikkaballapur_rain_3d_sum", 0) +
            0.25 * df_weather.get("madanapalle_rain_3d_sum", 0) +
            0.10 * df_weather.get("dharmapuri_rain_3d_sum", 0)
        ).round(1)

        df_weather["tomato_weather_stress_index"] = np.clip(
            (df_weather["tomato_corridor_rain_3d"] / 60.0) * 0.70 +
            df_weather.get("kolar_heatwave_shock", 0) * 0.30,
            0.0, 1.0
        ).round(3)

        # Onion Corridor (Nashik 65%, Chitradurga 20%, Hubballi 15%)
        df_weather["onion_corridor_rain_7d"] = (
            0.65 * df_weather.get("nashik_rain_7d_sum", 0) +
            0.20 * df_weather.get("chitradurga_rain_7d_sum", 0) +
            0.15 * df_weather.get("hubballi_rain_7d_sum", 0)
        ).round(1)

        df_weather["onion_weather_stress_index"] = np.clip(
            (df_weather["onion_corridor_rain_7d"] / 70.0) * 0.60 +
            df_weather.get("nashik_unseasonal_rain_shock", 0) * 0.40,
            0.0, 1.0
        ).round(3)

        # Potato Corridor (Hassan 65%, Agra 35%)
        df_weather["potato_corridor_rain_7d"] = (
            0.65 * df_weather.get("hassan_rain_7d_sum", 0) +
            0.35 * df_weather.get("agra_rain_7d_sum", 0)
        ).round(1)

        df_weather["potato_weather_stress_index"] = np.clip(
            df_weather.get("hassan_blight_risk_index", 0) * 0.60 +
            (df_weather["potato_corridor_rain_7d"] / 60.0) * 0.40,
            0.0, 1.0
        ).round(3)

        df_weather["bangalore_rain_disruption"] = (df_weather.get("bangalore_precip_mm", 0) >= 20.0).astype(int)

        # 3. Fuel & Freight Calculations
        df_fuel = self.generate_fuel_price_series(dates)
        
        # Route distances: Kolar=70km, Chikkaballapur=60km, Nashik=1050km, Hassan=180km, Agra=1950km
        diesel = df_fuel["diesel_price_per_litre"]
        df_fuel["freight_cost_tomato_per_kg"] = (70.0 / (3.8 * 10000.0) * diesel * 9.5 + 0.65).round(2)
        df_fuel["freight_cost_onion_per_kg"] = (1050.0 / (3.8 * 10000.0) * diesel * 1.55 + 1.20).round(2)
        df_fuel["freight_cost_potato_per_kg"] = (180.0 / (3.8 * 10000.0) * diesel * 3.5 + 0.90).round(2)
        df_fuel["freight_cost_agra_potato_per_kg"] = (1950.0 / (3.8 * 10000.0) * diesel * 1.45 + 1.80).round(2)

        # 4. Government, Macro Inflation, Demand & Seasonality
        df_govt = self.generate_government_policy_indicators(dates)
        df_season = self.generate_crop_seasonality_cycles(dates)
        df_cpi = self.generate_cpi_food_inflation(dates)
        df_demand = self.generate_festival_demand_indices(dates)

        # Merge all into master dataframe
        df_master = pd.merge(df_weather, df_fuel, on="date", how="left")
        df_master = pd.merge(df_master, df_govt, on="date", how="left")
        df_master = pd.merge(df_master, df_season, on="date", how="left")
        df_master = pd.merge(df_master, df_cpi, on="date", how="left")
        df_master = pd.merge(df_master, df_demand, on="date", how="left")

        # Save to processed directory
        os.makedirs(os.path.dirname(ECONOMIC_DATA_FILE), exist_ok=True)
        df_master.to_csv(ECONOMIC_DATA_FILE, index=False, encoding="utf-8")
        
        logger.info(f"Master Economic Factor Store created: {len(df_master):,} days, {df_master.shape[1]} features.")
        logger.info(f"Saved to: {ECONOMIC_DATA_FILE}")
        return df_master


def get_economic_feature_store() -> pd.DataFrame:
    """Loads or creates the unified economic feature store."""
    if os.path.exists(ECONOMIC_DATA_FILE):
        df = pd.read_csv(ECONOMIC_DATA_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    engine = EconomicFactorEngine()
    return engine.build_economic_feature_store()


if __name__ == "__main__":
    engine = EconomicFactorEngine(force_refresh_weather=True)
    df = engine.build_economic_feature_store()
    print("Master Economic & Agrometeorological Store Features:")
    print(df.columns.tolist())
    print("\nLatest Snapshot:")
    print(df.tail(1).T)
