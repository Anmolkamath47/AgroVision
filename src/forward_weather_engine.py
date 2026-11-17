"""
Live Forward Agrometeorological Forecast Engine (Open-Meteo 14-Day Global Weather API).
Connects real-time forward weather projections directly into multi-horizon price forecasting:
- Kolar & Madanapalle (Tomato supply corridor)
- Nashik & Hubballi (National & regional Onion corridor)
- Hassan & Belagavi (Karnataka Potato corridor)
- Bangalore APMC (Consumption center)

Provides forward weather shock indicators for Day 1, Day 3, Day 7, and Day 14.
"""

import os
import sys
import json
import time
import logging
import requests
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("ForwardWeatherEngine")

WEATHER_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "forward_weather_14d.json"
)

PRODUCING_CORRIDORS = {
    "Tomato": [
        {"name": "Kolar & Malur", "lat": 13.1367, "lon": 78.1291, "weight": 0.65},
        {"name": "Madanapalle AP", "lat": 13.5500, "lon": 78.5000, "weight": 0.35},
    ],
    "Onion": [
        {"name": "Nashik Lasalgaon", "lat": 19.9975, "lon": 73.7898, "weight": 0.70},
        {"name": "Hubballi Dharwad", "lat": 15.3647, "lon": 75.1240, "weight": 0.30},
    ],
    "Potato": [
        {"name": "Hassan Table Belt", "lat": 13.0033, "lon": 76.1004, "weight": 0.80},
        {"name": "Bangalore APMC", "lat": 12.9716, "lon": 77.5946, "weight": 0.20},
    ],
}


def _create_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class LiveForwardWeatherEngine:
    """Fetches real-time 14-day weather forecasts from Open-Meteo to model forward harvest shocks."""

    CACHE_TTL_SECONDS = 3600  # 1 hour cache
    _cached_weather: Optional[Dict[str, Any]] = None
    _last_fetched_ts: float = 0.0

    @classmethod
    def get_forward_weather_profiles(cls, force_refresh: bool = False) -> Dict[str, Any]:
        """Returns 14-day forward agrometeorology for all target commodities."""
        now = time.time()
        if not force_refresh and cls._cached_weather and (now - cls._last_fetched_ts < cls.CACHE_TTL_SECONDS):
            return cls._cached_weather

        # Check disk cache
        if not force_refresh and os.path.exists(WEATHER_CACHE_FILE):
            try:
                with open(WEATHER_CACHE_FILE, "r", encoding="utf-8") as f:
                    disk_cache = json.load(f)
                    cache_ts = disk_cache.get("_timestamp", 0)
                    if now - cache_ts < cls.CACHE_TTL_SECONDS and "commodities" in disk_cache:
                        cls._cached_weather = disk_cache["commodities"]
                        cls._last_fetched_ts = cache_ts
                        return cls._cached_weather
            except Exception as e:
                logger.debug(f"Note reading weather disk cache: {e}")

        # Fetch live from Open-Meteo API
        profiles = cls._fetch_live_open_meteo()
        if profiles:
            cls._cached_weather = profiles
            cls._last_fetched_ts = now
            cls._save_disk_cache(profiles)
            return profiles

        # Fallback to defaults if network unreachable
        fallback = cls._generate_fallback_profiles()
        cls._cached_weather = fallback
        cls._last_fetched_ts = now
        return fallback

    @classmethod
    def _fetch_live_open_meteo(cls) -> Dict[str, Any]:
        """Queries Open-Meteo Global Forecast API for all hubs using requests session."""
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgroVision/2.0"})
        results = {}

        for comm, hubs in PRODUCING_CORRIDORS.items():
            daily_rains = [0.0] * 14
            daily_temps = [28.0] * 14
            forecast_dates = []

            for hub in hubs:
                lat = hub["lat"]
                lon = hub["lon"]
                wt = hub["weight"]

                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
                    "timezone": "Asia/Kolkata",
                    "forecast_days": 14,
                }
                url = "https://api.open-meteo.com/v1/forecast"

                try:
                    res = session.get(url, params=params, timeout=4)
                    if res.status_code == 200:
                        data = res.json()
                        daily = data.get("daily", {})
                        p_sum = daily.get("precipitation_sum", [])
                        t_max = daily.get("temperature_2m_max", [])
                        dates = daily.get("time", [])

                        if dates and not forecast_dates:
                            forecast_dates = dates

                        for d_idx in range(min(14, len(p_sum))):
                            r_val = float(p_sum[d_idx] or 0.0)
                            t_val = float(t_max[d_idx] or 28.0)
                            daily_rains[d_idx] += r_val * wt
                            daily_temps[d_idx] = max(daily_temps[d_idx], t_val)
                except Exception as e:
                    logger.debug(f"Weather forecast note for {hub['name']}: {e}")

            # Compute forward shocks for key horizons (D+1, D+3, D+7, D+14)
            # Cumulative rainfall in preceding window creates harvesting delays
            d1_rain = round(daily_rains[1], 1) if len(daily_rains) > 1 else 0.0
            d3_rain = round(sum(daily_rains[1:4]), 1) if len(daily_rains) > 3 else 0.0
            d7_rain = round(sum(daily_rains[1:8]), 1) if len(daily_rains) > 7 else 0.0
            d14_rain = round(sum(daily_rains[1:14]), 1) if len(daily_rains) > 13 else 0.0

            # Calculate price shock impact (Rs/kg) from forward weather
            if comm == "Tomato":
                # Heavy rain (>15mm) causes fruit rot & impassable fields
                d1_shock = 0.50 if d1_rain > 15.0 else 0.0
                d3_shock = 1.20 if d3_rain > 20.0 else (0.50 if d3_rain > 10.0 else 0.0)
                d7_shock = 1.80 if d7_rain > 25.0 else (0.80 if d7_rain > 12.0 else -0.30)
                d14_shock = 0.0 if d14_rain < 30.0 else 1.00
            elif comm == "Onion":
                # Storage & curing belt in Nashik: clear weather facilitates dispatch
                d1_shock = 0.0
                d3_shock = 0.80 if d3_rain > 20.0 else 0.0
                d7_shock = 1.50 if d7_rain > 25.0 else -0.40  # Clear weather cools prices
                d14_shock = -0.50 if d14_rain < 15.0 else 0.50
            else:  # Potato
                # Hassan blight & transport
                d1_shock = 0.0
                d3_shock = 0.40 if d3_rain > 15.0 else 0.0
                d7_shock = 0.90 if d7_rain > 20.0 else -0.20
                d14_shock = 0.0

            results[comm] = {
                "commodity": comm,
                "forecast_dates": forecast_dates,
                "daily_rainfall_mm": [round(x, 1) for x in daily_rains],
                "daily_max_temp_c": [round(x, 1) for x in daily_temps],
                "shock_analysis": {
                    "tomorrow_d1": {"rainfall_mm": d1_rain, "price_impact_rs": d1_shock},
                    "day_3_ahead": {"rainfall_mm": d3_rain, "price_impact_rs": d3_shock},
                    "day_7_ahead": {"rainfall_mm": d7_rain, "price_impact_rs": d7_shock},
                    "day_14_ahead": {"rainfall_mm": d14_rain, "price_impact_rs": d14_shock},
                },
                "source": "Open-Meteo Global Agrometeorology 14-Day Live API",
            }

        return results

    @classmethod
    def _generate_fallback_profiles(cls) -> Dict[str, Any]:
        """Graceful fallback profile if API is temporarily unavailable."""
        results = {}
        for comm in PRODUCING_CORRIDORS:
            results[comm] = {
                "commodity": comm,
                "daily_rainfall_mm": [0.0] * 14,
                "daily_max_temp_c": [28.0] * 14,
                "shock_analysis": {
                    "tomorrow_d1": {"rainfall_mm": 0.0, "price_impact_rs": 0.0},
                    "day_3_ahead": {"rainfall_mm": 0.0, "price_impact_rs": 0.0},
                    "day_7_ahead": {"rainfall_mm": 0.0, "price_impact_rs": 0.0},
                    "day_14_ahead": {"rainfall_mm": 0.0, "price_impact_rs": 0.0},
                },
                "source": "Open-Meteo Global Agrometeorology (Cached / Fallback)",
            }
        return results

    @classmethod
    def _save_disk_cache(cls, data: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(WEATHER_CACHE_FILE), exist_ok=True)
            payload = {
                "_timestamp": time.time(),
                "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
                "commodities": data,
            }
            with open(WEATHER_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug(f"Error saving weather disk cache: {e}")


if __name__ == "__main__":
    weather = LiveForwardWeatherEngine.get_forward_weather_profiles(force_refresh=True)
    print(json.dumps(weather, indent=2))
