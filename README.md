# AI Agricultural Economist: Bangalore Grocery Price Prediction
## Multi-Factor APMC Wholesale & Consumer Retail Forecasting Engine

A production-grade **Agricultural Economics & Machine Learning forecasting engine** designed to forecast next-day **APMC Mandi Wholesale Modal Rates (₹/kg)** and **Consumer Retail Prices across multiple channels (Pushcarts, Supermarkets, Quick-Commerce)** for essential agricultural commodities (**Tomato, Onion, Potato**) in **Bangalore, Karnataka, India**.

Built to predict like an **Agricultural Economist** by factoring in:
1. **Source-Region Agrometeorology & Weather Shocks**: Daily rainfall anomalies, deluge events, and heatwaves across primary producing hubs (**Kolar & Madanapalle** for Tomato, **Nashik & North Karnataka** for Onion, **Hassan & Belagavi** for Potato).
2. **Fuel & Interstate Logistics**: Daily Karnataka diesel prices and distance-weighted transit freight pass-through costs.
3. **Government Interventions & Schemes**: Buffer stock releases (NAFED/NCCF *Bharat Onion*), export restrictions/bans, and Operation Greens (TOP) transport subsidies.
4. **Harvest Cycles & Seasonality**: Rabi/Kharif harvest arrival flushes vs. lean deficit periods.
5. **Economic Factor Attribution**: Full decomposition of price changes into quantitative fundamental drivers.

---

## 1. System Architecture & Economic Pipeline

```
+---------------------------------------------------------------------------------------------------+
|                                  MULTI-FACTOR REAL-TIME DATA INGESTION                            |
+------------------------------------+--------------------------------+-----------------------------+
| 1. Live Bangalore Retail Scraper   | 2. Open-Meteo 14-Day Forecast  | 3. Fuel & Government Policy |
|    - vegetablemarketprice.com      |    - Kolar, Malur, Madanapalle |    - Karnataka Diesel Rate  |
|    - Real-time D0 wholesale/retail |    - Nashik Lasalgaon, Dharwad |    - NAFED Buffer Releases  |
|    - High/Low/Average spreads      |    - Hassan Belt, Belagavi     |    - Operation Greens (TOP) |
+------------------------------------+--------------------------------+-----------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------+
|                        Phase A: Economic Factor & Feature Engineering Engine                      |
|                                     (src/economic_factors.py)                                     |
|  - Source Region Rainfall Anomalies & Cumulative Disruption (1d, 3d, 7d precipitation)            |
|  - Heatwave & Blight Agrometeorological Stress Indices                                            |
|  - Freight Cost Index (Distance x Diesel Price) & Transport Pass-Through Rates                    |
|  - Government Policy Pressure Index & Buffer Stock Subsidy Flags                                  |
|  - Kharif / Rabi Production & Supply Seasonality Indices                                          |
+---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------+
|                        Phase B: Multi-Factor AI Model Training & Tuning                           |
|                                    (src/train.py, train_delta.py)                                 |
|  - Multi-Factor XGBoost Direct Regressors & Residual Delta Forecasters                            |
|  - Directional Accuracy & Extreme Event Spike Optimization                                       |
|  - Rigorous Time-Series Cross-Validation & Metric Tracking                                        |
+---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------+
|                  Phase C: AI Agricultural Economist Inference & Attribution Engine                |
|                                          (src/predict.py)                                         |
|  - Mandi Wholesale (APMC) & DCA Retail Next-Day Forecasts                                         |
|  - Multi-Channel Shelf Breakdown (Pushcart, Supermarket, Quick-Commerce)                          |
|  - Economic Factor Attribution Waterfall (Weather + Fuel + Govt Policy + Momentum + Supply)       |
|  - Economist Market Intelligence Commentary, Risk Rating & Procurement Advisory                  |
+---------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------+
|                        Phase D: Automated Daily Multi-Factor Synchronization                      |
|                                       (src/daily_update.py)                                       |
|  - Auto-fetches live Mandi rates + regional weather + fuel + policy indicators on every run       |
|  - Fast local caching and offline graceful resilience                                             |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Key Economic Drivers by Commodity

| Commodity | Primary Producing Belts | Transit Distance | Critical Agrometeorological & Economic Drivers |
| :--- | :--- | :---: | :--- |
| **Tomato** | **Kolar**, Chikkaballapur (Karnataka), Madanapalle (AP) | ~70 - 125 km | • Extreme sensitivity to **harvest rain shocks** (>25mm causes fruit rot & field impassability)<br>• Heatwaves (>36°C) cause blossom drop<br>• Short shelf life (18% transit spoilage loss) |
| **Onion** | **Nashik / Lasalgaon** (Maharashtra), Hubballi (Karnataka) | ~1,050 km | • High sensitivity to **interstate diesel freight** (1,050 km transit)<br>• **Unseasonal rains** in Oct-Nov damage curing and storage stocks<br>• **Govt Policy**: NAFED buffer releases (*Bharat Onion* @ ₹25/kg), export bans |
| **Potato** | **Hassan**, Belagavi (Karnataka), Agra cold storage (UP) | ~180 - 1,800 km | • Long cold storage durability (8-10 months)<br>• Late-blight risk in Hassan under wet/cold spells<br>• Operation Greens transport subsidies |

---

## 3. Real-Time Inference & CLI Usage

### 1. All-in-One Complete Market Intelligence (All 3 Factors in 1 Command)
Predicts dual Mandi wholesale & consumer retail prices, TreeSHAP economic cause factors, and 14-day spike radar in a single unified dashboard:
```powershell
python src/predict.py
```

### 2. Single Commodity Forecast (e.g. Tomato)
```powershell
python src/predict.py --commodity Tomato
```

### 3. Machine-Readable Full JSON Export
```powershell
python src/predict.py --commodity all --json
```

### 4. Force Daily Multi-Factor Synchronization (Mandi + Regional Weather + Diesel)
```powershell
python src/daily_update.py --sync
```

---

## 4. Continuous Forecasting, Tracking & Backtesting Pipeline

The system includes a production-grade automated continuous tracking and evaluation engine:
1. **Live Ground-Truth Ingestion**: Automatically fetches daily Bangalore prices.
2. **Automated Verification**: Matches past pending predictions against ground-truth actuals.
3. **Accuracy & Tolerance Scoring**: Calculates rolling MAE, RMSE, MAPE, Directional Accuracy, and Precision Hit Rates ($\pm 5\%$, $\pm 10\%$, $\pm ₹2/\text{kg}$).
4. **Permanent Append-Only Ledger**: Stored permanently in `data/tracking/prediction_ledger.csv`.
5. **Interactive Dashboard & Charts**: Generates `reports/dashboard.html` and high-res plots in `reports/figures/`.
6. **Periodic Automated Retraining**: Retrains Wholesale and Retail delta models as new data accumulates.

### Key Commands:
```powershell
# 1. Run full daily continuous cycle (Sync -> Verify -> Predict -> Retrain check -> Dashboard)
python src/continuous_pipeline.py --once

# 2. View current tracking scorecard and accuracy metrics
python src/continuous_pipeline.py --status

# 3. Seed historical 60-day walk-forward backtest
python src/continuous_pipeline.py --backtest --days 60

# 4. Open/refresh interactive HTML dashboard
python src/continuous_pipeline.py --dashboard

# 5. Run as continuous automated background service (runs every 24 hours)
python src/continuous_pipeline.py --daemon --interval-hours 24
```

---

## 5. PostgreSQL Enterprise Database Setup & Migration

AgroVision supports high-throughput relational persistence using **PostgreSQL** alongside SQLAlchemy ORM with automatic zero-downtime CSV fallback.

### Database Setup
1. Configure your database connection in `.env`:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=agrovision
DB_USER=postgres
DB_PASSWORD=your_password
```
*(Or provide `DATABASE_URL=postgresql://postgres:password@localhost:5432/agrovision`)*

### Schema Initialization & CSV-to-PostgreSQL Migration
Run the automated migration tool to create database tables and migrate existing historical datasets:
```powershell
# Migrate all datasets (Schema + Prediction Ledger + Historical Prices + Economic Indicators)
python scripts/migrate_to_postgres.py --all

# Or initialize schema only:
python scripts/migrate_to_postgres.py --init-schema
```

---

## 6. Daily Continuous Retraining & Real Market Reconciliation

AgroVision operates an automated daily continual learning pipeline using 100% verified ground-truth market data without synthetic or dummy data:

### Core Data Integrity Principles
1. **Zero Dummy Data**: All market baseline prices are strictly ingested from live Bangalore APMC mandi auctions and verified daily market portals (`vegetablemarketprice.com`).
2. **Authentic Out-of-Sample Verification**: Tomorrow's forecasts ($D+1$) remain strictly in `PENDING` state until tomorrow's real trading session closes and actual prices are recorded. No fake zero-error verifications are manufactured.
3. **Daily Autonomous Retraining**:
   - Ingests today's realized market prices and economic factors.
   - Re-engineers the 221-factor feature matrix (`data/processed/bangalore_grocery_features.csv`).
   - Retrains XGBoost Delta regressors and direct models on all accumulated historical samples (8,369 samples).
   - Generates tomorrow's real forward forecast and logs to the Prediction Ledger as `PENDING`.

### API Endpoints
- `POST /api/daily-train`: Execute full daily continuous retraining cycle on real market data.
- `GET /api/daily-train/status`: Get telemetry, samples count, directional accuracy, and retraining history.
- `GET /api/ledger`: Access the permanent prediction audit ledger.

---

## 7. Directory Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── api_response_metadata.json
│   │   ├── bangalore_grocery_prices_raw.csv
│   │   ├── source_region_weather.csv           <-- Cached Open-Meteo regional weather
│   │   └── validation_report.json
│   └── processed/
│       ├── bangalore_grocery_prices_processed.csv
│       ├── bangalore_economic_factors.csv      <-- Master economic & weather store
│       └── bangalore_grocery_features.csv      <-- 124 multi-factor features
├── models/
│   ├── feature_columns.json
│   ├── metrics.json
│   ├── delta_models_metrics.json
│   ├── tomato_xgboost.json
│   ├── tomato_delta_xgboost.json
│   ├── tomato_retail_xgboost.json
│   ├── onion_xgboost.json
│   ├── onion_delta_xgboost.json
│   ├── onion_retail_xgboost.json
│   ├── potato_xgboost.json
│   ├── potato_delta_xgboost.json
│   ├── potato_retail_xgboost.json
│   └── figures/
└── src/
    ├── __init__.py
    ├── baseline.py
    ├── clean_data.py
    ├── collect_data.py
    ├── config.py
    ├── continuous_pipeline.py     <-- Continuous tracking & backtesting orchestrator
    ├── daily_auto_trainer.py      <-- Autonomous daily verification, feature rebuild & retraining
    ├── daily_update.py            <-- Multi-factor live daily synchronization
    ├── economic_factors.py        <-- Agrometeorology, fuel, govt schemes, arrival cycle
    ├── evaluate.py
    ├── features.py                <-- 221-factor feature engineering engine
    ├── forecasting_engine.py      <-- Adaptive Structural State-Space & ML (ASML) Ensemble
    ├── forward_weather_engine.py  <-- Live Open-Meteo 14-day agrometeorological forward feeds
    ├── live_market_scraper.py     <-- Live Bangalore market scraper (vegetablemarketprice.com)
    ├── predict.py                 <-- AI Economist factor attribution & multi-horizon forecasting
    ├── real_time_prices.py        <-- Live market price fusion & retail spread guarantee
    ├── train.py                   <-- Multi-factor XGBoost training & hyperparameter tuning
    ├── train_delta.py             <-- Residual delta price shift forecasters
    └── validate_data.py
```

---

## 6. Execution Pipeline

```bash
# 1. Collect Mandi Historical & Live Data
python src/collect_data.py

# 2. Build Unified Economic & Regional Weather Store
python src/economic_factors.py

# 3. Clean and Validate
python src/clean_data.py

# 4. Generate Multi-Factor Features
python src/features.py

# 5. Train AI Models
python src/train.py
python src/train_delta.py

# 6. Evaluate & Benchmark
python src/evaluate.py

# 7. Run AI Economist Predictions
python src/predict.py --commodity all
```
