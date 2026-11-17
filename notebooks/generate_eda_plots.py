"""
Script to generate all 9 required EDA figures for Bangalore Grocery Price Analysis.
Saves high-resolution PNG plots to notebooks/figures/.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_DATA_FILE, TARGET_COMMODITIES

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

COMMODITY_COLORS = {
    "Tomato": "#e74c3c",
    "Onion": "#8e44ad",
    "Potato": "#d35400",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def generate_all_eda_plots():
    df = load_data()
    print(f"Loaded {len(df):,} records for EDA plotting.")

    # 1. Daily Price Time Series (2021-2026)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].sort_values("date")
        ax.plot(sub["date"], sub["modal_price_per_kg"], label=f"{comm} (Daily Modal Price)", color=COMMODITY_COLORS[comm], alpha=0.85, lw=1.3)
    ax.set_title("1. Daily Modal Wholesale Price Trends (Bangalore Mandi: 2021 - 2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (Rs. / kg)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_daily_price_trends.png"), dpi=150)
    plt.close(fig)

    # 2. 7-Day Moving Average
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].sort_values("date").copy()
        sub["ma7"] = sub["modal_price_per_kg"].rolling(window=7, min_periods=2).mean()
        ax.plot(sub["date"], sub["ma7"], label=f"{comm} (7-Day MA)", color=COMMODITY_COLORS[comm], lw=1.6)
    ax.set_title("2. 7-Day Moving Average Price Trends (Short-Term Smoothing)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (Rs. / kg)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_7day_moving_average.png"), dpi=150)
    plt.close(fig)

    # 3. 30-Day Moving Average
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].sort_values("date").copy()
        sub["ma30"] = sub["modal_price_per_kg"].rolling(window=30, min_periods=5).mean()
        ax.plot(sub["date"], sub["ma30"], label=f"{comm} (30-Day MA)", color=COMMODITY_COLORS[comm], lw=2.0)
    ax.set_title("3. 30-Day Moving Average Price Trends (Medium-Term Trend)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (Rs. / kg)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_30day_moving_average.png"), dpi=150)
    plt.close(fig)

    # 4. Monthly Average Price
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["year_month", "commodity"])["modal_price_per_kg"].mean().unstack()
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    monthly.plot(kind="bar", ax=ax, color=[COMMODITY_COLORS[c] for c in monthly.columns], width=0.8)
    ax.set_title("4. Monthly Average Modal Price (Bangalore)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Average Price (Rs. / kg)", fontsize=11)
    # Sample every 4th tick for readability
    ticks = ax.get_xticks()
    ax.set_xticks(ticks[::3])
    ax.set_xticklabels([monthly.index[i].strftime("%Y-%m") for i in ticks[::3]], rotation=45, ha="right")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_monthly_average_prices.png"), dpi=150)
    plt.close(fig)

    # 5. Price Distribution & Boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm]
        sns.kdeplot(sub["modal_price_per_kg"], ax=ax1, label=comm, color=COMMODITY_COLORS[comm], fill=True, alpha=0.3, lw=1.8)
    ax1.set_title("Price Density Distributions (KDE)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Modal Price (Rs. / kg)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(frameon=True, facecolor="white")

    sns.boxplot(data=df, x="commodity", y="modal_price_per_kg", ax=ax2, palette=COMMODITY_COLORS, width=0.5, fliersize=3)
    ax2.set_title("Price Dispersion & Outliers (Boxplot)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Commodity", fontsize=10)
    ax2.set_ylabel("Modal Price (Rs. / kg)", fontsize=10)

    plt.suptitle("5. Commodity Price Distributions and Dispersion", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_price_distributions.png"), dpi=150)
    plt.close(fig)

    # 6. Price Volatility (30-Day Rolling Standard Deviation)
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm].sort_values("date").copy()
        sub["rolling_std30"] = sub["modal_price_per_kg"].rolling(window=30, min_periods=5).std()
        ax.plot(sub["date"], sub["rolling_std30"], label=f"{comm} (30-Day Rolling Std)", color=COMMODITY_COLORS[comm], lw=1.6)
    ax.set_title("6. Price Volatility: 30-Day Rolling Standard Deviation", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price Standard Deviation (Rs. / kg)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_price_volatility.png"), dpi=150)
    plt.close(fig)

    # 7. Trading Days vs Calendar Continuity (Missing Data Analysis)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    summary_counts = []
    for comm in TARGET_COMMODITIES:
        sub = df[df["commodity"] == comm]
        total_span = (sub["date"].max() - sub["date"].min()).days + 1
        active_days = len(sub)
        closed_days = total_span - active_days
        summary_counts.append({
            "Commodity": comm,
            "Active Trading Days": active_days,
            "Market Closed / Non-Trading Days": closed_days,
        })
    df_counts = pd.DataFrame(summary_counts).set_index("Commodity")
    df_counts.plot(kind="bar", stacked=True, ax=ax, color=["#27ae60", "#bdc3c7"], width=0.45)
    ax.set_title("7. Mandi Trading Days vs Market Closures (Calendar Coverage)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Commodity", fontsize=10)
    ax.set_ylabel("Calendar Days", fontsize=10)
    ax.legend(frameon=True, facecolor="white")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_trading_days_coverage.png"), dpi=150)
    plt.close(fig)

    # 8. Seasonal Patterns (Monthly Climatology by Month of Year)
    df["month_num"] = df["date"].dt.month
    monthly_seasonality = df.groupby(["month_num", "commodity"])["modal_price_per_kg"].mean().unstack()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for comm in TARGET_COMMODITIES:
        if comm in monthly_seasonality.columns:
            ax.plot(range(1, 13), monthly_seasonality[comm], marker="o", label=comm, color=COMMODITY_COLORS[comm], lw=2.2)
    ax.set_title("8. Intra-Year Seasonal Price Pattern (Average Price by Calendar Month)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Month of Year", fontsize=10)
    ax.set_ylabel("Average Modal Price (Rs. / kg)", fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_intra_year_seasonality.png"), dpi=150)
    plt.close(fig)

    # 9. Cross-Commodity Correlation
    price_pivot = df.pivot(index="date", columns="commodity", values="modal_price_per_kg").dropna()
    corr = price_pivot.corr()
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson Correlation (r)"})
    ax.set_title("9. Cross-Commodity Price Correlation Matrix (Bangalore)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_cross_commodity_correlation.png"), dpi=150)
    plt.close(fig)

    print(f"Successfully generated all 9 EDA plots into: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_eda_plots()
