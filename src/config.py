import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRACKING_DIR = DATA_DIR / "tracking"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure essential directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Data.gov.in API Configuration
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "")
DATA_GOV_RESOURCE_ID = os.getenv(
    "DATA_GOV_RESOURCE_ID", "9ef84268-d588-465a-a308-a864a43d0070"
)
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource"

# Target Commodities & Location
TARGET_COMMODITIES = ["Tomato", "Onion", "Potato"]
TARGET_STATE = "Karnataka"
TARGET_DISTRICT_KEYWORDS = ["Bangalore", "Bengaluru"]

# File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "bangalore_grocery_prices_raw.csv"
RAW_METADATA_FILE = RAW_DATA_DIR / "api_response_metadata.json"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "bangalore_grocery_prices_processed.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "bangalore_grocery_features.csv"
PREDICTION_LEDGER_FILE = TRACKING_DIR / "prediction_ledger.csv"
TRACKING_METRICS_FILE = TRACKING_DIR / "tracking_metrics.json"
DASHBOARD_HTML_FILE = REPORTS_DIR / "dashboard.html"
RETRAINING_LOG_FILE = MODELS_DIR / "retraining_log.json"

# Database Configuration (PostgreSQL)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "agrovision")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if DB_PASSWORD:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        DATABASE_URL = f"postgresql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
