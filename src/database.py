"""
AgroVision Database Module (PostgreSQL + SQLAlchemy).
Handles connection pooling, schema initialization, ORM models, time-series querying,
prediction ledger persistence, and seamless fallback to CSV files when offline.
"""

import os
import sys
import time
import logging
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple, Generator
from contextlib import contextmanager

import pandas as pd
import numpy as np
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    Boolean,
    Date,
    DateTime,
    Text,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    DATABASE_URL,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    PREDICTION_LEDGER_FILE,
    PROCESSED_DATA_FILE,
    FEATURE_DATA_FILE,
    RAW_DATA_FILE,
    TRACKING_METRICS_FILE,
)

logger = logging.getLogger("AgroVisionDB")

Base = declarative_base()


# =====================================================================
# SQLAlchemy ORM Models
# =====================================================================

class PredictionLedgerModel(Base):
    """
    Permanent Append-Only & Reconciliation Ledger for daily commodity forecasts.
    Tracks predictions vs out-of-sample ground-truth actuals.
    """
    __tablename__ = "prediction_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(64), unique=True, index=True, nullable=False)
    date_predicted = Column(Date, index=True, nullable=False)
    target_date = Column(Date, index=True, nullable=False)
    commodity = Column(String(50), index=True, nullable=False)
    market = Column(String(100), default="Bangalore")
    
    # Wholesale Metrics
    current_wholesale_price = Column(Float, nullable=True)
    predicted_wholesale_price = Column(Float, nullable=False)
    predicted_wholesale_delta = Column(Float, nullable=True)
    actual_wholesale_price = Column(Float, nullable=True)
    wholesale_error_rs = Column(Float, nullable=True)
    wholesale_pct_error = Column(Float, nullable=True)
    
    # Retail Metrics
    current_retail_price = Column(Float, nullable=True)
    predicted_retail_price = Column(Float, nullable=False)
    predicted_retail_delta = Column(Float, nullable=True)
    actual_retail_price = Column(Float, nullable=True)
    actual_retail_delta = Column(Float, nullable=True)
    retail_error_rs = Column(Float, nullable=True)
    retail_pct_error = Column(Float, nullable=True)
    
    # Intelligence & Attribution
    predicted_trend = Column(String(20), nullable=True)
    actual_trend = Column(String(20), nullable=True)
    spike_risk_pct = Column(Float, default=0.0)
    primary_driver = Column(String(255), nullable=True)
    
    # Evaluation Flags
    is_direction_correct = Column(Boolean, nullable=True)
    is_within_5pct = Column(Boolean, nullable=True)
    is_within_10pct = Column(Boolean, nullable=True)
    is_within_2rs = Column(Boolean, nullable=True)
    
    # Lifecycle Status: PENDING, VERIFIED, EXPIRED
    status = Column(String(20), default="PENDING", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_ledger_target_commodity", "target_date", "commodity"),
        Index("idx_ledger_status_target", "status", "target_date"),
    )


class ProcessedPriceModel(Base):
    """
    Cleaned daily historical & live prices for Bangalore APMC mandi and retail benchmarks.
    """
    __tablename__ = "processed_prices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True, nullable=False)
    commodity = Column(String(50), index=True, nullable=False)
    market = Column(String(100), default="Bangalore", index=True, nullable=False)
    
    wholesale_price = Column(Float, nullable=False)
    retail_price = Column(Float, nullable=True)
    modal_price = Column(Float, nullable=True)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    arrivals_tonnes = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("date", "commodity", "market", name="uq_date_commodity_market"),
        Index("idx_processed_date_commodity", "date", "commodity"),
    )


class EconomicFactorModel(Base):
    """
    Daily agrometeorology, fuel, transport logistics, and government policy indicators.
    """
    __tablename__ = "economic_factors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, unique=True, index=True, nullable=False)
    
    # Agrometeorology (Producing Hubs)
    kolar_rain_1d = Column(Float, default=0.0)
    kolar_rain_3d = Column(Float, default=0.0)
    kolar_rain_7d = Column(Float, default=0.0)
    kolar_temp_max = Column(Float, nullable=True)
    
    nashik_rain_1d = Column(Float, default=0.0)
    nashik_rain_3d = Column(Float, default=0.0)
    nashik_rain_7d = Column(Float, default=0.0)
    
    hassan_rain_1d = Column(Float, default=0.0)
    hassan_rain_3d = Column(Float, default=0.0)
    hassan_rain_7d = Column(Float, default=0.0)
    
    # Fuel & Transport
    karnataka_diesel_price = Column(Float, nullable=True)
    freight_cost_index = Column(Float, nullable=True)
    
    # Policy & Buffer Stock
    buffer_stock_release_flag = Column(Boolean, default=False)
    export_ban_flag = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RawMandiArrivalModel(Base):
    """
    Raw ingested records from AGMARKNET / data.gov.in.
    """
    __tablename__ = "raw_mandi_arrivals"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    arrival_date = Column(Date, index=True, nullable=False)
    state = Column(String(50), default="Karnataka")
    district = Column(String(100), default="Bangalore")
    market = Column(String(100), nullable=False)
    commodity = Column(String(50), index=True, nullable=False)
    variety = Column(String(100), nullable=True)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    modal_price = Column(Float, nullable=True)
    raw_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyRetrainingLogModel(Base):
    """
    Audit log of daily continuous model retraining events and metric evolutions.
    """
    __tablename__ = "daily_retraining_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    retrained_at = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(20), default="SUCCESS")
    samples_count = Column(Integer, nullable=False)
    tomato_mae = Column(Float, nullable=True)
    onion_mae = Column(Float, nullable=True)
    potato_mae = Column(Float, nullable=True)
    directional_accuracy = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    details = Column(Text, nullable=True)


# =====================================================================
# Connection Management & Engine Setup
# =====================================================================

_engine = None
_SessionFactory = None
_db_checked = False
_db_available = False


def get_engine():
    """Returns or lazily initializes the SQLAlchemy database engine."""
    global _engine, _SessionFactory
    if _engine is None:
        try:
            # Create engine with connection pooling and fast connection timeout
            _engine = create_engine(
                DATABASE_URL,
                pool_size=10,
                max_overflow=20,
                pool_timeout=5,
                pool_recycle=1800,
                connect_args={"connect_timeout": 3},
            )
            _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
        except Exception as e:
            logger.warning(f"Could not create SQLAlchemy engine: {e}")
            _engine = None
            _SessionFactory = None
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        get_engine()
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    factory = get_session_factory()
    if factory is None:
        raise RuntimeError("Database session factory unavailable.")
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_db_connection() -> Dict[str, Any]:
    """
    Tests the PostgreSQL database connection and returns health telemetry.
    """
    global _db_available, _db_checked
    engine = get_engine()
    if engine is None:
        _db_available = False
        _db_checked = True
        return {
            "connected": False,
            "engine": "PostgreSQL",
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "error": "Engine initialization failed",
            "latency_ms": None,
            "fallback_active": True,
        }

    start_time = time.perf_counter()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        _db_available = True
        _db_checked = True
        return {
            "connected": True,
            "engine": "PostgreSQL",
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "user": DB_USER,
            "latency_ms": latency_ms,
            "fallback_active": False,
            "error": None,
        }
    except Exception as e:
        _db_available = False
        _db_checked = True
        return {
            "connected": False,
            "engine": "PostgreSQL",
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "error": str(e),
            "latency_ms": None,
            "fallback_active": True,
        }


def is_database_connected() -> bool:
    """Quick boolean check if PostgreSQL is currently accessible."""
    status = check_db_connection()
    return status.get("connected", False)


def init_db() -> bool:
    """
    Creates all database tables if they do not exist.
    """
    engine = get_engine()
    if engine is None:
        logger.warning("Database engine is not configured.")
        return False
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Successfully initialized PostgreSQL database schema.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL schema: {e}")
        return False


# =====================================================================
# Database High-Level Data Accessors with Fallback
# =====================================================================

def get_ledger_records(status_filter: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
    """
    Retrieves prediction ledger records from PostgreSQL, or falls back to CSV.
    """
    if is_database_connected():
        try:
            with get_engine().connect() as conn:
                query = "SELECT * FROM prediction_ledger"
                params = {}
                if status_filter and status_filter.upper() != "ALL":
                    query += " WHERE status = :status"
                    params["status"] = status_filter.upper()
                query += " ORDER BY target_date DESC, id DESC LIMIT :limit"
                params["limit"] = limit
                df = pd.read_sql(text(query), conn, params=params)
                if not df.empty and "id" in df.columns:
                    df = df.drop(columns=["id", "created_at", "updated_at"], errors="ignore")
                return df
        except Exception as e:
            logger.warning(f"PostgreSQL ledger query failed, falling back to CSV: {e}")

    # Fallback to CSV
    if os.path.exists(PREDICTION_LEDGER_FILE):
        try:
            df = pd.read_csv(PREDICTION_LEDGER_FILE)
            if status_filter and status_filter.upper() != "ALL" and "status" in df.columns:
                df = df[df["status"] == status_filter.upper()]
            return df
        except Exception as e:
            logger.error(f"Failed to read ledger CSV: {e}")
    return pd.DataFrame()


def save_prediction_record(record: Dict[str, Any]) -> bool:
    """
    Saves a prediction record to PostgreSQL if connected.
    """
    if not is_database_connected():
        return False

    try:
        with get_db_session() as session:
            # Check if prediction_id already exists
            pred_id = str(record.get("prediction_id", ""))
            bool_cols = {"is_direction_correct", "is_within_5pct", "is_within_10pct", "is_within_2rs"}
            def _clean_val(k, v):
                if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).lower() in ("nan", "none", ""):
                    return None
                if k in bool_cols:
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    if s in ("true", "1", "1.0", "t"):
                        return True
                    if s in ("false", "0", "0.0", "f"):
                        return False
                    return bool(v)
                if isinstance(v, (np.integer, np.int64)):
                    return int(v)
                if isinstance(v, (np.floating, np.float64)):
                    return float(v)
                if isinstance(v, np.bool_):
                    return bool(v)
                return v

            existing = session.query(PredictionLedgerModel).filter_by(prediction_id=pred_id).first()
            if existing:
                for k, v in record.items():
                    if hasattr(existing, k):
                        val = _clean_val(k, v)
                        if val is not None:
                            setattr(existing, k, val)
                existing.updated_at = datetime.utcnow()
            else:
                clean_rec = {}
                for k, v in record.items():
                    if hasattr(PredictionLedgerModel, k):
                        clean_rec[k] = _clean_val(k, v)
                ledger_obj = PredictionLedgerModel(**clean_rec)
                session.add(ledger_obj)
        return True
    except Exception as e:
        logger.warning(f"Could not persist prediction to PostgreSQL: {e}")
        return False


def get_db_telemetry_stats() -> Dict[str, Any]:
    """
    Returns record counts across all PostgreSQL tables and general statistics.
    """
    status = check_db_connection()
    if not status["connected"]:
        # Count rows in local CSVs for status reporting
        ledger_count = 0
        processed_count = 0
        if os.path.exists(PREDICTION_LEDGER_FILE):
            try:
                ledger_count = len(pd.read_csv(PREDICTION_LEDGER_FILE))
            except Exception:
                pass
        if os.path.exists(PROCESSED_DATA_FILE):
            try:
                processed_count = len(pd.read_csv(PROCESSED_DATA_FILE))
            except Exception:
                pass

        return {
            "status": "offline_fallback",
            "engine": "Local CSV Store",
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "tables": {
                "prediction_ledger": {"rows": ledger_count, "storage": "CSV"},
                "processed_prices": {"rows": processed_count, "storage": "CSV"},
            },
            "connected": False,
            "message": status.get("error", "Database offline"),
        }

    try:
        engine = get_engine()
        stats = {}
        with engine.connect() as conn:
            for table_name in ["prediction_ledger", "processed_prices", "economic_factors", "raw_mandi_arrivals", "daily_retraining_log"]:
                try:
                    res = conn.execute(text(f"SELECT count(*) FROM {table_name}"))
                    count = res.scalar() or 0
                    stats[table_name] = {"rows": count, "storage": "PostgreSQL"}
                except Exception:
                    stats[table_name] = {"rows": 0, "storage": "PostgreSQL (uninitialized)"}

        return {
            "status": "connected",
            "engine": "PostgreSQL 14+",
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
            "user": DB_USER,
            "latency_ms": status.get("latency_ms"),
            "connected": True,
            "tables": stats,
            "message": "PostgreSQL active and synchronized",
        }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "message": str(e),
        }


def save_daily_retrain_event(event: Dict[str, Any]) -> bool:
    """
    Persists an automated model retraining audit log to PostgreSQL.
    """
    if not is_database_connected():
        return False
    try:
        with get_db_session() as session:
            obj = DailyRetrainingLogModel(
                retrained_at=datetime.utcnow(),
                status=event.get("status", "SUCCESS"),
                samples_count=int(event.get("samples_count", 0)),
                tomato_mae=float(event.get("tomato_mae", 0.0)),
                onion_mae=float(event.get("onion_mae", 0.0)),
                potato_mae=float(event.get("potato_mae", 0.0)),
                directional_accuracy=float(event.get("directional_accuracy", 0.0)),
                reason=str(event.get("reason", "")),
                details=json.dumps(event.get("details", {})) if isinstance(event.get("details"), dict) else str(event.get("details", ""))
            )
            session.add(obj)
        return True
    except Exception as e:
        logger.warning(f"Could not persist retraining log to PostgreSQL: {e}")
        return False
