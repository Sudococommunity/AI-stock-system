"""
Utilities for ingesting Indian market historical data.

Primary mode:
  - try internet providers for fresh data
Fallback mode:
  - ingest local CSV history when the machine has no network access
"""
from dataclasses import dataclass
import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from stock_recommender.data.database import DatabaseManager

logger = logging.getLogger(__name__)

# ── Ticker constants for benchmark indices stored in the DB ───────────────────
NIFTY50_TICKER = "INDEX:NIFTY50"
SENSEX_TICKER = "INDEX:SENSEX"
INDIA_VIX_TICKER = "INDEX:INDIAVIX"

# Yahoo Finance symbols for the above
_YF_NIFTY50 = "^NSEI"
_YF_SENSEX = "^BSESN"
_YF_INDIAVIX = "^INDIAVIX"


REQUIRED_PRICE_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
COLUMN_ALIASES = {
    "timestamp": "date",
    "datetime": "date",
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "close",
    "Volume": "volume",
}


@dataclass
class IngestionResult:
    imported_stocks: int
    imported_rows: int
    skipped_files: List[str]

    def to_dict(self) -> Dict:
        return {
            "imported_stocks": self.imported_stocks,
            "imported_rows": self.imported_rows,
            "skipped_files": self.skipped_files,
        }


@dataclass
class CorporateActionSyncResult:
    synced_stocks: int
    inserted_actions: int
    skipped_symbols: List[str]

    def to_dict(self) -> Dict:
        return {
            "synced_stocks": self.synced_stocks,
            "inserted_actions": self.inserted_actions,
            "skipped_symbols": self.skipped_symbols,
        }


def ingest_indian_market_history(
    db: DatabaseManager,
    data_dir: str,
    metadata_path: Optional[str] = None,
    market_prefix: str = "NSE",
) -> IngestionResult:
    """
    Ingest local CSV history files into the project database.

    Expected shape:
    - one CSV per stock under `data_dir`
    - file name becomes ticker if metadata file is absent
    - metadata file, if supplied, may map ticker -> name/sector/market_cap/file
    """
    data_root = Path(data_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    metadata = _load_metadata(metadata_path)
    imported_stocks = 0
    imported_rows = 0
    skipped: List[str] = []

    for csv_path in sorted(data_root.rglob("*.csv")):
        rel_name = str(csv_path.relative_to(data_root))
        try:
            ticker, stock_meta = _resolve_ticker_and_meta(csv_path, metadata, market_prefix)
            df = _read_price_csv(csv_path)
            if df.empty:
                skipped.append(rel_name)
                continue

            stock_id = db.upsert_stock(
                ticker=ticker,
                name=stock_meta.get("name", ticker),
                sector=stock_meta.get("sector", ""),
                market_cap=float(stock_meta.get("market_cap", 0.0) or 0.0),
            )
            records = df.to_dict("records")
            db.insert_price_batch(stock_id, records)
            imported_stocks += 1
            imported_rows += len(records)
        except Exception:
            skipped.append(rel_name)

    return IngestionResult(
        imported_stocks=imported_stocks,
        imported_rows=imported_rows,
        skipped_files=skipped,
    )


def ingest_indian_market_dataset(
    db: DatabaseManager,
    source: str = "internet_or_csv",
    data_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    market_prefix: str = "NSE",
) -> IngestionResult:
    """
    Load Indian stock history from internet when possible, otherwise from local CSV.

    `source` values:
    - internet
    - csv
    - internet_or_csv
    """
    if source not in {"internet", "csv", "internet_or_csv"}:
        raise ValueError("source must be one of: internet, csv, internet_or_csv")

    internet_errors: List[str] = []
    if source in {"internet", "internet_or_csv"}:
        try:
            if not symbols:
                raise ValueError("symbols are required for internet ingestion")
            return ingest_from_yfinance(
                db=db,
                symbols=symbols,
                start=start,
                end=end,
                market_prefix=market_prefix,
            )
        except Exception as exc:
            if source == "internet":
                raise
            internet_errors.append(str(exc))

    if not data_dir:
        raise ValueError(
            "CSV fallback requires data_dir. Internet ingestion failed with: "
            + "; ".join(internet_errors)
        )
    return ingest_indian_market_history(
        db=db,
        data_dir=data_dir,
        metadata_path=metadata_path,
        market_prefix=market_prefix,
    )


def ingest_from_yfinance(
    db: DatabaseManager,
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    market_prefix: str = "NSE",
) -> IngestionResult:
    """
    Fetch historical bars from Yahoo Finance.
    Expected symbols for India: RELIANCE.NS, TCS.NS, INFY.NS, etc.
    """
    yf = importlib.import_module("yfinance")
    imported_stocks = 0
    imported_rows = 0
    skipped: List[str] = []

    for symbol in symbols:
        try:
            ticker = str(symbol).strip().upper()
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(start=start, end=end, auto_adjust=False)
            if hist is None or hist.empty:
                skipped.append(ticker)
                continue

            hist = hist.reset_index()
            if "Date" not in hist.columns and "Datetime" in hist.columns:
                hist["Date"] = hist["Datetime"]

            df = hist.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
            if df.empty:
                skipped.append(ticker)
                continue

            db_ticker = ticker if ":" in ticker else f"{market_prefix}:{ticker}"
            stock_id = db.upsert_stock(db_ticker, name=ticker)
            records = df.to_dict("records")
            db.insert_price_batch(stock_id, records)
            imported_stocks += 1
            imported_rows += len(records)
        except Exception:
            skipped.append(symbol)

    return IngestionResult(
        imported_stocks=imported_stocks,
        imported_rows=imported_rows,
        skipped_files=skipped,
    )


def sync_corporate_actions_from_rapidapi(
    db: DatabaseManager,
    stock_names: Optional[List[str]] = None,
    rapidapi_key: Optional[str] = None,
    host: str = "indian-stock-exchange-api2.p.rapidapi.com",
) -> CorporateActionSyncResult:
    """
    Fetch corporate actions via RapidAPI and store them locally.
    This enriches the training dataset but does not replace OHLCV ingestion.
    """
    rapidapi_key = rapidapi_key or os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key:
        raise ValueError("RapidAPI key not set. Use RAPIDAPI_KEY env var.")

    stocks = db.get_all_stocks()
    if stock_names:
        wanted = {name.strip().lower() for name in stock_names if name.strip()}
        stocks = [
            stock for stock in stocks
            if stock.get("name", "").strip().lower() in wanted
            or stock.get("ticker", "").split(":")[-1].strip().lower() in wanted
        ]

    synced_stocks = 0
    inserted_actions = 0
    skipped: List[str] = []

    for stock in stocks:
        stock_name = (stock.get("name") or stock.get("ticker", "")).split(":")[-1].strip()
        if not stock_name:
            skipped.append(str(stock.get("ticker", "")))
            continue
        try:
            payload = fetch_rapidapi_corporate_actions(stock_name, rapidapi_key=rapidapi_key, host=host)
            normalized = normalize_rapidapi_corporate_actions(payload)
            inserted_actions += db.insert_corporate_actions(stock["stock_id"], normalized)
            synced_stocks += 1
        except Exception:
            skipped.append(stock_name)

    return CorporateActionSyncResult(
        synced_stocks=synced_stocks,
        inserted_actions=inserted_actions,
        skipped_symbols=skipped,
    )


def fetch_rapidapi_corporate_actions(
    stock_name: str,
    rapidapi_key: str,
    host: str = "indian-stock-exchange-api2.p.rapidapi.com",
) -> Dict:
    params = urlencode({"stock_name": stock_name})
    url = f"https://{host}/corporate_actions?{params}"
    request = Request(
        url,
        headers={
            "Content-Type": "application/json",
            "x-rapidapi-host": host,
            "x-rapidapi-key": rapidapi_key,
        },
        method="GET",
    )
    with urlopen(request, timeout=20) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def normalize_rapidapi_corporate_actions(payload: Dict) -> List[Dict]:
    """
    Normalize several likely RapidAPI response shapes into a stable local schema.
    """
    raw_items = payload
    if isinstance(payload, dict):
        for key in ("data", "results", "corporate_actions", "actions"):
            if isinstance(payload.get(key), list):
                raw_items = payload[key]
                break

    if not isinstance(raw_items, list):
        return []

    normalized: List[Dict] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        action_date = (
            item.get("action_date")
            or item.get("date")
            or item.get("ex_date")
            or item.get("record_date")
        )
        action_type = (
            item.get("action_type")
            or item.get("type")
            or item.get("category")
            or "corporate_action"
        )
        title = (
            item.get("title")
            or item.get("purpose")
            or item.get("subject")
            or action_type
        )
        description = (
            item.get("description")
            or item.get("details")
            or item.get("message")
            or ""
        )
        normalized.append(
            {
                "action_date": str(action_date) if action_date is not None else None,
                "action_type": str(action_type),
                "title": str(title),
                "description": str(description),
                "source": "rapidapi:indian-stock-exchange-api2",
                "raw_payload": item,
            }
        )
    return normalized


def _load_metadata(metadata_path: Optional[str]) -> Dict[str, Dict]:
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return {str(k): dict(v) for k, v in raw.items()}
        raise ValueError("JSON metadata must be an object keyed by ticker")

    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Metadata CSV must include a 'ticker' column")
    return {
        str(row["ticker"]): {
            "name": row.get("name", ""),
            "sector": row.get("sector", ""),
            "market_cap": row.get("market_cap", 0.0),
            "file": row.get("file", ""),
        }
        for _, row in df.iterrows()
    }


def _resolve_ticker_and_meta(csv_path: Path, metadata: Dict[str, Dict], market_prefix: str) -> tuple[str, Dict]:
    bare_ticker = csv_path.stem.upper()
    prefix_token = f"{market_prefix.upper()}_"
    if bare_ticker.startswith(prefix_token):
        bare_ticker = bare_ticker[len(prefix_token):]
    stock_meta = metadata.get(bare_ticker, {})

    for ticker, meta in metadata.items():
        file_name = str(meta.get("file", "")).strip()
        if file_name and Path(file_name).name.lower() == csv_path.name.lower():
            bare_ticker = ticker.upper()
            stock_meta = meta
            break

    ticker = bare_ticker if ":" in bare_ticker else f"{market_prefix}:{bare_ticker}"
    return ticker, stock_meta


def ingest_india_vix(
    db: DatabaseManager,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> IngestionResult:
    """
    Fetch India VIX history from Yahoo Finance (^INDIAVIX) and store it in the DB.

    India VIX is NSE's fear gauge — a 30-day implied volatility index.
    It is stored as ticker INDEX:INDIAVIX so MarketRegimeAnalyzer._read_india_vix()
    picks it up automatically.

    Typical values:
      < 14  → calm market (low fear)
      14-20 → normal
      > 20  → elevated fear
    """
    yf = importlib.import_module("yfinance")
    ticker_obj = yf.Ticker(_YF_INDIAVIX)
    hist = ticker_obj.history(start=start, end=end, auto_adjust=False)
    if hist is None or hist.empty:
        logger.warning("[VIX] No India VIX history returned from Yahoo Finance")
        return IngestionResult(imported_stocks=0, imported_rows=0, skipped_files=[_YF_INDIAVIX])

    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else "Datetime"
    df = hist.rename(columns={date_col: "date", "Open": "open", "High": "high",
                               "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    # VIX has no volume — fill with 0
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0)
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

    stock_id = db.upsert_stock(
        ticker=INDIA_VIX_TICKER,
        name="India VIX",
        sector="Index",
        market_cap=0.0,
    )
    db.insert_price_batch(stock_id, df.to_dict("records"))
    logger.info("[VIX] Ingested %d India VIX rows (stock_id=%d)", len(df), stock_id)
    return IngestionResult(imported_stocks=1, imported_rows=len(df), skipped_files=[])


def ingest_benchmark_indices(
    db: DatabaseManager,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> IngestionResult:
    """
    Fetch Nifty 50 (^NSEI) and Sensex (^BSESN) history and store them in the DB.

    These are used as benchmarks for:
      - Beta / Jensen's Alpha vs actual Indian market
      - Nifty/Sensex correlation in RiskProfile
      - Market regime breadth signals

    Stored as:
      INDEX:NIFTY50  ← Nifty 50 (NSE benchmark)
      INDEX:SENSEX   ← BSE Sensex (BSE benchmark)
    """
    yf = importlib.import_module("yfinance")
    benchmarks = [
        (_YF_NIFTY50, NIFTY50_TICKER, "Nifty 50"),
        (_YF_SENSEX, SENSEX_TICKER, "BSE Sensex"),
    ]
    total_stocks = 0
    total_rows = 0
    skipped: List[str] = []

    for yf_symbol, db_ticker, display_name in benchmarks:
        try:
            ticker_obj = yf.Ticker(yf_symbol)
            hist = ticker_obj.history(start=start, end=end, auto_adjust=False)
            if hist is None or hist.empty:
                skipped.append(yf_symbol)
                continue

            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else "Datetime"
            df = hist.rename(columns={date_col: "date", "Open": "open", "High": "high",
                                       "Low": "low", "Close": "close", "Volume": "volume"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            if "volume" not in df.columns:
                df["volume"] = 0.0
            df["volume"] = df["volume"].fillna(0.0)
            df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])

            stock_id = db.upsert_stock(
                ticker=db_ticker,
                name=display_name,
                sector="Index",
                market_cap=0.0,
            )
            db.insert_price_batch(stock_id, df.to_dict("records"))
            total_stocks += 1
            total_rows += len(df)
            logger.info("[Benchmark] Ingested %d rows for %s (stock_id=%d)",
                        len(df), display_name, stock_id)
        except Exception as exc:
            logger.warning("[Benchmark] Failed to ingest %s: %s", yf_symbol, exc)
            skipped.append(yf_symbol)

    return IngestionResult(imported_stocks=total_stocks, imported_rows=total_rows,
                           skipped_files=skipped)


def apply_corporate_action_adjustments(
    db: DatabaseManager,
    stock_ids: Optional[List[int]] = None,
) -> Dict[str, int]:
    """
    Apply split and bonus corporate actions to historical price data.

    For each stock with a split/bonus event:
      1. Parse the ratio from the action title/description (e.g. "2:1 split" → ratio=2.0)
      2. Divide all OHLC prices (and multiply volume) *before* the ex-date by the ratio.
         Post-split prices are NOT changed (they are already on the adjusted scale).
      3. Update the price_history rows in-place.

    This ensures all historical indicators (moving averages, RSI, etc.) remain
    comparable across the split date.

    WARNING: This is a destructive update.  Run only once after fresh ingestion.
    To avoid double-adjusting, corporate_actions rows are marked as applied by
    setting source = source + ':adjusted'.

    Returns a dict with counts: {adjusted_stocks, adjusted_rows, skipped_stocks}
    """
    all_stocks = db.get_all_stocks() if stock_ids is None else [
        db.get_stock_info(sid) for sid in stock_ids if db.get_stock_info(sid)
    ]

    adjusted_stocks = 0
    adjusted_rows = 0
    skipped_stocks = 0

    for stock in all_stocks:
        sid = int(stock["stock_id"])
        try:
            actions = db.get_corporate_actions(sid)
        except Exception:
            skipped_stocks += 1
            continue

        # Filter to splits and bonuses that haven't been applied yet
        pending = [
            a for a in actions
            if a.get("action_type", "").lower() in {"split", "bonus", "stock split",
                                                      "sub-division", "subdivision"}
            and ":adjusted" not in str(a.get("source", ""))
        ]
        if not pending:
            continue

        history = db.get_price_history(sid, limit=10_000)
        if not history:
            continue

        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        changed = False

        for action in sorted(pending, key=lambda a: str(a.get("action_date") or ""), reverse=True):
            ex_date_str = action.get("action_date")
            if not ex_date_str:
                continue
            try:
                ex_date = pd.to_datetime(ex_date_str)
            except Exception:
                continue

            ratio = _parse_split_ratio(action.get("title", ""), action.get("description", ""))
            if ratio is None or ratio <= 0 or abs(ratio - 1.0) < 0.001:
                continue

            # Adjust all rows strictly before ex_date
            mask = df["date"] < ex_date
            if mask.sum() == 0:
                continue

            df.loc[mask, "open"] = (df.loc[mask, "open"] / ratio).round(4)
            df.loc[mask, "high"] = (df.loc[mask, "high"] / ratio).round(4)
            df.loc[mask, "low"] = (df.loc[mask, "low"] / ratio).round(4)
            df.loc[mask, "close"] = (df.loc[mask, "close"] / ratio).round(4)
            df.loc[mask, "volume"] = (df.loc[mask, "volume"] * ratio).round(0)
            changed = True
            adjusted_rows += int(mask.sum())
            logger.info("[CorpAdj] stock_id=%d  ex=%s  ratio=%.4f  rows=%d",
                        sid, ex_date_str, ratio, int(mask.sum()))

            # Mark the action as applied so we don't re-run it
            try:
                db.mark_corporate_action_applied(action["action_id"])
            except Exception:
                pass

        if changed:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            db.replace_price_history(sid, df.to_dict("records"))
            adjusted_stocks += 1

    return {
        "adjusted_stocks": adjusted_stocks,
        "adjusted_rows": adjusted_rows,
        "skipped_stocks": skipped_stocks,
    }


# ── helper for parsing split / bonus ratios ───────────────────────────────────

def _parse_split_ratio(title: str, description: str) -> Optional[float]:
    """
    Extract the split/bonus multiplier from an action title or description.

    Handles patterns:
      "2:1 split"       → 2.0
      "Split 5:1"       → 5.0
      "Bonus 1:1"       → 2.0  (1 extra share per existing = price halved)
      "Bonus 3:2"       → 2.5  (new_total = existing + 3/2 * existing)
      "Sub-division 10:1" → 10.0
    """
    text = f"{title} {description}".lower()

    # Match "N:M" or "N/M" ratio patterns
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)")
    for match in pattern.finditer(text):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator <= 0:
            continue
        ratio = numerator / denominator
        # For bonuses: "bonus 1:1" means 1 extra per 1 existing → ratio = 2.0
        if "bonus" in text:
            ratio = 1.0 + ratio
        if ratio >= 1.2:   # ignore trivial ratios (rounding noise)
            return ratio

    return None


def _read_price_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={col: COLUMN_ALIASES.get(col, col.lower()) for col in df.columns})
    missing = REQUIRED_PRICE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}")

    df = df[list(REQUIRED_PRICE_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df
