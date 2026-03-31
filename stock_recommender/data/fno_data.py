"""
F&O (Futures & Options) data ingestion for NSE stocks.

Fetches three key signals used by every serious Indian market analyst:

  1. Put-Call Ratio (PCR) — bearish when > 1.5, bullish when < 0.7
  2. Open Interest (OI) — rising OI with rising price = strong trend
  3. Delivery Percentage — high delivery pct = genuine buying, not speculation

Data sources (in priority order):
  A. NSE official bhav copy (free, no API key)
  B. RapidAPI Indian Stock Exchange API (requires RAPIDAPI_KEY)

Usage:
    from stock_recommender.data.fno_data import ingest_fno_snapshot
    result = ingest_fno_snapshot(db)  # ingest today's F&O snapshot for all stocks
"""
from __future__ import annotations

import csv
import importlib
import io
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from stock_recommender.data.database import DatabaseManager

logger = logging.getLogger(__name__)

# ── NSE public endpoints ──────────────────────────────────────────────────────
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
_NSE_FO_BHAV_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
_NSE_CM_BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
_NSE_PCR_URL     = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"


@dataclass
class FnoSnapshot:
    """F&O metrics for a single stock on a single date."""
    ticker: str
    snapshot_date: str
    pcr: Optional[float] = None            # put-call ratio (OI-based)
    oi_calls: Optional[float] = None       # total call OI (contracts)
    oi_puts: Optional[float] = None        # total put OI (contracts)
    total_oi: Optional[float] = None       # futures + options OI combined
    delivery_pct: Optional[float] = None   # delivery % of traded volume
    source: str = "nse_bhav"


@dataclass
class FnoIngestionResult:
    ingested_stocks: int = 0
    ingested_snapshots: int = 0
    skipped_symbols: List[str] = field(default_factory=list)
    source_used: str = "none"


# ── Public entry-point ────────────────────────────────────────────────────────

def ingest_fno_snapshot(
    db: DatabaseManager,
    snapshot_date: Optional[str] = None,
    rapidapi_key: Optional[str] = None,
    rapidapi_host: str = "indian-stock-exchange-api2.p.rapidapi.com",
) -> FnoIngestionResult:
    """
    Ingest one day's F&O snapshot for all stocks in the DB.

    Pipeline:
      1. Try NSE bhav copy (free).
      2. Fall back to RapidAPI if bhav copy fails and a key is available.
      3. Store each snapshot via db.upsert_fno_snapshot().

    Args:
        db:            DatabaseManager instance.
        snapshot_date: ISO date string (YYYY-MM-DD).  Defaults to yesterday.
        rapidapi_key:  Optional RapidAPI key; also read from RAPIDAPI_KEY env var.
        rapidapi_host: RapidAPI host for Indian stock exchange API.
    """
    if snapshot_date is None:
        snapshot_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    all_stocks = {
        s["ticker"].split(":")[-1].upper(): s["stock_id"]
        for s in db.get_all_stocks()
        if not s["ticker"].startswith("INDEX:")
    }

    snapshots: List[FnoSnapshot] = []
    source_used = "none"

    # ── Attempt 1: NSE bhav copy ──────────────────────────────────────────────
    try:
        snapshots = _fetch_nse_bhav(snapshot_date, set(all_stocks.keys()))
        source_used = "nse_bhav"
        logger.info("[F&O] NSE bhav copy: fetched %d snapshots for %s",
                    len(snapshots), snapshot_date)
    except Exception as exc:
        logger.warning("[F&O] NSE bhav fetch failed: %s", exc)

    # ── Attempt 2: RapidAPI fallback ──────────────────────────────────────────
    if not snapshots:
        api_key = rapidapi_key or os.getenv("RAPIDAPI_KEY")
        if api_key:
            try:
                snapshots = _fetch_rapidapi_fno(
                    list(all_stocks.keys()), snapshot_date, api_key, rapidapi_host
                )
                source_used = "rapidapi"
                logger.info("[F&O] RapidAPI: fetched %d snapshots", len(snapshots))
            except Exception as exc:
                logger.warning("[F&O] RapidAPI F&O fetch failed: %s", exc)

    if not snapshots:
        logger.warning("[F&O] No F&O data available for %s", snapshot_date)
        return FnoIngestionResult(source_used=source_used)

    # ── Persist snapshots ─────────────────────────────────────────────────────
    result = FnoIngestionResult(source_used=source_used)
    for snap in snapshots:
        bare_ticker = snap.ticker.split(":")[-1].upper()
        stock_id = all_stocks.get(bare_ticker)
        if stock_id is None:
            result.skipped_symbols.append(snap.ticker)
            continue
        try:
            db.upsert_fno_snapshot(
                stock_id=stock_id,
                snapshot_date=snap.snapshot_date,
                pcr=snap.pcr,
                oi_calls=snap.oi_calls,
                oi_puts=snap.oi_puts,
                total_oi=snap.total_oi,
                delivery_pct=snap.delivery_pct,
            )
            result.ingested_snapshots += 1
            result.ingested_stocks += 1
        except Exception as exc:
            logger.warning("[F&O] Failed to persist %s: %s", snap.ticker, exc)
            result.skipped_symbols.append(snap.ticker)

    return result


def get_fno_features(db: DatabaseManager, stock_id: int, lookback: int = 20) -> Dict:
    """
    Return a dict of recent F&O metrics for a stock, suitable for use
    as extra features in ranking or risk scoring.

    Keys:
        pcr_latest, pcr_mean, pcr_trend,
        oi_trend (recent OI growth rate),
        delivery_pct_latest, delivery_pct_mean
    """
    rows = db.get_fno_snapshots(stock_id, limit=lookback)
    if not rows:
        return {}

    import numpy as np

    pcrs = [r["pcr"] for r in rows if r["pcr"] is not None]
    ois  = [r["total_oi"] for r in rows if r["total_oi"] is not None]
    dels = [r["delivery_pct"] for r in rows if r["delivery_pct"] is not None]

    features: Dict = {}
    if pcrs:
        features["pcr_latest"] = float(pcrs[-1])
        features["pcr_mean"]   = float(np.mean(pcrs))
        # Positive trend → puts being added (bearish signal)
        features["pcr_trend"]  = float(pcrs[-1] - pcrs[0]) if len(pcrs) > 1 else 0.0
    if ois and len(ois) >= 2:
        features["oi_trend"] = float((ois[-1] - ois[0]) / max(abs(ois[0]), 1.0))
    if dels:
        features["delivery_pct_latest"] = float(dels[-1])
        features["delivery_pct_mean"]   = float(np.mean(dels))

    return features


# ── NSE bhav copy parser ──────────────────────────────────────────────────────

def _fetch_nse_bhav(date_str: str, wanted_symbols: set) -> List[FnoSnapshot]:
    """
    Download NSE's combined F&O + cash bhav copies and extract OI / delivery data.

    NSE bhav copies are public files served from nsearchives.nseindia.com.
    The F&O bhav has option OI columns; the CM (cash-market) bhav has delivery %.
    """
    import zipfile

    # Date format for NSE archives: DDMMYYYY
    d = datetime.strptime(date_str, "%Y-%m-%d")
    date_ddmmyyyy = d.strftime("%d%m%Y")

    # ── Cash market bhav (delivery %) ────────────────────────────────────────
    cm_url = _NSE_CM_BHAV_URL.format(date=date_ddmmyyyy)
    delivery_map: Dict[str, float] = {}
    try:
        req = Request(cm_url, headers=_NSE_HEADERS)
        with urlopen(req, timeout=20) as resp:
            cm_csv = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(cm_csv))
        for row in reader:
            sym = str(row.get("SYMBOL", "") or row.get("Symbol", "")).strip().upper()
            tot_trd_qty = float(row.get("TTL_TRD_QNTY", 0) or 0)
            dlv_qty = float(row.get("DELIV_QTY", 0) or 0)
            if sym and tot_trd_qty > 0:
                delivery_map[sym] = round(100.0 * dlv_qty / tot_trd_qty, 2)
    except Exception as exc:
        logger.debug("[F&O bhav] CM bhav unavailable: %s", exc)

    # ── F&O bhav (option OI) ─────────────────────────────────────────────────
    fo_url = _NSE_FO_BHAV_URL.format(date=date_ddmmyyyy)
    call_oi: Dict[str, float] = {}
    put_oi: Dict[str, float] = {}
    futures_oi: Dict[str, float] = {}
    try:
        req = Request(fo_url, headers=_NSE_HEADERS)
        with urlopen(req, timeout=30) as resp:
            zip_data = io.BytesIO(resp.read())
        with zipfile.ZipFile(zip_data) as zf:
            csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            if csv_name:
                fo_csv = zf.read(csv_name).decode("utf-8")
                reader = csv.DictReader(io.StringIO(fo_csv))
                for row in reader:
                    sym = str(row.get("SYMBOL", "") or "").strip().upper()
                    inst = str(row.get("INSTRUMENT", "") or "").strip().upper()
                    try:
                        oi_val = float(row.get("OPEN_INT", 0) or 0)
                    except ValueError:
                        continue
                    if "CE" in inst:
                        call_oi[sym] = call_oi.get(sym, 0.0) + oi_val
                    elif "PE" in inst:
                        put_oi[sym] = put_oi.get(sym, 0.0) + oi_val
                    elif "FUT" in inst:
                        futures_oi[sym] = futures_oi.get(sym, 0.0) + oi_val
    except Exception as exc:
        logger.debug("[F&O bhav] F&O bhav unavailable: %s", exc)

    all_syms = (set(call_oi) | set(put_oi) | set(futures_oi) | set(delivery_map)) & wanted_symbols
    snapshots: List[FnoSnapshot] = []

    for sym in all_syms:
        c_oi = call_oi.get(sym)
        p_oi = put_oi.get(sym)
        f_oi = futures_oi.get(sym, 0.0)
        pcr = round(p_oi / c_oi, 4) if (c_oi and p_oi and c_oi > 0) else None
        total = (c_oi or 0) + (p_oi or 0) + f_oi or None
        snapshots.append(
            FnoSnapshot(
                ticker=sym,
                snapshot_date=date_str,
                pcr=pcr,
                oi_calls=c_oi,
                oi_puts=p_oi,
                total_oi=total,
                delivery_pct=delivery_map.get(sym),
                source="nse_bhav",
            )
        )

    return snapshots


# ── RapidAPI fallback ─────────────────────────────────────────────────────────

def _fetch_rapidapi_fno(
    symbols: List[str],
    date_str: str,
    api_key: str,
    host: str,
) -> List[FnoSnapshot]:
    """Fetch F&O data via RapidAPI for a list of NSE symbols."""
    snapshots: List[FnoSnapshot] = []
    for sym in symbols[:50]:   # RapidAPI free tier: rate-limit guard
        try:
            params = urlencode({"stock_name": sym})
            url = f"https://{host}/fno_data?{params}"
            req = Request(url, headers={
                "x-rapidapi-host": host,
                "x-rapidapi-key": api_key,
                "Content-Type": "application/json",
            })
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if not isinstance(data, dict):
                continue
            snapshots.append(
                FnoSnapshot(
                    ticker=sym,
                    snapshot_date=date_str,
                    pcr=_safe_float(data.get("pcr") or data.get("put_call_ratio")),
                    oi_calls=_safe_float(data.get("call_oi") or data.get("oi_calls")),
                    oi_puts=_safe_float(data.get("put_oi") or data.get("oi_puts")),
                    total_oi=_safe_float(data.get("total_oi") or data.get("open_interest")),
                    delivery_pct=_safe_float(data.get("delivery_pct") or data.get("delivery_percentage")),
                    source="rapidapi",
                )
            )
        except Exception:
            continue
    return snapshots


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
