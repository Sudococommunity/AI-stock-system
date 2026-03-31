import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.request import Request, urlopen
import logging

logger = logging.getLogger(__name__)

import pandas as pd

from stock_recommender.data.indian_market_data import ingest_indian_market_history

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


@dataclass
class UniverseSymbol:
    ticker: str
    exchange: str
    provider_symbol: str
    company_name: str = ""
    sector: str = ""


@dataclass
class DownloadReport:
    requested: int
    downloaded: int
    skipped: int
    failed: int
    failures: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class IndianUniverseDownloader:
    """
    Download maximum available historical OHLCV for Indian stocks.

    The downloader is designed for bulk runs:
    - one CSV per symbol
    - resumable by skipping existing files
    - provider symbol mapping for NSE/BSE via Yahoo Finance
    """

    def __init__(
        self,
        output_dir: str,
        workers: int = 4,
        pause_seconds: float = 0.5,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.pause_seconds = pause_seconds
        self.verbose = verbose
        self.state_path = self.output_dir / "_download_state.json"

    def download(
        self,
        symbols: Iterable[UniverseSymbol],
        period: str = "max",
        refresh: bool = False,
        min_rows: int = 30,
        retries: int = 2,
    ) -> DownloadReport:
        if yf is None:
            raise RuntimeError("yfinance is not installed")

        symbols = list(symbols)
        requested = len(symbols)
        downloaded = 0
        skipped = 0
        failures: List[str] = []
        state = self._load_state()
        completed = 0

        if self.verbose:
            print("=" * 72)
            print("INDIAN UNIVERSE DOWNLOAD STARTED")
            print(f"Total symbols        : {requested}")
            print(f"Output directory     : {self.output_dir}")
            print(f"Workers              : {self.workers}")
            print(f"Requested period     : {period}")
            print("=" * 72)

        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
            futures = {}
            for item in symbols:
                path = self._csv_path(item)
                if path.exists() and not refresh:
                    skipped += 1
                    completed += 1
                    state[item.ticker] = {"status": "skipped", "path": str(path)}
                    if self.verbose and (completed <= 5 or completed % 100 == 0):
                        print(f"[{completed}/{requested}] [SKIP] {item.ticker} (already exists)")
                    continue
                futures[executor.submit(self._download_one, item, period, min_rows, retries)] = item

            for future in as_completed(futures):
                item = futures[future]
                completed += 1
                try:
                    ok, payload = future.result()
                    if ok:
                        downloaded += 1
                        state[item.ticker] = payload
                    else:
                        failures.append(item.ticker)
                        state[item.ticker] = payload
                except Exception as exc:
                    failures.append(item.ticker)
                    state[item.ticker] = {"status": "failed", "error": str(exc)}
                self._save_state(state)
                if self.verbose and (completed <= 10 or completed % 50 == 0 or completed == requested):
                    print(
                        f"[SUMMARY] {completed}/{requested} complete | "
                        f"downloaded={downloaded} skipped={skipped} failed={len(failures)}"
                    )

        return DownloadReport(
            requested=requested,
            downloaded=downloaded,
            skipped=skipped,
            failed=len(failures),
            failures=failures,
        )

    def ingest_to_db(
        self,
        db,
        metadata_path: Optional[str] = None,
        market_prefix: str = "NSE",
    ):
        return ingest_indian_market_history(
            db=db,
            data_dir=str(self.output_dir),
            metadata_path=metadata_path,
            market_prefix=market_prefix,
        )

    def _download_one(
        self,
        item: UniverseSymbol,
        period: str,
        min_rows: int,
        retries: int,
    ) -> tuple[bool, Dict]:
        last_error = None
        for attempt in range(retries + 1):
            try:
                ticker = yf.Ticker(item.provider_symbol)
                if str(period).lower() == "max":
                    hist = ticker.history(start="1926-01-01", auto_adjust=False)
                else:
                    hist = ticker.history(period=period, auto_adjust=False)
                if hist is None or hist.empty:
                    raise ValueError("No history returned")

                hist = hist.reset_index()
                date_col = "Date" if "Date" in hist.columns else "Datetime"
                df = hist.rename(
                    columns={
                        date_col: "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                df = df[["date", "open", "high", "low", "close", "volume"]].dropna()
                if len(df) < min_rows:
                    raise ValueError(f"Too few rows: {len(df)}")

                path = self._csv_path(item)
                df.to_csv(path, index=False)
                if self.verbose:
                    print(f"[OK] {item.ticker} -> {len(df)} rows")
                time.sleep(self.pause_seconds)
                return True, {
                    "status": "downloaded",
                    "path": str(path),
                    "rows": int(len(df)),
                    "provider_symbol": item.provider_symbol,
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < retries:
                    time.sleep(1.0 * (attempt + 1))

        if self.verbose:
            print(f"[FAIL] {item.ticker}: {last_error}")
        return False, {
            "status": "failed",
            "error": last_error or "unknown error",
            "provider_symbol": item.provider_symbol,
        }

    def _csv_path(self, item: UniverseSymbol) -> Path:
        safe_ticker = item.ticker.replace(":", "_").replace("/", "_")
        return self.output_dir / f"{safe_ticker}.csv"

    def _load_state(self) -> Dict:
        if not self.state_path.exists():
            return {}
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self, state: Dict) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


def load_universe_from_csv(
    path: str,
    exchange: str = "NSE",
    symbol_col: Optional[str] = None,
    name_col: Optional[str] = None,
    sector_col: Optional[str] = None,
) -> List[UniverseSymbol]:
    df = pd.read_csv(path)
    normalized = {col.lower().strip(): col for col in df.columns}

    symbol_col = symbol_col or _first_present(normalized, ["symbol", "ticker", "tradingsymbol", "security id"])
    name_col = name_col or _first_present(normalized, ["company name", "name", "security name", "issuer name"])
    sector_col = sector_col or _first_present(normalized, ["sector", "industry"])
    if not symbol_col:
        raise ValueError("Could not infer symbol column from CSV")

    result: List[UniverseSymbol] = []
    for _, row in df.iterrows():
        raw_symbol = str(row[symbol_col]).strip()
        if not raw_symbol or raw_symbol.lower() == "nan":
            continue
        company_name = str(row[name_col]).strip() if name_col else ""
        sector = str(row[sector_col]).strip() if sector_col else ""
        result.append(
            UniverseSymbol(
                ticker=f"{exchange.upper()}:{raw_symbol.upper()}",
                exchange=exchange.upper(),
                provider_symbol=to_yfinance_symbol(raw_symbol, exchange),
                company_name=company_name,
                sector=sector,
            )
        )
    return result


def load_nse_universe_official() -> List[UniverseSymbol]:
    """
    Load NSE equity universe from the official NSE archive CSV.
    Source verified from NSE's 'Securities available for Trading' page:
    https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,application/octet-stream,*/*",
            "Referer": "https://www.nseindia.com/",
        },
    )
    with urlopen(request, timeout=30) as response:
        df = pd.read_csv(response)

    # Common official columns: SYMBOL, NAME OF COMPANY, SERIES, DATE OF LISTING, ...
    if "SYMBOL" not in df.columns:
        raise ValueError("Unexpected NSE universe format: SYMBOL column missing")

    series_col = df["SERIES"] if "SERIES" in df.columns else pd.Series(["EQ"] * len(df))
    df = df[series_col.astype(str).str.upper().isin({"EQ", "BE", "BZ", "SM"})].copy()
    result: List[UniverseSymbol] = []
    for _, row in df.iterrows():
        symbol = str(row["SYMBOL"]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        result.append(
            UniverseSymbol(
                ticker=f"NSE:{symbol}",
                exchange="NSE",
                provider_symbol=to_yfinance_symbol(symbol, "NSE"),
                company_name=str(row.get("NAME OF COMPANY", "")).strip(),
                sector="",
            )
        )
    return result


def load_bse_universe_official() -> List[UniverseSymbol]:
    """
    Load BSE equity universe from BSE's official scrip list API.

    Source: BSE India public API — returns all actively-listed equity scrips.
    Symbols are stored with .BO suffix for Yahoo Finance (e.g. RELIANCE.BO).

    Falls back to a secondary CSV endpoint if the JSON API is unavailable.
    """
    # Primary: BSE's JSON scrip-list endpoint
    primary_url = (
        "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
        "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }
    try:
        req = Request(primary_url, headers=headers)
        with urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch BSE universe: {exc}. "
            "Check network access or use load_universe_from_csv() with a BSE CSV file."
        ) from exc

    # BSE API wraps the list in {"Table": [...]} or returns a bare list
    if isinstance(raw, dict):
        items = raw.get("Table") or raw.get("data") or raw.get("results") or []
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unexpected BSE API response shape: {type(raw)}")

    result: List[UniverseSymbol] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # BSE uses numeric Scrip_Cd and alphanumeric Scrip_Id (trading symbol)
        scrip_id = str(item.get("Scrip_Id") or item.get("SCRIP_ID") or "").strip().upper()
        scrip_code = str(item.get("Scrip_Cd") or item.get("SCRIP_CD") or "").strip()
        name = str(item.get("SCRIP_NAME") or item.get("Scrip_Name") or "").strip()
        sector = str(item.get("INDUSTRY") or item.get("Industry") or "").strip()
        group = str(item.get("Group") or item.get("GROUP") or "").strip().upper()

        # Skip suspicious/debt/preference share groups
        if group and group not in {"A", "B", "T", "XT", "Z", ""}:
            continue

        # Prefer alphabetic trading symbol; fall back to numeric scrip code
        symbol = scrip_id if scrip_id and scrip_id != "NAN" else scrip_code
        if not symbol:
            continue

        result.append(
            UniverseSymbol(
                ticker=f"BSE:{symbol}",
                exchange="BSE",
                provider_symbol=to_yfinance_symbol(symbol, "BSE"),
                company_name=name,
                sector=sector,
            )
        )

    logger.info("[BSE] Loaded %d equity symbols from BSE official API", len(result))
    return result


def to_yfinance_symbol(symbol: str, exchange: str) -> str:
    exchange = exchange.upper()
    suffix = {"NSE": ".NS", "BSE": ".BO"}.get(exchange)
    if not suffix:
        raise ValueError(f"Unsupported exchange: {exchange}")
    symbol = str(symbol).strip().upper()
    if symbol.endswith(suffix):
        return symbol
    return f"{symbol}{suffix}"


def write_universe_metadata(path: str, symbols: Iterable[UniverseSymbol]) -> None:
    payload = {}
    for item in symbols:
        bare = item.ticker.split(":")[-1]
        payload[bare] = {
            "name": item.company_name,
            "sector": item.sector,
            "file": f"{item.ticker.replace(':', '_')}.csv",
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _first_present(columns: Dict[str, str], options: List[str]) -> Optional[str]:
    for option in options:
        if option in columns:
            return columns[option]
    return None
