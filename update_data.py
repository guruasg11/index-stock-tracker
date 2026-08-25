"""
update_data.py
==============
Standalone daily data builder for the NSE Index Returns Dashboard.

Run by GitHub Actions once per weekday evening (IST). Fetches EOD data from
NSE India's official API via the `nse` package, computes nine trailing return
periods for every index and every constituent stock, and writes three JSON
files into ./data/ which the Streamlit app reads from disk.

This script NEVER aborts on a single failure. Any index, symbol, or network
error is logged and skipped so the JSON files are still written with whatever
succeeded.

Environment overrides (all optional):
    NSE_THROTTLE    seconds to sleep between NSE calls        (default 0.40)
    NSE_ATTEMPTS    retry attempts per NSE call               (default 3)
    HISTORY_DAYS    days of history to request per symbol     (default 400)
    MAX_INDICES     cap indices processed, 0 = all            (default 0)
    SKIP_STOCKS     set to 1 to build indices.json only       (default 0)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta

try:
    from nse import NSE
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "FATAL: the `nse` package is not installed. "
        "Run: pip install -r requirements.txt\n"
    )
    raise


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

INDICES_FILE = os.path.join(DATA_DIR, "indices.json")
STOCKS_FILE = os.path.join(DATA_DIR, "stocks.json")
LAST_UPDATED_FILE = os.path.join(DATA_DIR, "last_updated.json")

THROTTLE_SECONDS = float(os.getenv("NSE_THROTTLE", "0.40"))
MAX_ATTEMPTS = int(os.getenv("NSE_ATTEMPTS", "3"))
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "400"))
MAX_INDICES = int(os.getenv("MAX_INDICES", "0"))
SKIP_STOCKS = os.getenv("SKIP_STOCKS", "0") == "1"

# Flush partial results to disk every N indices so a timeout never loses a run.
CHECKPOINT_EVERY = 5

IST = timezone(timedelta(hours=5, minutes=30))

INDEX_DATE_FORMAT = "%d-%b-%Y"
STOCK_DATE_FORMAT = "%d-%b-%Y"

# label -> function mapping the latest trading date to the lookback target date.
# "1D" is special-cased (previous available row, not a calendar offset).
PERIODS: List[Tuple[str, Optional[Callable[[date], date]]]] = [
    ("1D", None),
    ("3D", lambda d: d - timedelta(days=3)),
    ("1W", lambda d: d - timedelta(weeks=1)),
    ("2W", lambda d: d - timedelta(weeks=2)),
    ("1M", lambda d: d - relativedelta(months=1)),
    ("2M", lambda d: d - relativedelta(months=2)),
    ("3M", lambda d: d - relativedelta(months=3)),
    ("6M", lambda d: d - relativedelta(months=6)),
    ("1Y", lambda d: d - relativedelta(years=1)),
]

PERIOD_LABELS = [label for label, _ in PERIODS]


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def log(message: str) -> None:
    """Timestamped, immediately-flushed stdout logging for Action logs."""
    stamp = datetime.now(IST).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# --------------------------------------------------------------------------
# Series construction and return maths
# --------------------------------------------------------------------------

def build_series(
    records: Any,
    close_key: str,
    date_key: str,
    date_format: str,
) -> Optional[pd.Series]:
    """
    Turn a list of NSE record dicts into a float Series indexed by date,
    ascending, de-duplicated. Returns None when nothing usable is present.
    """
    if not isinstance(records, list) or not records:
        return None

    rows: List[Tuple[pd.Timestamp, float]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_date = record.get(date_key)
        raw_close = record.get(close_key)
        if raw_date is None or raw_close is None:
            continue
        try:
            parsed_date = datetime.strptime(str(raw_date).strip(), date_format).date()
        except (ValueError, TypeError):
            continue
        try:
            close = float(str(raw_close).replace(",", "").strip())
        except (ValueError, TypeError):
            continue
        if close <= 0:
            continue
        rows.append((pd.Timestamp(parsed_date), close))

    if not rows:
        return None

    frame = pd.DataFrame(rows, columns=["date", "close"])
    frame = frame.drop_duplicates(subset="date", keep="last")
    frame = frame.sort_values("date")

    return pd.Series(
        frame["close"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(frame["date"]),
    )


def compute_returns(
    series: Optional[pd.Series],
) -> Tuple[Optional[float], Optional[str], Dict[str, Optional[float]]]:
    """
    Compute the nine trailing returns.

    Returns (latest_close, as_of_iso_date, {period_label: percent_or_None}).
    Missing data yields None for that period, never an exception.
    """
    returns: Dict[str, Optional[float]] = {label: None for label in PERIOD_LABELS}

    if series is None or series.empty:
        return None, None, returns

    latest_timestamp = series.index[-1]
    latest_close = float(series.iloc[-1])
    as_of = latest_timestamp.strftime("%Y-%m-%d")

    # 1 Day: previous available trading row.
    if len(series) >= 2:
        previous_close = float(series.iloc[-2])
        if previous_close > 0:
            returns["1D"] = round(
                (latest_close - previous_close) / previous_close * 100.0, 2
            )

    latest_date = latest_timestamp.to_pydatetime().date()

    for label, offset in PERIODS:
        if offset is None:
            continue
        try:
            target = pd.Timestamp(offset(latest_date))
        except (ValueError, OverflowError):
            continue
        try:
            # asof() returns the last value at or before target, which is
            # exactly the closest earlier trading day across weekends/holidays.
            base = series.asof(target)
        except (TypeError, ValueError, KeyError):
            base = float("nan")
        if base is None or pd.isna(base):
            continue
        base_close = float(base)
        if base_close <= 0:
            continue
        returns[label] = round((latest_close - base_close) / base_close * 100.0, 2)

    return round(latest_close, 2), as_of, returns


# --------------------------------------------------------------------------
# Throttled, retrying NSE access
# --------------------------------------------------------------------------

def nse_call(fn: Callable[[], Any], label: str) -> Optional[Any]:
    """
    Execute an NSE call with throttling, retries and exponential backoff.
    Returns None on total failure. Never raises.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if THROTTLE_SECONDS > 0:
                time.sleep(THROTTLE_SECONDS)
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            reason = f"{type(exc).__name__}: {exc}"
            if attempt == MAX_ATTEMPTS:
                log(f"    ! {label} failed after {MAX_ATTEMPTS} attempts ({reason})")
                return None
            backoff = 2.0 * (2 ** (attempt - 1))
            log(f"    ~ {label} attempt {attempt} failed ({reason}); retry in {backoff:.0f}s")
            time.sleep(backoff)
    return None


def date_window() -> Tuple[date, date]:
    today = datetime.now(IST).date()
    return today - timedelta(days=HISTORY_DAYS), today


# --------------------------------------------------------------------------
# Fetchers (stock fetches are globally cached — each symbol hit once per run)
# --------------------------------------------------------------------------

_STOCK_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def fetch_index_row(client: "NSE", index_name: str) -> Optional[Dict[str, Any]]:
    from_date, to_date = date_window()
    raw = nse_call(
        lambda: client.fetch_historical_index_data(
            index_name, from_date=from_date, to_date=to_date
        ),
        f"history for index '{index_name}'",
    )
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("indexCloseOnlineRecords", []))

    series = build_series(raw, "EOD_CLOSE_INDEX_VAL", "EOD_TIMESTAMP", INDEX_DATE_FORMAT)
    close, as_of, returns = compute_returns(series)
    if close is None:
        log(f"    ! no usable history for index '{index_name}'")
        return None

    row: Dict[str, Any] = {"index": index_name, "close": close, "as_of": as_of}
    row.update(returns)
    return row


def fetch_stock_row(client: "NSE", symbol: str) -> Optional[Dict[str, Any]]:
    if symbol in _STOCK_CACHE:
        return _STOCK_CACHE[symbol]

    from_date, to_date = date_window()
    raw = nse_call(
        lambda: client.fetch_equity_historical_data(
            symbol, from_date=from_date, to_date=to_date
        ),
        f"history for symbol '{symbol}'",
    )
    if isinstance(raw, dict):
        raw = raw.get("data", [])

    series = build_series(raw, "CH_CLOSING_PRICE", "mTIMESTAMP", STOCK_DATE_FORMAT)
    close, as_of, returns = compute_returns(series)

    if close is None:
        log(f"    ! no usable history for symbol '{symbol}'")
        _STOCK_CACHE[symbol] = None
        return None

    row: Dict[str, Any] = {"symbol": symbol, "close": close, "as_of": as_of}
    row.update(returns)
    _STOCK_CACHE[symbol] = row
    return row


def fetch_constituents(client: "NSE", index_name: str) -> List[str]:
    payload = nse_call(
        lambda: client.listEquityStocksByIndex(index_name),
        f"constituents of '{index_name}'",
    )
    if not payload:
        return []
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []

    symbols: List[str] = []
    seen = set()
    normalised_index = index_name.strip().upper()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if not symbol:
            continue
        symbol = str(symbol).strip()
        # NSE includes the index itself as a pseudo-row; drop it.
        if not symbol or symbol.upper() == normalised_index:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    """Atomic-ish write: temp file in the same dir, then replace."""
    ensure_data_dir()
    directory = os.path.dirname(path)
    handle, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=1, sort_keys=False)
        os.replace(temp_path, path)
    except Exception:  # noqa: BLE001
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def checkpoint(
    indices_payload: List[Dict[str, Any]],
    stocks_payload: Dict[str, List[Dict[str, Any]]],
    finished: bool,
) -> None:
    try:
        write_json(INDICES_FILE, indices_payload)
        write_json(STOCKS_FILE, stocks_payload)
        write_json(
            LAST_UPDATED_FILE,
            {
                "last_updated": datetime.now(IST).isoformat(timespec="seconds"),
                "last_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "indices_count": len(indices_payload),
                "stocks_count": sum(len(v) for v in stocks_payload.values()),
                "unique_symbols_fetched": len(_STOCK_CACHE),
                "complete": finished,
            },
        )
    except Exception as exc:  # noqa: BLE001
        log(f"! checkpoint write failed: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    started = time.time()
    ensure_data_dir()
    log("NSE Index Returns Dashboard - daily data build starting")
    log(
        f"config: history={HISTORY_DAYS}d throttle={THROTTLE_SECONDS}s "
        f"attempts={MAX_ATTEMPTS} max_indices={MAX_INDICES or 'all'} "
        f"skip_stocks={SKIP_STOCKS}"
    )

    download_folder = tempfile.mkdtemp(prefix="nse_dl_")
    try:
        client = NSE(download_folder=download_folder, server=True)
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: could not initialise NSE client: {type(exc).__name__}: {exc}")
        checkpoint([], {}, finished=False)
        return 1

    indices_payload: List[Dict[str, Any]] = []
    stocks_payload: Dict[str, List[Dict[str, Any]]] = {}

    try:
        listing = nse_call(lambda: client.listIndices(), "listIndices")
        raw_indices = listing.get("data", []) if isinstance(listing, dict) else []
        index_names: List[str] = []
        seen = set()
        for item in raw_indices:
            if not isinstance(item, dict):
                continue
            name = item.get("index")
            if not name:
                continue
            name = str(name).strip()
            if name and name not in seen:
                seen.add(name)
                index_names.append(name)

        if MAX_INDICES > 0:
            index_names = index_names[:MAX_INDICES]

        if not index_names:
            log("! listIndices returned nothing usable - writing empty payloads")
            checkpoint([], {}, finished=False)
            return 1

        total = len(index_names)
        log(f"{total} indices to process")

        for position, index_name in enumerate(index_names, start=1):
            log(f"[{position}/{total}] {index_name}")

            try:
                index_row = fetch_index_row(client, index_name)
            except Exception as exc:  # noqa: BLE001
                log(f"    ! unexpected error on index '{index_name}': {exc}")
                index_row = None

            if index_row:
                indices_payload.append(index_row)
                log(f"    close={index_row['close']} as_of={index_row['as_of']} 1D={index_row['1D']}")

            if SKIP_STOCKS:
                continue

            try:
                symbols = fetch_constituents(client, index_name)
            except Exception as exc:  # noqa: BLE001
                log(f"    ! unexpected error listing constituents: {exc}")
                symbols = []

            if not symbols:
                log("    (no equity constituents - likely a non-equity index)")
                stocks_payload[index_name] = []
                continue

            cached = sum(1 for s in symbols if s in _STOCK_CACHE)
            log(f"    {len(symbols)} constituents ({cached} already cached)")

            rows: List[Dict[str, Any]] = []
            for symbol in symbols:
                try:
                    stock_row = fetch_stock_row(client, symbol)
                except Exception as exc:  # noqa: BLE001
                    log(f"    ! unexpected error on symbol '{symbol}': {exc}")
                    stock_row = None
                if stock_row:
                    rows.append(dict(stock_row))

            stocks_payload[index_name] = rows
            log(f"    {len(rows)}/{len(symbols)} constituents computed")

            if position % CHECKPOINT_EVERY == 0:
                checkpoint(indices_payload, stocks_payload, finished=False)
                log(f"    ~ checkpoint written ({position}/{total})")

    except KeyboardInterrupt:
        log("! interrupted - writing partial results")
    except Exception as exc:  # noqa: BLE001
        log(f"! unexpected top-level error: {type(exc).__name__}: {exc}")
    finally:
        try:
            client.exit()
        except Exception:  # noqa: BLE001
            pass

    checkpoint(indices_payload, stocks_payload, finished=True)

    elapsed = time.time() - started
    log(
        f"done in {elapsed / 60:.1f} min - "
        f"{len(indices_payload)} indices, "
        f"{sum(len(v) for v in stocks_payload.values())} index-stock rows, "
        f"{len(_STOCK_CACHE)} unique symbols fetched"
    )
    # Exit 0 even on partial success so the Action still commits what it got.
    return 0


if __name__ == "__main__":
    sys.exit(main())
