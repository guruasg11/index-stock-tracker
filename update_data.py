"""
update_data.py
==============
Daily data builder for the NSE Index Returns Dashboard.

Runs in GitHub Actions each weekday evening (IST). Pulls EOD data from NSE
India's official API via the `nse` package, computes nine trailing return
periods for every index and every constituent stock, and writes JSON into
./data/ which the Streamlit app reads from disk.

Never aborts on a single failure. Any index, symbol or network error is
logged and skipped so the JSON is still written with whatever succeeded.

CORPORATE ACTIONS
-----------------
NSE's historical equity endpoint returns RAW TRADED closes. It does not
adjust for splits, bonuses or consolidations. A 1:5 split therefore reads as
a -80% move on every horizon spanning the ex-date, which is why unadjusted
figures diverge from NSE's published stock returns.

This script back-adjusts. Any single-session move beyond the CORP_ACTION_BAND
threshold is treated as a corporate action (NSE price bands cap genuine
single-day moves well below this) and all prior closes are rescaled onto the
post-event basis. Every adjustment is recorded in data/diagnostics.json so it
can be audited against NSE's corporate action calendar.

Indices are NOT adjusted - index levels are already continuous through
constituent corporate actions.

Environment overrides (all optional):
    NSE_THROTTLE        seconds between NSE calls           (default 0.35)
    NSE_ATTEMPTS        retry attempts per call             (default 3)
    HISTORY_DAYS        days of history per symbol          (default 400)
    MAX_INDICES         cap indices, 0 = all                (default 0)
    SKIP_STOCKS         1 = build indices only              (default 0)
    CORP_ACTION_BAND    single-day move treated as an       (default 0.25)
                        action, as a fraction
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
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
STOCKS_DIR = os.path.join(DATA_DIR, "stocks")

INDICES_FILE = os.path.join(DATA_DIR, "indices.json")
STOCKS_MAP_FILE = os.path.join(DATA_DIR, "stocks_index.json")
ALL_STOCKS_FILE = os.path.join(DATA_DIR, "all_stocks.json")
MEMBERSHIP_FILE = os.path.join(DATA_DIR, "symbol_membership.json")
LAST_UPDATED_FILE = os.path.join(DATA_DIR, "last_updated.json")
DIAGNOSTICS_FILE = os.path.join(DATA_DIR, "diagnostics.json")

THROTTLE_SECONDS = float(os.getenv("NSE_THROTTLE", "0.35"))
MAX_ATTEMPTS = int(os.getenv("NSE_ATTEMPTS", "3"))
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "400"))
MAX_INDICES = int(os.getenv("MAX_INDICES", "0"))
SKIP_STOCKS = os.getenv("SKIP_STOCKS", "0") == "1"
CORP_ACTION_BAND = float(os.getenv("CORP_ACTION_BAND", "0.25"))

CHECKPOINT_EVERY = 5

IST = timezone(timedelta(hours=5, minutes=30))

# Periods measured in TRADING SESSIONS, not calendar days. Three calendar days
# back from a Monday lands on the same Friday close as one session back, so
# every Monday build would report an identical 1D and 3D. Sessions avoid that.
SESSION_PERIODS: Dict[str, int] = {"1D": 1, "3D": 3}

PERIODS: List[Tuple[str, Optional[Callable[[date], date]]]] = [
    ("1D", None),
    ("3D", None),
    ("1W", lambda d: d - timedelta(weeks=1)),
    ("2W", lambda d: d - timedelta(weeks=2)),
    ("1M", lambda d: d - relativedelta(months=1)),
    ("2M", lambda d: d - relativedelta(months=2)),
    ("3M", lambda d: d - relativedelta(months=3)),
    ("6M", lambda d: d - relativedelta(months=6)),
    ("1Y", lambda d: d - relativedelta(years=1)),
]

PERIOD_LABELS = [label for label, _ in PERIODS]


def log(message: str) -> None:
    stamp = datetime.now(IST).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def slugify(name: str) -> str:
    """Filesystem-safe shard name. Mirrored in the Streamlit detail page."""
    slug = re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")
    return (slug or "INDEX")[:80]


# --------------------------------------------------------------------------
# Series construction, corporate actions, return maths
# --------------------------------------------------------------------------

SCHEMA_SEEN: Dict[str, Any] = {}

DATE_FORMATS = (
    "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%Y %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y",
    "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%b %d, %Y",
)

DATE_KEY_HINT = re.compile(r"(TIMESTAMP|_DATE|^DATE|DT$|TRADE_DATE)", re.IGNORECASE)
CLOSE_KEY_HINT = re.compile(r"CLOS", re.IGNORECASE)
CLOSE_KEY_VETO = re.compile(r"(PREV|PRIOR|ADJ_?PREV)", re.IGNORECASE)


def parse_any_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse a date across every shape NSE's endpoints have used."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).normalize()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # epoch seconds or milliseconds
        try:
            unit = "ms" if float(value) > 1e11 else "s"
            return pd.Timestamp(pd.to_datetime(float(value), unit=unit)).normalize()
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1]
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt)).normalize()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    except (ValueError, TypeError):
        return None
    if parsed is None or pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def parse_any_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number > 0 else None
    text = str(value).replace(",", "").replace("\u20b9", "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number > 0 else None


def unwrap_records(payload: Any) -> List[Dict[str, Any]]:
    """Flatten whatever the endpoint returned into a list of record dicts."""
    if payload is None:
        return []
    if isinstance(payload, pd.DataFrame):
        return payload.to_dict("records")
    if isinstance(payload, dict):
        for key in ("data", "indexCloseOnlineRecords", "records", "grapthData", "result"):
            if key in payload:
                return unwrap_records(payload[key])
        # a dict of dicts
        values = list(payload.values())
        if values and all(isinstance(v, dict) for v in values):
            return values
        return []
    if isinstance(payload, (list, tuple)):
        flat: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                flat.append(item)
            elif isinstance(item, (list, tuple, dict)):
                flat.extend(unwrap_records(item))
        return flat
    return []


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """One level of nesting - some NSE payloads bury fields under 'meta'."""
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat.setdefault(str(inner_key), inner_value)
        else:
            flat[str(key)] = value
    return flat


def detect_fields(
    records: List[Dict[str, Any]],
    prefer_close: str,
    prefer_date: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the close and date columns by probing actual values, so a renamed
    field or a changed date format degrades to a different column rather
    than to silence.
    """
    sample = records[: min(len(records), 25)]
    if not sample:
        return None, None
    keys: List[str] = []
    for record in sample:
        for key in record:
            if key not in keys:
                keys.append(key)

    def usable_date(key: str) -> bool:
        hits = sum(1 for r in sample if parse_any_date(r.get(key)) is not None)
        return hits >= max(1, len(sample) // 2)

    def usable_number(key: str) -> bool:
        hits = sum(1 for r in sample if parse_any_number(r.get(key)) is not None)
        return hits >= max(1, len(sample) // 2)

    date_key = None
    if prefer_date in keys and usable_date(prefer_date):
        date_key = prefer_date
    else:
        for key in keys:
            if DATE_KEY_HINT.search(key) and usable_date(key):
                date_key = key
                break
        if date_key is None:
            for key in keys:
                if usable_date(key) and not usable_number(key):
                    date_key = key
                    break

    close_key = None
    if prefer_close in keys and usable_number(prefer_close):
        close_key = prefer_close
    else:
        candidates = [
            k for k in keys
            if CLOSE_KEY_HINT.search(k) and not CLOSE_KEY_VETO.search(k) and usable_number(k)
        ]
        if candidates:
            close_key = candidates[0]

    return close_key, date_key


def build_series(
    payload: Any,
    close_key: str,
    date_key: str,
    schema_label: str = "",
) -> Optional[pd.Series]:
    """
    NSE payload -> float Series indexed by date, ascending, de-duplicated.

    Field names and date formats are detected from the payload itself. The
    close_key/date_key arguments are only the preferred names.
    """
    records = [flatten_record(r) for r in unwrap_records(payload)]
    if not records:
        return None

    resolved_close, resolved_date = detect_fields(records, close_key, date_key)

    if schema_label and schema_label not in SCHEMA_SEEN:
        SCHEMA_SEEN[schema_label] = {
            "keys": sorted(records[0].keys()),
            "using_close": resolved_close,
            "using_date": resolved_date,
            "sample": {
                k: str(v)[:40] for k, v in list(records[0].items())[:14]
            },
        }
        log(f"  schema[{schema_label}] close={resolved_close!r} date={resolved_date!r}")
        log(f"  schema[{schema_label}] keys={sorted(records[0].keys())}")

    if not resolved_close or not resolved_date:
        return None

    rows: List[Tuple[pd.Timestamp, float]] = []
    for record in records:
        stamp = parse_any_date(record.get(resolved_date))
        close = parse_any_number(record.get(resolved_close))
        if stamp is None or close is None:
            continue
        rows.append((stamp, close))

    if not rows:
        return None

    frame = pd.DataFrame(rows, columns=["date", "close"])
    frame = frame.drop_duplicates(subset="date", keep="last").sort_values("date")

    return pd.Series(
        frame["close"].to_numpy(dtype="float64"),
        index=pd.DatetimeIndex(frame["date"]),
    )


def adjust_for_corporate_actions(
    series: Optional[pd.Series],
    band: float = CORP_ACTION_BAND,
) -> Tuple[Optional[pd.Series], List[Dict[str, Any]]]:
    """
    Back-adjust raw NSE closes for splits, bonuses and consolidations.

    Walks the series backwards. Any session-over-session ratio outside
    [1-band, 1/(1-band)] is treated as a corporate action; every earlier close
    is multiplied by the cumulative ratio so the whole series sits on the
    current, post-action basis. Returns the adjusted series and the events.
    """
    if series is None or len(series) < 3:
        return series, []

    raw = series.to_numpy(dtype="float64")
    count = len(raw)
    factors = np.ones(count, dtype="float64")
    events: List[Dict[str, Any]] = []

    lower = 1.0 - band
    upper = 1.0 / lower if lower > 0 else float("inf")

    cumulative = 1.0
    for position in range(count - 1, 0, -1):
        previous = raw[position - 1]
        if previous <= 0:
            factors[position - 1] = cumulative
            continue
        ratio = raw[position] / previous
        if ratio <= lower or ratio >= upper:
            cumulative *= ratio
            events.append(
                {
                    "date": series.index[position].strftime("%Y-%m-%d"),
                    "ratio": round(float(ratio), 6),
                    "raw_prev_close": round(float(previous), 2),
                    "raw_close": round(float(raw[position]), 2),
                }
            )
        factors[position - 1] = cumulative

    if not events:
        return series, []

    adjusted = pd.Series(raw * factors, index=series.index)
    events.reverse()
    return adjusted, events


def compute_returns(
    series: Optional[pd.Series],
) -> Tuple[Optional[float], Optional[str], Dict[str, Optional[float]]]:
    """
    Nine trailing returns.

    Returns (latest_close, as_of_iso, {period: percent_or_None}).
    Missing history yields None for that period, never an exception.
    """
    returns: Dict[str, Optional[float]] = {label: None for label in PERIOD_LABELS}

    if series is None or series.empty:
        return None, None, returns

    latest_timestamp = series.index[-1]
    latest_close = float(series.iloc[-1])
    as_of = latest_timestamp.strftime("%Y-%m-%d")

    for label, sessions in SESSION_PERIODS.items():
        if len(series) < sessions + 1:
            continue
        base_close = float(series.iloc[-(sessions + 1)])
        if base_close > 0:
            returns[label] = round(
                (latest_close - base_close) / base_close * 100.0, 2
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
            # asof() returns the last value at or before target - exactly the
            # closest earlier trading day across weekends and holidays.
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

FAILURES: List[Dict[str, str]] = []


def nse_call(fn: Callable[[], Any], label: str) -> Optional[Any]:
    """Throttled call with retries and exponential backoff. Never raises."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if THROTTLE_SECONDS > 0:
                time.sleep(THROTTLE_SECONDS)
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            reason = f"{type(exc).__name__}: {exc}"
            if attempt == MAX_ATTEMPTS:
                log(f"    ! {label} failed after {MAX_ATTEMPTS} attempts ({reason})")
                FAILURES.append({"target": label, "reason": reason})
                return None
            backoff = 2.0 * (2 ** (attempt - 1))
            log(f"    ~ {label} attempt {attempt} failed ({reason}); retry in {backoff:.0f}s")
            time.sleep(backoff)
    return None


def date_window() -> Tuple[date, date]:
    today = datetime.now(IST).date()
    return today - timedelta(days=HISTORY_DAYS), today


# --------------------------------------------------------------------------
# Fetchers - stock fetches are globally cached, one call per unique symbol
# --------------------------------------------------------------------------

_STOCK_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
ADJUSTMENTS: Dict[str, List[Dict[str, Any]]] = {}
# symbol -> every index that contains it. Free: built from data already fetched.
MEMBERSHIP: Dict[str, List[str]] = {}
SPAN_LOGGED: set = set()


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

    series = build_series(raw, "EOD_CLOSE_INDEX_VAL", "EOD_TIMESTAMP", "index")
    close, as_of, returns = compute_returns(series)
    if close is None:
        log(f"    ! no usable history for index '{index_name}'")
        return None

    row: Dict[str, Any] = {
        "index": index_name,
        "close": close,
        "as_of": as_of,
        "sessions": int(len(series)) if series is not None else 0,
        "from": series.index[0].strftime("%Y-%m-%d") if series is not None else None,
    }
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
    series = build_series(raw, "CH_CLOSING_PRICE", "mTIMESTAMP", "equity")
    series, events = adjust_for_corporate_actions(series)
    close, as_of, returns = compute_returns(series)

    if close is None:
        log(f"    ! no usable history for symbol '{symbol}'")
        _STOCK_CACHE[symbol] = None
        return None

    if len(SPAN_LOGGED) < 5:
        SPAN_LOGGED.add(symbol)
        log(
            f"    span[{symbol}] {len(series)} sessions "
            f"{series.index[0].date()} -> {series.index[-1].date()}"
        )
    if len(series) < 150:
        log(f"    ? {symbol}: only {len(series)} sessions - long horizons will be thin")

    if events:
        ADJUSTMENTS[symbol] = events
        ratios = ", ".join(f"{e['date']} x{e['ratio']:.4f}" for e in events)
        log(f"    * {symbol}: corporate action adjusted ({ratios})")

    row: Dict[str, Any] = {
        "symbol": symbol,
        "close": close,
        "as_of": as_of,
        "adjusted": bool(events),
        "sessions": int(len(series)),
        "from": series.index[0].strftime("%Y-%m-%d"),
    }
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
    rows = [flatten_record(r) for r in unwrap_records(payload)]
    if not rows:
        return []

    if "constituents" not in SCHEMA_SEEN:
        SCHEMA_SEEN["constituents"] = {"keys": sorted(rows[0].keys())}
        log(f"  schema[constituents] keys={sorted(rows[0].keys())}")

    symbols: List[str] = []
    seen = set()
    normalised_index = index_name.strip().upper()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol") or row.get("Symbol") or row.get("SYMBOL")
        if not symbol:
            continue
        symbol = str(symbol).strip()
        # NSE returns the index itself as a pseudo-row; drop it.
        if not symbol or symbol.upper() == normalised_index:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


# --------------------------------------------------------------------------
# Persistence - stocks are sharded one file per index so daily git diffs stay
# small and the detail page loads only the shard it needs.
# --------------------------------------------------------------------------

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STOCKS_DIR, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    ensure_dirs()
    directory = os.path.dirname(path)
    handle, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=1)
        os.replace(temp_path, path)
    except Exception:  # noqa: BLE001
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def write_stock_shard(index_name: str, rows: List[Dict[str, Any]]) -> str:
    filename = f"{slugify(index_name)}.json"
    write_json(os.path.join(STOCKS_DIR, filename), rows)
    return filename


def checkpoint(
    indices_payload: List[Dict[str, Any]],
    shard_map: Dict[str, str],
    finished: bool,
    started: float,
) -> None:
    try:
        write_json(INDICES_FILE, indices_payload)
        write_json(STOCKS_MAP_FILE, shard_map)
        # Flat universe for the lookup page: every symbol priced this run.
        write_json(
            ALL_STOCKS_FILE,
            sorted(
                (row for row in _STOCK_CACHE.values() if row),
                key=lambda r: r["symbol"],
            ),
        )
        write_json(
            MEMBERSHIP_FILE,
            {symbol: sorted(set(names)) for symbol, names in MEMBERSHIP.items()},
        )
        write_json(
            LAST_UPDATED_FILE,
            {
                "last_updated": datetime.now(IST).isoformat(timespec="seconds"),
                "last_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "indices_count": len(indices_payload),
                "indices_with_stocks": len(shard_map),
                "unique_symbols": len([r for r in _STOCK_CACHE.values() if r]),
                "symbols_adjusted": len(ADJUSTMENTS),
                "failures": len(FAILURES),
                "runtime_minutes": round((time.time() - started) / 60.0, 1),
                "complete": finished,
            },
        )
        write_json(
            DIAGNOSTICS_FILE,
            {
                "corporate_action_band": CORP_ACTION_BAND,
                "history_days": HISTORY_DAYS,
                "detected_schema": SCHEMA_SEEN,
                "adjustments": ADJUSTMENTS,
                "failures": FAILURES[-200:],
            },
        )
    except Exception as exc:  # noqa: BLE001
        log(f"! checkpoint write failed: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    started = time.time()
    ensure_dirs()
    log("NSE Index Returns Dashboard - daily data build starting")
    log(
        f"config: history={HISTORY_DAYS}d throttle={THROTTLE_SECONDS}s "
        f"attempts={MAX_ATTEMPTS} max_indices={MAX_INDICES or 'all'} "
        f"skip_stocks={SKIP_STOCKS} corp_action_band={CORP_ACTION_BAND:.0%}"
    )

    download_folder = tempfile.mkdtemp(prefix="nse_dl_")
    try:
        client = NSE(download_folder=download_folder, server=True)
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: could not initialise NSE client: {type(exc).__name__}: {exc}")
        FAILURES.append({"target": "NSE client init", "reason": str(exc)})
        checkpoint([], {}, finished=False, started=started)
        return 1

    indices_payload: List[Dict[str, Any]] = []
    shard_map: Dict[str, str] = {}

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
            checkpoint([], {}, finished=False, started=started)
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
                log(
                    f"    close={index_row['close']} as_of={index_row['as_of']} "
                    f"1D={index_row['1D']} 1Y={index_row['1Y']}"
                )

            if SKIP_STOCKS:
                continue

            try:
                symbols = fetch_constituents(client, index_name)
            except Exception as exc:  # noqa: BLE001
                log(f"    ! unexpected error listing constituents: {exc}")
                symbols = []

            if not symbols:
                log("    (no equity constituents - likely a non-equity index)")
                shard_map[index_name] = write_stock_shard(index_name, [])
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
                    MEMBERSHIP.setdefault(symbol, []).append(index_name)

            shard_map[index_name] = write_stock_shard(index_name, rows)
            log(f"    {len(rows)}/{len(symbols)} constituents computed")

            if position % CHECKPOINT_EVERY == 0:
                checkpoint(indices_payload, shard_map, finished=False, started=started)
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

    checkpoint(indices_payload, shard_map, finished=True, started=started)

    priced = [r for r in _STOCK_CACHE.values() if r]
    if priced:
        spans = sorted(r.get("sessions", 0) for r in priced)
        median_span = spans[len(spans) // 2]
        thin = sum(1 for n in spans if n < 150)
        log(
            f"coverage: median {median_span} sessions per symbol, "
            f"min {spans[0]}, max {spans[-1]}, {thin} symbols under 150 sessions"
        )

    log(
        f"done in {(time.time() - started) / 60:.1f} min - "
        f"{len(indices_payload)} indices, {len(shard_map)} shards, "
        f"{len(_STOCK_CACHE)} unique symbols, "
        f"{len(ADJUSTMENTS)} corporate-action adjustments, "
        f"{len(FAILURES)} failures"
    )
    # Exit 0 even on partial success so the Action still commits what it got.
    return 0


if __name__ == "__main__":
    sys.exit(main())
