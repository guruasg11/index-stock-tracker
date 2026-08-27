"""
update_data.py
==============
Daily data builder for the NSE Index Returns Dashboard.

Runs in GitHub Actions each weekday evening (IST). Computes nine trailing
return periods for every NSE index and every constituent stock, and writes
JSON into ./data/ which the Streamlit app reads from disk.

DATA SOURCE - ARCHIVE FILES, NOT THE PER-SYMBOL API
---------------------------------------------------
Earlier builds called NSE's historical API once per index and once per symbol,
roughly 1100 requests per run. NSE rate-limits datacentre IP ranges, so a
GitHub Actions run lost most of those calls to throttling and the dashboard
silently served carried-forward data.

This build reads NSE's own daily archive files instead:

    equityBhavcopy(date)   -> BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
                              every equity close for that session
    indicesBhavcopy(date)  -> ind_close_all_DDMMYYYY.csv
                              every index close for that session

Two downloads per trading day. Each session is cached under data/history/ as
an immutable gzipped file, so a normal daily run fetches exactly two files and
appends. Only the first run backfills the full window.

Request volume per daily run drops from ~1100 to 2, plus one weekly refresh of
index constituents.

CORPORATE ACTIONS
-----------------
Bhavcopy closes are RAW TRADED prices. A 1:5 split reads as a -80% move on
every horizon spanning the ex-date. Any single-session move beyond
CORP_ACTION_BAND is treated as a corporate action and all prior closes are
rescaled onto the post-event basis. Every adjustment is recorded in
data/diagnostics.json so it can be audited against NSE's corporate action
calendar.

Indices are NOT adjusted - index levels are already continuous through
constituent corporate actions.

Environment overrides (all optional):
    NSE_THROTTLE        seconds between archive downloads   (default 1.0)
    NSE_ATTEMPTS        retry attempts per download         (default 4)
    HISTORY_DAYS        calendar days of history to keep    (default 400)
    MAX_NEW_DAYS        cap sessions fetched per run,       (default 0)
                        0 = every missing session
    REBUILD_EQUITY_HISTORY  1 = discard and refetch equity   (default 0)
                        sessions. Run once after changing
                        which bhavcopy rows are kept.
    CONSTITUENT_MAX_AGE days before constituents refresh    (default 7)
    FORCE_CONSTITUENTS  1 = refresh every constituent list  (default 0)
    CONSTITUENT_BATCH   constituent lists refreshed per run (default 40)
    MAX_INDICES         cap indices, 0 = all                (default 0)
    SKIP_STOCKS         1 = build indices only              (default 0)
    CORP_ACTION_BAND    single-day move treated as an       (default 0.25)
                        action, as a fraction
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import shutil
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
HISTORY_DIR = os.path.join(DATA_DIR, "history")
EQ_HISTORY_DIR = os.path.join(HISTORY_DIR, "eq")
IDX_HISTORY_DIR = os.path.join(HISTORY_DIR, "idx")

INDICES_FILE = os.path.join(DATA_DIR, "indices.json")
STOCKS_MAP_FILE = os.path.join(DATA_DIR, "stocks_index.json")
ALL_STOCKS_FILE = os.path.join(DATA_DIR, "all_stocks.json")
MEMBERSHIP_FILE = os.path.join(DATA_DIR, "symbol_membership.json")
LAST_UPDATED_FILE = os.path.join(DATA_DIR, "last_updated.json")
DIAGNOSTICS_FILE = os.path.join(DATA_DIR, "diagnostics.json")
CONSTITUENTS_FILE = os.path.join(DATA_DIR, "constituents.json")

THROTTLE_SECONDS = float(os.getenv("NSE_THROTTLE", "1.0"))
MAX_ATTEMPTS = int(os.getenv("NSE_ATTEMPTS", "4"))
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "400"))
MAX_NEW_DAYS = int(os.getenv("MAX_NEW_DAYS", "0"))
# 1 = discard the stored equity sessions and refetch them. Needed once after
# a change to which bhavcopy rows are kept, since stored files are filtered.
REBUILD_EQUITY_HISTORY = os.getenv("REBUILD_EQUITY_HISTORY", "0") == "1"
CONSTITUENT_MAX_AGE = int(os.getenv("CONSTITUENT_MAX_AGE", "7"))
FORCE_CONSTITUENTS = os.getenv("FORCE_CONSTITUENTS", "0") == "1"
# Constituent lists are refreshed in batches, oldest first. NSE throttles this
# endpoint hard from datacentre IPs, so asking for 164 in one run loses most of
# them. 60 a run converges the whole set inside a week and lands a far higher
# share of what it asks for.
CONSTITUENT_BATCH = int(os.getenv("CONSTITUENT_BATCH", "25"))
# NSE throttles the live constituents endpoint after roughly 15-20 calls and
# then answers 200 with an empty payload rather than an error. A slow pace, a
# fresh session every few calls, and stopping early on a run of empties keep
# the run inside that limit instead of collecting worthless responses.
CONSTITUENT_THROTTLE = float(os.getenv("CONSTITUENT_THROTTLE", "3.0"))
SESSION_RECYCLE_EVERY = int(os.getenv("SESSION_RECYCLE_EVERY", "12"))
SESSION_COOLDOWN = float(os.getenv("SESSION_COOLDOWN", "20.0"))
DEFER_STREAK_LIMIT = int(os.getenv("DEFER_STREAK_LIMIT", "6"))
# Archive URLs probed per index before giving up. Each probe is one request.
CSV_PROBE_LIMIT = int(os.getenv("CSV_PROBE_LIMIT", "8"))

# NSE's live-market constituents endpoint only serves the ~50 indices in its
# Live Equity Market dropdown. Everything else answers 200 with an empty
# payload. The archive publishes a constituent CSV per index instead, on the
# same host as the bhavcopies, so that is the fallback.
# NSE's own archive carries the headline indices; niftyindices.com carries the
# strategy and thematic ones that nsearchives does not publish. Both are tried.
INDEX_CSV_BASES = [
    "https://nsearchives.nseindia.com/content/indices",
    "https://www.niftyindices.com/IndexConstituent",
]

# Hand-written {index name: csv url} entries, for any index whose file name
# cannot be derived. Edit this file in the repo; it is never overwritten.
CSV_OVERRIDES_FILE = os.path.join(DATA_DIR, "constituent_url_overrides.json")

# Index families that hold no equities at all. Probing these wastes requests
# and their empty result is correct, not a failure.
NON_EQUITY_INDEX = re.compile(
    r"G-SEC|GSEC|BHARAT BOND|INDIA VIX|1D RATE|DIVIDEND POINTS|"
    r"PR 1X|PR 2X|TR 1X|TR 2X|\bUSD\b|ARBITRAGE|FUTURES INDEX|FUTURES TR|"
    r"CLEAN PRICE",
    re.IGNORECASE,
)
MAX_INDICES = int(os.getenv("MAX_INDICES", "0"))
SKIP_STOCKS = os.getenv("SKIP_STOCKS", "0") == "1"
CORP_ACTION_BAND = float(os.getenv("CORP_ACTION_BAND", "0.25"))

# Consecutive download failures walking backwards before the backfill gives up.
# Diwali plus a weekend is the longest realistic NSE gap; 8 clears it.
MAX_CONSECUTIVE_MISSES = int(os.getenv("MAX_CONSECUTIVE_MISSES", "8"))

WK52_SESSIONS = 252
IST = timezone(timedelta(hours=5, minutes=30))

# Cash-segment series to keep when the bhavcopy carries no instrument-type
# column. With UDiFF the FinInstrmTp == 'STK' filter already removes futures
# and options, so the series whitelist is only a fallback for the older layout.
# It includes REIT and InvIT codes; a whitelist that omitted them silently
# dropped every REIT and InvIT constituent from the price history.
EQUITY_SERIES = {
    "EQ", "BE", "BZ", "SM", "ST", "IL",
    "RR", "IV", "MF", "RT", "IN",
}

# Series that are never a tradable share line: rights entitlements and
# partly-paid forms. Dropped whenever the column is present.
EXCLUDED_SERIES = {"RE", "E1", "E2", "PP", "W1", "W3", "Y1", "Z1"}

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

SCHEMA_SEEN: Dict[str, Any] = {}
FAILURES: List[Dict[str, str]] = []
ADJUSTMENTS: Dict[str, List[Dict[str, Any]]] = {}
MEMBERSHIP: Dict[str, List[str]] = {}

PRIOR_INDICES: Dict[str, Dict[str, Any]] = {}
PRIOR_STOCKS: Dict[str, Dict[str, Any]] = {}
PRIOR_SHARDS: Dict[str, str] = {}
PRIOR_MEMBERSHIP: Dict[str, List[str]] = {}
PRIOR_CONSTITUENTS: Dict[str, List[str]] = {}
PRIOR_CONSTITUENT_DATES: Dict[str, str] = {}
PRIOR_CONSTITUENT_EMPTY: Dict[str, str] = {}
PRIOR_CONSTITUENT_CSV: Dict[str, str] = {}
CSV_OVERRIDES: Dict[str, str] = {}


def log(message: str) -> None:
    stamp = datetime.now(IST).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def slugify(name: str) -> str:
    """Filesystem-safe shard name. Mirrored in the Streamlit detail page."""
    slug = re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")
    return (slug or "INDEX")[:80]


def normalise_name(name: str) -> str:
    """Loose key for matching index names across NSE's inconsistent casing."""
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


# --------------------------------------------------------------------------
# Corporate actions, return maths
# --------------------------------------------------------------------------

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


def window_extremes(series: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """52-week high/low and the distance from each, off the same adjusted closes."""
    blank: Dict[str, Optional[float]] = {
        "wk52_high": None, "wk52_low": None,
        "from_high": None, "from_low": None, "wk52_sessions": 0,
    }
    if series is None or series.empty:
        return blank

    window = series.iloc[-WK52_SESSIONS:] if len(series) > WK52_SESSIONS else series
    high = float(window.max())
    low = float(window.min())
    latest = float(series.iloc[-1])

    return {
        "wk52_high": round(high, 2),
        "wk52_low": round(low, 2),
        # negative or zero: how far below the high it currently trades
        "from_high": round((latest - high) / high * 100.0, 2) if high > 0 else None,
        # positive or zero: how far above the low
        "from_low": round((latest - low) / low * 100.0, 2) if low > 0 else None,
        "wk52_sessions": int(len(window)),
    }


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
# Bhavcopy parsing - column names are detected, never assumed. NSE has changed
# this layout twice (old cmDDMMMYYYYbhav, then UDiFF in July 2024) and will
# again; detection keeps a rename from silently zeroing the build.
# --------------------------------------------------------------------------

EQ_SYMBOL_KEYS = ("TCKRSYMB", "SYMBOL")
EQ_CLOSE_KEYS = ("CLSPRIC", "CLOSE", "CLOSE_PRICE")
EQ_SERIES_KEYS = ("SCTYSRS", "SERIES")
EQ_INSTRUMENT_KEYS = ("FININSTRMTP", "INSTRUMENT")

IDX_NAME_KEYS = ("INDEXNAME", "INDEX_NAME", "INDEX")
IDX_CLOSE_KEYS = ("CLOSINGINDEXVALUE", "CLOSING_INDEX_VALUE", "CLOSE")


def _column_lookup(frame: pd.DataFrame) -> Dict[str, str]:
    """Map NORMALISED column name -> actual column name."""
    return {re.sub(r"[^A-Z0-9]+", "", str(c).upper()): str(c) for c in frame.columns}


def _pick_column(frame: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lookup = _column_lookup(frame)
    for candidate in candidates:
        key = re.sub(r"[^A-Z0-9]+", "", candidate.upper())
        if key in lookup:
            return lookup[key]
    return None


def parse_equity_bhavcopy(path: str) -> Optional[pd.DataFrame]:
    """Return a two-column frame: symbol, close. None if unparseable."""
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        log(f"    ! could not read equity bhavcopy {os.path.basename(path)}: {exc}")
        return None

    frame.columns = [str(c).strip() for c in frame.columns]

    symbol_col = _pick_column(frame, EQ_SYMBOL_KEYS)
    close_col = _pick_column(frame, EQ_CLOSE_KEYS)
    if not symbol_col or not close_col:
        log(
            f"    ! equity bhavcopy layout not recognised. columns="
            f"{sorted(frame.columns)[:25]}"
        )
        return None

    if "equity_bhavcopy" not in SCHEMA_SEEN:
        SCHEMA_SEEN["equity_bhavcopy"] = {
            "symbol": symbol_col,
            "close": close_col,
            "columns": [str(c) for c in frame.columns],
        }
        log(f"  schema[equity] symbol='{symbol_col}' close='{close_col}'")

    instrument_col = _pick_column(frame, EQ_INSTRUMENT_KEYS)
    if instrument_col:
        # UDiFF carries futures and options in the same layout family. STK is
        # the cash-segment stock row; keep it when the column exists.
        values = frame[instrument_col].astype(str).str.strip().str.upper()
        if (values == "STK").any():
            frame = frame[values == "STK"]

    series_col = _pick_column(frame, EQ_SERIES_KEYS)
    if series_col:
        values = frame[series_col].astype(str).str.strip().str.upper()
        if instrument_col:
            # STK has already removed derivatives, so only drop the series
            # that are definitely not a share line. Keeping the rest is what
            # lets REITs and InvITs through.
            frame = frame[~values.isin(EXCLUDED_SERIES)]
        else:
            frame = frame[values.isin(EQUITY_SERIES)]
        if "equity_series_kept" not in SCHEMA_SEEN:
            kept = sorted(
                set(frame[series_col].astype(str).str.strip().str.upper())
            )[:20]
            SCHEMA_SEEN["equity_series_kept"] = kept
            log(f"  schema[equity] series kept: {kept}")

    out = pd.DataFrame(
        {
            "symbol": frame[symbol_col].astype(str).str.strip().str.upper(),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    out = out[(out["symbol"] != "") & out["close"].notna() & (out["close"] > 0)]
    out = out.drop_duplicates(subset="symbol", keep="last")
    return out if not out.empty else None


def parse_indices_bhavcopy(path: str) -> Optional[pd.DataFrame]:
    """Return a two-column frame: index, close. None if unparseable."""
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        log(f"    ! could not read indices bhavcopy {os.path.basename(path)}: {exc}")
        return None

    frame.columns = [str(c).strip() for c in frame.columns]

    name_col = _pick_column(frame, IDX_NAME_KEYS)
    close_col = _pick_column(frame, IDX_CLOSE_KEYS)
    if not name_col or not close_col:
        log(
            f"    ! indices bhavcopy layout not recognised. columns="
            f"{sorted(frame.columns)[:25]}"
        )
        return None

    if "indices_bhavcopy" not in SCHEMA_SEEN:
        SCHEMA_SEEN["indices_bhavcopy"] = {
            "index": name_col,
            "close": close_col,
            "columns": [str(c) for c in frame.columns],
        }
        log(f"  schema[indices] index='{name_col}' close='{close_col}'")

    out = pd.DataFrame(
        {
            "index": frame[name_col].astype(str).str.strip(),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
        }
    )
    out = out[(out["index"] != "") & out["close"].notna() & (out["close"] > 0)]
    out = out.drop_duplicates(subset="index", keep="last")
    return out if not out.empty else None


# --------------------------------------------------------------------------
# History store - one immutable gzipped file per session per segment. Git
# stores each blob once, so the repo grows by roughly 30 KB per trading day
# instead of rewriting a monolithic history file every run.
# --------------------------------------------------------------------------

def stored_sessions(folder: str) -> List[date]:
    dates: List[date] = []
    for path in glob.glob(os.path.join(folder, "*.csv.gz")):
        stem = os.path.basename(path)[: -len(".csv.gz")]
        try:
            dates.append(datetime.strptime(stem, "%Y-%m-%d").date())
        except ValueError:
            continue
    return sorted(dates)


def write_session(folder: str, session: date, frame: pd.DataFrame) -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{session:%Y-%m-%d}.csv.gz")
    handle, temp_path = tempfile.mkstemp(dir=folder, suffix=".tmp")
    os.close(handle)
    try:
        # mtime=0 so an identical rebuild produces an identical blob and git
        # records no change.
        with gzip.GzipFile(temp_path, "wb", compresslevel=9, mtime=0) as stream:
            stream.write(frame.to_csv(index=False).encode("utf-8"))
        os.replace(temp_path, path)
    except Exception:  # noqa: BLE001
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def prune_history(folder: str, cutoff: date) -> int:
    removed = 0
    for session in stored_sessions(folder):
        if session < cutoff:
            try:
                os.remove(os.path.join(folder, f"{session:%Y-%m-%d}.csv.gz"))
                removed += 1
            except OSError:
                pass
    return removed


def download_with_retry(
    fetch: Callable[[], Any],
    label: str,
) -> Optional[str]:
    """
    One archive download with throttle, retries and backoff.

    A RuntimeError or FileNotFoundError from the `nse` package means the file
    does not exist for that date - a holiday, a weekend, or a session whose
    report is not published yet. That is not a failure and is not retried.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if THROTTLE_SECONDS > 0:
                time.sleep(THROTTLE_SECONDS)
            return str(fetch())
        except (RuntimeError, FileNotFoundError):
            return None
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            reason = f"{type(exc).__name__}: {exc}"
            if attempt == MAX_ATTEMPTS:
                log(f"    ! {label} failed after {MAX_ATTEMPTS} attempts ({reason})")
                FAILURES.append({"target": label, "reason": reason})
                return None
            backoff = 3.0 * (2 ** (attempt - 1))
            log(f"    ~ {label} attempt {attempt} failed ({reason}); retry in {backoff:.0f}s")
            time.sleep(backoff)
    return None


def sync_history(client: "NSE", scratch: str) -> Dict[str, int]:
    """
    Bring data/history/ up to date.

    Walks backwards from today to the start of the window, downloading any
    session not already stored. Weekends are skipped without a request. A run
    of consecutive misses ends the walk, which is what stops the backfill from
    grinding through years of pre-history on a fresh repo.
    """
    today = datetime.now(IST).date()
    window_start = today - timedelta(days=HISTORY_DAYS)

    if REBUILD_EQUITY_HISTORY:
        wiped = 0
        for session in stored_sessions(EQ_HISTORY_DIR):
            try:
                os.remove(os.path.join(EQ_HISTORY_DIR, f"{session:%Y-%m-%d}.csv.gz"))
                wiped += 1
            except OSError:
                pass
        log(f"REBUILD_EQUITY_HISTORY set - discarded {wiped} stored equity sessions")

    have_eq = set(stored_sessions(EQ_HISTORY_DIR))
    have_idx = set(stored_sessions(IDX_HISTORY_DIR))
    log(
        f"history on disk: {len(have_eq)} equity sessions, "
        f"{len(have_idx)} index sessions"
    )

    stats = {"eq_added": 0, "idx_added": 0, "sessions_probed": 0, "misses": 0}
    consecutive_misses = 0
    cursor = today
    budget = MAX_NEW_DAYS if MAX_NEW_DAYS > 0 else 10 ** 6
    added_sessions = 0

    while cursor >= window_start and added_sessions < budget:
        if cursor.weekday() >= 5:
            cursor -= timedelta(days=1)
            continue

        needs_eq = cursor not in have_eq
        needs_idx = cursor not in have_idx
        if not needs_eq and not needs_idx:
            consecutive_misses = 0
            cursor -= timedelta(days=1)
            continue

        stats["sessions_probed"] += 1
        stamp = datetime(cursor.year, cursor.month, cursor.day)
        got_anything = False

        if needs_idx:
            path = download_with_retry(
                lambda s=stamp: client.indicesBhavcopy(s, folder=scratch),
                f"indices bhavcopy {cursor:%Y-%m-%d}",
            )
            if path:
                frame = parse_indices_bhavcopy(path)
                if frame is not None:
                    write_session(IDX_HISTORY_DIR, cursor, frame)
                    stats["idx_added"] += 1
                    got_anything = True

        if needs_eq:
            path = download_with_retry(
                lambda s=stamp: client.equityBhavcopy(s, folder=scratch),
                f"equity bhavcopy {cursor:%Y-%m-%d}",
            )
            if path:
                frame = parse_equity_bhavcopy(path)
                if frame is not None:
                    write_session(EQ_HISTORY_DIR, cursor, frame)
                    stats["eq_added"] += 1
                    got_anything = True

        if got_anything:
            added_sessions += 1
            consecutive_misses = 0
            if added_sessions % 20 == 0:
                log(f"  backfill: {added_sessions} sessions retrieved, at {cursor:%Y-%m-%d}")
        else:
            stats["misses"] += 1
            consecutive_misses += 1
            if consecutive_misses >= MAX_CONSECUTIVE_MISSES:
                log(
                    f"  stopping walk at {cursor:%Y-%m-%d} after "
                    f"{consecutive_misses} consecutive unavailable sessions"
                )
                break

        # Free the scratch dir so a 400-day backfill does not fill the runner.
        for junk in glob.glob(os.path.join(scratch, "*")):
            try:
                os.remove(junk)
            except (OSError, IsADirectoryError):
                pass

        cursor -= timedelta(days=1)

    cutoff = today - timedelta(days=HISTORY_DAYS + 10)
    pruned = prune_history(EQ_HISTORY_DIR, cutoff) + prune_history(IDX_HISTORY_DIR, cutoff)
    if pruned:
        log(f"  pruned {pruned} session files older than {cutoff:%Y-%m-%d}")

    log(
        f"history sync: +{stats['eq_added']} equity, +{stats['idx_added']} index "
        f"sessions, {stats['misses']} unavailable"
    )
    return stats


def load_history(folder: str, key: str) -> pd.DataFrame:
    """
    Read every stored session into a wide frame: rows = dates, columns = keys.
    """
    sessions = stored_sessions(folder)
    if not sessions:
        return pd.DataFrame()

    parts: List[pd.DataFrame] = []
    for session in sessions:
        path = os.path.join(folder, f"{session:%Y-%m-%d}.csv.gz")
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if key not in frame.columns or "close" not in frame.columns:
            continue
        frame = frame[[key, "close"]].copy()
        frame["date"] = pd.Timestamp(session)
        parts.append(frame)

    if not parts:
        return pd.DataFrame()

    long_frame = pd.concat(parts, ignore_index=True)
    long_frame["close"] = pd.to_numeric(long_frame["close"], errors="coerce")
    long_frame = long_frame[long_frame["close"].notna() & (long_frame["close"] > 0)]
    wide = long_frame.pivot_table(
        index="date", columns=key, values="close", aggfunc="last"
    ).sort_index()
    return wide


# --------------------------------------------------------------------------
# Constituents - the only remaining per-index API call. Membership changes on
# a semi-annual review cycle, so it is refreshed weekly and carried forward in
# between. 139 calls a week instead of 139 a day.
# --------------------------------------------------------------------------

def read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_prior() -> None:
    """Read the last committed build so this run can only add to it."""
    indices = read_json(INDICES_FILE)
    if isinstance(indices, list):
        for row in indices:
            if isinstance(row, dict) and row.get("index"):
                PRIOR_INDICES[str(row["index"])] = row

    stocks = read_json(ALL_STOCKS_FILE)
    if isinstance(stocks, list):
        for row in stocks:
            if isinstance(row, dict) and row.get("symbol"):
                PRIOR_STOCKS[str(row["symbol"])] = row

    shards = read_json(STOCKS_MAP_FILE)
    if isinstance(shards, dict):
        PRIOR_SHARDS.update({str(k): str(v) for k, v in shards.items()})

    membership = read_json(MEMBERSHIP_FILE)
    if isinstance(membership, dict):
        for symbol, names in membership.items():
            if isinstance(names, list):
                PRIOR_MEMBERSHIP[str(symbol)] = [str(n) for n in names]

    overrides = read_json(CSV_OVERRIDES_FILE)
    if isinstance(overrides, dict):
        for name, url in overrides.items():
            if isinstance(url, str) and url.startswith("http"):
                CSV_OVERRIDES[str(name)] = url
                CSV_OVERRIDES[normalise_name(name)] = url
        if CSV_OVERRIDES:
            log(f"loaded {len(overrides)} constituent URL overrides")

    stored = read_json(CONSTITUENTS_FILE)
    if isinstance(stored, dict):
        for name, symbols in (stored.get("indices") or {}).items():
            if isinstance(symbols, list):
                PRIOR_CONSTITUENTS[str(name)] = [str(s) for s in symbols]
        for name, stamp in (stored.get("fetched_at") or {}).items():
            if stamp:
                PRIOR_CONSTITUENT_DATES[str(name)] = str(stamp)
        poisoned = 0
        for name, note in (stored.get("no_equity_constituents") or {}).items():
            name = str(name)
            if NON_EQUITY_INDEX.search(name):
                # Genuinely holds no equities. Keep it out of the queue.
                PRIOR_CONSTITUENT_EMPTY[name] = str(note)
                continue
            # Recorded empty by an earlier build that mistook a throttled
            # response for a real answer. Drop the stamp so it is retried.
            PRIOR_CONSTITUENT_DATES.pop(name, None)
            poisoned += 1
        if poisoned:
            log(f"cleared {poisoned} indices wrongly marked as having no constituents")
        for name, url in (stored.get("constituent_csv_url") or {}).items():
            PRIOR_CONSTITUENT_CSV[str(name)] = str(url)
        # Migrate a single whole-file stamp from the previous format.
        legacy = stored.get("fetched")
        if legacy and not PRIOR_CONSTITUENT_DATES:
            for name in PRIOR_CONSTITUENTS:
                PRIOR_CONSTITUENT_DATES[name] = str(legacy)

    if PRIOR_INDICES or PRIOR_STOCKS:
        log(
            f"carrying forward previous build: {len(PRIOR_INDICES)} indices, "
            f"{len(PRIOR_STOCKS)} stocks, {len(PRIOR_CONSTITUENTS)} constituent lists"
        )


def constituent_age_days(name: str) -> int:
    """Days since this index's membership was last fetched. Large if never."""
    stamp = PRIOR_CONSTITUENT_DATES.get(name)
    if not stamp:
        return 10 ** 6
    try:
        fetched = datetime.strptime(stamp, "%Y-%m-%d").date()
    except ValueError:
        return 10 ** 6
    return (datetime.now(IST).date() - fetched).days


def unwrap_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "records", "rows", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def constituent_csv_urls(index_name: str) -> List[str]:
    """
    Candidate constituent CSV URLs for an index, most likely first.

    NSE's naming is mostly ind_<alphanumeric lowercase>list.csv - NIFTY 50 ->
    ind_nifty50list.csv - but it also publishes underscore-separated forms, and
    strategy indices live on niftyindices.com rather than nsearchives. Both
    hosts and several name forms are tried; a hit is cached so later runs go
    straight to it.
    """
    override = CSV_OVERRIDES.get(str(index_name)) or CSV_OVERRIDES.get(
        normalise_name(index_name)
    )
    if override:
        return [override]

    raw = str(index_name).upper()
    forms: List[str] = []

    def add(text: str) -> None:
        compact = re.sub(r"[^A-Z0-9]+", "", text).lower()
        snake = re.sub(r"[^A-Z0-9]+", "_", text).strip("_").lower()
        for slug in (compact, snake):
            if slug and slug not in forms:
                forms.append(slug)
            if slug.endswith("index"):
                trimmed = slug[:-5].rstrip("_")
                if trimmed and trimmed not in forms:
                    forms.append(trimmed)

    add(raw)
    add(raw.replace("&", " AND "))
    add(re.sub(r"\s*\(.*?\)", "", raw))          # drop parenthetical suffixes
    add(raw.replace("-", " "))

    urls: List[str] = []
    for form in forms[:5]:
        for base in INDEX_CSV_BASES:
            urls.append(f"{base}/ind_{form}list.csv")
    return urls[:CSV_PROBE_LIMIT]


def parse_constituent_csv(path: str, index_name: str) -> List[str]:
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return []
    frame.columns = [str(c).strip() for c in frame.columns]
    symbol_col = _pick_column(frame, ("SYMBOL", "TCKRSYMB"))
    if not symbol_col:
        return []
    series_col = _pick_column(frame, ("SERIES", "SCTYSRS"))
    if series_col:
        values = frame[series_col].astype(str).str.strip().str.upper()
        keep = frame[values.isin(EQUITY_SERIES)]
        if not keep.empty:
            frame = keep
    symbols: List[str] = []
    seen = set()
    normalised_index = normalise_name(index_name)
    for value in frame[symbol_col].astype(str):
        symbol = value.strip().upper()
        if not symbol or symbol in seen or normalise_name(symbol) == normalised_index:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def fetch_constituents_csv(
    client: "NSE", index_name: str, scratch: str, known_url: Optional[str] = None
) -> Tuple[List[str], str]:
    """Archive CSV fallback. Returns (symbols, url_that_worked)."""
    urls = [known_url] if known_url else constituent_csv_urls(index_name)
    for url in urls:
        if not url:
            continue
        try:
            if THROTTLE_SECONDS > 0:
                time.sleep(THROTTLE_SECONDS)
            path = client.download_document(url, folder=scratch)
        except Exception:  # noqa: BLE001 - a 404 here just means wrong guess
            continue
        symbols = parse_constituent_csv(str(path), index_name)
        try:
            os.remove(str(path))
        except OSError:
            pass
        if symbols:
            return symbols, url
    return [], ""


def fetch_constituents(
    client: "NSE",
    index_name: str,
    scratch: str,
    api_name: Optional[str] = None,
    known_csv: Optional[str] = None,
) -> Tuple[List[str], str, str]:
    """
    Fetch one index's equity membership.

    The archive CSV is tried first. It is a static file on the same host that
    serves the bhavcopies, which tolerates far more volume than the live API.
    The live endpoint is the fallback, not the default.

    Returns (symbols, status, note). Status is one of:
      ok        - rows returned. note holds the CSV URL when the archive
                  supplied them, so later runs skip straight to it.
      empty     - a known non-equity index. Correct and final.
      deferred  - nothing came back, but this index should have constituents.
                  NOT stamped, so it stays in the queue and is retried. This
                  is what a throttled response looks like.
    """
    if NON_EQUITY_INDEX.search(str(index_name)):
        return [], "empty", "non-equity index family, not probed"

    symbols, url = fetch_constituents_csv(client, index_name, scratch, known_csv)
    if symbols:
        return symbols, "ok", url

    query = api_name or index_name
    payload: Any = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if CONSTITUENT_THROTTLE > 0:
                time.sleep(CONSTITUENT_THROTTLE)
            payload = client.listEquityStocksByIndex(query)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_ATTEMPTS:
                note = f"{type(exc).__name__}: {exc}"
                FAILURES.append(
                    {"target": f"constituents of '{index_name}'", "reason": note}
                )
                return [], "deferred", note
            time.sleep(5.0 * (2 ** (attempt - 1)))

    rows = unwrap_records(payload)
    symbols = []
    seen = set()
    normalised_index = normalise_name(index_name)
    for row in rows:
        symbol = row.get("symbol") or row.get("Symbol") or row.get("SYMBOL")
        if not symbol:
            continue
        symbol = str(symbol).strip().upper()
        if not symbol or normalise_name(symbol) == normalised_index:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)

    if symbols:
        return symbols, "ok", ""

    if isinstance(payload, dict):
        shape = f"dict keys={sorted(payload.keys())[:8]} rows={len(rows)}"
    elif isinstance(payload, list):
        shape = f"list len={len(payload)}"
    else:
        shape = f"{type(payload).__name__}"
    return [], "deferred", f"live '{query}' -> {shape} (throttled or unsupported)"


def build_api_name_map(client: "NSE", archive_names: List[str]) -> Dict[str, str]:
    """
    One listIndices call per run, purely to translate archive index names into
    the exact strings NSE's constituents endpoint accepts. The archive writes
    'Nifty India Digital'; the endpoint wants its own spelling. Matching is on
    the alphanumeric-only form, so casing and punctuation differences resolve.
    """
    try:
        if THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)
        payload = client.listIndices()
    except Exception as exc:  # noqa: BLE001
        log(f"  ~ listIndices unavailable ({type(exc).__name__}); using archive names")
        FAILURES.append({"target": "listIndices", "reason": str(exc)})
        return {}

    api_names: List[str] = []
    for row in unwrap_records(payload):
        name = row.get("index") or row.get("indexName") or row.get("key")
        if name:
            api_names.append(str(name).strip())

    by_key = {normalise_name(n): n for n in api_names}
    mapping = {}
    for archive_name in archive_names:
        match = by_key.get(normalise_name(archive_name))
        if match and match != archive_name:
            mapping[archive_name] = match
    log(f"  listIndices: {len(api_names)} names, {len(mapping)} differ from the archive")
    return mapping


def refresh_constituents(
    client: "NSE",
    index_names: List[str],
    scratch: str,
    new_client: Optional[Callable[[], "NSE"]] = None,
) -> Tuple[Dict[str, List[str]], "NSE"]:
    """
    Refresh index membership in a small aged batch.

    Only two outcomes stamp a date: a real list, and a known non-equity index.
    Anything else is deferred, keeps no date, and is retried on the next run.
    That is what stops a throttled empty response from being cached as fact.

    The session is recycled every SESSION_RECYCLE_EVERY calls, and the batch
    stops early after DEFER_STREAK_LIMIT consecutive deferrals - once NSE is
    throttling, further calls only waste time.
    """
    held: Dict[str, List[str]] = dict(PRIOR_CONSTITUENTS)
    dates: Dict[str, str] = dict(PRIOR_CONSTITUENT_DATES)
    empties: Dict[str, str] = dict(PRIOR_CONSTITUENT_EMPTY)
    csv_urls: Dict[str, str] = dict(PRIOR_CONSTITUENT_CSV)
    deferred_notes: Dict[str, str] = {}

    if FORCE_CONSTITUENTS:
        due = [n for n in index_names if not NON_EQUITY_INDEX.search(n)]
    else:
        due = [n for n in index_names if constituent_age_days(n) >= CONSTITUENT_MAX_AGE]
    # Indices with no list at all come first; a stale list still renders.
    due.sort(key=lambda n: (1 if held.get(n) else 0, -constituent_age_days(n), n))

    capped = due[:CONSTITUENT_BATCH] if CONSTITUENT_BATCH > 0 else due

    if not capped:
        log(f"constituents current - reusing {len(held)} stored lists")
        return held, client

    api_map = build_api_name_map(client, capped)

    log(
        f"constituents: {len(due)} of {len(index_names)} due, attempting "
        f"{len(capped)} this run (batch={CONSTITUENT_BATCH or 'all'}, "
        f"throttle={CONSTITUENT_THROTTLE}s)"
    )

    today = datetime.now(IST).strftime("%Y-%m-%d")
    counts = {"ok": 0, "empty": 0, "deferred": 0, "via_csv": 0}
    defer_streak = 0
    calls = 0

    for position, name in enumerate(capped, start=1):
        if new_client and calls and calls % SESSION_RECYCLE_EVERY == 0:
            log(f"  recycling NSE session after {calls} calls, {SESSION_COOLDOWN:.0f}s cooldown")
            try:
                client.exit()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(SESSION_COOLDOWN)
            try:
                client = new_client()
            except Exception as exc:  # noqa: BLE001
                log(f"  ! could not reopen session ({type(exc).__name__}); stopping batch")
                break

        symbols, status, note = fetch_constituents(
            client, name, scratch, api_map.get(name), csv_urls.get(name)
        )
        calls += 1
        counts[status] += 1

        if status == "ok":
            held[name] = symbols
            dates[name] = today
            empties.pop(name, None)
            deferred_notes.pop(name, None)
            defer_streak = 0
            if note.startswith("http"):
                csv_urls[name] = note
                counts["via_csv"] += 1
            log(f"  [{position}/{len(capped)}] ok       {name} ({len(symbols)})")
        elif status == "empty":
            held.setdefault(name, [])
            dates[name] = today
            empties[name] = note
            defer_streak = 0
        else:
            deferred_notes[name] = note
            defer_streak += 1
            log(f"  [{position}/{len(capped)}] deferred {name}")
            if defer_streak >= DEFER_STREAK_LIMIT:
                log(
                    f"  stopping batch after {defer_streak} consecutive deferrals "
                    f"- NSE is throttling. The rest stay queued for the next run."
                )
                break

    write_json(
        CONSTITUENTS_FILE,
        {
            "updated": today,
            "refreshed_this_run": counts["ok"],
            "from_archive_csv": counts["via_csv"],
            "deferred_this_run": counts["deferred"],
            "non_equity_this_run": counts["empty"],
            "lists_held": len(held),
            "lists_with_stocks": sum(1 for v in held.values() if v),
            "fetched_at": dates,
            "constituent_csv_url": csv_urls,
            "no_equity_constituents": empties,
            "deferred": deferred_notes,
            "indices": held,
        },
    )

    populated = sum(1 for v in held.values() if v)
    outstanding = len(due) - counts["ok"] - counts["empty"]
    log(
        f"constituents: ok={counts['ok']} (csv={counts['via_csv']}) "
        f"non-equity={counts['empty']} deferred={counts['deferred']}, "
        f"{populated}/{len(held)} indices now have a stock list, "
        f"{outstanding} still due"
    )
    return held, client


# --------------------------------------------------------------------------
# Persistence - stocks are sharded one file per index so daily git diffs stay
# small and the detail page loads only the shard it needs.
# --------------------------------------------------------------------------

def ensure_dirs() -> None:
    for folder in (DATA_DIR, STOCKS_DIR, HISTORY_DIR, EQ_HISTORY_DIR, IDX_HISTORY_DIR):
        os.makedirs(folder, exist_ok=True)


def write_json(path: str, payload: Any, compact: bool = False) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            if compact:
                json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
            else:
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
    path = os.path.join(STOCKS_DIR, filename)
    if not rows:
        existing = read_json(path)
        if isinstance(existing, list) and existing:
            return filename
    write_json(path, rows, compact=True)
    return filename


def merge_rows(
    prior: Dict[str, Dict[str, Any]],
    fresh: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fresh rows win. Anything not refreshed this run is kept and flagged."""
    merged: Dict[str, Dict[str, Any]] = {}
    for key, row in prior.items():
        carried = dict(row)
        carried["stale"] = True
        merged[key] = carried
    for key, row in fresh.items():
        current = dict(row)
        current["stale"] = False
        merged[key] = current
    return list(merged.values())


def write_outputs(
    index_rows: Dict[str, Dict[str, Any]],
    stock_rows: Dict[str, Dict[str, Any]],
    shard_map: Dict[str, str],
    history_stats: Dict[str, int],
    started: float,
) -> Tuple[int, int]:
    merged_indices = merge_rows(PRIOR_INDICES, index_rows)
    merged_indices.sort(key=lambda r: str(r.get("index", "")))
    write_json(INDICES_FILE, merged_indices, compact=True)

    merged_shards = dict(PRIOR_SHARDS)
    merged_shards.update(shard_map)
    write_json(STOCKS_MAP_FILE, merged_shards)

    merged_stocks = merge_rows(PRIOR_STOCKS, stock_rows)
    merged_stocks.sort(key=lambda r: str(r.get("symbol", "")))
    write_json(ALL_STOCKS_FILE, merged_stocks, compact=True)

    merged_membership = dict(PRIOR_MEMBERSHIP)
    merged_membership.update(
        {symbol: sorted(set(names)) for symbol, names in MEMBERSHIP.items()}
    )
    write_json(MEMBERSHIP_FILE, merged_membership, compact=True)

    write_json(
        LAST_UPDATED_FILE,
        {
            "last_updated": datetime.now(IST).isoformat(timespec="seconds"),
            "last_updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "NSE daily archive bhavcopy",
            "indices_count": len(merged_indices),
            "indices_refreshed": len(index_rows),
            "indices_carried_forward": len(merged_indices) - len(index_rows),
            "indices_with_stocks": len(merged_shards),
            "unique_symbols": len(merged_stocks),
            "symbols_refreshed": len(stock_rows),
            "symbols_adjusted": len(ADJUSTMENTS),
            "equity_sessions_stored": len(stored_sessions(EQ_HISTORY_DIR)),
            "index_sessions_stored": len(stored_sessions(IDX_HISTORY_DIR)),
            "equity_sessions_added": history_stats.get("eq_added", 0),
            "index_sessions_added": history_stats.get("idx_added", 0),
            "failures": len(FAILURES),
            "runtime_minutes": round((time.time() - started) / 60.0, 1),
            "complete": True,
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
    return len(merged_indices), len(merged_stocks)


# --------------------------------------------------------------------------
# Row builders - pure computation over the history frames, no network
# --------------------------------------------------------------------------

def series_for(frame: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if frame.empty or key not in frame.columns:
        return None
    series = frame[key].dropna()
    return series if len(series) >= 2 else None


def build_index_row(name: str, series: pd.Series) -> Dict[str, Any]:
    close, as_of, returns = compute_returns(series)
    row: Dict[str, Any] = {
        "index": name,
        "close": close,
        "as_of": as_of,
        "sessions": int(len(series)),
        "from": series.index[0].strftime("%Y-%m-%d"),
    }
    row.update(returns)
    row.update(window_extremes(series))
    return row


def build_stock_row(symbol: str, series: pd.Series) -> Optional[Dict[str, Any]]:
    adjusted, events = adjust_for_corporate_actions(series)
    close, as_of, returns = compute_returns(adjusted)
    if close is None:
        return None

    if events:
        ADJUSTMENTS[symbol] = events

    row: Dict[str, Any] = {
        "symbol": symbol,
        "close": close,
        "as_of": as_of,
        "adjusted": bool(events),
        "sessions": int(len(adjusted)),
        "from": adjusted.index[0].strftime("%Y-%m-%d"),
    }
    row.update(returns)
    row.update(window_extremes(adjusted))
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    started = time.time()
    ensure_dirs()
    log("NSE Index Returns Dashboard - daily data build starting")
    log(
        f"config: history={HISTORY_DAYS}d throttle={THROTTLE_SECONDS}s "
        f"attempts={MAX_ATTEMPTS} max_new_days={MAX_NEW_DAYS or 'all'} "
        f"max_indices={MAX_INDICES or 'all'} skip_stocks={SKIP_STOCKS} "
        f"corp_action_band={CORP_ACTION_BAND:.0%}"
    )

    load_prior()

    scratch = tempfile.mkdtemp(prefix="nse_dl_")

    def open_client() -> "NSE":
        return NSE(download_folder=scratch, server=True)

    try:
        client = open_client()
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: could not initialise NSE client: {type(exc).__name__}: {exc}")
        FAILURES.append({"target": "NSE client init", "reason": str(exc)})
        log("previous build left untouched")
        return 1

    history_stats: Dict[str, int] = {}
    try:
        history_stats = sync_history(client, scratch)
    except Exception as exc:  # noqa: BLE001
        log(f"! history sync aborted: {type(exc).__name__}: {exc}")
        FAILURES.append({"target": "history sync", "reason": str(exc)})

    index_frame = load_history(IDX_HISTORY_DIR, "index")
    equity_frame = load_history(EQ_HISTORY_DIR, "symbol")
    log(
        f"history loaded: indices {index_frame.shape[1]} x {index_frame.shape[0]} sessions, "
        f"equities {equity_frame.shape[1]} x {equity_frame.shape[0]} sessions"
    )

    if index_frame.empty and equity_frame.empty:
        log("! no history available - previous build left untouched")
        try:
            client.exit()
        except Exception:  # noqa: BLE001
            pass
        return 1

    # Index names come from the archive file itself, so listIndices is no
    # longer needed. Keep the name each index already carries in the committed
    # build so shards, membership and the app's links stay stable.
    prior_by_key = {normalise_name(n): n for n in PRIOR_INDICES}
    index_names: List[str] = []
    bhav_to_canonical: Dict[str, str] = {}
    for raw_name in index_frame.columns:
        canonical = prior_by_key.get(normalise_name(raw_name), str(raw_name))
        bhav_to_canonical[str(raw_name)] = canonical
        index_names.append(canonical)
    index_names = sorted(set(index_names))
    if MAX_INDICES > 0:
        index_names = index_names[:MAX_INDICES]
    log(f"{len(index_names)} indices in the archive file")

    constituents: Dict[str, List[str]] = {}
    if not SKIP_STOCKS:
        try:
            constituents, client = refresh_constituents(
                client, index_names, scratch, open_client
            )
        except Exception as exc:  # noqa: BLE001
            log(f"! constituent refresh aborted: {type(exc).__name__}: {exc}")
            constituents = dict(PRIOR_CONSTITUENTS)

    try:
        client.exit()
    except Exception:  # noqa: BLE001
        pass

    # ---- indices -------------------------------------------------------
    canonical_to_bhav = {v: k for k, v in bhav_to_canonical.items()}
    index_rows: Dict[str, Dict[str, Any]] = {}
    for name in index_names:
        series = series_for(index_frame, canonical_to_bhav.get(name, name))
        if series is None:
            log(f"    ! no usable history for index '{name}'")
            continue
        index_rows[name] = build_index_row(name, series)

    log(f"{len(index_rows)}/{len(index_names)} indices computed")

    # ---- stocks --------------------------------------------------------
    stock_rows: Dict[str, Dict[str, Any]] = {}
    shard_map: Dict[str, str] = {}
    missing_symbols: List[str] = []

    if not SKIP_STOCKS:
        wanted: set = set()
        for name in index_names:
            for symbol in constituents.get(name, []):
                wanted.add(symbol)
        log(f"{len(wanted)} unique constituent symbols to compute")

        for symbol in sorted(wanted):
            series = series_for(equity_frame, symbol)
            if series is None:
                missing_symbols.append(symbol)
                continue
            row = build_stock_row(symbol, series)
            if row:
                stock_rows[symbol] = row

        for name in index_names:
            rows = [
                dict(stock_rows[s])
                for s in constituents.get(name, [])
                if s in stock_rows
            ]
            for row in rows:
                MEMBERSHIP.setdefault(str(row["symbol"]), []).append(name)
            shard_map[name] = write_stock_shard(name, rows)

        log(
            f"{len(stock_rows)}/{len(wanted)} symbols computed, "
            f"{len(missing_symbols)} absent from the bhavcopy"
        )
        if missing_symbols:
            log(f"    absent sample: {', '.join(missing_symbols[:12])}")
        if ADJUSTMENTS:
            sample = list(ADJUSTMENTS.items())[:5]
            for symbol, events in sample:
                ratios = ", ".join(f"{e['date']} x{e['ratio']:.4f}" for e in events)
                log(f"    * {symbol}: corporate action adjusted ({ratios})")

    total_indices, total_stocks = write_outputs(
        index_rows, stock_rows, shard_map, history_stats, started
    )

    if stock_rows:
        spans = sorted(row["sessions"] for row in stock_rows.values())
        thin = sum(1 for n in spans if n < 150)
        log(
            f"coverage: median {spans[len(spans) // 2]} sessions per symbol, "
            f"min {spans[0]}, max {spans[-1]}, {thin} symbols under 150 sessions"
        )

    if PRIOR_INDICES and len(index_rows) < len(PRIOR_INDICES) * 0.8:
        log(
            f"! WARNING: only {len(index_rows)} of {len(PRIOR_INDICES)} known "
            f"indices refreshed. The rest were carried forward and marked stale."
        )

    log(
        f"done in {(time.time() - started) / 60:.1f} min - "
        f"{len(index_rows)} indices refreshed ({total_indices} held), "
        f"{len(shard_map)} shards, {len(stock_rows)} symbols refreshed "
        f"({total_stocks} held), {len(ADJUSTMENTS)} corporate-action adjustments, "
        f"{len(FAILURES)} failures"
    )
    shutil.rmtree(scratch, ignore_errors=True)
    # Exit 0 even on partial success so the Action still commits what it got.
    return 0


if __name__ == "__main__":
    sys.exit(main())
