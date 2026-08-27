"""
probe_constituents.py
=====================
One-off discovery job. Finds the real constituent-CSV URL for every index that
has no stock list, and writes the answers to data/constituent_url_overrides.json
so update_data.py can use them.

This exists because NSE's constituent file names cannot be derived reliably.
Rather than guessing, this probes a wide candidate set from the GitHub runner -
the only machine that can actually reach NSE - and records what works.

Run it once from the Actions tab. It changes nothing except the overrides file.

Environment overrides:
    PROBE_THROTTLE      seconds between probes            (default 0.4)
    PROBE_LIMIT         max URLs tried per index          (default 40)
    PROBE_ONLY          comma-separated index names, or   (default all missing)
                        blank for every index with no list
    PROBE_INCLUDE_OK    1 = also probe indices that       (default 0)
                        already have a list
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from nse import NSE
except ImportError:  # pragma: no cover
    sys.stderr.write("FATAL: the `nse` package is not installed.\n")
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
INDICES_FILE = os.path.join(DATA_DIR, "indices.json")
CONSTITUENTS_FILE = os.path.join(DATA_DIR, "constituents.json")
OVERRIDES_FILE = os.path.join(DATA_DIR, "constituent_url_overrides.json")
REPORT_FILE = os.path.join(DATA_DIR, "constituent_probe_report.json")

THROTTLE = float(os.getenv("PROBE_THROTTLE", "0.4"))
PROBE_LIMIT = int(os.getenv("PROBE_LIMIT", "40"))
PROBE_ONLY = [s.strip() for s in os.getenv("PROBE_ONLY", "").split(",") if s.strip()]
PROBE_INCLUDE_OK = os.getenv("PROBE_INCLUDE_OK", "0") == "1"

BASES = [
    "https://nsearchives.nseindia.com/content/indices",
    "https://www.niftyindices.com/IndexConstituent",
    "https://niftyindices.com/IndexConstituent",
    "https://archives.nseindia.com/content/indices",
]

SUFFIXES = ["list.csv", "_list.csv", ".csv"]

NON_EQUITY = re.compile(
    r"G-SEC|GSEC|BHARAT BOND|INDIA VIX|1D RATE|DIVIDEND POINTS|"
    r"PR 1X|PR 2X|TR 1X|TR 2X|\bUSD\b|ARBITRAGE|FUTURES INDEX|FUTURES TR|"
    r"CLEAN PRICE",
    re.IGNORECASE,
)


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1, sort_keys=True)


def name_forms(index_name: str) -> List[str]:
    """Every plausible file-name stem for this index, most likely first."""
    raw = str(index_name).upper().strip()

    bases: List[str] = [raw]
    bases.append(raw.replace("&", " AND "))
    bases.append(raw.replace("&", " "))
    bases.append(re.sub(r"\s*\(.*?\)", "", raw))          # drop "(MAATR)" etc
    bases.append(re.sub(r"\bINDEX\b", "", raw))
    bases.append(re.sub(r"\bINDEX\b", "", raw.replace("&", " AND ")))
    bases.append(raw.replace(":", " "))
    bases.append(raw.replace("/", " "))
    bases.append(raw.replace("%", " "))
    # NSE writes some as "NIFTY500" and others as "NIFTY 500"
    bases.append(re.sub(r"\bNIFTY\s*(\d)", r"NIFTY\1", raw))
    bases.append(re.sub(r"\bNIFTY(\d)", r"NIFTY \1", raw))

    forms: List[str] = []
    for base in bases:
        base = re.sub(r"\s+", " ", base).strip()
        if not base:
            continue
        compact = re.sub(r"[^A-Z0-9]+", "", base).lower()
        snake = re.sub(r"[^A-Z0-9]+", "_", base).strip("_").lower()
        for form in (compact, snake):
            if form and form not in forms:
                forms.append(form)
    return forms


def candidate_urls(index_name: str) -> List[str]:
    urls: List[str] = []
    for form in name_forms(index_name):
        for suffix in SUFFIXES:
            for base in BASES:
                url = f"{base}/ind_{form}{suffix}"
                if url not in urls:
                    urls.append(url)
    return urls[:PROBE_LIMIT]


def csv_symbols(path: str) -> List[str]:
    """Symbols from a constituent CSV, or [] if the file isn't one."""
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001 - a 404 page parses as garbage
        return []
    frame.columns = [str(c).strip() for c in frame.columns]
    lookup = {re.sub(r"[^A-Z0-9]+", "", str(c).upper()): str(c) for c in frame.columns}
    symbol_col = lookup.get("SYMBOL") or lookup.get("TCKRSYMB")
    if not symbol_col:
        return []
    out: List[str] = []
    seen = set()
    for value in frame[symbol_col].astype(str):
        symbol = value.strip().upper()
        if not symbol or symbol == "NAN" or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def probe(client: "NSE", url: str, scratch: str) -> Optional[List[str]]:
    try:
        if THROTTLE > 0:
            time.sleep(THROTTLE)
        path = client.download_document(url, folder=scratch)
    except Exception:  # noqa: BLE001
        return None
    symbols = csv_symbols(str(path))
    try:
        os.remove(str(path))
    except OSError:
        pass
    return symbols or None


def main() -> int:
    indices = read_json(INDICES_FILE)
    if not isinstance(indices, list) or not indices:
        log("FATAL: data/indices.json is missing or empty. Run the main build first.")
        return 1

    stored = read_json(CONSTITUENTS_FILE) or {}
    held: Dict[str, List[str]] = stored.get("indices") or {}

    all_names = [str(row["index"]) for row in indices if row.get("index")]

    if PROBE_ONLY:
        targets = [n for n in all_names if n in PROBE_ONLY]
        missing = [n for n in PROBE_ONLY if n not in all_names]
        if missing:
            log(f"! not found in indices.json, skipped: {missing}")
    else:
        targets = [
            name
            for name in all_names
            if (PROBE_INCLUDE_OK or not held.get(name))
            and not NON_EQUITY.search(name)
        ]

    if not targets:
        log("Nothing to probe - every equity index already has a stock list.")
        return 0

    log(f"probing {len(targets)} indices, up to {PROBE_LIMIT} URLs each")
    log(f"worst case {len(targets) * PROBE_LIMIT} requests at {THROTTLE}s")

    scratch = tempfile.mkdtemp(prefix="probe_")
    try:
        client = NSE(download_folder=scratch, server=True)
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: could not initialise NSE client: {exc}")
        return 1

    existing = read_json(OVERRIDES_FILE)
    overrides: Dict[str, str] = {
        k: v
        for k, v in (existing or {}).items()
        if isinstance(v, str) and v.startswith("http")
    }
    report: Dict[str, Any] = {"found": {}, "not_found": {}}
    requests_made = 0

    for position, name in enumerate(targets, start=1):
        urls = candidate_urls(name)
        hit_url = None
        hit_symbols: List[str] = []
        for url in urls:
            requests_made += 1
            symbols = probe(client, url, scratch)
            if symbols:
                hit_url, hit_symbols = url, symbols
                break

        if hit_url:
            overrides[name] = hit_url
            report["found"][name] = {"url": hit_url, "symbols": len(hit_symbols)}
            log(f"[{position}/{len(targets)}] OK   {name} -> {hit_url} ({len(hit_symbols)})")
            write_json(OVERRIDES_FILE, overrides)
        else:
            report["not_found"][name] = urls[:6]
            log(f"[{position}/{len(targets)}] MISS {name} ({len(urls)} urls tried)")

    try:
        client.exit()
    except Exception:  # noqa: BLE001
        pass

    report["summary"] = {
        "probed": len(targets),
        "found": len(report["found"]),
        "not_found": len(report["not_found"]),
        "requests": requests_made,
    }
    write_json(REPORT_FILE, report)
    write_json(OVERRIDES_FILE, overrides)

    log("")
    log(f"found {len(report['found'])} of {len(targets)} in {requests_made} requests")
    if report["not_found"]:
        log("still missing:")
        for name in sorted(report["not_found"]):
            log(f"    {name}")
    log(f"wrote {OVERRIDES_FILE} and {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
