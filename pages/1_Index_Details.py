"""
pages/1_Index_Details.py
========================
Detail page for a single NSE index.

Reads data/indices.json and data/stocks.json from disk only.
Makes no live NSE calls.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Index Details",
    page_icon=":bar_chart:",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")

RETURN_COLS = ["1D", "3D", "1W", "2W", "1M", "2M", "3M", "6M", "1Y"]
HOME_PAGE = "app.py"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_json(filename: str) -> Optional[Any]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if payload in (None, [], {}, ""):
        return None
    return payload


def format_last_updated(payload: Optional[dict]) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    raw = payload.get("last_updated")
    if not raw:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        return str(raw)
    return parsed.strftime("%d %b %Y, %H:%M IST")


def stock_column_config() -> dict:
    config = {
        "symbol": st.column_config.TextColumn("Symbol", width="medium"),
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "as_of": st.column_config.TextColumn("As of"),
    }
    for column in RETURN_COLS:
        config[column] = st.column_config.NumberColumn(column, format="%.2f%%")
    return config


def back_button(key: str) -> None:
    if st.button(":arrow_left: Back to all indices", key=key):
        st.switch_page(HOME_PAGE)


# --------------------------------------------------------------------------
# Guard: an index must have been selected
# --------------------------------------------------------------------------

selected_index = st.session_state.get("selected_index")

if not selected_index:
    st.warning("No index selected. Go back and pick one from the list.")
    back_button("back_no_selection")
    st.stop()

back_button("back_top")
st.title(str(selected_index))

meta = load_json("last_updated.json")
st.caption(f"Data last updated: **{format_last_updated(meta)}**")

indices_data = load_json("indices.json")

if not indices_data:
    st.warning(
        "No data yet — waiting for the first scheduled run. "
        "Trigger *Update NSE data* from the repo's **Actions** tab to build it now."
    )
    st.stop()

index_row = next(
    (
        row
        for row in indices_data
        if isinstance(row, dict) and str(row.get("index", "")) == str(selected_index)
    ),
    None,
)

if index_row is None:
    st.error(f"'{selected_index}' is not present in `data/indices.json`.")
    back_button("back_missing_index")
    st.stop()


# --------------------------------------------------------------------------
# Index-level returns
# --------------------------------------------------------------------------

summary_left, summary_right = st.columns(2)
with summary_left:
    close = index_row.get("close")
    st.metric("Close", f"{close:,.2f}" if isinstance(close, (int, float)) else "—")
with summary_right:
    st.metric("As of", index_row.get("as_of") or "—")

st.subheader("Index returns")


def render_metrics(row: dict, labels: list) -> None:
    columns = st.columns(len(labels))
    for column, label in zip(columns, labels):
        value = row.get(label)
        if isinstance(value, (int, float)):
            column.metric(label, f"{value:.2f}%", delta=f"{value:.2f}%")
        else:
            column.metric(label, "—")


render_metrics(index_row, RETURN_COLS[:5])
render_metrics(index_row, RETURN_COLS[5:])

st.divider()


# --------------------------------------------------------------------------
# Constituent stocks
# --------------------------------------------------------------------------

st.subheader("Constituent stocks")

stocks_data = load_json("stocks.json")
constituents = []
if isinstance(stocks_data, dict):
    constituents = stocks_data.get(str(selected_index)) or []

if not constituents:
    st.info(
        "No constituent data for this index. "
        "This is expected for non-equity indices (G-Sec, fixed income, strategy indices), "
        "or the constituents may not have been fetched in the last run."
    )
    back_button("back_no_constituents")
    st.stop()

frame = pd.DataFrame(constituents)

if "symbol" not in frame.columns:
    st.error("`data/stocks.json` is malformed — no `symbol` column found.")
    back_button("back_malformed_stocks")
    st.stop()

for column in RETURN_COLS + ["close"]:
    if column not in frame.columns:
        frame[column] = None
    frame[column] = pd.to_numeric(frame[column], errors="coerce")

if "as_of" not in frame.columns:
    frame["as_of"] = None

frame = frame[["symbol", "close", "as_of"] + RETURN_COLS]

search = st.text_input(
    "Search symbols",
    value="",
    placeholder="e.g. HDFCBANK, INFY",
    label_visibility="collapsed",
)

view = frame
if search.strip():
    view = frame[frame["symbol"].str.contains(search.strip(), case=False, na=False)]

if view.empty:
    st.info(f"No symbol matches '{search}'.")
else:
    st.caption(f"{len(view)} of {len(frame)} constituents")
    st.dataframe(
        view.reset_index(drop=True),
        column_config=stock_column_config(),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
back_button("back_bottom")

st.caption(
    "Source: NSE India official EOD API via the `nse` package. "
    "Returns are computed against the closest earlier trading day for each period. "
    "Blank cells mean history was unavailable for that period."
)
