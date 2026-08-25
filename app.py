"""
app.py
======
NSE Index Returns Dashboard - home page.

Reads data/indices.json and data/last_updated.json from disk only.
Makes no live NSE calls. Data is refreshed by update_data.py, which runs
daily via GitHub Actions and commits the JSON back to the repo.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NSE Index Returns Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

RETURN_COLS = ["1D", "3D", "1W", "2W", "1M", "2M", "3M", "6M", "1Y"]
DETAIL_PAGE = "pages/1_Index_Details.py"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_json(filename: str) -> Optional[Any]:
    """Load a JSON file from data/. Returns None if missing, empty or invalid."""
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


def return_column_config() -> dict:
    config = {
        "index": st.column_config.TextColumn("Index", width="large"),
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "as_of": st.column_config.TextColumn("As of"),
    }
    for column in RETURN_COLS:
        config[column] = st.column_config.NumberColumn(column, format="%.2f%%")
    return config


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("NSE Index Returns Dashboard")

indices_data = load_json("indices.json")
meta = load_json("last_updated.json")

if not indices_data:
    st.warning(
        "No data yet — waiting for the first scheduled run.\n\n"
        "The daily GitHub Actions job builds `data/indices.json`. "
        "Trigger it manually from the repo's **Actions** tab "
        "(*Update NSE data* → *Run workflow*) if you don't want to wait."
    )
    st.stop()

frame = pd.DataFrame(indices_data)

if "index" not in frame.columns:
    st.error("`data/indices.json` is malformed — no `index` column found.")
    st.stop()

for column in RETURN_COLS + ["close"]:
    if column not in frame.columns:
        frame[column] = None
    frame[column] = pd.to_numeric(frame[column], errors="coerce")

if "as_of" not in frame.columns:
    frame["as_of"] = None

frame = frame[["index", "close", "as_of"] + RETURN_COLS]

header_left, header_right = st.columns([3, 1])
with header_left:
    st.caption(f"Data last updated: **{format_last_updated(meta)}**")
    if isinstance(meta, dict) and meta.get("complete") is False:
        st.caption(":warning: Last run finished early — this snapshot may be partial.")
with header_right:
    st.metric("Indices tracked", len(frame))

st.divider()

search = st.text_input(
    "Search indices",
    value="",
    placeholder="e.g. NIFTY BANK, MIDCAP, AUTO",
    label_visibility="collapsed",
)

view = frame
if search.strip():
    view = frame[frame["index"].str.contains(search.strip(), case=False, na=False)]

if view.empty:
    st.info(f"No index matches '{search}'.")
    st.stop()

view = view.reset_index(drop=True)

st.write("Click a row to open that index's constituent detail.")

event = st.dataframe(
    view,
    column_config=return_column_config(),
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
    key="indices_table",
)


def open_index(name: str) -> None:
    st.session_state["selected_index"] = name
    st.switch_page(DETAIL_PAGE)


selected_rows = []
if event is not None and getattr(event, "selection", None):
    selected_rows = event.selection.get("rows", []) or []

if selected_rows:
    position = selected_rows[0]
    if 0 <= position < len(view):
        open_index(str(view.iloc[position]["index"]))

st.divider()

fallback_left, fallback_right = st.columns([3, 1])
with fallback_left:
    manual_choice = st.selectbox(
        "Or pick an index",
        options=view["index"].tolist(),
        index=0,
        key="manual_index_choice",
    )
with fallback_right:
    st.write("")
    st.write("")
    if st.button("View details", use_container_width=True, type="primary"):
        open_index(str(manual_choice))

st.caption(
    "Source: NSE India official EOD API via the `nse` package. "
    "Returns are computed against the closest earlier trading day for each period. "
    "Blank cells mean history was unavailable for that period."
)
