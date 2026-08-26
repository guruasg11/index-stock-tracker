"""
pages/2_Lookup.py
=================
Search any single index or stock and read all nine returns.

For a stock, also shows every index it belongs to and its return relative to
each of them — the excess is computed here from the same stored figures, so
no extra data is needed.

Reads data/*.json from disk only. Makes no live NSE calls.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Lookup",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")

RETURN_COLS = ["1D", "3D", "1W", "2W", "1M", "2M", "3M", "6M", "1Y"]
HORIZON_NAMES = {
    "1D": "1 day", "3D": "3 days", "1W": "1 week", "2W": "2 weeks",
    "1M": "1 month", "2M": "2 months", "3M": "3 months",
    "6M": "6 months", "1Y": "1 year",
}
HOME_PAGE = "app.py"
DETAIL_PAGE = "pages/1_Index_Details.py"

RANGE_COLS = ["wk52_high", "wk52_low", "from_high", "from_low"]
RANGE_LABELS = {
    "wk52_high": "52W high", "wk52_low": "52W low",
    "from_high": "% off high", "from_low": "% off low",
}


def nav_bar(current: str) -> None:
    """Explicit tabs - sidebar navigation is switched off in config.toml."""
    tabs = [
        ("Indices", "app.py"),
        ("Lookup", "pages/2_Lookup.py"),
        ("Compare", "pages/3_Compare.py"),
    ]
    columns = st.columns(len(tabs) + 4)
    for column, (label, target) in zip(columns, tabs):
        with column:
            st.button(
                label,
                key=f"nav_{label}",
                width="stretch",
                type="primary" if label == current else "secondary",
                disabled=label == current,
                on_click=None if label == current else st.switch_page,
                args=None if label == current else (target,),
            )


def range_column_config() -> Dict[str, Any]:
    return {
        "wk52_high": st.column_config.NumberColumn("52W high", format="%.2f"),
        "wk52_low": st.column_config.NumberColumn("52W low", format="%.2f"),
        "from_high": st.column_config.NumberColumn(
            "% off high", format="%.2f%%", help="Distance below the 52-week high"
        ),
        "from_low": st.column_config.NumberColumn(
            "% off low", format="%.2f%%", help="Distance above the 52-week low"
        ),
    }


INK = "#12263A"
RULE = "#DCE1E8"
PETROL = "#1F5673"
UP = (15, 122, 78)
DOWN = (192, 57, 43)


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');

        html, body, [class*="st-"] {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
        .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
        #MainMenu, footer {{ visibility: hidden; }}

        .masthead {{
            border-top: 3px solid {INK}; border-bottom: 1px solid {RULE};
            padding: 0.9rem 0 1.1rem 0; margin-bottom: 1.4rem;
            display: flex; flex-wrap: wrap;
            align-items: baseline; justify-content: space-between; gap: 1rem;
        }}
        .masthead .eyebrow {{
            font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
            letter-spacing: 0.16em; text-transform: uppercase; color: {PETROL};
            display: block; margin-bottom: 0.35rem;
        }}
        .masthead h1 {{
            font-family: 'IBM Plex Serif', Georgia, serif; font-size: 2.05rem;
            font-weight: 600; line-height: 1.1; color: {INK}; margin: 0; padding: 0;
        }}
        .masthead .stamp {{
            font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem;
            color: #5B6B7C; text-align: right; line-height: 1.6;
        }}
        .masthead .stamp b {{ color: {INK}; font-weight: 600; }}

        .chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.2rem 0 1.4rem 0; }}
        .chip {{
            flex: 1 1 104px; border: 1px solid {RULE}; border-top: 2px solid {INK};
            background: #FFFFFF; padding: 0.6rem 0.7rem 0.65rem 0.7rem;
        }}
        .chip .k {{
            font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
            letter-spacing: 0.12em; color: #5B6B7C; text-transform: uppercase;
        }}
        .chip .v {{
            font-family: 'IBM Plex Mono', monospace; font-size: 1.16rem;
            font-weight: 600; margin-top: 0.2rem; font-variant-numeric: tabular-nums;
        }}
        .chip.up {{ border-top-color: rgb{UP}; }}
        .chip.up .v {{ color: rgb{UP}; }}
        .chip.down {{ border-top-color: rgb{DOWN}; }}
        .chip.down .v {{ color: rgb{DOWN}; }}
        .chip.flat .v {{ color: #5B6B7C; }}

        .note {{
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
            color: #5B6B7C; border-left: 2px solid {RULE};
            padding-left: 0.7rem; margin-top: 1.2rem; line-height: 1.7;
        }}
        div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_json(relative_path: str) -> Optional[Any]:
    path = os.path.join(DATA_DIR, relative_path)
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


def stamp(meta: Optional[dict]) -> str:
    if not isinstance(meta, dict):
        return "not yet built"
    raw = meta.get("last_updated")
    if not raw:
        return "not yet built"
    try:
        return datetime.fromisoformat(str(raw)).strftime("%d %b %Y &middot; %H:%M IST")
    except ValueError:
        return str(raw)


def masthead(eyebrow: str, heading: str, right_html: str) -> None:
    st.markdown(
        f'<div class="masthead"><div><span class="eyebrow">{eyebrow}</span>'
        f'<h1>{heading}</h1></div><div class="stamp">{right_html}</div></div>',
        unsafe_allow_html=True,
    )


def range_chips(row: Dict[str, Any]) -> None:
    cards = []
    for column in RANGE_COLS:
        value = row.get(column)
        if isinstance(value, (int, float)) and not pd.isna(value):
            if column.startswith("from_"):
                tone = "up" if value > 0 else ("down" if value < 0 else "flat")
                text = f"{float(value):+.2f}%"
            else:
                tone, text = "flat", f"{float(value):,.2f}"
        else:
            tone, text = "flat", "—"
        cards.append(
            f'<div class="chip {tone}"><div class="k">{RANGE_LABELS[column]}</div>'
            f'<div class="v">{text}</div></div>'
        )
    st.markdown(f'<div class="chips">{"".join(cards)}</div>', unsafe_allow_html=True)


def return_chips(row: Dict[str, Any]) -> None:
    cards = []
    for column in RETURN_COLS:
        value = row.get(column)
        if isinstance(value, (int, float)) and not pd.isna(value):
            tone = "up" if value > 0 else ("down" if value < 0 else "flat")
            text = f"{float(value):+.2f}%"
        else:
            tone, text = "flat", "—"
        cards.append(
            f'<div class="chip {tone}"><div class="k">{HORIZON_NAMES[column]}</div>'
            f'<div class="v">{text}</div></div>'
        )
    st.markdown(f'<div class="chips">{"".join(cards)}</div>', unsafe_allow_html=True)


def excess_css(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "color:#AAB3BD;"
    try:
        magnitude = min(abs(float(value)) / 12.0, 1.0)
    except (TypeError, ValueError):
        return ""
    rgb = UP if float(value) >= 0 else DOWN
    return (
        f"background-color:rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.08 + 0.5 * magnitude:.3f});"
        f"color:{INK};font-weight:{'600' if magnitude > 0.55 else '500'};"
    )


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

inject_css()
nav_bar("Lookup")

meta = load_json("last_updated.json")
indices_data = load_json("indices.json") or []
all_stocks = load_json("all_stocks.json") or []
membership = load_json("symbol_membership.json") or {}

index_by_name = {
    str(row["index"]): row
    for row in indices_data
    if isinstance(row, dict) and row.get("index")
}
stock_by_symbol = {
    str(row["symbol"]): row
    for row in all_stocks
    if isinstance(row, dict) and row.get("symbol")
}

masthead(
    "NSE &middot; single security lookup",
    "Lookup",
    f"<b>{len(index_by_name)}</b> indices &middot; <b>{len(stock_by_symbol)}</b> stocks<br>"
    f"built {stamp(meta)}",
)

if not index_by_name and not stock_by_symbol:
    st.warning(
        "**Nothing to search yet.** Run **Update NSE data** from the repository's "
        "Actions tab with both inputs set to `0`, then reload this page."
    )
    st.stop()

if not stock_by_symbol:
    st.warning(
        "**Indices only.** `data/all_stocks.json` is missing, so stock lookup is "
        "unavailable. The last build ran with **Build indices only** set to `1`, or "
        "predates this file. Re-run **Update NSE data** with both inputs at `0`."
    )

options = [f"{name}  ·  index" for name in sorted(index_by_name)] + [
    f"{symbol}  ·  stock" for symbol in sorted(stock_by_symbol)
]

choice = st.selectbox(
    "Search",
    options=options,
    index=None,
    placeholder="Type a stock symbol or index name — RELIANCE, HDFCBANK, NIFTY BANK",
    label_visibility="collapsed",
)

if not choice:
    st.caption(
        f"Search across {len(index_by_name)} indices and {len(stock_by_symbol)} stocks. "
        "The stock universe is every constituent of every NSE index — a symbol in no "
        "index at all will not appear."
    )
    st.stop()

name, _, kind = choice.rpartition("  ·  ")


# --------------------------------------------------------------------------
# Index result
# --------------------------------------------------------------------------

if kind == "index":
    row = index_by_name[name]
    close = row.get("close")
    st.subheader(name)
    st.caption(
        f"Index &middot; close {close:,.2f} &middot; as of {row.get('as_of') or '—'}"
        if isinstance(close, (int, float))
        else f"Index &middot; as of {row.get('as_of') or '—'}"
    )
    return_chips(row)
    range_chips(row)

    if st.button("Open constituents", type="primary"):
        st.session_state["selected_index"] = name
        st.switch_page(DETAIL_PAGE)

    st.markdown(
        '<div class="note">Price returns, not total returns. Each period is measured '
        'against the closest earlier trading day.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# --------------------------------------------------------------------------
# Stock result
# --------------------------------------------------------------------------

row = stock_by_symbol[name]
close = row.get("close")
parents = membership.get(name, []) if isinstance(membership, dict) else []

st.subheader(name)
st.caption(
    f"Stock &middot; close ₹{close:,.2f} &middot; as of {row.get('as_of') or '—'}"
    f" &middot; in {len(parents)} index{'' if len(parents) == 1 else 'es'}"
    if isinstance(close, (int, float))
    else f"Stock &middot; as of {row.get('as_of') or '—'}"
)
if row.get("adjusted"):
    st.caption(
        ":warning: Prices back-adjusted for a corporate action. Check the ex-date "
        "against NSE's calendar if a long-horizon figure looks unexpected."
    )

return_chips(row)
range_chips(row)

if not parents:
    st.markdown(
        '<div class="note">No index membership on file for this symbol.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown("**Return against each index it belongs to**")
st.caption(
    "Excess return — the stock's figure minus the index's, in percentage points. "
    "Positive means it outperformed that index over the period."
)

records: List[Dict[str, Any]] = []
for parent in parents:
    parent_row = index_by_name.get(parent)
    if not parent_row:
        continue
    record: Dict[str, Any] = {"Index": parent}
    for column in RETURN_COLS:
        stock_value = row.get(column)
        index_value = parent_row.get(column)
        if isinstance(stock_value, (int, float)) and isinstance(index_value, (int, float)):
            record[column] = round(float(stock_value) - float(index_value), 2)
        else:
            record[column] = np.nan
    records.append(record)

if not records:
    st.info("None of this symbol's indices are in the current build.")
    st.stop()

excess = pd.DataFrame(records).sort_values("1Y", ascending=False, na_position="last")
excess = excess.reset_index(drop=True)

styler = excess.style
for column in RETURN_COLS:
    styler = styler.apply(lambda col: [excess_css(v) for v in col], subset=[column])

st.dataframe(
    styler,
    column_config={
        "Index": st.column_config.TextColumn("Index", width="large"),
        **{
            column: st.column_config.NumberColumn(
                column, format="%.2f", help=f"Excess over {HORIZON_NAMES[column]}"
            )
            for column in RETURN_COLS
        },
    },
    hide_index=True,
    width="stretch",
    height=min(420, 60 + 36 * len(excess)),
)

st.markdown(
    '<div class="note">Excess is arithmetic difference in percentage points, not a '
    'beta-adjusted or risk-adjusted figure. Price returns on both sides, so the '
    'dividend treatment is consistent.</div>',
    unsafe_allow_html=True,
)
