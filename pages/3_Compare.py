"""
pages/3_Compare.py
==================
Compare any number of indices and stocks side by side.

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
    page_title="Compare",
    page_icon=":scales:",
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
    """Explicit tabs - sidebar navigation is switched off in config.toml.

    st.switch_page cannot be called from an on_click callback, so each button
    is checked inline instead.
    """
    tabs = [("Indices", "app.py"), ("Compare", "pages/3_Compare.py")]
    columns = st.columns(len(tabs) + 5)
    for column, (label, target) in zip(columns, tabs):
        with column:
            clicked = st.button(
                label,
                key=f"nav_{label}",
                width="stretch",
                type="primary" if label == current else "secondary",
                disabled=label == current,
            )
        if clicked:
            st.switch_page(target)


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


def heatmap(frame: pd.DataFrame, columns: List[str]) -> Any:
    """Per-column normalised diverging tint, so each measure reads on its own range."""
    styler = frame.style
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        scale = float(np.percentile(series.abs(), 85)) if len(series) else 0.0
        scale = max(scale, 0.5)
        styler = styler.apply(
            lambda col, sc=scale: [excess_css_scaled(v, sc) for v in col], subset=[column]
        )
    return styler


def excess_css_scaled(value: Any, scale: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "color:#AAB3BD;"
    try:
        magnitude = min(abs(float(value)) / scale, 1.0)
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
nav_bar("Compare")

meta = load_json("last_updated.json")
indices_data = load_json("indices.json") or []
all_stocks = load_json("all_stocks.json") or []

index_by_name = {
    str(r["index"]): r for r in indices_data if isinstance(r, dict) and r.get("index")
}
stock_by_symbol = {
    str(r["symbol"]): r for r in all_stocks if isinstance(r, dict) and r.get("symbol")
}

masthead(
    "NSE &middot; side by side",
    "Compare",
    f"<b>{len(index_by_name)}</b> indices &middot; <b>{len(stock_by_symbol)}</b> stocks<br>"
    f"built {stamp(meta)}",
)

if not index_by_name and not stock_by_symbol:
    st.warning(
        "**Nothing to compare yet.** Run **Update NSE data** from the repository's "
        "Actions tab with both inputs set to `0`, then reload."
    )
    st.stop()

INDEX_TAG = "  \u00b7  index"
STOCK_TAG = "  \u00b7  stock"
options = [f"{n}{INDEX_TAG}" for n in sorted(index_by_name)] + [
    f"{s}{STOCK_TAG}" for s in sorted(stock_by_symbol)
]

picked = st.multiselect(
    "Compare",
    options=options,
    placeholder="Add indices and stocks — NIFTY 50, NIFTY BANK, HDFCBANK, INFY",
    label_visibility="collapsed",
    key="compare_picked",
)

if not picked:
    st.caption(
        "Add two or more from the box above. Indices and stocks mix freely — both are "
        "measured the same way, so a stock sits beside its own index on one scale."
    )
    st.stop()


def resolve(label: str) -> Optional[Dict[str, Any]]:
    if label.endswith(INDEX_TAG):
        name = label[: -len(INDEX_TAG)]
        row = index_by_name.get(name)
        return {**row, "Name": name, "Type": "Index"} if row else None
    name = label[: -len(STOCK_TAG)]
    row = stock_by_symbol.get(name)
    return {**row, "Name": name, "Type": "Stock"} if row else None


rows = [r for r in (resolve(p) for p in picked) if r]
if not rows:
    st.info("None of those selections are in the current build.")
    st.stop()

MEASURES = RETURN_COLS + RANGE_COLS
# Excess makes sense for returns and for distance-from-extreme. Subtracting one
# security's 52-week high from another's is meaningless, so those stay absolute.
EXCESS_COLS = RETURN_COLS + ["from_high", "from_low"]

mode_col, bench_col = st.columns([2, 3])
with mode_col:
    mode = st.radio(
        "Show", options=["Absolute", "Excess over benchmark"],
        horizontal=True, label_visibility="collapsed",
    )
benchmark = None
if mode == "Excess over benchmark":
    with bench_col:
        benchmark = st.selectbox(
            "Benchmark", options=[r["Name"] for r in rows], label_visibility="collapsed",
        )

records: List[Dict[str, Any]] = []
for row in rows:
    record: Dict[str, Any] = {
        "Name": f"{row['Name']} *" if row.get("stale") else row["Name"],
        "Type": row["Type"],
        "close": row.get("close"),
        "as_of": row.get("as_of"),
    }
    for column in MEASURES:
        value = row.get(column)
        record[column] = float(value) if isinstance(value, (int, float)) else np.nan
    records.append(record)

frame = pd.DataFrame(records)

if benchmark:
    base = next((r for r in rows if r["Name"] == benchmark), None)
    if base:
        for column in EXCESS_COLS:
            reference = base.get(column)
            if isinstance(reference, (int, float)):
                frame[column] = (frame[column] - float(reference)).round(2)
            else:
                frame[column] = np.nan
        frame = frame[frame["Name"].str.removesuffix(" *") != benchmark]
        st.caption(
            f"Percentage points above or below **{benchmark}** over each period. "
            "The benchmark row is removed; raw 52-week levels stay absolute."
        )

if frame.empty:
    st.info("Add something besides the benchmark to measure against it.")
    st.stop()

frame = frame.reset_index(drop=True)

st.dataframe(
    heatmap(frame, EXCESS_COLS),
    column_config={
        "Name": st.column_config.TextColumn("Name", width="medium"),
        "Type": st.column_config.TextColumn("Type", width="small"),
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "as_of": st.column_config.TextColumn("As of", width="small"),
        **{
            c: st.column_config.NumberColumn(c, format="%.2f%%", help=HORIZON_NAMES[c])
            for c in RETURN_COLS
        },
        **range_column_config(),
    },
    hide_index=True,
    width="stretch",
    height=min(560, 70 + 36 * len(frame)),
)

st.markdown("**One horizon, ranked**")
horizon = st.radio(
    "Horizon", options=RETURN_COLS, index=RETURN_COLS.index("1Y"),
    horizontal=True, format_func=lambda h: HORIZON_NAMES[h],
    label_visibility="collapsed",
)

chart = frame[["Name", horizon]].dropna(subset=[horizon])
if chart.empty:
    st.info(f"No {HORIZON_NAMES[horizon]} figure available for any selection.")
else:
    chart = chart.sort_values(horizon, ascending=False).set_index("Name")
    st.bar_chart(chart, height=max(200, 36 * len(chart)), horizontal=True)

if any(r.get("stale") for r in rows):
    st.caption("* carried forward from an earlier build, not refreshed in the last run.")

st.markdown(
    '<div class="note">'
    'Price returns, not total returns. Each period is measured against the closest '
    'earlier trading day.<br>'
    'Excess is an arithmetic difference in percentage points, not beta-adjusted.<br>'
    '52-week high and low come from the trailing 252 sessions of the same adjusted '
    'closes used for the returns.'
    '</div>',
    unsafe_allow_html=True,
)
