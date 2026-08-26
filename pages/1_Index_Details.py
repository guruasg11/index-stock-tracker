"""
pages/1_Index_Details.py
========================
Constituents and returns for one NSE index.

Reads data/*.json from disk only. Makes no live NSE calls.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Index constituents",
    page_icon=":bar_chart:",
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

        .tape {{ margin: 0 0 1.5rem 0; }}
        .tape .bar {{
            display: flex; height: 9px; width: 100%;
            border-radius: 1px; overflow: hidden; background: {RULE};
        }}
        .tape .bar span {{ display: block; height: 100%; }}
        .tape .legend {{
            display: flex; justify-content: space-between;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
            color: #5B6B7C; margin-top: 0.4rem;
        }}
        .tape .legend b {{ color: {INK}; font-weight: 600; }}

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


def slugify(name: str) -> str:
    """Must mirror update_data.slugify exactly."""
    slug = re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")
    return (slug or "INDEX")[:80]


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


def breadth_tape(values: pd.Series, horizon: str, unit: str) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return
    advancing = int((clean > 0).sum())
    declining = int((clean < 0).sum())
    unchanged = int(len(clean) - advancing - declining)
    total = max(len(clean), 1)
    up_pct = advancing / total * 100.0
    down_pct = declining / total * 100.0
    flat_pct = max(0.0, 100.0 - up_pct - down_pct)
    st.markdown(
        f"""
        <div class="tape">
          <div class="bar">
            <span style="width:{up_pct:.2f}%;background:rgb{UP};"></span>
            <span style="width:{flat_pct:.2f}%;background:{RULE};"></span>
            <span style="width:{down_pct:.2f}%;background:rgb{DOWN};"></span>
          </div>
          <div class="legend">
            <div><b>{advancing}</b> advancing &nbsp;/&nbsp; <b>{unchanged}</b> flat
                 &nbsp;/&nbsp; <b>{declining}</b> declining</div>
            <div>{unit} breadth over {HORIZON_NAMES[horizon]} &nbsp;&middot;&nbsp;
                 median <b>{clean.median():+.2f}%</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cell_css(value: Any, scale: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "color:#AAB3BD;"
    try:
        magnitude = min(abs(float(value)) / scale, 1.0)
    except (TypeError, ValueError):
        return ""
    alpha = 0.08 + 0.5 * magnitude
    rgb = UP if float(value) >= 0 else DOWN
    weight = "600" if magnitude > 0.55 else "500"
    return (
        f"background-color:rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f});"
        f"color:{INK};font-weight:{weight};"
    )


def heatmap(frame: pd.DataFrame, columns: List[str]) -> Any:
    styler = frame.style
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        scale = float(np.percentile(series.abs(), 85)) if len(series) else 0.0
        scale = max(scale, 0.5)
        styler = styler.apply(
            lambda col, sc=scale: [cell_css(v, sc) for v in col], subset=[column]
        )
    return styler


def return_columns() -> Dict[str, Any]:
    return {
        column: st.column_config.NumberColumn(
            column, format="%.2f%%", help=HORIZON_NAMES[column]
        )
        for column in RETURN_COLS
    }


def back(key: str) -> None:
    if st.button("← All indices", key=key):
        st.switch_page(HOME_PAGE)


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------

inject_css()
nav_bar("Indices")

selected_index = st.session_state.get("selected_index")
horizon = st.session_state.get("horizon", "1M")
if horizon not in RETURN_COLS:
    horizon = "1M"

if not selected_index:
    masthead("NSE &middot; constituents", "No index chosen", "")
    st.info("Pick an index from the main table to see its constituents.")
    back("back_none")
    st.stop()

meta = load_json("last_updated.json")
indices_data = load_json("indices.json")

if not indices_data:
    masthead("NSE &middot; constituents", str(selected_index), "no data on disk")
    st.warning("`data/indices.json` is missing. Run **Update NSE data** from the Actions tab.")
    back("back_nodata")
    st.stop()

index_row = next(
    (
        row for row in indices_data
        if isinstance(row, dict) and str(row.get("index", "")) == str(selected_index)
    ),
    None,
)

if index_row is None:
    masthead("NSE &middot; constituents", str(selected_index), "not in this build")
    st.error(f"“{selected_index}” is not in the current data. It may have been renamed or delisted.")
    back("back_missing")
    st.stop()


# --------------------------------------------------------------------------
# Index header and its own nine returns
# --------------------------------------------------------------------------

back("back_top")

close = index_row.get("close")
masthead(
    "NSE &middot; index constituents &middot; price returns",
    str(selected_index),
    f"close <b>{close:,.2f}</b><br>as of <b>{index_row.get('as_of') or '—'}</b><br>"
    f"built {stamp(meta)}" if isinstance(close, (int, float))
    else f"as of <b>{index_row.get('as_of') or '—'}</b><br>built {stamp(meta)}",
)

cards = []
for column in RETURN_COLS:
    value = index_row.get(column)
    if isinstance(value, (int, float)):
        tone = "up" if value > 0 else ("down" if value < 0 else "flat")
        text = f"{value:+.2f}%"
    else:
        tone, text = "flat", "—"
    cards.append(
        f'<div class="chip {tone}"><div class="k">{HORIZON_NAMES[column]}</div>'
        f'<div class="v">{text}</div></div>'
    )
st.markdown(f'<div class="chips">{"".join(cards)}</div>', unsafe_allow_html=True)

range_cards = []
for column in RANGE_COLS:
    value = index_row.get(column)
    if isinstance(value, (int, float)):
        if column.startswith("from_"):
            tone = "up" if value > 0 else ("down" if value < 0 else "flat")
            text = f"{float(value):+.2f}%"
        else:
            tone, text = "flat", f"{float(value):,.2f}"
    else:
        tone, text = "flat", "—"
    range_cards.append(
        f'<div class="chip {tone}"><div class="k">{RANGE_LABELS[column]}</div>'
        f'<div class="v">{text}</div></div>'
    )
st.markdown(f'<div class="chips">{"".join(range_cards)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Constituents
# --------------------------------------------------------------------------

shard_map = load_json("stocks_index.json") or {}
shard_name = shard_map.get(str(selected_index)) if isinstance(shard_map, dict) else None
if not shard_name:
    shard_name = f"{slugify(selected_index)}.json"

constituents = load_json(os.path.join("stocks", shard_name)) or []

if not constituents:
    st.info(
        "No constituents on file for this index. NSE returns an equity list only for "
        "equity indices — fixed income, G-Sec and some strategy indices have none. "
        "If you expected stocks here, check the last job ran with **Build indices "
        "only** set to `0`."
    )
    back("back_nostocks")
    st.stop()

frame = pd.DataFrame(constituents)
if "symbol" not in frame.columns:
    st.error("The constituent file has no `symbol` column. Re-run the daily job.")
    back("back_malformed")
    st.stop()

for column in RETURN_COLS + RANGE_COLS + ["close"]:
    if column not in frame.columns:
        frame[column] = np.nan
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
if "as_of" not in frame.columns:
    frame["as_of"] = None
if "adjusted" not in frame.columns:
    frame["adjusted"] = False

adjusted_symbols = frame.loc[frame["adjusted"].fillna(False).astype(bool), "symbol"].tolist()
frame = frame[["symbol", "close", "as_of"] + RETURN_COLS + RANGE_COLS]

horizon = st.radio(
    "Rank by",
    options=RETURN_COLS,
    index=RETURN_COLS.index(horizon),
    horizontal=True,
    format_func=lambda h: HORIZON_NAMES[h],
    key="detail_horizon",
)
st.session_state["horizon"] = horizon

breadth_tape(frame[horizon], horizon, "Constituent")

search = st.text_input(
    "Filter",
    placeholder="Filter by symbol — HDFCBANK, INFY, RELIANCE",
    label_visibility="collapsed",
)

view = frame
if search.strip():
    view = frame[frame["symbol"].str.contains(search.strip(), case=False, na=False)]

if view.empty:
    st.info(f"No symbol contains “{search.strip()}”. Clear the filter to see all {len(frame)}.")
else:
    view = view.sort_values(horizon, ascending=False, na_position="last").reset_index(drop=True)
    st.caption(
        f"{len(view)} of {len(frame)} constituents, strongest {HORIZON_NAMES[horizon]} first."
    )
    st.dataframe(
        heatmap(view, RETURN_COLS + ["from_high", "from_low"]),
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", width="medium"),
            "close": st.column_config.NumberColumn("Close", format="%.2f"),
            "as_of": st.column_config.TextColumn("As of", width="small"),
            **return_columns(),
            **range_column_config(),
        },
        hide_index=True,
        width="stretch",
        height=620,
    )

if adjusted_symbols:
    with st.expander(f"{len(adjusted_symbols)} symbols back-adjusted for corporate actions"):
        st.write(
            "NSE's historical endpoint returns raw traded prices. These symbols showed a "
            "single-session move too large to be a market move, so earlier closes were "
            "rescaled onto the current basis. Check them against NSE's corporate action "
            "calendar if a figure looks wrong."
        )
        st.code(", ".join(sorted(adjusted_symbols)), language=None)

back("back_bottom")

st.markdown(
    """
    <div class="note">
    Source: NSE India official EOD API. Price returns, not total returns.<br>
    Each period is measured against the closest earlier trading day. Blank means
    no history for that span — usually a recent listing.<br>
    Colour is scaled within each column, so intensity is comparable across
    horizons rather than absolute.
    </div>
    """,
    unsafe_allow_html=True,
)
