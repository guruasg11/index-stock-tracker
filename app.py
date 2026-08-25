"""
app.py
======
NSE Index Returns Dashboard - all indices.

Reads data/*.json from disk only. Makes no live NSE calls. Data is rebuilt
daily by update_data.py via GitHub Actions.
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
    page_title="NSE Index Returns",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

RETURN_COLS = ["1D", "3D", "1W", "2W", "1M", "2M", "3M", "6M", "1Y"]
HORIZON_NAMES = {
    "1D": "1 day", "3D": "3 days", "1W": "1 week", "2W": "2 weeks",
    "1M": "1 month", "2M": "2 months", "3M": "3 months",
    "6M": "6 months", "1Y": "1 year",
}
DETAIL_PAGE = "pages/1_Index_Details.py"

INK = "#12263A"
PAPER = "#F7F8FA"
RULE = "#DCE1E8"
PETROL = "#1F5673"
UP = (15, 122, 78)
DOWN = (192, 57, 43)


# --------------------------------------------------------------------------
# Shared presentation layer
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');

        html, body, [class*="st-"] {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
        .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
        #MainMenu, footer {{ visibility: hidden; }}

        .masthead {{
            border-top: 3px solid {INK};
            border-bottom: 1px solid {RULE};
            padding: 0.9rem 0 1.1rem 0;
            margin-bottom: 1.4rem;
            display: flex; flex-wrap: wrap;
            align-items: baseline; justify-content: space-between; gap: 1rem;
        }}
        .masthead .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.68rem; letter-spacing: 0.16em;
            text-transform: uppercase; color: {PETROL};
            display: block; margin-bottom: 0.35rem;
        }}
        .masthead h1 {{
            font-family: 'IBM Plex Serif', Georgia, serif;
            font-size: 2.05rem; font-weight: 600; line-height: 1.1;
            color: {INK}; margin: 0; padding: 0;
        }}
        .masthead .stamp {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.74rem; color: #5B6B7C; text-align: right; line-height: 1.6;
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
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem; color: #5B6B7C; margin-top: 0.4rem;
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
    """Load JSON from data/. Returns None if missing, empty or invalid."""
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


def breadth_tape(values: pd.Series, horizon: str, unit: str) -> None:
    """Advancers vs decliners over the chosen horizon, as a proportional rule."""
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
    median = clean.median()
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
                 median <b>{median:+.2f}%</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cell_css(value: Any, scale: float) -> str:
    """Diverging tint, normalised per column so every horizon reads on its own range."""
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
    """Per-column normalised diverging heatmap. This is the point of the page."""
    styler = frame.style
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        scale = float(np.percentile(series.abs(), 85)) if len(series) else 0.0
        scale = max(scale, 0.5)
        styler = styler.apply(
            lambda col, sc=scale: [cell_css(v, sc) for v in col], subset=[column]
        )
    return styler


def return_columns(label_prefix: str = "") -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for column in RETURN_COLS:
        config[column] = st.column_config.NumberColumn(
            f"{label_prefix}{column}", format="%.2f%%", help=HORIZON_NAMES[column]
        )
    return config


def normalise(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    for column in RETURN_COLS + ["close"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "as_of" not in frame.columns:
        frame["as_of"] = None
    return frame[[key, "close", "as_of"] + RETURN_COLS]


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

inject_css()

indices_data = load_json("indices.json")
meta = load_json("last_updated.json")

if not indices_data:
    masthead("NSE &middot; end of day", "NSE Index Returns", "no data on disk")
    st.warning(
        "**Nothing to show yet.** The daily job has not written `data/indices.json`.\n\n"
        "Open the repository's **Actions** tab, choose **Update NSE data**, and run it. "
        "Leave both inputs at `0` for a full build of every index and constituent."
    )
    st.stop()

frame = pd.DataFrame(indices_data)
if "index" not in frame.columns:
    masthead("NSE &middot; end of day", "NSE Index Returns", "malformed data")
    st.error("`data/indices.json` has no `index` column. Re-run the daily job.")
    st.stop()

frame = normalise(frame, "index")

partial = isinstance(meta, dict) and meta.get("complete") is False
symbols = meta.get("unique_symbols", 0) if isinstance(meta, dict) else 0
adjusted = meta.get("symbols_adjusted", 0) if isinstance(meta, dict) else 0
as_of_dates = frame["as_of"].dropna()
trading_day = as_of_dates.mode().iat[0] if len(as_of_dates) else "—"

masthead(
    "NSE &middot; end of day &middot; price returns",
    "NSE Index Returns",
    f"<b>{len(frame)}</b> indices &middot; <b>{symbols}</b> stocks priced<br>"
    f"trading day <b>{trading_day}</b><br>built {stamp(meta)}",
)

if partial:
    st.warning(
        "The last job ended before finishing. These figures are a partial snapshot — "
        "re-run **Update NSE data** to complete it."
    )

horizon = st.radio(
    "Rank by",
    options=RETURN_COLS,
    index=RETURN_COLS.index("1M"),
    horizontal=True,
    format_func=lambda h: HORIZON_NAMES[h],
)

breadth_tape(frame[horizon], horizon, "Index")

leaders = frame.dropna(subset=[horizon]).sort_values(horizon, ascending=False)
if not leaders.empty:
    picks = pd.concat([leaders.head(3), leaders.tail(3)])
    cards = []
    for _, row in picks.iterrows():
        value = float(row[horizon])
        tone = "up" if value > 0 else ("down" if value < 0 else "flat")
        cards.append(
            f'<div class="chip {tone}"><div class="k">{row["index"][:26]}</div>'
            f'<div class="v">{value:+.2f}%</div></div>'
        )
    st.markdown(f'<div class="chips">{"".join(cards)}</div>', unsafe_allow_html=True)

search = st.text_input(
    "Filter",
    placeholder="Filter by name — NIFTY BANK, MIDCAP, AUTO, PSU",
    label_visibility="collapsed",
)

view = frame
if search.strip():
    view = frame[frame["index"].str.contains(search.strip(), case=False, na=False)]

if view.empty:
    st.info(f"No index name contains “{search.strip()}”. Clear the filter to see all {len(frame)}.")
    st.stop()

view = view.sort_values(horizon, ascending=False, na_position="last").reset_index(drop=True)

st.caption(
    f"{len(view)} of {len(frame)} indices, strongest {HORIZON_NAMES[horizon]} first. "
    "Colour is scaled within each column, so a deep green under 1D and under 1Y "
    "mean the same thing relative to that horizon. Click a row for constituents."
)

event = st.dataframe(
    heatmap(view, RETURN_COLS),
    column_config={
        "index": st.column_config.TextColumn("Index", width="large"),
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "as_of": st.column_config.TextColumn("As of", width="small"),
        **return_columns(),
    },
    hide_index=True,
    width="stretch",
    height=560,
    on_select="rerun",
    selection_mode="single-row",
    key="indices_table",
)


def open_index(name: str) -> None:
    st.session_state["selected_index"] = name
    st.session_state["horizon"] = horizon
    st.switch_page(DETAIL_PAGE)


rows = []
if event is not None and getattr(event, "selection", None):
    rows = event.selection.get("rows", []) or []
if rows and 0 <= rows[0] < len(view):
    open_index(str(view.iloc[rows[0]]["index"]))

picker, action = st.columns([4, 1])
with picker:
    choice = st.selectbox(
        "Open an index", options=view["index"].tolist(), label_visibility="collapsed"
    )
with action:
    if st.button("Open", width="stretch", type="primary"):
        open_index(str(choice))

st.markdown(
    f"""
    <div class="note">
    Source: NSE India official EOD API. Price returns, not total returns — NSE's
    TR series will read higher by roughly the dividend yield.<br>
    Each period is measured against the closest earlier trading day, so weekends
    and exchange holidays resolve backwards. Blank means no history for that span.<br>
    Constituent prices are back-adjusted for splits, bonuses and consolidations
    ({adjusted} symbol{"" if adjusted == 1 else "s"} adjusted in the last build).
    </div>
    """,
    unsafe_allow_html=True,
)
