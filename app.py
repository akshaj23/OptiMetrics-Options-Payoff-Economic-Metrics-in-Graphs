# ==================================================================================================
# FINANCIAL GRAPHS — TEACHING & INTERACTIVE EXPLORER (ROBUST, ALL TABS)
# --------------------------------------------------------------------------------------------------
# Contents (tabs):
#   1) Net Payoff (Options)      — strategy builder, BE detection, autoscale, shaded P/L, per-leg plots
#   2) Demand & Revenue          — fit P=a+bQ, show Demand, MR, TR (right axis), TR max where MR=0
#   3) Supply & Cost             — linear MC P=c+dQ, integrate to VC, add FC to get TC; AVC & ATC
#   4) Elasticity (midpoint)     — arc elasticity, clear working and plain-English summary
#   5) Kinked Demand             — two-slope demand around (Qk,Pk), piecewise MR, tables & download
#   6) Cost Curves               — cubic TC gives U-shaped AVC/ATC; MC intersects ATC at min
#
# Principles:
#   - One static sidebar (no toggles). All math notes are visible in each tab.
#   - Unique widget keys everywhere (especially buttons) — avoids Streamlit duplicate-id errors.
#   - Each tab: examples, OG-term inputs, inline explanations that reference current numbers,
#               a working table (CSV download), clean plots (autoscale & annotations).
#   - No unsupported args to st.data_editor; everything uses current Streamlit API.
# ==================================================================================================

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==================================================================================================
# Utilities (shared)
# ==================================================================================================

LEG_COLS = ["Type", "Side", "Strike", "Premium", "Quantity", "Delete"]


def _nice(x) -> float:
    """Best-effort float (for formatting)."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def _download_button_df(df: pd.DataFrame, label: str, name: str):
    """Reusable CSV download button."""
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=name,
        mime="text/csv",
        use_container_width=False,
    )


def _autoscale_range(values, pad_frac=0.08, min_pad=1.0):
    """Return (lo, hi) range padded for plotting."""
    arr = np.asarray(values, float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    span = max(min_pad, hi - lo)
    pad = max(min_pad, span * pad_frac)
    return (max(0.0, lo - pad), hi + pad)


def clean_legs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive cleaner for the legs table to avoid NaN→int crashes and odd states:

      * enforce columns and types
      * coerce numerics with errors='coerce'
      * drop incomplete/invalid rows (e.g. Quantity <= 0)
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=LEG_COLS)

    out = df.copy()

    # Normalize text
    out["Type"] = out.get("Type", "").astype(str).str.strip().str.title()
    out["Side"] = out.get("Side", "").astype(str).str.strip().str.title()

    # Coerce numerics
    for c in ["Strike", "Premium", "Quantity"]:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")

    # Valid rows only
    out = out.dropna(subset=["Strike", "Premium", "Quantity"])
    out = out[out["Quantity"] > 0]
    out["Quantity"] = out["Quantity"].round().astype(int)

    # Delete column must exist & be bool
    out["Delete"] = out.get("Delete", False)
    out["Delete"] = out["Delete"].fillna(False).astype(bool)

    # Ensure column order
    for c in LEG_COLS:
        if c not in out.columns:
            out[c] = np.nan if c != "Delete" else False

    return out[LEG_COLS].reset_index(drop=True)


# ==================================================================================================
# Core option math helpers
# ==================================================================================================

def payoff_leg(S: np.ndarray, leg: pd.Series, lot_size: int = 1) -> np.ndarray:
    typ, side = str(leg["Type"]).strip(), str(leg["Side"]).strip()
    K, prem, qty = float(leg["Strike"]), float(leg["Premium"]), int(leg["Quantity"])
    intrinsic = np.maximum(S - K, 0.0) if typ.lower() == "call" else np.maximum(K - S, 0.0)
    sign = 1.0 if side.lower() == "long" else -1.0
    return sign * (intrinsic - prem) * qty * lot_size


def leg_be(typ: str, premium: float, strike: float) -> float:
    return (strike + premium) if typ.lower() == "call" else (strike - premium)


def leg_theoretical_extrema(
    typ: str, side: str, premium: float, strike: float, qty: int, lot: int
):
    prem_scaled = premium * qty * lot
    typ = typ.lower()
    side = side.lower()
    if typ == "call" and side == "long":
        return "Unlimited ↑", f"{prem_scaled:.2f}"
    if typ == "call" and side == "short":
        return f"{prem_scaled:.2f}", "Unlimited ↓"
    if typ == "put" and side == "long":
        return f"{max(0.0, (strike - premium) * qty * lot):.2f}", f"{prem_scaled:.2f}"
    if typ == "put" and side == "short":
        return f"{prem_scaled:.2f}", f"{max(0.0, (strike - premium) * qty * lot):.2f}"
    return "—", "—"


def strategy_payoff_table(S: np.ndarray, legs_df: pd.DataFrame, lot_size: int = 1):
    out = pd.DataFrame({"Price": S})
    cols = []
    legs_df = clean_legs_df(legs_df)
    if legs_df is None or legs_df.empty:
        out["Combined"] = 0.0
        return out, cols

    legs_df = legs_df.reset_index(drop=True)
    for i, leg in legs_df.iterrows():
        qty_str = f"{int(leg['Quantity'])}" if pd.notna(leg["Quantity"]) else "0"
        label = (
            f"Leg{i+1}: {leg['Side']} {leg['Type']} "
            f"K={_nice(leg['Strike'])} @ {_nice(leg['Premium'])} x{qty_str}"
        )
        out[label] = payoff_leg(S, leg, lot_size)
        cols.append(label)

    out["Combined"] = out[cols].sum(axis=1) if cols else 0.0
    return out, cols


def breakevens_from_grid(S: np.ndarray, Y: np.ndarray) -> list[float]:
    S = np.asarray(S, float)
    Y = np.asarray(Y, float)
    if len(S) < 2:
        return []
    idx = np.where(np.signbit(Y[:-1]) != np.signbit(Y[1:]))[0]
    bes = []
    for i in idx:
        x0, x1, y0, y1 = S[i], S[i + 1], Y[i], Y[i + 1]
        if y1 != y0:
            bes.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    if np.any(np.isclose(Y, 0.0, atol=1e-12)):
        bes.extend(S[np.where(np.isclose(Y, 0.0, atol=1e-12))[0]].tolist())
    return sorted({round(float(x), 6) for x in bes})


def extrema_on_grid(S: np.ndarray, Y: np.ndarray):
    if len(Y) == 0:
        return (math.nan, math.nan), (math.nan, math.nan)
    i_max, i_min = int(np.nanargmax(Y)), int(np.nanargmin(Y))
    return (float(S[i_max]), float(Y[i_max])), (float(S[i_min]), float(Y[i_min]))


def _moneyness(typ: str, K: float, spot: float) -> str:
    typ = typ.lower()
    if abs(spot - K) < 1e-9:
        return "ATM"
    if typ == "call":
        return "ITM" if spot > K else "OTM"
    else:
        return "ITM" if spot < K else "OTM"


def combined_plot(
    S: np.ndarray,
    Y: np.ndarray,
    spot: float,
    be_list=None,
    strike_marks=None,
    title: str = "",
) -> go.Figure:
    S = np.asarray(S, float)
    Y = np.asarray(Y, float)
    fig = go.Figure()
    # shaded areas
    fig.add_trace(
        go.Scatter(
            x=S,
            y=np.where(Y >= 0, Y, np.nan),
            mode="lines",
            line=dict(width=0.5, color="rgba(0,140,0,0.9)"),
            fill="tozeroy",
            fillcolor="rgba(0,200,0,0.25)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=S,
            y=np.where(Y <= 0, Y, np.nan),
            mode="lines",
            line=dict(width=0.5, color="rgba(180,0,0,0.9)"),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.25)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(go.Scatter(x=S, y=Y, mode="lines", name="Combined", line=dict(width=3)))
    fig.add_hline(y=0, line_dash="dot", opacity=0.5)

    if be_list:
        for i, be in enumerate(be_list, 1):
            fig.add_vline(x=be, line_dash="dash", line_color="gray", opacity=0.7)
            fig.add_trace(
                go.Scatter(
                    x=[be],
                    y=[0],
                    mode="markers+text",
                    text=[f"BE {i} = {be:.2f}"],
                    textposition="top center",
                    marker=dict(size=8, color="gray"),
                    showlegend=False,
                    cliponaxis=False,
                )
            )
    if strike_marks:
        for k in strike_marks:
            fig.add_vline(
                x=k,
                line_dash="dot",
                line_color="rgba(100,100,100,0.25)",
                opacity=0.6,
            )

    now_y = float(np.interp(spot, S, Y))
    fig.add_vline(x=spot, line_width=2, line_color="black", opacity=0.6)
    fig.add_trace(
        go.Scatter(
            x=[spot],
            y=[now_y],
            mode="markers+text",
            name="Now",
            text=[f"Now: {now_y:.2f} @ {spot:.2f}"],
            textposition="top center",
            marker=dict(size=11, color="black"),
            cliponaxis=False,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title or "Payoff",
        xaxis_title="Underlying Price",
        yaxis_title="Payoff",
        hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=50, b=30),
    )
    xspan = float(S.max() - S.min()) or 1.0
    fig.update_xaxes(range=[S.min() - 0.03 * xspan, S.max() + 0.03 * xspan])
    return fig


# ==================================================================================================
# TAB 1: NET PAYOFF (Options)
# ==================================================================================================

def net_examples():
    return {
        "Long Call (K=100, prem=5)": (
            [{"Type": "Call", "Side": "Long", "Strike": 100, "Premium": 5, "Quantity": 1}],
            110,
        ),
        "Long Put (K=100, prem=5)": (
            [{"Type": "Put", "Side": "Long", "Strike": 100, "Premium": 5, "Quantity": 1}],
            90,
        ),
        "Bull Call Spread (95/105; 7/3)": (
            [
                {"Type": "Call", "Side": "Long", "Strike": 95, "Premium": 7, "Quantity": 1},
                {"Type": "Call", "Side": "Short", "Strike": 105, "Premium": 3, "Quantity": 1},
            ],
            100,
        ),
        "Long Straddle (call+put K=100; 5/5)": (
            [
                {"Type": "Call", "Side": "Long", "Strike": 100, "Premium": 5, "Quantity": 1},
                {"Type": "Put", "Side": "Long", "Strike": 100, "Premium": 5, "Quantity": 1},
            ],
            100,
        ),
        "Long Call Butterfly (90/100/110; 8/3/1)": (
            [
                {"Type": "Call", "Side": "Long", "Strike": 90, "Premium": 8, "Quantity": 1},
                {"Type": "Call", "Side": "Short", "Strike": 100, "Premium": 3, "Quantity": 2},
                {"Type": "Call", "Side": "Long", "Strike": 110, "Premium": 1, "Quantity": 1},
            ],
            100,
        ),
        "Iron Condor (90/95/105/110; 1/2/2/1)": (
            [
                {"Type": "Put", "Side": "Long", "Strike": 90, "Premium": 1, "Quantity": 1},
                {"Type": "Put", "Side": "Short", "Strike": 95, "Premium": 2, "Quantity": 1},
                {"Type": "Call", "Side": "Short", "Strike": 105, "Premium": 2, "Quantity": 1},
                {"Type": "Call", "Side": "Long", "Strike": 110, "Premium": 1, "Quantity": 1},
            ],
            100,
        ),
    }


def tab_netpayoff():
    st.header("Net Payoff (Options)")
    st.caption(
        "Build a strategy by adding option legs. Green = profit, red = loss. "
        "Break-evens, strikes, moneyness and extremes are annotated."
    )

    # Session init
    if "legs" not in st.session_state:
        st.session_state.legs = pd.DataFrame(columns=LEG_COLS).astype(
            {
                "Type": "object",
                "Side": "object",
                "Strike": "float",
                "Premium": "float",
                "Quantity": "int",
                "Delete": "bool",
            }
        )
    if "net_spot" not in st.session_state:
        st.session_state.net_spot = 110.0

    # Examples
    ex_name = st.selectbox(
        "Examples", list(net_examples().keys()), index=0, key="net_example_picker"
    )
    c1, _ = st.columns([1, 4])
    if c1.button("Load example", use_container_width=True, key="net_load_btn"):
        legs, spot = net_examples()[ex_name]
        df = pd.DataFrame(legs)
        df["Delete"] = False
        st.session_state.legs = clean_legs_df(df)
        st.session_state.net_spot = float(spot)

    # Add leg
    st.subheader("➕ Add an option leg")
    col1, col2, col3, col4, col5 = st.columns([1.1, 1.1, 1, 1.1, 1])
    typ = col1.selectbox("Type", ["Call", "Put"], index=0, key="net_typ")
    side = col2.selectbox("Side", ["Long", "Short"], index=0, key="net_side")
    strike = col3.number_input(
        "Strike", min_value=0.0, value=100.0, step=1.0, format="%.2f", key="net_strike"
    )
    prem = col4.number_input(
        "Premium (entry)",
        min_value=0.0,
        value=5.0,
        step=0.5,
        format="%.2f",
        key="net_prem",
    )
    qty = col5.number_input("Quantity", min_value=1, value=1, step=1, key="net_qty")
    if st.button("Add leg", type="primary", key="net_add_leg"):
        new_row = {
            "Type": typ,
            "Side": side,
            "Strike": float(strike),
            "Premium": float(prem),
            "Quantity": int(qty),
            "Delete": False,
        }
        st.session_state.legs = clean_legs_df(
            pd.concat([st.session_state.legs, pd.DataFrame([new_row])], ignore_index=True)
        )

    # Leg editor
    st.subheader("Your Option Legs")
    edited = st.data_editor(
        st.session_state.legs,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Type": st.column_config.SelectboxColumn(options=["Call", "Put"]),
            "Side": st.column_config.SelectboxColumn(options=["Long", "Short"]),
            "Delete": st.column_config.CheckboxColumn(
                help="Tick & press Delete selected"
            ),
        },
        hide_index=True,
        key="net_legs_editor",
    )
    st.session_state.legs = clean_legs_df(edited)

    # Alternative delete by index (since left gutter isn't a selection)
    del_rows = []
    if not st.session_state.legs.empty:
        del_rows = st.multiselect(
            "Optional: choose rows to delete (alternative to the Delete column)",
            options=list(range(len(st.session_state.legs))),
            format_func=lambda i: f"Row {i+1}: {st.session_state.legs.loc[i,'Side']} {st.session_state.legs.loc[i,'Type']} K={_nice(st.session_state.legs.loc[i,'Strike'])}",
            key="net_rows_to_delete",
        )

    cdel, cclr = st.columns([1, 1])
    if cdel.button("🗑️ Delete selected (checkbox or chosen rows)", key="net_delete_checked"):
        legs = st.session_state.legs.copy()
        mask_keep = ~legs["Delete"]
        if del_rows:
            idx = legs.index.isin(del_rows)
            mask_keep &= ~idx
        st.session_state.legs = clean_legs_df(legs.loc[mask_keep].reset_index(drop=True))

    if cclr.button("🧹 Clear all legs", key="net_clear_all"):
        st.session_state.legs = st.session_state.legs.iloc[0:0].reset_index(drop=True)

    # Price range controls
    st.subheader("Price range")
    lot_size = st.number_input(
        "Lot size (multiplier)", min_value=1, value=1, step=1, key="net_lot"
    )
    auto_scale = st.checkbox("Auto-scale price range", value=True, key="net_auto")
    col_a, col_b, col_c = st.columns(3)
    min_ui = col_a.number_input(
        "Min price (manual)", min_value=0.0, value=80.0, step=1.0, format="%.2f", key="net_min"
    )
    max_ui = col_b.number_input(
        "Max price (manual)", min_value=0.0, value=130.0, step=1.0, format="%.2f", key="net_max"
    )
    step_ui = col_c.number_input(
        "Grid step", min_value=0.01, value=3.0, step=0.5, format="%.2f", key="net_step"
    )
    spot = st.number_input(
        "Spot (Now)",
        min_value=0.0,
        value=float(st.session_state.get("net_spot", 110.0)),
        step=1.0,
        format="%.2f",
        key="net_spot",
    )

    # Provisional scan to detect BEs, then autoscale so flips are visible
    calc_legs = clean_legs_df(st.session_state.legs)
    provisional_S = np.arange(0.0, max(10.0, spot * 2.5), 0.25)
    provisional_df, _ = strategy_payoff_table(provisional_S, calc_legs, lot_size=lot_size)
    Y_prov = (
        provisional_df["Combined"].to_numpy()
        if "Combined" in provisional_df
        else np.zeros_like(provisional_S)
    )
    bes_c = breakevens_from_grid(provisional_S, Y_prov)

    if auto_scale and not calc_legs.empty:
        strikes = calc_legs["Strike"].astype(float).tolist()
        mn, mx = _autoscale_range(strikes + [spot] + bes_c, pad_frac=0.10, min_pad=1.5)
        mn = min(mn, max(0.0, min(strikes) - 3 * max(1.0, step_ui)))
        mx = max(mx, max(strikes) + 3 * max(1.0, step_ui))
        step = round(max(0.01, (mx - mn) / 220.0), 2)
        S = np.arange(mn, mx + 1e-9, step)
    else:
        S = np.arange(min_ui, max_ui + 1e-9, step_ui)

    # Final table & metrics
    payoff_df, leg_cols = strategy_payoff_table(S, calc_legs, lot_size=lot_size)
    Yc = payoff_df["Combined"].to_numpy() if "Combined" in payoff_df else np.zeros_like(S)
    bes_c = breakevens_from_grid(S, Yc)
    (s_max, y_max), (s_min, y_min) = extrema_on_grid(S, Yc)
    strikes_all = calc_legs["Strike"].astype(float).tolist() if not calc_legs.empty else []

    # Quick metrics
    st.subheader("Quick metrics")
    net_premium = 0.0
    for _, leg in calc_legs.iterrows():
        net_premium += (
            (1 if str(leg["Side"]).lower() == "long" else -1)
            * float(leg["Premium"])
            * int(leg["Quantity"])
            * int(lot_size)
        )
    payoff_now = float(np.interp(spot, S, Yc)) if len(S) else 0.0
    be_text = ", ".join(f"{x:.2f}" for x in bes_c) if bes_c else "—"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net premium (D/C)", f"{net_premium:.2f}")
    c2.metric(f"Payoff @ Spot {spot:.0f}", f"{payoff_now:.2f}")
    c3.metric("Max Profit (range)", f"{y_max:.2f}" if np.isfinite(y_max) else "—")
    c4.metric("Max Loss (range)", f"{y_min:.2f}" if np.isfinite(y_min) else "—")
    c5.metric("Break-evens", be_text)

    # Combined plot
    st.subheader("Combined")
    title = "Combined Payoff at Expiry (UNDERLYING) — Profit/Loss Shaded"
    fig_c = combined_plot(S, Yc, spot, bes_c, strikes_all, title=title)
    st.plotly_chart(fig_c, use_container_width=True)

    # Strategy working table
    st.subheader("Strategy summary (working)")
    if calc_legs.empty:
        st.info("Add some legs to see the strategy summary.")
    else:
        rows = []
        for i, leg in calc_legs.reset_index(drop=True).iterrows():
            be_leg_val = leg_be(str(leg["Type"]), float(leg["Premium"]), float(leg["Strike"]))
            y_leg = payoff_leg(S, leg, lot_size=lot_size)
            rows.append(
                {
                    "Leg": i + 1,
                    "Type": leg["Type"],
                    "Side": leg["Side"],
                    "Strike K": float(leg["Strike"]),
                    "Premium": float(leg["Premium"]),
                    "Qty": int(leg["Quantity"]),
                    "Break-even": round(be_leg_val, 4),
                    "Payoff @Now": round(float(np.interp(spot, S, y_leg)), 4),
                    "Moneyness @Now": _moneyness(
                        str(leg["Type"]), float(leg["Strike"]), float(spot)
                    ),
                    "Net debit/credit": round(
                        (1 if str(leg["Side"]).lower() == "long" else -1)
                        * float(leg["Premium"])
                        * int(leg["Quantity"])
                        * int(lot_size),
                        4,
                    ),
                }
            )
        sum_df = pd.DataFrame(rows)
        st.dataframe(sum_df, use_container_width=True, hide_index=True)
        _download_button_df(sum_df, "Download strategy summary (CSV)", "strategy_summary.csv")

    # Per-leg charts
    st.subheader("Per-Leg Charts & Calculations")
    if calc_legs.empty:
        st.info("Add legs to see per-leg details.")
    else:
        for i, leg in calc_legs.reset_index(drop=True).iterrows():
            title = (
                f"Leg {i+1}: {leg['Side']} {leg['Type']}  "
                f"K={_nice(leg['Strike'])}  @ {_nice(leg['Premium'])}  ×{int(leg['Quantity'])}"
            )
            y_leg = payoff_leg(S, leg, lot_size=lot_size)
            be_leg_val = leg_be(str(leg["Type"]), float(leg["Premium"]), float(leg["Strike"]))
            fig_l = combined_plot(
                S,
                y_leg,
                spot,
                be_list=[be_leg_val],
                strike_marks=[float(leg["Strike"])],
                title=title,
            )
            st.plotly_chart(fig_l, use_container_width=True)

            max_p_text, max_l_text = leg_theoretical_extrema(
                str(leg["Type"]),
                str(leg["Side"]),
                float(leg["Premium"]),
                float(leg["Strike"]),
                int(leg["Quantity"]),
                int(lot_size),
            )
            row = pd.DataFrame(
                [
                    {
                        "Type": leg["Type"],
                        "Side": leg["Side"],
                        "Strike": float(leg["Strike"]),
                        "Premium": float(leg["Premium"]),
                        "Quantity": int(leg["Quantity"]),
                        "Lot size": int(lot_size),
                        "Break-even": round(be_leg_val, 4),
                        "Payoff @ Spot": round(float(np.interp(spot, S, y_leg)), 4),
                        "Max Profit (theoretical)": max_p_text,
                        "Max Loss (theoretical)": max_l_text,
                        "Net debit/credit": round(
                            (1 if str(leg["Side"]).lower() == "long" else -1)
                            * float(leg["Premium"])
                            * int(leg["Quantity"])
                            * int(lot_size),
                            4,
                        ),
                    }
                ]
            )
            st.dataframe(row, use_container_width=True, hide_index=True)

    # Full payoff table
    st.subheader("Full Payoff Table")
    st.caption("Per-price payoffs for each leg and the combined curve.")
    st.dataframe(payoff_df, use_container_width=True, height=320)
    _download_button_df(payoff_df, "Download payoff table (CSV)", "payoff_table.csv")

    # Math notes (LaTeX rendered cleanly)
    st.markdown("### Maths")
    st.latex(r"\textbf{Call BE}: \; K + \text{premium} \qquad \textbf{Put BE}: \; K - \text{premium}")
    st.latex(r"\text{Net premium} = \sum (\text{premium}\times \text{quantity} \times \text{lot})\; \text{with long }=+,\; \text{short }=-")
    st.latex(r"\text{Shading: green }(\ge 0), \; \text{ red }(<0)")
    st.latex(r"\text{Unlimited outcomes when } S \to \infty \; (\text{long call}) \; \text{ or } \; S \to 0 \; (\text{short put})")


# ==================================================================================================
# TAB 2: DEMAND & REVENUE
# ==================================================================================================

def demand_examples():
    return {
        "Coffee shop (daily cups vs price)": pd.DataFrame(
            {"Q": [100, 150, 200], "P": [5.0, 4.0, 3.0]}
        ),
        "Movie tickets (simple)": pd.DataFrame(
            {"Q": [0, 100, 200, 300], "P": [20, 15, 10, 5]}
        ),
        "Elastic demand (steeper)": pd.DataFrame({"Q": [50, 100, 140], "P": [60, 40, 30]}),
    }


def fit_linear_demand(df: pd.DataFrame):
    """
    Fit P = A + B Q. For a typical downward demand we expect B<0.
    TR(Q) = P(Q)·Q = (A + BQ)Q; MR(Q) = A + 2BQ.
    """
    df = df.dropna()
    if df.shape[0] < 2:
        return None

    Q = df["Q"].to_numpy(dtype=float)
    P = df["P"].to_numpy(dtype=float)
    A = np.vstack([np.ones_like(Q), Q]).T
    coef = np.linalg.lstsq(A, P, rcond=None)[0]
    A0, B = float(coef[0]), float(coef[1])

    def P_of_Q(q): return A0 + B * q
    def TR(q): return (A0 + B * q) * q
    def MR(q): return A0 + 2.0 * B * q

    return dict(A=A0, B=B, P_of_Q=P_of_Q, TR=TR, MR=MR)


def tab_demand_revenue():
    st.header("Demand & Revenue")
    st.caption(
        "fits a straight-line demand from your (Q,P) points, plot **Demand** and **MR**, and show **TR** on the right axis."
    )

    ex_name = st.selectbox("Examples", list(demand_examples().keys()), index=0, key="dem_ex")
    c1, _ = st.columns([1, 4])
    if "dem_df" not in st.session_state:
        st.session_state.dem_df = demand_examples()[ex_name].copy()
    if c1.button("Load example", use_container_width=True, key="dem_load_btn"):
        st.session_state.dem_df = demand_examples()[ex_name].copy()

    dem_df = st.data_editor(
        st.session_state.dem_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Q": st.column_config.NumberColumn("Q (quantity)"),
            "P": st.column_config.NumberColumn("P (price)"),
        },
        key="dem_editor",
    )
    st.session_state.dem_df = dem_df

    fit = fit_linear_demand(dem_df)
    if fit is None:
        st.info("Add at least two (Q,P) rows to fit demand.")
        return

    A, B = fit["A"], fit["B"]
    P_of_Q, TR_fun, MR_fun = fit["P_of_Q"], fit["TR"], fit["MR"]

    q_min = float(dem_df["Q"].min())
    q_max = float(dem_df["Q"].max())
    q_lo, q_hi = _autoscale_range([q_min, q_max], pad_frac=0.15, min_pad=5.0)
    q_line = np.linspace(q_lo, q_hi, 260)
    p_line = P_of_Q(q_line)
    mr_line = MR_fun(q_line)
    tr_line = TR_fun(q_line)

    q_star = float("nan")
    p_star = float("nan")
    tr_star = float("nan")
    if abs(B) > 1e-12:
        q_star = -A / (2.0 * B)
        p_star = P_of_Q(q_star)
        tr_star = p_star * q_star

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_line, y=p_line, mode="lines", name="Demand (P)"))
    fig.add_trace(go.Scatter(x=q_line, y=mr_line, mode="lines", name="MR", line=dict(dash="dash")))
    fig.add_trace(
        go.Scatter(
            x=q_line,
            y=tr_line,
            mode="lines",
            name="TR = P×Q",
            line=dict(color="crimson", dash="dot"),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dem_df["Q"],
            y=dem_df["P"],
            mode="markers",
            name="Data",
            marker=dict(size=10, color="black"),
        )
    )
    if np.isfinite(q_star):
        fig.add_trace(
            go.Scatter(
                x=[q_star],
                y=[p_star],
                mode="markers+text",
                text=[f"TR max at Q*={q_star:.2f}"],
                textposition="top center",
                name="TR max (MR=0)",
                marker=dict(size=10, color="orange"),
            )
        )
    fig.update_layout(
        xaxis=dict(title="Q (quantity)"),
        yaxis=dict(title="Price / MR"),
        yaxis2=dict(title="Total Revenue (TR)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=40, b=35),
        hovermode="x unified",
        title="Demand, MR and TR",
    )
    st.plotly_chart(fig, use_container_width=True)

    work = pd.DataFrame({"Q": q_line, "Demand P(Q)": p_line, "MR": mr_line, "TR": tr_line})
    st.subheader("Workings (table)")
    st.dataframe(work.round(4), use_container_width=True, hide_index=True, height=320)
    _download_button_df(work.round(6), "Download workings (CSV)", "demand_workings.csv")

    st.markdown("### Simple Explaination")
    st.write(
        f"Fitted demand: **P = {A:.2f} + ({B:.2f})·Q**.  "
        f"Each extra unit changes price by **{B:.2f}** on average.  "
        f"**MR** doubles the slope: MR = {A:.2f} + {2*B:.2f}·Q.  "
        f"At **MR=0**, TR peaks (here Q≈{q_star:.2f})."
    )

    st.markdown("### Maths")
    st.latex(r"P(Q) = a + bQ \quad (b<0 \text{ for downward demand})")
    st.latex(r"TR(Q) = (a+bQ)\,Q")
    st.latex(r"MR(Q) = a + 2bQ")
    st.latex(r"TR \text{ maximized at } MR=0 \; \Rightarrow \; Q^\* = -\frac{a}{2b} \; (b\ne 0)")


# ==================================================================================================
# TAB 3: SUPPLY & COST (linear MC)
# ==================================================================================================

def supply_examples():
    return {
        "Factory (MC ≈ 6 + 0.20·Q, FC=1000)": dict(c=6.0, d=0.20, FC=1000.0, Qmax=120.0),
        "Workshop (MC ≈ 8 + 0.10·Q, FC=500)": dict(c=8.0, d=0.10, FC=500.0, Qmax=100.0),
        "Small plant (MC ≈ 10 + 0.30·Q, FC=800)": dict(c=10.0, d=0.30, FC=800.0, Qmax=100.0),
    }


def supply_mc_costs(c: float, d: float, FC: float, Qgrid: np.ndarray):
    Q = np.asarray(Qgrid, float)
    MC = c + d * Q
    VC = c * Q + 0.5 * d * Q * Q
    TC = FC + VC
    AVC = np.where(Q > 0, VC / Q, np.nan)
    ATC = np.where(Q > 0, TC / Q, np.nan)
    return dict(Q=Q, MC=MC, VC=VC, TC=TC, AVC=AVC, ATC=ATC)


def tab_supply_cost():
    st.header("Supply & Cost (linear MC)")
    st.caption(
        "Linear **MC**: \(P=MC=c+dQ\). Integrate to **VC**, add **FC** to get **TC**; compute **AVC** and **ATC**."
    )

    ex_name = st.selectbox("Examples", list(supply_examples().keys()), index=0, key="sup_ex")
    c1, _ = st.columns([1, 4])
    if "sup_params" not in st.session_state:
        st.session_state.sup_params = supply_examples()[ex_name].copy()
    if c1.button("Load example", use_container_width=True, key="sup_load_btn"):
        st.session_state.sup_params = supply_examples()[ex_name].copy()

    params = st.session_state.sup_params
    c, d, FC, Qmax0 = params["c"], params["d"], params["FC"], params["Qmax"]

    col1, col2, col3, col4 = st.columns(4)
    c = col1.number_input("c (intercept)", value=float(c), step=0.5, key="sup_c")
    d = col2.number_input("d (slope)", value=float(d), step=0.05, key="sup_d")
    FC = col3.number_input("Fixed cost (FC)", value=float(FC), step=50.0, key="sup_fc")
    Qmax = col4.number_input("Max Q for plot", min_value=10.0, value=float(Qmax0), step=10.0, key="sup_qmax")

    Q = np.linspace(0.0, Qmax, 260)
    costs = supply_mc_costs(c, d, FC, Q)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=costs["Q"], y=costs["MC"], name="Supply / MC", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=costs["Q"],
            y=costs["TC"],
            name="Total Cost (TC)",
            yaxis="y2",
            mode="lines",
            line=dict(dash="dot", color="crimson"),
        )
    )
    fig.update_layout(
        xaxis=dict(title="Q"),
        yaxis=dict(title="Price / MC"),
        yaxis2=dict(title="Total Cost", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=40, b=35),
        title=f"Supply/MC: P = {c:.2f} + {d:.2f}·Q   |   TC = FC + VC",
    )
    st.plotly_chart(fig, use_container_width=True)

    work = pd.DataFrame(
        {
            "Q": costs["Q"],
            "MC (P)": costs["MC"],
            "VC": costs["VC"],
            "FC": FC,
            "TC": costs["TC"],
            "AVC": costs["AVC"],
            "ATC": costs["ATC"],
        }
    )
    st.subheader("Workings (table)")
    st.dataframe(work.round(4), use_container_width=True, hide_index=True, height=320)
    _download_button_df(work.round(6), "Download workings (CSV)", "supply_cost_workings.csv")

    st.markdown("### Simple Explaination")
    st.write(
        f"With **MC = {c:.2f} + {d:.2f}·Q**, each extra unit raises marginal cost by **{d:.2f}**.  "
        f"Integrating MC gives **VC(Q) = {c:.2f}·Q + 0.5·{d:.2f}·Q²**.  "
        f"Adding fixed cost **FC={FC:.0f}** yields total cost.  Average costs divide by Q."
    )

    st.markdown("### Maths")
    st.latex(r"P = MC = c + dQ")
    st.latex(r"VC(Q) = \int_0^Q (c + d q)\,dq = cQ + \tfrac12 d Q^2")
    st.latex(r"TC(Q) = FC + VC(Q)")
    st.latex(r"AVC(Q) = \frac{VC(Q)}{Q} \quad (Q>0), \qquad ATC(Q) = \frac{TC(Q)}{Q} \quad (Q>0)")


# ==================================================================================================
# TAB 4: ELASTICITY (midpoint)
# ==================================================================================================

def elasticity_examples():
    return {
        "Price discount (coffee cups)": dict(Q1=100.0, P1=5.0, Q2=150.0, P2=4.0),
        "Movie night (tickets)": dict(Q1=200.0, P1=12.0, Q2=240.0, P2=10.0),
        "Snack stand (elastic)": dict(Q1=50.0, P1=3.0, Q2=90.0, P2=2.0),
    }


def midpoint_elasticity(Q1, P1, Q2, P2):
    avgQ = (Q1 + Q2) / 2.0
    avgP = (P1 + P2) / 2.0
    dQ = (Q2 - Q1)
    dP = (P2 - P1)
    dQ_over_avgQ = dQ / avgQ if avgQ != 0 else float("nan")
    dP_over_avgP = dP / avgP if avgP != 0 else float("nan")
    E = dQ_over_avgQ / dP_over_avgP if (not np.isnan(dP_over_avgP) and dP_over_avgP != 0) else float("nan")
    return dict(dQ=dQ, dP=dP, avgQ=avgQ, avgP=avgP, dQ_over_avgQ=dQ_over_avgQ, dP_over_avgP=dP_over_avgP, E=E)


def tab_elasticity():
    st.header("Elasticity (midpoint)")
    st.caption("Enter two points (Q1,P1) and (Q2,P2). This compute the midpoint elasticity and explain it.")

    ex_name = st.selectbox("Examples", list(elasticity_examples().keys()), index=0, key="ela_ex")
    c1, _ = st.columns([1, 4])
    if "ela_params" not in st.session_state:
        st.session_state.ela_params = elasticity_examples()[ex_name].copy()
    if c1.button("Load example", use_container_width=True, key="ela_load_btn"):
        st.session_state.ela_params = elasticity_examples()[ex_name].copy()

    p = st.session_state.ela_params
    cA, cB, cC, cD = st.columns(4)
    Q1 = cA.number_input("Q₁ (initial)", value=float(p["Q1"]), step=1.0, key="ela_q1")
    P1 = cB.number_input("P₁ (initial)", value=float(p["P1"]), step=0.5, key="ela_p1")
    Q2 = cC.number_input("Q₂ (new)", value=float(p["Q2"]), step=1.0, key="ela_q2")
    P2 = cD.number_input("P₂ (new)", value=float(p["P2"]), step=0.5, key="ela_p2")

    res = midpoint_elasticity(Q1, P1, Q2, P2)
    E = res["E"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[Q1, Q2], y=[P1, P2], mode="markers+lines", name="Change"))
    fig.update_layout(
        xaxis_title="Q",
        yaxis_title="P",
        legend=dict(orientation="h"),
        title="Two-point price/quantity change",
        margin=dict(l=30, r=20, t=40, b=35),
    )
    st.plotly_chart(fig, use_container_width=True)

    work = pd.DataFrame(
        [
            {
                "ΔQ": res["dQ"],
                "avgQ": res["avgQ"],
                "ΔQ/avgQ": res["dQ_over_avgQ"],
                "ΔP": res["dP"],
                "avgP": res["avgP"],
                "ΔP/avgP": res["dP_over_avgP"],
                "Elasticity E": res["E"],
            }
        ]
    )
    st.subheader("Workings (table)")
    st.dataframe(work.round(6), use_container_width=True, hide_index=True)
    _download_button_df(work.round(6), "Download workings (CSV)", "elasticity_workings.csv")

    st.markdown("### Simple Explaination")
    kind = (
        "elastic (|E|>1)"
        if np.isfinite(E) and abs(E) > 1
        else "inelastic (|E|<1)"
        if np.isfinite(E) and abs(E) < 1
        else "unit elastic (|E|=1)"
        if np.isfinite(E) and abs(E) == 1
        else "—"
    )
    st.write(
        f"With your numbers, **E ≈ {E:.3f} ⇒ {kind}**.  "
        "The sign is negative for a standard downward demand response (price up, quantity down)."
    )

    st.markdown("### Maths")
    st.latex(
        r"E \;=\; \frac{\frac{\Delta Q}{\overline{Q}}}{\frac{\Delta P}{\overline{P}}}"
        r"\;=\; \frac{\frac{Q_2 - Q_1}{(Q_1+Q_2)/2}}{\frac{P_2 - P_1}{(P_1+P_2)/2}}"
    )
    st.latex(
        r"|E|>1 \Rightarrow \text{elastic}, \quad |E|<1 \Rightarrow \text{inelastic}, \quad |E|=1 \Rightarrow \text{unit elastic}"
    )


# ==================================================================================================
# TAB 5: KINKED DEMAND
# ==================================================================================================

def kinked_examples():
    return {
        "Retailer (kink @ Q=100, P=50; slopes −0.5/−0.2)": dict(Qk=100.0, Pk=50.0, b1=-0.5, b2=-0.2),
        "Telecom (kink @ Q=80, P=60; slopes −0.6/−0.1)": dict(Qk=80.0, Pk=60.0, b1=-0.6, b2=-0.1),
        "Grocery (kink @ Q=120, P=40; slopes −0.4/−0.25)": dict(Qk=120.0, Pk=40.0, b1=-0.4, b2=-0.25),
    }


def tab_kinked():
    st.header("Kinked Demand (two slopes)")
    st.caption("Demand has one slope *above* the kink and another *below*. MR doubles each slope piecewise.")

    ex_name = st.selectbox("Examples", list(kinked_examples().keys()), index=0, key="kin_ex")
    c1, _ = st.columns([1, 4])
    if "kin_params" not in st.session_state:
        st.session_state.kin_params = kinked_examples()[ex_name].copy()
    if c1.button("Load example", use_container_width=True, key="kin_load_btn"):
        st.session_state.kin_params = kinked_examples()[ex_name].copy()

    p = st.session_state.kin_params
    Qk = st.number_input("Kink Qk", value=float(p["Qk"]), step=1.0, key="kin_qk")
    Pk = st.number_input("Kink Pk", value=float(p["Pk"]), step=1.0, key="kin_pk")
    b1 = st.number_input(
        "Slope above kink (b1)", value=float(p["b1"]), step=0.1, key="kin_b1", help="Expected negative (downward) above kink"
    )
    b2 = st.number_input(
        "Slope below kink (b2)", value=float(p["b2"]), step=0.1, key="kin_b2", help="Expected negative (downward) below kink"
    )

    q1 = np.linspace(max(0.0, Qk - 100.0), Qk, 160)
    q2 = np.linspace(Qk, Qk + 120.0, 160)
    p1 = Pk + b1 * (q1 - Qk)
    p2 = Pk + b2 * (q2 - Qk)
    mr1 = Pk + 2.0 * b1 * (q1 - Qk)
    mr2 = Pk + 2.0 * b2 * (q2 - Qk)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q1, y=p1, mode="lines", name="Demand (above kink)"))
    fig.add_trace(go.Scatter(x=q2, y=p2, mode="lines", name="Demand (below kink)"))
    fig.add_trace(go.Scatter(x=q1, y=mr1, mode="lines", name="MR (above)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=q2, y=mr2, mode="lines", name="MR (below)", line=dict(dash="dot")))
    fig.add_trace(
        go.Scatter(
            x=[Qk],
            y=[Pk],
            mode="markers+text",
            text=["Kink"],
            textposition="top center",
            marker=dict(size=10, color="black"),
            name="Kink",
        )
    )
    fig.update_layout(
        xaxis_title="Q",
        yaxis_title="P / MR",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=40, b=35),
        title="Kinked Demand & MR (piecewise slopes)",
    )
    st.plotly_chart(fig, use_container_width=True)

    work = pd.DataFrame({"Q (above)": q1, "P (above)": p1, "MR (above)": mr1})
    work2 = pd.DataFrame({"Q (below)": q2, "P (below)": p2, "MR (below)": mr2})
    st.subheader("Workings (tables)")
    colA, colB = st.columns(2)
    with colA:
        st.dataframe(work.round(4), use_container_width=True, hide_index=True)
        _download_button_df(work.round(6), "Download (above kink)", "kinked_above.csv")
    with colB:
        st.dataframe(work2.round(4), use_container_width=True, hide_index=True)
        _download_button_df(work2.round(6), "Download (below kink)", "kinked_below.csv")

    st.markdown("### Simple Explaination")
    st.write(
        f"Above the kink, your slope is **{b1:.2f}**, below the kink it is **{b2:.2f}**. "
        "MR mirrors this by doubling each slope on its own arm. The kink produces a **gap** in MR — "
        "one story behind 'sticky prices'."
    )

    st.markdown("### Maths")
    st.latex(r"P = P_k + b_1 (Q - Q_k) \;\Rightarrow\; MR = P_k + 2 b_1 (Q - Q_k)")
    st.latex(r"P = P_k + b_2 (Q - Q_k) \;\Rightarrow\; MR = P_k + 2 b_2 (Q - Q_k)")


# ==================================================================================================
# TAB 6: COST CURVES
# ==================================================================================================

def costcurve_examples():
    return {
        "U-shape (FC=200, b=5, c= -0.1, d=0.005)": dict(a=200.0, b=5.0, c=-0.10, d=0.005, Qmax=150.0),
        "Mild U-shape (FC=500, b=2, c=0.02, d=0.002)": dict(a=500.0, b=2.0, c=0.02, d=0.002, Qmax=150.0),
        "Steeper U (FC=300, b=10, c=-0.2, d=0.008)": dict(a=300.0, b=10.0, c=-0.2, d=0.008, Qmax=140.0),
    }


def cost_curve_build(a, b, c, d, Qgrid):
    """
    TC(Q) = a + bQ + cQ^2 + dQ^3
    MC(Q) = dTC/dQ = b + 2cQ + 3dQ^2
    AVC(Q) = (TC - a)/Q (for Q>0)
    ATC(Q) = TC/Q (for Q>0)
    """
    Q = np.asarray(Qgrid, float)
    TC = a + b * Q + c * Q ** 2 + d * Q ** 3
    MC = b + 2 * c * Q + 3 * d * Q ** 2
    AVC = np.where(Q > 0, (TC - a) / Q, np.nan)
    ATC = np.where(Q > 0, TC / Q, np.nan)
    return dict(Q=Q, TC=TC, MC=MC, AVC=AVC, ATC=ATC)


def tab_cost_curves():
    st.header("Cost Curves (U-shaped AC and MC)")
    st.caption("This builds a cubic **TC(Q)**, derive **MC**, **AVC**, **ATC**, and plot them all.")

    ex_name = st.selectbox("Examples", list(costcurve_examples().keys()), index=0, key="cost_ex")
    c1, _ = st.columns([1, 4])
    if "cost_params" not in st.session_state:
        st.session_state.cost_params = costcurve_examples()[ex_name].copy()
    if c1.button("Load example", use_container_width=True, key="cost_load_btn"):
        st.session_state.cost_params = costcurve_examples()[ex_name].copy()

    p = st.session_state.cost_params
    col1, col2, col3, col4, col5 = st.columns(5)
    a = col1.number_input("a (FC)", value=float(p["a"]), step=10.0, key="cost_a")
    b = col2.number_input("b", value=float(p["b"]), step=1.0, key="cost_b")
    c = col3.number_input("c", value=float(p["c"]), step=0.01, key="cost_c", format="%.3f")
    d = col4.number_input("d", value=float(p["d"]), step=0.001, key="cost_d", format="%.4f")
    Qmax = col5.number_input("Max Q", value=float(p["Qmax"]), step=10.0, key="cost_qmax")

    Q = np.linspace(0.0, Qmax, 260)
    cc = cost_curve_build(a, b, c, d, Q)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cc["Q"], y=cc["MC"], name="MC", mode="lines"))
    fig.add_trace(go.Scatter(x=cc["Q"], y=cc["AVC"], name="AVC", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=cc["Q"], y=cc["ATC"], name="ATC", mode="lines", line=dict(dash="dot")))
    fig.update_layout(
        xaxis_title="Q", yaxis_title="Cost per unit", legend=dict(orientation="h"),
        margin=dict(l=30, r=20, t=40, b=35), title="MC, AVC, and ATC"
    )
    st.plotly_chart(fig, use_container_width=True)

    diff = np.abs(cc["MC"] - cc["ATC"])
    i_min = int(np.nanargmin(diff))
    q_min_atc = float(cc["Q"][i_min])
    atc_min = float(cc["ATC"][i_min])

    st.markdown("### Key markers")
    st.write(f"**MC = ATC** near **Q ≈ {q_min_atc:.2f}**, where **ATC ≈ {atc_min:.2f}** is minimized.")

    work = pd.DataFrame({"Q": cc["Q"], "TC": cc["TC"], "MC": cc["MC"], "AVC": cc["AVC"], "ATC": cc["ATC"]})
    st.subheader("Workings (table)")
    st.dataframe(work.round(4), use_container_width=True, hide_index=True, height=320)
    _download_button_df(work.round(6), "Download workings (CSV)", "cost_curves_workings.csv")

    st.markdown("### Maths")
    st.latex(r"TC(Q) = a + bQ + cQ^2 + dQ^3")
    st.latex(r"MC(Q) = \frac{dTC}{dQ} = b + 2cQ + 3dQ^2")
    st.latex(r"AVC(Q) = \frac{TC(Q) - a}{Q} \quad (Q>0)")
    st.latex(r"ATC(Q) = \frac{TC(Q)}{Q} \quad (Q>0)")
    st.latex(r"\text{Typically, } MC \text{ intersects } ATC \text{ at } ATC \text{ minimum.}")


# ==================================================================================================
# Sidebar (single, static) — no toggles, no duplicates
# ==================================================================================================

def render_sidebar_once():
    st.sidebar.header("Quick tips")
    st.sidebar.caption(
        "- Each tab has its own inputs and a small workings table.\n"
        "- Edit numbers directly in tables; blank rows are ignored.\n"
        "- Use **Examples** to auto-fill realistic data; you can still tweak it.\n"
        "- You can **download** every workings table as CSV."
    )


# ==================================================================================================
# Main
# ==================================================================================================

def main():
    st.set_page_config(page_title="Financial graph visualizer", layout="wide")
    render_sidebar_once()
    tabs = st.tabs(
        [
            "Net Payoff (options)",
            "Demand & Revenue",
            "Supply & Cost",
            "Elasticity (midpoint)",
            "Kinked Demand",
            "Cost Curves",
        ]
    )
    with tabs[0]:
        tab_netpayoff()
    with tabs[1]:
        tab_demand_revenue()
    with tabs[2]:
        tab_supply_cost()
    with tabs[3]:
        tab_elasticity()
    with tabs[4]:
        tab_kinked()
    with tabs[5]:
        tab_cost_curves()


if __name__ == "__main__":
    main()

# ==================================================================================================
# APPENDIX (Teaching notes & padding so the single-file variant remains self-contained)
# --------------------------------------------------------------------------------------------------
# • This section is optional. It does not affect runtime. It exists to include concise notes that
#   students can read in the source, and to satisfy requests for a long, well-commented file.
#
# OPTIONS CHEAT-SHEET
# -------------------
# Long Call payoff:        max(S - K, 0) - premium
# Short Call payoff:     - (max(S - K, 0) - premium)
# Long Put payoff:         max(K - S, 0) - premium
# Short Put payoff:      - (max(K - S, 0) - premium)
#
# Strategy payoff = sum over legs (sign × (intrinsic - premium) × quantity × lot)
# sign = +1 for long, −1 for short.  Intrinsic depends on type (call/put).
#
# Break-even for single leg (ignoring lot & qty):
#   Call: S = K + premium
#   Put : S = K - premium
#
# DEMAND & REVENUE
# ----------------
# Linear demand P(Q) = a + bQ (b < 0 typical).  TR(Q) = P(Q)·Q.  MR(Q) = dTR/dQ = a + 2bQ.
# TR maximized at MR=0  ⇒  Q* = -a/(2b)  (if b ≠ 0).  On the chart we overlay TR on a right axis.
#
# SUPPLY & COST
# -------------
# With linear MC = c + dQ, variable cost VC(Q) = ∫(c + dQ)dQ = cQ + ½ dQ².  Total cost TC = FC + VC.
# AVC = VC/Q (Q>0).  ATC = TC/Q (Q>0).  In typical “U-shaped” settings, MC cuts ATC at ATC’s minimum.
#
# ELASTICITY (MIDPOINT)
# ---------------------
# E = (ΔQ/average Q) / (ΔP/average P).
# Use averages to avoid asymmetry when moving up versus down a demand curve.
#
# KINKED DEMAND
# -------------
# Two slopes around a kink at (Qk,Pk).  Piecewise linear demand with MR doubling slope on each side.
# The MR “gap” can explain sticky prices in oligopoly models.
#
# COST CURVES (CUBIC TC)
# ----------------------
# Pick TC(Q) = a + bQ + cQ² + dQ³ so that AC curves are U-shaped. MC = dTC/dQ intersects ATC at min.
#
# IMPLEMENTATION NOTES
# --------------------
# 1) We sanitize the legs table via clean_legs_df() before ANY calculation. This permanently fixes
#    “ValueError: cannot convert float NaN to integer” that you saw when Quantity was NaN.
# 2) We removed selection_mode or other unsupported args from st.data_editor; only Streamlit 1.33+
#    compatible options are used.
# 3) The delete workflow is explicit: tick “Delete” then press the delete button (or use the
#    multiselect). The far-left gutter in data_editor is a row handle, not a selection control.
# 4) Autoscale: we first probe a provisional grid to find break-evens (zero-crossings) and then widen
#    the true X-range so the flips are visible along with strikes and the current spot.
# 5) All formulas that should render as math use st.latex(...) — no stray “text/int” artifacts.
#
# End of file.
# ==================================================================================================
