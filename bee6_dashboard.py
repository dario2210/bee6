"""
Bee6 Dashboard — port 8066.
"""

from __future__ import annotations

import json
import threading
import time as _time
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots

from bee6_binance import update_csv_cache
from bee6_data import load_klines, prepare_indicators
from bee6_params import (
    BINANCE_INTERVAL,
    BINANCE_MARKET,
    BINANCE_START_DATE,
    BINANCE_SYMBOL,
    CSV_PATH,
    DEFAULT_PARAMS,
    INITIAL_CAPITAL,
    LIVE_DAYS,
    OPT_DAYS,
    save_params,
)
from bee6_stats import (
    breakdown_by_regime,
    breakdown_by_side,
    breakdown_by_strategy,
    compute_stats,
    wfo_summary,
)
from bee6_strategy import Bee6Strategy
from bee6_wfo import walk_forward_optimization

C = {
    "bg": "#07101c",
    "surface": "rgba(10, 20, 35, 0.88)",
    "surf2": "rgba(16, 28, 50, 0.92)",
    "border": "rgba(120, 150, 185, 0.18)",
    "text": "#eef6ff",
    "muted": "#8899bb",
    "green": "#1de9b6",
    "red": "#ff4d6d",
    "blue": "#5eb8ff",
    "amber": "#ffc046",
    "purple": "#a78bfa",
    "coral": "#ff7a59",
    "teal": "#00bcd4",
}

_SERVER_TOKEN = str(int(_time.time()))
_lock = threading.Lock()
_state = {
    "running": False,
    "stop": False,
    "status": "Ready.",
    "progress": "",
    "result": None,
    "result_version": 0,
}
_APP_DIR = Path(__file__).resolve().parent


def gs() -> dict:
    with _lock:
        return dict(_state)


def ss(**kwargs) -> None:
    with _lock:
        if "result" in kwargs:
            _state["result_version"] += 1
        _state.update(kwargs)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def lbl(text):
    return html.Div(text, style={
        "fontSize": "11px", "color": C["muted"], "marginBottom": "4px",
        "fontWeight": "600", "letterSpacing": "0.05em", "textTransform": "uppercase",
    })


def inp(id_, value, **kwargs):
    return dcc.Input(id=id_, value=value, debounce=True, style={
        "width": "100%", "background": C["surf2"],
        "border": f"1px solid {C['border']}", "borderRadius": "12px",
        "color": C["text"], "padding": "9px 12px", "fontSize": "13px",
        "boxSizing": "border-box",
    }, **kwargs)


def drp(id_, options, value):
    return dcc.Dropdown(id=id_, options=options, value=value, clearable=False,
                        style={"color": "#0b1220"})


def field(label, ctrl):
    return html.Div([lbl(label), ctrl], style={"marginBottom": "10px"})


def btn(id_, label, color=C["blue"], text_color="#fff"):
    return html.Button(label, id=id_, n_clicks=0, style={
        "width": "100%", "background": color, "border": "none",
        "borderRadius": "14px", "color": text_color, "padding": "11px 14px",
        "fontSize": "13px", "fontWeight": "800", "cursor": "pointer",
        "marginBottom": "8px",
    })


card_s = {
    "background": C["surface"],
    "border": f"1px solid {C['border']}",
    "borderRadius": "22px",
    "padding": "16px 18px",
    "marginBottom": "12px",
    "boxShadow": "0 16px 48px rgba(0,0,0,0.32)",
    "backdropFilter": "blur(20px)",
}


def sec(text):
    return html.Div(text, style={
        "fontSize": "11px", "fontWeight": "700", "color": C["muted"],
        "textTransform": "uppercase", "letterSpacing": "0.06em",
        "marginBottom": "8px", "marginTop": "4px",
    })


def _num(v, default, cast=float):
    if v in (None, ""):
        return default
    try:
        return cast(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Params from controls
# ---------------------------------------------------------------------------

def _params_from_controls(
    direction, initial_capital, risk_pct, fee_rate, slippage_bps,
    adx_trend, adx_range, bb_range, bb_compression, volume_spike,
    donchian_len, trend_rsi_min, trend_rsi_max,
    macd_mode, range_require_williams, use_candle_filter, wfo_mode, score_mode,
) -> tuple[dict, float, str, str]:
    params = dict(DEFAULT_PARAMS)
    params.update({
        "trade_direction": direction or DEFAULT_PARAMS["trade_direction"],
        "risk_pct": _num(risk_pct, DEFAULT_PARAMS["risk_pct"]),
        "fee_rate": _num(fee_rate, DEFAULT_PARAMS["fee_rate"]),
        "slippage_bps": _num(slippage_bps, DEFAULT_PARAMS["slippage_bps"]),
        "adx_trend_threshold": _num(adx_trend, DEFAULT_PARAMS["adx_trend_threshold"]),
        "adx_range_threshold": _num(adx_range, DEFAULT_PARAMS["adx_range_threshold"]),
        "bb_width_range_max": _num(bb_range, DEFAULT_PARAMS["bb_width_range_max"]),
        "bb_width_compression_max": _num(bb_compression, DEFAULT_PARAMS["bb_width_compression_max"]),
        "volume_spike_mult": _num(volume_spike, DEFAULT_PARAMS["volume_spike_mult"]),
        "donchian_len": _num(donchian_len, DEFAULT_PARAMS["donchian_len"], int),
        "trend_rsi_min": _num(trend_rsi_min, DEFAULT_PARAMS["trend_rsi_min"]),
        "trend_rsi_max": _num(trend_rsi_max, DEFAULT_PARAMS["trend_rsi_max"]),
        "macd_mode": macd_mode or DEFAULT_PARAMS["macd_mode"],
        "range_require_williams": bool(range_require_williams),
        "use_candle_filter": bool(use_candle_filter),
    })
    return params, _num(initial_capital, INITIAL_CAPITAL), wfo_mode or "rolling", score_mode or "balanced"


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------

def _load_data(update: bool = False) -> pd.DataFrame:
    csv_path = str(_APP_DIR / CSV_PATH)
    if update or not Path(csv_path).exists():
        update_csv_cache(
            csv_path=csv_path, symbol=BINANCE_SYMBOL, interval=BINANCE_INTERVAL,
            start_date=BINANCE_START_DATE, market=BINANCE_MARKET, verbose=True,
        )
    return prepare_indicators(load_klines(csv_path))


def _run_update_data():
    try:
        ss(running=True, stop=False, status="Downloading ETHUSDT 1H from Binance...", progress="")
        df = _load_data(update=True)
        ss(running=False,
           status=f"Data updated: {len(df):,} candles.",
           progress=f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}",
           result={"mode": "data", "df": df})
    except Exception as exc:
        ss(running=False, status=f"Data update failed: {exc!r}", progress="")


def _run_backtest(params: dict, initial_capital: float):
    try:
        ss(running=True, stop=False, status="Running backtest...", progress="")
        df = _load_data()
        trades, equity, final_cap = Bee6Strategy(params).run(df, initial_capital)
        stats = compute_stats(trades, equity, initial_capital, print_output=False)
        ss(running=False,
           status=f"Backtest done: {len(trades)} trades | final {final_cap:,.0f} USD",
           progress="",
           result={"mode": "backtest", "df": df, "trades": trades,
                   "equity": equity, "stats": stats, "params": params,
                   "windows": pd.DataFrame()})
    except Exception as exc:
        ss(running=False, status=f"Backtest failed: {exc!r}", progress="")


def _run_wfo(params: dict, initial_capital: float, wfo_mode: str, score_mode: str):
    try:
        ss(running=True, stop=False, status="Running WFO...", progress="")
        df = _load_data()

        def on_combo(window_id, total_windows, combo_idx, combo_total):
            ss(status=f"WFO window {window_id + 1}/{max(total_windows, 1)}",
               progress=f"Combo {combo_idx}/{combo_total}")

        def on_window(row):
            ss(status=f"Window {row['window_id']} done",
               progress=f"Live ret {row['live_return_pct']:+.2f}% | trades {row['n_trades_live']}")

        trades, equity, windows, final_cap, stopped, best_params = walk_forward_optimization(
            df, score_mode=score_mode, wfo_mode=wfo_mode, verbose=True,
            on_combo_progress=on_combo, on_window_done=on_window,
            should_stop=lambda: gs().get("stop", False),
            opt_days=OPT_DAYS, live_days=LIVE_DAYS,
            initial_capital=initial_capital, base_params=params,
        )
        stats = compute_stats(trades, equity, initial_capital, print_output=False)
        if best_params:
            save_params(best_params, str(_APP_DIR / "bee6_wfo_best_params.json"))
        summary = wfo_summary(windows)
        status = (
            "WFO stopped." if stopped
            else f"WFO done: {len(windows)} windows | {summary.get('pct_profitable_windows', 0):.0f}% profitable | final {final_cap:,.0f} USD"
        )
        ss(running=False, stop=False, status=status, progress="Best params saved." if best_params else "",
           result={"mode": "wfo", "df": df, "trades": trades, "equity": equity,
                   "stats": stats, "params": best_params or params, "windows": windows})
    except Exception as exc:
        ss(running=False, stop=False, status=f"WFO failed: {exc!r}", progress="")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _empty_fig(msg="Run data update, backtest or WFO."):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=700,
        annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                       "x": 0.5, "y": 0.5, "showarrow": False,
                       "font": {"color": C["muted"], "size": 16}}],
    )
    return fig


def _build_figure(result: dict | None):
    if not result:
        return _empty_fig()
    df = result.get("df")
    if df is None or df.empty:
        return _empty_fig("No data loaded.")

    trades = result.get("trades", pd.DataFrame())
    equity = result.get("equity", pd.DataFrame())
    plot_df = df.tail(900).copy()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.028,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        specs=[[{}], [{}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # Row 1: Candles + EMAs + BB
    fig.add_trace(go.Candlestick(
        x=plot_df["time"], open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"], name="ETHUSDT",
        increasing_line_color=C["green"], decreasing_line_color=C["red"],
    ), row=1, col=1)

    for col, color, name in [
        ("ema50", C["blue"], "EMA50"), ("ema200", C["purple"], "EMA200"),
        ("bb_upper", "rgba(255,192,70,0.4)", "BB+"), ("bb_lower", "rgba(255,192,70,0.4)", "BB-"),
    ]:
        if col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["time"], y=plot_df[col], mode="lines", name=name,
                line={"color": color, "width": 1},
            ), row=1, col=1)

    # Trade markers
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
        t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
        t = t[t["entry_time"] >= plot_df["time"].min()]

        for side, marker, color in [("long", "triangle-up", C["green"]), ("short", "triangle-down", C["red"])]:
            sub = t[t["side"] == side]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["entry_time"], y=sub["entry_price"], mode="markers",
                    name=f"{side.title()} entry",
                    marker={"symbol": marker, "color": color, "size": 12},
                    text=sub["strategy"],
                ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=t["exit_time"], y=t["exit_price"], mode="markers", name="Exit",
            marker={"symbol": "x", "color": C["amber"], "size": 9},
            text=t["reason"],
        ), row=1, col=1)

    # Row 2: Equity
    if equity is not None and not equity.empty:
        fig.add_trace(go.Scatter(
            x=equity["time"], y=equity["equity"], mode="lines", name="Equity",
            line={"color": C["green"], "width": 2},
        ), row=2, col=1)

    # Row 3: RSI + Stoch K + Williams %R
    for col, color, name, row, sec_y in [
        ("rsi", C["amber"], "RSI", 3, False),
        ("stoch_k", C["teal"], "Stoch K", 3, True),
    ]:
        if col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df["time"], y=plot_df[col], mode="lines", name=name,
                line={"color": color, "width": 1.3},
            ), row=row, col=1, secondary_y=sec_y)

    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,77,109,0.5)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(29,233,182,0.5)", row=3, col=1)

    # Row 4: MACD histogram + ADX
    if "macd_hist" in plot_df.columns:
        colors = [C["green"] if v >= 0 else C["red"] for v in plot_df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=plot_df["time"], y=plot_df["macd_hist"], name="MACD hist",
            marker_color=colors, opacity=0.75,
        ), row=4, col=1, secondary_y=False)

    if "adx" in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df["time"], y=plot_df["adx"], mode="lines", name="ADX",
            line={"color": C["blue"], "width": 1.3},
        ), row=4, col=1, secondary_y=True)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=700,
        margin={"l": 20, "r": 20, "t": 16, "b": 16},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        font={"color": C["text"]},
        xaxis_rangeslider_visible=False,
        barmode="relative",
    )
    fig.update_xaxes(gridcolor="rgba(120,150,185,0.08)")
    fig.update_yaxes(gridcolor="rgba(120,150,185,0.08)")
    return fig


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

def _kpi(title, value, color=C["text"]):
    return html.Div([
        html.Div(title, style={"fontSize": "10px", "color": C["muted"], "marginBottom": "3px"}),
        html.Div(value, style={"fontSize": "18px", "fontWeight": "800", "color": color}),
    ], style={
        "background": C["surf2"], "border": f"1px solid {C['border']}",
        "borderRadius": "16px", "padding": "12px 13px",
    })


def _fmt(v, suffix="", dec=2):
    if v is None:
        return "n/d"
    try:
        if np.isnan(float(v)):
            return "n/d"
        return f"{float(v):,.{dec}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _kpis(result: dict | None):
    stats = (result or {}).get("stats") or {}
    if not stats:
        return html.Div("No results yet.", style={"color": C["muted"], "fontSize": "13px"})
    ret = stats.get("net_return_pct")
    dd = stats.get("max_drawdown_pct")
    return html.Div([
        _kpi("Net return",     _fmt(ret, "%"),                    C["green"] if (ret or 0) >= 0 else C["red"]),
        _kpi("CAGR",           _fmt(stats.get("cagr_pct"), "%"),  C["green"]),
        _kpi("Max DD",         _fmt(dd, "%"),                     C["red"] if (dd or 0) < 0 else C["muted"]),
        _kpi("Profit factor",  _fmt(stats.get("profit_factor")),  C["amber"]),
        _kpi("Trades",         _fmt(stats.get("n_trades"), dec=0), C["blue"]),
        _kpi("Win %",          _fmt(stats.get("winrate_pct"), "%"), C["teal"]),
        _kpi("Sharpe",         _fmt(stats.get("sharpe_ratio")),   C["text"]),
        _kpi("Sortino",        _fmt(stats.get("sortino_ratio")),  C["text"]),
        _kpi("Avg R",          _fmt(stats.get("avg_r_multiple")), C["green"]),
        _kpi("Consistency",    _fmt(stats.get("consistency_pct"), "%"), C["purple"]),
        _kpi("Fees",           _fmt(stats.get("fee_total_usd"), " USD"), C["muted"]),
        _kpi("Stability",      _fmt(stats.get("stability_score")), C["teal"]),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
        "gap": "8px",
    })


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _table_payload(df: pd.DataFrame, max_rows: int = 80):
    if df is None or df.empty:
        return [], []
    out = df.copy().tail(max_rows)
    for col in out.columns:
        if "time" in col or col.endswith("_start") or col.endswith("_end"):
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: None if pd.isna(x) else round(float(x), 4))
    return out.to_dict("records"), [{"name": str(c), "id": str(c)} for c in out.columns]


def _breakdown_payload(result: dict | None):
    trades = (result or {}).get("trades")
    if trades is None or trades.empty:
        return [], []
    parts = []
    for label, fn in [("side", breakdown_by_side), ("strategy", breakdown_by_strategy), ("regime", breakdown_by_regime)]:
        df = fn(trades)
        if not df.empty:
            df.insert(0, "breakdown", label)
            parts.append(df)
    combo = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _table_payload(combo, 40)


tbl_style = {
    "style_table": {"overflowX": "auto"},
    "style_cell": {
        "backgroundColor": C["surf2"], "color": C["text"],
        "border": f"1px solid {C['border']}", "fontSize": "12px",
        "padding": "7px", "maxWidth": "160px",
        "overflow": "hidden", "textOverflow": "ellipsis",
    },
    "style_header": {"backgroundColor": "#0c1829", "fontWeight": "800"},
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="bee6")
server = app.server

sidebar = html.Div([
    html.Div("bee6", style={"fontSize": "26px", "fontWeight": "900"}),
    html.Div("ETH/USDT 1H · Multi-indicator · DEX-ready",
             style={"color": C["muted"], "fontSize": "12px", "marginBottom": "16px"}),

    html.Div([
        sec("Execution"),
        field("Direction", drp("direction",
            [{"label": "Both", "value": "both"},
             {"label": "Long only", "value": "long"},
             {"label": "Short only", "value": "short"}],
            DEFAULT_PARAMS["trade_direction"])),
        field("Initial capital", inp("initial-capital", INITIAL_CAPITAL, type="number")),
        field("Risk per trade", inp("risk-pct", DEFAULT_PARAMS["risk_pct"], type="number")),
        field("Fee rate", inp("fee-rate", DEFAULT_PARAMS["fee_rate"], type="number")),
        field("Slippage bps", inp("slippage-bps", DEFAULT_PARAMS["slippage_bps"], type="number")),
    ], style=card_s),

    html.Div([
        sec("Regime filters"),
        field("ADX trend >", inp("adx-trend", DEFAULT_PARAMS["adx_trend_threshold"], type="number")),
        field("ADX range <", inp("adx-range", DEFAULT_PARAMS["adx_range_threshold"], type="number")),
        field("BB range max", inp("bb-range", DEFAULT_PARAMS["bb_width_range_max"], type="number")),
        field("BB compression max", inp("bb-compression", DEFAULT_PARAMS["bb_width_compression_max"], type="number")),
        field("Volume spike x", inp("volume-spike", DEFAULT_PARAMS["volume_spike_mult"], type="number")),
        field("Donchian len", inp("donchian-len", DEFAULT_PARAMS["donchian_len"], type="number")),
        field("Trend RSI min", inp("trend-rsi-min", DEFAULT_PARAMS["trend_rsi_min"], type="number")),
        field("Trend RSI max", inp("trend-rsi-max", DEFAULT_PARAMS["trend_rsi_max"], type="number")),
    ], style=card_s),

    html.Div([
        sec("New indicators"),
        field("MACD mode", drp("macd-mode",
            [{"label": "Histogram cross", "value": "histogram"},
             {"label": "Signal line cross", "value": "signal_cross"}],
            DEFAULT_PARAMS["macd_mode"])),
        field("Williams %R filter (range)", drp("range-require-williams",
            [{"label": "Enabled", "value": True},
             {"label": "Disabled", "value": False}],
            DEFAULT_PARAMS["range_require_williams"])),
        field("Candle pattern filter", drp("use-candle-filter",
            [{"label": "Enabled", "value": True},
             {"label": "Disabled", "value": False}],
            DEFAULT_PARAMS["use_candle_filter"])),
    ], style=card_s),

    html.Div([
        sec("WFO settings"),
        field("WFO mode", drp("wfo-mode",
            [{"label": "Rolling (classic)", "value": "rolling"},
             {"label": "Anchored (expanding)", "value": "anchored"}],
            "rolling")),
        field("Score mode", drp("score-mode",
            [{"label": "Balanced", "value": "balanced"},
             {"label": "Defensive", "value": "defensive"},
             {"label": "Return only", "value": "return_only"}],
            "balanced")),
    ], style=card_s),

    html.Div([
        btn("update-data", "Update data", C["purple"]),
        btn("run-backtest", "Run backtest", C["blue"]),
        btn("run-wfo", "Run WFO", C["green"], "#051a0e"),
        btn("stop-job", "Stop", C["red"]),
    ], style=card_s),

], style={"width": "310px", "minWidth": "310px"})

main_area = html.Div([
    html.Div([
        html.Div(id="status", style={"fontWeight": "800", "fontSize": "15px"}),
        html.Div(id="progress", style={"color": C["muted"], "fontSize": "12px", "marginTop": "3px"}),
    ], style=card_s),

    html.Div(id="kpis", style=card_s),

    html.Div([dcc.Graph(id="main-chart", figure=_empty_fig(),
                        config={"displayModeBar": True})], style=card_s),

    html.Div([
        sec("Trades"),
        dash_table.DataTable(id="trades-table", data=[], columns=[],
                             page_size=12, sort_action="native", **tbl_style),
    ], style=card_s),

    html.Div([
        sec("WFO windows"),
        dash_table.DataTable(id="windows-table", data=[], columns=[],
                             page_size=8, sort_action="native", **tbl_style),
    ], style=card_s),

    html.Div([
        sec("Breakdown by side / strategy / regime"),
        dash_table.DataTable(id="breakdown-table", data=[], columns=[],
                             page_size=12, sort_action="native", **tbl_style),
    ], style=card_s),

    html.Pre(id="params-preview", style={
        **card_s, "whiteSpace": "pre-wrap",
        "color": C["muted"], "fontSize": "12px",
    }),
], style={"flex": "1", "minWidth": "0"})

app.layout = html.Div([
    dcc.Interval(id="poll", interval=1400, n_intervals=0),
    dcc.Store(id="server-token", data=_SERVER_TOKEN),
    html.Div([sidebar, main_area],
             style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
], style={
    "minHeight": "100vh", "background": C["bg"],
    "color": C["text"],
    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
    "padding": "20px", "boxSizing": "border-box",
})

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("status", "children", allow_duplicate=True),
    Input("update-data", "n_clicks"),
    Input("run-backtest", "n_clicks"),
    Input("run-wfo", "n_clicks"),
    Input("stop-job", "n_clicks"),
    State("direction", "value"),
    State("initial-capital", "value"),
    State("risk-pct", "value"),
    State("fee-rate", "value"),
    State("slippage-bps", "value"),
    State("adx-trend", "value"),
    State("adx-range", "value"),
    State("bb-range", "value"),
    State("bb-compression", "value"),
    State("volume-spike", "value"),
    State("donchian-len", "value"),
    State("trend-rsi-min", "value"),
    State("trend-rsi-max", "value"),
    State("macd-mode", "value"),
    State("range-require-williams", "value"),
    State("use-candle-filter", "value"),
    State("wfo-mode", "value"),
    State("score-mode", "value"),
    prevent_initial_call=True,
)
def start_job(
    _u, _b, _w, _s,
    direction, initial_capital, risk_pct, fee_rate, slippage_bps,
    adx_trend, adx_range, bb_range, bb_compression, volume_spike,
    donchian_len, trend_rsi_min, trend_rsi_max,
    macd_mode, range_require_williams, use_candle_filter,
    wfo_mode, score_mode,
):
    trigger = dash.ctx.triggered_id
    state = gs()

    if trigger == "stop-job":
        ss(stop=True, status="Stop requested.", progress="Finishing current combo…")
        return "Stop requested."
    if state.get("running"):
        return "A job is already running."

    params, init_cap, wfo_mode, score_mode = _params_from_controls(
        direction, initial_capital, risk_pct, fee_rate, slippage_bps,
        adx_trend, adx_range, bb_range, bb_compression, volume_spike,
        donchian_len, trend_rsi_min, trend_rsi_max,
        macd_mode, range_require_williams, use_candle_filter,
        wfo_mode, score_mode,
    )

    if trigger == "update-data":
        threading.Thread(target=_run_update_data, daemon=True).start()
    elif trigger == "run-backtest":
        threading.Thread(target=_run_backtest, args=(params, init_cap), daemon=True).start()
    elif trigger == "run-wfo":
        threading.Thread(target=_run_wfo, args=(params, init_cap, wfo_mode, score_mode), daemon=True).start()
    else:
        return state.get("status", "Ready.")
    return "Starting…"


@app.callback(
    Output("status", "children"),
    Output("progress", "children"),
    Output("kpis", "children"),
    Output("main-chart", "figure"),
    Output("trades-table", "data"),
    Output("trades-table", "columns"),
    Output("windows-table", "data"),
    Output("windows-table", "columns"),
    Output("breakdown-table", "data"),
    Output("breakdown-table", "columns"),
    Output("params-preview", "children"),
    Input("poll", "n_intervals"),
)
def refresh(_):
    state = gs()
    result = state.get("result")
    trades_data, trades_cols = _table_payload((result or {}).get("trades", pd.DataFrame()), 120)
    windows_data, windows_cols = _table_payload((result or {}).get("windows", pd.DataFrame()), 80)
    breakdown_data, breakdown_cols = _breakdown_payload(result)
    params = (result or {}).get("params") or {}
    params_text = json.dumps(params, indent=2) if params else "No active params yet."
    return (
        state.get("status", ""),
        state.get("progress", ""),
        _kpis(result),
        _build_figure(result),
        trades_data, trades_cols,
        windows_data, windows_cols,
        breakdown_data, breakdown_cols,
        params_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8066, debug=False)
