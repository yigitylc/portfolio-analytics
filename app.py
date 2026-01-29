"""
================================================================================
Advanced Portfolio Analytics Dashboard - Streamlit App
================================================================================
Two-portfolio comparison tool with:
- Cumulative returns (log scale), drawdown, annual returns, distribution
- Monthly heatmap, worst drawdowns table
- Rolling Sharpe & Beta (adjustable window)
- Correlation matrix (trailing window) and pair rolling correlation
- Full metrics: CAGR, Sharpe, Sortino, Calmar, VaR, CVaR, Beta, Alpha, IR
- Portfolio modes: Daily Rebalanced (constant weights) vs Buy & Hold (drift)
================================================================================
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import warnings
from typing import List, Tuple, Optional, Set, Dict, Any

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CONSTANTS & TICKER UNIVERSE
# =============================================================================

PORT_COLOR = "#1f77b4"
BMK_COLOR = "#d62728"
AVG_PORT = "#7fb3d5"
AVG_BMK = "#f5b7b1"

DOT_TO_DASH = {"BRK.B": "BRK-B", "BF.B": "BF-B", "PBR.A": "PBR-A", "EBR.B": "EBR-B"}

BASE_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLY", "XLP", "XLF", "XLV", "XLI", "XLB", "XLK", "XBI", "SMH","IGV", "XLC", "XLU",
    "XME", "GDX", "URA", "XOP", "XHB", "XLRE", "XRT",
    "IBIT", "BIL",
    "EFA", "VGK", "VEA", "EEM", "EUFN",
    "EWA", "EWC", "EWQ", "EWG", "EWI", "EWJ", "EWP", "EWU", "EWL",
    "FXI", "KWEB", "EWZ", "EWW", "EWS", "EWY", "EWT", "INDA", "EWH", "EZA",
    "SHY", "IEI", "IEF", "TLT", "TIP", "STIP", "IGSB", "HYG", "EMB", "BNDX", "BWX", "HYEM", "IAGG",
    "DBC", "GSG", "USO", "UNG", "GLD", "SLV", "DBA",
    "VNQ", "REM",
    "UUP", "VIXY",
    "TUR",
    "AAPL", "BABA", "BIDU", "BILI", "UNH", "PANW", "PLTR", "SNOW", "TSLA", "NVDA", "MU", "JD",
]

ADR_TICKERS = [
    "TSM", "AZN", "ASML", "SMFG", "HSBC", "TM", "NVS", "NVO", "SAP", "SHEL",
    "MUFG", "SAN", "BHP", "HDB", "TTE", "RIO", "BBVA", "UL", "BUD", "SONY",
    "BTI", "ARM", "SNY", "MFG", "GSK", "GDS", "VNET", "KC", "XPEV", "TME",
    "TCOM", "BP", "BCS", "PBR", "PBR-A", "ING",
    "LYG", "NTES", "PKX", "SE", "INFY", "VALE", "RELX", "EQNR", "ARGX",
    "IBN", "TAK", "GFI", "DEO", "HLN", "ABEV", "VIPS", "BBD",
    "ASX", "MT", "PUK", "HMC", "NOK", "CUK", "PDD", "TEVA", "ERIC",
    "KB", "RYAAY", "VOD", "IX", "CHT", "WDS", "BSBR", "UMC", "E",
]


def sanitize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if t in DOT_TO_DASH:
        return DOT_TO_DASH[t]
    return t.replace(".", "-")


def unique_order(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        xu = x.upper()
        if xu not in seen:
            seen.add(xu)
            out.append(xu)
    return out


def _read_html_with_headers(url: str) -> List[pd.DataFrame]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return pd.read_html(r.text)


@st.cache_data(ttl=3600, show_spinner=False)
def get_extended_tickers() -> List[str]:
    """Build ticker universe from Wikipedia + base + ADR lists."""
    sp500: List[str] = []
    nasdaq100: List[str] = []

    try:
        tables = _read_html_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            for col in ["Symbol", "Ticker symbol", "Ticker"]:
                if col in t.columns:
                    sp500 = [sanitize_ticker(x) for x in t[col].dropna().astype(str).tolist()]
                    break
            if sp500:
                break
    except Exception:
        pass

    try:
        tables = _read_html_with_headers("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            for col in ["Ticker", "Ticker symbol", "Symbol"]:
                if col in t.columns:
                    nasdaq100 = [sanitize_ticker(x) for x in t[col].dropna().astype(str).tolist()]
                    break
            if nasdaq100:
                break
    except Exception:
        pass

    all_tickers = unique_order(
        [sanitize_ticker(t) for t in BASE_TICKERS]
        + [sanitize_ticker(t) for t in ADR_TICKERS]
        + sp500
        + nasdaq100
    )
    return sorted(all_tickers)


# =============================================================================
# DATA FETCHING
# =============================================================================

def _fetch_one(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        df = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d",
            auto_adjust=True, actions=False, prepost=False, timeout=30,
        )
        if df.empty or "Close" not in df.columns:
            return None
        s = df["Close"].rename(ticker)
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices. No aggressive ffill - only within-column ffill for weekends."""
    if not tickers:
        return pd.DataFrame()

    end_inc = (pd.to_datetime(end) + pd.Timedelta(days=1)).date().isoformat()
    series_list = []
    for t in tickers:
        s = _fetch_one(t, start, end_inc)
        if s is not None:
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1).sort_index()
    # Only ffill gaps up to 5 days (weekends/holidays), not IPO gaps
    df = df.ffill(limit=5)
    return df


# =============================================================================
# PORTFOLIO MATH
# =============================================================================

def normalize_weights(weights: List[float]) -> List[float]:
    w = [max(float(v), 0.0) for v in weights]
    s = sum(w)
    if s == 0:
        return [1.0 / len(w)] * len(w)
    return [v / s for v in w]


def calculate_portfolio_rebalanced(weights: np.ndarray, price_df: pd.DataFrame) -> pd.Series:
    """Daily rebalanced portfolio: constant weights, re-normalized for missing data."""
    daily = price_df.pct_change()
    # Drop the first row (NaN from pct_change)
    daily = daily.iloc[1:]

    port_returns = []
    dates = []
    for date, row in daily.iterrows():
        valid = row.notna()
        if not valid.any():
            continue
        w_day = weights[valid.values].copy()
        w_sum = w_day.sum()
        if w_sum == 0:
            continue
        w_day = w_day / w_sum  # re-normalize for missing tickers
        ret = (row[valid] * w_day).sum()
        port_returns.append(ret)
        dates.append(date)

    return pd.Series(port_returns, index=dates, name="Portfolio")


def calculate_portfolio_buyhold(weights: np.ndarray, price_df: pd.DataFrame) -> pd.Series:
    """Buy & hold portfolio: weights drift with prices. V_t = sum(w_i * P_i,t / P_i,0)."""
    # Find first valid price for each ticker
    first_valid = price_df.apply(lambda col: col.dropna().iloc[0] if col.dropna().shape[0] > 0 else np.nan)
    valid_mask = first_valid.notna()
    if not valid_mask.any():
        return pd.Series(dtype=float)

    # Normalize prices to start at 1
    norm_prices = price_df.loc[:, valid_mask] / first_valid[valid_mask]
    w = weights[valid_mask.values]
    w = w / w.sum()

    # Portfolio value = sum(w_i * normalized_price_i)
    portfolio_value = (norm_prices * w).sum(axis=1).dropna()
    # Convert to returns
    pr = portfolio_value.pct_change().dropna()
    pr.name = "Portfolio"
    return pr, portfolio_value


def _safe_div(a, b):
    return (a / b) if (b is not None and b != 0 and np.isfinite(b)) else np.nan


def rolling_beta(series_p, series_b, window):
    df = pd.concat([series_p, series_b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    p, b = df.iloc[:, 0], df.iloc[:, 1]
    return p.rolling(window).cov(b) / b.rolling(window).var()


def calculate_portfolio_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, str]:
    out = {}
    try:
        p = portfolio_returns.dropna()
        if p.empty or len(p) < 2:
            return {"Error": "Not enough data"}

        if benchmark_returns is not None and len(benchmark_returns) > 1:
            idx = p.index.intersection(benchmark_returns.index)
            p = p.reindex(idx)
            b = benchmark_returns.reindex(idx)
        else:
            b = None

        # Risk-free daily rate (compounded)
        rf_d = (1 + risk_free_rate) ** (1 / 252) - 1

        total_return = (1 + p).prod() - 1
        out["Cumulative Return"] = f"{total_return:.2%}"

        years = max((p.index[-1] - p.index[0]).days / 365.25, 1 / 252)
        cagr = (1 + total_return) ** (1 / years) - 1
        out["CAGR"] = f"{cagr:.2%}"

        vol = p.std() * np.sqrt(252)
        out["Volatility (ann.)"] = f"{vol:.2%}"

        # Sharpe: excess = pr - rf_d
        excess = p - rf_d
        sharpe = np.sqrt(252) * _safe_div(excess.mean(), excess.std())
        out["Sharpe Ratio"] = "N/A" if np.isnan(sharpe) else f"{sharpe:.2f}"

        # Sortino: downside_dev = sqrt(mean(min(excess,0)^2)) * sqrt(252)
        downside = np.minimum(excess, 0)
        downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        sortino = _safe_div(excess.mean() * 252, downside_dev)
        out["Sortino Ratio"] = "N/A" if np.isnan(sortino) else f"{sortino:.2f}"

        cum = (1 + p).cumprod()
        peak = cum.cummax()
        dd = (cum / peak) - 1
        max_dd = dd.min()
        out["Max Drawdown"] = f"{max_dd:.2%}"

        # Calmar: CAGR / abs(max_dd)
        calmar = _safe_div(cagr, abs(max_dd))
        out["Calmar Ratio"] = "N/A" if np.isnan(calmar) else f"{calmar:.2f}"

        out["Skew"] = f"{p.skew():.2f}"
        out["Kurtosis"] = f"{p.kurtosis():.2f}"

        # Expected returns via log/geometric
        mu_log = np.log1p(p).mean()
        out["Expected Daily Return"] = f"{np.expm1(mu_log):.4%}"
        out["Expected Monthly Return"] = f"{np.expm1(21 * mu_log):.2%}"
        out["Expected Yearly Return"] = f"{np.expm1(252 * mu_log):.2%}"

        # VaR/CVaR as POSITIVE loss numbers
        var95 = -np.percentile(p, 5)
        cvar95 = -p[p <= -var95].mean() if (p <= -var95).any() else var95
        out["VaR (95%)"] = f"{var95:.2%}"
        out["CVaR (95%)"] = f"{cvar95:.2%}"

        out["Win Rate"] = f"{(p > 0).mean():.2%}"
        out["Risk-Free Rate"] = f"{risk_free_rate:.2%}"

        dd_only = dd[dd < 0]
        if not dd_only.empty:
            out["Average Drawdown"] = f"{dd_only.mean():.2%}"

        if b is not None and len(b) > 1:
            # Jensen's Alpha via OLS: y = (pr - rf_d), x = (bench - rf_d)
            y = p - rf_d
            x = b - rf_d
            x_clean = x.dropna()
            y_clean = y.reindex(x_clean.index).dropna()
            x_clean = x_clean.reindex(y_clean.index)

            if len(x_clean) > 2:
                x_arr = x_clean.values
                y_arr = y_clean.values
                X = np.column_stack([np.ones(len(x_arr)), x_arr])
                try:
                    coeffs = np.linalg.lstsq(X, y_arr, rcond=None)[0]
                    alpha_d = coeffs[0]
                    beta = coeffs[1]
                    out["Beta"] = f"{beta:.2f}"
                    out["Alpha (Jensen)"] = f"{alpha_d * 252:.2%}"
                except np.linalg.LinAlgError:
                    cov = np.cov(p, b)
                    beta = _safe_div(cov[0, 1], cov[1, 1])
                    if not np.isnan(beta):
                        out["Beta"] = f"{beta:.2f}"

            ar = p - b
            te = ar.std() * np.sqrt(252)
            ir = _safe_div(ar.mean() * 252, te)
            if not np.isnan(ir):
                out["Information Ratio"] = f"{ir:.2f}"
            corr = p.corr(b)
            if not np.isnan(corr):
                out["Correlation"] = f"{corr:.2%}"
                out["R-squared"] = f"{(corr ** 2):.2%}"

            rb6 = rolling_beta(p, b, 126).dropna()
            rb12 = rolling_beta(p, b, 252).dropna()
            if len(rb6) > 0:
                out["Avg Beta (6M)"] = f"{rb6.mean():.2f}"
            if len(rb12) > 0:
                out["Avg Beta (12M)"] = f"{rb12.mean():.2f}"

        def _max_run(mask):
            if len(mask) == 0:
                return 0
            groups = (mask != mask.shift()).cumsum()
            return int(mask.groupby(groups).cumsum()[mask].max() or 0)

        out["Max Consecutive Wins"] = _max_run(p > 0)
        out["Max Consecutive Losses"] = _max_run(p <= 0)

    except Exception as e:
        out = {"Error": f"Statistics failed: {e}"}
    return out


# =============================================================================
# SELF-TEST
# =============================================================================

def run_self_test():
    """Validate key formulas. Returns list of (test_name, passed, detail)."""
    results = []

    np.random.seed(42)
    n = 504
    ret = pd.Series(np.random.normal(0.0004, 0.01, n),
                     index=pd.bdate_range("2020-01-01", periods=n))
    bench = pd.Series(np.random.normal(0.0003, 0.012, n), index=ret.index)
    rf = 0.04

    # Build price df for buy&hold vs rebalanced comparison
    prices = pd.DataFrame({
        "A": (1 + pd.Series(np.random.normal(0.001, 0.015, n), index=ret.index)).cumprod() * 100,
        "B": (1 + pd.Series(np.random.normal(0.0005, 0.01, n), index=ret.index)).cumprod() * 50,
    })
    w = np.array([0.6, 0.4])

    # Test 1: Buy & hold vs Rebalanced should differ
    pr_reb = calculate_portfolio_rebalanced(w, prices)
    pr_bh_ret, _ = calculate_portfolio_buyhold(w, prices)
    cum_reb = (1 + pr_reb).prod() - 1
    cum_bh = (1 + pr_bh_ret).prod() - 1
    diff = abs(cum_reb - cum_bh)
    results.append(("Buy&Hold vs Rebalanced differ", diff > 1e-6,
                     f"Rebalanced={cum_reb:.4%}, B&H={cum_bh:.4%}, diff={diff:.6f}"))

    # Test 2: Sharpe uses excess returns
    rf_d = (1 + rf) ** (1 / 252) - 1
    excess = ret - rf_d
    sharpe = np.sqrt(252) * excess.mean() / excess.std()
    sharpe_raw = np.sqrt(252) * ret.mean() / ret.std()
    results.append(("Sharpe uses excess returns", abs(sharpe - sharpe_raw) > 0.01,
                     f"With excess={sharpe:.3f}, raw={sharpe_raw:.3f}"))

    # Test 3: Sortino uses downside RMS (not std of negatives)
    downside = np.minimum(excess, 0)
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
    neg_only = excess[excess < 0]
    wrong_dev = neg_only.std() * np.sqrt(252) if len(neg_only) > 0 else 0
    results.append(("Sortino uses downside RMS", abs(downside_dev - wrong_dev) > 0.001,
                     f"RMS={downside_dev:.4f}, std(neg)={wrong_dev:.4f}"))

    # Test 4: Calmar positive when CAGR positive
    total = (1 + ret).prod() - 1
    years = n / 252
    cagr = (1 + total) ** (1 / years) - 1
    cum = (1 + ret).cumprod()
    max_dd = ((cum / cum.cummax()) - 1).min()
    calmar = cagr / abs(max_dd)
    results.append(("Calmar positive when CAGR>0", (cagr > 0 and calmar > 0) or cagr <= 0,
                     f"CAGR={cagr:.4%}, MaxDD={max_dd:.4%}, Calmar={calmar:.2f}"))

    # Test 5: VaR/CVaR are positive loss numbers
    var95 = -np.percentile(ret, 5)
    results.append(("VaR is positive loss", var95 > 0,
                     f"VaR(95%)={var95:.4%}"))

    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def legend_value(value_text, color, legendgroup=None, legendrank=None):
    return go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=0, color=color),
        name=value_text, showlegend=True, hoverinfo="skip",
        legendgroup=legendgroup,
        legendrank=legendrank if legendrank is not None else 1000,
    )


def fig_cumulative(pr, br=None, name="Portfolio"):
    if isinstance(pr, pd.DataFrame):
        pr = pr.iloc[:, 0]
    traces = []
    p_cum = (1 + pr).cumprod()
    traces.append(go.Scatter(
        x=p_cum.index, y=p_cum, name=name, mode="lines",
        line=dict(width=3, color=PORT_COLOR),
    ))
    if br is not None and len(br) > 0:
        if isinstance(br, pd.DataFrame):
            br = br.iloc[:, 0]
        b_cum = (1 + br).cumprod()
        traces.append(go.Scatter(
            x=b_cum.index, y=b_cum, name="Benchmark (SPY)", mode="lines",
            line=dict(width=3, dash="dash", color=BMK_COLOR),
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{name} vs Benchmark - Cumulative (log)",
        xaxis_title="Date",
        yaxis=dict(title="Cumulative", type="log"),
        template="plotly_white",
        hovermode="x unified",
        height=500, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_underwater(pr, br=None):
    if isinstance(pr, pd.DataFrame):
        pr = pr.iloc[:, 0]
    p_cum = (1 + pr).cumprod()
    p_dd = (p_cum / p_cum.cummax()) - 1
    p_avg = p_dd[p_dd < 0].mean() if (p_dd < 0).any() else np.nan

    traces = [go.Scatter(
        x=p_dd.index, y=p_dd, name="Portfolio DD", mode="lines",
        fill="tozeroy", line=dict(color=PORT_COLOR),
    )]
    if np.isfinite(p_avg):
        traces.append(go.Scatter(
            x=[p_dd.index.min(), p_dd.index.max()], y=[p_avg, p_avg],
            mode="lines", name="Avg DD (Portfolio)",
            line=dict(color=AVG_PORT, width=2, dash="dash"),
            hoverinfo="skip", legendgroup="avg_port", legendrank=10,
        ))
        traces.append(legend_value(f"{p_avg:.2%}", AVG_PORT, legendgroup="avg_port", legendrank=11))

    if br is not None and len(br) > 0:
        if isinstance(br, pd.DataFrame):
            br = br.iloc[:, 0]
        b_cum = (1 + br).cumprod()
        b_dd = (b_cum / b_cum.cummax()) - 1
        b_avg = b_dd[b_dd < 0].mean() if (b_dd < 0).any() else np.nan
        traces.append(go.Scatter(
            x=b_dd.index, y=b_dd, name="Benchmark DD", mode="lines",
            line=dict(color=BMK_COLOR), legendrank=20,
        ))
        if np.isfinite(b_avg):
            traces.append(go.Scatter(
                x=[b_dd.index.min(), b_dd.index.max()], y=[b_avg, b_avg],
                mode="lines", name="Avg DD (Benchmark)",
                line=dict(color=AVG_BMK, width=2, dash="dash"),
                hoverinfo="skip", legendgroup="avg_bmk", legendrank=21,
            ))
            traces.append(legend_value(f"{b_avg:.2%}", AVG_BMK, legendgroup="avg_bmk", legendrank=22))

    fig = go.Figure(traces)
    fig.update_layout(
        title="Underwater (Drawdown)",
        xaxis_title="Date",
        yaxis=dict(title="Drawdown", tickformat=".0%"),
        template="plotly_white",
        hovermode="x unified",
        height=500, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_annual_returns(pr, br):
    df = pd.concat([pr, br], axis=1).dropna()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for annual returns", x=0.5, y=0.5, showarrow=False)
        return fig

    p_year = df.iloc[:, 0].resample("YE").apply(lambda x: (1 + x).prod() - 1)
    b_year = df.iloc[:, 1].resample("YE").apply(lambda x: (1 + x).prod() - 1)

    years = sorted(set(p_year.index.year.tolist()) | set(b_year.index.year.tolist()))
    cats = [str(y) for y in years]

    p_map = {d.year: v for d, v in zip(p_year.index, p_year.values)}
    b_map = {d.year: v for d, v in zip(b_year.index, b_year.values)}
    p_vals = [float(p_map.get(y, np.nan)) for y in years]
    b_vals = [float(b_map.get(y, np.nan)) for y in years]

    traces = [
        go.Bar(x=cats, y=p_vals, name="Portfolio", marker_color=PORT_COLOR, legendrank=1),
        go.Bar(x=cats, y=b_vals, name="Benchmark", marker_color=BMK_COLOR, legendrank=2),
    ]

    p_avg = np.nanmean([v for v in p_vals if np.isfinite(v)])
    b_avg = np.nanmean([v for v in b_vals if np.isfinite(v)])
    if np.isfinite(p_avg):
        traces.append(go.Scatter(
            x=cats, y=[p_avg] * len(cats), mode="lines",
            name="Avg (Portfolio)", line=dict(color=AVG_PORT, dash="dash", width=2),
            hoverinfo="skip", legendgroup="avg_port", legendrank=3,
        ))
        traces.append(legend_value(f"{p_avg:.2%}", AVG_PORT, legendgroup="avg_port", legendrank=4))
    if np.isfinite(b_avg):
        traces.append(go.Scatter(
            x=cats, y=[b_avg] * len(cats), mode="lines",
            name="Avg (Benchmark)", line=dict(color=AVG_BMK, dash="dash", width=2),
            hoverinfo="skip", legendgroup="avg_bmk", legendrank=5,
        ))
        traces.append(legend_value(f"{b_avg:.2%}", AVG_BMK, legendgroup="avg_bmk", legendrank=6))

    fig = go.Figure(traces)
    fig.update_layout(
        title="Annual Returns Comparison",
        xaxis=dict(title="Year", type="category"),
        yaxis=dict(title="Annual Return", tickformat=".0%"),
        barmode="group",
        template="plotly_white",
        height=500, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_distribution(pr):
    mean_ret = pr.mean() * 100
    fig = go.Figure([go.Histogram(x=pr * 100, nbinsx=50, opacity=0.8, name="Daily Returns")])
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="black",
                  annotation_text=f"Mean: {mean_ret:.2f}%")
    fig.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
        height=500, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_monthly_heatmap(pr):
    pr = pd.Series(pr).dropna()
    if pr.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for monthly heatmap", x=0.5, y=0.5, showarrow=False)
        return fig

    monthly = pr.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = monthly.to_frame("ret")
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    piv = df.pivot(index="Year", columns="Month", values="ret").reindex(columns=range(1, 13))

    z, text = [], []
    for _, row in piv.iterrows():
        z_row, t_row = [], []
        for val in row.values:
            if pd.isna(val):
                z_row.append(None)
                t_row.append("")
            else:
                z_row.append(float(val))
                t_row.append(f"{val:.1%}")
        z.append(z_row)
        text.append(t_row)

    finite_vals = piv.values[np.isfinite(piv.values)]
    bound = max(
        abs(np.quantile(finite_vals, 0.05)) if finite_vals.size else 0.1,
        abs(np.quantile(finite_vals, 0.95)) if finite_vals.size else 0.1,
    )
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    y_labels = [str(int(y)) for y in piv.index]

    fig = go.Figure(go.Heatmap(
        z=z, x=month_names, y=y_labels, colorscale="RdYlGn",
        zmin=-bound, zmax=bound, zmid=0,
        colorbar=dict(title="Return", tickformat=".0%"),
        text=text, texttemplate="%{text}",
        hovertemplate="Year %{y}<br>%{x}: %{z:.2%}<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly Returns Heatmap",
        xaxis_title="Month", yaxis_title="Year",
        yaxis=dict(type="category"),
        template="plotly_white",
        height=500, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def compute_worst_drawdowns(pr: pd.Series, top_n: int = 10) -> pd.DataFrame:
    pr = pr.dropna()
    if pr.empty:
        return pd.DataFrame()

    cum = (1 + pr).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    last_date = pr.index[-1]

    episodes = []
    in_dd = False
    start = None
    trough_val = 0.0
    trough_date = None

    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = date
            trough_val = val
            trough_date = date
        elif in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = date
            if val >= 0:
                episodes.append({
                    "Start": start, "Trough": trough_date, "Recovery": date,
                    "Depth": trough_val,
                    "DD Days": (trough_date - start).days,
                    "To Recovery Days": (date - trough_date).days,
                    "Total Days": (date - start).days,
                })
                in_dd = False

    if in_dd and start is not None:
        trough_date = dd.loc[start:].idxmin()
        trough_val = dd.loc[start:].min()
        episodes.append({
            "Start": start, "Trough": trough_date, "Recovery": None,
            "Depth": trough_val,
            "DD Days": (trough_date - start).days,
            "To Recovery Days": None,
            "Total Days": (last_date - start).days,
        })

    if not episodes:
        return pd.DataFrame()

    df = pd.DataFrame(episodes).sort_values("Depth").head(top_n)
    df["Start"] = pd.to_datetime(df["Start"]).dt.strftime("%Y-%m-%d")
    df["Trough"] = pd.to_datetime(df["Trough"]).dt.strftime("%Y-%m-%d")
    df["Recovery"] = df["Recovery"].apply(
        lambda x: pd.to_datetime(x).strftime("%Y-%m-%d") if pd.notna(x) else "Open"
    )
    df["Depth"] = df["Depth"].apply(lambda x: f"{x:.2%}")
    df["DD Days"] = df["DD Days"].apply(lambda x: int(x) if pd.notna(x) else "-")
    df["To Recovery Days"] = df["To Recovery Days"].apply(lambda x: int(x) if pd.notna(x) else "-")
    df["Total Days"] = df["Total Days"].apply(lambda x: int(x) if pd.notna(x) else "-")
    return df.reset_index(drop=True)


def fig_rolling_metrics(pr, br, window, rf_annual=0.0):
    if pr is None or len(pr) == 0 or br is None or len(br) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", x=0.5, y=0.5, showarrow=False)
        return fig

    idx = pr.index.intersection(br.index)
    p = pr.reindex(idx).dropna()
    b = br.reindex(idx).dropna()
    idx = p.index.intersection(b.index)
    if len(idx) < max(21, window):
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for selected window", x=0.5, y=0.5, showarrow=False)
        return fig

    p = p.reindex(idx)
    b = b.reindex(idx)

    # Rolling Sharpe with excess returns
    rf_d = (1 + rf_annual) ** (1 / 252) - 1
    excess = p - rf_d
    roll_sharpe = (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(252)
    roll_beta = p.rolling(window).cov(b) / b.rolling(window).var()

    window_years = max(1, round(window / 252))
    traces = [
        go.Scatter(x=roll_sharpe.index, y=roll_sharpe, mode="lines",
                   name=f"Rolling Sharpe ({window_years}Y)", line=dict(color=PORT_COLOR), legendrank=1),
        go.Scatter(x=roll_beta.index, y=roll_beta, mode="lines",
                   name=f"Rolling Beta ({window_years}Y)", line=dict(color=BMK_COLOR), yaxis="y2", legendrank=2),
    ]

    sh_avg = roll_sharpe.dropna().mean()
    bt_avg = roll_beta.dropna().mean()
    if np.isfinite(sh_avg):
        traces.append(go.Scatter(
            x=[roll_sharpe.index.min(), roll_sharpe.index.max()], y=[sh_avg, sh_avg],
            mode="lines", name="Avg Sharpe",
            line=dict(color=AVG_PORT, width=2, dash="dash"),
            hoverinfo="skip", legendgroup="avg_sh", legendrank=3,
        ))
        traces.append(legend_value(f"{sh_avg:.2f}", AVG_PORT, legendgroup="avg_sh", legendrank=4))
    if np.isfinite(bt_avg):
        traces.append(go.Scatter(
            x=[roll_beta.index.min(), roll_beta.index.max()], y=[bt_avg, bt_avg],
            mode="lines", name="Avg Beta", yaxis="y2",
            line=dict(color=AVG_BMK, width=2, dash="dash"),
            hoverinfo="skip", legendgroup="avg_bt", legendrank=5,
        ))
        traces.append(legend_value(f"{bt_avg:.2f}", AVG_BMK, legendgroup="avg_bt", legendrank=6))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Rolling Sharpe & Beta (window ~ {window_years} year{'s' if window_years > 1 else ''})",
        xaxis_title="Date",
        yaxis=dict(title="Sharpe"),
        yaxis2=dict(title="Beta", overlaying="y", side="right"),
        template="plotly_white",
        hovermode="x unified",
        height=500, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_corr_matrix(price_df: pd.DataFrame, horizon_days: Optional[int]):
    if price_df is None or price_df.empty or len(price_df.columns) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for correlation matrix", x=0.5, y=0.5, showarrow=False)
        return fig

    cols = list(price_df.columns)[:25]
    px = price_df[cols]
    ret = px.pct_change().dropna()
    if horizon_days is not None:
        ret = ret.iloc[-horizon_days:].copy() if len(ret) >= horizon_days else ret
    corr = ret.corr()
    tickers = corr.columns.tolist()
    z = corr.values
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=tickers, y=tickers, colorscale="RdBu", zmin=-1, zmax=1, zmid=0,
        colorbar=dict(title="Corr"),
        text=text, texttemplate="%{text}",
        hovertemplate="X %{x}<br>Y %{y}<br>Corr %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Correlation Matrix (trailing window)",
        xaxis=dict(type="category", side="bottom"),
        yaxis=dict(type="category", autorange="reversed"),
        template="plotly_white",
        height=600, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_pair_rolling_corr(price_df: pd.DataFrame, a: str, b: str, window: int):
    if price_df is None or price_df.empty or a not in price_df.columns or b not in price_df.columns or a == b:
        fig = go.Figure()
        msg = "Select two different assets" if a == b else "Select valid pair"
        fig.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False)
        return fig

    ret_a = price_df[a].pct_change().dropna()
    ret_b = price_df[b].pct_change().dropna()
    common = ret_a.index.intersection(ret_b.index)
    ret_a, ret_b = ret_a.reindex(common), ret_b.reindex(common)
    rc = ret_a.rolling(window).corr(ret_b)

    traces = [go.Scatter(
        x=rc.index, y=rc, name=f"Rolling Corr ({a},{b}) {window}d",
        mode="lines", line=dict(color=PORT_COLOR),
    )]
    avg = rc.dropna().mean()
    if np.isfinite(avg):
        traces.append(go.Scatter(
            x=[rc.index.min(), rc.index.max()], y=[avg, avg],
            mode="lines", name="Avg Corr",
            line=dict(color=AVG_PORT, dash="dash"),
            hoverinfo="skip", legendgroup="avg_corr", legendrank=10,
        ))
        traces.append(legend_value(f"{avg:.2f}", AVG_PORT, legendgroup="avg_corr", legendrank=11))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Pair Rolling Correlation: {a} vs {b}",
        xaxis_title="Date",
        yaxis=dict(title="Correlation", range=[-1, 1]),
        template="plotly_white",
        hovermode="x unified",
        height=450, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


def fig_weight_drift(portfolio_value_series: pd.Series, weights: np.ndarray, price_df: pd.DataFrame):
    """Show how weights drift over time in Buy & Hold mode."""
    first_valid = price_df.apply(lambda col: col.dropna().iloc[0] if col.dropna().shape[0] > 0 else np.nan)
    valid_mask = first_valid.notna()
    if not valid_mask.any():
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    norm_prices = price_df.loc[:, valid_mask] / first_valid[valid_mask]
    w = weights[valid_mask.values]
    w = w / w.sum()

    # Each ticker's $ value = w_i * normalized_price_i
    dollar_values = norm_prices * w
    total = dollar_values.sum(axis=1)
    weight_pcts = dollar_values.div(total, axis=0).dropna()

    fig = go.Figure()
    for col in weight_pcts.columns:
        fig.add_trace(go.Scatter(
            x=weight_pcts.index, y=weight_pcts[col],
            name=col, mode="lines", stackgroup="one",
        ))
    fig.update_layout(
        title="Weight Drift (Buy & Hold)",
        xaxis_title="Date",
        yaxis=dict(title="Weight", tickformat=".0%"),
        template="plotly_white",
        hovermode="x unified",
        height=450, margin=dict(l=60, r=60, t=70, b=60),
    )
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def process_portfolio(
    name: str,
    selected_tickers: List[str],
    manual_input: str,
    weight_values: Dict[str, float],
    start_date: str,
    end_date: str,
    rf: float,
    mode: str,
):
    """Process a single portfolio: fetch data, compute metrics, render charts."""

    # Build ticker list
    tickers = [sanitize_ticker(t) for t in (selected_tickers or [])]
    if manual_input:
        tickers += [sanitize_ticker(t) for t in manual_input.split(",") if t.strip()]
    tickers = unique_order(tickers)

    if not tickers:
        st.warning(f"{name}: No tickers selected.")
        return

    # Build weights from user input
    raw_weights = [weight_values.get(t, 1.0 / len(tickers)) for t in tickers]
    w = normalize_weights(raw_weights)

    # Fetch data
    with st.spinner(f"Fetching data for {name}..."):
        px = get_stock_data(tuple(tickers), start_date, end_date)

    if px.empty:
        st.error(f"{name}: Could not fetch data. Check tickers or connectivity.")
        return

    # Align weights to available columns
    cols = list(px.columns)
    w_aligned = []
    for c in cols:
        try:
            i = tickers.index(c)
            w_aligned.append(w[i])
        except ValueError:
            w_aligned.append(0.0)
    w_aligned = normalize_weights(w_aligned)
    w_arr = np.array(w_aligned)

    # Check for missing data and warn
    missing_pct = px.isna().mean()
    high_missing = missing_pct[missing_pct > 0.05]
    if not high_missing.empty:
        st.warning(f"{name}: Tickers with >5% missing data: {', '.join(high_missing.index.tolist())}. "
                   "Returns are computed only on available dates.")

    # Portfolio returns based on mode
    portfolio_value = None
    if mode == "Buy & Hold":
        pr, portfolio_value = calculate_portfolio_buyhold(w_arr, px)
    else:
        pr = calculate_portfolio_rebalanced(w_arr, px)

    if pr.empty or len(pr) < 2:
        st.error(f"{name}: Not enough return data after processing.")
        return

    # Benchmark (SPY)
    spy_data = get_stock_data(("SPY",), pr.index.min().date().isoformat(), pr.index.max().date().isoformat())
    br = spy_data["SPY"].pct_change().dropna() if not spy_data.empty and "SPY" in spy_data.columns else pd.Series(dtype=float)

    # Metrics
    stats = calculate_portfolio_metrics(pr, br if not br.empty else None, rf)

    # --- Tabs ---
    tab_perf, tab_hold, tab_charts = st.tabs(["Performance", "Holdings", "Charts"])

    with tab_perf:
        st.caption(f"Mode: **{mode}**")
        stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True, height=700)

    with tab_hold:
        holdings_df = pd.DataFrame({"Ticker": cols, "Weight (initial)": [f"{wt:.1%}" for wt in w_aligned]})
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        if mode == "Buy & Hold" and portfolio_value is not None:
            st.plotly_chart(fig_weight_drift(portfolio_value, w_arr, px), use_container_width=True)

    with tab_charts:
        # 1. Cumulative
        st.plotly_chart(fig_cumulative(pr, br if not br.empty else None, name), use_container_width=True)

        # 2. Underwater
        st.plotly_chart(fig_underwater(pr, br if not br.empty else None), use_container_width=True)

        # 3. Annual Returns
        if not br.empty:
            st.plotly_chart(fig_annual_returns(pr, br), use_container_width=True)

        # 4. Distribution
        st.plotly_chart(fig_distribution(pr), use_container_width=True)

        # 5. Monthly Heatmap
        st.plotly_chart(fig_monthly_heatmap(pr), use_container_width=True)

        # 6. Worst Drawdowns Table
        st.subheader("Worst 10 Drawdowns")
        dd_table = compute_worst_drawdowns(pr)
        if not dd_table.empty:
            st.dataframe(dd_table, use_container_width=True, hide_index=True)
        else:
            st.info("No drawdown episodes detected.")

        # 7. Rolling Sharpe & Beta
        st.subheader("Rolling Sharpe & Beta")
        roll_window = st.radio(
            "Rolling window:",
            options=[252, 252 * 3, 252 * 5, 252 * 10],
            format_func=lambda x: {252: "1Y", 756: "3Y", 1260: "5Y", 2520: "10Y"}[x],
            horizontal=True,
            key=f"roll_window_{name}",
        )
        st.plotly_chart(fig_rolling_metrics(pr, br, roll_window, rf), use_container_width=True)

        # 8. Correlation Matrix
        st.subheader("Correlation Matrix (trailing window)")
        corr_horizon = st.radio(
            "Horizon:",
            options=[252, 756, 1260, "ALL"],
            format_func=lambda x: {252: "1Y", 756: "3Y", 1260: "5Y", "ALL": "All"}[x],
            horizontal=True,
            key=f"corr_horizon_{name}",
        )
        horizon_val = None if corr_horizon == "ALL" else int(corr_horizon)
        st.plotly_chart(fig_corr_matrix(px, horizon_val), use_container_width=True)

        # 9. Pair Rolling Correlation
        st.subheader("Pair Rolling Correlation")
        pair_cols = list(px.columns)
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            pair_a = st.selectbox("Asset A", pair_cols, index=0, key=f"pair_a_{name}")
        with pcol2:
            pair_b = st.selectbox("Asset B", pair_cols, index=min(1, len(pair_cols) - 1), key=f"pair_b_{name}")
        with pcol3:
            pair_window = st.radio(
                "Window:",
                options=[30, 60, 126, 252],
                format_func=lambda x: {30: "30d", 60: "60d", 126: "126d (~6M)", 252: "252d (~1Y)"}[x],
                horizontal=True,
                key=f"pair_win_{name}",
            )
        st.plotly_chart(fig_pair_rolling_corr(px, pair_a, pair_b, pair_window), use_container_width=True)


def main():
    st.title("Portfolio Analytics Dashboard")
    st.markdown("**Compare two portfolios side-by-side: cumulative returns, drawdowns, Sharpe, Beta, correlations, and more.**")

    # Build ticker universe
    with st.spinner("Loading ticker universe..."):
        all_tickers = get_extended_tickers()

    # --- Sidebar ---
    st.sidebar.header("Global Settings")

    import datetime as dt

    start_date = st.sidebar.date_input(
        "Start date",
        value=dt.date.today() - dt.timedelta(days=365 * 3),
    )
    end_date = st.sidebar.date_input("End date", value=dt.date.today())
    rf_pct = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    rf = rf_pct / 100.0

    # Portfolio Mode toggle
    port_mode = st.sidebar.radio(
        "Portfolio Mode",
        options=["Daily Rebalanced", "Buy & Hold"],
        index=0,
        help="Daily Rebalanced: constant weights every day. Buy & Hold: initial weights drift with prices.",
    )

    st.sidebar.markdown("---")

    # --- Portfolio 1 Settings ---
    st.sidebar.header("Portfolio 1")
    sel1 = st.sidebar.multiselect(
        "Select tickers (P1)",
        options=all_tickers,
        default=[t for t in ["AAPL", "MSFT"] if t in all_tickers],
        key="sel1",
    )
    man1 = st.sidebar.text_input("Additional tickers (comma-separated, P1)", key="man1")

    # Build ticker list for weight inputs
    tickers1 = unique_order([sanitize_ticker(t) for t in sel1] + [sanitize_ticker(t) for t in man1.split(",") if t.strip()])

    weights1: Dict[str, float] = {}
    if tickers1:
        st.sidebar.caption("Weights (P1) - auto-normalized")
        eq = round(1.0 / len(tickers1), 3)
        for t in tickers1:
            weights1[t] = st.sidebar.number_input(f"{t}", min_value=0.0, max_value=1.0, value=eq, step=0.001, key=f"w1_{t}")

    st.sidebar.markdown("---")

    # --- Portfolio 2 Settings ---
    st.sidebar.header("Portfolio 2")
    sel2 = st.sidebar.multiselect(
        "Select tickers (P2)",
        options=all_tickers,
        default=[t for t in ["NVDA", "GOOGL"] if t in all_tickers],
        key="sel2",
    )
    man2 = st.sidebar.text_input("Additional tickers (comma-separated, P2)", key="man2")

    tickers2 = unique_order([sanitize_ticker(t) for t in sel2] + [sanitize_ticker(t) for t in man2.split(",") if t.strip()])

    weights2: Dict[str, float] = {}
    if tickers2:
        st.sidebar.caption("Weights (P2) - auto-normalized")
        eq = round(1.0 / len(tickers2), 3)
        for t in tickers2:
            weights2[t] = st.sidebar.number_input(f"{t}", min_value=0.0, max_value=1.0, value=eq, step=0.001, key=f"w2_{t}")

    # --- Analyze Button ---
    st.sidebar.markdown("---")
    analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

    # Self-test expander
    with st.sidebar.expander("Self-Test"):
        if st.button("Run Self-Test", key="selftest"):
            results = run_self_test()
            for test_name, passed, detail in results:
                icon = "PASS" if passed else "FAIL"
                st.text(f"[{icon}] {test_name}")
                st.caption(detail)

    if not analyze:
        st.info("Configure portfolios in the sidebar and click Analyze.")
        return

    # --- Results ---
    col1, col2 = st.columns(2)

    with col1:
        st.header("Portfolio 1")
        process_portfolio(
            "Portfolio 1", sel1, man1, weights1,
            start_date.isoformat(), end_date.isoformat(), rf, port_mode,
        )

    with col2:
        st.header("Portfolio 2")
        process_portfolio(
            "Portfolio 2", sel2, man2, weights2,
            start_date.isoformat(), end_date.isoformat(), rf, port_mode,
        )


if __name__ == "__main__":
    main()
