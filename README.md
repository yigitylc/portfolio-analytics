# Advanced Portfolio Analytics Dashboard - Streamlit App

Two-portfolio comparison tool with comprehensive analytics: cumulative returns, drawdowns, Sharpe, Beta, correlations, and more.

## Features

- **Two-Portfolio Side-by-Side Comparison**
- **Large Ticker Universe**: S&P 500 + Nasdaq-100 + ETFs + ADRs
- **Custom Weights**: Per-ticker weight inputs, auto-normalized
- **SPY Benchmark**: All metrics computed relative to SPY

### Charts and Analytics (per portfolio)
1. Cumulative Return (log scale) vs Benchmark
2. Underwater (Drawdown) with average DD lines
3. Annual Returns Comparison (grouped bar)
4. Distribution of Daily Returns (histogram)
5. Monthly Returns Heatmap (Year x Month)
6. Worst 10 Drawdowns Table
7. Rolling Sharpe and Beta (adjustable window: 1Y/3Y/5Y/10Y)
8. Correlation Matrix Heatmap (trailing: 1Y/3Y/5Y/All)
9. Pair Rolling Correlation (selectable pair and window)

### Metrics Table
- Cumulative Return, CAGR, Volatility
- Sharpe, Sortino, Calmar Ratios
- Max Drawdown, Average Drawdown
- VaR (95%), CVaR (95%)
- Beta, Alpha, Information Ratio, Correlation, R-squared
- Rolling Beta averages (6M, 12M)
- Win Rate, Max Consecutive Wins/Losses
- Skew, Kurtosis

- **Ticker Universe**: Wikipedia (S&P 500, Nasdaq-100) with hardcoded fallbacks
- **Market Data**: yfinance (Yahoo Finance, adjusted prices via auto_adjust=True)
