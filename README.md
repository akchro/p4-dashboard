# P4 Trading Dashboard

A Dash (Plotly) dashboard for analyzing IMC Prosperity-style algorithmic trading log files. Visualizes order book snapshots, trade executions, PnL, net position, and algorithm debug logs in a single interactive view.



## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
```

Open http://localhost:8050, select a log file from the dropdown, and choose a product.

## Log File Format

Place `.log` files in the `logs/` directory. Each file is a JSON object with:

| Key | Description |
|---|---|
| `activitiesLog` | Semicolon-delimited CSV of order book snapshots (bid/ask levels 1-3, mid price, PnL) |
| `tradeHistory` | List of trade objects with timestamp, buyer, seller, symbol, price, quantity |
| `logs` | List of per-timestamp debug output (lambdaLog, sandboxLog) |
| `submissionId` | Unique identifier for the submission |

## Dashboard Sections

| # | Section | Description |
|---|---|---|
| 1 | **Main Chart** | Order book levels (bid 1-3 in blue, ask 1-3 in red), mid price, and trade markers (green/red/gray) |
| 2 | **PnL Chart** | Profit and loss over time |
| 3 | **Position Chart** | Cumulative net position derived from trades |
| 4 | **Log Viewer** | Lambda/sandbox log output, synced to the hovered timestamp on the main chart |
| 5 | **Controls** | File selector, product selector, day selector |
| 6 | **Trade Filters** | Toggle individual bid/ask levels, show/hide trades, filter by quantity range |
| 7 | **Performance** | Downsample slider (1-20x) for faster rendering on large datasets |

## Project Structure

```
app.py                          # Entry point: layout, callback registration
data/
  loader.py                     # JSON log parsing -> DataFrames
  store.py                      # Module-level data singleton with accessor functions
components/
  main_chart.py                 # Order book + trade markers chart
  pnl_chart.py                  # PnL line chart
  position_chart.py             # Net position line chart
  log_viewer.py                 # Hover-synced log display
  controls.py                   # File/product/day selectors
  trade_filters.py              # Level toggles, trade toggle, qty filter
  performance_controls.py       # Downsample slider
utils/
  position.py                   # Cumulative position calculation from trades
```

## Dependencies

- dash >= 2.14.0
- plotly >= 5.18.0
- pandas >= 2.0.0
