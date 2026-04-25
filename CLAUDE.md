# CLAUDE.md — Claude's reference for this repo

This file exists so I can orient myself quickly after a context reset. Keep it
tight; point to sources of truth rather than duplicating them.

## Project

IMC Prosperity Round 3 trading dashboard (Dash/Plotly). Visualizes order book,
trades, PnL, positions, and — new in Round 3 — options analytics (overlay,
volatility smile, IV time series). The user uses this to inspect both live log
files (`logs/*.log`) and historical CSVs (`historical/ROUND_*/`).

The round trades three asset classes:
- `HYDROGEL_PACK` (delta-1, position limit 200)
- `VELVETFRUIT_EXTRACT` (delta-1, position limit 200) — the options underlying
- 10 vouchers `VEV_{K}` where `K ∈ {4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500}` — European calls, position limit 300 each

See `docs/ROUND_3.md` for details on how the round works

See `docs/round3_options_primer.md` for theory and strategy menu.


## Data conventions

- Timestamps per day: `0, 100, 200, …, 999_900` (step 100; 10_000 ticks/day).
- `TIMESTAMP_PER_DAY = 1_000_000` in `utils/options.py` — use this to convert
  a timestamp to fractional days: `t / TIMESTAMP_PER_DAY`.
- Time to expiry (TTE) convention for `ROUND_3` historical folder:
  - day 0 → TTE = 8 days (tutorial round)
  - day 1 → TTE = 7 days (round 1)
  - day 2 → TTE = 6 days (round 2)
  - live round 3 starts at **TTE = 5 days** (`opts.LIVE_TTE_AT_START`).
- Units: T in days, σ per √day, r = 0. So a σ of 0.013 ≈ 21% annualized.
- Historical ROUND_3 smile sits around **σ ≈ 0.012–0.013 per √day** across strikes 5000–5500 (very flat); deep OTM (6000/6500) pin at the 0.5 price floor and give no IV signal.

## Where things live

```
app.py                          # Dash entry: tabs + sub-tabs wiring
CLAUDE.md                       # this file
README.md                       # user-facing intro
docs/
  round2_speed_profit.md
  round3_options_primer.md      # BS, greeks, IV, strategy menu
data/
  loader.py / store.py          # live log parsing (from logs/*.log)
  historical_loader.py / historical_store.py   # CSV parsing for historical/
components/
  main_chart.py, volume_chart.py, order_chart.py, pnl_chart.py,
  position_chart.py, log_viewer.py                        # live Trading
  historical_chart.py, historical_volume.py, …            # historical Trading
  options_core.py               # store-agnostic options figure builders
  options_live.py               # Live Options sub-tab
  options_historical.py         # Historical Options sub-tab
utils/
  position.py
  options.py                    # BS, IV solver, greeks, strike constants
tools/
  probe.py                      # CLI to inspect the same data the dashboard shows
historical/
  ROUND_3/prices_round_3_day_{0,1,2}.csv
  ROUND_3/trades_round_3_day_{0,1,2}.csv
logs/*.log                      # live submission logs (JSON)
```

## Tools

One CLI, `tools/probe.py`, with subcommands. Run from the repo root.
**Do not name any tool `inspect.py`** — it shadows Python's stdlib `inspect`
module when numpy is imported from `tools/`.

All commands default to `--round ROUND_3`. Most require `--day {0,1,2}`.
Most support `--json` for structured output.

### `summary` — per-product mid stats
```bash
python3 tools/probe.py summary --day 2
python3 tools/probe.py summary                  # all days
```
Fields: n, min, max, mean, std.

### `price` — price window / stats for one product
```bash
python3 tools/probe.py price --day 2 --product VEV_5200 --range 500000:500500
python3 tools/probe.py price --day 2 --product VEV_5200 --stats
python3 tools/probe.py price --day 2 --product VEV_5200 --sample 20
```
Prints `timestamp | mid | bid_1 | ask_1 | vol_bid_1 | vol_ask_1`. `--stats`
prints n/min/max/mean/std instead.

### `book` — order book snapshot
```bash
python3 tools/probe.py book --day 2 --product VEV_5200 --ts 500000
```
Snaps to the nearest timestamp; prints all three levels on each side.

### `trades` — trade log filter
```bash
python3 tools/probe.py trades --day 2 --product VEV_5200 --range 0:1000000 --min-qty 5
```

### `smile` — IV across all strikes at one timestamp
```bash
python3 tools/probe.py smile --day 2 --ts 500000
python3 tools/probe.py smile --day 2 --ts 500000 --tte 6.0   # override TTE
```
Columns: K, C, intrinsic, extrinsic, log-moneyness m, IV. Prints S and T. NaN
IVs mean the voucher price is at/below intrinsic or stuck at the 0.5 floor
(unsolvable).

### `iv` — IV distribution per strike across a full day
```bash
python3 tools/probe.py iv --day 2 --strikes 5100 5200 5300
python3 tools/probe.py iv --day 2                           # all strikes
```
Columns: strike, n, min, p5, median, mean, p95, max, std — useful for judging
whether IV is stable or moves.

### `arb` — find no-arb bound violations
```bash
python3 tools/probe.py arb --day 2                          # default tol=0.5
python3 tools/probe.py arb --day 2 --tol 0.0 --strikes 5200 5300
```
Flags voucher `ask < max(S_bid − K, 0)` or `bid > S_ask`. These are free-money
opportunities when they show up.

### `greeks` — Δ/Γ/Θ/ν at a timestamp
```bash
python3 tools/probe.py greeks --day 2 --ts 500000 --strikes 5100 5200 5300
```
Uses market IV per strike; falls back to σ=0.015 if IV is unsolvable (marked
in the IV column as `"0.015 (fallback)"`).

## Common workflows

**User says "something weird around t=X for VEV_K on day D":**
1. `probe.py book --day D --product VEV_K --ts X` — see the book
2. `probe.py price --day D --product VEV_K --range (X-5000):(X+5000)` — nearby prices
3. `probe.py price --day D --product VELVETFRUIT_EXTRACT --range … --stats` — compare underlying
4. `probe.py smile --day D --ts X` — full cross-section snapshot

**User asks about vol regime:**
- `probe.py iv --day D` to see per-strike distribution
- Flat distribution (low std) → market thinks vol is known
- Wide distribution → IV mean-reversion candidate

**User asks about arb:**
- `probe.py arb --day D --tol 0.0` for the strictest check
- Deep ITM strikes (4000, 4500) most likely to throw stale-book violations

## The options math (from `utils/options.py`)

- Normal CDF: A&S 26.2.17 polynomial approximation (no scipy dep).
- IV solver: vectorized Newton-Raphson, rejects non-converged points as NaN
  (used to silently return 0.0 for deep-OTM floors — the post-convergence
  residual check catches that now).
- Greeks: closed-form with r=0.
- All prices quoted on a tick grid of 0.5 for deep OTM (the 0.5 floor).

## Gotchas I've hit

- **Filename shadowing:** never name a script under `tools/` `inspect.py`,
  `logging.py`, `types.py`, etc. Python auto-prepends the script's directory
  to `sys.path`, so stdlib imports get shadowed. Present name `probe.py`.
- **Plotly array encoding:** Plotly JSON may encode numeric arrays as
  `{"dtype": "...", "bdata": "..."}`. `len(trace["y"])` returns 2 (dict keys)
  not the true length — always inspect figures by rendering, not by counting
  JSON elements.
- **Realized σ > IV:** tick-level realized vol on `VELVETFRUIT_EXTRACT` comes
  out ~0.022 per √day vs implied ~0.012. The gap is microstructure noise
  (bid-ask bounce), not a real vol-risk premium. Bucket the underlying before
  computing realized vol if this matters.
- **Dash IDs must be globally unique.** Sub-tabs within a top-level tab all
  render into the DOM at once. When reusing an ID (like `hist-day-selector`)
  across sub-tabs, it works — but duplicating the same ID in two sub-tabs breaks.
