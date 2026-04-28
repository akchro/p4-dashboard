# Round 3 — voucher bid-capture strategy (replaces voucher_pure_mm)

## Why pure MM is dead

`voucher_pure_mm.py` v3 treated VEV_5000–5500 as a normal MM book: quote bid
and ask at the touch, capture spread, recycle inventory. The historical data
proves this doesn't work:

- **100% of OTM voucher trades print at the bid.** Aggressive flow is purely
  one-direction (the OTM blanket-seller bot dumping). Asks never get lifted.
- **No round-trip is possible.** Every fill on our bid accumulates long
  inventory with no order-book exit.
- Without an exit, position limit (300/strike) caps in ~1 day and we brick.

## The new model: bid capture + EOD liquidation + delta hedge

Three insights together change the framing entirely:

1. **The seller's flow is uninformed and structural.** ~95 sweeps/day, fires
   blanket-style across 5300/5400/5500/6000/6500 with same qty per strike.
   Will keep firing as long as it's coded into the simulator.
2. **EOD liquidation at fair mid is the exit.** No need to recycle inventory
   on the order book — the engine marks-out positions at end-of-day. Per
   `prosperity_pnl_floor_settle.md` (reverse-engineered from log 403865):
   - **Normal vouchers (5300/5400/5500)**: mark-to-mid → buying at bid
     captures **half-spread**.
   - **Floor-pinned (6000/6500)**: mark-to-bid (0) → no edge from being
     long; skip.
3. **No bot in the population delta-hedges.** Confirmed: VFE trades within
   ±200 ticks of an OTM sweep at 20.1% (vs 22.4% random baseline). Hedging
   is uncontested.

The strategy reduces to:

> Sit on the touch bid for VEV_5300/5400/5500. Get filled by the seller bot.
> Hedge the accumulated delta in VFE. Let EOD liquidate the positions at mid
> and realize the half-spread. Skip VEV_6000/6500. Keep VEV_4000 on its own
> two-way pennying logic.

## Per-strike edge

| strike | mid (avg) | bid (typical) | half-spread | sweeps/day | avg qty | max contracts/day | max edge/day (ticks) |
|------:|----------:|--------------:|------------:|-----------:|--------:|------------------:|---------------------:|
| 5300 | 46.8 | mid − ~1.0 | ~1.0 | 40 | 3.5 | 140 | ~140 |
| 5400 | 16.0 | mid − ~0.7 | ~0.7 | 75 | 3.5 | 263 | ~184 |
| 5500 | 6.6 | mid − ~0.6 | ~0.6 | 89 | 3.5 | 312 (capped at 300) | ~180 |
| 6000 | 0.5 | 0 | mark-to-bid → 0 edge | 95 | 3.5 | — | 0 (skip) |
| 6500 | 0.5 | 0 | mark-to-bid → 0 edge | 95 | 3.5 | — | 0 (skip) |

Total upper-bound capture across 5300/5400/5500: **~500 ticks/day** if we win
every fill on the touch.

Caveat: "max contracts/day" assumes we are alone on the bid AND the
position limit doesn't bind sooner. Currently a different bot is holding
the touch — to capture the flow we have to outbid them by joining the touch
with priority (FIFO) or pennying inside if the spread allows.

## Strategy components

### 1. Bid placement
- For each of {VEV_5300, VEV_5400, VEV_5500}: post a bid at `bid_1` (the
  touch). Don't penny inside — the simulator's matching only fires at the
  external touch on tight books.
- Size = `min(remaining_position_limit, max_per_quote)`. Choose
  `max_per_quote` to give multiple chances per sweep (seller's qty is 2-5,
  so 5 is a natural cap).
- No ask quote. The market has no aggressive buyers to lift it. (Optional:
  post an ask at far-OTM or `mid + N` as a "if regime changes" safety —
  costs nothing if it never fills.)

### 2. Position limit management
- Hard cap: 300 long per voucher.
- When current_long >= 295, pull the bid (stop accumulating).
- After EOD liquidation, position resets and we start fresh next day.

### 3. Delta hedge in VFE
- Every fill on a voucher adds Δ to total long delta on S.
- For each voucher, compute Δ = N(d1) using current S, K, T, σ. Store per
  position.
- Total target VFE position = −Σ(qty_i × Δ_i)
- Re-hedge every N ticks (e.g. every 100 ticks or after every voucher fill).
- VFE position limit: 200. Bound voucher accumulation so total |Δ| stays
  within hedgeable range.

Sizing example: if all three OTM strikes fill near limit (300 each):
- VEV_5300 Δ ≈ 0.30, contributes 90 long delta
- VEV_5400 Δ ≈ 0.16, contributes 48 long delta
- VEV_5500 Δ ≈ 0.06, contributes 18 long delta
- Total long delta ≈ 156 → hedge by short 156 VFE (within 200 limit, OK)

### 4. EOD liquidation
- Engine marks all open positions to fair mid at end-of-day (per
  `prosperity_pnl_floor_settle.md`).
- For 5300/5400/5500: this realizes the half-spread on every contract held.
- For VEV_6000/6500: would realize 0 on long positions (mark-to-bid). Don't
  hold these.
- For VFE hedge: marks at VFE mid. Hedge cost is the spread on entries plus
  any drift between entry and EOD mid.

## Risks

1. **Mid drift during the day.** Edge = mid_eod − bid_t1, not mid_t1 − bid_t1.
   If S falls, OTM mids fall, and our captured edge shrinks (or goes
   negative). Delta hedge mostly handles this.

2. **Vega exposure.** If implied vol drops between fill and EOD, OTM mids
   drop. We'd need to vega-hedge with another option to neutralize, but no
   asset in the round provides clean vega exposure (all OTM vouchers move
   together). Accept this as residual risk.

3. **Delta-hedge slippage.** VFE has its own bid-ask spread. Each hedge
   trade pays half-spread. Don't over-hedge (small Δ shifts don't merit
   re-hedging — set a deadband, e.g. only re-hedge when |Δ_drift| > 5).

4. **Competition for the bid.** The current passive bidder is *some other
   organizer bot*. To displace it we either:
   - Match their bid (FIFO determines who fills first — depends on order
     submission timing)
   - Penny inside (works only on wide-spread strikes; tight ones won't
     respond per simulator matching rules)
   - In live, we may only catch a fraction of theoretical max.

5. **Floor-pinning interpretation.** If "fair mid" liquidation is universal
   (memory's empirical floor-pinned exception is wrong), then VEV_6000/6500
   become the highest-edge strikes (0 → 0.5 = guaranteed +0.5/contract).
   Worth re-validating with a fresh probe submission.

## Implementation sketch

```python
# Per-tick loop (rough pseudocode)

OTM_STRIKES = [5300, 5400, 5500]  # 6000/6500 skipped per memory
MAX_VOUCHER_POS = 295  # leave 5 headroom on 300 limit
VFE_LIMIT = 195
HEDGE_DEADBAND = 5  # don't re-hedge < 5 deltas

def on_tick(state):
    S = mid(state, "VELVETFRUIT_EXTRACT")
    T = days_to_expiry(state.timestamp)
    sigma = SIGMA_EST  # or compute rolling

    # 1. Quote bids on OTM vouchers
    orders = []
    for K in OTM_STRIKES:
        sym = f"VEV_{K}"
        pos = state.position.get(sym, 0)
        if pos >= MAX_VOUCHER_POS:
            continue  # at limit, pull bid
        bid_px = state.order_depth[sym].best_bid
        size = min(MAX_VOUCHER_POS - pos, 5)  # match seller's max sweep
        orders.append(Order(sym, bid_px, size))

    # 2. Compute target delta hedge
    total_delta = 0.0
    for K in OTM_STRIKES:
        sym = f"VEV_{K}"
        pos = state.position.get(sym, 0)
        if pos == 0: continue
        delta = bs_delta(S, K, T, sigma)
        total_delta += pos * delta

    # 3. Hedge in VFE
    vfe_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
    target_vfe = -round(total_delta)
    target_vfe = max(-VFE_LIMIT, min(VFE_LIMIT, target_vfe))
    drift = target_vfe - vfe_pos
    if abs(drift) >= HEDGE_DEADBAND:
        if drift > 0:
            ask_px = state.order_depth["VELVETFRUIT_EXTRACT"].best_ask
            orders.append(Order("VELVETFRUIT_EXTRACT", ask_px, drift))
        else:
            bid_px = state.order_depth["VELVETFRUIT_EXTRACT"].best_bid
            orders.append(Order("VELVETFRUIT_EXTRACT", bid_px, drift))

    return orders
```

## Open questions / next steps

1. **Validate floor-pinned liquidation behavior** — rerun a probing
   submission to confirm whether memory's "mark-to-bid for VEV_6000/6500"
   still holds, or if it's actually mark-to-mid (which would unlock
   ~+150 ticks/day per pinned strike).
2. **Estimate fill rate vs. competing bidder** — backtest hypothetical
   bid-quote strategy against historical book + sweep timing to see what
   fraction of the seller's ~95 sweeps/day we'd actually catch.
3. **VEV_4000 strategy stays separate** — keep the wide-spread pennying-MM
   logic from `voucher_pure_mm.py` v3's `trade_deep_voucher`. That bot is
   different (two-way flow, qty 1-3) and can be MM'd normally.
4. **Hedge the hedge-slippage** — model VFE bid-ask cost into the per-fill
   edge to find true profitable threshold. If half-spread on VEV_5500 is
   0.6 but VFE hedge eats 0.3, real edge is 0.3 not 0.6.
