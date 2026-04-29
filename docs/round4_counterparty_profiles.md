# Round 4 — Counterparty (Mark) profiles

Round 4 exposes counterparty IDs in the trade log. There are seven non-player
bots — `Mark 01, 14, 22, 38, 49, 55, 67`. This document classifies each one
based on its behavior across all three historical days
(`historical/ROUND_4/trades_round_4_day_{1,2,3}.csv`,
`historical/ROUND_4/prices_round_4_day_{1,2,3}.csv`).

The reproduction scripts are `tools/counterparty_profile.py` and
`tools/counterparty_extra.py`.

## Methodology

For every trade I look up the prevailing top-of-book (`bid_1`, `ask_1`,
`mid`) and classify the trade location:

- `price >= ask_1` → **buyer was aggressor** (lifted the offer)
- `price <= bid_1` → **seller was aggressor** (hit the bid)
- otherwise → trade printed inside the spread (rare; ~0% in this data)

A trader's **aggressor rate** is the share of their trades where they crossed
the spread. **Passive rate** is its complement — someone else crossed and
filled their resting quote. From there I add: trade size distribution, product
mix, buy/sell skew, end-of-day inventory drift, and which other Marks they
trade against.

Round numbers below aggregate all three days unless noted.

## Headline classification

| Mark    | Trades | Aggr% | Pass% | Buy% | Net qty   | Primary products                       | Role                                      |
|---------|-------:|------:|------:|-----:|----------:|----------------------------------------|-------------------------------------------|
| Mark 14 |  2172 |    0% |  100% |  52% |     +302  | HYDROGEL, UNDERLYING, VEV_4000          | **True market maker** (two-sided, flat)   |
| Mark 01 |  1843 |    0% |  100% |  87% |   +4,678  | UNDERLYING + high-strike vouchers       | Passive directional **buyer** (vouchers)  |
| Mark 22 |  1584 |   91% |    9% |   3% |   −5,477  | High-strike vouchers                    | Aggressive directional **seller** (vouchers) |
| Mark 38 |  1478 |  100% |    0% |  50% |      −14  | HYDROGEL, VEV_4000                      | Aggressive **liquidity taker** (flat)     |
| Mark 55 |  1198 |  100% |    0% |  50% |      −43  | UNDERLYING only                         | Aggressive **liquidity taker** (flat)     |
| Mark 67 |   165 |   99% |    1% | 100% |   +1,510  | UNDERLYING only                         | Aggressive directional **buyer** (underlying) |
| Mark 49 |   122 |    2% |   98% |  14% |     −956  | UNDERLYING only                         | Passive directional **seller** (underlying) |

Behavior is essentially identical across day 1, 2, and 3 — every Mark shows
the same role each day (see "Per-day stability" at the end).

---

## Per-Mark profile

### Mark 14 — *Market Maker* (canonical)

- **Aggressor rate: 0% across all 3 days.** Every Mark 14 trade is a resting
  quote being filled. As buyer, the print is at-bid (100%); as seller it is
  at-ask (100%).
- **Buy/sell symmetric** (52%/48% on both UNDERLYING and HYDROGEL_PACK).
- **Inventory always reverts toward zero**: HYDROGEL_PACK pos range
  [−148, +57], UNDERLYING [−30, +137], end-of-window net qty +302 across
  ~8.7k units traded.
- **Trade sizes small and stable** (median 4, max 8).
- **Product mix:** HYDROGEL_PACK (1003), VELVETFRUIT_EXTRACT (647), VEV_4000
  (439), plus tiny passive long-only fills on VEV_5200–5500 (33/30/13/7 trades).

Mark 14 is the textbook MM in the simulation: tight, two-sided, never crosses,
goes home flat. They're the main provider of liquidity in HYDROGEL_PACK,
UNDERLYING and VEV_4000.

### Mark 01 — *Passive directional voucher buyer*

- **Aggressor rate: 0%** — also 100% passive, but the resemblance to a market
  maker stops there.
- **Buys vouchers one-way only.** On every voucher in their book (5200, 5300,
  5400, 5500, 6000, 6500), buy share = **100%**: they post bids and never lift
  inventory back out. End-of-window long positions:
  - VEV_6000: **+1,105**
  - VEV_6500: **+1,105**
  - VEV_5500: +1,042
  - VEV_5400: +911
  - VEV_5300: +439
- **On UNDERLYING only**, they look more MM-like (52% buy, end pos +42).
- **Counterparty:** **Mark 22 supplies ~95% of the voucher flow** (1339 of
  1599 buy-side trades). For VEV_6000 / VEV_6500 it's literally 317-of-317 vs.
  Mark 22.
- Average size 4, max 8.

Net: Mark 01 is not a market maker on vouchers — they are a structural,
passive bidder that accumulates a long lottery-ticket book. Per memory note,
floor-pinned 6000/6500 settle at bid=0, so this passive-buy book is largely a
nothing-burger on the deep-OTM strikes (Mark 22 sells those at 0 too, so no
premium changes hands). Mark 01's economic exposure is in **5300/5400/5500**,
where Mark 22 is selling them real premium.

### Mark 22 — *Aggressive directional voucher seller* (mirror of Mark 01)

- **Aggressor rate: 91% (day 1) → 93% (day 3).** The seller side is
  overwhelmingly aggressive — Mark 22 hits the bid 93% of the time.
- **Sell-only on vouchers** (sell share 97–100% across VEV_5200–6500).
- **Inventory mirrors Mark 01 almost exactly**:
  - VEV_6000: **−1,105** (Mark 01 +1,105)
  - VEV_6500: **−1,105** (Mark 01 +1,105)
  - VEV_5500: −1,069 (Mark 01 +1,042)
  - VEV_5400: −959 (Mark 01 +911)
- **Also a passive seller of UNDERLYING:** 126 trades on VELVETFRUIT_EXTRACT,
  end pos −551. Here aggression rate drops to 13%, meaning on the underlying
  Mark 22 sits on the offer like Mark 49 does.
- Average size 3.7, max 10.

Mark 22 is the systematic counterparty to Mark 01 in vouchers. They never
provide liquidity in vouchers — they consume Mark 01's bids. The mirror is
near-perfect, so for modelling purposes "Mark 01 + Mark 22" can be treated as
one structural pair.

### Mark 38 — *Aggressive liquidity taker (flat)*

- **Aggressor rate: 100%** across every day, every product they touch.
- **Buy/sell balanced** (50/50), so they're not directional — they cross both
  sides of the book.
- **Counterparty: ~98% Mark 14.** They lift Mark 14's offers and hit Mark 14's
  bids on HYDROGEL_PACK (1003 of 1022 product-trades) and on VEV_4000 (439
  of 442). They are essentially Mark 14's exclusive counterparty.
- **Position always reverts** (HYDROGEL pos range [−55, +139], end +34).
- Average size 3.4, max 6 — *smaller than Mark 14's max*, so Mark 38 takes in
  small clips, not large blocks.

Mark 38 is "the Mark 14 dance partner" — a flow-trader that pays the spread
to Mark 14 in HYDROGEL_PACK and VEV_4000. Whether it's a stat-arb bot or a
hedger generating noise, behaviorally it's a taker that never has a strong
inventory.

### Mark 55 — *Aggressive liquidity taker (UNDERLYING only)*

- **Aggressor rate: 100%.** Trades only VELVETFRUIT_EXTRACT, balanced 50/50
  buy/sell.
- **Inventory churns:** day-end positions −87, +91, −47.
- Largest typical clip of any active MM-counterparty bot — average size 5.5,
  max 8.
- **Counterparties:** Mark 14 (~647 trades) and Mark 01 (~504 trades),
  i.e. it lifts/hits whichever passive Mark is at the inside on the
  underlying.

Looks like an HFT-style two-sided taker on the underlying. Whatever signal
it's chasing, it pays spread to Mark 14 and Mark 01.

### Mark 67 — *Aggressive directional buyer (UNDERLYING only)*

- **Aggressor rate: 99%, buy share 100%.** Every trade is Mark 67 lifting an
  offer for VELVETFRUIT_EXTRACT.
- **Accumulates aggressively:** end-of-day positions +519, +567, +424.
- **Largest size of any bot:** mean 9.2, p95 14, max 15.
- **Counterparty mix:** lifts Mark 49 (89 trades), Mark 22 (75 trades),
  Mark 55 (1). Notably *not* Mark 14 — Mark 67's clips are bigger than what
  Mark 14 has resting at the inside, so it gets filled deeper down the book by
  Mark 49/22.

This is a one-way buyer of the underlying. Probably represents structural
demand the simulator wants present in the market.

### Mark 49 — *Passive directional seller (UNDERLYING only)*

- **Aggressor rate: 2%, sell share 86%.** Trades only VELVETFRUIT_EXTRACT.
- **Always at the offer when selling** (99% at-ask), and at the bid the few
  times they buy — so they're a quote-only bot like Mark 01/14, just
  one-sided.
- **Builds a heavy short:** end pos −304 / −360 / −292. They never cover.
- **Large size:** mean 9.7, max 15. Same size profile as Mark 67.
- **Counterparty:** primarily Mark 67 (89 of 105 sells). Mark 67 lifts Mark 49.

Mark 49 is the structural seller of the underlying — they post offers in
size, get lifted by Mark 67, and don't unwind. Together Mark 49 ↔ Mark 67
look like a coordinated pair on the underlying analogous to Mark 01 ↔ Mark 22
on vouchers.

---

## Pair structures

Three near-exclusive bot-vs-bot pairs emerge:

| Provider (passive)        | Taker (aggressive)        | Products                        | Volume | Pattern |
|---------------------------|---------------------------|---------------------------------|-------:|---------|
| Mark 01 (bids only)       | Mark 22 (sells only)      | VEV_5200–6500                   |  1,339 | Mark 01 accumulates long, Mark 22 accumulates short — perfect mirror |
| Mark 14 (two-sided MM)    | Mark 38 (two-sided taker) | HYDROGEL_PACK, VEV_4000         |  1,442 | Mark 38 pays the spread; both end flat |
| Mark 49 (asks only, big)  | Mark 67 (lifts only, big) | VELVETFRUIT_EXTRACT             |     89 | Mark 49 accumulates short, Mark 67 accumulates long |

Mark 55 is the one trader that doesn't have a single dedicated dance partner —
it sweeps both Mark 14 and Mark 01 on the underlying.

The voucher and underlying pairs are interesting because the inventory drift
is **unbounded** (Mark 01 ends day 3 at +1,105 VEV_6000; Mark 67 ends at +519
VEV_UNDERLYING). These are clearly not market makers — they are scripted
counterparties whose role is to provide one-way flow.

---

## What this means for our strategy

1. **Liquidity in HYDROGEL_PACK and VEV_4000 is *only* Mark 14.** If Mark 38
   isn't trading and you want to take, you're crossing Mark 14's spread.
   Conversely if you want to *provide*, you're competing for Mark 38's flow,
   and Mark 14 is the incumbent.

2. **Voucher liquidity is bifurcated.** For VEV_5200–6500, Mark 01 is the
   primary bidder and Mark 22 is the primary asker. The "mid" you see is
   really `Mark 01's bid / Mark 22's ask` for these strikes. There is almost
   no other flow on deep OTM strikes.

3. **VEV_6000 / VEV_6500 are economically dead.** Every single fill in those
   products is Mark 22 → Mark 01 at price = 0. They mark to 0 at settle.
   Buying or selling at 0 has zero PnL impact, so no edge for us either way
   on lottery-ticket strikes.

4. **The underlying has the most diverse flow.** Mark 14 quotes (passive
   two-sided), Mark 49 sells (passive one-sided), Mark 01 also takes the
   bid sometimes, while Mark 55 + Mark 67 + Mark 22 take liquidity. Spreads
   here will be the most competitive but also the most defensible if we
   want to provide.

5. **Passive bots never widen** (the spread positioning is always exactly at
   ±1 half-spread for the passive Marks). So we can model the bot quote
   layer as a static function of mid; it's not adaptive to size or speed.

6. **All seven Marks have *stable* roles across days** (aggressor rate
   varies by ≤4% between days for every Mark). Whatever pattern you key off
   on day 1 is still true on day 3.

---

## Per-day stability

```
trader  day    n   agg_rate  pas_rate
Mark 01   1   550     0.000     1.000
Mark 01   2   573     0.000     1.000
Mark 01   3   720     0.000     1.000
Mark 14   1   764     0.000     1.000
Mark 14   2   668     0.000     0.999
Mark 14   3   740     0.000     0.999
Mark 22   1   474     0.886     0.114
Mark 22   2   471     0.892     0.106
Mark 22   3   639     0.934     0.064
Mark 38   1   544     1.000     0.000
Mark 38   2   439     1.000     0.000
Mark 38   3   495     1.000     0.000
Mark 49   1    40     0.025     0.975
Mark 49   2    43     0.023     0.977
Mark 49   3    39     0.000     1.000
Mark 55   1   384     1.000     0.000
Mark 55   2   411     1.000     0.000
Mark 55   3   403     1.000     0.000
Mark 67   1    58     1.000     0.000
Mark 67   2    61     1.000     0.000
Mark 67   3    46     0.978     0.022
```

No regime shifts — each Mark's role is the same from day 1 to day 3.

---

# Appendix: Does Mark 67 move the market?

Hypothesis to test: Mark 67 is a 100% aggressive buyer of VELVETFRUIT_EXTRACT
that prints in clips of 2-15. Maybe small clips are noise and large clips
carry information / move the market.

Reproduction: `tools/mark67_signal.py` and `tools/mark67_signal2.py`.

## Setup

For each Mark 67 trade I measure forward mid-returns at horizons 100, 500,
1k, 2k, 5k, 10k, 20k, 50k timesteps. As a control I do the same for an
unconditional baseline (random tick samples) and for every other trader
that buys VELVETFRUIT_EXTRACT.

## What every Mark 67 trade does to the market

```
    who side    n  avg_qty  fwd_100  fwd_500  fwd_1k  fwd_5k  fwd_10k  fwd_50k
Mark 67  buy  165     9.15    +1.97    +1.95   +2.24   +1.92    +1.57    +1.37
Mark 55  buy  598     5.44    +0.01    +0.20   +0.01   +0.39    +0.52    +0.48
Mark 14  buy  316     5.57    -0.19    -0.03   -0.26   -0.53    -1.23    +0.13
Mark 01  buy  260     5.45    +0.23    +0.29   +0.24   -0.14    -0.64    -0.63
baseline ---  9k       —     -0.00    -0.04   -0.06   -0.03    -0.09    +0.06
```

Observations:

1. **Mark 67's effect is dramatic and persistent.** Every Mark 67 trade
   moves mid up by ~2 ticks within 100 timesteps and stays elevated by
   +1.4 even at 50,000 timesteps later.
2. **No other trader has this signature.** Mark 55 (the other 100%
   aggressive buyer of underlying) shows ~0 forward return. Mark 14 and
   Mark 01 (passive buyers, i.e. their bid is being hit) show small
   negative drift, which is the mechanical opposite (mid moves down when
   their bid is hit).
3. **Mark 49's sells mirror Mark 67's buys exactly.** Their fwd_100 is
   +1.90 because most Mark 49 sells *are* the same trades (Mark 67 lifts
   Mark 49 89 of 105 times).

## How much of this is mechanical?

When Mark 67 lifts the offer, the bid/ask refresh on the next tick can
auto-bump the mid. To separate mechanical from informational impact I
computed `mid(next tick) − mid(this tick)` as the **instantaneous jump**.

| Trader / side | n | mean instantaneous jump | median |
|---|---:|---:|---:|
| Mark 67 buys | 165 | **+1.97** | +2.0 |
| Mark 55 buys | 598 | +0.01 | 0 |
| Mark 55 sells | 600 | −0.04 | 0 |

Distribution of the Mark 67 trade-tick jump:

```
jump   count
-0.5      2
 0.0      5
 0.5     14
 1.0     12
 1.5     33
 2.0     34
 2.5     32
 3.0     15
 3.5     15
 4.0      3
```

97% of Mark 67 trades produce a positive instantaneous mid jump; ~80%
produce a jump of 1.5 or more. Subtracting the jump from the forward
returns leaves residuals scattered around zero — i.e. **almost all of the
Mark 67 forward-return effect is the mechanical mid bump from consuming
the offer level**, not subsequent price discovery.

So why does the bump persist instead of decaying? Because Mark 49 keeps
re-quoting the offer at higher levels after each lift, and Mark 14 (the MM)
adjusts both quotes around the new level. The market doesn't snap back —
the mid stays at the new higher level until the next directional move.
Velvet has thin two-sided liquidity, so a single 9-clip lift permanently
shifts the equilibrium quote.

## Does volume modulate the effect?

Bucketed by Mark 67's trade quantity:

```
  bucket   n   avg_q  fwd_100  fwd_1k  fwd_5k  fwd_50k  jump  resid_5k  resid_50k
small_2-5  18    4.6    +2.11   +2.83   +4.42    -1.31  +2.11    +2.31     -3.42
med_6-9    81    7.8    +1.86   +2.28   +1.62    +3.59  +1.86    -0.24     +1.73
big_10-12  41   10.8    +2.02   +2.10   +1.21    -0.37  +2.02    -0.82     -2.39
xl_13-15   25   14.0    +2.14   +1.94   +2.26    -1.06  +2.14    +0.12     -3.20
```

The instantaneous jump is **~+2 across every bucket** — it does not scale
with quantity. A 5-share lift produces the same mechanical effect as a
15-share lift. This is consistent with Mark 67 always taking exactly
`ask_volume_1`-worth of contracts at most, never punching through to
deeper levels.

The residuals (forward return after subtracting the jump) are noisy and
do not show a monotonic pattern in size — small bucket has +2.3 residual
at 5k, big bucket has −0.8, xl has +0.1. Sample sizes are small (n=18-81
per bucket), so I'd not draw a strong conclusion either way.

**Bottom line on size: filtering by volume does not separate Mark 67's
trades into "noise" vs "signal" buckets — every Mark 67 trade has roughly
the same impact.**

## Is it just clustering?

Mark 67 fires ~50-60 times per day, with median gap of ~13,000 timesteps
between consecutive trades:

```
day    n   gap_p25  gap_median  gap_p75  gap_mean
  1   58     6,000    10,300    23,600   17,121
  2   61     5,450    13,650    25,625   16,543
  3   46     6,800    14,800    33,400   21,176
```

For "isolated" Mark 67 trades only (no follow-up Mark 67 trade within 5,000
timesteps; n=130 of 165), forward returns are essentially identical:

```
horizon       100    500   1000   2000   5000  10000  20000  50000
mean fwd     +1.97  +2.00  +2.34  +1.73  +1.89  +1.62  +1.80  +1.25
```

So clustering doesn't explain the effect — each Mark 67 print individually
moves the mid by ~+2 and the move sticks.

## What does this mean for trading

1. **Mark 67's print is a buy event you can react to.** In the live log,
   any Mark 67 trade on VELVETFRUIT_EXTRACT comes with a +2 tick mid bump
   that is realized within one tick of the trade and persists. If our
   submission can re-quote within one tick of seeing the trade, we can
   front-run the next Mark 67 trade by joining Mark 49's offer side
   slightly higher.
2. **Volume filtering does not help.** Every clip — 2-share or 15-share —
   moves the mid the same +2 ticks. The user's hypothesis ("maybe big
   ones are signal") is not supported in the data: Mark 67's *appearance*
   is the signal, not their size.
3. **The signal is mostly mechanical, so by the time we see Mark 67's
   print, the move has already happened.** What we can capture is the
   subsequent quote-stickiness — the bid-ask doesn't snap back, so a long
   position taken right after a Mark 67 print can be exited at the new
   higher mid with no adverse selection from a reversion.
4. **Mark 49 is the other side of the same coin.** Tracking Mark 49 offer
   refreshes (especially when their offer creeps up between consecutive
   prints) is the symmetric signal — it tells us they're being lifted.
5. **Be wary of attribution to "informed flow".** This isn't a "Mark 67
   knows something" story. The market on velvet is thin and one-sided
   (Mark 49 is the only structural seller in size), so a single
   ~9-share lift permanently shifts the equilibrium quote. The +2 ticks
   is the price of taking liquidity, not a private-info premium.
