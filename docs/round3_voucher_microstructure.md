# Round 3 — voucher microstructure findings

Analysis of `historical/ROUND_3/trades_round_3_day_{0,1,2}.csv` and the
matching prices CSVs. Every trade was classified as buy- or sell-initiated by
comparing trade price to the contemporaneous `bid_1` / `ask_1`.

> **Multi-bot context.** The Prosperity market consists of multiple bots
> hand-coded by the organizers, each with its own strategy and quirks. Other
> teams' submissions are **not** in the market — we trade only against the
> organizer bots. This has two important consequences:
>
> 1. **The patterns identified here are stable.** The same organizer-made
>    bots run in live Round 3 as in the historical CSVs. Signatures we
>    observe (e.g. the OTM blanket-seller, the qty=3 dumb buyer) will
>    persist; behavior is not going to shift because of a different
>    competitive field.
> 2. **The "passive side" of every trade is also an organizer bot.** When
>    the OTM seller dumps into a bid, the resting bid is some other
>    organizer bot's MM. To capture that flow, we have to outcompete that
>    bot for the touch — not other teams.
>
> Findings below identify *signatures* (recurring trade patterns); a
> signature usually maps to one bot when the evidence is mechanical
> (identical timestamps + quantities), but can also reflect aggregate
> behavior of multiple bots sharing a quantity bucket.

## TL;DR

The voucher tape splits into **three distinct regimes** with completely
different bots driving each. Two of them point at concrete strategy edges.

| regime | strikes | what's happening | edge |
|---|---|---|---|
| **A. Wide-spread balanced** | VEV_4000 | 21-tick spread; ~155 trd/day; balanced 51/49 buy/sell; qty ∈ {1,2,3} | MM with pennying inside the spread (per v3 file) |
| **B. Dead** | VEV_4500, 5000, 5100 | 1 trade per voucher across 3 days; 100% buys | Pure intrinsic-arb opportunities only |
| **C. One-sided OTM dump** | VEV_5200 → 6500 | Tight spreads; **100% sell-initiated** flow on 5400+ | Join the bid → collect the seller's spread risk-free |

## Per-voucher overview

| strike | #trades | trd/day | qty range | avg mid | avg spread | %spread≤1 |
|-------:|--------:|--------:|----------:|--------:|-----------:|----------:|
| 4000 | 464 | 154.7 | **1–3** | 1250.11 | **20.81** | 0.0% |
| 4500 | 1 | 0.3 | 1 | 750.11 | 15.85 | 0.0% |
| 5000 | 1 | 0.3 | 1 | 255.02 | 6.04 | 0.0% |
| 5100 | 1 | 0.3 | 1 | 166.81 | 4.30 | 0.3% |
| 5200 | 18 | 6.0 | 1–5 | 95.55 | 2.89 | 0.9% |
| 5300 | 121 | 40.3 | 1–5 | 46.76 | 2.11 | 2.9% |
| 5400 | 225 | 75.0 | 2–5 | 15.95 | 1.38 | 61.9% |
| 5500 | 267 | 89.0 | 2–5 | 6.64 | 1.15 | 85.0% |
| 6000 | 284 | 94.7 | 2–5 | 0.50 | 1.00 | 100.0% |
| 6500 | 284 | 94.7 | 2–5 | 0.50 | 1.00 | 100.0% |

## Trade execution location (where do trades print?)

| voucher | #trd | @ bid | @ ask | inside | outside |
|--------:|-----:|------:|------:|-------:|--------:|
| 4000 | 464 | 51.3% | 48.7% | 0.0% | 0.0% |
| 4500 | 1 | 0.0% | 100.0% | – | – |
| 5000 | 1 | 0.0% | 100.0% | – | – |
| 5100 | 1 | 0.0% | 100.0% | – | – |
| 5200 | 18 | 94.4% | 5.6% | 0.0% | 0.0% |
| 5300 | 121 | **98.3%** | 0.8% | 0.8% | 0.0% |
| 5400 | 225 | **100.0%** | 0.0% | 0.0% | 0.0% |
| 5500 | 267 | **100.0%** | 0.0% | 0.0% | 0.0% |
| 6000 | 284 | **100.0%** | 0.0% | 0.0% | 0.0% |
| 6500 | 284 | **100.0%** | 0.0% | 0.0% | 0.0% |

**Zero trades print inside the spread.** The simulator's matching only fires
at `bid_1` / `ask_1` (confirms the v3 file's diagnosis). Pennying is dead air
on tight books; only on VEV_4000's 21-tick spread can pennying actually catch
flow because the simulator's market orders cross several ticks.

## Net signed quantity per voucher per day

| voucher | day0 | day1 | day2 | total |
|--------:|-----:|-----:|-----:|------:|
| 4000 | +15 | −37 | −8 | **−30** (balanced) |
| 4500 | 0 | +1 | 0 | +1 |
| 5000 | 0 | +1 | 0 | +1 |
| 5100 | 0 | +1 | 0 | +1 |
| 5200 | −15 | −20 | −26 | **−61** |
| 5300 | −128 | −128 | −157 | **−413** |
| 5400 | −218 | −286 | −283 | **−787** |
| 5500 | −281 | −321 | −335 | **−937** |
| 6000 | −320 | −345 | −337 | **−1002** |
| 6500 | −320 | −345 | −337 | **−1002** |

**Key observation:** VEV_6000 and VEV_6500 have **identical** trade counts and
net flow on every single day. Almost certainly a single bot trades both pinned
strikes symmetrically.

The selling pressure scales monotonically with how OTM the strike is. There is
a "lottery-ticket dumper" bot whose entire program is *sell deep OTM calls
into whatever bid is showing*.

## Is the OTM seller informed?

If the seller is informed, voucher prices should fall after their sells.
Signed forward returns (positive = voucher fell, in seller's favor):

| voucher | n | mean ΔP+5k | t(+5k) | mean ΔP+20k | t(+20k) |
|--------:|--:|-----------:|-------:|------------:|--------:|
| 5200 | 17 | −0.618 | −0.48 | −0.688 | −0.33 |
| 5300 | 119 | +0.340 | +1.40 | +0.004 | +0.01 |
| 5400 | 225 | +0.036 | +0.39 | +0.181 | +1.20 |
| 5500 | 267 | −0.009 | −0.23 | +0.045 | +0.70 |

**Verdict: NOT informed.** All t-stats are inside ±2; the "predictive power"
is noise. Same result if we test against the underlying (`VELVETFRUIT_EXTRACT`).

This is the **strongest microstructure signal in the dataset**: the OTM
seller is dumping size into the book with no edge, paying spread on every
trade. The PASSIVE side of these trades — the resting bidder — is collecting
spread risk-free.

## Strategy implications

### 1. MM the OTM strikes by joining the bid (VEV_5300 / 5400 / 5500)
The v3 file's "join the external touch" approach on these strikes is exactly
right. Every trade hits the bid; if you have a bid at the touch you'll get
filled by the seller and the price doesn't move adversely. This is **paid
liquidity provision against an uninformed taker**. The size-asymmetry trick
(bigger ask when long, bigger bid when short) is gravy.

Note: VEV_5200 is borderline (only 6 trd/day). Probably not worth the inventory.

### 2. VEV_4000 needs different treatment
This is the only voucher with two-way flow, and quantity distribution {1,2,3}
suggests a **different bot** trades it (note: qty=1 is unique to VEV_4000 —
it appears on no other voucher). Wide spread (21 ticks) means pennying inside
the spread *does* work because the simulator's market orders sweep multiple
ticks. v3 already does this via `trade_deep_voucher`.

### 3. VEV_4500 / 5000 / 5100 are dead pools
1 trade per voucher across 3 days. The only PnL on these comes from
intrinsic arbitrage (ask < S − K) which fires rarely but should be caught
when it does. v3's `trade_deep_voucher` covers this.

### 4. VEV_6000 / 6500: skip them
Bid=0, ask=1 always. Every trade prints at 0 and contracts settle at 0
(per `prosperity_pnl_floor_settle.md`). Trading these is **pure 0 P&L** for
both sides. The fact that 568 trades happened across 3 days with zero
expected value tells you the simulator is firing them as noise. Don't waste
inventory limit (300 per voucher) on a guaranteed-zero asset.

## Cross-strike co-firing — IDENTIFIED THE BOT

The OTM seller is **one bot** that simultaneously sweeps multiple strikes on
each fire. Smoking gun: **VEV_6000 and VEV_6500 trade at exactly the same
284 timestamps with exactly the same 284 quantities.** Zero solo fires.

Distribution of strikes hit per (day, timestamp) when this bot fires:

| #strikes | count |
|---------:|------:|
| 1 (solo) | **0** |
| 2 | 4 |
| 3 | 39 |
| 4 | 149 |
| 5 (all OTM) | 92 |

Of 284 unique sweeps, **0 are solo** — when the bot fires, it always hits
multiple strikes. 92 sweeps hit all 5 OTM strikes (5300, 5400, 5500, 6000,
6500) with **identical quantity on every strike** in the sweep.

### The bot's universe is K ≥ 5300 (answers "why does VEV_5100 have no flow")

- VEV_4500 / 5000 / 5100: **excluded** — bot logic requires K ≥ 5300
- VEV_5200: only 18 trades in 3 days — borderline; bot occasionally includes it
- VEV_5300+: always included in sweeps

The "no man's land" between K=4000 (intrinsic-arb bot) and K=5300 (blanket
seller bot) has no bot logic targeting it. That's why near-ATM strikes have
no flow despite being closest to fair value.

### Firing cadence — irregular, event-triggered

| day | n_fires | median gap | mean gap | min | max |
|----:|--------:|-----------:|---------:|----:|----:|
| 0 | 91 | 8,750 | 11,016 | 100 | 44,300 |
| 1 | 98 | 6,900 | 10,163 | 100 | 51,500 |
| 2 | 95 | 8,850 | 10,433 | 200 | 39,300 |

Not a periodic timer. Min gap of 100 ticks (one snap apart) means the bot can
fire multiple times in immediate succession.

### The bot fires at S extremes (mild vol trigger)

S rank within the past 5k-tick window at sweep time (0=local low, 1=local high):

| S rank bucket | sweeps |
|:-------------|------:|
| 0–20% (local low) | 76 |
| 20–40% | 41 |
| 40–60% | 40 |
| 60–80% | 37 |
| 80–100% (local high) | **90** |

**Bimodal**: the bot fires more at local extremes than in the middle. 58% of
sweeps happen when S is in the top or bottom 20% of recent range. So it has
*some* vol-event trigger logic, but it doesn't matter which direction S
moved — just that it moved.

### Strategy implication: BLANKET BID across OTM strikes

Because the bot sweeps all 5 OTM strikes simultaneously with the same qty:
- A single bid at the touch on each of VEV_5300/5400/5500 catches the seller's
  full sweep (qty 2-5 per strike) every time it fires — **but only if our
  bid is in front of the resident organizer-bot bid that's currently
  collecting that flow**. Today an organizer MM bot is the passive side.
- Per-fire potential: ~3 strikes × 3.5 avg qty × ~1 tick spread = ~10 ticks profit
  per sweep, ~95 sweeps/day → **~950 ticks/day theoretical capture** if we
  fully displace the organizer's resting bidder.
- Skip VEV_6000/6500: trades print at 0, settle at 0, zero P&L on either side.
- VEV_5200 borderline: only 6 trd/day, probably not worth inventory.

## Bots operate in product silos — no cross-product hedging

Multi-product co-firing analysis (same timestamp, ≥2 products with prints):

| combo | occurrences |
|---|---:|
| VFE only | 1,298 |
| HYDROGEL only | 946 |
| VEV_4000 only | 433 |
| VEV_5400+5500+6000+6500 (OTM sweep, no 5300) | 106 |
| VEV_5300+5400+5500+6000+6500 (full OTM sweep) | 72 |
| HYDROGEL + VFE | 39 |
| VFE + VEV_4000 | 15 |
| HYDROGEL + VEV_4000 | 12 |

**The OTM blanket-seller does NOT delta-hedge.** VFE trade within ±200 ticks
of an OTM sweep occurs **20.1%** of the time, vs random-baseline **22.4%**.
The seller is naked short vol — never touches the underlying after firing.

Same pattern across the board: bots are single-product. None of the voucher
bots hedge in VFE; HYDROGEL bots don't trade options.

### Why this matters

1. **The vol surface isn't being arbitraged.** Nobody trades multiple
   strikes coherently except the OTM blanket-seller (who only sells, no
   pricing logic). That means the surface is shaped purely by where the
   seller dumps and where the resident bidder sits — there's no smart
   bot enforcing fair vol across strikes. Cross-strike strategies
   (calendar spreads, butterflies, vol-curve fits) have no competition.
2. **Delta hedging is uncontested edge.** If our voucher MM accumulates
   long delta from getting hit on bids, hedging in VFE is something no
   organizer bot does. We can size voucher quotes more aggressively
   knowing our delta is hedgeable cheaply.

## Caveats

- 3 days × ~270 trades/day per voucher is a small sample. Per-quantity
  analysis on VEV_5200 (n=17 sells) is essentially noise.
- Buyer/seller fields are empty in Prosperity historical CSVs, so the
  "one bot" inference rests on identical aggregates and 100% co-firing.
  Worth confirming on live logs.
- The blanket-sweep capture math assumes we displace the resident
  organizer-bot bidder on each strike. In live, FIFO priority at the
  touch will determine actual capture.
