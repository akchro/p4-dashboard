# Round 3: Options Primer

A `VEV_K` voucher is a **European call** on `VELVETFRUIT_EXTRACT` with strike `K`,
expiring at end of round 7 (no early exercise, no carry between rounds). At expiry,
each voucher is worth `max(S_T − K, 0)`. Before expiry it has additional **time
value**, and the whole game is figuring out what that's worth.

## Variables

| symbol  | meaning                                               | source                            |
| ------- | ----------------------------------------------------- | --------------------------------- |
| `S`     | spot price of `VELVETFRUIT_EXTRACT`                   | mid-price of underlying           |
| `K`     | voucher strike                                        | label (4000…6500)                 |
| `T`     | time to expiry, in days                               | 5d at start of round 3, 4d at end |
| `σ`     | volatility of `VELVETFRUIT_EXTRACT` (per-day)         | **estimate from data**            |
| `r`     | risk-free rate                                        | assume `0`                        |
| `C`     | observed voucher mid-price                            | order book                        |
| `IV(K)` | implied vol — `σ` such that `BS(S,K,T,σ)=C`           | numerical solve                   |

## Decomposition

Every option price decomposes:

```
C = intrinsic + extrinsic
intrinsic(S,K)   = max(S − K, 0)
extrinsic(S,K,T,σ) ≥ 0,  → 0 as T → 0
```

Snapshot from `historical/ROUND_3/prices_round_3_day_2.csv` (avg `S ≈ 5255`,
TTE=6d):

| strike | avg C | intrinsic | extrinsic | moneyness        |
| -----: | ----: | --------: | --------: | ---------------- |
|   4000 |  1255 |      1255 |      ~0   | deep ITM         |
|   4500 |   755 |       755 |      ~0   | deep ITM         |
|   5000 |   259 |       255 |      ~4   | ITM              |
|   5100 |   167 |       155 |      ~12  | ITM              |
|   5200 |    94 |        55 |      ~39  | slightly ITM     |
|   5300 |    44 |         0 |      ~44  | slightly OTM     |
|   5400 |    14 |         0 |      ~14  | OTM              |
|   5500 |     5 |         0 |      ~5   | OTM              |
|   6000 |   0.5 |         0 |     0.5   | floor (no signal)|
|   6500 |   0.5 |         0 |     0.5   | floor (no signal)|

The 6000/6500 strikes pin at the 0.5 quote floor — they carry no usable price
signal. Real action is at strikes 5000–5500.

## No-arb bounds (free PnL if violated)

```
max(S − K, 0) ≤ C ≤ S
C(K1) ≥ C(K2)   for K1 < K2          (monotone in strike)
C(K1) − C(K2) ≤ K2 − K1              (slope bounded)
C convex in K                        (butterflies ≥ 0)
```

If the book ever quotes outside these, lift/hit it.

## Black–Scholes (with r=0)

```
d1  = ( ln(S/K) + ½ σ² T ) / ( σ √T )
d2  = d1 − σ √T
C   = S · N(d1) − K · N(d2)
```

`N` is the standard normal CDF. Closed form, vectorisable, ~µs per quote.

## Greeks (sensitivities)

| greek | formula (r=0)               | meaning                                      |
| ----- | --------------------------- | -------------------------------------------- |
| Δ     | `N(d1)`                     | ∂C/∂S — share-equivalents per voucher        |
| Γ     | `φ(d1) / (S σ √T)`          | ∂Δ/∂S — how fast Δ moves                     |
| Θ     | `−S φ(d1) σ / (2 √T)`       | ∂C/∂t — time decay (per day, σ per day)      |
| ν     | `S φ(d1) √T`                | ∂C/∂σ — vega                                 |

`φ` is the standard normal PDF. Δ runs 0 (deep OTM) → 1 (deep ITM); Γ and ν peak
ATM. Θ is most negative ATM with small T — these are the strikes that bleed
fastest into expiry.

## Implied volatility & the smile

Given `C`, invert BS for `σ`. Bisection on `[1e-6, 5]` converges in ~20 iterations.

In a flat-vol world, IV would be the same for every strike. In practice it
isn't — plotting `IV(K)` vs `K` (or vs **log-moneyness** `m = ln(K/S)/√T`) gives
a curve, usually a smile or skew. Two uses:

1. **Cross-sectional fit.** At each timestamp, fit `IV(m) ≈ a + b·m + c·m²`
   over the liquid strikes (5000–5500). Vouchers above the fit are rich, below
   are cheap — trade the residual.
2. **Time-series mean reversion.** Track `IV(K)` per strike over time; when one
   strike's IV deviates from its rolling mean by some z-score, fade it.

## Estimating σ

Two options:

- **Realised vol** — compute log-returns of `VELVETFRUIT_EXTRACT` mid-price,
  scale to per-day. This is a *prior* for σ.
- **Implied vol** — invert observed voucher prices. The market's σ. Differs
  from realised → directional vol opinion baked into voucher prices.

Trading the gap between realised and implied is "vol arbitrage."

## Delta hedging

A long voucher position is exposed to S moving. To isolate vol exposure (so the
PnL only depends on σ being right), short Δ shares of `VELVETFRUIT_EXTRACT` per
long voucher. Re-hedge as Δ drifts.

```
hedge_VE = − Σ_K  (position_K · Δ_K)
```

Caveats specific to this round:

- Position limit on `VELVETFRUIT_EXTRACT` is 200. A maxed-out 300-voucher
  position with Δ=0.7 implies a hedge of 210 shares — already over the cap.
  Greeks-aware sizing matters.
- `HYDROGEL_PACK` is unrelated to vouchers; it doesn't hedge anything here.

## Time decay across rounds

Same strike, three days of history, falling TTE:

| strike | day 0 (TTE=8d) | day 1 (TTE=7d) | day 2 (TTE=6d) |
| -----: | -------------: | -------------: | -------------: |
|   5300 |          48.9 |          46.9 |          44.5 |
|   5400 |          18.5 |          15.6 |          13.7 |
|   5500 |           8.1 |           6.6 |           5.3 |

OTM extrinsic decays roughly as `√T` (Θ accelerates near expiry). At TTE=5d the
5400/5500 strikes are bleeding ~1–2 ticks of value per day even with S flat —
short these unhedged and theta works for you, long them and it works against.

## Strategy menu

| approach                  | edge                                | risk                              |
| ------------------------- | ----------------------------------- | --------------------------------- |
| no-arb monitor            | book mistakes (bounds violations)   | rare; size when seen              |
| smile-fit residual        | relative value across strikes       | model risk (smile shape wrong)    |
| IV mean-reversion         | per-strike IV is autocorrelated     | regime shift in σ                 |
| realised-vs-implied       | vol risk premium                    | path-dependent; needs hedging     |
| delta-neutral market make | quote both sides, hedge Δ in VE     | inventory limits on VE (±200)     |
| theta harvest (short OTM) | Θ < 0 means short collects decay    | left tail if S spikes through K   |

The cleanest first build: BS pricer → IV solver → fit a quadratic smile across
the five liquid strikes per timestamp → quote against the fit, hedge net Δ in
`VELVETFRUIT_EXTRACT` subject to the 200 cap. Add theta-harvest on the wings
only after the core hedger is stable.
