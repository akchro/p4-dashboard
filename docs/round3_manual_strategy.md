# Round 3 manual: The Celestial Gardeners' Guild — strategy

## TL;DR

| Belief about field | Bid | Expected profit / counterparty |
|---|---|---|
| Field plays roughly Nash (avg b2 ≈ 836) | **b1 = 751, b2 = 836** | ~84.3 |
| Field bids visibly higher (avg b2 > 836) | **b1 = (670 + ⌈avg⌉)/2 + 1, b2 = ⌈avg⌉ + 1** | depends |
| Field bids low (avg b2 < 836) | **b1 = 751, b2 = 836** (Nash unchanged) | ~84.3 |

The two-bid Nash equilibrium of this game splits the price range into thirds:
`b1 ≈ 753.33`, `b2 ≈ 836.67`. Profit at the equilibrium is `(920-670)/3 ≈ 83.33`
per trade. The discrete optimum (placing each bid one tick above a reserve grid
point) gives `(751, 836)` with **84.33**.

The Nash is also the *best response* to any field that itself plays at or below
Nash. The only time you should deviate is when you genuinely believe the average
second bid will exceed 836 — and then you bid just above the average.

---

## Game recap

Reserves `R ∈ {670, 675, ..., 920}` (51 equiprobable values). You sell at
920 the next day. Submit two bids `b1`, `b2`. Per counterparty:

```
R < b1                          → trade at b1, profit  = 920 − b1
b1 ≤ R < b2 and b2 >  avg_b2    → trade at b2, profit  = 920 − b2
b1 ≤ R < b2 and b2 ≤  avg_b2    → trade at b2, profit ×= ((920 − avg_b2) / (920 − b2))³
R ≥ b2                          → no trade
```

`avg_b2` is the mean of every (human) player's second bid, including yours.
The bid penalty is **cubic** in the under-shoot, so under-bidding by a few
ticks is much worse than over-bidding by the same amount.

If `b1 > b2`, your second bid is wasted (`R < b1 ⇒ R < b2` so the lower bid
always fires first). Always pick `b1 < b2`.

---

## Solving for Nash

Write the symmetric-equilibrium expected profit per counterparty (continuous
approximation, `b2 = avg_b2` so penalty multiplier = 1):

```
E[π] = (b1 − 670)(920 − b1)/250  +  (b2 − b1)(920 − b2)/250
```

First-order conditions:

```
∂E[π]/∂b1 = 0   ⇒  b1 = (670 + b2)/2
∂E[π]/∂b2 = 0   ⇒  b2 = (920 + b1)/2
```

Solving simultaneously:

```
b1* = 670 + (920 − 670)/3   = 753.33
b2* = 670 + 2(920 − 670)/3  = 836.67
E[π*] = (920 − 670)/3       = 83.33
```

Geometrically: the bids partition `[670, 920]` into three equal thirds. With
`n` bids the analogue is to split into `n+1` equal segments, giving
`E[π] = 125 · n / (n+1)`. So the marginal value of having a second bid is
`83.33 − 62.5 = 20.83` per counterparty.

### Why no profitable deviations exist

- **Upward deviation** to `b2' > 836.67`: the FOC `b2 = (920 + b1)/2 = 836.67`
  is exactly the global maximum. Any move up reduces profit (second order).
- **Downward deviation** to `b2' < 836.67`: penalty kicks in and the
  `(b2' − b1)/(920 − b2')²` term is monotonically increasing in `b2'` over the
  feasible region — so the best penalised bid asymptotes the equilibrium from
  below.

Note: by the same logic, *any* `b2* ≥ 836.67` (with matching
`b1 = (670+b2*)/2`) is also a Nash equilibrium. They form a Pareto-ordered
chain: bidding higher hurts everyone equally, but no one profits by deviating
down. **836.67 is the Pareto-best Nash** and is the natural focal point —
the only equilibrium where the FOC for `b2` is interior rather than corner.

### Discrete tightening

Because reserves are at multiples of 5, you can shave the bid down to the next
integer above any reserve. With `b1 = 751` you catch all `R ≤ 750`; with
`b2 = 836` you catch all `R ≤ 835`:

```
E[π] = (17/51)·(920−751) + (17/51)·(920−836)
     = (17/51)·169 + (17/51)·84
     = 17·253/51
     = 84.33
```

This is **+1.0 per counterparty** above the continuous Nash, just from
discretisation. Equivalent answers exist on the same grid: `(751, 841)`,
`(756, 836)` are within rounding error.

---

## Best response to a non-Nash field

Given a fixed `avg_b2`, the best two-bid strategy depends on where the average
falls:

### Case 1: `avg_b2 ≤ 836`

Nash dominates — `(751, 836)` has `b2 > avg_b2`, no penalty, full profit.
There is no exploit; pushing b2 lower would forfeit profit per trade and
trigger penalty.

### Case 2: `avg_b2 > 836`

Nash now triggers the penalty. The cubic shrinks profit fast: at `avg_b2 = 845`
the multiplier is `(75/84)³ ≈ 0.71`. Better to bid `b2'` just above `avg_b2`.

Once `b2'` is fixed at `⌈avg_b2⌉ + 1`, the optimal `b1` follows from the same
FOC:

```
b1' = (670 + b2')/2
```

Worked example with `avg_b2 = 845`:

- Pure Nash (751, 836): 17/51·169 + 17/51·(84·0.71) = 76.2
- Exploit (758, 846): 18/51·162 + 18/51·74 = **83.3**

That's +7 per counterparty by stepping over the average.

The deeper the field over-bids, the worse Nash gets — but the exploit also
yields less because you're paying more per trade. Past `avg_b2 ≈ 870` the
two-bid game devolves into something close to a single-bid problem.

---

## Reading the dashboard

The Round 3 page in the **Manual** tab takes a 5-component player population:

| Group | What they bid | Default share |
|---|---|---|
| Perfect Nash | exactly `nash_b2` (default 836) | 15% |
| Concentrated | normal around `conc_mid` (default 795) | 25% |
| Slightly higher | normal around `nash_b2 + offset` (default 846) | 50% |
| Random | uniform over a band | 5% |
| Nice numbers | random pick from `{700, 750, 800, 850, 900}` | 5% |

It samples `n_players` second bids from the mixture, computes `avg_b2`, then
draws the full E[π] surface over (b1, b2). Three markers:

- **× white** — theoretical Nash (751, 836)
- **★ yellow** — the optimal exploit given the realised `avg_b2`
- **● red** — your tunable strategy (b1, b2 sliders)

Tune the population sliders to stress-test your read on the field. The summary
table at the bottom gives the profit of (a) continuous Nash, (b) discrete
Nash, (c) the best single-bid strategy as a sanity floor, (d) your manual
strategy, and (e) the optimal exploit.

### What the population mixture says

With the user's stated weights (15/25/50/5/5):

- If "concentrated" players bid the **naive midpoint 795** and "slightly
  higher" players bid `+10` (= 846), the mass at 795 pulls `avg_b2` to
  **~827** — *below* Nash. The exploit is just **(751, 836)** at 84.3.
- If "concentrated" players actually understand Nash and bid around **836**,
  the 50% bidding `+10` dominates and `avg_b2 ≈ 837` — barely above Nash.
  The exploit shifts to roughly **(751, 841)** for ~84.2 (negligible gain).
- If "slightly higher" players go aggressive (`+25` → 861), `avg_b2 ≈ 847`.
  Now the exploit is **(756–760, 851)** for ~83.5 — a meaningful +9 over a
  naive Nash play that eats the penalty.

The crucial unknown is what the "concentrated" group thinks "the optimal mid"
is. The slider lets you flip between the naive (795) and Nash-aware (836)
interpretation in one click.

---

## Recommended strategy

1. **Default to discrete Nash: `b1 = 751, b2 = 836`.**
   This is the strictly best response to anyone who reasons their way to
   Nash, and a near-best response to fields biased low or moderately high.

2. **If your prior is that >50% of the field will visibly out-bid Nash by
   ≥ 10**, shift to **`b1 = 756, b2 = 846`** (or wherever the dashboard
   exploit star lands). This is the GTO-aware exploit; the asymmetry of
   the cubic penalty makes "slightly above the expected average" much
   safer than playing Nash flat.

3. **Don't bid below Nash on b2.** The cubic penalty wipes out the
   per-trade gain almost immediately. Even if you think the field is
   conservative, the upside of a lower `b2` is bounded by ~5 per trade
   while the downside on a slightly-higher field is ~10–20.

4. **Don't fancy `b1`.** Once `b2` is set, the b1 FOC `(670+b2)/2` is
   robust — there's no profitable creative move on the lower bid.

5. **Pick odd-looking ticks (751, 836, 846)**, not multiples of 5. Reserves
   sit on the multiples-of-5 grid; bidding at a multiple of 5 wastes the
   chance to capture the reserve at that exact price. `750` catches the same
   reserves as `746` but pays 5 more.

### Single answer if forced to pick one set of bids

**`b1 = 751, b2 = 846`.** This gives up only ~0.1 vs pure Nash on a Nash-like
field, but earns ~5–9 if the "slightly higher" group meaningfully drags the
average up — which the user's own population model says is the most likely
outcome.
