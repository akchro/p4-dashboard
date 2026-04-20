# Round 2: Speed vs Profit Analysis

## Structure

From `components/manual_round2.py:5-14`:

```
research(r) = 200_000 · log(1+r) / log(101)
scale(s)    = 7s/100
P(r,s)      = research(r) · scale(s)        # gross profit
PnL         = P(r*,s*) · speed_mult(sp) − 500·(r+s+sp)
```

Since `r + s = B = budget_cap - sp`, the inner optimization collapses to a single
function of remaining budget: `P*(B) = max_{r+s=B} research(r)·scale(s)`.

## Closed form for P*(B)

Let `u = 1 + r*`. The first-order condition on `log(1+r)·(B−r)` gives:

```
u · (1 + log u) = B + 1
P*(B)           = K · u · (log u)²           K = 14,000 / log(101) ≈ 3,033.5
```

Differentiating (the `(2 + log u)` factor cancels cleanly):

```
dP*/dB = K · log u
```

## Shape

Computed values across the speed grid:

| sp  | B   | P*(sp)  | dP/dsp  |
| --: | --: | ------: | ------: |
|   0 | 100 | 742,336 |  −9,600 |
|  25 |  75 | 509,141 |  −8,884 |
|  50 |  50 | 296,207 |  −7,888 |
|  75 |  25 | 113,629 |  −6,196 |
|  90 |  10 |  29,664 |  −3,930 |
|  95 |   5 |  10,012 |  −2,002 |

**It is not logarithmic — the shape is `B · log(B)`.**

Checking ratios vs the B=100 baseline (742k):

| B   | actual / P*(100) | linear | pure log | B·log B |
| --: | ---------------: | -----: | -------: | ------: |
| 50  | 0.40             | 0.50   | 0.85     | 0.43 ✓  |
| 25  | 0.15             | 0.25   | 0.70     | 0.17 ✓  |

For large B, `u ≈ B / log B`, so `P*(B) ≈ K · B · log B`. The logarithm only
appears in the **slope** `dP*/dB = K · log u`, not in P itself.

## Practical takeaway

- `P*(sp)` is **convex decreasing** in speed: marginal profit lost per extra 1%
  of speed shrinks as you invest more (from ~−9.6k at sp=0 to ~−4k at sp=90).
- Roughly linear-with-gentle-concavity in `sp` over the middle range
  (sp ∈ [10, 70]) — the log factor moves slowly there.
- Clean abstraction:

  ```
  PnL(sp) ≈ K · (100 − sp) · log(101 − sp) · m(sp) − 50,000
  ```

  where `m(sp)` is the rank-based speed multiplier. The optimum sits where the
  (roughly linear) growth of `m(sp)` overtakes the `B · log B` decay of `P*`.
