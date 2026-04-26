# Probe trader · log 403865 · liquidation prices

- Day 2; trader bought 5 of each voucher at ts=99000; run ended at ts=99900.
- Underlying S = 5264.00 at liquidation; TTE = 5.9001 days.
- **Realized σ** on VELVETFRUIT_EXTRACT, Roll-bounce-corrected at k=1: 1.7869 %/√day (raw 2.2065).
- **Flat ATM IV** (median IV across K∈[5000,5500]): 1.2767 %/√day.

**liq** = effective liquidation price = `cost + PnL/qty`. **BS-IV(K)** prices each strike at its own IV (so it equals mid by construction — sanity check). **BS-IV(flat)** prices every strike at the ATM IV — shows what the *no-smile* model says. **BS-RV** uses the Roll-corrected realized σ (one number, applied flat).

| K | buy | liq | mid | bid | ask | PnL/5 | IV % | BS-IV(K) | BS-IV(flat) | BS-RV | BS-RV(raw) | liq − mid | mid − BS-RV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 1276 | 1264.077 | 1264.00 | 1253 | 1275 | -59.613 | — | — | 1264.000 | 1264.000 | 1264.000 | +0.077 | -0.000 |
| 4500 | 773 | 764.077 | 764.00 | 756 | 772 | -44.613 | — | — | 764.000 | 764.008 | 764.129 | +0.077 | -0.008 |
| 5000 | 271 | 267.203 | 267.00 | 264 | 270 | -18.983 | 1.260 | 267.000 | 267.211 | 276.868 | 288.702 | +0.203 | -9.868 |
| 5100 | 179 | 175.979 | 176.00 | 174 | 178 | -15.105 | 1.247 | 176.000 | 176.868 | 194.568 | 211.557 | -0.021 | -18.568 |
| 5200 | 105 | 102.554 | 102.50 | 101 | 104 | -12.229 | 1.294 | 102.500 | 101.696 | 126.162 | 146.758 | +0.054 | -23.662 |
| 5300 | 52 | 50.135 | 50.00 | 49 | 51 | -9.326 | 1.298 | 50.000 | 48.915 | 74.579 | 95.838 | +0.135 | -24.579 |
| 5400 | 17 | 16.101 | 16.00 | 15 | 17 | -4.496 | 1.191 | 16.000 | 19.095 | 39.815 | 58.664 | +0.101 | -23.815 |
| 5500 | 7 | 6.483 | 6.50 | 6 | 7 | -2.587 | 1.306 | 6.500 | 5.928 | 19.071 | 33.557 | -0.017 | -12.571 |
| 6000 | 1 | 0.001 | 0.50 | 0 | 1 | -4.996 | 2.116 | 0.500 | 0.000 | 0.088 | 0.722 | -0.499 | +0.412 |
| 6500 | 1 | 0.000 | 0.50 | 0 | 1 | -5.000 | — | — | 0.000 | 0.000 | 0.003 | -0.500 | +0.500 |

## What to notice

- For all ITM/ATM strikes, **liq ≈ mid** (within ±0.2 ticks), so the engine settles at mid.
- For **VEV_6000 and VEV_6500** (price floor: bid=0, ask=1), liq = 0.0, NOT the mid of 0.5. The engine effectively closes floor-pinned positions at the bid. Buying these as 'lottery tickets' loses the full premium.
- **BS-IV(K) = mid** by construction; included as numerical sanity check.
- **BS-IV(flat) ≠ mid** at OTM wings — that's the smile. ATM IV understates the wing IVs the market is actually quoting.
- **BS-RV >> mid** for ATM strikes because the per-tick realized σ (~1.79%/√day) is still inflated by bid-ask bounce — even after Roll correction. CLAUDE.md notes the 'true' σ from longer windows is closer to ~1.3%/√day. The market IV (~1.28%) is the more trustworthy vol estimate here.
