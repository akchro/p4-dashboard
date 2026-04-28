#!/usr/bin/env python3
"""Extract tier-2 voucher fills from a Prosperity backtest log into CSV.

One row per SUBMISSION trade on VEV_5000/5100/5200, with the detector that
triggered the order (replayed from activitiesLog) and the mid 20 ticks later.

Usage:
    python3 tools/extract_t2_fills.py logs/<file>.log [-o out.csv]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Mirror voucher_long_short.py constants
TIER2_STRIKES = [5000, 5100, 5200]
SIGNAL_STRIKES = [5300, 5400, 5500]
LOOKBACK = 200
DIP_PCTL = 0.20
DIV_LOOKBACK = 20
DIV_SMOOTH = 10
DIV_MIN_BULL = 10
DIV_DECAY = 0.5
PEAK_QTY = 30
DIP_QTY = 30
TIER2_QUOTE_QTY = 10
BULL_HIST_LEN = 30
MID_HIST_LEN = 200
LOOKAHEAD_TS = 2000  # 20 ticks * 100 ts/tick


def smoothed(seq, j, window):
    s = max(0, j - window + 1)
    sub = seq[s:j + 1]
    return sum(sub) / len(sub) if sub else 0.0


def detect_peak(bull_hist, mid_hist):
    n = len(bull_hist)
    if n < 1 or len(mid_hist) <= DIV_LOOKBACK:
        return False
    i = n - 1
    j_lo = max(0, i - DIV_LOOKBACK + 1)
    bs_max = max(smoothed(bull_hist, j, DIV_SMOOTH) for j in range(j_lo, i + 1))
    bs_now = smoothed(bull_hist, i, DIV_SMOOTH)
    mid_chg = mid_hist[-1] - mid_hist[-DIV_LOOKBACK - 1]
    return mid_chg > 0 and bs_max >= DIV_MIN_BULL and bs_now <= DIV_DECAY * bs_max


def detect_dip(bull_hist, mid_hist):
    n = len(bull_hist)
    if n < 1 or len(mid_hist) <= DIV_LOOKBACK:
        return False
    i = n - 1
    j_lo = max(0, i - DIV_LOOKBACK + 1)
    bs_min = min(smoothed(bull_hist, j, DIV_SMOOTH) for j in range(j_lo, i + 1))
    bs_now = smoothed(bull_hist, i, DIV_SMOOTH)
    mid_chg = mid_hist[-1] - mid_hist[-DIV_LOOKBACK - 1]
    return mid_chg < 0 and bs_min <= -DIV_MIN_BULL and bs_now >= DIV_DECAY * bs_min


def parse_activities(csv_text):
    """Return book[ts][product] = {bid_1, ask_1, bid_vol_1, ask_vol_1, mid}."""
    book = defaultdict(dict)
    lines = csv_text.split('\n')
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split(';')
        ts = int(parts[1])
        product = parts[2]

        def f(i):
            v = parts[i]
            return float(v) if v not in ('', None) else None

        def n(i):
            v = parts[i]
            return int(v) if v not in ('', None) else None

        book[ts][product] = {
            'bid_1': f(3),
            'bid_vol_1': n(4),
            'ask_1': f(9),
            'ask_vol_1': n(10),
            'mid': f(15),
        }
    return book


def replay_flags(book, timestamps):
    """Replay strategy bull/mid history to compute peak/dip/pctl per (ts, K).

    Returns flags[(ts, K)] = {'peak': bool, 'dip': bool, 'pctl': bool}.
    Skips ticks where mid is unavailable (matching the strategy's `continue`).
    """
    bull_hist = []
    mid_hist = {K: [] for K in TIER2_STRIKES}
    flags = {}

    for ts in timestamps:
        snap = book[ts]
        bull = 0
        for K in SIGNAL_STRIKES:
            r = snap.get(f'VEV_{K}')
            if r and r['bid_1'] is not None and r['ask_1'] is not None:
                bull += (r['bid_vol_1'] or 0) - (r['ask_vol_1'] or 0)
        bull_hist.append(int(bull))
        if len(bull_hist) > BULL_HIST_LEN:
            bull_hist = bull_hist[-BULL_HIST_LEN:]

        for K in TIER2_STRIKES:
            r = snap.get(f'VEV_{K}')
            if not r or r['bid_1'] is None or r['ask_1'] is None:
                continue
            mid = (r['bid_1'] + r['ask_1']) / 2
            hist = mid_hist[K]
            hist.append(mid)
            if len(hist) > MID_HIST_LEN:
                hist = hist[-MID_HIST_LEN:]
                mid_hist[K] = hist
            if len(hist) >= LOOKBACK:
                peak = detect_peak(bull_hist, hist)
                dip = detect_dip(bull_hist, hist)
                rank = sum(1 for x in hist if x <= mid) / len(hist)
                pctl = (rank <= DIP_PCTL) and (bull > 0)
                flags[(ts, K)] = {'peak': peak, 'dip': dip, 'pctl': pctl}
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', type=Path)
    ap.add_argument('-o', '--out', type=Path, default=None)
    args = ap.parse_args()

    with open(args.log) as f:
        data = json.load(f)

    book = parse_activities(data['activitiesLog'])
    timestamps = sorted(book.keys())
    flags = replay_flags(book, timestamps)

    # Walk SUBMISSION T2 trades in order; assign detector via FIFO budget per
    # (ts, product, side). The matching engine fills orders in submission
    # order, so the dip order (submitted first) consumes its budget before
    # the pctl order.
    #
    # Positions reset at each day boundary (ts = 1_000_000, 2_000_000) — the
    # Prosperity engine zeroes positions between days even though timestamps
    # keep climbing.
    DAY_LEN = 1_000_000
    rows = []
    pos = defaultdict(int)
    cur_day = 0
    tick_alloc = {}
    n_unknown = 0

    for trade in data['tradeHistory']:
        sym = trade['symbol']
        if not sym.startswith('VEV_'):
            continue
        try:
            K = int(sym.split('_')[1])
        except (IndexError, ValueError):
            continue
        if K not in TIER2_STRIKES:
            continue
        if trade.get('buyer') == 'SUBMISSION':
            side = 'BUY'
            signed = trade['quantity']
        elif trade.get('seller') == 'SUBMISSION':
            side = 'SELL'
            signed = -trade['quantity']
        else:
            continue

        ts = trade['timestamp']
        qty = trade['quantity']
        price = trade['price']

        day = ts // DAY_LEN
        if day != cur_day:
            pos.clear()
            cur_day = day

        f_now = flags.get((ts, K), {'peak': False, 'dip': False, 'pctl': False})
        key = (ts, sym, side)
        if key not in tick_alloc:
            if side == 'BUY':
                tick_alloc[key] = {
                    'dip_left': DIP_QTY if f_now['dip'] else 0,
                    'pctl_left': TIER2_QUOTE_QTY if f_now['pctl'] else 0,
                }
            else:
                tick_alloc[key] = {
                    'peak_left': PEAK_QTY if f_now['peak'] else 0,
                }
        alloc = tick_alloc[key]

        if side == 'BUY':
            if alloc['dip_left'] > 0:
                detector = 'E2_dip'
                alloc['dip_left'] = max(0, alloc['dip_left'] - qty)
            elif alloc['pctl_left'] > 0:
                detector = 'pctl_dip+bull'
                alloc['pctl_left'] = max(0, alloc['pctl_left'] - qty)
            else:
                detector = '?'
                n_unknown += 1
        else:
            if alloc['peak_left'] > 0:
                detector = 'E1_peak'
                alloc['peak_left'] = max(0, alloc['peak_left'] - qty)
            else:
                detector = '?'
                n_unknown += 1

        pos_before = pos[sym]
        pos_after = pos_before + signed
        pos[sym] = pos_after

        rec_now = book.get(ts, {}).get(sym)
        mid_at_fill = rec_now['mid'] if rec_now and rec_now.get('mid') is not None else ''
        rec_fwd = book.get(ts + LOOKAHEAD_TS, {}).get(sym)
        mid_fwd = rec_fwd['mid'] if rec_fwd and rec_fwd.get('mid') is not None else ''

        rows.append({
            'timestamp': ts,
            'strike': K,
            'side': side,
            'price': price,
            'qty': qty,
            'which_detector': detector,
            'position_before': pos_before,
            'position_after': pos_after,
            'mid_at_fill': mid_at_fill,
            'mid_at_+20_ticks': mid_fwd,
        })

    out = args.out or args.log.with_suffix('.t2_fills.csv')
    fields = ['timestamp', 'strike', 'side', 'price', 'qty', 'which_detector',
              'position_before', 'position_after', 'mid_at_fill', 'mid_at_+20_ticks']
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f'wrote {len(rows)} rows to {out}')
    print(f'  unknown detector: {n_unknown}')
    from collections import Counter
    det_counts = Counter(r['which_detector'] for r in rows)
    for k, v in det_counts.most_common():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
