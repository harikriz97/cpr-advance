"""
61_new_strategies_backtest.py
Test 2 new strategies using existing NIFTY tick data:

Strategy A — 9:20 Candle Contra Sell
  - First 5-min candle (9:15–9:20) direction
  - Bullish first candle → sell ATM CE (fade the move)
  - Bearish first candle → sell ATM PE
  - Entry: 9:20:02
  - Target: EP * 0.75 (25% profit for seller)
  - Hard SL: EP * 2.0
  - EOD: 15:20

Strategy B — ORB ATM Option Buy
  - Opening Range = first 15-min (9:15–9:30) high/low
  - Spot 1-min candle closes ABOVE ORB high → buy ATM CE
  - Spot 1-min candle closes BELOW ORB low  → buy ATM PE
  - Entry: next candle open + 2s
  - Target: EP * 2.0 (option doubles = 100% gain)
  - SL: EP * 0.6 (40% loss)
  - EOD: 15:00 (option buying — exit early)
"""
import sys, os, re
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
from plot_util import send_custom_chart
from my_util import list_trading_dates, load_spot_data

DATA_ROOT = os.environ['INTER_SERVER_DATA_PATH']
LOT_SIZE  = 75

def r2(v): return round(float(v), 2)

# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────
def find_option_file(date_str, strike, opt_type, date_folder):
    pattern = re.compile(rf'^NIFTY(\d{{6}}){strike}({opt_type})\.csv$', re.IGNORECASE)
    matches = []
    for f in os.listdir(date_folder):
        m = pattern.match(f)
        if m:
            try:
                exp_dt   = datetime.strptime('20' + m.group(1), '%Y%m%d')
                trade_dt = datetime.strptime(date_str, '%Y%m%d')
                dte = (exp_dt - trade_dt).days
                if dte >= 0:
                    matches.append((dte, f))
            except Exception:
                continue
    if not matches:
        return None
    matches.sort()
    return os.path.join(date_folder, matches[0][1])

def load_option(path):
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path, header=None, names=['date','time','price','vol','oi'])

def get_atm(spot_price, step=50):
    return int(round(spot_price / step) * step)

def find_opt(date_str, date_folder, spot_price, opt_type):
    strike = get_atm(spot_price)
    for adj in [0, step := 50, -step, 2*step, -2*step]:
        path = find_option_file(date_str, strike + adj, opt_type, date_folder)
        if path:
            return path, strike + adj
    return None, None

def simulate_exit_sell(opt_tks, entry_time, ep, target_mult, sl_mult, eod_str):
    """For option selling: target = price falls, SL = price rises."""
    target  = r2(ep * target_mult)   # e.g. 0.75 × ep
    hard_sl = r2(ep * sl_mult)       # e.g. 2.0 × ep
    after   = opt_tks[opt_tks['time'] >= entry_time]
    for _, row in after.iterrows():
        cp = float(row['price']); t = row['time']
        if t > eod_str:
            break
        if cp <= target:
            return r2(cp), t, 'target'
        if cp >= hard_sl:
            return r2(cp), t, 'hard_sl'
    eod = opt_tks[opt_tks['time'] <= eod_str]
    if eod.empty:
        return None, None, None
    return r2(float(eod['price'].iloc[-1])), eod.iloc[-1]['time'], 'eod'

def simulate_exit_buy(opt_tks, entry_time, ep, target_mult, sl_mult, eod_str):
    """For option buying: target = price rises, SL = price falls."""
    target  = r2(ep * target_mult)   # e.g. 2.0 × ep (doubled)
    hard_sl = r2(ep * sl_mult)       # e.g. 0.6 × ep (40% loss)
    after   = opt_tks[opt_tks['time'] >= entry_time]
    for _, row in after.iterrows():
        cp = float(row['price']); t = row['time']
        if t > eod_str:
            break
        if cp >= target:
            return r2(cp), t, 'target'
        if cp <= hard_sl:
            return r2(cp), t, 'hard_sl'
    eod = opt_tks[opt_tks['time'] <= eod_str]
    if eod.empty:
        return None, None, None
    return r2(float(eod['price'].iloc[-1])), eod.iloc[-1]['time'], 'eod'

def simulate_exit(opt_tks, entry_time, ep, target_mult, sl_mult, eod_str):
    return simulate_exit_sell(opt_tks, entry_time, ep, target_mult, sl_mult, eod_str)

# ═══════════════════════════════════════════════════════
# STRATEGY A — 9:20 Candle Contra Sell
# ═══════════════════════════════════════════════════════
def strategy_a_day(date_str):
    folder = os.path.join(DATA_ROOT, date_str)
    if not os.path.isdir(folder):
        return None
    spot = load_spot_data(date_str, 'NIFTY')
    if spot is None or spot.empty:
        return None

    # first 5-min candle: 9:15:00–9:19:59
    candle = spot[(spot['time'] >= '09:15:00') & (spot['time'] < '09:20:00')]
    if candle.empty:
        return None
    c_open  = float(candle['price'].iloc[0])
    c_close = float(candle['price'].iloc[-1])
    c_size  = abs(c_close - c_open)

    # skip doji (candle body < 5 points — noise)
    if c_size < 5:
        return None

    bullish   = c_close > c_open
    opt_type  = 'CE' if bullish else 'PE'   # contra: sell opposite

    # entry spot price at 9:20
    spot_920 = spot[spot['time'] >= '09:20:00']
    if spot_920.empty:
        return None
    spot_price = float(spot_920['price'].iloc[0])
    opt_path, strike = find_opt(date_str, folder, spot_price, opt_type)
    if opt_path is None:
        return None

    opt_tks = load_option(opt_path)
    if opt_tks is None or opt_tks.empty:
        return None

    entry_row = opt_tks[opt_tks['time'] >= '09:20:02']
    if entry_row.empty:
        return None
    ep         = r2(float(entry_row['price'].iloc[0]))
    entry_time = entry_row.iloc[0]['time']
    if ep <= 0:
        return None

    xp, exit_time, exit_reason = simulate_exit(
        opt_tks, entry_time, ep,
        target_mult=0.75, sl_mult=2.0, eod_str='15:20:00'
    )
    if xp is None:
        return None

    return {
        'date': date_str, 'strategy': '9:20_contra',
        'direction': 'sell_ce' if opt_type == 'CE' else 'sell_pe',
        'candle_body': r2(c_size), 'bullish_candle': bullish,
        'strike': strike, 'ep': ep, 'xp': xp,
        'entry_time': entry_time, 'exit_time': exit_time,
        'exit_reason': exit_reason,
        'pnl': r2((ep - xp) * LOT_SIZE)
    }

# ═══════════════════════════════════════════════════════
# STRATEGY B — ORB ATM Option Buy (15-min)
# ═══════════════════════════════════════════════════════
def strategy_b_day(date_str):
    folder = os.path.join(DATA_ROOT, date_str)
    if not os.path.isdir(folder):
        return None
    spot = load_spot_data(date_str, 'NIFTY')
    if spot is None or spot.empty:
        return None

    # Opening Range: 9:15–9:30
    orb = spot[(spot['time'] >= '09:15:00') & (spot['time'] < '09:30:00')]
    if orb.empty:
        return None
    orb_high = float(orb['price'].max())
    orb_low  = float(orb['price'].min())
    orb_rng  = r2(orb_high - orb_low)

    # skip very narrow ORB (< 20 pts) — no real range
    if orb_rng < 20:
        return None

    # scan 1-min closes after 9:30 for breakout
    # resample to 1-min OHLC
    spot['dt'] = pd.to_datetime(date_str + ' ' + spot['time'])
    spot_1m    = spot.set_index('dt')['price'].resample('1min').ohlc().dropna()
    after_orb  = spot_1m[spot_1m.index.time >= pd.Timestamp('09:30:00').time()]

    signal     = None
    signal_dt  = None
    for dt, row in after_orb.iterrows():
        if row['close'] > orb_high:
            signal    = 'buy_ce'
            signal_dt = dt
            break
        elif row['close'] < orb_low:
            signal    = 'buy_pe'
            signal_dt = dt
            break

    if signal is None:
        return None

    # entry at next candle open + 2s
    entry_time_str = (signal_dt + pd.Timedelta(minutes=1, seconds=2)).strftime('%H:%M:%S')

    # only enter before 11:30 (late breakouts have less momentum)
    if entry_time_str > '11:30:00':
        return None

    opt_type = 'CE' if signal == 'buy_ce' else 'PE'
    spot_at_entry = spot[spot['time'] >= entry_time_str]
    if spot_at_entry.empty:
        return None
    spot_price = float(spot_at_entry['price'].iloc[0])

    opt_path, strike = find_opt(date_str, folder, spot_price, opt_type)
    if opt_path is None:
        return None

    opt_tks = load_option(opt_path)
    if opt_tks is None or opt_tks.empty:
        return None

    entry_row = opt_tks[opt_tks['time'] >= entry_time_str]
    if entry_row.empty:
        return None
    ep         = r2(float(entry_row['price'].iloc[0]))
    entry_time = entry_row.iloc[0]['time']
    if ep <= 0:
        return None

    # for buying: target = 2x, SL = 0.6x (40% loss), EOD = 15:00
    xp, exit_time, exit_reason = simulate_exit_buy(
        opt_tks, entry_time, ep,
        target_mult=2.0, sl_mult=0.6, eod_str='15:00:00'
    )
    if xp is None:
        return None

    return {
        'date': date_str, 'strategy': 'orb_buy',
        'direction': signal, 'orb_range': orb_rng,
        'strike': strike, 'ep': ep, 'xp': xp,
        'entry_time': entry_time, 'exit_time': exit_time,
        'exit_reason': exit_reason,
        'pnl': r2((xp - ep) * LOT_SIZE)   # buying: xp - ep
    }

# ─────────────────────────────────────────────
# Run both backtests
# ─────────────────────────────────────────────
def run_all():
    dates  = list_trading_dates()
    a_trades, b_trades = [], []
    for d in dates:
        try:
            r = strategy_a_day(d)
            if r: a_trades.append(r)
        except Exception:
            pass
        try:
            r = strategy_b_day(d)
            if r: b_trades.append(r)
        except Exception:
            pass
    return pd.DataFrame(a_trades), pd.DataFrame(b_trades)

print("Running Strategy A (9:20 Contra Sell) + Strategy B (ORB Buy)...")
df_a, df_b = run_all()

# ─────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────
def print_results(df, name):
    if df.empty:
        print(f"{name}: No trades"); return
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['year']    = df['date_dt'].dt.year
    n   = len(df)
    wr  = r2((df.pnl > 0).mean() * 100)
    pnl = r2(df.pnl.sum())
    avg = r2(df.pnl.mean())
    eq  = df.sort_values('date_dt').pnl.cumsum()
    mdd = r2((eq - eq.cummax()).min())
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  Trades    : {n}")
    print(f"  Win Rate  : {wr}%")
    print(f"  Total P&L : Rs {pnl:,.0f}")
    print(f"  Avg P&L   : Rs {avg:,.0f}")
    print(f"  Max DD    : Rs {mdd:,.0f}")
    print(f"\n  Exit reasons:")
    for reason, g in df.groupby('exit_reason'):
        print(f"    {reason:<10} {len(g):>4} | WR {(g.pnl>0).mean()*100:.1f}% | Rs {g.pnl.sum():>10,.0f}")
    print(f"\n  Direction:")
    for d, g in df.groupby('direction'):
        print(f"    {d:<12} {len(g):>4} | WR {(g.pnl>0).mean()*100:.1f}% | Rs {g.pnl.sum():>10,.0f}")
    print(f"\n  Year-wise:")
    for yr, g in df.groupby('year'):
        print(f"    {yr}  {len(g):>4} | WR {(g.pnl>0).mean()*100:.1f}% | Rs {g.pnl.sum():>10,.0f}")

print_results(df_a, "Strategy A — 9:20 Candle Contra Sell")
print_results(df_b, "Strategy B — ORB ATM Option Buy (15-min)")

# ─────────────────────────────────────────────
# Save CSVs
# ─────────────────────────────────────────────
df_a.to_csv('data/20260428/61a_9x20_contra_trades.csv', index=False)
df_b.to_csv('data/20260428/61b_orb_buy_trades.csv', index=False)
print("\nCSVs saved.")

# ─────────────────────────────────────────────
# Charts: equity curves for both + comparison
# ─────────────────────────────────────────────
def make_equity_line(df, label, color):
    df = df.sort_values('date_dt').reset_index(drop=True)
    eq = df['pnl'].cumsum()
    return {
        'id': label.lower().replace(' ','_').replace(':',''),
        'label': label, 'color': color,
        'data': [{'time': int(r.date_dt.timestamp()), 'value': round(eq[i],2)}
                 for i, (_, r) in enumerate(df.iterrows())]
    }

if not df_a.empty:
    df_a['date_dt'] = pd.to_datetime(df_a['date'], format='%Y%m%d')
if not df_b.empty:
    df_b['date_dt'] = pd.to_datetime(df_b['date'], format='%Y%m%d')

# v17a for comparison
v17a = pd.read_csv('data/56_combined_trades.csv', parse_dates=['date'])
v17a = v17a[v17a.strategy=='v17a'].copy()
v17a['date_dt'] = v17a['date']
v17a_eq = v17a.sort_values('date_dt').pnl.cumsum()
v17a_line = {
    'id': 'v17a', 'label': 'v17a CPR+EMA (sell)', 'color': '#26a69a',
    'data': [{'time': int(pd.Timestamp(r.date).timestamp()), 'value': round(eq,2)}
             for (_, r), eq in zip(v17a.sort_values('date_dt').iterrows(), v17a_eq)]
}

lines = [v17a_line]
if not df_a.empty:
    lines.append(make_equity_line(df_a, '9:20 Contra Sell', '#4BC0C0'))
if not df_b.empty:
    lines.append(make_equity_line(df_b, 'ORB Buy', '#f59e0b'))

tv = {'lines': lines, 'candlestick': [], 'volume': [], 'isTvFormat': False}
send_custom_chart('strategy_comparison', tv,
                  title='Strategy Comparison — v17a vs 9:20 Contra vs ORB Buy')
print("📊 Comparison equity chart sent")

# Individual equity + DD for Strategy A
if not df_a.empty:
    df_as = df_a.sort_values('date_dt').reset_index(drop=True)
    eq_a  = df_as.pnl.cumsum()
    dd_a  = eq_a - eq_a.cummax()
    tv_a  = {
        'lines': [
            {'id':'eq_a','label':'Equity','data':[{'time':int(r.date_dt.timestamp()),'value':round(eq_a[i],2)} for i,(_, r) in enumerate(df_as.iterrows())],'seriesType':'baseline','baseValue':0},
            {'id':'dd_a','label':'Drawdown','data':[{'time':int(r.date_dt.timestamp()),'value':round(dd_a[i],2)} for i,(_, r) in enumerate(df_as.iterrows())],'seriesType':'baseline','baseValue':0,'isNewPane':True},
        ],
        'candlestick':[],'volume':[],'isTvFormat':False,
    }
    send_custom_chart('9x20_equity', tv_a,
                      title=f'9:20 Contra Sell — {len(df_as)} trades | WR {r2((df_as.pnl>0).mean()*100)}% | P&L Rs {r2(df_as.pnl.sum()):,.0f}')
    print("📊 9:20 contra equity chart sent")

# Individual equity + DD for Strategy B
if not df_b.empty:
    df_bs = df_b.sort_values('date_dt').reset_index(drop=True)
    eq_b  = df_bs.pnl.cumsum()
    dd_b  = eq_b - eq_b.cummax()
    tv_b  = {
        'lines': [
            {'id':'eq_b','label':'Equity','data':[{'time':int(r.date_dt.timestamp()),'value':round(eq_b[i],2)} for i,(_, r) in enumerate(df_bs.iterrows())],'seriesType':'baseline','baseValue':0},
            {'id':'dd_b','label':'Drawdown','data':[{'time':int(r.date_dt.timestamp()),'value':round(dd_b[i],2)} for i,(_, r) in enumerate(df_bs.iterrows())],'seriesType':'baseline','baseValue':0,'isNewPane':True},
        ],
        'candlestick':[],'volume':[],'isTvFormat':False,
    }
    send_custom_chart('orb_buy_equity', tv_b,
                      title=f'ORB ATM Buy — {len(df_bs)} trades | WR {r2((df_bs.pnl>0).mean()*100)}% | P&L Rs {r2(df_bs.pnl.sum()):,.0f}')
    print("📊 ORB buy equity chart sent")

print("\nAll done!")
