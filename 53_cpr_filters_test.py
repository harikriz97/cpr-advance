"""
CPR Filters Test — Virgin CPR, CPR Width, Weekly CPR Alignment
==============================================================
Tests 3 new filters on existing v17a trades (465 trades / 5yr):
  Filter 1: Virgin CPR  — prev day never touched CPR
  Filter 2: CPR Width   — wide CPR (>= threshold) = range day = better for selling
  Filter 3: Weekly CPR  — weekly CPR bias aligns with EMA daily bias

For each filter: shows WR, avg P&L, total P&L on IN subset vs OUT subset.
Goal: find if any filter reliably improves WR on subset of days.
"""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, list_trading_dates
import pandas as pd, numpy as np

OUT_DIR = 'data/20260428'
os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp = r2((h+l+c)/3); bc = r2((h+l)/2); tc = r2(2*pp-bc)
    r1 = r2(2*pp-l);   r2_ = r2(pp+(h-l));  r3 = r2(r1+(h-l)); r4 = r2(r2_+(h-l))
    s1 = r2(2*pp-h);   s2_ = r2(pp-(h-l));  s3 = r2(s1-(h-l)); s4 = r2(s2_-(h-l))
    return dict(pp=pp, bc=bc, tc=tc, r1=r1, r2=r2_, r3=r3, r4=r4,
                s1=s1, s2=s2_, s3=s3, s4=s4)

def stats(df):
    if df.empty: return 0, 0.0, 0.0, 0.0
    n = len(df)
    wr = round((df.pnl > 0).mean() * 100, 1)
    avg = round(df.pnl.mean(), 0)
    total = round(df.pnl.sum(), 0)
    return n, wr, avg, total

def pts_wk(total_5yr):
    return round(total_5yr / 5 / 52 / 75, 1)

# ── Load v17a trades ────────────────────────────────────────────────
trades = pd.read_csv('data/20260420/38_zone_v17a_trades.csv', parse_dates=['date'])
trades['date_str'] = trades['date'].dt.strftime('%Y%m%d')
print(f"Loaded {len(trades)} v17a trades\n")

# ── Build daily OHLC (5yr + seed) ──────────────────────────────────
print("Loading daily OHLC for all trade dates + prev days...")
all_dates = list_trading_dates()
trade_dates = sorted(trades['date_str'].unique().tolist())
# Need prev day for each trade date + prev week for weekly CPR
needed = set()
for d in trade_dates:
    idx = all_dates.index(d) if d in all_dates else -1
    if idx >= 1:  needed.add(all_dates[idx-1])
    if idx >= 5:  needed.add(all_dates[idx-5])   # prev week approx
    if idx >= 6:  needed.add(all_dates[idx-6])
    needed.add(d)
# Also need full prev week (Mon-Fri) for weekly CPR
for d in trade_dates:
    idx = all_dates.index(d) if d in all_dates else -1
    for k in range(1, 8):
        if idx-k >= 0: needed.add(all_dates[idx-k])

needed_sorted = sorted(needed)
daily_ohlc = {}
for d in needed_sorted:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty: continue
    h = float(tks['price'].max())
    l = float(tks['price'].min())
    c = float(tks[tks['time'] <= '15:30:00']['price'].iloc[-1])
    o = float(tks[tks['time'] >= '09:15:00']['price'].iloc[0])
    daily_ohlc[d] = (h, l, c, o)

print(f"  Loaded {len(daily_ohlc)} days\n")

# ── Compute filters for each trade ─────────────────────────────────
virgin_cpr    = []   # True if prev day never touched CPR
cpr_width_val = []   # tc - bc value
weekly_align  = []   # True if weekly CPR bias matches daily EMA bias

for _, row in trades.iterrows():
    date = row['date_str']
    idx  = all_dates.index(date) if date in all_dates else -1

    # ── Filter 1 & 2: need prev day OHLC ──
    if idx >= 1:
        prev = all_dates[idx-1]
        if prev in daily_ohlc:
            ph, pl, pc, po = daily_ohlc[prev]
            pvt = compute_pivots(ph, pl, pc)
            # Virgin CPR: prev day high < bc (all prices below CPR) OR prev day low > tc (all above CPR)
            is_virgin = (pl > pvt['tc']) or (ph < pvt['bc'])
            virgin_cpr.append(is_virgin)
            cpr_width_val.append(r2(pvt['tc'] - pvt['bc']))
        else:
            virgin_cpr.append(np.nan)
            cpr_width_val.append(np.nan)
    else:
        virgin_cpr.append(np.nan)
        cpr_width_val.append(np.nan)

    # ── Filter 3: Weekly CPR alignment ──
    # Collect last 5 trading days before today to build weekly OHLC
    if idx >= 5:
        wk_days = [all_dates[idx-k] for k in range(1,6) if all_dates[idx-k] in daily_ohlc]
        if len(wk_days) >= 3:
            wk_h = max(daily_ohlc[d][0] for d in wk_days)
            wk_l = min(daily_ohlc[d][1] for d in wk_days)
            wk_c = daily_ohlc[wk_days[0]][2]   # most recent day close
            wpvt = compute_pivots(wk_h, wk_l, wk_c)
            # Weekly bias: today's open vs weekly PP
            if date in daily_ohlc:
                today_op = daily_ohlc[date][3]
                w_bias = 'bull' if today_op > wpvt['pp'] else 'bear'
                d_bias = row['ema_bias']
                weekly_align.append(w_bias == d_bias)
            else:
                weekly_align.append(np.nan)
        else:
            weekly_align.append(np.nan)
    else:
        weekly_align.append(np.nan)

trades['virgin_cpr']   = virgin_cpr
trades['cpr_width']    = cpr_width_val
trades['weekly_align'] = weekly_align

# ── Print results ───────────────────────────────────────────────────
print("=" * 70)
print("BASELINE — ALL 465 TRADES")
print("=" * 70)
n, wr, avg, total = stats(trades)
print(f"  n={n}  WR={wr}%  avg=₹{avg:,.0f}  total=₹{total:,.0f}/5yr  {pts_wk(total)} pts/wk")

# Filter 1 — Virgin CPR
print("\n" + "=" * 70)
print("FILTER 1: Virgin CPR (prev day never touched TC/BC)")
print("=" * 70)
v_in  = trades[trades['virgin_cpr'] == True]
v_out = trades[trades['virgin_cpr'] == False]
ni, wri, avgi, toti = stats(v_in)
no, wro, avgo, toto = stats(v_out)
print(f"  Virgin CPR days (IN) : n={ni}  WR={wri}%  avg=₹{avgi:,.0f}  total=₹{toti:,.0f}  {pts_wk(toti)} pts/wk")
print(f"  Non-Virgin days (OUT): n={no}  WR={wro}%  avg=₹{avgo:,.0f}  total=₹{toto:,.0f}  {pts_wk(toto)} pts/wk")
wr_diff = round(wri - wro, 1)
print(f"  WR difference: {wr_diff:+.1f}%  {'✅ Virgin better' if wr_diff > 2 else '❌ No clear edge'}")

# Filter 2 — CPR Width buckets
print("\n" + "=" * 70)
print("FILTER 2: CPR Width (wide = range day = safer for selling)")
print("=" * 70)
valid_w = trades[trades['cpr_width'].notna()]
thresholds = [20, 25, 30, 35, 40]
print(f"  Width distribution: min={valid_w.cpr_width.min():.0f}  median={valid_w.cpr_width.median():.0f}  "
      f"max={valid_w.cpr_width.max():.0f}  p75={valid_w.cpr_width.quantile(0.75):.0f}")
for thr in thresholds:
    wide  = valid_w[valid_w.cpr_width >= thr]
    narrow = valid_w[valid_w.cpr_width < thr]
    nw, wrw, avgw, totw = stats(wide)
    nn2, wrn, avgn, totn = stats(narrow)
    print(f"  Width >= {thr:2d}: n={nw:3d} WR={wrw}%  avg=₹{avgw:,.0f}  | "
          f"Width < {thr:2d}: n={nn2:3d} WR={wrn}%  avg=₹{avgn:,.0f}  "
          f"{'✅' if wrw - wrn > 2 else '  '}")

# Filter 3 — Weekly CPR alignment
print("\n" + "=" * 70)
print("FILTER 3: Weekly CPR Alignment (weekly bias == daily EMA bias)")
print("=" * 70)
wa_in  = trades[trades['weekly_align'] == True]
wa_out = trades[trades['weekly_align'] == False]
ni3, wri3, avgi3, toti3 = stats(wa_in)
no3, wro3, avgo3, toto3 = stats(wa_out)
print(f"  Aligned (IN)    : n={ni3}  WR={wri3}%  avg=₹{avgi3:,.0f}  total=₹{toti3:,.0f}  {pts_wk(toti3)} pts/wk")
print(f"  Misaligned (OUT): n={no3}  WR={wro3}%  avg=₹{avgo3:,.0f}  total=₹{toto3:,.0f}  {pts_wk(toto3)} pts/wk")
wr_diff3 = round(wri3 - wro3, 1)
print(f"  WR difference: {wr_diff3:+.1f}%  {'✅ Alignment helps' if wr_diff3 > 2 else '❌ No clear edge'}")

# Combined Filter 1+2 (Virgin + Wide)
print("\n" + "=" * 70)
print("COMBINED: Virgin CPR AND Wide CPR (>= 25 pts)")
print("=" * 70)
best_w = 25
combined = trades[(trades['virgin_cpr'] == True) & (trades['cpr_width'] >= best_w)]
rest     = trades[~((trades['virgin_cpr'] == True) & (trades['cpr_width'] >= best_w))]
nc, wrc, avgc, totc = stats(combined)
nr2, wrr2, avgr2, totr2 = stats(rest)
print(f"  Virgin + Wide (IN): n={nc}  WR={wrc}%  avg=₹{avgc:,.0f}  total=₹{totc:,.0f}  {pts_wk(totc)} pts/wk")
print(f"  Others (OUT)      : n={nr2}  WR={wrr2}%  avg=₹{avgr2:,.0f}  total=₹{totr2:,.0f}  {pts_wk(totr2)} pts/wk")

# Combined Filter 1+2+3
print("\n" + "=" * 70)
print("COMBINED: Virgin + Wide + Weekly Aligned")
print("=" * 70)
all3 = trades[(trades['virgin_cpr'] == True) & (trades['cpr_width'] >= best_w) & (trades['weekly_align'] == True)]
rest3 = trades[~((trades['virgin_cpr'] == True) & (trades['cpr_width'] >= best_w) & (trades['weekly_align'] == True))]
na3, wra3, avga3, tota3 = stats(all3)
nr3, wrr3, avgr3, totr3 = stats(rest3)
print(f"  All 3 filters (IN): n={na3}  WR={wra3}%  avg=₹{avga3:,.0f}  total=₹{tota3:,.0f}  {pts_wk(tota3)} pts/wk")
print(f"  Others (OUT)      : n={nr3}  WR={wrr3}%  avg=₹{avgr3:,.0f}  total=₹{totr3:,.0f}  {pts_wk(totr3)} pts/wk")

# Save enriched trades
out = f'{OUT_DIR}/53_cpr_filters_test.csv'
trades.to_csv(out, index=False)
print(f"\nSaved → {out}")
