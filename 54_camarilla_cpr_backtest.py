"""
Camarilla R3/S3 inside CPR Zone — Backtest
============================================
Concept: When Camarilla H3 falls inside CPR zone (BC..TC) → strong resistance → sell CE
         When Camarilla L3 falls inside CPR zone (BC..TC) → strong support   → sell PE

Two passes:
  Pass 1 — Standalone: trade ONLY on Camarilla-CPR alignment days (new signal)
  Pass 2 — Filter on v17a: check if this alignment improves WR on existing trade days

Camarilla levels (from prev day H, L, C):
  H4 = C + (H−L) × 1.1/2     L4 = C − (H−L) × 1.1/2
  H3 = C + (H−L) × 1.1/4     L3 = C − (H−L) × 1.1/4
  H2 = C + (H−L) × 1.1/6     L2 = C − (H−L) × 1.1/6
  H1 = C + (H−L) × 1.1/12    L1 = C − (H−L) × 1.1/12
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np

OUT_DIR   = 'data/20260428'
LOT_SIZE  = 75
STRIKE_INT= 50
EMA_PERIOD= 20
EOD_EXIT  = '15:20:00'
YEARS     = 5

os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp = r2((h+l+c)/3); bc = r2((h+l)/2); tc = r2(2*pp-bc)
    r1 = r2(2*pp-l);   r2_ = r2(pp+(h-l));  r3 = r2(r1+(h-l)); r4 = r2(r2_+(h-l))
    s1 = r2(2*pp-h);   s2_ = r2(pp-(h-l));  s3 = r2(s1-(h-l)); s4 = r2(s2_-(h-l))
    return dict(pp=pp,bc=bc,tc=tc,r1=r1,r2=r2_,r3=r3,r4=r4,s1=s1,s2=s2_,s3=s3,s4=s4)

def compute_camarilla(h, l, c):
    rng = h - l
    return dict(
        h4=r2(c + rng*1.1/2), h3=r2(c + rng*1.1/4),
        h2=r2(c + rng*1.1/6), h1=r2(c + rng*1.1/12),
        l1=r2(c - rng*1.1/12),l2=r2(c - rng*1.1/6),
        l3=r2(c - rng*1.1/4), l4=r2(c - rng*1.1/2),
    )

def inside_cpr(level, cpr_lo, cpr_hi):
    return cpr_lo <= level <= cpr_hi

def get_strike(atm, opt_type, stype):
    if opt_type == 'CE':
        return {'OTM1':atm+STRIKE_INT,'ATM':atm,'ITM1':atm-STRIKE_INT}[stype]
    return {'OTM1':atm-STRIKE_INT,'ATM':atm,'ITM1':atm+STRIKE_INT}[stype]

def sim_pct(ts, ps, ep, eod_ns, tgt_pct, sl_pct):
    tgt=r2(ep*(1-tgt_pct)); hsl=r2(ep*(1+sl_pct)); sl=hsl; md=0.0
    for i in range(len(ts)):
        t=ts[i]; p=ps[i]
        if t>=eod_ns: return r2((ep-p)*LOT_SIZE),'eod',r2(p)
        d=(ep-p)/ep
        if d>md: md=d
        if   md>=0.60: sl=min(sl,r2(ep*(1-md*0.95)))
        elif md>=0.40: sl=min(sl,r2(ep*0.80))
        elif md>=0.25: sl=min(sl,ep)
        if p<=tgt: return r2((ep-p)*LOT_SIZE),'target',r2(p)
        if p>=sl:  return r2((ep-p)*LOT_SIZE),'lockin_sl' if sl<hsl else 'hard_sl',r2(p)
    return r2((ep-ps[-1])*LOT_SIZE),'eod',r2(ps[-1])

# ── Pass 0: Load OHLC + EMA ──────────────────────────────────────────
print(f"Loading NIFTY OHLC + EMA({EMA_PERIOD}) (5yr)...")
t0 = time.time()
all_dates = list_trading_dates()
latest    = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr = [d for d in all_dates
             if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)

daily_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty: continue
    daily_ohlc[d] = (
        float(tks['price'].max()),
        float(tks['price'].min()),
        float(tks[tks['time'] <= '15:30:00']['price'].iloc[-1]),
        float(tks[tks['time'] >= '09:15:00']['price'].iloc[0]),
    )

close_s = pd.Series({d: v[2] for d, v in daily_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD, adjust=False).mean().shift(1)
print(f"  {len(daily_ohlc)} days in {time.time()-t0:.0f}s")

# ── Pass 1: Standalone Camarilla-CPR alignment signal ────────────────
print("\nPass 1: Standalone Camarilla H3/L3 inside CPR (grid search)...")
t1 = time.time()

ENTRY_TIMES  = ['09:16:02','09:20:02','09:25:02','09:31:02']
STRIKE_TYPES = ['ATM','OTM1','ITM1']
TARGETS      = [0.20, 0.30, 0.40, 0.50]
SL_MULTS     = [0.50, 1.00, 1.50, 2.00]
BODY_MIN     = 0.10

standalone_records = []
day_counts = {'H3_in_CPR': 0, 'L3_in_CPR': 0, 'both': 0}

for date in dates_5yr:
    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx-1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph, pl, pc, _ = daily_ohlc[prev]
    _, _, _, today_op = daily_ohlc[date]

    # Body filter
    prev_open = daily_ohlc[prev][3]
    prev_body = round(abs(pc - prev_open) / prev_open * 100, 3)
    if prev_body <= BODY_MIN: continue

    pvt  = compute_pivots(ph, pl, pc)
    cam  = compute_camarilla(ph, pl, pc)
    e20  = ema_s.get(date, np.nan)
    if np.isnan(e20): continue

    cpr_lo = min(pvt['tc'], pvt['bc'])
    cpr_hi = max(pvt['tc'], pvt['bc'])

    h3_in = inside_cpr(cam['h3'], cpr_lo, cpr_hi)
    l3_in = inside_cpr(cam['l3'], cpr_lo, cpr_hi)

    if not h3_in and not l3_in: continue

    # Determine signal:
    # H3 inside CPR → resistance zone → sell CE (bearish reversal expected)
    # L3 inside CPR → support zone   → sell PE (bullish reversal expected)
    # EMA bias confirms direction
    bias = 'bull' if today_op > e20 else 'bear'

    signals = []
    if h3_in and bias == 'bear': signals.append(('CE', 'H3_in_CPR', cam['h3']))
    if l3_in and bias == 'bull': signals.append(('PE', 'L3_in_CPR', cam['l3']))
    if not signals: continue

    if h3_in: day_counts['H3_in_CPR'] += 1
    if l3_in: day_counts['L3_in_CPR'] += 1
    if h3_in and l3_in: day_counts['both'] += 1

    dstr   = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    exps   = list_expiry_dates(date)
    if not exps: continue
    expiry = exps[0]
    exp_dt = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte    = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: continue

    atm    = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    for opt, setup_type, cam_level in signals:
        for et in ENTRY_TIMES:
            for stype in STRIKE_TYPES:
                strike = get_strike(atm, opt, stype)
                inst   = f'NIFTY{expiry}{strike}{opt}'
                tk     = load_tick_data(date, inst, et[:5]+':00', EOD_EXIT)
                if tk is None or tk.empty: continue
                ep_mask = tk['time'] >= et
                if not ep_mask.any(): continue
                ep = r2(float(tk[ep_mask].iloc[0]['price']))
                if ep < 5 or ep / today_op * 100 <= 0.47: continue
                opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
                opt_ps = tk[ep_mask]['price'].values.astype(float)
                for tgt in TARGETS:
                    for slm in SL_MULTS:
                        pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt, slm)
                        standalone_records.append(dict(
                            date=dstr, setup=setup_type, opt=opt, bias=bias,
                            cam_level=cam_level, cpr_lo=cpr_lo, cpr_hi=cpr_hi,
                            strike_type=stype, entry_time=et,
                            ep=ep, xp=xp, exit_reason=reason, pnl=pnl,
                            target_pct=tgt, sl_pct=slm, dte=dte
                        ))

print(f"  Done {time.time()-t1:.0f}s — {day_counts} — {len(standalone_records)} records")

# ── Pass 2: Filter on v17a existing trades ───────────────────────────
print("\nPass 2: Camarilla-CPR alignment as filter on v17a trades...")
v17a = pd.read_csv('data/20260420/38_zone_v17a_trades.csv', parse_dates=['date'])
v17a['date_str'] = v17a['date'].dt.strftime('%Y%m%d')

cam_flag = []
for _, row in v17a.iterrows():
    date = row['date_str']
    idx  = all_dates.index(date) if date in all_dates else -1
    if idx < 1 or all_dates[idx-1] not in daily_ohlc:
        cam_flag.append(False); continue
    prev = all_dates[idx-1]
    ph, pl, pc, _ = daily_ohlc[prev]
    pvt  = compute_pivots(ph, pl, pc)
    cam  = compute_camarilla(ph, pl, pc)
    cpr_lo = min(pvt['tc'], pvt['bc'])
    cpr_hi = max(pvt['tc'], pvt['bc'])
    flag = inside_cpr(cam['h3'], cpr_lo, cpr_hi) or inside_cpr(cam['l3'], cpr_lo, cpr_hi)
    cam_flag.append(flag)

v17a['cam_cpr_align'] = cam_flag

# ── Print results ────────────────────────────────────────────────────
def stats(df, label):
    if df.empty: print(f"  {label}: 0 trades"); return
    n = len(df); wr = round((df.pnl>0).mean()*100,1)
    avg = round(df.pnl.mean(),0); total = round(df.pnl.sum(),0)
    pts = round(total/5/52/LOT_SIZE, 1)
    print(f"  {label}: n={n}  WR={wr}%  avg=₹{avg:,.0f}  total=₹{total:,.0f}/5yr  {pts}pts/wk")

print("\n" + "="*70)
print("PASS 1 — Standalone Camarilla-CPR Signal")
print("="*70)
if standalone_records:
    df_s = pd.DataFrame(standalone_records)
    grp  = df_s.groupby(['setup','strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1),
        avg=lambda x: round(x.mean(),0), total=lambda x: round(x.sum(),0)
    ).reset_index()
    for setup in ['H3_in_CPR','L3_in_CPR']:
        sg = grp[(grp.setup==setup)&(grp.n>=8)].sort_values('total',ascending=False)
        if sg.empty: print(f"\n  {setup}: no valid config"); continue
        print(f"\n  {setup} — Top 3 configs:")
        for _, b in sg.head(3).iterrows():
            print(f"    {b.strike_type} {b.entry_time} tgt={b.target_pct:.0%} sl={b.sl_pct}x"
                  f"  → n={int(b.n)} WR={b.wr}% avg=₹{b.avg:,.0f} total=₹{b.total:,.0f}/5yr")

print("\n" + "="*70)
print("PASS 2 — Camarilla-CPR as Filter on v17a Trades")
print("="*70)
aligned   = v17a[v17a['cam_cpr_align'] == True]
unaligned = v17a[v17a['cam_cpr_align'] == False]
stats(v17a,      "Baseline  (all)")
stats(aligned,   "Cam-CPR align (IN) ")
stats(unaligned, "No align (OUT)")
wr_diff = round((aligned.pnl>0).mean()*100 - (unaligned.pnl>0).mean()*100, 1)
print(f"\n  WR diff (aligned − unaligned): {wr_diff:+.1f}%  "
      f"{'✅ Filter works' if wr_diff > 3 else '❌ Weak edge'}")

# Best standalone config
print("\n" + "="*70)
print("BEST STANDALONE CONFIG PROJECTION")
print("="*70)
if standalone_records:
    df_s = pd.DataFrame(standalone_records)
    grp_all = df_s.groupby(['strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1),
        avg=lambda x: round(x.mean(),0), total=lambda x: round(x.sum(),0)
    ).reset_index()
    best = grp_all[grp_all.n >= 10].sort_values('total', ascending=False).head(5)
    print("  Top 5 configs across all setups:")
    for _, b in best.iterrows():
        print(f"    {b.strike_type} {b.entry_time} tgt={b.target_pct:.0%} sl={b.sl_pct}x"
              f"  → n={int(b.n)} WR={b.wr}% avg=₹{b.avg:,.0f} total=₹{b.total:,.0f}/5yr  "
              f"{round(b.total/5/52/LOT_SIZE,1)}pts/wk")

# Save
df_out = pd.DataFrame(standalone_records) if standalone_records else pd.DataFrame()
if not df_out.empty:
    df_out.to_csv(f'{OUT_DIR}/54_camarilla_standalone.csv', index=False)
v17a.to_csv(f'{OUT_DIR}/54_v17a_with_cam_flag.csv', index=False)
print(f"\nSaved → {OUT_DIR}/54_camarilla_standalone.csv + 54_v17a_with_cam_flag.csv")
