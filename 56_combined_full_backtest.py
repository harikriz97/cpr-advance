"""
Combined Full Backtest — v17a + Camarilla Touch
================================================
Strategy 1 (v17a): CPR zone + EMA(20) bias → sell CE/PE at open time
Strategy 2 (Cam) : Camarilla H3/L3 inside CPR → sell on intraday touch
                   Only on days where v17a does NOT trade (sequential rule)

Camarilla rules (from backtest analysis):
  H3 touch → sell OTM1 CE : tgt=50%  sl=1.0x  skip 09:20–09:30 touch window
  L3 touch → sell ITM1 PE : tgt=20%  sl=0.5x  all windows allowed

Output:
  data/20260428/56_combined_trades.csv
  equity curve chart pushed to chat
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np
from plot_util import plot_equity, send_custom_chart

OUT_DIR    = 'data/20260428'
LOT_SIZE   = 75
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_EXIT   = '15:20:00'
BODY_MIN   = 0.10
YEARS      = 5
os.makedirs(OUT_DIR, exist_ok=True)

# ── v17a best params ──────────────────────────────────────────────────
V17A_PARAMS = {
    ("above_r4",   "bull", "PE"): ("ITM1", "09:31:02", 0.50, 2.00),
    ("below_s4",   "bear", "CE"): ("ITM1", "09:16:02", 0.20, 0.50),
    ("pdh_to_r1",  "bear", "PE"): ("OTM1", "09:20:02", 0.50, 0.50),
    ("pdl_to_bc",  "bull", "PE"): ("OTM1", "09:31:02", 0.20, 1.50),
    ("pdl_to_s1",  "bear", "CE"): ("ITM1", "09:20:02", 0.20, 2.00),
    ("r1_to_r2",   "bear", "PE"): ("ATM",  "09:20:02", 0.50, 2.00),
    ("r1_to_r2",   "bull", "PE"): ("OTM1", "09:16:02", 0.50, 1.00),
    ("r2_to_r3",   "bull", "PE"): ("ATM",  "09:20:02", 0.20, 1.50),
    ("r2_to_r3",   "bear", "PE"): ("ATM",  "09:20:02", 0.20, 1.50),
    ("r3_to_r4",   "bull", "PE"): ("ITM1", "09:25:02", 0.20, 0.50),
    ("s1_to_s2",   "bear", "CE"): ("ATM",  "09:16:02", 0.50, 2.00),
    ("s3_to_s4",   "bear", "CE"): ("ITM1", "09:20:02", 0.40, 0.50),
    ("tc_to_pdh",  "bear", "PE"): ("OTM1", "09:31:02", 0.50, 0.50),
    ("tc_to_pdh",  "bull", "PE"): ("ITM1", "09:25:02", 0.20, 0.50),
    ("within_cpr", "bear", "CE"): ("ATM",  "09:16:02", 0.20, 2.00),
    ("within_cpr", "bull", "PE"): ("ATM",  "09:20:02", 0.30, 2.00),
}

# ── Camarilla params (from analysis) ─────────────────────────────────
CAM_H3 = dict(opt='CE', stype='OTM1', tgt=0.50, sl=1.00)   # H3 touch → sell CE
CAM_L3 = dict(opt='PE', stype='ITM1', tgt=0.20, sl=0.50)   # L3 touch → sell PE
CAM_SKIP_WINDOW = ('09:20:00', '09:30:00')                  # skip H3 touch in this window

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp=r2((h+l+c)/3); bc=r2((h+l)/2); tc=r2(2*pp-bc)
    r1=r2(2*pp-l); r2_=r2(pp+(h-l)); r3=r2(r1+(h-l)); r4=r2(r2_+(h-l))
    s1=r2(2*pp-h); s2_=r2(pp-(h-l)); s3=r2(s1-(h-l)); s4=r2(s2_-(h-l))
    return dict(pp=pp,bc=bc,tc=tc,r1=r1,r2=r2_,r3=r3,r4=r4,s1=s1,s2=s2_,s3=s3,s4=s4)

def compute_camarilla(h, l, c):
    rng = h - l
    return dict(h3=r2(c+rng*1.1/4), l3=r2(c-rng*1.1/4))

def classify_zone(op, pvt, pdh, pdl):
    if   op > pvt['r4']: return 'above_r4'
    elif op > pvt['r3']: return 'r3_to_r4'
    elif op > pvt['r2']: return 'r2_to_r3'
    elif op > pvt['r1']: return 'r1_to_r2'
    elif op > pdh:       return 'pdh_to_r1'
    elif op > pvt['tc']: return 'tc_to_pdh'
    elif op >= pvt['bc']:return 'within_cpr'
    elif op > pdl:       return 'pdl_to_bc'
    elif op > pvt['s1']: return 'pdl_to_s1'
    elif op > pvt['s2']: return 's1_to_s2'
    elif op > pvt['s3']: return 's2_to_s3'
    elif op > pvt['s4']: return 's3_to_s4'
    else:                return 'below_s4'

def get_v17a_signal(zone, bias):
    key = (zone, bias, 'PE')
    if key in V17A_PARAMS: return 'PE', V17A_PARAMS[key]
    key = (zone, bias, 'CE')
    if key in V17A_PARAMS: return 'CE', V17A_PARAMS[key]
    return None, None

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

def detect_touch(spot_tks, level, direction, scan_from='09:16:00', scan_to='15:15:00'):
    tol   = level * 0.05 / 100
    mask  = (spot_tks['time'] >= scan_from) & (spot_tks['time'] <= scan_to)
    scan  = spot_tks[mask].reset_index(drop=True)
    if len(scan) < 2: return None
    for i in range(1, len(scan)):
        price = float(scan.iloc[i]['price'])
        prev  = float(scan.iloc[i-1]['price'])
        t     = scan.iloc[i]['time']
        if direction == 'up'   and prev < level and price >= level - tol: return t, price
        if direction == 'down' and prev > level and price <= level + tol: return t, price
    return None

def add_seconds(hhmm_ss, secs):
    h,m,s = map(int, hhmm_ss.split(':'))
    total = h*3600 + m*60 + s + secs
    return f'{total//3600:02d}:{(total%3600)//60:02d}:{total%60:02d}'


# ═══════════════════════════════════════════════════════════════════
# PASS 0: Load OHLC + EMA
# ═══════════════════════════════════════════════════════════════════
print(f"Pass 0: Loading NIFTY OHLC + EMA({EMA_PERIOD}) ({YEARS}yr)...")
t0 = time.time()
all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest - pd.DateOffset(years=YEARS)]
extra      = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)

daily_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty: continue
    daily_ohlc[d] = (
        float(tks['price'].max()),
        float(tks['price'].min()),
        float(tks[tks['time']<='15:30:00']['price'].iloc[-1]),
        float(tks[tks['time']>='09:15:00']['price'].iloc[0]),
    )

close_s = pd.Series({d:v[2] for d,v in daily_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD, adjust=False).mean().shift(1)
print(f"  {len(daily_ohlc)} days loaded in {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════════
# PASS 1: Main backtest loop
# ═══════════════════════════════════════════════════════════════════
print(f"\nPass 1: Running combined strategy on {len(dates_5yr)} days...")
t1 = time.time()

records = []
counts  = dict(v17a=0, cam_h3=0, cam_l3=0, skip_body=0,
               no_signal=0, no_data=0, dte0=0)

for date in dates_5yr:
    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx-1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph, pl, pc, _ = daily_ohlc[prev]
    _, _, _, today_op = daily_ohlc[date]

    # Body filter
    prev_body = round(abs(pc - daily_ohlc[prev][3]) / daily_ohlc[prev][3] * 100, 3)
    if prev_body <= BODY_MIN:
        counts['skip_body'] += 1
        continue

    pvt  = compute_pivots(ph, pl, pc)
    cam  = compute_camarilla(ph, pl, pc)
    e20  = ema_s.get(date, np.nan)
    if np.isnan(e20): continue

    bias = 'bull' if today_op > e20 else 'bear'
    zone = classify_zone(today_op, pvt, ph, pl)
    dstr = f'{date[:4]}-{date[4:6]}-{date[6:]}'

    exps = list_expiry_dates(date)
    if not exps: counts['no_data'] += 1; continue
    expiry = exps[0]
    exp_dt = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte    = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: counts['dte0'] += 1; continue

    atm    = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    # ── Strategy 1: v17a ────────────────────────────────────────────
    opt, params = get_v17a_signal(zone, bias)
    if opt and params:
        stype, et, tgt, sl_p = params
        strike = get_strike(atm, opt, stype)
        inst   = f'NIFTY{expiry}{strike}{opt}'
        tk     = load_tick_data(date, inst, et[:5]+':00', EOD_EXIT)
        if tk is not None and not tk.empty:
            ep_mask = tk['time'] >= et
            if ep_mask.any():
                ep = r2(float(tk[ep_mask].iloc[0]['price']))
                if ep >= 5 and ep / today_op * 100 > 0.47:
                    opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
                    opt_ps = tk[ep_mask]['price'].values.astype(float)
                    pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt, sl_p)
                    records.append(dict(
                        date=dstr, strategy='v17a', zone=zone, bias=bias,
                        opt=opt, strike_type=stype, entry_time=et,
                        ep=ep, xp=xp, exit_reason=reason, pnl=pnl,
                        cam_level=np.nan, touch_time=''
                    ))
                    counts['v17a'] += 1
                    continue   # sequential: v17a traded → skip Camarilla today

    # ── Strategy 2: Camarilla touch (only if v17a didn't trade) ─────
    cpr_lo = min(pvt['tc'], pvt['bc'])
    cpr_hi = max(pvt['tc'], pvt['bc'])
    h3_in  = cpr_lo <= cam['h3'] <= cpr_hi
    l3_in  = cpr_lo <= cam['l3'] <= cpr_hi

    if not h3_in and not l3_in:
        counts['no_signal'] += 1
        continue

    # Load spot ticks once
    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty:
        counts['no_data'] += 1; continue

    traded_cam = False

    # H3 touch → sell CE
    if h3_in:
        touch = detect_touch(spot_tks, cam['h3'], 'up')
        if touch:
            touch_time, touch_price = touch
            # Skip 09:20–09:30 window for H3
            if CAM_SKIP_WINDOW[0] <= touch_time < CAM_SKIP_WINDOW[1]:
                pass   # skip this touch
            else:
                entry_time = add_seconds(touch_time, 2)
                if entry_time < EOD_EXIT:
                    p = CAM_H3
                    strike = get_strike(atm, p['opt'], p['stype'])
                    inst   = f'NIFTY{expiry}{strike}{p["opt"]}'
                    tk     = load_tick_data(date, inst, touch_time[:5]+':00', EOD_EXIT)
                    if tk is not None and not tk.empty:
                        ep_mask = tk['time'] >= entry_time
                        if ep_mask.any():
                            ep = r2(float(tk[ep_mask].iloc[0]['price']))
                            if ep >= 5:
                                opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
                                opt_ps = tk[ep_mask]['price'].values.astype(float)
                                pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, p['tgt'], p['sl'])
                                records.append(dict(
                                    date=dstr, strategy='cam_h3', zone=zone, bias=bias,
                                    opt=p['opt'], strike_type=p['stype'], entry_time=entry_time,
                                    ep=ep, xp=xp, exit_reason=reason, pnl=pnl,
                                    cam_level=cam['h3'], touch_time=touch_time
                                ))
                                counts['cam_h3'] += 1
                                traded_cam = True

    # L3 touch → sell PE (only if H3 didn't fire today)
    if l3_in and not traded_cam:
        touch = detect_touch(spot_tks, cam['l3'], 'down')
        if touch:
            touch_time, touch_price = touch
            entry_time = add_seconds(touch_time, 2)
            if entry_time < EOD_EXIT:
                p = CAM_L3
                strike = get_strike(atm, p['opt'], p['stype'])
                inst   = f'NIFTY{expiry}{strike}{p["opt"]}'
                tk     = load_tick_data(date, inst, touch_time[:5]+':00', EOD_EXIT)
                if tk is not None and not tk.empty:
                    ep_mask = tk['time'] >= entry_time
                    if ep_mask.any():
                        ep = r2(float(tk[ep_mask].iloc[0]['price']))
                        if ep >= 5:
                            opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
                            opt_ps = tk[ep_mask]['price'].values.astype(float)
                            pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, p['tgt'], p['sl'])
                            records.append(dict(
                                date=dstr, strategy='cam_l3', zone=zone, bias=bias,
                                opt=p['opt'], strike_type=p['stype'], entry_time=entry_time,
                                ep=ep, xp=xp, exit_reason=reason, pnl=pnl,
                                cam_level=cam['l3'], touch_time=touch_time
                            ))
                            counts['cam_l3'] += 1
                            traded_cam = True

    if not traded_cam:
        counts['no_signal'] += 1

elapsed = time.time() - t1
print(f"  Done in {elapsed:.0f}s")
print(f"  {counts}")


# ═══════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════
df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

def stats_block(sub, label):
    if sub.empty: print(f"  {label}: 0 trades"); return
    n   = len(sub)
    wr  = round((sub.pnl > 0).mean()*100, 1)
    avg = round(sub.pnl.mean(), 0)
    tot = round(sub.pnl.sum(), 0)
    mdd = round(sub['pnl'].cumsum().sub(sub['pnl'].cumsum().cummax()).min(), 0)
    pts = round(tot/5/52/LOT_SIZE, 1)
    print(f"  {label:<22}: n={n:4d}  WR={wr:5.1f}%  avg=₹{avg:>7,.0f}  "
          f"total=₹{tot:>10,.0f}/5yr  MDD=₹{mdd:>8,.0f}  {pts}pts/wk")

print("\n" + "="*75)
print("COMBINED STRATEGY RESULTS")
print("="*75)
stats_block(df,                    "ALL COMBINED")
stats_block(df[df.strategy=='v17a'],   "v17a only")
stats_block(df[df.strategy=='cam_h3'], "Cam H3 touch CE")
stats_block(df[df.strategy=='cam_l3'], "Cam L3 touch PE")

print("\nPer-year breakdown:")
print(f"  {'Year':<6}{'Trades':>7}{'WR':>7}{'PnL (₹)':>13}{'Pts/wk':>9}{'Strategy mix':>20}")
df['year'] = df['date'].dt.year
for yr, grp in df.groupby('year'):
    n   = len(grp); wr = round((grp.pnl>0).mean()*100,1)
    tot = round(grp.pnl.sum(),0)
    weeks = grp['date'].dt.isocalendar().week.nunique()
    pts = round(tot/weeks/LOT_SIZE, 1)
    mix = f"v17a={len(grp[grp.strategy=='v17a'])} h3={len(grp[grp.strategy=='cam_h3'])} l3={len(grp[grp.strategy=='cam_l3'])}"
    print(f"  {yr:<6}{n:>7}{wr:>6.1f}%{tot:>12,.0f}  {pts:>7}  {mix}")

print("\nExit reason breakdown:")
for reason, grp in df.groupby('exit_reason'):
    n=len(grp); wr=round((grp.pnl>0).mean()*100,1); avg=round(grp.pnl.mean(),0)
    print(f"  {reason:<12}: n={n:4d}  WR={wr}%  avg=₹{avg:,.0f}")


# ═══════════════════════════════════════════════════════════════════
# EQUITY CURVE CHART
# ═══════════════════════════════════════════════════════════════════
print("\nBuilding equity curve...")

df_v17a = df[df.strategy=='v17a'].copy()
df_cam  = df[df.strategy.isin(['cam_h3','cam_l3'])].copy()

eq_all   = df['pnl'].cumsum()
eq_v17a  = df_v17a['pnl'].cumsum()
eq_cam   = df_cam['pnl'].cumsum()
dd_all   = eq_all - eq_all.cummax()

def make_pts(dates, values):
    return [{"time": int(pd.Timestamp(t).timestamp()), "value": round(float(v), 2)}
            for t, v in zip(dates, values) if pd.notna(v)]

tv_json = {
    "isTvFormat": False,
    "candlestick": [],
    "volume": [],
    "lines": [
        {
            "id": "combined",
            "label": f"Combined ({len(df)} trades)",
            "color": "#26a69a",
            "data": make_pts(df['date'], eq_all),
            "seriesType": "line"
        },
        {
            "id": "v17a",
            "label": f"v17a only ({len(df_v17a)} trades)",
            "color": "#4BC0C0",
            "data": make_pts(df_v17a['date'], eq_v17a),
            "seriesType": "line"
        },
        {
            "id": "camarilla",
            "label": f"Camarilla add-on ({len(df_cam)} trades)",
            "color": "#FF9F40",
            "data": make_pts(df_cam['date'], eq_cam),
            "seriesType": "line"
        },
        {
            "id": "drawdown",
            "label": "Drawdown",
            "color": "#ef5350",
            "data": make_pts(df['date'], dd_all),
            "seriesType": "baseline",
            "baseValue": 0,
            "isNewPane": True
        }
    ]
}

tot_all   = round(df['pnl'].sum(), 0)
pts_1lot  = round(tot_all/5/52/LOT_SIZE, 1)
pts_2lots = round(pts_1lot*2, 1)

send_custom_chart("56_combined_equity",
    tv_json,
    title=f"Combined v17a + Camarilla Touch | 5yr 1-lot | {pts_1lot}pts/wk → {pts_2lots}pts/wk @2lots")

print(f"Chart pushed ✓")

# ── Save ─────────────────────────────────────────────────────────────
out = f'{OUT_DIR}/56_combined_trades.csv'
df.to_csv(out, index=False)
print(f"Saved → {out}  ({len(df)} records)")

print(f"\n{'='*55}")
print(f"FINAL SUMMARY")
print(f"{'='*55}")
print(f"  Total trades  : {len(df)}")
print(f"  Total P&L     : ₹{tot_all:,.0f} / 5yr")
print(f"  Per year      : ₹{round(tot_all/5,0):,.0f}")
print(f"  WR overall    : {round((df.pnl>0).mean()*100,1)}%")
print(f"  Max drawdown  : ₹{round(dd_all.min(),0):,.0f}")
print(f"  1 lot         : {pts_1lot} pts/wk")
print(f"  2 lots        : {pts_2lots} pts/wk")
print(f"  3 lots        : {round(pts_1lot*3,1)} pts/wk")
print(f"  4 lots        : {round(pts_1lot*4,1)} pts/wk  {'✅ ~100' if pts_1lot*4>=90 else ''}")
