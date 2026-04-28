"""
Camarilla R3/S3 Touch Trigger — Correct Backtest
=================================================
Real concept: Price TRAVELS to H3 (inside CPR) during the day → sell CE at that moment
              Price TRAVELS to L3 (inside CPR) during the day → sell PE at that moment

This is an intraday level-touch trigger, NOT an open-time signal.

Logic:
  1. Compute H3/L3 from prev day OHLC
  2. Check if H3 or L3 falls inside CPR zone (cpr_lo..cpr_hi)
  3. Scan NIFTY spot tick data intraday — detect first touch of H3 or L3
  4. Entry = next tick after touch + 2 seconds
  5. Sell OTM CE (if H3 touch) or OTM PE (if L3 touch)
  6. Grid search: strike_type × target_pct × sl_pct
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np

OUT_DIR    = 'data/20260428'
LOT_SIZE   = 75
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_EXIT   = '15:20:00'
SCAN_FROM  = '09:16:00'   # start scanning after first candle
SCAN_TO    = '13:00:00'   # only morning session touches (reversal logic)
YEARS      = 5
TOUCH_TOL  = 0.05         # % tolerance for touch detection (5 paise per 100)

os.makedirs(OUT_DIR, exist_ok=True)

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp = r2((h+l+c)/3); bc = r2((h+l)/2); tc = r2(2*pp-bc)
    return dict(pp=pp, bc=bc, tc=tc)

def compute_camarilla(h, l, c):
    rng = h - l
    return dict(h3=r2(c + rng*1.1/4), l3=r2(c - rng*1.1/4),
                h4=r2(c + rng*1.1/2), l4=r2(c - rng*1.1/2))

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

def detect_touch(spot_tks, level, direction, scan_from, scan_to):
    """
    Detect first tick where spot price touches a level.
    direction: 'from_below' (price rises to touch H3) or 'from_above' (price falls to touch L3)
    Returns: (touch_time_str, touch_price) or None
    """
    mask = (spot_tks['time'] >= scan_from) & (spot_tks['time'] <= scan_to)
    scan = spot_tks[mask].reset_index(drop=True)
    if len(scan) < 2: return None

    tol = level * TOUCH_TOL / 100

    for i in range(1, len(scan)):
        price = float(scan.iloc[i]['price'])
        prev_price = float(scan.iloc[i-1]['price'])
        t = scan.iloc[i]['time']

        if direction == 'from_below':
            # price rises to touch H3 (reversal: sell CE after touching H3)
            if prev_price < level and price >= level - tol:
                return (t, price)
        else:
            # price falls to touch L3 (reversal: sell PE after touching L3)
            if prev_price > level and price <= level + tol:
                return (t, price)

    return None

# ── Load OHLC + EMA ──────────────────────────────────────────────────
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

# ── Main backtest ─────────────────────────────────────────────────────
print("\nScanning intraday touches of H3/L3 inside CPR...")
t1 = time.time()

STRIKE_TYPES = ['ATM', 'OTM1', 'ITM1']
TARGETS      = [0.20, 0.30, 0.40, 0.50]
SL_MULTS     = [0.50, 1.00, 1.50, 2.00]
BODY_MIN     = 0.10

records = []
day_counts = {'eligible': 0, 'H3_touch': 0, 'L3_touch': 0, 'no_touch': 0}

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

    pvt = compute_pivots(ph, pl, pc)
    cam = compute_camarilla(ph, pl, pc)
    e20 = ema_s.get(date, np.nan)
    if np.isnan(e20): continue

    cpr_lo = min(pvt['tc'], pvt['bc'])
    cpr_hi = max(pvt['tc'], pvt['bc'])

    h3_in = cpr_lo <= cam['h3'] <= cpr_hi
    l3_in = cpr_lo <= cam['l3'] <= cpr_hi
    if not h3_in and not l3_in: continue

    day_counts['eligible'] += 1

    # Load spot ticks for today
    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty: continue

    dstr   = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    exps   = list_expiry_dates(date)
    if not exps: continue
    expiry = exps[0]
    exp_dt = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte    = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: continue

    atm    = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    touched_any = False

    # H3 touch → sell CE (price rises to resistance, expect reversal down)
    if h3_in:
        touch = detect_touch(spot_tks, cam['h3'], 'from_below', SCAN_FROM, SCAN_TO)
        if touch:
            touch_time, touch_price = touch
            # Entry 2 seconds after touch
            th, tm, ts_ = touch_time.split(':')
            entry_secs = int(th)*3600 + int(tm)*60 + int(ts_) + 2
            entry_h = entry_secs // 3600
            entry_m = (entry_secs % 3600) // 60
            entry_s = entry_secs % 60
            entry_time = f'{entry_h:02d}:{entry_m:02d}:{entry_s:02d}'
            if entry_time >= EOD_EXIT: continue

            day_counts['H3_touch'] += 1
            touched_any = True

            for stype in STRIKE_TYPES:
                strike = get_strike(atm, 'CE', stype)
                inst   = f'NIFTY{expiry}{strike}CE'
                tk     = load_tick_data(date, inst, touch_time[:5]+':00', EOD_EXIT)
                if tk is None or tk.empty: continue
                ep_mask = tk['time'] >= entry_time
                if not ep_mask.any(): continue
                ep = r2(float(tk[ep_mask].iloc[0]['price']))
                if ep < 5: continue
                opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
                opt_ps = tk[ep_mask]['price'].values.astype(float)
                for tgt in TARGETS:
                    for slm in SL_MULTS:
                        pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt, slm)
                        records.append(dict(
                            date=dstr, setup='H3_touch', opt='CE',
                            cam_level=cam['h3'], touch_time=touch_time,
                            entry_time=entry_time, touch_price=touch_price,
                            strike_type=stype, ep=ep, xp=xp,
                            exit_reason=reason, pnl=pnl,
                            target_pct=tgt, sl_pct=slm, dte=dte
                        ))

    # L3 touch → sell PE (price falls to support, expect reversal up)
    if l3_in:
        touch = detect_touch(spot_tks, cam['l3'], 'from_above', SCAN_FROM, SCAN_TO)
        if touch:
            touch_time, touch_price = touch
            th, tm, ts_ = touch_time.split(':')
            entry_secs = int(th)*3600 + int(tm)*60 + int(ts_) + 2
            entry_h = entry_secs // 3600
            entry_m = (entry_secs % 3600) // 60
            entry_s = entry_secs % 60
            entry_time = f'{entry_h:02d}:{entry_m:02d}:{entry_s:02d}'
            if entry_time >= EOD_EXIT: continue

            day_counts['L3_touch'] += 1
            touched_any = True

            for stype in STRIKE_TYPES:
                strike = get_strike(atm, 'PE', stype)
                inst   = f'NIFTY{expiry}{strike}PE'
                tk     = load_tick_data(date, inst, touch_time[:5]+':00', EOD_EXIT)
                if tk is None or tk.empty: continue
                ep_mask = tk['time'] >= entry_time
                if not ep_mask.any(): continue
                ep = r2(float(tk[ep_mask].iloc[0]['price']))
                if ep < 5: continue
                opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
                opt_ps = tk[ep_mask]['price'].values.astype(float)
                for tgt in TARGETS:
                    for slm in SL_MULTS:
                        pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt, slm)
                        records.append(dict(
                            date=dstr, setup='L3_touch', opt='PE',
                            cam_level=cam['l3'], touch_time=touch_time,
                            entry_time=entry_time, touch_price=touch_price,
                            strike_type=stype, ep=ep, xp=xp,
                            exit_reason=reason, pnl=pnl,
                            target_pct=tgt, sl_pct=slm, dte=dte
                        ))

    if not touched_any: day_counts['no_touch'] += 1

elapsed = time.time()-t1
print(f"  Done {elapsed:.0f}s — {day_counts}")
print(f"  {len(records)} records")

# ── Results ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULTS — Camarilla Level TOUCH trigger")
print("="*65)

if not records:
    print("  No records found.")
else:
    df = pd.DataFrame(records)

    for setup in ['H3_touch', 'L3_touch']:
        sg = df[df.setup == setup]
        if sg.empty: print(f"\n{setup}: no trades"); continue

        grp = sg.groupby(['strike_type','target_pct','sl_pct'])['pnl'].agg(
            n='count',
            wr=lambda x: round((x>0).mean()*100,1),
            avg=lambda x: round(x.mean(),0),
            total=lambda x: round(x.sum(),0)
        ).reset_index()

        valid = grp[grp.n >= 8].sort_values('total', ascending=False)
        n_days = sg['date'].nunique()
        print(f"\n{setup} — {n_days} unique days, top 5 configs:")
        for _, b in valid.head(5).iterrows():
            pts = round(b.total/5/52/75, 1)
            print(f"  {b.strike_type} tgt={b.target_pct:.0%} sl={b.sl_pct}x"
                  f"  → n={int(b.n)} WR={b.wr}%  avg=₹{b.avg:,.0f}  "
                  f"total=₹{b.total:,.0f}/5yr  {pts}pts/wk")

    # Touch time distribution
    print("\n" + "="*65)
    print("TOUCH TIME DISTRIBUTION")
    print("="*65)
    for setup in ['H3_touch','L3_touch']:
        sg = df[df.setup==setup].drop_duplicates('date')
        if sg.empty: continue
        hours = sg['touch_time'].str[:5].value_counts().sort_index()
        buckets = {'09:1':0,'09:2':0,'09:3':0,'09:4':0,'09:5':0,
                   '10:':0,'11:':0,'12:':0,'13:':0}
        for t, cnt in hours.items():
            for k in buckets:
                if t.startswith(k): buckets[k]+=cnt; break
        print(f"  {setup}:")
        for k,v in buckets.items():
            if v>0: print(f"    {k}xx  → {v} days")

    # Combined best projection
    print("\n" + "="*65)
    print("COMBINED: v17a + best touch configs")
    print("="*65)
    base_total = 407370; base_trades = 358

    # Best H3: pick top by total with n>=8
    h3g = df[df.setup=='H3_touch'].groupby(['strike_type','target_pct','sl_pct'])['pnl'].agg(
        n='count', total='sum').reset_index()
    h3g = h3g[h3g.n>=8].sort_values('total',ascending=False)

    l3g = df[df.setup=='L3_touch'].groupby(['strike_type','target_pct','sl_pct'])['pnl'].agg(
        n='count', total='sum').reset_index()
    l3g = l3g[l3g.n>=8].sort_values('total',ascending=False)

    add_h3 = 0; add_h3_n = 0
    add_l3 = 0; add_l3_n = 0

    if not h3g.empty:
        b = h3g.iloc[0]
        mask = (df.setup=='H3_touch')&(df.strike_type==b.strike_type)&\
               (df.target_pct==b.target_pct)&(df.sl_pct==b.sl_pct)
        add_h3 = round(df[mask].pnl.sum(),0)
        add_h3_n = df[mask]['date'].nunique()

    if not l3g.empty:
        b = l3g.iloc[0]
        mask = (df.setup=='L3_touch')&(df.strike_type==b.strike_type)&\
               (df.target_pct==b.target_pct)&(df.sl_pct==b.sl_pct)
        add_l3 = round(df[mask].pnl.sum(),0)
        add_l3_n = df[mask]['date'].nunique()

    new_total  = base_total + add_h3 + add_l3
    new_trades = base_trades + add_h3_n + add_l3_n
    pts = round(new_total/5/52/75, 1)

    print(f"  v17a base      : {base_trades} trades  ₹{base_total:,.0f}/5yr  20.9 pts/wk")
    print(f"  + H3 touch CE  : +{add_h3_n} days  +₹{add_h3:,.0f}/5yr")
    print(f"  + L3 touch PE  : +{add_l3_n} days  +₹{add_l3:,.0f}/5yr")
    print(f"  Combined (1lot): {new_trades} trades  ₹{new_total:,.0f}/5yr  {pts} pts/wk")
    print(f"  Combined (2lot): {pts*2:.1f} pts/wk  {'✅ NEAR 100' if pts*2>=80 else '⚠ below 80'}")

    df.to_csv(f'{OUT_DIR}/55_camarilla_touch.csv', index=False)
    print(f"\nSaved → {OUT_DIR}/55_camarilla_touch.csv  ({len(df)} records)")
