"""
Missing Zones Backtest — pdh_to_r1+bull + pdl_to_bc+bear
=========================================================
These two zone+bias combos have NO v17a signal currently.
Combined: 195 days/5yr sitting idle.

Grid search: entry_time × strike_type × target_pct × sl_pct
3-tier lock-in trail. Tick-level SL/target check.
Output: data/YYYYMMDD/52_missing_zones_trades.csv
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np

FOLDER    = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell'
OUT_DIR   = f'{FOLDER}/data/20260428'
LOT_SIZE  = 75
STRIKE_INT = 50
EMA_PERIOD = 20
BODY_MIN   = 0.10
EOD_EXIT   = '15:20:00'
YEARS      = 5

# Grid
ENTRY_TIMES  = ['09:16:02', '09:20:02', '09:25:02', '09:31:02']
STRIKE_TYPES = ['ATM', 'OTM1', 'ITM1']
TARGETS      = [0.20, 0.30, 0.40, 0.50]
SL_MULTS     = [0.50, 1.00, 1.50, 2.00]

# The two missing zones and their signals
MISSING_ZONES = {
    ('pdh_to_r1', 'bull'): 'PE',   # open just above PDH, bull EMA → sell PE
    ('pdl_to_bc', 'bear'): 'CE',   # open just below CPR, bear EMA → sell CE
}

def r2(v): return round(float(v), 2)

def compute_pivots(h, l, c):
    pp=r2((h+l+c)/3); bc=r2((h+l)/2); tc=r2(2*pp-bc)
    r1=r2(2*pp-l); r2_=r2(pp+(h-l)); r3=r2(r1+(h-l)); r4=r2(r2_+(h-l))
    s1=r2(2*pp-h); s2_=r2(pp-(h-l)); s3=r2(s1-(h-l)); s4=r2(s2_-(h-l))
    return dict(pp=pp,bc=bc,tc=tc,r1=r1,r2=r2_,r3=r3,r4=r4,s1=s1,s2=s2_,s3=s3,s4=s4)

def classify_zone(op, pvt, pdh, pdl):
    if   op>pvt['r4']:  return 'above_r4'
    elif op>pvt['r3']:  return 'r3_to_r4'
    elif op>pvt['r2']:  return 'r2_to_r3'
    elif op>pvt['r1']:  return 'r1_to_r2'
    elif op>pdh:        return 'pdh_to_r1'
    elif op>pvt['tc']:  return 'tc_to_pdh'
    elif op>=pvt['bc']: return 'within_cpr'
    elif op>pdl:        return 'pdl_to_bc'
    elif op>pvt['s1']:  return 'pdl_to_s1'
    elif op>pvt['s2']:  return 's1_to_s2'
    elif op>pvt['s3']:  return 's2_to_s3'
    elif op>pvt['s4']:  return 's3_to_s4'
    else:               return 'below_s4'

def get_strike(atm, opt_type, stype):
    if opt_type == 'CE':
        return {'OTM1': atm+STRIKE_INT, 'ATM': atm, 'ITM1': atm-STRIKE_INT}[stype]
    return {'OTM1': atm-STRIKE_INT, 'ATM': atm, 'ITM1': atm+STRIKE_INT}[stype]

def sim_pct(ts, ps, ep, eod_ns, tgt_pct, sl_pct):
    tgt=r2(ep*(1-tgt_pct)); hsl=r2(ep*(1+sl_pct)); sl=hsl; md=0.0
    for i in range(len(ts)):
        t=ts[i]; p=ps[i]
        if t >= eod_ns: return r2((ep-p)*LOT_SIZE), 'eod', r2(p)
        d=(ep-p)/ep
        if d>md: md=d
        if   md>=0.60: sl=min(sl, r2(ep*(1-md*0.95)))
        elif md>=0.40: sl=min(sl, r2(ep*0.80))
        elif md>=0.25: sl=min(sl, ep)
        if p<=tgt: return r2((ep-p)*LOT_SIZE), 'target', r2(p)
        if p>=sl:
            return r2((ep-p)*LOT_SIZE), 'lockin_sl' if sl<hsl else 'hard_sl', r2(p)
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', r2(ps[-1])


# ── Pass 1: daily OHLC + EMA ────────────────────────────────────────
print(f"Pass 1: NIFTY daily OHLC + EMA({EMA_PERIOD}) ({YEARS}yr)...")
t0 = time.time()

all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest-pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)
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


# ── Pass 2: scan missing zone days ──────────────────────────────────
print("Pass 2: scanning pdh_to_r1+bull and pdl_to_bc+bear days...")
t1 = time.time()
records = []
day_counts = {'pdh_to_r1+bull': 0, 'pdl_to_bc+bear': 0}

for date in dates_5yr:
    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx-1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph, pl, pc, _ = daily_ohlc[prev]
    _, _, _, today_op = daily_ohlc[date]
    prev_open = daily_ohlc[prev][3]
    prev_body = round(abs(pc-prev_open)/prev_open*100, 3)
    if prev_body <= BODY_MIN: continue

    pvt  = compute_pivots(ph, pl, pc)
    e20  = ema_s.get(date, np.nan)
    if np.isnan(e20): continue

    bias = 'bull' if today_op > e20 else 'bear'
    zone = classify_zone(today_op, pvt, ph, pl)

    if (zone, bias) not in MISSING_ZONES: continue
    signal = MISSING_ZONES[(zone, bias)]
    day_counts[f'{zone}+{bias}'] += 1

    dstr     = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    exps     = list_expiry_dates(date)
    if not exps: continue
    expiry   = exps[0]
    exp_dt   = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte      = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: continue

    atm    = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    for entry_time in ENTRY_TIMES:
        for stype in STRIKE_TYPES:
            strike = get_strike(atm, signal, stype)
            inst   = f'NIFTY{expiry}{strike}{signal}'

            tk = load_tick_data(date, inst, entry_time[:5]+':00', EOD_EXIT)
            if tk is None or tk.empty: continue

            ep_mask = tk['time'] >= entry_time
            if not ep_mask.any(): continue
            ep = r2(float(tk[ep_mask].iloc[0]['price']))
            if ep < 5: continue

            # IV filter — same as live strategy
            iv = ep / today_op * 100
            if iv <= 0.47: continue

            opt_ts = tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
            opt_ps = tk[ep_mask]['price'].values.astype(float)

            for tgt_pct in TARGETS:
                for sl_pct in SL_MULTS:
                    pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt_pct, sl_pct)
                    records.append(dict(
                        date=dstr, zone=zone, bias=bias, opt=signal,
                        strike_type=stype, strike=strike, dte=dte,
                        entry_time=entry_time, ep=ep, xp=xp,
                        exit_reason=reason, pnl=pnl,
                        target_pct=tgt_pct, sl_pct=sl_pct,
                    ))

print(f"  Done in {time.time()-t1:.0f}s")
print(f"  pdh_to_r1+bull: {day_counts['pdh_to_r1+bull']} days")
print(f"  pdl_to_bc+bear: {day_counts['pdl_to_bc+bear']} days")
print(f"  Records: {len(records)}")

if not records:
    print("No records. Check data/IV filter.")
    sys.exit(1)

df = pd.DataFrame(records)

# ── Analysis ────────────────────────────────────────────────────────
print("\n" + "="*72)
print("TOP CONFIGS — by total P&L (min 15 trades)")
print("="*72)
grp = df.groupby(['zone','strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
    n='count',
    wr=lambda x: round((x>0).mean()*100,1),
    avg=lambda x: round(x.mean(),0),
    total=lambda x: round(x.sum(),0)
).reset_index()
grp = grp[grp['n'] >= 15].sort_values('total', ascending=False)
print(grp.head(20).to_string(index=False))

print("\n" + "="*72)
print("BEST CONFIG PER ZONE")
print("="*72)
for zone_key in ['pdh_to_r1', 'pdl_to_bc']:
    z = grp[grp['zone'] == zone_key]
    if z.empty:
        print(f"\n{zone_key}: no config with 15+ trades")
        continue
    best = z.iloc[0]
    print(f"\n{zone_key} best: {best.strike_type} | {best.entry_time} | "
          f"tgt={best.target_pct:.0%} | sl={best.sl_pct}x")
    print(f"  Trades={int(best.n)}  WR={best.wr}%  Avg=Rs{best.avg:,.0f}  Total=Rs{best.total:,.0f}")

    # year-by-year
    mask = ((df.zone==zone_key) & (df.strike_type==best.strike_type) &
            (df.entry_time==best.entry_time) & (df.target_pct==best.target_pct) &
            (df.sl_pct==best.sl_pct))
    detail = df[mask].copy()
    detail['year'] = pd.to_datetime(detail['date']).dt.year
    yr = detail.groupby('year')['pnl'].agg(
        n='count', wr=lambda x:round((x>0).mean()*100,1),
        avg=lambda x:round(x.mean(),0), total=lambda x:round(x.sum(),0)
    )
    print(yr.to_string())

print("\n" + "="*72)
print("COMBINED BEST — both zones together")
print("="*72)
# Find single config that works for both zones
combined = df.groupby(['strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
    n='count', wr=lambda x:round((x>0).mean()*100,1),
    avg=lambda x:round(x.mean(),0), total=lambda x:round(x.sum(),0)
).reset_index()
combined = combined[combined['n'] >= 30].sort_values('total', ascending=False)
print(combined.head(10).to_string(index=False))

if not combined.empty:
    best_c = combined.iloc[0]
    print(f"\nBest combined: {best_c.strike_type} | {best_c.entry_time} | "
          f"tgt={best_c.target_pct:.0%} | sl={best_c.sl_pct}x")
    print(f"  Total trades={int(best_c.n)}  WR={best_c.wr}%  "
          f"Avg=Rs{best_c.avg:,.0f}  Total 5yr=Rs{best_c.total:,.0f}")
    yrs = 5
    print(f"  Per year = Rs{best_c.total/yrs:,.0f}  Per week = {best_c.total/yrs/52/75:.1f} pts")

# ── Save ────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = f'{OUT_DIR}/52_missing_zones_trades.csv'
df.to_csv(out, index=False)
print(f"\nSaved → {out}  ({len(df)} records)")
