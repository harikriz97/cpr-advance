"""
SENSEX Tuesday Backtest
=======================
On NIFTY expiry Tuesdays (DTE=0 for NIFTY → skip NIFTY):
  - Use NIFTY zone + EMA bias signal (correlation ~0.98)
  - Trade SENSEX options (DTE=2, fat premium, lot=20)
  - Grid search: strike_type × target_pct × sl_pct

5-year backtest. Output: data/YYYYMMDD/51_sensex_tuesday_trades.csv
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np

FOLDER    = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell'
OUT_DIR   = f'{FOLDER}/data/20260428'
LOT_SIZE  = 20          # SENSEX lot size
STRIKE_INT = 100        # SENSEX strike interval
EMA_PERIOD = 20
BODY_MIN   = 0.10
EOD_EXIT   = '15:20:00'
YEARS      = 1   # SENSEX option data available from Dec 2025 only

# Grid parameters
ENTRY_TIMES  = ['09:16:02', '09:20:02', '09:25:02', '09:31:02']
STRIKE_TYPES = ['ATM', 'OTM1', 'ITM1']
TARGETS      = [0.20, 0.30, 0.40, 0.50]
SL_MULTS     = [0.50, 1.00, 1.50, 2.00]

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

def get_v17a_signal(zone, bias):
    if zone in {'above_r4','r3_to_r4','r2_to_r3','r1_to_r2'}: return 'PE'
    if zone == 'pdh_to_r1'  and bias == 'bear': return 'PE'
    if zone == 'tc_to_pdh':                      return 'PE'
    if zone == 'within_cpr' and bias == 'bull':  return 'PE'
    if zone == 'within_cpr' and bias == 'bear':  return 'CE'
    if zone == 'pdl_to_bc'  and bias == 'bull':  return 'PE'
    if zone in {'pdl_to_s1','s1_to_s2','s3_to_s4','below_s4'} and bias=='bear': return 'CE'
    return None

def get_strike(atm, opt_type, stype):
    if opt_type == 'CE':
        return {'OTM1': atm+STRIKE_INT, 'ATM': atm, 'ITM1': atm-STRIKE_INT}[stype]
    return {'OTM1': atm-STRIKE_INT, 'ATM': atm, 'ITM1': atm+STRIKE_INT}[stype]

def sim_pct(ts, ps, ep, eod_ns, tgt_pct, sl_pct):
    """3-tier lock-in trail + pct hard SL."""
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*(1+sl_pct)); sl = hsl; md = 0.0
    for i in range(len(ts)):
        t = ts[i]; p = ps[i]
        if t >= eod_ns: return r2((ep-p)*LOT_SIZE), 'eod', p
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE), 'target', p
        if p >= sl:
            return r2((ep-p)*LOT_SIZE), 'lockin_sl' if sl<hsl else 'hard_sl', p
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', ps[-1]


# ── Pass 1: NIFTY daily OHLC + EMA (for signal) ────────────────────
print(f"Pass 1: NIFTY daily OHLC + EMA({EMA_PERIOD}) ({YEARS}yr)...")
t0 = time.time()

all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest-pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)

nifty_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None or tks.empty: continue
    nifty_ohlc[d] = (
        float(tks['price'].max()),
        float(tks['price'].min()),
        float(tks[tks['time'] <= '15:30:00']['price'].iloc[-1]),
        float(tks[tks['time'] >= '09:15:00']['price'].iloc[0]),
    )

close_s = pd.Series({d: v[2] for d, v in nifty_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD, adjust=False).mean().shift(1)  # prev day EMA
print(f"  {len(nifty_ohlc)} NIFTY days loaded in {time.time()-t0:.0f}s")


# ── Pass 2: scan Tuesdays — NIFTY signal → SENSEX trade ────────────
print("Pass 2: scanning Tuesdays for SENSEX trades...")
t1 = time.time()
records = []
skipped = 0
tuesday_count = 0

for date in dates_5yr:
    dt = pd.Timestamp(date[:4]+'-'+date[4:6]+'-'+date[6:])
    if dt.weekday() != 1: continue  # Tuesday only

    tuesday_count += 1
    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx-1]
    if prev not in nifty_ohlc or date not in nifty_ohlc: continue

    # NIFTY CPR + EMA → signal
    ph, pl, pc, _ = nifty_ohlc[prev]
    _, _, _, today_op = nifty_ohlc[date]
    prev_open = nifty_ohlc[prev][3]
    prev_body = round(abs(pc - prev_open)/prev_open*100, 3)
    if prev_body <= BODY_MIN:
        skipped += 1; continue

    pvt  = compute_pivots(ph, pl, pc)
    e20  = ema_s.get(date, np.nan)
    if np.isnan(e20):
        skipped += 1; continue

    bias   = 'bull' if today_op > e20 else 'bear'
    zone   = classify_zone(today_op, pvt, ph, pl)
    signal = get_v17a_signal(zone, bias)
    if signal is None:
        skipped += 1; continue

    # SENSEX expiry (DTE=2 typically)
    dstr       = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    exps       = list_expiry_dates(date, 'SENSEX')
    if not exps:
        skipped += 1; continue
    expiry     = exps[0]
    exp_dt     = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte        = (exp_dt - dt).days
    if dte == 0:
        skipped += 1; continue

    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    # For each entry time, find SENSEX ATM from first valid option tick
    for entry_time in ENTRY_TIMES:
        entry_hhmm = entry_time[:5]

        # Find SENSEX true ATM: strike where |CE - PE| is minimised
        # Scan 70000-90000, collect all strikes with both CE+PE valid,
        # then pick the one with smallest |CE-PE| difference
        candidates = {}
        for strike_try in range(70000, 91000, STRIKE_INT):
            tk_ce = load_tick_data(date, f'SENSEX{expiry}{strike_try}CE', entry_hhmm+':00', entry_hhmm+':59')
            tk_pe = load_tick_data(date, f'SENSEX{expiry}{strike_try}PE', entry_hhmm+':00', entry_hhmm+':59')
            if tk_ce is None or tk_pe is None or tk_ce.empty or tk_pe.empty: continue
            ce_p = float(tk_ce.iloc[0]['price'])
            pe_p = float(tk_pe.iloc[0]['price'])
            if ce_p > 50 and pe_p > 50:   # both have meaningful premium
                candidates[strike_try] = abs(ce_p - pe_p)

        if not candidates: continue
        # True ATM = strike where CE ≈ PE (min diff)
        sensex_atm = min(candidates, key=candidates.get)

        # Now load full day ticks for each strike type
        for stype in STRIKE_TYPES:
            strike = get_strike(sensex_atm, signal, stype)
            inst   = f'SENSEX{expiry}{strike}{signal}'

            # Entry price — load 1 min window around entry time
            tk_entry = load_tick_data(date, inst, entry_hhmm+':00', entry_hhmm+':59')
            if tk_entry is None or tk_entry.empty: continue
            ep_mask = tk_entry['time'] >= entry_time
            if not ep_mask.any(): continue
            ep = r2(float(tk_entry[ep_mask].iloc[0]['price']))
            if ep < 10: continue  # too cheap

            # Full tick data for simulation
            tk_full = load_tick_data(date, inst, entry_time[:5]+':'+entry_time[6:], EOD_EXIT)
            if tk_full is None or tk_full.empty: continue

            opt_ts = tk_full['date_time'].values.astype('datetime64[ns]').astype('int64')
            opt_ps = tk_full['price'].values.astype(float)

            for tgt_pct in TARGETS:
                for sl_pct in SL_MULTS:
                    pnl, reason, xp = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt_pct, sl_pct)
                    records.append(dict(
                        date=dstr, zone=zone, bias=bias, opt=signal,
                        strike_type=stype, strike=strike, expiry=expiry, dte=dte,
                        entry_time=entry_time, ep=ep, xp=r2(xp),
                        exit_reason=reason, pnl=pnl,
                        target_pct=tgt_pct, sl_pct=sl_pct,
                    ))

print(f"  Done in {time.time()-t1:.0f}s")
print(f"  Tuesdays scanned: {tuesday_count} | skipped: {skipped} | records: {len(records)}")

if not records:
    print("No records found. Check data availability.")
    sys.exit(1)

df = pd.DataFrame(records)

# ── Analysis ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BEST CONFIGS — by total P&L (min 10 trades)")
print("="*70)
grp = df.groupby(['strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
    n='count',
    wr=lambda x: round((x>0).mean()*100, 1),
    avg=lambda x: round(x.mean(), 0),
    total=lambda x: round(x.sum(), 0)
).reset_index()
grp = grp[grp['n'] >= 10].sort_values('total', ascending=False)
print(grp.head(20).to_string(index=False))

print("\n" + "="*70)
print("BEST CONFIG DETAIL — top performer")
print("="*70)
if not grp.empty:
    best = grp.iloc[0]
    mask = ((df.strike_type == best.strike_type) &
            (df.entry_time  == best.entry_time)  &
            (df.target_pct  == best.target_pct)  &
            (df.sl_pct      == best.sl_pct))
    detail = df[mask].copy()
    by_yr = detail.copy()
    by_yr['year'] = pd.to_datetime(by_yr['date']).dt.year
    print(f"Config: {best.strike_type} | {best.entry_time} | tgt={best.target_pct:.0%} | sl={best.sl_pct}x")
    print(f"Trades: {int(best.n)} | WR: {best.wr}% | Avg: Rs{best.avg:,.0f} | Total: Rs{best.total:,.0f}")
    print()
    print(by_yr.groupby('year')['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1),
        avg=lambda x: round(x.mean(),0), total=lambda x: round(x.sum(),0)
    ).to_string())
    print()
    print("By zone:")
    print(detail.groupby(['zone','bias'])['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1), avg='mean', total='sum'
    ).round(0).sort_values('total', ascending=False).to_string())

print("\n" + "="*70)
print("ENTRY TIME COMPARISON (best strike+params per time)")
print("="*70)
et_grp = df.groupby('entry_time')['pnl'].agg(
    n='count', wr=lambda x: round((x>0).mean()*100,1),
    avg=lambda x: round(x.mean(),0), total=lambda x: round(x.sum(),0)
).reset_index()
print(et_grp.to_string(index=False))

# ── Save ────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_path = f'{OUT_DIR}/51_sensex_tuesday_trades.csv'
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}  ({len(df)} records)")
