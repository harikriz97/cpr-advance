"""
Intraday v2 Strategy Optimization
==================================
Tests three improvements over the current backtest (107 trades):

  A. TC target sweep      — OTM1 SL=50%, targets = [0.20..0.40]
  B. Scan window sweep    — extend SCAN_TO to 10:55 / 11:20
  C. PDH as upside break  — add PDH to up_levels alongside R1/R2/TC

Runs only on NO-SIGNAL days (v17a has no entry signal).
Output: data/YYYYMMDD/50_intraday_v2_optimize.csv
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
sys.path.insert(0, os.path.expanduser('~') + '/.claude/skills/sa-kron-chart/scripts')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_tick_data, load_spot_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np
from datetime import timedelta

FOLDER   = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell'
OUT_DIR  = f'{FOLDER}/data/20260420'
LOT_SIZE = 75; STRIKE_INT = 50; EMA_PERIOD = 20; EOD_EXIT = '15:20:00'; YEARS = 5
IV_MIN = 0.47; BODY_MIN = 0.10

# ── Test parameters ────────────────────────────────────────────────
TC_TARGETS   = [0.20, 0.25, 0.30, 0.35, 0.40]   # Test A: TC target sweep
SCAN_WINDOWS = [                                   # Test B: scan window sweep
    ('09:30', '10:25'),   # current baseline
    ('09:30', '10:55'),   # +30 min
    ('09:30', '11:20'),   # +55 min
]
PDH_TEST = True                                    # Test C: PDH as upside break

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

def get_v17a_signal(zone, ema_bias):
    if zone in {'above_r4','r3_to_r4','r2_to_r3','r1_to_r2'}: return 'PE'
    if zone == 'pdh_to_r1' and ema_bias == 'bear': return 'PE'
    if zone == 'tc_to_pdh': return 'PE'
    if zone == 'within_cpr' and ema_bias == 'bull': return 'PE'
    if zone == 'within_cpr' and ema_bias == 'bear': return 'CE'
    if zone == 'pdl_to_bc'  and ema_bias == 'bull': return 'PE'
    if zone in {'pdl_to_s1','s1_to_s2','s3_to_s4','below_s4'} and ema_bias=='bear': return 'CE'
    return None

def get_strike(atm, opt_type, stype):
    if opt_type == 'CE':
        return {'OTM1': atm+STRIKE_INT, 'ATM': atm, 'ITM1': atm-STRIKE_INT}[stype]
    return {'OTM1': atm-STRIKE_INT, 'ATM': atm, 'ITM1': atm+STRIKE_INT}[stype]

def sim_pct(ts, ps, ep, eod_ns, tgt_pct, sl_pct):
    """% SL + 3-tier lock-in trail."""
    tgt = r2(ep*(1-tgt_pct)); hsl = r2(ep*(1+sl_pct)); sl = hsl; md = 0.0
    for i in range(len(ts)):
        t=ts[i]; p=ps[i]
        if t >= eod_ns: return r2((ep-p)*LOT_SIZE), 'eod', p, t
        d = (ep-p)/ep
        if d > md: md = d
        if   md >= 0.60: sl = min(sl, r2(ep*(1-md*0.95)))
        elif md >= 0.40: sl = min(sl, r2(ep*0.80))
        elif md >= 0.25: sl = min(sl, ep)
        if p <= tgt: return r2((ep-p)*LOT_SIZE), 'target', p, t
        if p >= sl:
            return r2((ep-p)*LOT_SIZE), 'lockin_sl' if sl<hsl else 'hard_sl', p, t
    return r2((ep-ps[-1])*LOT_SIZE), 'eod', ps[-1], ts[-1]

def detect_break(ohlc5, pvt, pdh, pdl, scan_from, scan_to, include_pdh=False):
    """Return first pivot break dict or None."""
    up_levels = []
    if include_pdh:
        up_levels.append(('PDH', pdh, 'PE'))
    up_levels += [('R1', pvt['r1'], 'PE'), ('R2', pvt['r2'], 'PE'), ('TC', pvt['tc'], 'PE')]
    dn_levels = [('PDL', pdl, 'CE'), ('S1', pvt['s1'], 'CE'), ('S2', pvt['s2'], 'CE')]

    try:
        scan = ohlc5.between_time(scan_from, scan_to)
    except Exception:
        return None
    if len(scan) < 2: return None

    candles = scan.reset_index()
    ts_col  = candles.columns[0]
    for idx in range(1, len(candles)):
        row = candles.iloc[idx]; prev = candles.iloc[idx-1]
        c_close = row['close']; p_close = prev['close']
        c_time  = row[ts_col]
        entry_dt = c_time + pd.Timedelta(minutes=5, seconds=2)
        for name, level, opt in up_levels:
            if p_close <= level < c_close:
                return dict(entry_dt=entry_dt, opt=opt, level=level, level_name=name)
        for name, level, opt in dn_levels:
            if p_close >= level > c_close:
                return dict(entry_dt=entry_dt, opt=opt, level=level, level_name=name)
    return None


# ── Pass 1: daily OHLC + EMA ───────────────────────────────────────
print(f"Pass 1: daily OHLC + EMA({EMA_PERIOD}) ({YEARS}yr)...")
all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest-pd.DateOffset(years=YEARS)]

extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)
t0 = time.time()
daily_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d, 'NIFTY')
    if tks is None: continue
    daily_ohlc[d] = (
        round(tks['price'].max(), 2),
        round(tks['price'].min(), 2),
        round(tks[tks['time'] <= '15:30:00']['price'].iloc[-1], 2),
        round(tks[tks['time'] >= '09:15:00']['price'].iloc[0],  2),
    )

close_s = pd.Series({d: v[2] for d, v in daily_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD, adjust=False).mean()
print(f"  {len(daily_ohlc)} days loaded in {time.time()-t0:.0f}s")


# ── Pass 2: scan no-signal days ────────────────────────────────────
print("Pass 2: scanning no-signal days with all test configurations...")

records = []
t1 = time.time()
processed = 0

for date in dates_5yr:
    idx = all_dates.index(date)
    if idx < 1: continue
    prev = all_dates[idx-1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph, pl, pc, _ = daily_ohlc[prev]
    _, _, _, today_op = daily_ohlc[date]

    pvt       = compute_pivots(ph, pl, pc)
    e20       = ema_s.get(prev, np.nan)   # FIX: use prev day EMA — today's close unknown at 09:15
    if np.isnan(e20): continue
    prev_body = round(abs(pc - daily_ohlc[prev][3]) / daily_ohlc[prev][3] * 100, 3)
    if prev_body <= BODY_MIN: continue

    bias   = 'bull' if today_op > e20 else 'bear'
    zone   = classify_zone(today_op, pvt, ph, pl)
    signal = get_v17a_signal(zone, bias)

    # Only intraday v2 days (no v17a signal)
    if signal is not None: continue

    dstr   = f'{date[:4]}-{date[4:6]}-{date[6:]}'
    expiries = list_expiry_dates(date)
    if not expiries: continue
    expiry = expiries[0]
    exp_dt = pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte    = (exp_dt - pd.Timestamp(dstr)).days
    if dte == 0: continue

    # Build 5-min OHLC directly from spot ticks
    spot_tks = load_spot_data(date, 'NIFTY')
    if spot_tks is None or spot_tks.empty: continue
    spot_tks['dt'] = pd.to_datetime(dstr + ' ' + spot_tks['time'])
    sp = spot_tks[['dt','price']].set_index('dt').rename(columns={'price':'p'})
    ohlc5 = sp['p'].resample('5min', closed='left', label='left').agg(
        open='first', high='max', low='min', close='last').dropna()
    if len(ohlc5) < 2: continue

    pdh = r2(ph); pdl = r2(pl)
    atm = int(round(today_op / STRIKE_INT) * STRIKE_INT)
    eod_ns = pd.Timestamp(dstr + ' ' + EOD_EXIT).value

    # ── Test A + B + C: for each (scan_window, include_pdh) combination
    for scan_from, scan_to in SCAN_WINDOWS:
        for include_pdh in ([False, True] if PDH_TEST else [False]):
            brk = detect_break(ohlc5, pvt, pdh, pdl, scan_from, scan_to, include_pdh)
            if brk is None: continue

            # Current INTRADAY_PARAMS mapping
            PARAMS = {
                ('PDL','CE'): ('ATM',  0.30, 2.00),
                ('R1', 'PE'): ('ATM',  0.20, 0.50),
                ('R2', 'PE'): ('ITM1', 0.50, 1.00),
                ('S1', 'CE'): ('ITM1', 0.30, 1.00),
                ('S2', 'CE'): ('OTM1', 0.40, 1.00),
                ('TC', 'PE'): ('OTM1', 0.20, 0.50),   # will be swept below
                ('PDH','PE'): ('ATM',  0.30, 1.00),    # Test C: new level params
            }
            key = (brk['level_name'], brk['opt'])
            if key not in PARAMS: continue

            stype, tgt_base, sl_pct = PARAMS[key]
            strike = get_strike(atm, brk['opt'], stype)

            ot = load_tick_data(date, f'NIFTY{expiry}{strike}{brk["opt"]}',
                                '09:15:00', '15:30:00')
            if ot is None or ot.empty: continue
            ot['dt'] = pd.to_datetime(dstr + ' ' + ot['time'])
            entry_mask = ot['dt'] >= brk['entry_dt']
            if not entry_mask.any(): continue
            ot_entry = ot[entry_mask]
            ep = float(ot_entry['price'].iloc[0])
            if ep <= 0: continue

            opt_ts = ot_entry['dt'].values.astype('datetime64[ns]').astype('int64')
            opt_ps = ot_entry['price'].values.astype(float)

            # For TC: sweep targets; for others: single simulation
            if brk['level_name'] == 'TC':
                targets_to_test = TC_TARGETS
            else:
                targets_to_test = [tgt_base]

            for tgt in targets_to_test:
                pnl, reason, xp, _ = sim_pct(opt_ts, opt_ps, ep, eod_ns, tgt, sl_pct)
                records.append(dict(
                    date=dstr, break_name=brk['level_name'], opt=brk['opt'],
                    strike_type=stype, ep=r2(ep), xp=r2(xp),
                    exit_reason=reason, pnl=pnl, dte=dte,
                    scan_from=scan_from, scan_to=scan_to,
                    include_pdh=include_pdh, target_pct=tgt, sl_pct=sl_pct,
                    entry_time=brk['entry_dt'].strftime('%H:%M:%S'),
                ))

    processed += 1
    if processed % 50 == 0:
        print(f"  {processed} no-signal days processed...")

print(f"Pass 2 done in {time.time()-t1:.0f}s. {len(records)} raw records.")

if not records:
    print("No records — check data path or date range.")
    sys.exit(1)

df = pd.DataFrame(records)

# ── Analysis ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("TEST A: TC TARGET SWEEP (OTM1, SL=50%, baseline window 09:30-10:25)")
print("="*70)
tc_base = df[(df.break_name=='TC') & (df.scan_to=='10:25') & (~df.include_pdh)]
if not tc_base.empty:
    grp = tc_base.groupby('target_pct')['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1),
        avg='mean', total='sum'
    ).round(0)
    print(grp.to_string())

print("\n" + "="*70)
print("TEST B: SCAN WINDOW COMPARISON (no PDH, baseline TC target=0.20)")
print("="*70)
# For each window, compare: same break levels, same params
win_base = df[~df.include_pdh & (df.target_pct == df.apply(
    lambda r: 0.20 if r.break_name=='TC' else
    {'PDL':0.30,'R1':0.20,'R2':0.50,'S1':0.30,'S2':0.40}.get(r.break_name, 0.20), axis=1))]

if not win_base.empty:
    grp2 = win_base.groupby('scan_to')['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1),
        avg='mean', total='sum'
    ).round(0)
    print(grp2.to_string())
    print()
    print("By level within each window:")
    grp3 = win_base.groupby(['scan_to','break_name'])['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1), avg='mean', total='sum'
    ).round(0)
    print(grp3.to_string())

print("\n" + "="*70)
print("TEST C: PDH AS NEW UPSIDE BREAK LEVEL (baseline window 09:30-10:25)")
print("="*70)
pdh_comp = df[df.scan_to=='10:25']
no_pdh = pdh_comp[~pdh_comp.include_pdh]
with_pdh = pdh_comp[pdh_comp.include_pdh]
pdh_trades = with_pdh[with_pdh.break_name=='PDH']
print(f"WITHOUT PDH level: {len(no_pdh)} trades  WR={round((no_pdh.pnl>0).mean()*100,1)}%  "
      f"avg={round(no_pdh.pnl.mean(),0)}  total={round(no_pdh.pnl.sum(),0)}")
print(f"PDH-only trades:   {len(pdh_trades)} trades  "
      f"WR={round((pdh_trades.pnl>0).mean()*100,1) if len(pdh_trades)>0 else 'N/A'}%  "
      f"avg={round(pdh_trades.pnl.mean(),0) if len(pdh_trades)>0 else 0}  "
      f"total={round(pdh_trades.pnl.sum(),0) if len(pdh_trades)>0 else 0}")
if len(pdh_trades) > 0:
    print("\nPDH break exit reasons:")
    print(pdh_trades['exit_reason'].value_counts().to_string())
    print("\nPDH break by DTE:")
    print(pdh_trades.groupby('dte')['pnl'].agg(
        n='count', wr=lambda x: round((x>0).mean()*100,1), avg='mean'
    ).round(0).to_string())

# ── Save raw results ───────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_path = f'{OUT_DIR}/50_intraday_v2_optimize.csv'
df.to_csv(out_path, index=False)
print(f"\nRaw results saved → {out_path}")
print(f"Total records: {len(df)}")
