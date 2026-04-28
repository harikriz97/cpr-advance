"""
More Trades Backtest — All NIFTY Improvements
==============================================
Tests 4 ideas to increase trade count + profit:

  A. Missing zones  — pdh_to_r1+bull (sell ATM PE) + pdl_to_bc+bear (sell ATM CE)
  B. Body filter    — lower from 0.10% to 0.05%
  C. Second trade   — after v17a exits before 11:00, scan intraday v2
  D. s2_to_s3+bear  — 8 additional days, missing from v17a params

Output: data/20260428/52_more_trades_backtest.csv
"""
import sys, os, warnings, time
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')
os.chdir('/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell')

from my_util import load_spot_data, load_tick_data, list_expiry_dates, list_trading_dates
import pandas as pd, numpy as np
from datetime import timedelta

FOLDER     = '/home/hesham/workspace/share/super_agent_data/WfLlFj/01_cpr_pivot_ema_sell'
OUT_DIR    = f'{FOLDER}/data/20260428'
LOT_SIZE   = 75
STRIKE_INT = 50
EMA_PERIOD = 20
EOD_EXIT   = '15:20:00'
YEARS      = 5

# Existing best params for reference
EXISTING_PARAMS = {
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

# NEW zones to test — grid search best params
NEW_ZONES = {
    ('pdh_to_r1', 'bull'): 'PE',   # Idea A
    ('pdl_to_bc', 'bear'): 'CE',   # Idea A
    ('s2_to_s3',  'bear'): 'CE',   # Idea D
}
ENTRY_TIMES  = ['09:16:02', '09:20:02', '09:25:02', '09:31:02']
STRIKE_TYPES = ['ATM', 'OTM1', 'ITM1']
TARGETS      = [0.20, 0.30, 0.40, 0.50]
SL_MULTS     = [0.50, 1.00, 1.50, 2.00]

BODY_MIN_STRICT = 0.10   # current
BODY_MIN_LOOSE  = 0.05   # Idea B

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

def detect_intraday_break(ohlc5, pvt, pdh, pdl, scan_from='09:30', scan_to='11:20'):
    up_levels = [('R1',pvt['r1'],'PE'),('R2',pvt['r2'],'PE')]
    dn_levels = [('PDL',pdl,'CE'),('S1',pvt['s1'],'CE'),('S2',pvt['s2'],'CE')]
    try: scan = ohlc5.between_time(scan_from, scan_to)
    except: return None
    if len(scan) < 2: return None
    candles = scan.reset_index(); ts_col = candles.columns[0]
    for idx in range(1, len(candles)):
        row=candles.iloc[idx]; prev=candles.iloc[idx-1]
        c_close=row['close']; p_close=prev['close']; c_time=row[ts_col]
        entry_dt = c_time + pd.Timedelta(minutes=5, seconds=2)
        for name,level,opt in up_levels:
            if p_close<=level<c_close: return dict(entry_dt=entry_dt,opt=opt,level=level,level_name=name)
        for name,level,opt in dn_levels:
            if p_close>=level>c_close: return dict(entry_dt=entry_dt,opt=opt,level=level,level_name=name)
    return None

INTRADAY_PARAMS = {
    ('PDL','CE'):('ATM', 0.30,2.00),
    ('R1', 'PE'):('ATM', 0.20,0.50),
    ('R2', 'PE'):('ITM1',0.50,1.00),
    ('S1', 'CE'):('ITM1',0.30,1.00),
    ('S2', 'CE'):('OTM1',0.40,1.00),
}


# ── Pass 1: daily OHLC + EMA ────────────────────────────────────────
print(f"Pass 1: NIFTY OHLC + EMA({EMA_PERIOD}) ({YEARS}yr)...")
t0=time.time()
all_dates  = list_trading_dates()
latest     = pd.Timestamp(all_dates[-1][:4]+'-'+all_dates[-1][4:6]+'-'+all_dates[-1][6:])
dates_5yr  = [d for d in all_dates
              if pd.Timestamp(d[:4]+'-'+d[4:6]+'-'+d[6:]) >= latest-pd.DateOffset(years=YEARS)]
extra = max(0, all_dates.index(dates_5yr[0]) - EMA_PERIOD - 20)
daily_ohlc = {}
for d in all_dates[extra:]:
    tks = load_spot_data(d,'NIFTY')
    if tks is None or tks.empty: continue
    daily_ohlc[d]=(
        float(tks['price'].max()), float(tks['price'].min()),
        float(tks[tks['time']<='15:30:00']['price'].iloc[-1]),
        float(tks[tks['time']>='09:15:00']['price'].iloc[0]),
    )
close_s = pd.Series({d:v[2] for d,v in daily_ohlc.items()}).sort_index()
ema_s   = close_s.ewm(span=EMA_PERIOD,adjust=False).mean().shift(1)
print(f"  {len(daily_ohlc)} days in {time.time()-t0:.0f}s")


# ── Pass 2: new zones grid search ───────────────────────────────────
print("Pass 2: new zones grid search (pdh_to_r1+bull, pdl_to_bc+bear, s2_to_s3+bear)...")
t1=time.time(); new_records=[]; day_count={}

for date in dates_5yr:
    idx=all_dates.index(date)
    if idx<1: continue
    prev=all_dates[idx-1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph,pl,pc,_=daily_ohlc[prev]; _,_,_,today_op=daily_ohlc[date]
    prev_open=daily_ohlc[prev][3]
    prev_body=round(abs(pc-prev_open)/prev_open*100,3)
    if prev_body<=BODY_MIN_STRICT: continue

    pvt=compute_pivots(ph,pl,pc)
    e20=ema_s.get(date,np.nan)
    if np.isnan(e20): continue

    bias=('bull' if today_op>e20 else 'bear')
    zone=classify_zone(today_op,pvt,ph,pl)
    if (zone,bias) not in NEW_ZONES: continue

    signal=NEW_ZONES[(zone,bias)]
    key=f'{zone}+{bias}'
    day_count[key]=day_count.get(key,0)+1

    dstr=f'{date[:4]}-{date[4:6]}-{date[6:]}'
    exps=list_expiry_dates(date)
    if not exps: continue
    expiry=exps[0]
    exp_dt=pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte=(exp_dt-pd.Timestamp(dstr)).days
    if dte==0: continue

    atm=int(round(today_op/STRIKE_INT)*STRIKE_INT)
    eod_ns=pd.Timestamp(dstr+' '+EOD_EXIT).value

    for et in ENTRY_TIMES:
        for stype in STRIKE_TYPES:
            strike=get_strike(atm,signal,stype)
            inst=f'NIFTY{expiry}{strike}{signal}'
            tk=load_tick_data(date,inst,et[:5]+':00',EOD_EXIT)
            if tk is None or tk.empty: continue
            ep_mask=tk['time']>=et
            if not ep_mask.any(): continue
            ep=r2(float(tk[ep_mask].iloc[0]['price']))
            if ep<5 or ep/today_op*100<=0.47: continue
            opt_ts=tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
            opt_ps=tk[ep_mask]['price'].values.astype(float)
            for tgt in TARGETS:
                for sl in SL_MULTS:
                    pnl,reason,xp=sim_pct(opt_ts,opt_ps,ep,eod_ns,tgt,sl)
                    new_records.append(dict(idea='A_new_zones',date=dstr,zone=zone,
                        bias=bias,opt=signal,strike_type=stype,entry_time=et,
                        ep=ep,xp=xp,exit_reason=reason,pnl=pnl,target_pct=tgt,sl_pct=sl))

print(f"  Done {time.time()-t1:.0f}s — {dict(day_count)} — {len(new_records)} records")


# ── Pass 3: body filter 0.05% extra days ────────────────────────────
print("Pass 3: body filter 0.05% — extra days between 0.05–0.10%...")
t2=time.time(); body_records=[]; body_days=0

for date in dates_5yr:
    idx=all_dates.index(date)
    if idx<1: continue
    prev=all_dates[idx-1]
    if prev not in daily_ohlc or date not in daily_ohlc: continue

    ph,pl,pc,_=daily_ohlc[prev]; _,_,_,today_op=daily_ohlc[date]
    prev_open=daily_ohlc[prev][3]
    prev_body=round(abs(pc-prev_open)/prev_open*100,3)
    # Only days that pass 0.05 but fail 0.10 (the extra days from loosening filter)
    if not (0.05 < prev_body <= 0.10): continue

    pvt=compute_pivots(ph,pl,pc)
    e20=ema_s.get(date,np.nan)
    if np.isnan(e20): continue

    bias=('bull' if today_op>e20 else 'bear')
    zone=classify_zone(today_op,pvt,ph,pl)
    if (zone,bias,get_v17a_signal(zone,bias)) == (zone,bias,None): continue
    signal=get_v17a_signal(zone,bias)
    if signal is None: continue
    if (zone,bias,signal) not in EXISTING_PARAMS: continue

    stype,et,tgt,sl_p=EXISTING_PARAMS[(zone,bias,signal)]
    body_days+=1
    dstr=f'{date[:4]}-{date[4:6]}-{date[6:]}'
    exps=list_expiry_dates(date)
    if not exps: continue
    expiry=exps[0]
    exp_dt=pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte=(exp_dt-pd.Timestamp(dstr)).days
    if dte==0: continue

    atm=int(round(today_op/STRIKE_INT)*STRIKE_INT)
    strike=get_strike(atm,signal,stype)
    inst=f'NIFTY{expiry}{strike}{signal}'
    eod_ns=pd.Timestamp(dstr+' '+EOD_EXIT).value

    tk=load_tick_data(date,inst,et[:5]+':00',EOD_EXIT)
    if tk is None or tk.empty: continue
    ep_mask=tk['time']>=et
    if not ep_mask.any(): continue
    ep=r2(float(tk[ep_mask].iloc[0]['price']))
    if ep<5 or ep/today_op*100<=0.47: continue
    opt_ts=tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
    opt_ps=tk[ep_mask]['price'].values.astype(float)
    pnl,reason,xp=sim_pct(opt_ts,opt_ps,ep,eod_ns,tgt,sl_p)
    body_records.append(dict(idea='B_body_filter',date=dstr,zone=zone,
        bias=bias,opt=signal,strike_type=stype,entry_time=et,
        ep=ep,xp=xp,exit_reason=reason,pnl=pnl,target_pct=tgt,sl_pct=sl_p))

print(f"  Done {time.time()-t2:.0f}s — {body_days} extra days — {len(body_records)} trades")


# ── Pass 4: second trade after early v17a exit ──────────────────────
print("Pass 4: second trade after early v17a exit (before 11:00)...")
t3=time.time(); second_records=[]; second_days=0

# Load existing v17a trades to find early exits
v17a_df = pd.read_csv('data/20260420/38_zone_v17a_trades.csv', parse_dates=['date'])
early_exits = v17a_df[v17a_df['entry_time'].notna()].copy()
early_exits['exit_dt'] = pd.to_datetime(early_exits['date'].dt.strftime('%Y-%m-%d') + ' ' + early_exits['entry_time'])
# Approximate exit time: entry_time + 30min for target, 60min for others (rough)
# Better: filter by exit_reason=target AND entry_time before 10:30
early_exits = early_exits[
    (early_exits['exit_reason']=='target') &
    (early_exits['entry_time'] <= '10:30:00')
]
print(f"  Early target exits (before 10:30): {len(early_exits)}")

for _, row in early_exits.iterrows():
    date=row['date'].strftime('%Y%m%d')
    dstr=row['date'].strftime('%Y-%m-%d')
    if date not in daily_ohlc: continue

    ph_=None
    idx=all_dates.index(date) if date in all_dates else -1
    if idx<1: continue
    prev=all_dates[idx-1]
    if prev not in daily_ohlc: continue

    ph,pl,pc,_=daily_ohlc[prev]; _,_,_,today_op=daily_ohlc[date]
    pvt=compute_pivots(ph,pl,pc)

    # Build 5min OHLC from spot ticks
    spot_tks=load_spot_data(date,'NIFTY')
    if spot_tks is None or spot_tks.empty: continue
    spot_tks['dt']=pd.to_datetime(dstr+' '+spot_tks['time'])
    sp=spot_tks[['dt','price']].set_index('dt').rename(columns={'price':'p'})
    ohlc5=sp['p'].resample('5min',closed='left',label='left').agg(
        open='first',high='max',low='min',close='last').dropna()
    if len(ohlc5)<2: continue

    # Scan starts after v17a exit time
    exit_hhmm=row['entry_time'][:5]
    brk=detect_intraday_break(ohlc5,pvt,ph,pl,
                              scan_from=exit_hhmm, scan_to='11:20')
    if brk is None: continue
    key2=(brk['level_name'],brk['opt'])
    if key2 not in INTRADAY_PARAMS: continue

    stype2,tgt2,sl2=INTRADAY_PARAMS[key2]
    exps=list_expiry_dates(date)
    if not exps: continue
    expiry=exps[0]
    exp_dt=pd.Timestamp(f'20{expiry[:2]}-{expiry[2:4]}-{expiry[4:]}')
    dte=(exp_dt-pd.Timestamp(dstr)).days
    if dte==0: continue

    atm=int(round(today_op/STRIKE_INT)*STRIKE_INT)
    strike=get_strike(atm,brk['opt'],stype2)
    inst=f'NIFTY{expiry}{strike}{brk["opt"]}'
    entry_str=brk['entry_dt'].strftime('%H:%M:%S')
    eod_ns=pd.Timestamp(dstr+' '+EOD_EXIT).value

    tk=load_tick_data(date,inst,entry_str[:5]+':00',EOD_EXIT)
    if tk is None or tk.empty: continue
    ep_mask=tk['time']>=entry_str
    if not ep_mask.any(): continue
    ep=r2(float(tk[ep_mask].iloc[0]['price']))
    if ep<5: continue

    opt_ts=tk[ep_mask]['date_time'].values.astype('datetime64[ns]').astype('int64')
    opt_ps=tk[ep_mask]['price'].values.astype(float)
    pnl,reason,xp=sim_pct(opt_ts,opt_ps,ep,eod_ns,tgt2,sl2)
    second_days+=1
    second_records.append(dict(idea='C_second_trade',date=dstr,
        zone=brk['level_name'],bias='—',opt=brk['opt'],
        strike_type=stype2,entry_time=entry_str,
        ep=ep,xp=xp,exit_reason=reason,pnl=pnl,
        target_pct=tgt2,sl_pct=sl2))

print(f"  Done {time.time()-t3:.0f}s — {second_days} second trades")


# ── Summary ─────────────────────────────────────────────────────────
all_records = new_records + body_records + second_records
df_all = pd.DataFrame(all_records)

print("\n" + "="*65)
print("SUMMARY — Additional Trades & Profit")
print("="*65)

# Idea A — best config per new zone
print("\nIDEA A: New zones (pdh_to_r1+bull, pdl_to_bc+bear, s2_to_s3+bear)")
if new_records:
    df_new = pd.DataFrame(new_records)
    grp = df_new.groupby(['zone','strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
        n='count', wr=lambda x:round((x>0).mean()*100,1),
        avg=lambda x:round(x.mean(),0), total=lambda x:round(x.sum(),0)
    ).reset_index()
    for z in ['pdh_to_r1','pdl_to_bc','s2_to_s3']:
        zg = grp[(grp.zone==z) & (grp.n>=10)].sort_values('total',ascending=False)
        if zg.empty: print(f"  {z}: no valid config (insufficient trades)"); continue
        b=zg.iloc[0]
        print(f"  {z}: {b.strike_type} {b.entry_time} tgt={b.target_pct:.0%} sl={b.sl_pct}x"
              f"  →  n={int(b.n)} WR={b.wr}% avg=Rs{b.avg:,.0f} total=Rs{b.total:,.0f}/5yr")

print("\nIDEA B: Body filter 0.05%")
if body_records:
    df_b=pd.DataFrame(body_records)
    n=len(df_b); wr=round((df_b.pnl>0).mean()*100,1)
    total=round(df_b.pnl.sum(),0); avg=round(df_b.pnl.mean(),0)
    print(f"  Extra trades: {n}/5yr ({n/5:.0f}/yr)  WR={wr}%  Avg=Rs{avg:,.0f}  Total=Rs{total:,.0f}/5yr")
else:
    print("  No extra trades found")

print("\nIDEA C: Second trade after early exit")
if second_records:
    df_c=pd.DataFrame(second_records)
    n=len(df_c); wr=round((df_c.pnl>0).mean()*100,1)
    total=round(df_c.pnl.sum(),0); avg=round(df_c.pnl.mean(),0)
    print(f"  Extra trades: {n}/5yr ({n/5:.0f}/yr)  WR={wr}%  Avg=Rs{avg:,.0f}  Total=Rs{total:,.0f}/5yr")
else:
    print("  No second trades found")

# Combined projection
print("\n" + "="*65)
print("COMBINED IMPACT PROJECTION")
print("="*65)
base_total=480435; base_trades=465; base_yrs=5
add_total=0; add_trades=0

if new_records:
    df_new=pd.DataFrame(new_records)
    # Use best config per zone
    for z,bias_z,sig_z in [('pdh_to_r1','bull','PE'),('pdl_to_bc','bear','CE'),('s2_to_s3','bear','CE')]:
        zg=df_new[(df_new.zone==z)]
        if zg.empty: continue
        grp2=zg.groupby(['strike_type','entry_time','target_pct','sl_pct'])['pnl'].agg(
            n='count',total='sum').reset_index()
        grp2=grp2[grp2.n>=10].sort_values('total',ascending=False)
        if grp2.empty: continue
        b=grp2.iloc[0]
        mask2=(zg.strike_type==b.strike_type)&(zg.entry_time==b.entry_time)&\
              (zg.target_pct==b.target_pct)&(zg.sl_pct==b.sl_pct)
        add_total+=round(zg[mask2].pnl.sum(),0)
        add_trades+=len(zg[mask2])

if body_records:
    df_b=pd.DataFrame(body_records)
    add_total+=round(df_b.pnl.sum(),0); add_trades+=len(df_b)

if second_records:
    df_c=pd.DataFrame(second_records)
    add_total+=round(df_c.pnl.sum(),0); add_trades+=len(df_c)

new_total=base_total+add_total; new_trades=base_trades+add_trades
new_per_yr=new_total/base_yrs; new_per_wk=new_per_yr/52
new_pts_wk=new_per_wk/75

print(f"  Base (current)    : {base_trades} trades  Rs{base_total:,.0f}/5yr  "
      f"Rs{base_total/base_yrs:,.0f}/yr  {base_total/base_yrs/52/75:.1f}pts/wk")
print(f"  Additional        : +{add_trades} trades  +Rs{add_total:,.0f}/5yr")
print(f"  Combined (1 lot)  : {new_trades} trades  Rs{new_total:,.0f}/5yr  "
      f"Rs{new_per_yr:,.0f}/yr  {new_pts_wk:.1f}pts/wk")
print(f"  Combined (2 lots) : {new_pts_wk*2:.1f}pts/wk  {'✅ HITS 100' if new_pts_wk*2>=100 else '❌ below 100'}")

# Save
os.makedirs(OUT_DIR, exist_ok=True)
out=f'{OUT_DIR}/52_more_trades_backtest.csv'
df_all.to_csv(out,index=False)
print(f"\nSaved → {out}  ({len(df_all)} records)")
