# CPR Advanced — v17a + Camarilla Touch Strategy

NIFTY weekly options selling strategy based on CPR zone classification, EMA(20) bias, and Camarilla pivot touch triggers.

## Strategy Overview

### Strategy 1: v17a (Morning Open)
- Compute previous day's CPR (Central Pivot Range) and Pivot levels
- Classify today's opening price into one of 15 zones
- Determine EMA(20) bias (bull/bear)
- If zone+bias matches a known signal → sell CE or PE at the configured entry time
- Body filter: skip if previous day candle body < 0.10% (flat day = no edge)

### Strategy 2: Camarilla Touch (Intraday — only on non-v17a days)
- Compute Camarilla H3/L3 levels from previous day's H/L/C
- If H3 falls inside CPR band → sell CE OTM1 when spot touches H3 intraday
  - Skip 09:20–09:30 window (false-touch zone)
- If L3 falls inside CPR band → sell PE ITM1 when spot touches L3 intraday
- Sequential rule: if v17a already traded today, skip Camarilla

### Exit Rules (3-tier trailing)
- Target: 50% / 20% decay depending on zone
- Break-even trail: locks SL at entry when 25% decay reached
- 80% lock: when 40% decay reached
- 95% trail: when 60% decay reached
- EOD exit: 15:20 IST

## Backtest Results (5 years, 1 lot = 75 units)
- Combined: 480 trades, WR=71.2%, P&L=Rs.4,99,181/5yr
- Max drawdown: Rs.-21,158
- 25.6 pts/week (1 lot) → ~102 pts/week at 4 lots

## Files

| File | Description |
|------|-------------|
| `config.py` | Strategy parameters, API credentials (fill before use) |
| `strategy.py` | Core logic: pivots, zone classifier, EMA, TradeState |
| `trader.py` | Live trader: Angel One + OpenAlgo integration |
| `angelone.py` | Angel One Smart API client |
| `openalgo.py` | OpenAlgo paper/live order client |
| `dashboard.py` | Streamlit performance dashboard |
| `my_util.py` | Backtest utility: data loading, OHLC, option chain |
| `forward_bias_test.py` | Forward bias audit for signal indicators |
| `optimize_intraday_v2.py` | Intraday v2 optimization script |
| `51_*.py` | SENSEX Tuesday backtest research |
| `52_*.py` | More trades / additional zones research |
| `53_*.py` | CPR filters test (Virgin CPR, Width, Weekly alignment) |
| `54_*.py` | Camarilla open-time backtest (superseded) |
| `55_*.py` | Camarilla touch trigger backtest |
| `56_*.py` | Combined full backtest (v17a + Camarilla) — final |

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.py` with your credentials:
```python
ANGELONE_API_KEY   = "..."
ANGELONE_CLIENT_ID = "..."
ANGELONE_PASSWORD  = "..."
ANGELONE_TOTP_KEY  = "..."
OPENALGO_API_KEY   = "..."
```

## Running

```bash
# Paper trade (default)
python trader.py

# Dry run — signals only, no orders
python trader.py --dry-run

# Live orders
python trader.py --live

# Dashboard
streamlit run dashboard.py
```

## Expiry Handling
- NIFTY weekly expiry: Tuesday
- On expiry day (DTE=0): automatically skips to next week's expiry
- Holiday adjustment: Monday used when Tuesday is a market holiday

## Notes
- EMA(20) requires 40+ days of seed data — trader fetches 50 days OHLC
- Body filter applied to previous day (not today) to avoid forward bias
- No overlapping trades — one position at a time
- Data path: set `INTER_SERVER_DATA_PATH` env var for backtest data
