#!/usr/bin/env python3
"""
NIFTY Options Backtesting Utility Library

QUICK REFERENCE:
================
| Function              | Example                                                      |
|-----------------------|--------------------------------------------------------------|
| load_spot_data        | load_spot_data("20250115", "NIFTY")                          |
| load_option_data      | load_option_data("20250115", 21350, "250116", "CE")          |
| fetch_spot_price      | fetch_spot_price("20250115", "09:16:00")                     |
| create_spot_ohlc      | create_spot_ohlc(df, "1min")                                 |
| create_option_ohlc    | create_option_ohlc(df, "CE", 21350)                          |
| calculate_strike      | calculate_strike(24556, "PE", strike_position="otm_1")       |
| build_instrument_name | build_instrument_name("NIFTY", 25450, "20251024", True, "PE")|
| list_expiry_dates     | list_expiry_dates("20250115")                                |
| super_plotter         | super_plotter(".", fig, "chart", "20250115", ["svg"])        |
| save_dataframe        | save_dataframe(df, ".", "ohlc", "20250115")                  |
| list_trading_dates    | list_trading_dates("202501")                                 |

DATA FORMAT:
- Tick CSV: date,time,price,volume,open_interest (no header)
- Spot file: {DATA_FOLDER}/{YYYYMMDD}/NIFTY.csv
- Option file: {DATA_FOLDER}/{YYYYMMDD}/NIFTY{YYMMDD}{strike}{CE|PE}.csv
"""

import os
import glob
import shutil
import pandas as pd
from datetime import datetime
import calendar
import subprocess
import json
import plotly.io as pio

# Global configuration
DATA_FOLDER = os.environ.get("INTER_SERVER_DATA_PATH", "/mnt/data/day-wise")  

def load_spot_data(date_str, instrument='NIFTY'):
    """>>> df = load_spot_data("20250115", "NIFTY")"""
    file_path = f"{DATA_FOLDER}/{date_str}/{instrument}.csv"
    try:
        df = pd.read_csv(file_path, header=None, names=['date', 'time', 'price', 'volume', 'open_interest'])
        df['date_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
        return df.sort_values('date_time').reset_index(drop=True)
    except FileNotFoundError:
        return None


def load_option_data(date_str, strike, expiry, option_type, instrument='NIFTY'):
    """>>> df = load_option_data("20250115", 21350, "250116", "CE", "NIFTY")"""
    file_path = f"{DATA_FOLDER}/{date_str}/{instrument}{expiry}{strike}{option_type}.csv"
    df = pd.read_csv(file_path, header=None, names=['date', 'time', 'price', 'volume', 'open_interest'])
    df['date_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'])
    return df.sort_values('date_time').reset_index(drop=True)


def fetch_spot_price(date_str, time='09:16:00', instrument='NIFTY'):
    """>>> price = fetch_spot_price("20250115", "09:16:00", "NIFTY")"""
    df = load_spot_data(date_str, instrument)
    if df is None:
        return None

    spot_at_time = df[df['time'] == time]
    if spot_at_time.empty:
        return None

    spot_price = spot_at_time.iloc[0]['price']
    return spot_price


def lookup_strike_interval(index_name):
    """>>> lookup_strike_interval("NIFTY")  # Returns 50"""
    strike_intervals = {
        'NIFTY': 50,
        'BANKNIFTY': 100,
        'FINNIFTY': 50,
        'SENSEX': 100
    }

    if index_name not in strike_intervals:
        raise ValueError(f"Invalid index_name: {index_name}. Must be one of {list(strike_intervals.keys())}")

    return strike_intervals[index_name]


def list_expiry_dates(date_str, index_name='NIFTY'):
    """>>> list_expiry_dates('20181010')  # Returns ['181025', '181129', ...]"""
    data_dir = f"{DATA_FOLDER}/{date_str}/"

    if not os.path.exists(data_dir):
        return []

    pattern = f"{index_name}[0-9]*[CE|PE].csv"
    files = glob.glob(os.path.join(data_dir, pattern))

    expiry_dates_set = set()

    for file in files:
        filename = os.path.basename(file)

        if filename.startswith(index_name) and (filename.endswith('CE.csv') or filename.endswith('PE.csv')):
            start_pos = len(index_name)

            if len(filename) > start_pos + 6:
                expiry_6digit = filename[start_pos:start_pos + 6]

                if expiry_6digit.isdigit():
                    expiry_dates_set.add(expiry_6digit)

    return sorted(list(expiry_dates_set))


def build_instrument_name(index_name, strike, current_date, is_current_expiry, option_type):
    """>>> build_instrument_name('NIFTY', 25450, '20251024', True, 'PE')"""
    if option_type not in ['CE', 'PE']:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'CE' or 'PE'")

    expiry_dates = list_expiry_dates(current_date, index_name)

    if not expiry_dates:
        raise ValueError(f"No expiry dates found for {index_name} on {current_date}")

    current_date_6digit = current_date[2:]

    current_expiry = None
    next_expiry = None

    for i, expiry in enumerate(expiry_dates):
        if expiry >= current_date_6digit:
            current_expiry = expiry
            if i + 1 < len(expiry_dates):
                next_expiry = expiry_dates[i + 1]
            break

    if current_expiry is None:
        raise ValueError(f"No valid expiry found for {index_name} on or after {current_date}")

    selected_expiry = current_expiry if is_current_expiry else next_expiry

    if selected_expiry is None:
        raise ValueError(f"Next expiry not available for {index_name} on {current_date}")

    instrument_name = f"{index_name}{selected_expiry}{strike}{option_type}"

    return instrument_name


def load_tick_data(date_str, instrument_name, entry_time, exit_time='15:20:00'):
    """>>> df = load_tick_data('20181010', 'NIFTY18102510350PE', '09:16:00')"""
    df = load_spot_data(date_str, instrument=instrument_name)

    if df is None:
        return None

    filtered_df = df[(df['time'] >= entry_time) & (df['time'] <= exit_time)]

    return filtered_df.reset_index(drop=True)


def calculate_strike(spot_price, option_type, index_name='NIFTY', strike_position='atm'):
    """>>> calculate_strike(24556, 'PE', strike_position='otm_1')  # Returns 24500"""
    interval = lookup_strike_interval(index_name)

    if option_type not in ['CE', 'PE']:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'CE' or 'PE'")

    strike_position = strike_position.lower()
    if strike_position == 'atm':
        position_value = 0
    elif strike_position.startswith('otm_'):
        try:
            position_value = int(strike_position.split('_')[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid strike_position: {strike_position}. Expected format: 'otm_1', 'otm_2', etc.")
    elif strike_position.startswith('itm_'):
        try:
            position_value = int(strike_position.split('_')[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid strike_position: {strike_position}. Expected format: 'itm_1', 'itm_2', etc.")
    else:
        raise ValueError(f"Invalid strike_position: {strike_position}. Must be 'atm', 'otm_N', or 'itm_N'")

    atm_strike = int(round(spot_price / interval) * interval)

    if strike_position == 'atm':
        return atm_strike
    elif strike_position.startswith('otm_'):
        if option_type == 'PE':
            return atm_strike - (interval * position_value)
        else:
            return atm_strike + (interval * position_value)
    else:
        if option_type == 'PE':
            return atm_strike + (interval * position_value)
        else:
            return atm_strike - (interval * position_value)


def create_folder_structure(output_path):
    """>>> create_folder_structure("./output")"""
    os.makedirs(output_path, exist_ok=True)
    return output_path


def create_option_ohlc(tick_data, option_type, strike, timeframe='1min'):
    """>>> ohlc = create_option_ohlc(tick_df, "CE", 21350, "1min")"""
    trading_data = tick_data[
        (tick_data['date_time'].dt.time >= pd.Timestamp('09:15:00').time()) &
        (tick_data['date_time'].dt.time <= pd.Timestamp('15:30:00').time()) &
        (tick_data['volume'] > 0)
    ].copy()

    trading_data['time_group'] = trading_data['date_time'].dt.floor(timeframe)

    ohlc_agg = trading_data.groupby('time_group').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum',
        'open_interest': 'last'
    }).reset_index()

    ohlc_agg.columns = ['date_time', 'open', 'high', 'low', 'close', 'volume', 'open_interest']
    ohlc_agg['oi_change'] = ohlc_agg['open_interest'].diff()
    ohlc_agg['strike'] = strike
    ohlc_agg['option'] = option_type

    return ohlc_agg


def create_spot_ohlc(tick_data, timeframe='1min'):
    """>>> ohlc = create_spot_ohlc(tick_df, "1min")"""
    trading_data = tick_data[
        (tick_data['date_time'].dt.time >= pd.Timestamp('09:15:00').time()) &
        (tick_data['date_time'].dt.time <= pd.Timestamp('15:30:00').time())
    ].copy()

    trading_data['time_group'] = trading_data['date_time'].dt.floor(timeframe)

    ohlc_agg = trading_data.groupby('time_group').agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    }).reset_index()

    ohlc_agg.columns = ['date_time', 'open', 'high', 'low', 'close', 'volume']

    return ohlc_agg


def save_dataframe(df, folder_path, base_name, date_str, ext='csv'):
    """>>> path = save_dataframe(df, ".", "ohlc", "20250115")"""
    date_folder = os.path.join(folder_path, 'data', date_str)
    os.makedirs(date_folder, exist_ok=True)
    file_path = os.path.join(date_folder, f"{base_name}.{ext}")
    df.to_csv(file_path, index=False)
    return file_path


def generate_ohlc_pair(date_str, strike, expiry, folder_path, timeframe='1min', instrument='NIFTY'):
    """>>> ce_ohlc, pe_ohlc = generate_ohlc_pair("20250115", 21350, "250116", ".")"""
    ce_data = load_option_data(date_str, strike, expiry, 'CE', instrument)
    ce_ohlc = create_option_ohlc(ce_data, 'CE', strike, timeframe)
    ce_file = save_dataframe(ce_ohlc, folder_path, f'01_atm_ce_ohlc_{timeframe}', date_str)

    pe_data = load_option_data(date_str, strike, expiry, 'PE', instrument)
    pe_ohlc = create_option_ohlc(pe_data, 'PE', strike, timeframe)
    pe_file = save_dataframe(pe_ohlc, folder_path, f'02_atm_pe_ohlc_{timeframe}', date_str)

    return ce_ohlc, pe_ohlc


def generate_calendar_dates(date_input):
    """>>> generate_calendar_dates("202501")  # Returns ['20250101', '20250102', ...]"""
    if not date_input.isdigit():
        raise ValueError(f"Invalid format: {date_input}. Must be digits only.")

    if len(date_input) == 8:
        try:
            datetime.strptime(date_input, '%Y%m%d')
            return [date_input]
        except ValueError:
            raise ValueError(f"Invalid date: {date_input}")

    elif len(date_input) == 6:
        try:
            year = int(date_input[:4])
            month = int(date_input[4:6])
            if month < 1 or month > 12:
                raise ValueError(f"Invalid month: {month}")

            _, last_day = calendar.monthrange(year, month)
            dates = []
            for day in range(1, last_day + 1):
                dates.append(f"{year:04d}{month:02d}{day:02d}")
            return dates
        except ValueError:
            raise ValueError(f"Invalid year/month: {date_input}")

    elif len(date_input) == 4:
        try:
            year = int(date_input)
            dates = []
            for month in range(1, 13):
                _, last_day = calendar.monthrange(year, month)
                for day in range(1, last_day + 1):
                    dates.append(f"{year:04d}{month:02d}{day:02d}")
            return dates
        except ValueError:
            raise ValueError(f"Invalid year: {date_input}")

    else:
        raise ValueError(f"Invalid format: {date_input}. Must be YYYY (4), YYYYMM (6), or YYYYMMDD (8) digits.")


def list_trading_dates(date_input=None):
    """>>> list_trading_dates("202501")  # Returns trading dates for Jan 2025"""
    all_dates = sorted([d for d in os.listdir(DATA_FOLDER) if d.isdigit() and len(d) == 8])

    if date_input is None:
        return all_dates

    if not date_input.isdigit():
        raise ValueError(f"Invalid format: {date_input}. Must be digits only.")

    if len(date_input) == 8:
        return [date_input] if date_input in all_dates else []
    elif len(date_input) == 6:
        return [d for d in all_dates if d.startswith(date_input)]
    elif len(date_input) == 4:
        return [d for d in all_dates if d.startswith(date_input)]
    else:
        raise ValueError(f"Invalid format: {date_input}. Must be YYYY (4), YYYYMM (6), or YYYYMMDD (8) digits.")


def calculate_margin(action, ltp):
    """>>> margin = calculate_margin("SELL", 150.0)"""
    if action == 'BUY':
        return 75 * ltp * 1.10
    elif action == 'SELL':
        return 250000
    else:
        raise ValueError("Action must be 'BUY' or 'SELL'")


def fetch_option_chain(date_str, time_str, strikes_range, expiry_offset=0):
    """
    >>> chain = fetch_option_chain("20250115", "09:16:00", 5, 0)

    Example output:
        {
          "analysis": {
            "date": "20181010",
            "atm_strike": 10350,
            "expiry": "181025",
            "nifty_level": 10354.3
          },
          "option_chain": [
            {"strike": 10350, "ce_price": 165.75, "pe_price": 146.95, "is_atm": true}
          ]
        }
    """
    try:
        nifty_fetcher_path = os.path.expanduser('~/bin/nifty_fetcher_v2.out')
        result = subprocess.run(
            [nifty_fetcher_path, date_str, time_str, str(strikes_range), str(expiry_offset)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            json_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break

            if json_start >= 0:
                json_text = '\n'.join(lines[json_start:])
                data = json.loads(json_text)
                # Convert 8-digit expiry (YYYYMMDD) to 6-digit (YYMMDD) to match file naming
                if data and 'analysis' in data and 'expiry' in data['analysis']:
                    expiry_8digit = data['analysis']['expiry']
                    data['analysis']['expiry'] = expiry_8digit[2:]  # Strip first 2 digits
                return data

        return None
    except Exception:
        return None


def apply_with_history(current_day_df, instrument_name, calculation_func):
    """>>> result = apply_with_history(current_df, 'NIFTY', add_sma_func)"""
    if 'date_time' not in current_day_df.columns:
        raise ValueError("current_day_df must have 'date_time' column")

    if not pd.api.types.is_datetime64_any_dtype(current_day_df['date_time']):
        current_day_df = current_day_df.copy()
        current_day_df['date_time'] = pd.to_datetime(current_day_df['date_time'])

    current_date_str = current_day_df['date_time'].iloc[0].strftime('%Y%m%d')

    if len(current_day_df) >= 2:
        time_diff = current_day_df['date_time'].iloc[1] - current_day_df['date_time'].iloc[0]
        timeframe = f"{int(time_diff.total_seconds() / 60)}min"
    else:
        timeframe = '1min'

    all_dates = list_trading_dates()
    current_date_index = None

    try:
        current_date_index = all_dates.index(current_date_str)
    except ValueError:
        raise ValueError(f"Current date {current_date_str} not found in trading dates")

    if current_date_index == 0:
        result_df = current_day_df.copy()
        result_df = calculation_func(result_df)
        return result_df

    prev_date_str = all_dates[current_date_index - 1]
    is_option = any(instrument_name.endswith(suffix) for suffix in ['CE', 'PE'])
    prev_tick_data = load_spot_data(prev_date_str, instrument=instrument_name)

    if prev_tick_data is None:
        result_df = current_day_df.copy()
        result_df = calculation_func(result_df)
        return result_df

    if is_option:
        index_name = None
        for idx in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX']:
            if instrument_name.startswith(idx):
                index_name = idx
                break

        if index_name is None:
            raise ValueError(f"Cannot parse instrument name: {instrument_name}")

        option_type = instrument_name[-2:]
        strike_start = len(index_name) + 6
        strike = int(instrument_name[strike_start:-2])

        prev_ohlc = create_option_ohlc(prev_tick_data, option_type, strike, timeframe)
    else:
        prev_ohlc = create_spot_ohlc(prev_tick_data, timeframe)

    # Remove 15:30 candle from previous day (partial/incomplete data)
    prev_ohlc = prev_ohlc[prev_ohlc['date_time'].dt.time < pd.Timestamp('15:30:00').time()].reset_index(drop=True)

    combined_df = pd.concat([prev_ohlc, current_day_df], ignore_index=True)
    combined_df = calculation_func(combined_df)

    prev_rows_count = len(prev_ohlc)
    result_df = combined_df.iloc[prev_rows_count:].reset_index(drop=True)

    return result_df


def load_instrument_data(instrument_name, entry_date, entry_time):
    """>>> df = load_instrument_data('NIFTY181025010350CE', '20181010', '09:25:00')"""
    # Check if instrument is an option (has expiry embedded)
    is_option = any(instrument_name.endswith(suffix) for suffix in ['CE', 'PE'])

    if not is_option:
        # For spot instruments, just return single day data
        return load_tick_data(entry_date, instrument_name, entry_time, '15:30:00')

    # Extract index name
    index_name = None
    for idx in ['BANKNIFTY', 'FINNIFTY', 'SENSEX', 'NIFTY']:  # Check longer names first
        if instrument_name.startswith(idx):
            index_name = idx
            break

    if index_name is None:
        raise ValueError(f"Cannot parse instrument name: {instrument_name}")

    # Extract 6-digit expiry (YYMMDD) from instrument name
    expiry_6digit = instrument_name[len(index_name):len(index_name) + 6]

    if not expiry_6digit.isdigit() or len(expiry_6digit) != 6:
        raise ValueError(f"Invalid expiry format in instrument name: {instrument_name}")

    # Convert 6-digit expiry to 8-digit (YYMMDD → YYYYMMDD)
    expiry_date = '20' + expiry_6digit

    # Get all trading dates between entry and expiry
    all_trading_dates = list_trading_dates()
    dates_in_range = [d for d in all_trading_dates if entry_date <= d <= expiry_date]

    if not dates_in_range:
        return None

    all_data = []

    for i, date_str in enumerate(dates_in_range):
        if i == 0:
            # First day: from entry_time to 15:30:00
            df = load_tick_data(date_str, instrument_name, entry_time, '15:30:00')
        else:
            # Subsequent days: full trading day
            df = load_tick_data(date_str, instrument_name, '09:15:00', '15:30:00')

        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        return None

    # Concatenate all data
    result_df = pd.concat(all_data, ignore_index=True)
    return result_df


# ─── Chart functions: use the sa-kron-chart skill ───
# Do NOT import plot_util here. Use /sa-kron-chart skill for all chart operations.

