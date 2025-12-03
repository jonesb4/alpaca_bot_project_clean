#!/usr/bin/env python3
import json
import os
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import requests # Necessary for Telegram API calls
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# ---------------- CONFIGURATION ----------------
try:
    API_KEY = os.environ['ALPACA_API_KEY']
    API_SECRET = os.environ['ALPACA_SECRET_KEY']

    if not API_KEY or not API_SECRET:
        raise ValueError("Alpaca credentials are empty")

    print("✅ Alpaca credentials loaded successfully")

except KeyError as e:
    print(f"❌ ERROR: Missing required environment variable: {e}")
    raise SystemExit("Cannot start bot without Alpaca credentials")

BASE_URL = "https://paper-api.alpaca.markets"
# --- Strategy Configuration Dictionary ---
STRATEGY_CONFIG = {
    "MoneyInvested": 2000,                # Total dollar amount available to invest per trade
    "LOOKBACK_BARS": 60,                  # 5-minute bars (5 hours of data lookback)
    "MIN_REQUIRED_SUPPORT_HITS": 3,       # Minimum required rising 5-minute support points
    "SHORT_TERM_SLOPE_LOOKBACK": 12,      # Number of 5-minute bars for short-term trend check (1 hour)
    "STOP_LOSS_PCT": 0.055,               # Catastrophic Stop Loss attached to bracket order
    "TAKE_PROFIT_PCT": 0.08,              # Local target for 50% profit taking
    "TRAILING_STOP_PCT": 0.04             # Trailing stop percentage after TP is hit
}

POLL_INTERVAL_SECONDS = 64          # Delay between main loops
DATA_FEED = 'iex'                   # Alpaca data feed selection (e.g., iex, sip)

# ---------------- TELEGRAM CONFIGURATION (CLEAN & FINAL) ----------------
# !!! REPLACE THESE WITH YOUR OWN VALUES !!!
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

try:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # Check if either variable is None or empty
        raise KeyError('One or more Telegram credentials are missing or empty')

    print("✅ Telegram credentials loaded successfully")
    TELEGRAM_ENABLED = True

except KeyError as e:
    # This block runs if credentials are not found in the environment
    print("⚠️ WARNING: Telegram credentials not found - notifications disabled")
    # You can keep the log message simple now that we know the fix works
    TELEGRAM_BOT_TOKEN = None
    TELEGRAM_CHAT_ID = None
    TELEGRAM_ENABLED = False

    print("bot initialized") # This is when you send your text!

while True:
    try:
        # 1. Place your core trading logic function call here
        perform_trade_check() 
        
        # 2. Wait for a set amount of time before the next check
        print(f"[{datetime.datetime.now()}] Sleeping for 5 minutes...")
        time.sleep(300) # Sleeps for 300 seconds (5 minutes)

    except Exception as e:
        # Crucial: Catch errors and print them so they appear in logs!
        print(f"An error occurred during runtime: {e}")
        # Consider a small sleep to avoid thrashing restarts
        time.sleep(10)
# ------------------------------------------------------------------------

# Global State Variables
symbol_state = {}
positions = {}
last_summary_time = datetime.now()
closed_trades_pl = {} # Tracks P/L for trades that have been fully closed.
is_first_run = True # Flag to force log the first analysis run


# Initialize Alpaca API connection
try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account_status = api.get_account().status
    print(f"✅ Connected to Alpaca PAPER account. Status: {account_status}")
except Exception as e:
    print(f"❌ Failed to connect to Alpaca API: {e}")
    raise SystemExit("Cannot start bot without Alpaca connection")

# ---------------- TELEGRAM & LOGGING FUNCTIONS ----------------

def send_telegram_message(message):
    """Sends a message to the configured Telegram chat using HTML parsing."""
    if not TELEGRAM_ENABLED:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # 1. Prepare message for HTML (replace Markdown bolding with <b> tags)
    # This ensures consistency with the EOD report and log_trade replacements.
    html_message = message.replace('**', '<b>').replace('</b>', '</b>')

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": html_message,
        "parse_mode": "HTML" # <--- CRITICAL FIX: Use HTML for reliability
    }

    try:
        response = requests.post(url, data=payload, timeout=5)
        response.raise_for_status()

        # 2. Check the JSON response for actual delivery status
        json_response = response.json()
        if not json_response.get('ok'):
            log_trade(f"❌ Telegram API failure ('ok': false, HTTP 200). Response: {response.text}")

    except requests.exceptions.Timeout:
        log_trade(f"❌ Telegram alert failed: Request timed out.")
    except requests.exceptions.RequestException as e:
        log_trade(f"❌ Telegram alert failed: Network or API Error: {e}")
    except Exception as e:
        log_trade(f"❌ Telegram alert failed: Unknown error: {e}")


def log_trade(message):
    """Prints a timestamped log message and sends it to Telegram if it's a trade alert."""
    full_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}"
    print(full_message)

    # Keywords that indicate a trade execution or critical event
    is_trade_alert = any(keyword in message for keyword in ["🟢 BOUGHT", "⭐ TAKE PROFIT", "🛑 TRAILING STOP HIT", "🟥 EOD SELL", "❌ Buy failed", "❌ EOD SELL FAILED", "🛑 CATASTROPHIC STOP HIT"])

    if is_trade_alert:
        # Prepare a formatted message for Telegram using HTML tags for coloring/emphasis
        formatted_message = full_message.replace('🟢', '✅ <b>BUY</b>').replace('⭐', '💰 <b>TP/TS</b>').replace('🛑', '🚨 <b>TS HIT</b>').replace('🟥', '🔻 <b>EOD SELL</b>').replace('❌', '🔴 <b>FAILURE</b>')
        send_telegram_message(formatted_message)

# ---------------- STATE PERSISTENCE CONFIG ----------------
STATE_FILE = 'bot_state.json'

# ---------------- STATE PERSISTENCE FUNCTIONS ----------------

def save_state():
    """Saves the local 'positions' dictionary to a JSON file."""
    # Convert datetime objects (if any) to string format for JSON compatibility
    data_to_save = {}
    for symbol, data in positions.items():
        # Make a copy to avoid modifying the live dictionary
        clean_data = data.copy()
        # Convert any potential non-JSON types (like Decimal if present, or datetime) to strings
        for key, value in clean_data.items():
            if isinstance(value, datetime):
                clean_data[key] = value.isoformat()
        data_to_save[symbol] = clean_data

    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        # log_trade(f"✅ State saved to {STATE_FILE}.") # Optional: Uncomment for verbose logging
    except Exception as e:
        log_trade(f"❌ Failed to save state to file: {e}")

def load_state():
    """Loads the local 'positions' dictionary from a JSON file on startup."""
    global positions
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                loaded_data = json.load(f)

            # Convert string back to required types (like floats) if necessary
            for symbol, data in loaded_data.items():
                # Ensure all numeric values that were saved as numbers are loaded as floats
                # This step is mostly defensive, as JSON handles numbers well.
                pass

            positions = loaded_data
            log_trade(f"✅ Loaded {len(positions)} open positions from persistent file.")
        except Exception as e:
            log_trade(f"❌ Failed to load state from file. Starting fresh. Error: {e}")
    else:
        log_trade(f"ℹ️ No state file found. Starting fresh.")

# ---------------- PRE-MARKET DATA ----------------
def fetch_premarket_tickers():
    """Scrapes TradingView for high pre-market gainers (>20%)."""
    url = "https://www.tradingview.com/markets/stocks-usa/market-movers-pre-market-gainers/"
    headers = {"User-Agent": "Mozilla/5.0"}
    tickers = []
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.find_all("tr")

        for row in rows[1:]:
            cols = row.find_all("td")
            if not cols: continue

            ticker_tag = cols[0].find("a")
            if not ticker_tag: continue

            ticker = ticker_tag.get_text(strip=True)
            if not (1 < len(ticker) <= 5 and ticker.isalpha() and ticker.isupper()): continue

            texts = [c.get_text(strip=True) for c in cols]
            if not any("M" in t for t in texts): continue

            change_str = next((t for t in texts if "%" in t and ("+" in t or "−" in t)), None)
            if not change_str: continue

            try:
                # Safely parse percentage change
                change_pct = float(change_str.replace("%", "").replace("+", "").replace("−", "-").replace(",", ""))
            except:
                continue

            if change_pct >= 20:
                tickers.append(ticker)

        tickers = list(dict.fromkeys(tickers))
        log_trade(f"Found {len(tickers)} pre-market gainer(s): {', '.join(tickers) if tickers else 'None'}")
    except Exception as e:
        log_trade(f"Scraper error: {e}")

    return pd.DataFrame({"Ticker": tickers})

# ---------------- POSITION RECOVERY ----------------

def find_sl_id_from_history(symbol, suppress_log=False):
    """Searches order history for the parent order and retrieves the active SL leg ID."""
    try:
        recent_orders = api.list_orders(
            status='all',
            symbols=[symbol],
            limit=50,
            direction='desc'
        )

        parent_order_id = None

        for order in recent_orders:
            if order.side == 'buy' and order.type == 'market' and order.status == 'filled':
                if hasattr(order, 'order_class') and order.order_class == 'bracket':
                    parent_order_id = order.id
                    if not suppress_log:
                        log_trade(f"🔎 Found parent bracket order ID {parent_order_id} for {symbol}.")
                    break

        if not parent_order_id:
            return 'UNKNOWN'

        full_parent_order = api.get_order(parent_order_id)

        for leg in full_parent_order.legs:
            if leg.type in ['stop', 'stop_limit'] and leg.side == 'sell':
                if leg.status in ['open', 'held']:
                    return leg.id

        return 'UNKNOWN'

    except Exception as e:
        log_trade(f"❌ Error during historical SL search for {symbol}: {e}")
        return 'UNKNOWN'


# ---------------- POSITION RECOVERY ----------------

# NOTE: The helper function find_sl_id_from_history() remains UNCHANGED.

def load_open_positions():
    """
    1. Loads existing positions from the local state file.
    2. Fetches positions from Alpaca API.
    3. Merges and recovers SL IDs if they were missing locally.
    """
    global positions

    # 1. Load data from the persistence file first
    load_state()

    try:
        alpaca_positions = api.list_positions()

        if alpaca_positions:
            log_trade(f"--- Reconciling {len(alpaca_positions)} existing position(s) from Alpaca ---")

            all_open_orders = api.list_orders(status='open')
            sl_order_map = {}

            # 2. Populate the SL map from Alpaca's open orders
            for order in all_open_orders:
                if order.type in ['stop', 'stop_limit'] and order.side == 'sell':
                    sl_order_map[order.symbol] = order.id

            # 3. Process Positions: Use Alpaca as the source of truth for quantity/price
            for p in alpaca_positions:
                symbol = p.symbol
                qty = int(p.qty)
                buy_price = float(p.avg_entry_price)
                current_price = float(p.current_price)

                # --- MERGE LOGIC ---
                if symbol in positions:
                    # Use stored data for TP hit status and trailing stop (if available)
                    pos_data = positions[symbol]

                    # Update shares and price from Alpaca (source of truth)
                    pos_data["shares_remaining"] = qty
                    pos_data["buy_price"] = buy_price

                    # Update SL ID if Alpaca has a live one (and the file might be outdated)
                    alpaca_sl_order_id = sl_order_map.get(symbol, pos_data.get("alpaca_sl_order_id", 'UNKNOWN'))

                else:
                    # Brand new position found on Alpaca, calculate state from scratch
                    pos_data = {
                        "buy_price": buy_price,
                        "shares_remaining": qty,
                        "stop_loss": buy_price * (1 - STRATEGY_CONFIG["STOP_LOSS_PCT"]),
                        "take_profit": buy_price * (1 + STRATEGY_CONFIG["TAKE_PROFIT_PCT"]),
                        "take_profit_hit": False,
                        "trailing_stop": None,
                        "alpaca_sl_order_id": sl_order_map.get(symbol, 'UNKNOWN')
                    }
                    alpaca_sl_order_id = pos_data["alpaca_sl_order_id"]
                # --- END MERGE LOGIC ---

                # Secondary Historical Search if the ID is still missing
                if alpaca_sl_order_id == 'UNKNOWN':
                    alpaca_sl_order_id = find_sl_id_from_history(symbol, True)
                    if alpaca_sl_order_id != 'UNKNOWN':
                        log_trade(f"✅ Recovered SL ID {alpaca_sl_order_id} for {symbol} via historical search.")

                pos_data["alpaca_sl_order_id"] = alpaca_sl_order_id

                # Check if TP was hit based on current price if starting fresh
                if not pos_data["take_profit_hit"] and current_price >= pos_data["take_profit"]:
                    # If TP was missed while the bot was down, assume TP state is active
                    pos_data["take_profit_hit"] = True
                    pos_data["trailing_stop"] = current_price * (1 - STRATEGY_CONFIG["TRAILING_STOP_PCT"])

                positions[symbol] = pos_data

                log_trade(f"Reconciled: {symbol} - {qty} shares @ ${buy_price:.2f} (TP Hit: {pos_data['take_profit_hit']})")

        # Cleanup: Remove positions from local state that are no longer on Alpaca
        # This handles cases where Alpaca SL was hit while the bot was offline.
        for symbol in list(positions.keys()):
            if symbol not in [p.symbol for p in alpaca_positions]:
                log_trade(f"⚠️ CLEANUP: Removing stale position {symbol} (closed on Alpaca while offline).")
                del positions[symbol]

        save_state() # Save the merged/reconciled state immediately

    except Exception as e:
        log_trade(f"❌ Failed to load existing positions from Alpaca: {e}")

# ---------------- HELPER P/L CALCULATION ----------------
def calculate_trade_pl(symbol, position_data, final_sell_price, sell_type):
    """Calculates the P/L and percentage gain for a fully closed trade."""

    # We assume the position initially had full shares equal to the starting MoneyInvested amount
    initial_shares = int(STRATEGY_CONFIG["MoneyInvested"] / position_data["buy_price"])
    buy_price = position_data["buy_price"]

    initial_cost = initial_shares * buy_price

    if position_data["take_profit_hit"]:
        # Case 1: TP was hit (50% sold at TP, 50% sold at final_sell_price)
        shares_at_tp = initial_shares // 2
        shares_at_final_sell = initial_shares - shares_at_tp

        tp_price = position_data["take_profit"]

        # Total Revenue
        revenue = (shares_at_tp * tp_price) + (shares_at_final_sell * final_sell_price)

    else:
        # Case 2: Catastrophic SL hit before TP (100% sold at final_sell_price)
        revenue = initial_shares * final_sell_price

    pl_dollar = revenue - initial_cost
    pl_percent = (pl_dollar / initial_cost) * 100 if initial_cost else 0

    return pl_dollar, pl_percent
# --------------------------------------------------------

# ---------------- HELPER SHARE CALCULATION (NEW) ----------------
def calculate_shares(symbol, current_price):
    """Calculates the number of shares to buy based on MoneyInvested and current price."""
    # Ensure price is valid before division
    if current_price <= 0:
        log_trade(f"❌ Cannot calculate shares for {symbol}: price is ${current_price:.2f}.")
        return 0

    cash_to_invest = STRATEGY_CONFIG["MoneyInvested"]
    # Shares must be an integer (whole number)
    shares = int(cash_to_invest / current_price)

    if shares == 0:
        log_trade(f"⚠️ Not enough money to buy one share of {symbol} at ${current_price:.2f}.")

    return shares
# --------------------------------------------------------


# ---------------- DATA FETCHING & ANALYSIS ----------------

def get_symbol_metrics(symbol, supports, short_slope):
    """Returns a dictionary of metrics for sorting and logging."""
    return {
        "symbol": symbol,
        "supports_count": len(supports),
        "short_slope": short_slope,
        "raw_supports": supports,
    }

def fetch_5min_bars(symbol, max_retries=3):
    """Fetches the required historical 5-minute bars from Alpaca."""
    for attempt in range(max_retries):
        try:
            bars = api.get_bars(symbol, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
                                limit=STRATEGY_CONFIG["LOOKBACK_BARS"], adjustment='all', feed=DATA_FEED).df
            if bars.empty:
                return pd.DataFrame()
            return bars.tz_convert('America/New_York').tz_localize(None)
        except Exception as e:
            if attempt == max_retries - 1:
                log_trade(f"API 5-Min Bars fetch failed for {symbol}: {e}")
            else:
                time.sleep(2 ** attempt)
        return pd.DataFrame()

def analyze_symbol(symbol):
    """Analyzes the symbol for rising supports and short-term slope."""
    global symbol_state
    now = datetime.now()
    if symbol not in symbol_state:
        symbol_state[symbol] = {
            "supports": [], "short_term_slope": 0.0,
            "last_log": now - timedelta(minutes=5),
            "last_supports_count": 0, "trend_break_message": None
        }

    MIN_REQUIRED_BARS = STRATEGY_CONFIG["SHORT_TERM_SLOPE_LOOKBACK"]
    bars = fetch_5min_bars(symbol)

    # Analysis start filter (11:15 AM ET)
    analysis_start_time = now.replace(hour=10, minute=15, second=0, microsecond=0)
    bars = bars[bars.index >= analysis_start_time]

    if bars.empty or len(bars) < MIN_REQUIRED_BARS:
        return get_symbol_metrics(symbol, [], 0.0)

    prices = bars['close']
    times = bars.index

    # 1. HISTORICAL LOCAL MINIMA DETECTION
    # (Used ONLY for identifying potential support points, not for breaking the trend)
    minima_prices, minima_times = [], []
    num_historical_bars = len(prices)

    if num_historical_bars >= 5:
        for i in range(2, num_historical_bars - 2):
            left = prices.iloc[i-2:i]
            right = prices.iloc[i+1:i+3]

            # Simplified minima check: Price must be the lowest of 5 consecutive bars
            if prices.iloc[i] < left.min() and prices.iloc[i] < right.min():
                minima_prices.append(prices.iloc[i])
                minima_times.append(times[i])

    # 2. BUILD RISING SUPPORTS ARRAY
    # (This section determines the Count: 4/3)
    supports = []
    last_price = -np.inf

    # Build the sequence of strictly rising supports from the detected minima
    for t, price in zip(minima_times, minima_prices):
        if price > last_price:
            supports.append((t, price))
            last_price = price
        else:
            # Trend Broken via a lower minima: Reset the rising sequence to the current low point
            if len(supports) >= 2:
                symbol_state[symbol]['trend_break_message'] = f"🔴 {symbol} | Trend BROKEN (Minima sequence reset) at {t.strftime('%H:%M')}@{price:.4f} (Below prior support of {last_price:.4f})."

            supports = [(t, price)] # Start a new sequence with the current low point
            last_price = price

    # --- NEW FIX: CHECK ALL BARS FOR BREAK BELOW HIGHEST SUPPORT (The Price Action Rule) ---
    # This implements the user's rule: Break if price closes below the last support
    if len(supports) > 0:
        highest_support_price = supports[-1][1]
        last_support_time = supports[-1][0]

        # Filter all bars that occurred AFTER the last support confirmation time
        bars_since_last_support = bars[bars.index > last_support_time]
        
        # Only proceed if we have bars since the last support was found
        if not bars_since_last_support.empty:
            
            # Check if the lowest close price since that support has broken below it
            lowest_close_since = bars_since_last_support['close'].min()
            
            if lowest_close_since < highest_support_price:
                # Find the time this break occurred for logging (use the first time)
                break_time_index = bars_since_last_support['close'].idxmin()
                
                # Log the break and force the supports array to be reset to empty (count = 0).
                symbol_state[symbol]['trend_break_message'] = f"🔴 {symbol} | Trend BROKEN (Price close below support) at {break_time_index.strftime('%H:%M')}@{lowest_close_since:.4f} (Below highest support of {highest_support_price:.4f})."
                
                supports = [] # Reset supports to empty because the trend is broken (Count = 0)

    # 3. SHORT-TERM TREND SLOPE CALCULATION
    short_slope = 0.0
    if len(prices) >= STRATEGY_CONFIG["SHORT_TERM_SLOPE_LOOKBACK"]:
        short_term_prices = prices.iloc[-STRATEGY_CONFIG["SHORT_TERM_SLOPE_LOOKBACK"]:]
        x_short = np.arange(STRATEGY_CONFIG["SHORT_TERM_SLOPE_LOOKBACK"])
        y_short = short_term_prices.values
        short_slope, _ = np.polyfit(x_short, y_short, 1)
        symbol_state[symbol]["short_term_slope"] = short_slope


    # FINAL CHECK AND LOGGING HOOK
    final_supports_count = len(supports)
    if 'trend_break_message' in symbol_state[symbol] and final_supports_count < symbol_state[symbol]["last_supports_count"]:
        symbol_state[symbol]['trend_break_message'] # Use the message set above
    elif 'trend_break_message' in symbol_state[symbol]:
        # Clear message if the count wasn't actually reduced (e.g., if supports[] was already empty)
        symbol_state[symbol]['trend_break_message'] = None


    return get_symbol_metrics(symbol, supports, short_slope)


def log_and_execute(metrics, force_log=False):
    """Handles logging and trade execution based on sorted metrics."""
    symbol = metrics["symbol"]
    supports = metrics["raw_supports"]
    short_slope = metrics["short_slope"]
    new_supports_count = metrics["supports_count"]

    global symbol_state
    global is_first_run
    now = datetime.now()


    # Trend Break Logging
    if 'trend_break_message' in symbol_state[symbol] and symbol_state[symbol]['trend_break_message']:
        log_trade(symbol_state[symbol]['trend_break_message'])
        symbol_state[symbol]['trend_break_message'] = None
        force_log = True

    # LOGGING
    should_log_analysis = is_first_run or \
                         new_supports_count != symbol_state[symbol]["last_supports_count"] or \
                         force_log

    if should_log_analysis:
        print()
        support_emoji = "🟢" if new_supports_count == STRATEGY_CONFIG["MIN_REQUIRED_SUPPORT_HITS"] else ("🟡" if new_supports_count > 0 else "⚫")
        slope_emoji = "🟢" if short_slope > 0 else ("🔴" if short_slope < 0 else "⚪")

        # Format Supports String
        formatted_supports_list = []
        for s in supports:
            close_time = s[0] + timedelta(minutes=5)
            formatted_supports_list.append(f"{close_time.strftime('%H:%M')} @ {s[1]:.4f}")
        formatted_support_string = " -> ".join(formatted_supports_list) if supports else 'N/A'

        log_trade(
            f"{support_emoji}{slope_emoji} {symbol} | Supports Count: {new_supports_count}/{STRATEGY_CONFIG['MIN_REQUIRED_SUPPORT_HITS']} | ST Slope: {short_slope:+.5f} | Supports: {formatted_support_string}"
        )

        symbol_state[symbol]["last_log"] = now
        symbol_state[symbol]["last_supports_count"] = new_supports_count


    # BUY SIGNAL CHECK and EXECUTION
    rising_supports_met = new_supports_count >= STRATEGY_CONFIG["MIN_REQUIRED_SUPPORT_HITS"]
    short_term_trend_positive = short_slope > 0

    if rising_supports_met and short_term_trend_positive:
        # Only buy if not currently holding the symbol
        if symbol not in positions or positions[symbol]["shares_remaining"] == 0:
            
            # --- FIX: LOG THE INTENT TO BUY EVERY TIME THE SIGNAL IS MET ---
            # This ensures the bot confirms it is attempting to execute the trade
            if symbol_state[symbol]["last_supports_count"] >= STRATEGY_CONFIG["MIN_REQUIRED_SUPPORT_HITS"]:
                 log_trade(f"💰 BUY SIGNAL! {symbol} | 3+ supports confirmed | Short-term trend UP ({short_slope:+.5f})")
            # -----------------------------------------------------------------

            # --- EXECUTION LOGIC ---
            try:
                # CRITICAL: This line will fail if Alpaca does not have a price.
                price = api.get_latest_trade(symbol).price
                shares_to_buy = calculate_shares(symbol, price)

                if shares_to_buy > 0:
                    place_initial_buy(symbol, shares_to_buy)
                else:
                    log_trade(f"⚠️ Not buying {symbol}: Shares to buy is 0.")

            except Exception as e:
                # This log ensures you see the error when the buy fails.
                log_trade(f"❌ Buy setup failed for {symbol}: Could not get latest price or snapshot data. Error: {e}")
            # --- END EXECUTION LOGIC ---


# ---------------- TRADING EXECUTION ----------------
# ... (rest of the file remains the same)

# ---------------- TRADING EXECUTION ----------------
def place_initial_buy(symbol, shares):
    """Places the initial buy order as a bracket order."""
    sl_order_id = 'UNKNOWN'
    try:
        price = api.get_latest_trade(symbol).price

        # FIX: Explicitly round stop price to 2 decimal places (pennies)
        stop = round(price * (1 - STRATEGY_CONFIG["STOP_LOSS_PCT"]), 2)
        tp = price * (1 + STRATEGY_CONFIG["TAKE_PROFIT_PCT"])

        # Placeholder TP limit needs to be rounded too
        placeholder_tp_limit = round(price * 6.0, 2)

        # Submit the MARKET order with the bracket legs
        order = api.submit_order(
            symbol=symbol,
            qty=shares,
            side='buy',
            type='market',
            time_in_force='day',
            order_class='bracket',
            # Use the rounded stop price
            stop_loss={'stop_price': stop},
            take_profit={'limit_price': placeholder_tp_limit}
        )

        # Retrieve the SL Order ID for local management (cancellation later)
        time.sleep(1)
        full_order = api.get_order(order.id)

        for leg in full_order.legs:
            if leg.type == 'stop':
                sl_order_id = leg.id
                break

        # Store all necessary data locally
        positions[symbol] = {
            "buy_price": price,
            "shares_remaining": shares,
            "stop_loss": stop,
            "take_profit": tp,
            "take_profit_hit": False,
            "trailing_stop": None,
            "alpaca_sl_order_id": sl_order_id
        }
        log_trade(f"🟢 BOUGHT {shares} {symbol} @ ${price:.2f} | TP ${tp:.2f} (Local) | SL ${stop:.2f} (Alpaca)")

        # --- ADDED THIS LINE ---
        save_state()

        # Immediate SL Status Check
        if sl_order_id != 'UNKNOWN':
            try:
                sl_order_status = api.get_order(sl_order_id).status
                log_trade(f"✅ Verified SL Order ID {sl_order_id} status on Alpaca: {sl_order_status}")
            except Exception as e:
                log_trade(f"⚠️ Failed to verify SL order {sl_order_id}: {e}")

    except Exception as e:
        log_trade(f"❌ Buy failed {symbol}: {e}")

def manage_positions():
    """Manages open positions using local logic for TP and TS."""
    global closed_trades_pl # Need this to store P/L

    for symbol, pos in list(positions.items()):
        if pos["shares_remaining"] <= 0:
            continue

        # Check Alpaca SL Status (Ensure SL wasn't hit externally)
        alpaca_sl_status = None
        if pos["alpaca_sl_order_id"] != 'UNKNOWN':
            try:
                alpaca_sl_status = api.get_order(pos["alpaca_sl_order_id"]).status
            except:
                alpaca_sl_status = 'filled' # Assume filled/cancelled if API call fails

        # If the catastrophic SL was triggered by Alpaca, update shares and clean up
        if pos["alpaca_sl_order_id"] != 'UNKNOWN' and alpaca_sl_status in ['filled', 'canceled']:
            try:
                current_position = api.get_position(symbol)
                pos["shares_remaining"] = int(current_position.qty)
            except:
                pos["shares_remaining"] = 0

            # --- P/L CALCULATION AND STORAGE FOR SL ---
            sl_price = pos["stop_loss"]
            pl_dollar, pl_percent = calculate_trade_pl(symbol, pos, sl_price, "SL")

            log_trade(f"🛑 CATASTROPHIC STOP HIT {symbol} - Position closed by Alpaca SL. | P/L: ${pl_dollar:+.2f} ({pl_percent:+.2f}%)")

            closed_trades_pl[symbol] = {"pl_dollar": pl_dollar, "pl_percent": pl_percent}
            del positions[symbol] # Remove from positions immediately after closing/logging
            # ----------------------------------------------
            save_state() # Save state immediately after deletion
            continue


        # Proceed with LOCAL TP/TS logic
        try:
            price = api.get_latest_trade(symbol).price

            # --- Take Profit Logic (Local) ---
            if not pos["take_profit_hit"] and price >= pos["take_profit"]:

                # CRITICAL: Cancel the Alpaca SL order before selling the first part
                if pos["alpaca_sl_order_id"] != 'UNKNOWN':
                    try:
                        api.cancel_order(pos["alpaca_sl_order_id"])
                        log_trade(f"🟡 CANCELLING Alpaca SL for {symbol} to engage local TP/TS.")
                    except:
                        log_trade(f"❌ Failed to cancel Alpaca SL {pos['alpaca_sl_order_id']} for {symbol}. Proceeding with local sell.")


                # Sell 50% at TP
                half = pos["shares_remaining"] // 2
                if half > 0:
                    api.submit_order(symbol=symbol, qty=half, side='sell', type='market', time_in_force='day')
                    pos["shares_remaining"] -= half
                    pos["take_profit_hit"] = True
                    # FIX: Correctly use STRATEGY_CONFIG for TS
                    pos["trailing_stop"] = price * (1 - STRATEGY_CONFIG["TRAILING_STOP_PCT"])
                    log_trade(f"⭐ TAKE PROFIT 50% {symbol} @ ${price:.2f} | TS activated @ ${pos['trailing_stop']:.2f}")

            # --- Trailing Stop Logic (Local) ---
            elif pos["take_profit_hit"] and pos["shares_remaining"] > 0 and pos["trailing_stop"] is not None:
                new_trailing_stop = price * (1 - STRATEGY_CONFIG["TRAILING_STOP_PCT"])
                pos["trailing_stop"] = max(pos["trailing_stop"], new_trailing_stop)

                if price <= pos["trailing_stop"]:
                    # Sell the remainder at TS
                    api.submit_order(symbol=symbol, qty=pos["shares_remaining"], side='sell', type='market', time_in_force='day')

                    # --- P/L CALCULATION AND STORAGE ---
                    final_sell_price = price
                    pl_dollar, pl_percent = calculate_trade_pl(symbol, pos, final_sell_price, "TS")

                    log_trade(f"🛑 TRAILING STOP HIT {symbol} @ ${final_sell_price:.2f} → sold remaining | P/L: ${pl_dollar:+.2f} ({pl_percent:+.2f}%)")

                    closed_trades_pl[symbol] = {"pl_dollar": pl_dollar, "pl_percent": pl_percent}
                    pos["shares_remaining"] = 0
                    del positions[symbol] # Remove from positions immediately after closing/logging
                    # ----------------------------------------

        except Exception as e:
            log_trade(f"❌ Manage error {symbol}: {e}")
        # --- ADD save_state() HERE ---
        # This ensures that all changes (deletions, updates to trailing_stop/TP hit status)
        # made within the loop are written to the JSON file before the function exits.
    save_state()

def sell_all_positions():
    """Sells all remaining open positions at the end of the trading day."""
    log_trade("--- END OF DAY CLOSE ---")

    # Iterate over a copy of the keys
    for symbol in list(positions.keys()):
        shares_remaining = positions.get(symbol, {}).get("shares_remaining", 0)
        alpaca_sl_order_id = positions.get(symbol, {}).get("alpaca_sl_order_id", None)

        # 1. Attempt to cancel the Alpaca SL leg (and the placeholder TP leg)
        if alpaca_sl_order_id != 'UNKNOWN':
            try:
                api.cancel_order(alpaca_sl_order_id)
            except:
                pass

        # 2. Attempt to sell the remaining shares
        if shares_remaining > 0:
            try:
                # Get current price for P/L calculation proxy
                price = api.get_latest_trade(symbol).price
                api.submit_order(symbol=symbol, qty=shares_remaining, side='sell', type='market', time_in_force='day')

                # P/L Calculation for EOD Sell (using current price as proxy for filled price)
                pl_dollar, pl_percent = calculate_trade_pl(symbol, positions[symbol], price, "EOD")
                closed_trades_pl[symbol] = {"pl_dollar": pl_dollar, "pl_percent": pl_percent}

                log_trade(f"🟥 EOD SELL → {symbol} ({shares_remaining} shares) | P/L: ${pl_dollar:+.2f} ({pl_percent:+.2f}%)")
            except Exception as e:
                log_trade(f"❌ EOD SELL FAILED → {symbol}. Error: {e}")

    # FIX: Crucial: Clear the local state (including open positions) for a clean restart.
    positions.clear()
    symbol_state.clear()
    log_trade("✅ Local state (positions/analysis) cleared.")

    # --- ADDED THIS LINE ---
    save_state()
    # ---------------------

# ---------------- HELPER MARKET CLOCK FUNCTION ----------------
def get_next_market_open(api):
    """Uses Alpaca's clock to find the next market open datetime, set to strategy start time (11:15 ET)."""
    try:
        clock = api.get_clock()
        # Define the 11:15 AM strategy start time
        start_time_obj = datetime.strptime("11:15", "%H:%M").time()

        # If the market is open, we need to find the next *trading* day
        if clock.is_open:
            # Start search from tomorrow
            search_date = clock.timestamp.date() + timedelta(days=1)
        else:
            # Market is closed, use Alpaca's next scheduled open date
            search_date = clock.next_open.date()

        # Fetch calendar until a market day is found
        while True:
            # Only need one day, starting from search_date
            calendar_entries = api.get_calendar(start=search_date, end=search_date)

            if calendar_entries:
                calendar_entry = calendar_entries[0]
                # Return the start date of the trading day combined with your strategy's start time
                return datetime.combine(calendar_entry.date, start_time_obj)

            # If nothing found (weekend/holiday), move to the next day
            search_date += timedelta(days=1)

    except Exception as e:
        log_trade(f"❌ Failed to get next market open from Alpaca. Defaulting to 12 hours sleep. Error: {e}")
        # Default fallback sleep
        return datetime.now() + timedelta(hours=12)
# ---------------- MAIN EXECUTION LOOP ----------------
log_trade("Starting Alpaca 3-Support Rising Trend Bot (PAPER)")
UseThis = fetch_premarket_tickers()
load_open_positions()
last_summary_time = datetime.now()

send_telegram_message("🤖 Bot Initialized: Telegram connection verified! Running in PAPER mode.")

while True:
    try:
        now = datetime.now()

        # Define market hours (Eastern Time)
        start_time = datetime.strptime("11:15", "%H:%M").time()
        end_time = datetime.strptime("15:55", "%H:%M").time()

        if now.time() < start_time:
            time.sleep(30)
            continue

        if now.time() >= end_time:

            sell_all_positions()

            # --- EOD FINAL P/L SUMMARY ---
            if closed_trades_pl:
                total_pl = sum(d["pl_dollar"] for d in closed_trades_pl.values())
                avg_pl_percent = sum(d["pl_percent"] for d in closed_trades_pl.values()) / len(closed_trades_pl)

                log_trade(f"--- 🏁 END OF DAY PERFORMANCE REPORT 🏁 ---")
                log_trade(f"Total Closed Trades: <b>{len(closed_trades_pl)}</b>") # Use <b> tags
                log_trade(f"Total P/L: <b>${total_pl:+.2f}</b>")                # Use <b> tags
                log_trade(f"Average % Gain/Loss per Trade: <b>{avg_pl_percent:+.2f}%</b>") # Use <b> tags
                log_trade(f"----------------------------------------------")

                # Send summary to Telegram using HTML
                summary_message = f"🏁 <i>EOD REPORT:</i>\nTrades Closed: {len(closed_trades_pl)}\nTotal P/L: <b>${total_pl:+.2f}</b>\nAvg %: <b>{avg_pl_percent:+.2f}%</b>"
                send_telegram_message(summary_message)
            # --- END OF DAY P/L SUMMARY ---

            # Clear the total P/L tracker so it resets for the next day
            closed_trades_pl.clear()

            # --- CRITICAL FIX: CALCULATE NEXT MARKET OPEN ---
            next_market_start = get_next_market_open(api)
            sleep_duration = (next_market_start - now).total_seconds()

            log_trade(f"Market closed. Sleeping {sleep_duration/3600:.1f}h until {next_market_start.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(sleep_duration)

            # Re-initialize for the new day
            UseThis = fetch_premarket_tickers()
            load_open_positions()
            last_summary_time = datetime.now()
            is_first_run = True # Reset for the next day's first run
            continue # Go back to the top of the loop

        # --- 1. RUN ANALYSIS FOR ALL STOCKS ---
        all_metrics = []
        for _, row in UseThis.iterrows():
            sym = row["Ticker"]
            metrics = analyze_symbol(sym)
            all_metrics.append(metrics)

        # Sort by support count then slope (best signals first)
        all_metrics.sort(key=lambda m: (m["supports_count"], m["short_slope"]), reverse=True)

        # --- 2. LOG & EXECUTE TRADES ---
        for metrics in all_metrics:
            log_and_execute(metrics, force_log=False)

        # Reset the first run flag after logging the initial analysis
        if is_first_run:
            is_first_run = False

        # --- 3. MANAGE POSITIONS (TP/TS) ---
        manage_positions()

        # --- 4. 5-MINUTE STATUS SUMMARY ---
        if (now - last_summary_time).total_seconds() >= 300:
            log_trade("\n--- START 5-MINUTE STATUS SUMMARY ---")

            # Re-collect metrics for held stocks (in case they weren't in the main analysis)
            held_stocks_metrics = []
            for sym in positions.keys():
                # Avoid re-analyzing stocks already in the all_metrics list to save API calls
                if sym not in [m['symbol'] for m in all_metrics]:
                    held_stocks_metrics.append(analyze_symbol(sym))

            summary_metrics = all_metrics + held_stocks_metrics
            summary_metrics.sort(key=lambda m: (m["supports_count"], m["short_slope"]), reverse=True)

            for metrics in summary_metrics:
                log_and_execute(metrics, force_log=True) # Force log ensures all are printed

            log_trade("--- END 5-MINUTE STATUS SUMMARY ---\n")

            # --- 1. REALIZED P/L (Your Closed Trades) ---
            if closed_trades_pl:
                total_pl = sum(d["pl_dollar"] for d in closed_trades_pl.values())

                # Calculate average only if there are closed trades
                if len(closed_trades_pl) > 0:
                    avg_pl_percent = sum(d["pl_percent"] for d in closed_trades_pl.values()) / len(closed_trades_pl)
                else:
                    avg_pl_percent = 0.0

                log_trade(f"--- 📊 RUNNING PERFORMANCE REPORT (Realized) 📊 ---")
                log_trade(f"Closed Trades Today: **{len(closed_trades_pl)}**")
                log_trade(f"Total Closed P/L: **${total_pl:+.2f}**")
                log_trade(f"Average % Gain/Loss: **{avg_pl_percent:+.2f}%**")
                log_trade(f"---------------------------------------------------")
            # --- END REALIZED P/L REPORT ---


            # --- 2. TOTAL ACCOUNT P/L (Alpaca Official) ---
            try:
                # Fetches the overall account balance and P/L from the previous close
                account = api.get_account()
                current_equity = float(account.equity)
                # Use last_equity as the baseline for the previous day's close
                previous_close_equity = float(account.last_equity)

                daily_pl_dollar = current_equity - previous_close_equity
                daily_pl_percent = (daily_pl_dollar / previous_close_equity) * 100 if previous_close_equity else 0

                log_trade(f"--- 💸 TOTAL ACCOUNT P/L (Alpaca Official) 💸 ---")
                log_trade(f"Today's Portfolio Change: **${daily_pl_dollar:+.2f} ({daily_pl_percent:+.2f}%)**")
                log_trade(f"-------------------------------------------------")
            except Exception as e:
                log_trade(f"❌ Failed to fetch official Account P/L: {e}")
            # --- END TOTAL ACCOUNT P/L REPORT ---

            last_summary_time = now # Reset the timer

        time.sleep(POLL_INTERVAL_SECONDS)

    except Exception as e:
        log_trade(f"Main loop error: {e}")
        time.sleep(10)