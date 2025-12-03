


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
      
       # --- ADD THIS LINE ---
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


   # --- ADD THIS LINE ---
   save_state()
   # ---------------------


# ---------------- HELPER MARKET CLOCK FUNCTION ----------------
def get_next_market_open(api):
   # YOUR NEW CODE GOES HERE
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



