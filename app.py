from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time
import threading
import numpy as np
import math
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

stock_symbols = [
        'ABB.NS',
        'ABBOTINDIA.NS',
        'ACC.NS',
        'ADANIENT.NS',
        'ADANIPORTS.NS',
        'AMBUJACEM.NS',
        'APLAPOLLO.NS',
        'APOLLO.NS',
        'APOLLOTYRE.NS',
        'APOLLOHOSP.NS',
        'ASIANPAINT.NS',
        'ASHOKLEY.NS',
        'ASTRAL.NS',
        'AWL.NS',
        'AXISBANK.NS',
        'BAJAJ-AUTO.NS',
        'BAJAJFINSV.NS',
        'BAJAJHIND.NS',
        'BAJFINANCE.NS',
        'BALRAMCHIN.NS',
        'BANKBARODA.NS',
        'BATAINDIA.NS',
        'BEL.NS',
        'BERGEPAINT.NS',
        'BHARATFORG.NS',
        'BHARTIARTL.NS',
        'BHEL.NS',
        'BPCL.NS',
        'BRITANNIA.NS',
        'BSOFT.NS',
        'CANBK.NS',
        'CENTRALBK.NS',
        'CHAMBLFERT.NS',
        'CIPLA.NS',
        'COALINDIA.NS',
        'COFORGE.NS',
        'COLPAL.NS',
        'CONCOR.NS',
        'CROMPTON.NS',
        'CUMMINSIND.NS',
        'DABUR.NS',
        'DALBHARAT.NS',
        'DEEPAKFERT.NS',
        'DEVYANI.NS',
        'DIVISLAB.NS',
        'DRREDDY.NS',
        'EICHERMOT.NS',
        'FSL.NS',
        'GAIL.NS',
        'GLENMARK.NS',
        'GODFRYPHLP.NS',
        'GODREJCP.NS',
        'GRANULES.NS',
        'GRASIM.NS',
        'GRSE.NS',
        'HAL.NS',
        'HAVELLS.NS',
        'HCLTECH.NS',
        'HDFCBANK.NS',
        'HEROMOTOCO.NS',
        'HGS.NS',
        'HINDALCO.NS',
        'HINDCOPPER.NS',
        'HINDPETRO.NS',
        'HINDUNILVR.NS',
        'HINDZINC.NS',
        'ICICIBANK.NS',
        'IDEA.NS',
        'IDFCFIRSTB.NS',
        'IEX.NS',
        'IGL.NS',
        'INDHOTEL.NS',
        'INDIACEM.NS',
        'INDIAMART.NS',
        'INDUSINDBK.NS',
        'INDUSTOWER.NS',
        'INFY.NS',
        'INTELLECT.NS',
        'IOC.NS',
        'IRCTC.NS',
        'IRFC.NS',
        'ITC.NS',
        'JINDALSTEL.NS',
        'JSWSTEEL.NS',
        'KOTAKBANK.NS',
        'KPITTECH.NS',
        'LICI.NS',
        'LT.NS',
        'LTIM.NS',
        'LTTS.NS',
        'LUPIN.NS',
        'M&M.NS',
        'MARICO.NS',
        'MARUTI.NS',
        'MCX.NS',
        'MPHASIS.NS',
        'MRF.NS',
        'MRPL.NS',
        'NATIONALUM.NS',
        'NESTLEIND.NS',
        'NMDC.NS',
        'NTPC.NS',
        'NYKAA.NS',
        'ONGC.NS',
        'PETRONET.NS',
        'PFC.NS',
        'PIDILITIND.NS',
        'POLYCAB.NS',
        'POLYPLEX.NS',
        'POWERGRID.NS',
        'RAMCOCEM.NS',
        'RAYMOND.NS',
        'RELIANCE.NS',
        'RENUKA.NS',
        'RVNL.NS',
        'SAIL.NS',
        'SBIN.NS',
        'SIEMENS.NS',
        'SIRCA.NS',
        'SRF.NS',
        'STARCEMENT.NS',
        'SUNPHARMA.NS',
        'SUZLON.NS',
        'TATACHEM.NS',
        'TATACONSUM.NS',
        'TATAELXSI.NS',
        'TATAMOTORS.NS',
        'TATAPOWER.NS',
        'TATASTEEL.NS',
        'TCS.NS',
        'TECHM.NS',
        'TITAN.NS',
        'TORNTPHARM.NS',
        'TRENT.NS',
        'TTKPRESTIG.NS',
        'TVSMOTOR.NS',
        'UBL.NS',
        'ULTRACEMCO.NS',
        'UPL.NS',
        'VEDL.NS',
        'VOLTAS.NS',
        'WIPRO.NS',
        'YESBANK.NS',
        'ZENTEC.NS',
        'ZOMATO.NS',
        'ZYDUSWELL.NS',
    ]



# Step 1: Fetch Stock Data
def fetch_stock_data(stock_symbols):
    try:
        # Fetch 1-hour data over 3 months for existing indicators
        stock_data_1h = yf.download(stock_symbols, period="3mo", interval="1h", group_by='ticker', threads=True)
        # Fetch daily data over 2 years for additional metrics
        stock_data_daily = yf.download(stock_symbols, period="2y", interval="1d", group_by='ticker', threads=True)
        # Fetch daily data over 2 years for additional metrics
        # Fetch minute-level data for the current day
        stock_data_minute = yf.download(stock_symbols, period="1d", interval="1m", group_by='ticker', threads=True)
        # # Fetch minute-level data for the current day
        # stock_data_minute = {}
        # for symbol in stock_symbols:
        #     data_minute = yf.download(symbol, period="1d", interval="1m", group_by='ticker', threads=True)
        #     if not data_minute.empty:
        #         stock_data_minute[symbol] = data_minute
        #     else:
        #         print(f"No minute-level data for {symbol}")
        
        if stock_data_1h.empty or stock_data_daily.empty or stock_data_minute.empty:
            print(f"Error: No data returned for symbols {stock_symbols}")
            return None, None, None
        return stock_data_1h, stock_data_daily, stock_data_minute
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None, None, None

# Step 2: Calculate Bollinger %b and RSI for a single stock
def calculate_bollinger_and_rsi(data_1h, data_daily, data_minute):
    try:
        # Ensure indices are DateTimeIndex and sorted
        data_1h = data_1h.sort_index()
        data_daily = data_daily.sort_index()
        data_minute = data_minute.sort_index()
    
        # RSI on 1-hour data
        data_1h['OHLC4'] = data_1h[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        data_1h['RSI'] = ta.rsi(data_1h['OHLC4'], length=14)
        rsi_current = data_1h['RSI'].iloc[-1]
    
        # Bollinger Bands on 2-hour data
        data_2h = data_1h.resample('2h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'OHLC4': 'mean'
        }).dropna()
    
        if len(data_2h) < 20:
            print("Not enough data after resampling to calculate Bollinger Bands")
            return None, None, None, None
    
        bbands = ta.bbands(data_2h['OHLC4'], length=20, std=2, mamode='ema')
        data_2h = data_2h.join(bbands)
        data_2h['Bollinger_%b'] = ((data_2h['OHLC4'] - data_2h['BBL_20_2.0']) /
                                    (data_2h['BBU_20_2.0'] - data_2h['BBL_20_2.0'])) * 100
        bollinger_b = data_2h['Bollinger_%b'].iloc[-1]
    
        # Current value
        value_current = data_1h['Close'].iloc[-1]
    
        additional_metrics = {}
    
        # Day's high and low from minute-level data
        if not data_minute.empty:
            day_high = data_minute['High'].max()
            day_low = data_minute['Low'].min()
            additional_metrics['Day High'] = day_high
            additional_metrics['Day Low'] = day_low
        else:
            print("Minute-level data is empty for today's high and low.")
    
        # Value changes and percentage changes
        periods = {
            '1 day': 1,
            '5 days': 5,
            '20 days': 20,
            '1 year': 252  # Approximate trading days in a year
        }
    
        for period_name, period_length in periods.items():
            if len(data_daily) >= period_length + 1:
                past_close = data_daily['Close'].iloc[-(period_length + 1)]
                value_change = value_current - past_close
                percent_change = (value_change / past_close) * 100
                additional_metrics[f'Value change ({period_name})'] = value_change
                additional_metrics[f'% change ({period_name})'] = percent_change
                additional_metrics[f'High ({period_name} ago)'] = data_daily['High'].iloc[-(period_length + 1)]
                additional_metrics[f'Low ({period_name} ago)'] = data_daily['Low'].iloc[-(period_length + 1)]
            else:
                print(f"Not enough data to calculate {period_name} change.")
    
        return bollinger_b, rsi_current, value_current, additional_metrics
    
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None, None, None, None

# Step 3: Process all stocks and compute indicators
def process_stock_data(stock_data_1h, stock_data_daily, stock_data_minute, stock_symbols):
    results = {}
    for symbol in stock_symbols:
        try:
            data_1h = stock_data_1h[symbol]
            data_daily = stock_data_daily[symbol]
            data_minute = stock_data_minute.get(symbol, pd.DataFrame())
            bollinger_b, rsi, value, additional_metrics = calculate_bollinger_and_rsi(data_1h, data_daily, data_minute)
            if bollinger_b is not None and rsi is not None and value is not None:
                results[symbol] = {
                    'Bollinger_%b': bollinger_b,
                    'RSI': rsi,
                    'Value': value,
                    **additional_metrics  # Additional metrics including day's high and low
                }
            else:
                print(f"Insufficient data for {symbol}, skipping...")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    return results

# Step 4: Emit alerts based on conditions
def check_and_emit_alerts(results):
    import math
    from datetime import datetime

    def convert_value(val):
        if isinstance(val, (np.generic, np.number, float, int)):
            val = float(val)
            if math.isnan(val) or math.isinf(val):
                return None
            else:
                return val
        elif isinstance(val, (pd.Timestamp, datetime)):
            return val.isoformat()
        else:
            return val

    for symbol, indicators in results.items():
        bollinger_b = indicators['Bollinger_%b']
        rsi = indicators['RSI']
        value = indicators['Value']
        # Collect the additional metrics
        additional_metrics = {key: indicators[key] for key in indicators if key not in ['Bollinger_%b', 'RSI', 'Value']}

        # Convert the values
        bollinger_b = convert_value(bollinger_b)
        rsi = convert_value(rsi)
        value = convert_value(value)
        additional_metrics = {k: convert_value(v) for k, v in additional_metrics.items()}

        if bollinger_b is not None and rsi is not None:
            # Create the alert data
            alert_data = {
                'symbol': symbol,
                'type': 'green',  # default type
                'Bollinger_%b': bollinger_b,
                'RSI': rsi,
                'Value': value,
                **additional_metrics
            }

            # Remove keys with None values
            alert_data = {k: v for k, v in alert_data.items() if v is not None}

            # Determine the alert type
            if bollinger_b < -10:
                alert_data['type'] = 'green'
                socketio.emit('new_alert', alert_data)
            elif bollinger_b < 0:
                alert_data['type'] = 'blue'
                socketio.emit('new_alert', alert_data)
            elif bollinger_b > 120:
                alert_data['type'] = 'red'
                socketio.emit('new_alert', alert_data)
            elif bollinger_b > 100:
                alert_data['type'] = 'orange'
                socketio.emit('new_alert', alert_data)

            # RSI alerts
            if rsi < 5:
                alert_data['type'] = 'green'
                socketio.emit('new_alert', alert_data)
            elif rsi < 10:
                alert_data['type'] = 'blue'
                socketio.emit('new_alert', alert_data)
            elif rsi > 95:
                alert_data['type'] = 'red'
                socketio.emit('new_alert', alert_data)
            elif rsi > 90:
                alert_data['type'] = 'orange'
                socketio.emit('new_alert', alert_data)
        else:
            print(f"Indicators for {symbol} are None, skipping...")


@app.route('/')
def index():
    return render_template('index.html')

# Create a lock for thread safety
data_processing_lock = threading.Lock()

# Step 5: Main monitoring function
def monitor_stock_indicators():
    while True:
        with data_processing_lock:
            stock_data_1h, stock_data_daily, stock_data_minute = fetch_stock_data(stock_symbols)
            if (stock_data_1h is not None and not stock_data_1h.empty and
                stock_data_daily is not None and not stock_data_daily.empty):
                results = process_stock_data(stock_data_1h, stock_data_daily, stock_data_minute, stock_symbols)
                if results:
                    check_and_emit_alerts(results)
                else:
                    print("No valid indicators calculated, retrying...")
            else:
                print("No valid stock data found, retrying...")
        time.sleep(15 * 60)

@socketio.on('refresh_request')
def handle_refresh_request():
    with data_processing_lock:
        try:
            print("Received refresh request from client.")
            stock_data_1h, stock_data_daily, stock_data_minute = fetch_stock_data(stock_symbols)
            if (stock_data_1h is not None and not stock_data_1h.empty and
                stock_data_daily is not None and not stock_data_daily.empty):
                results = process_stock_data(stock_data_1h, stock_data_daily, stock_data_minute, stock_symbols)
                if results:
                    check_and_emit_alerts(results)
                    emit('refresh_complete', {'status': 'success'})
                else:
                    print("No valid indicators calculated.")
                    emit('refresh_complete', {'status': 'failure', 'message': 'No valid indicators calculated.'})
            else:
                print("No valid stock data found.")
                emit('refresh_complete', {'status': 'failure', 'message': 'No valid stock data found.'})
        except Exception as e:
            print(f"Error during refresh: {e}")
            emit('refresh_complete', {'status': 'failure', 'message': str(e)})

if __name__ == '__main__':
    # Run stock monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_stock_indicators)
    monitor_thread.daemon = True
    monitor_thread.start()
    socketio.run(app, host='0.0.0.0', port=5001)
    
