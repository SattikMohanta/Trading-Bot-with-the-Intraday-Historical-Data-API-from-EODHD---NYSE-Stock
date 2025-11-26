import requests
import json
import pandas as pd
import datetime
import time
import os
import pandas_ta as ta
import logging
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# 1. Configuration
API_KEY = '6_________________________4'   # Replace with your actual API Key
symbol = 'NVDA.US'               
interval = '1m'                  # 1-minute intervals
from_date_str = '01-01-2025'     # MM-DD-Y (yyyy)
to_date_str   = '30-03-2025'     # MM-DD-Y (yyyy)
RSI_PERIOD = 14                  # default

# 2. Convert to Python datetime
from_date_obj = datetime.datetime.strptime(from_date_str, '%d-%m-%Y')
to_date_obj   = datetime.datetime.strptime(to_date_str, '%d-%m-%Y')

# 3. Convert to Unix timestamps
from_timestamp = int(from_date_obj.timestamp())
to_timestamp   = int(to_date_obj.timestamp())


#4. The Intraday Historical Data API endpoint typically follows this format:
url = (
    f'https://eodhistoricaldata.com/api/intraday/{symbol}'
    f'?api_token={API_KEY}'
    f'&interval={interval}'
    f'&from={from_timestamp}'
    f'&to={to_timestamp}'
    f'&fmt=json'
)

# 6. Convert to a Pandas DataFrame
if response.status_code == 200:             
    data_json = json.loads(response.text)
    df = pd.DataFrame(data_json)
    print(df)
else:
    print("Failed to retrieve data:", response.status_code)
    
    df.rename(columns={ 
        't': 'timestamp',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }, inplace=True)
    # Convert timestamps to UTC datetime and set as index
   
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)

    print(df.head())
    print(df.tail())
    print(f"Error retrieving data: {response.status_code} - {response.text}")


    ------------------------------------------------------------------------------------------------------------------------------------------------------

Once we have reliable market data at your fingertips, the next logical step is to automate how you act on it. A *trading bot* allows us to execute strategies systematically, free from human bias and around the clock. Whether you’re focusing on short-term signals or multi-day patterns, bots help you streamline the process of analyzing data, entering positions, and managing risk. In the next section, we’ll explore how to create a simple RSI-based trading bot using the data you’ve extracted.


-------------------------------------------------------------------------------------------------------------------------------------------------------
**_Phase 2: Simple RSI-Based Trading Bot_**

With high-quality intraday data in hand, you can begin experimenting with trading strategies. A good starting point is the RSI (Relative Strength Index), a momentum oscillator that fluctuates between 0 and 100. Traditionally:

    - RSI below 30: Potentially oversold (a buy signal for some traders).
    - RSI above 70: Potentially overbought (a sell signal for some traders).

Below is an example script that downloads the same data, calculates the RSI using pandas_ta, and outputs hypothetical buy/sell signals based on RSI thresholds.


    #  Calculate RSI with a 14-bar period
    df['RSI'] = ta.rsi(df['close'], length=14)

    #  Generate trading signals
    df['Signal'] = 0
    df.loc[df['RSI'] <= 30, 'Signal'] = 1   # Buy signal
    df.loc[df['RSI'] >= 70, 'Signal'] = -1  # Sell signal

    print(df[['open', 'high', 'low', 'close', 'volume', 'RSI', 'Signal']].head(100))

    #  Simulating trades
    position = 0  # 1 if holding a buy, -1 if short, 0 if neutral
    for idx, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            print(f"{idx} => BUY at {row['close']:.2f} (RSI={row['RSI']:.2f})")
            position = 1
        elif row['Signal'] == -1 and position == 1:
            print(f"{idx} => SELL at {row['close']:.2f} (RSI={row['RSI']:.2f})")
            position = 0
    else:
            print(f"Error retrieving data: {response.status_code} - {response.text}") 


--------------------------------------------------------------------------------------------------------------------
***_Summary_***

1. Importing Libraries:

        requests for API requests,
        json for handling data,
        datetime for date manipulation,
        pandas for data processing,
        pandas_ta for technical analysis.

2. Configuration Parameters:

    API_KEY: API access key.
    symbol: Apple stock (AAPL.US).
    interval: Data with a 1-minute interval.
    from_date_str and to_date_str: Date range to fetch.

3. Date Conversion:

    Converts string dates into UNIX timestamps.

4. Constructing the URL and sending an HTTP request to the API:

    The request is sent to eodhistoricaldata.com with the defined parameters.

5. Data Processing:

    If the request is successful (status_code == 200), the response is converted into a Pandas DataFrame.
    Columns (o, h, l, c, v) are renamed to more meaningful names (open, high, low, close, volume).
    Timestamps are adjusted.

6. Calculating RSI:

    Uses pandas_ta.rsi with a 14-period setting.

7. Generating Trading Signals:

    Buy (Signal = 1): When RSI ≤ 30 (oversold).
    Sell (Signal = -1): When RSI ≥ 70 (overbought).

8. Simulating Trades:

    A position variable tracks if a trade is open.
    When a buy signal is generated, it prints a buy action with price and RSI.
    When a sell signal is generated, it prints a sell action with price and RSI.

9. Error Handling:

    If the API request fails, it displays an error message with the status code.



def plot_closing_price(
    ax: plt.Axes, 
    dataframe: pd.DataFrame, 
    ticker: str = "NVDA.US", 
    color: str = 'blue',  # Default color changed to match the plot
    linewidth: float = 2.0
) -> None:
    """ 
    (Docstring)
    Plots the closing price of a stock on the provided Axes object.

    Parameters:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        dataframe (pd.DataFrame): DataFrame containing stock data with a 'close' column.
        ticker (str): The stock ticker symbol for the title.
        color (str): Color of the line in the plot.
        linewidth (float): Width of the line in the plot.

    Returns:
        None
    """
    
    # Plot Closing Price
    ax.plot(dataframe.index, dataframe['close'], label='Close Price', color=color, linewidth=linewidth)
    
    # Set Titles & Labels
    ax.set_title(f"{ticker} - Closing Price", fontsize=14, fontweight='bold')
    ax.set_ylabel("Price ($)", fontsize=12)
    
    # Improve Readability
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for better visibility

# Example usage:
#fig, ax = plt.subplots()
#plot_closing_price(ax, your_dataframe_here, ticker="NVDA.US")
#plt.show()


def create_price_rsi_plot(dataframe: pd.DataFrame, rsi: pd.Series, figsize: tuple = (12, 6)) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Creates a plot with two subplots: one for the closing price and another for the RSI.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing stock data with a 'close' column.
        rsi (pd.Series): Series containing RSI values.
        figsize (tuple): Size of the figure.

    Returns:
        tuple: A tuple containing the Figure and a list of Axes objects.
    """
    fig, axs = plt.subplots(nrows=2, figsize=figsize, sharex=True)

    # Plot Closing Price
    axs[0].plot(dataframe.index, dataframe['close'], label='Close Price', color='blue', linewidth=1.5)
    axs[0].set_title("Closing Price", fontsize=14, fontweight='bold')
    axs[0].set_ylabel("Price (USD)", fontsize=12)
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Plot RSI
    axs[1].plot(rsi.index, rsi, label='RSI', color='orange', linewidth=1.5)
    axs[1].axhline(70, linestyle='--', color='red', alpha=0.5)  # Overbought line
    axs[1].axhline(30, linestyle='--', color='green', alpha=0.5)  # Oversold line
    axs[1].set_title("Relative Strength Index (RSI)", fontsize=14, fontweight='bold')
    axs[1].set_ylabel("RSI", fontsize=12)
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.6)

    # Improve x-axis visibility
    axs[1].tick_params(axis="x", rotation=45)  # Rotate x-axis labels for better visibility

    plt.xlabel("Date", fontsize=12)
    
    return fig, axs



    def plot_rsi(ax, df):
   
    #Ensure RSI exists in the dataframe
    if 'RSI' not in df.columns:
        print("Error: RSI column not found in DataFrame.")
        return
    
    #Plot RSI line
    ax.plot(df.index, df['RSI'], label="RSI", color="blue", linewidth=2.0)
    
    #Overbought & Oversold Levels
    overbought_level, oversold_level = 70, 30
    ax.axhline(overbought_level, linestyle="--", color="green", label=f"Overbought ({overbought_level})")
    ax.axhline(oversold_level, linestyle="--", color="red", label=f"Oversold ({oversold_level})")
    
    #Format the chart
    ax.set_title("Relative Strength Index (RSI)", fontsize=14, fontweight="bold")
    ax.set_ylabel("RSI Value", fontsize=12)
    ax.set_xlabel("Date/Time", fontsize=12)
    
    #Set Y-axis limits (optional)
    ax.set_ylim(0, 100)
    
    #Enable grid and legend
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

#Usage Example
fig, axs = plt.subplots(2, figsize=(12, 6), sharex=True)
plot_rsi(axs[1], df)
plt.show()


# Configuration
symbol = "NVDA.US"
interval = "10m"
range_days = "90d"
api_key = "6_________________________4"   # Replace with your actual API Key
from_date_str = '01-01-2025'     # MM-DD-Y (yyyy)
to_date_str   = '31-03-2025'     # MM-DD-Y (yyyy)

url = f"https://eodhistoricaldata.com/api/eod/NVDA.US?api_token=6_________________________4&from=01-01-2025&to=31-03-2025&fmt=json"

# Fetch historical/eod data
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)

# Convert and sort data
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

# Calculate RSI
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(RSI_PERIOD).mean()
avg_loss = loss.rolling(RSI_PERIOD).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Plot RSI with Plotly
fig = go.Figure()

# RSI Line
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['RSI'],
    mode='lines',
    name='RSI',
    line=dict(color='dodgerblue', width=2)
))

# Overbought line (70)
fig.add_trace(go.Scatter(
    x=[df.index[0], df.index[-1]],
    y=[70, 70],
    mode='lines',
    name='Overbought (70)',
    line=dict(color='red', width=1, dash='dash'),
    showlegend=True
))

# Oversold line (30)
fig.add_trace(go.Scatter(
    x=[df.index[0], df.index[-1]],
    y=[30, 30],
    mode='lines',
    name='Oversold (30)',
    line=dict(color='green', width=1, dash='dash'),
    showlegend=True
))

# Text annotations
fig.add_annotation(x=df.index[-1], y=70, text="Overbought", showarrow=False, font=dict(color="red"))
fig.add_annotation(x=df.index[-1], y=30, text="Oversold", showarrow=False, font=dict(color="green"))

# Layout
fig.update_layout(
    title=f"RSI Chart for {symbol}",
    xaxis_title="Date",
    yaxis_title="RSI",
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    template="simple_white",
    height=500
)

fig.show()



# Configuration
symbol = "NVDA.US"
interval = "15m"
range_days = "90d"
api_key = "6_________________________4"   # Replace with your actual API Key
from_date_str = '01-01-2025'     # MM-DD-Y (yyyy)
to_date_str   = '31-03-2025'     # MM-DD-Y (yyyy)

url = f"https://eodhistoricaldata.com/api/eod/NVDA.US?api_token=6_________________________4&from=01-01-2025&to=31-03-2025&fmt=json"
# Fetch data
response = requests.get(url)
data = response.json()

# Check structure
print("Sample record from response:")
print(data[:5]) # Preview first 5 sliced entry


# Convert to DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])  # Correct column name
df.set_index('date', inplace=True)
df.sort_index(inplace=True)


# Plot candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

fig.update_layout(
    title=f'{symbol} Intraday Candlestick Chart ({interval})',
    xaxis_title='Time',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='seaborn',
    height=600
)

fig.show()



# Configuration
symbol = "NVDA.US"
interval = "30m"
range_days = "90d"
api_key = "6_________________________4"   # Replace with your actual API Key
from_date_str = '01-01-2025'     # MM-DD-Y (yyyy)
to_date_str   = '31-03-2025'     # MM-DD-Y (yyyy)

url = f"https://eodhd.com/api/eod/NVDA.US?interval=5m&range=1d&api_token=6_________________________4&fmt=json"
# Fetch data
response = requests.get(url)
data = response.json()

# === API REQUEST ===
url = f"https://eodhistoricaldata.com/api/eod/NVDA.US?api_token=6_________________________4&from=01-01-2025&to=31-03-2025&fmt=json"
response = requests.get(url)

# === ERROR HANDLING ===
if response.status_code != 200:
    raise Exception(f"API request failed with status {response.status_code}: {response.text}")

# === PARSE JSON TO DATAFRAME ===
data = response.json()
df = pd.DataFrame(data)

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)


# === GROUP BY WEEK ===
df['week'] = df.index.to_period('W')
weekly_close = df.groupby('week')['close'].apply(list)

# === PREPARE DATA FOR BOXPLOT ===
box_data = weekly_close.tolist()
week_labels = [str(week) for week in weekly_close.index]

# === PLOT BOXPLOT ===
plt.figure(figsize=(14, 6))
box = plt.boxplot(box_data, patch_artist=True)

# === COLOR STYLING ===
colors = ['lightblue', 'lightgreen', 'lightcoral', 'wheat', 'lavender']
for patch, color in zip(box['boxes'], colors * (len(box['boxes']) // len(colors) + 1)):
    patch.set_facecolor(color)

plt.title(f'Weekly Closing Price Distribution for {symbol}', fontsize=16)
plt.xlabel('Week')
plt.ylabel('Closing Price')
plt.xticks(ticks=range(1, len(week_labels)+1), labels=week_labels, rotation=45, fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
