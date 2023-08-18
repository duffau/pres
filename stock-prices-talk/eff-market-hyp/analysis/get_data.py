import os
import pandas as pd
from io import StringIO
import time

# from settings import ALPHA_VANTAGE_API_KEY, RAPIDAPI_API_KEY
# from apis import alphavantage as av
from apis import yahoofinance as yf
# from apis import worldbank as wb
# from apis import eodhistoricaldata as eod

# API_KEY = os.getenv("EOD_API_KEY")
# API_KEY = os.getenv("API_DOJO_API_KEY")
DATE_KEY = "Date"
DELAY = 5

tickers = ["^GSPC", "TSLA", "^DJI", "GME", "HPE", "GOOG"]
data_dir = "./data"
for ticker in tickers:
    print(f"Getting {ticker}...")
    
    # csv = eod.get_historical_data(symbol=ticker, api_key=API_KEY)
    csv = yf.get_historical_data_v7(symbol=ticker)
    df = pd.read_csv(StringIO(csv))
    df[DATE_KEY] = pd.to_datetime(df[DATE_KEY])
    
    df.set_index("Date", inplace=True)
    df.to_pickle(f"{data_dir}/{ticker}_prices.pickle")
    print("Sleeping...")
    time.sleep(DELAY)
