import requests
import time
import csv
from io import StringIO

def get_historical_data(symbol, api_key):
    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"

    querystring = {"symbol":symbol, "region":"US"}

    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com",
        }

    json_resp = requests.get(url, headers=headers, params=querystring).json()
    breakpoint()
    price_records = json_resp["prices"]
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=price_records[0].keys())
    writer.writeheader()
    writer.writerows(price_records)
    csv_string = output.getvalue()
    return csv_string


def get_historical_data_v7(symbol):                                                                                             
    url = "https://query1.finance.yahoo.com/v7/finance/download"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    url = "/".join([url, symbol])
    query_params = None
    query_params = {
        "period1":-1325635200,
        "period2": int(time.time()),
        "interval":"1d",
        "events": "history", 
        "includeAdjustedClose": "true"
    }
    return requests.get(url, params=query_params, headers=headers).text
