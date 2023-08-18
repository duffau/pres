import pandas as pd
from datetime import datetime, date
import requests

DAILY_FREQ = "d"
WEEKLY_FREQ = "w"
MONTHLY_FREQ = "m"
CSV_HEADER = "Date,Open,High,Low,Close,Adjusted_close,Volume"


def get_historical_data(
    symbol,
    freq="d",
    api_key=None,
    from_date=None,
    to_date=str(date.today()),
) -> pd.DataFrame:
    if isinstance(from_date, (date, datetime)):
        from_date = from_date.strftime("%Y-%m-%d")

    if isinstance(to_date, (date, datetime)):
        to_date = to_date.strftime("%Y-%m-%d")

    if freq not in [DAILY_FREQ, WEEKLY_FREQ, MONTHLY_FREQ]:
        raise ValueError(f"freq: '{freq}' not recognized")

    url = f"https://eodhistoricaldata.com/api/eod/{symbol}"
    url_params = {"period": freq, "api_token": api_key, "fmt": "csv", "to": to_date}
    if from_date is not None:
        url_params.update({"from": from_date})
    resp = requests.get(url, params=url_params)
    resp.raise_for_status()
    csv_content = resp.text
    if csv_content.strip() == "Value" or csv_content is None:
        csv_content = CSV_HEADER + "\n"
    return csv_content
