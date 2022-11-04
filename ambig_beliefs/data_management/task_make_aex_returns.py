"""
Computes 6 month ahead and 12 month ahead returns from daily yahoo aex data
"""
import pandas as pd
import pytask

from config import IN_DATA
from config import OUT_DATA


DEPENDS_ON = IN_DATA / "aex" / "aex_yahoo.csv"
PRODUCES = OUT_DATA / "aex_returns.pickle"


@pytask.mark.depends_on(DEPENDS_ON)
@pytask.mark.produces(PRODUCES)
def task_make_aex_returns(depends_on, produces):

    aex_data_raw = pd.read_csv(depends_on)[["Date", "Close"]]
    aex_data_raw["Date"] = pd.to_datetime(aex_data_raw["Date"])
    aex_data_raw.set_index("Date", inplace=True)

    # Create new dataframe with a row for every day in between first and last observation
    date_idx = pd.date_range(
        start=aex_data_raw.index.min(),
        end=aex_data_raw.index.max(),
        freq="D",
    )
    aex_returns = pd.DataFrame(index=date_idx)
    aex_returns.index.name = "date"
    aex_returns["closing_price"] = aex_data_raw["Close"]

    # carry last known value forward to days on which there is no trade or missing data
    aex_returns["closing_price"] = aex_returns["closing_price"].ffill()

    # create returns
    for m in [6, 12]:
        current_price = aex_returns["closing_price"]
        future_date = aex_returns.index + pd.DateOffset(n=m, months=1)
        future_price = aex_returns["closing_price"].reindex(future_date).values
        aex_returns[f"aex_{m}m_ahead_return"] = (
            future_price - current_price
        ) / current_price

    aex_returns.to_pickle(produces)
