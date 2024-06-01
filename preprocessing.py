import numpy as np
import pickle
import pandas as pd

# Define the file paths
file_paths = [
    "index_history.pkl",
    "saved_indexes.pkl",
    "saved_symbols.pkl",
    "ticker_history.pkl",
    "event_clean.pkl",
]

# Load the data from each file
data_dict = {}

for path in file_paths:
    with open("raw_data/" + path, "rb") as file:
        data = pickle.load(file, encoding="latin1")
        data_dict[path] = data

# Access the raw data
index_history = data_dict["index_history.pkl"]
saved_indexes = data_dict["saved_indexes.pkl"]
saved_tickers = data_dict["saved_symbols.pkl"]
ticker_history = data_dict["ticker_history.pkl"]
event_history = data_dict["event_clean.pkl"]


# Define important indexes
us_indices = ["^GSPC", "^DJI", "^IXIC", "^NYA", "^XAX", "^RUT"]
closing_time_from_midnight = pd.DateOffset(
    hours=16
)  # Closing time of NYSE(4pm). Closing time matters because we want
# to determine if the closing time is before or after event. # For pre-market earning events,
# the 'before' data should not include the data from that day.
closing_time_from_midnight_ind = {}
us_indexes = ["^GSPC", "^DJI", "^IXIC", "^NYA", "^XAX", "^RUT"]
for ind in us_indexes:
    closing_time_from_midnight_ind[ind] = pd.DateOffset(
        hours=16
    )  # Closing time of NYSE
closing_time_from_midnight_ind["^VIX"] = pd.DateOffset(hours=15.25)
closing_time_from_midnight_ind["^BUK100P"] = pd.DateOffset(hours=17.5)
closing_time_from_midnight_ind["^FTSE"] = pd.DateOffset(hours=16.5)
closing_time_from_midnight_ind["^GDAXI"] = pd.DateOffset(hours=18)
closing_time_from_midnight_ind["^FCHI"] = pd.DateOffset(hours=17.5)
closing_time_from_midnight_ind["^STOXX50E"] = pd.DateOffset(hours=18)
closing_time_from_midnight_ind["^N100"] = pd.DateOffset(hours=17.5)
closing_time_from_midnight_ind["^BFX"] = pd.DateOffset(hours=17.5)
closing_time_from_midnight_ind["IMOEX.ME"] = pd.DateOffset(hours=16)
closing_time_from_midnight_ind["^N225"] = pd.DateOffset(hours=15)
closing_time_from_midnight_ind["^HSI"] = pd.DateOffset(hours=16)
closing_time_from_midnight_ind["000001.SS"] = pd.DateOffset(hours=15)
closing_time_from_midnight_ind["399001.SZ"] = pd.DateOffset(hours=15)
closing_time_from_midnight_ind["^STI"] = pd.DateOffset(hours=17)

# Using last num_days_before trading days. We dont care about when those days occured relative to each other


def earnings_data_no_of_trading_days(num_days_before, num_days_after):

    d = {}

    indices = ["^GSPC", "^IXIC"]

    for sym in event_history:
        dum = []
        event_date = event_history[sym].index

        for j in range(len(event_history[sym])):
            dic = {"Before": 0, "After": 0, "Earning": 0}
            dic["Earning"] = event_history[sym].iloc[[j]]

            ticker_closing = ticker_history[sym].index + closing_time_from_midnight
            dfb = ticker_history[sym][["Close", "Volume", "High", "Low", "Open"]][
                ticker_closing <= event_date[j]
            ]
            dfb.sort_index(inplace=True)
            dfb = dfb[-num_days_before:]
            dfb = dfb.rename(columns={"Close": sym})
            dfb.index = range(-len(dfb), 0, 1)
            for ind in indices:
                index_closing = (
                    index_history[ind].index + closing_time_from_midnight_ind[ind]
                )
                dfbi = index_history[ind][["Close"]][index_closing <= event_date[j]]
                dfbi.sort_index(inplace=True)
                dfbi = dfbi[-num_days_before:]
                dfbi = dfbi.rename(columns={"Close": ind})
                dfbi.index = range(-len(dfbi), 0, 1)
                dfb = dfb.join(dfbi)
            dic["Before"] = dfb

            ticker_closing = ticker_history[sym].index + closing_time_from_midnight
            dfa = ticker_history[sym][["Close", "Volume", "High", "Low", "Open"]][
                ticker_closing > event_date[j]
            ]
            dfa.sort_index(inplace=True)
            dfa = dfa[:num_days_after]
            dfa = dfa.rename(columns={"Close": sym})
            dfa.index = range(1, len(dfa) + 1)
            for ind in indices:
                index_closing = (
                    index_history[ind].index + closing_time_from_midnight_ind[ind]
                )
                dfai = index_history[ind][["Close"]][index_closing > event_date[j]]
                dfai.sort_index(inplace=True)
                dfai = dfai[:num_days_after]
                dfai = dfai.rename(columns={"Close": ind})
                dfai.index = range(1, len(dfai) + 1)
                dfa = dfa.join(dfai)
            dic["After"] = dfa

            dum.append(dic)
        d[sym] = dum
    return d


def daily_volatility(df):
    return (df["High"] - df["Low"]) / df["Open"]


def priced_in_label(ticker_0, ticker_a, index_0, index_a, surprise):
    return surprise


def generate_variables():
    """
    Generate input and output variables, standardize them.
    """
    df = earnings_data_no_of_trading_days(60, 60)
    X = []
    Y = []
    tickers = []
    for sym in df:
        for evt_idx in range(len(df[sym])):
            df_sym_idx = df[sym][evt_idx]

            ticker_b = df_sym_idx["Before"][sym].values
            index_b = df_sym_idx["Before"]["^GSPC"].values
            volume_b = df_sym_idx["Before"]["Volume"].values
            volatility_b = daily_volatility(df_sym_idx["Before"]).values

            ticker_a = df_sym_idx["After"][sym].values
            index_a = df_sym_idx["After"]["^GSPC"].values
            volume_a = df_sym_idx["After"]["Volume"].values
            volatility_a = daily_volatility(df_sym_idx["After"]).values

            if len(ticker_b) == 60 and len(index_b) == 60:
                ticker_0 = ticker_b[-1]
                index_0 = index_b[-1]
                volume_0 = volume_b[-1]
                volatility_0 = volatility_b[-1]

                ticker_b = ticker_b / ticker_0
                index_b = index_b / index_0
                volume_b = volume_b / volume_0
                volatility_b = volatility_b / volatility_0

                surprise = df_sym_idx["Earning"]["Surprise(%)"].values

                y = priced_in_label(ticker_0, ticker_a, index_0, index_a, surprise)

                ticker_a = ticker_a / ticker_0
                index_a = index_a / index_0
                volume_a = volume_a / volume_0
                volatility_a = volatility_a / volatility_0

                x = np.concatenate(
                    (
                        ticker_b,
                        ticker_a,
                        index_b,
                        index_a,
                        volume_b,
                        volume_a,
                        volatility_b,
                        volatility_a,
                    )
                )
                X.append(x)
                Y.append(y)
                tickers.append(sym)

    X = np.array(X)
    Y = np.array(Y).flatten()
    tickers = np.array(tickers)

    X = np.nan_to_num(X, nan=0.0)
    Y = np.nan_to_num(Y, nan=0.0)

    X = (X - X.mean()) / (X.std())
    Y = (Y - Y.mean()) / (Y.std())

    np.save("processed_data/X.npy", X)
    np.save("processed_data/Y.npy", Y)
    np.save("processed_data/ticker_labels.npy", tickers)
    return X, Y, tickers
