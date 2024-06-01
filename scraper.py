import yfinance as yf  
import requests  
import pandas as pd  
import pickle  


#Defining functions
def list_wikipedia_nasdaq100() -> pd.DataFrame:
    # Function to scrape the list of Nasdaq-100 companies from Wikipedia
    # Ref: https://stackoverflow.com/a/75846060/ (Reference to the source of the solution)
    url = 'https://en.m.wikipedia.org/wiki/Nasdaq-100'  # URL of the Wikipedia page containing Nasdaq-100 companies
    return pd.read_html(url, attrs={'id': "constituents"}, index_col='Ticker')[0]  # Reads the HTML table with id 'constituents' and sets 'Ticker' as the index



def list_ticker() -> pd.DataFrame:
    # Function to scrape the list of world indices from Yahoo Finance
    # Ref: https://stackoverflow.com/a/75846060/ (Reference to the source of the solution)
    url = "https://finance.yahoo.com/world-indices/"  # URL of the Yahoo Finance page containing world indices
    html = requests.get(url).content  # Sends an HTTP GET request to the URL and gets the content of the page
    return pd.read_html(html)  # Reads the HTML content and returns all tables found on the page as a list of DataFrames


# Get the DataFrame of Nasdaq-100 companies
index_wiki = list_wikipedia_nasdaq100()

# Extract the list of ticker symbols from the DataFrame index
symbols = index_wiki.index.to_list()

# Create a Tickers object from yfinance for the list of Nasdaq-100 symbols
big = yf.Tickers(symbols)
# Get the list of DataFrames containing world indices
index_list = list_ticker()

# Extract the 'Symbol' column from the first DataFrame in the list of world indices
# Convert it to a list and select the first 20 symbols
major_indices = list(index_list[0]["Symbol"])[:20]

# Create a dictionary where the keys are ticker symbols and the values are historical price data
dict_tickers = {i: big.tickers[i].history(period="60mo") for i in symbols} 
# 'big.tickers[i].history(period="60mo")' fetches the historical data for the past 60 months for each ticker


# Make a copy of the event_tickers_1 DataFrame
split_data = event_tickers_1.copy()

# Loop through the first 10 symbols in the list
for sym in symbols[:10]:
    # Get the historical stock price data for the current symbol
    stock_price = dict_tickers[sym]
    # Get the earnings data for the current symbol from event_tickers_1
    earnings_data = event_tickers_1[sym]
    # Initialize a list to hold the split data
    split = list(range(30))
    
    # Loop through the range 1 to 29
    for i in range(1, 30):
        # Get the current earnings date
        a = earnings_data.index[i]
        # Get the previous earnings date
        b = earnings_data.index[i-1]
        # Select stock prices before the current earnings date
        split[i] = stock_price[stock_price.index < a]
        # Further filter stock prices to include only those after the previous earnings date
        split[i] = stock_price[stock_price.index > b]
    
    # Update the split_data dictionary with the new split data for the current symbol
    split_data.update({sym: split})


# Create a Tickers object from yfinance for the list of major indices
big_index = yf.Tickers(major_indices)

# Create a dictionary where the keys are major index symbols and the values are historical price data
dict_index = {i: big_index.tickers[i].history(period="60mo") for i in major_indices}



# Save the list of symbols to a file using pickle
with open('raw_data/saved_symbols.pkl', 'wb') as f:
    pickle.dump(symbols, f)

# Save the list of major indices to a file using pickle
with open('raw_data/saved_indexes.pkl', 'wb') as f:
    pickle.dump(major_indices, f)

# Save the dictionary of historical stock price data to a file using pickle
with open('raw_data/ticker_history.pkl', 'wb') as f:
    pickle.dump(dict_tickers, f)

# Save the dictionary of historical index price data to a file using pickle
with open('raw_data/index_history.pkl', 'wb') as f:
    pickle.dump(dict_index, f)