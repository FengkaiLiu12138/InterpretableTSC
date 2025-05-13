import datetime as dt
import pandas as pd
import yfinance as yf

'''
Example usage:
symbol = "^FTSE"
interval = "1d"
chunk_days = 30
start_date = dt.datetime(2000, 1, 1)
end_date = dt.datetime(2025, 4, 22)
'''
class FTSEDataCatcher:
    def __init__(self, symbol, interval, chunk_days, start_date, end_date):
        self.symbol = symbol
        self.interval = interval
        self.chunk_days = chunk_days
        self.start_date = start_date
        self.end_date = end_date
        self.df_list = []

    def catch_data(self, save_path=None):
        try:
            temp_start = self.start_date

            while temp_start < self.end_date:
                temp_end = temp_start + dt.timedelta(days=self.chunk_days)

                if temp_end > self.end_date:
                    temp_end = self.end_date

                print(f"Fetching data from {temp_start} to {temp_end}...")
                data = yf.download(
                    self.symbol,
                    start=temp_start,
                    end=temp_end,
                    interval=self.interval,
                    progress=False
                )

                if not data.empty:
                    self.df_list.append(data)

                temp_start = temp_end

            if self.df_list:
                full_data = pd.concat(self.df_list)
                full_data = full_data[~full_data.index.duplicated(keep='first')]
                full_data.sort_index(inplace=True)

                print("Number of data：", full_data.shape[0])
                print(full_data.head())

                full_data.to_csv(save_path, encoding="utf-8")
                print(f"Saved to {save_path}")
            else:
                print("Data fetch failed.")
        except Exception as e:
            print(f"An error occurred: {e}")



if __name__ == "__main__":
    symbol = "^FTSE"
    interval = "1m"
    chunk_days = 7
    start_date = dt.datetime(2025, 4, 1)
    end_date = dt.datetime(2025, 5, 11)

    ftse_data_catcher = FTSEDataCatcher(symbol, interval, chunk_days, start_date, end_date)
    ftse_data_catcher.catch_data("../Dataset/ftse_minute_data_may.csv")