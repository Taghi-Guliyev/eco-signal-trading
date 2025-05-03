from meteostat import Stations, Daily
import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np

#%% Climate Stress Index Calculation

# Step 1: Define the locations (city centers in key states)
locations = {
    'Illinois': {'lat': 40.1164, 'lon': -88.2434},  # Champaign
    'Iowa': {'lat': 41.5868, 'lon': -93.6250},      # Des Moines
    'Nebraska': {'lat': 40.8136, 'lon': -96.7026}   # Lincoln
}

# Step 2: Define time period
start = datetime(2006, 1, 1)
end = datetime(2025, 12, 31)

# Step 3: Fetch monthly average temperature and total precipitation
climate_data = {}

for state, coords in locations.items():
    stations = Stations()
    station = stations.nearby(coords['lat'], coords['lon']).fetch(1)
    station_id = station.index[0]

    daily_data = Daily(station_id, start, end)
    daily_data = daily_data.fetch()

    # Resample to monthly
    monthly = daily_data.resample('M').agg({
        'tavg': 'mean',  # Average monthly temperature
        'prcp': 'sum'    # Total monthly precipitation
    })

    climate_data[state] = monthly

# Step 4: Create DataFrame combining all states
climate_df = pd.DataFrame(index=climate_data['Illinois'].index)

for state in locations.keys():
    climate_df[f'Temp_{state}'] = climate_data[state]['tavg']
    climate_df[f'Precip_{state}'] = climate_data[state]['prcp']

# Step 5: Calculate Temperature and Precipitation Anomalies
# (relative to mean across the entire sample)
climate_df['Temp_Anomaly'] = climate_df[[f'Temp_{state}' for state in locations]].mean(axis=1) - \
                              climate_df[[f'Temp_{state}' for state in locations]].mean(axis=1).mean()

climate_df['Precip_Anomaly'] = climate_df[[f'Precip_{state}' for state in locations]].mean(axis=1) - \
                                climate_df[[f'Precip_{state}' for state in locations]].mean(axis=1).mean()

# Step 6: Build Climate Stress Index
climate_df['Climate_Stress_Index'] = climate_df['Temp_Anomaly'] - climate_df['Precip_Anomaly']
climate_df = climate_df.shift(1)

#%% ADM stock information gathering

adm = yf.download('ADM', start='2004-12-01', end='2025-04-01', interval='1mo')
adm_monthly = adm['Close'].resample('M').last()
adm_return = adm_monthly.pct_change()
adm_return = adm_return.iloc[14:]

# Merge ADM with Climate Stress Index
final_df = pd.merge(pd.merge(adm_monthly, adm_return, left_index=True, right_index=True),
                     climate_df[['Climate_Stress_Index']], left_index=True, right_index=True)

import pandas_datareader.data as web

# Define time range
start = datetime(2005, 11, 1)
end = datetime(2025, 4, 1)

#%% Economic variables gathering

# Fetch CPI Food Inflation directly from FRED
cpi_food = web.DataReader('CPIFABSL', 'fred', start, end)

# Calculate monthly percent change
cpi_food['CPI_Food_Returns'] = cpi_food['CPIFABSL'].pct_change()

# Shift CPI Food to end of previous month
cpi_food.index = cpi_food.index + pd.offsets.MonthEnd(0)
cpi_food = cpi_food.shift(1)


usd_index = yf.download('DX-Y.NYB', start='2005-12-01', end='2025-04-01', interval='1mo')['Close']
usd_index.index = usd_index.index + pd.offsets.MonthEnd(0)
usd_index = usd_index.shift(1)

usd_index_monthly = usd_index.resample('M').last()
usd_index_return = usd_index_monthly.pct_change()

# Assuming usd_index is your DataFrame
usd_index = usd_index.iloc[2:]

# Merge correctly
macro_df = pd.concat([cpi_food['CPI_Food_Returns'], usd_index, usd_index_return], axis=1)
macro_df.columns = ['CPI_Food_Returns', 'USD_Index' ,'USD_Index_Return']

merged_df = pd.merge(final_df, macro_df, left_index=True, right_index=True, how='inner')

#%% Other alternative datas gathering

# Load Fertilizer Prices
fertilizer = pd.read_excel('fert.xlsx')

# Step 1: Force numeric conversion
cols_to_convert = ['P-rock', 'DAP', 'TSP', 'urea', 'Potas-c']

for col in cols_to_convert:
    fertilizer[col] = pd.to_numeric(fertilizer[col], errors='coerce')

# Step 2: Now safely take the mean
fertilizer['Fertilizer_Index'] = fertilizer[cols_to_convert].mean(axis=1)

print(fertilizer[['Fertilizer_Index']].head())

# Fix Date format: '2000M01' → '2000-01-01'
fertilizer['Date'] = pd.to_datetime(fertilizer['Date'], format='%YM%m')
fertilizer = fertilizer.set_index('Date')

# Keep only the index column for modeling
fertilizer_monthly = fertilizer[['Fertilizer_Index']]
fertilizer_monthly.index = fertilizer_monthly.index + pd.offsets.MonthEnd(0)
fertilizer_monthly = fertilizer_monthly.shift(1)
fertilizer_monthly = fertilizer_monthly.iloc[2:]

# Reload drought data with correct separator
drought = pd.read_csv('drought.csv', delimiter=';')

# Correct parsing
drought['MapDate'] = pd.to_datetime(drought['MapDate'], format='%Y%m%d')
drought = drought.set_index('MapDate')

# Optional: Filter states if necessary
drought = drought[drought['StateAbbreviation'].isin(['IA', 'IL', 'NE'])]

# Calculate Drought Severity Score (example: weighted sum)
drought['Drought_Severity'] = (drought['D1'] * 1 + drought['D2'] * 2 + drought['D3'] * 3 + drought['D4'] * 4)

# Resample to monthly average
drought_monthly = drought['Drought_Severity'].resample('M').mean()
drought_monthly = drought_monthly.shift(1)
drought_monthly = drought_monthly.iloc[2:]

alt_df = pd.merge(fertilizer_monthly, drought_monthly, left_index=True, right_index=True, how='inner')
merged_df = pd.merge(merged_df, alt_df, left_index=True, right_index=True, how='inner')

#%% Interaction terms and RSI calculation & dataframe merging

import ta

# Fetch ADM daily prices
adm_full = adm_monthly
adm_full2 = adm['High'].resample('M').last()
adm_full3 =  adm['Low'].resample('M').last()

# Relative Strength Index (RSI)
rsi = ta.momentum.RSIIndicator(close=adm_full['ADM'], window=14).rsi().shift(1)
merged_df = pd.merge(merged_df, rsi, left_index=True, right_index=True, how='inner')

# Define conditions
conditions = [
    merged_df['Climate_Stress_Index'] > 22,
    merged_df['Climate_Stress_Index'] < -22
]

# Define outputs
choices = [1, -1]

# Create the new column
merged_df['Climate_Signal'] = np.select(conditions, choices, default=0)

merged_df['interaction_term'] = merged_df['Climate_Stress_Index'] / merged_df['Drought_Severity']
merged_df['interaction_term'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

merged_df = merged_df.drop('Climate_Stress_Index', axis=1)
merged_df.index.name = 'Date'


merged_df.to_csv('final_merged_features.csv')

