# imports: panda for dataframe, prophet for time series model, sklearn for model evaluation metrics
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# read the CSV file - this file has been formatted to be fed directly into Prophet
df = pd.read_csv('../data/Sydney-Auckland_Prophet_Ready - Sheet1.csv')

print(df.head())

# rename columns to ds and y as expected by Prophet
df.rename(columns={'Month': 'ds', 'Passengers_Total': 'y'}, inplace=True)

# fit the model and add extra seasonality and holiday parameters
m = Prophet(
    seasonality_prior_scale=0.35,
)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
m.add_country_holidays(country_name='Australia')
m.fit(df)

# make predictions for the next 12 months
future = m.make_future_dataframe(periods=12, freq='M')
print(future.tail())

# predict and save predictions
forecast = m.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
forecast.to_csv('forecast_results.csv', index=False)

# calculate mean absolute error to evaluate model accuracy
mae = mean_absolute_error(df['y'], forecast['yhat'][:len(df)])
print(f"Mean Absolute Error: {mae}")

# plot the results and save plots
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

fig1.savefig('prophet_forecast.png')
fig2.savefig('prophet_components.png')
