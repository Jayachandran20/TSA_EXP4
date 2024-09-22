### DEVELOPED BY: M.JAYACHANDRAN
### REGISTER NO: 212222240038
### DATE:


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# AIM:
To implement ARMA model in python.
# ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
# PROGRAM:
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


file_path = 'future_gold_price.csv'  
data = pd.read_csv(file_path)


data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data_values = data['Close'].dropna().values

# 1. ARMA(1,1) Process

ar1 = np.array([1, -0.5])  # AR(1) coefficient
ma1 = np.array([1, 0.5])   # MA(1) coefficient


arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=1000)

# Plot the ARMA(1,1) time series
plt.figure(figsize=(10, 6))
plt.plot(arma11_sample)
plt.title('Generated ARMA(1,1) Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(0, 1000)
plt.grid(True)
plt.show()

# Display ACF and PACF plots for ARMA(1,1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(arma11_sample, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(arma11_sample, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF for ARMA(1,1)')
plt.tight_layout()
plt.show()

# 2. Fit ARMA(2,2) model to 'Close' prices from dataset

# Define ARMA(2,2) model for actual data
model = ARIMA(data_values, order=(2, 0, 2))  # ARMA(2,2) is equivalent to ARIMA(2,0,2)
arma22_fit = model.fit()

# Print summary of the model
print(arma22_fit.summary())

# Plot residuals to check model fitting
residuals = arma22_fit.resid

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARMA(2,2) Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Plot ACF and PACF for residuals to assess white noise
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuals, lags=35, ax=plt.gca())
plt.subplot(122)
plot_pacf(residuals, lags=35, ax=plt.gca())
plt.suptitle('ACF and PACF of Residuals from ARMA(2,2) Model')
plt.tight_layout()
plt.show()


```


# OUTPUT:
## SIMULATED ARMA(1,1) PROCESS:
![Screenshot 2024-09-22 182307](https://github.com/user-attachments/assets/4191ef7e-c74f-492e-b860-8495c1d481fa)


## Partial Autocorrelation
![Screenshot 2024-09-22 182558](https://github.com/user-attachments/assets/f5fac65a-831c-4de7-b7e5-4821c7e873a7)


## Autocorrelation

![Screenshot 2024-09-22 182344](https://github.com/user-attachments/assets/2dc83867-efb4-47fc-b452-0f46c13700e7)


## SIMULATED ARMA(2,2) PROCESS:
![Screenshot 2024-09-22 183034](https://github.com/user-attachments/assets/82f4a7ea-8374-4a34-83a8-31e24e7d8183)


## Partial Autocorrelation

![Screenshot 2024-09-22 183109](https://github.com/user-attachments/assets/698ecbec-9af4-463d-9357-c3e28d60f61a)


## Autocorrelation
![Screenshot 2024-09-22 183050](https://github.com/user-attachments/assets/89980ab4-de4b-4e86-9ba1-445a430807be)



# RESULT:
Thus, a python program is created to fit ARMA Model for Time Series successfully.
