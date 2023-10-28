import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import math

# Have pandas read in the .csv file
# Use path location on computer to read it in
data = pd.read_csv("/Users/richardzhang/Desktop/Bitcoin S2F Data.csv")

# Convert the "Date" column values into datetime objects
# This way, the dates can be neatly ordered on the x-axis when we graph the data
data["Date"] = pd.to_datetime(data.Date)

# Setting the index as the dates
data.set_index("Date", inplace=True)

# Taking the log of all of the available prices before 1st halving
log_price1 = pd.DataFrame(np.log(data["PriceUSD"].loc["2010-07-18":"2011-12-31"]))

# Taking the log of all of the available prices before 2nd halving
log_price2 = pd.DataFrame(np.log(data["PriceUSD"].loc["2010-07-18":"2015-12-31"]))

# Taking the log of all of the available prices before 3rd halving
log_price3 = pd.DataFrame(np.log(data["PriceUSD"].loc["2010-07-18":"2019-12-31"]))

# 30 day S2F ratios
ratio_30_day = pd.DataFrame(data["S2F 30 Day"].loc["2010-07-18":])

# 90 day S2F ratios
ratio_90_day = pd.DataFrame(data["S2F 90 Day"].loc["2010-07-18":])

# 180 day S2F ratios
ratio_180_day = pd.DataFrame(data["S2F 180 Day"].loc["2010-07-18":])

# 365 day S2F ratios
ratio_365_day = pd.DataFrame(data["S2F 365 Day"].loc["2010-07-18":])

#------------------------------------------------------------------------

log_ratio_30_day_1 = pd.DataFrame(np.log(data["S2F 30 Day"].loc["2010-07-18":"2011-12-31"]))
log_ratio_90_day_1 = pd.DataFrame(np.log(data["S2F 90 Day"].loc["2010-07-18":"2011-12-31"]))
log_ratio_180_day_1 = pd.DataFrame(np.log(data["S2F 180 Day"].loc["2010-07-18":"2011-12-31"]))
log_ratio_365_day_1 = pd.DataFrame(np.log(data["S2F 365 Day"].loc["2010-07-18":"2011-12-31"]))

log_ratio_30_day_2 = pd.DataFrame(np.log(data["S2F 30 Day"].loc["2010-07-18":"2015-12-31"]))
log_ratio_90_day_2 = pd.DataFrame(np.log(data["S2F 90 Day"].loc["2010-07-18":"2015-12-31"]))
log_ratio_180_day_2 = pd.DataFrame(np.log(data["S2F 180 Day"].loc["2010-07-18":"2015-12-31"]))
log_ratio_365_day_2 = pd.DataFrame(np.log(data["S2F 365 Day"].loc["2010-07-18":"2015-12-31"]))

log_ratio_30_day_3 = pd.DataFrame(np.log(data["S2F 30 Day"].loc["2010-07-18":"2019-12-31"]))
log_ratio_90_day_3 = pd.DataFrame(np.log(data["S2F 90 Day"].loc["2010-07-18":"2019-12-31"]))
log_ratio_180_day_3 = pd.DataFrame(np.log(data["S2F 180 Day"].loc["2010-07-18":"2019-12-31"]))
log_ratio_365_day_3 = pd.DataFrame(np.log(data["S2F 365 Day"].loc["2010-07-18":"2019-12-31"]))

model_1_1 = LinearRegression()
model_1_1.fit(log_ratio_30_day_1, log_price1)
model_1_2 = LinearRegression()
model_1_2.fit(log_ratio_90_day_1, log_price1)
model_1_3 = LinearRegression()
model_1_3.fit(log_ratio_180_day_1, log_price1)
model_1_4 = LinearRegression()
model_1_4.fit(log_ratio_365_day_1, log_price1)

print("Model 1_1 b_0 = ", model_1_1.intercept_)
print("Model 1_1 b_1 = ", model_1_1.coef_)
print("Model 1_2 b_0 = ", model_1_2.intercept_)
print("Model 1_2 b_1 = ", model_1_2.coef_)
print("Model 1_3 b_0 = ", model_1_3.intercept_)
print("Model 1_3 b_1 = ", model_1_3.coef_)
print("Model 1_4 b_0 = ", model_1_4.intercept_)
print("Model 1_4 b_1 = ", model_1_4.coef_)

model_2_1 = LinearRegression()
model_2_1.fit(log_ratio_30_day_2, log_price2)
model_2_2 = LinearRegression()
model_2_2.fit(log_ratio_90_day_2, log_price2)
model_2_3 = LinearRegression()
model_2_3.fit(log_ratio_180_day_2, log_price2)
model_2_4 = LinearRegression()
model_2_4.fit(log_ratio_365_day_2, log_price2)

print("Model 2_1 b_0 = ", model_2_1.intercept_)
print("Model 2_1 b_1 = ", model_2_1.coef_)
print("Model 2_2 b_0 = ", model_2_2.intercept_)
print("Model 2_2 b_1 = ", model_2_2.coef_)
print("Model 2_3 b_0 = ", model_2_3.intercept_)
print("Model 2_3 b_1 = ", model_2_3.coef_)
print("Model 2_4 b_0 = ", model_2_4.intercept_)
print("Model 2_4 b_1 = ", model_2_4.coef_)

model_3_1 = LinearRegression()
model_3_1.fit(log_ratio_30_day_3, log_price3)
model_3_2 = LinearRegression()
model_3_2.fit(log_ratio_90_day_3, log_price3)
model_3_3 = LinearRegression()
model_3_3.fit(log_ratio_180_day_3, log_price3)
model_3_4 = LinearRegression()
model_3_4.fit(log_ratio_365_day_3, log_price3)

print("Model 3_1 b_0 = ", model_3_1.intercept_)
print("Model 3_1 b_1 = ", model_3_1.coef_)
print("Model 3_2 b_0 = ", model_3_2.intercept_)
print("Model 3_2 b_1 = ", model_3_2.coef_)
print("Model 3_3 b_0 = ", model_3_3.intercept_)
print("Model 3_3 b_1 = ", model_3_3.coef_)
print("Model 3_4 b_0 = ", model_3_4.intercept_)
print("Model 3_4 b_1 = ", model_3_4.coef_)

# plt.plot(math.exp(model_1_1.intercept_) * ratio_30_day**model_1_1.coef_, label="Pre-1st Halving 30 Day Price")
# plt.plot(math.exp(model_2_1.intercept_) * ratio_30_day**model_2_1.coef_, label="Pre-2nd Halving 30 Day Price")
# plt.plot(math.exp(model_3_1.intercept_) * ratio_30_day**model_3_1.coef_, label="Pre-3rd Halving 30 Day Price")

# plt.plot(math.exp(model_1_2.intercept_) * ratio_90_day**model_1_2.coef_, label="Pre-1st Halving 90 Day Price")
# plt.plot(math.exp(model_2_2.intercept_) * ratio_90_day**model_2_2.coef_, label="Pre-2nd Halving 90 Day Price")
# plt.plot(math.exp(model_3_2.intercept_) * ratio_90_day**model_3_2.coef_, label="Pre-3rd Halving 90 Day Price")

# plt.plot(math.exp(model_1_3.intercept_) * ratio_180_day**model_1_3.coef_, label="Pre-1st Halving 180 Day Price")
# plt.plot(math.exp(model_2_3.intercept_) * ratio_180_day**model_2_3.coef_, label="Pre-2nd Halving 180 Day Price")
# plt.plot(math.exp(model_3_3.intercept_) * ratio_180_day**model_3_3.coef_, label="Pre-3rd Halving 180 Day Price")

# plt.plot(math.exp(model_1_4.intercept_) * ratio_365_day**model_1_4.coef_, label="Pre-1st Halving 365 Day Price")
# plt.plot(math.exp(model_2_4.intercept_) * ratio_365_day**model_2_4.coef_, label="Pre-2nd Halving 365 Day Price")
# plt.plot(math.exp(model_3_4.intercept_) * ratio_365_day**model_3_4.coef_, label="Pre-3rd Halving 365 Day Price")

log_price4 = pd.DataFrame(np.log(data["PriceUSD"].loc["2010-07-18":"2021-09-17"]))
model_4 = LinearRegression()
log_ratio_4 = pd.DataFrame(np.log(data["S2F 90 Day"].loc["2010-07-18":"2021-09-17"]))
model_4.fit(log_ratio_4, log_price4)
print("Model 4 (all data) b_0 = ", model_4.intercept_)
print("Model 4 (all data) b_1 = ", model_4.coef_)
plt.plot(math.exp(model_4.intercept_) * ratio_30_day**model_4.coef_, label="All Data 30 Day Price")
plt.plot(math.exp(model_4.intercept_) * ratio_90_day**model_4.coef_, label="All Data 90 Day Price")
plt.plot(math.exp(model_4.intercept_) * ratio_180_day**model_4.coef_, label="All Data 180 Day Price")
plt.plot(math.exp(model_4.intercept_) * ratio_365_day**model_4.coef_, label="All Data 365 Day Price")

# Model 1_1 b_0 =  [-1.15222679]
# Model 1_1 b_1 =  [[3.22325073]]
# Model 1_2 b_0 =  [-1.26130418]
# Model 1_2 b_1 =  [[3.66733313]]
# Model 1_3 b_0 =  [-1.44959138]
# Model 1_3 b_1 =  [[4.46339145]]
# Model 1_4 b_0 =  [-2.10213828]
# Model 1_4 b_1 =  [[5.87623583]]
# Model 2_1 b_0 =  [-1.18854364]
# Model 2_1 b_1 =  [[3.14959001]]
# Model 2_2 b_0 =  [-1.21448887]
# Model 2_2 b_1 =  [[3.19939776]]
# Model 2_3 b_0 =  [-1.21405236]
# Model 2_3 b_1 =  [[3.24830617]]
# Model 2_4 b_0 =  [-1.29200546]
# Model 2_4 b_1 =  [[3.38703266]]
# Model 3_1 b_0 =  [-1.00970001]
# Model 3_1 b_1 =  [[2.98805136]]
# Model 3_2 b_0 =  [-1.01385564]
# Model 3_2 b_1 =  [[3.01286129]]
# Model 3_3 b_0 =  [-0.99684947]
# Model 3_3 b_1 =  [[3.03779607]]
# Model 3_4 b_0 =  [-1.01000443]
# Model 3_4 b_1 =  [[3.10453477]]
# Model 4 (all data) b_0 =  [-0.91766008]
# Model 4 (all data) b_1 =  [[3.04435176]]

#------------------------------------------------------------------------

# 2010-07-18 is the earliest day with BTC price data
# Plot the data
plt.plot(data["PriceUSD"], label="Real Price")
plt.title("Bitcoin Price vs. Time Graph")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.yscale("log")
legend1 = plt.legend(loc="upper left", frameon=False, fontsize=9)
horz0 = plt.axhline(y=10, linestyle=":", color = "black", alpha=0.7)
horz1 = plt.axhline(y=100, linestyle=":", color = "gray", alpha=0.7)
horz2 = plt.axhline(y=300, linestyle=":", color = "red", alpha=0.7)
horz3 = plt.axhline(y=500, linestyle=":", color = "blue", alpha=0.7)
horz4 = plt.axhline(y=3000, linestyle=":", color = "green", alpha=0.7)
horz5 = plt.axhline(y=5000, linestyle=":", color = "purple", alpha=0.7)
horz6 = plt.axhline(y=30000, linestyle=":", color = "orange", alpha=0.7)
horz7 = plt.axhline(y=50000, linestyle=":", color = "navy", alpha=0.7)
horz8 = plt.axhline(y=300000, linestyle=":", color = "gold", alpha=0.7)
horz9 = plt.axhline(y=500000, linestyle=":", color = "magenta", alpha=0.7)
plt.legend([horz9, horz8, horz7, horz6, horz5, horz4, horz3, horz2, horz1, horz0], ["$500,000", "$300,000", "$50,000", "$30,000", "$5,000", "$3,000", "$500", "$300", "$100", "$10"], loc ="lower right", fontsize = 9)
plt.gca().add_artist(legend1)
plt.show()
