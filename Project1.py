import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SYMBOL="GOOGL"
START_DATE="2022-01-01"
END_DATE="2026-02-21"


df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
df = df.reset_index()
df.columns = df.columns.get_level_values(0).str.lower()

df["return"]= df["close"].pct_change()
df["volatility"]=df["return"].rolling(5).std()
df["ma_3"]= df["close"].rolling(3).mean()
df["ma_5"]= df["close"].rolling(5).mean()
df["ma_10"]= df["close"].rolling(10).mean()

df["target_return"]=df["return"].shift(-1)
df= df.dropna()

features= ["close","volume","volatility","ma_3","ma_5","ma_10"]
X= df[features]
Y= df["target_return"]

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, shuffle=False)

model= RandomForestRegressor(
    n_estimators=300,
    max_depth= 10,
    random_state= 42
)

model.fit(X_train, Y_train)

pred_returns= model.predict(X_test)
all_tree_preds= np.stack([tree.predict(X_test) for tree in model.estimators_], axis=0)
pred_std= all_tree_preds.std(axis=0)
lower_bound= pred_returns- 2*pred_std
upper_bound= pred_returns + 2*pred_std

pred_prices= X_test["close"].values*(1+ pred_returns)
lower_prices= X_test["close"].values*(1+ lower_bound)
upper_prices = X_test["close"].values * (1 + upper_bound)
y_true_prices= X_test["close"].values*(1+Y_test.values)

mae= mean_absolute_error(y_true_prices,pred_prices)
rmse= np.sqrt(mean_squared_error(y_true_prices,pred_prices))
r2= r2_score(y_true_prices,pred_prices)

print(f"MAE: ${mae: 2f}")
print(f"RMSE: ${rmse: 2f}")
print(f"R^2: {r2:3f}")

for i in range(5):
    print(f"Predicted: ${pred_prices[i]:.2f}, 95% CI: ${lower_prices[i]:.2f}- ${upper_prices[i]:.2f}")

plt.style.use('ggplot')
plt.figure(figsize=(12,6))
plt.plot(y_true_prices, label="Actual Price", color= "#1f77b4", linewidth=2)
plt.plot(pred_prices,label="Predicted Price",color="#ff7f0e",linewidth=2)
plt.fill_between(range(len(pred_prices)), lower_prices, upper_prices, color='orange', alpha=0.5)
plt.title("GOOGL price prediction( Random Forest ML)", fontsize=16)

plt.xlabel("Days", fontsize=12)
plt.ylabel("Price(USD)",fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

latest_idx= df.index[-1]
latest_features= pd.DataFrame({
    "close": [df.loc[latest_idx, "close"]],
    "volume":[df.loc[latest_idx,"volume"]],
    "volatility":[df["return"].iloc[-5:].std()],
    "ma_3":[df["close"].iloc[-3:].mean()],
    "ma_5":[df["close"].iloc[-5:].mean()],
    "ma_10": [df["close"].iloc[-10:].mean()]
})

for col in latest_features.columns:
    latest_features[col]= latest_features[col].clip(
        lower= X_train[col].min(),
        upper= X_train[col].max()
    )

pred_return_next= model.predict(latest_features)[0]
next_day_price= latest_features["close"].values[0]*(1+ pred_return_next)
print(f"Next Day predicted close for GOOGL: ${next_day_price:.2f}")
