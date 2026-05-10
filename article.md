---
author: "Kyle Jones"
date_published: "July 15, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/forecasting-solar-irradiance-with-regime-aware-lstm-29830cbe220e"
---

# Forecasting Solar Irradiance with Regime-Aware LSTM Utilities and grid operators rely on accurate solar forecasts to manage
generation, maintain reliability, and avoid unnecessary...

### Forecasting Solar Irradiance with Regime-Aware LSTM 

Utilities and grid operators rely on accurate solar forecasts to manage generation, maintain reliability, and avoid unnecessary curtailment. As solar penetration increases, errors in irradiance forecasting can translate directly into balancing costs, reserve overcommitment, or missed financial targets.

This post shows how to improve solar irradiance forecasting using a regime-aware LSTM model trained on real SolarAnywhere® data. We compare this model to a vanilla LSTM and find that even a basic seasonal regime feature improves predictive accuracy.

### The Business Context
Short-term solar forecasting supports:

- **Grid planning** (when and where energy is available)
- **Reserve scheduling** (how much backup is needed)
- **Curtailment prevention** (avoiding wasted power)

Forecasting solar irradiance is difficult due to rapid weather changes, cloud formations, and seasonal patterns. Machine learning models often struggle with these nonlinear shifts.

To address this, we use regime-aware modeling. Regimes represent distinct operating conditions like sunny vs. cloudy seasons or periods of high vs. low irradiance variability. These regimes act as context features, allowing the model to condition its predictions on recent temporal structure.

### The Data
We use **SolarAnywhere Time Series** data for a location near Bellevue, WA. It includes hourly observations of Global Horizontal Irradiance (GHI), Ambient Temperature, Relative Humidity, and Wind Speed.

We resample the data to daily frequency and create four seasonal regimes by dividing the year into quarters. Each row contains the average GHI and weather conditions for a day, plus a regime label (0--3).

### Modeling Approach
We train two models:

1.  [**Vanilla LSTM** --- standard recurrent neural network using only time series features.]
2.  [**Regime-Aware LSTM** --- same model, but with a learned embedding for the regime label, which augments each input sequence.]

Each model learns to predict the next day's GHI based on the past 30 days of features. We compare mean squared error (MSE) after 10 epochs.

### Results
Model MSE

Vanilla LSTM 0.0372

Regime-Aware LSTM 0.0345 ✅

Adding a regime embedding improves MSE by \~7%, despite no additional data.

Let's plot the predictions from both models against the true GHI values:


Even simple regime features --- like seasonal groupings --- can boost the accuracy of deep learning models in energy forecasting. This pattern can generalize to other domains: power price prediction, wind energy forecasting, or even grid congestion modeling.

Future work could include:

- Using **Markov Switching models** to define latent regimes
- Incorporating **cloud cover forecasts** or **satellite imagery**
- Extending to **probabilistic LSTM** or **quantile regression**

### Full Code
```python
# 1. Load and preprocess SolarAnywhere data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Bellevue SolarAnywhere Time Series.csv", encoding="cp1252", header=1)
df["time"] = pd.to_datetime(df["ObservationTime(LST)"])
df = df.set_index("time")
df = df[[
    "Global Horizontal Irradiance (GHI) W/m2",
    "AmbientTemperature (deg C)",
    "Relative Humidity (%)",
    "Wind Speed (m/s)"
]]
df.columns = ["GHI", "Temp", "Humidity", "Wind"]
df = df.resample("D").mean().dropna()
df["Regime"] = pd.cut(df.index.dayofyear, bins=4, labels=False)
features = ["GHI", "Temp", "Humidity", "Wind"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# 2. Define dataset
import torch
from torch.utils.data import Dataset, DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=30):
        self.X, self.y, self.r = [], [], []
        for i in range(len(df) - seq_len):
            self.X.append(df.iloc[i:i+seq_len][features].values)
            self.y.append(df.iloc[i+seq_len]["GHI"])
            self.r.append(df.iloc[i+seq_len]["Regime"])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)
        self.r = torch.tensor(self.r, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.r[idx], self.y[idx]
dataset = TimeSeriesDataset(df)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define models
import torch.nn as nn
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x, *_):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
class RegimeLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_regimes=4):
        super().__init__()
        self.embedding = nn.Embedding(num_regimes, 4)
        self.lstm = nn.LSTM(input_dim + 4, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x, regime_id):
        regime_embed = self.embedding(regime_id)
        regime_expanded = regime_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x_augmented = torch.cat([x, regime_expanded], dim=2)
        out, _ = self.lstm(x_augmented)
        return self.fc(out[:, -1, :])

# 4. Train
def train_model(model, loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, r, y in loader:
            optimizer.zero_grad()
            pred = model(x, r)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")
vanilla = VanillaLSTM()
train_model(vanilla, train_loader)
regime_model = RegimeLSTM()
train_model(regime_model, train_loader)

# 5. Evaluate
from sklearn.metrics import mean_squared_error
vanilla.eval()
regime_model.eval()
y_true, y_vanilla, y_regime = [], [], []
with torch.no_grad():
    for x, r, y in train_loader:
        y_true.append(y)
        y_vanilla.append(vanilla(x))
        y_regime.append(regime_model(x, r))
y_true = torch.cat(y_true).numpy()
y_vanilla = torch.cat(y_vanilla).numpy()
y_regime = torch.cat(y_regime).numpy()
print(f"Vanilla LSTM MSE: {mean_squared_error(y_true, y_vanilla):.4f}")
print(f"Regime-Aware LSTM MSE: {mean_squared_error(y_true, y_regime):.4f}")

# 6. Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(y_true[:100], label="True", linewidth=2)
plt.plot(y_vanilla[:100], label="Vanilla LSTM", linestyle="--")
plt.plot(y_regime[:100], label="Regime-Aware LSTM", linestyle=":")
plt.title("Predicted vs. True GHI (First 100 Days)")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend()
plt.tight_layout()
plt.savefig("ghi_lstm_comparison.png")
plt.show()
```
