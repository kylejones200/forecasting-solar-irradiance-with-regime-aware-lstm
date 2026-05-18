"""Compare vanilla LSTM vs regime-aware LSTM for solar GHI forecasting."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, features: list[str], seq_len: int = 30):
        self.X, self.y, self.r = [], [], []
        for i in range(len(frame) - seq_len):
            self.X.append(frame.iloc[i : i + seq_len][features].values)
            self.y.append(frame.iloc[i + seq_len]["GHI"])
            self.r.append(int(frame.iloc[i + seq_len]["Regime"]))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)
        self.r = torch.tensor(self.r, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.r[idx], self.y[idx]


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
        x_aug = torch.cat([x, regime_expanded], dim=2)
        out, _ = self.lstm(x_aug)
        return self.fc(out[:, -1, :])


def train_model(model, loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, r, y in loader:
            optimizer.zero_grad()
            pred = model(x, r)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")


def main() -> None:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    df = pd.DataFrame(
        {
            "GHI": 200
            + 80 * np.sin(2 * np.pi * dates.dayofyear / 365)
            + rng.normal(0, 10, len(dates)),
            "Temp": 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365),
            "Humidity": rng.uniform(30, 90, len(dates)),
            "Wind": rng.uniform(0, 15, len(dates)),
        },
        index=dates,
    )
    df["Regime"] = pd.cut(df.index.dayofyear, bins=4, labels=False).astype(int)
    features = ["GHI", "Temp", "Humidity", "Wind"]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    dataset = TimeSeriesDataset(df, features, seq_len=30)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    vanilla = VanillaLSTM()
    train_model(vanilla, loader, epochs=5)
    regime_model = RegimeLSTM()
    train_model(regime_model, loader, epochs=5)
    vanilla.eval()
    regime_model.eval()
    with torch.no_grad():
        preds_v, preds_r, actual = [], [], []
        for x, r, y in loader:
            preds_v.append(vanilla(x).numpy())
            preds_r.append(regime_model(x, r).numpy())
            actual.append(y.numpy())
    y_true = np.vstack(actual).ravel()
    rmse_v = np.sqrt(mean_squared_error(y_true, np.vstack(preds_v).ravel()))
    rmse_r = np.sqrt(mean_squared_error(y_true, np.vstack(preds_r).ravel()))
    print(f"Vanilla LSTM RMSE: {rmse_v:.4f}")
    print(f"Regime LSTM RMSE: {rmse_r:.4f}")
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:100], label="Actual", color="black")
    plt.plot(np.vstack(preds_v).ravel()[:100], label="Vanilla LSTM", alpha=0.8)
    plt.plot(np.vstack(preds_r).ravel()[:100], label="Regime LSTM", alpha=0.8)
    plt.legend()
    plt.title("Solar GHI: Vanilla vs Regime-Aware LSTM")
    plt.tight_layout()
    plt.savefig("lstm_regime_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
