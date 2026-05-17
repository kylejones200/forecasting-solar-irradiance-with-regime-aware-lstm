
def main() -> None:
    # --- notebook cell (unparsed) ---
    # """Generated from Jupyter notebook: 2025-07-15 LSTM v regime aware LSTM-1

    # Magics and shell lines are commented out. Run with a normal Python interpreter."""


    # # --- code cell ---

    # from sklearn.metrics import mean_squared_errorfrom sklearn.preprocessing import MinMaxScalerfrom torch.utils.data import Dataset, DataLoaderimport matplotlib.pyplot as pltimport pandas as pdimport torchimport torch.nn as nnfile_path = "Bellevue SolarAnywhere Time Series 20230101 to 20240101 Lat_47_615 Lon_ - 122_175 SA format.csv"# Load SolarAnywhere CSV (Hourly)df = pd.read_csv(file_path, encoding="cp1252", header = 1)# Parse datetime and set indexdf["time"] = pd.to_datetime(df["ObservationTime(LST)"])df = df.set_index("time")# Select and rename relevant columnsdf = df[["Global Horizontal Irradiance (GHI) W / m2","AmbientTemperature (deg C)","Relative Humidity (%)","Wind Speed (m / s)",]]df.columns = ["GHI", "Temp", "Humidity", "Wind"]# Resample to daily mean valuesdf = df.resample("D").mean().dropna()# Add regime placeholder (seasonal quarter bins)df["Regime"] = pd.cut(df.index.dayofyear, bins = 4, labels = False)# Normalize the featuresfeatures = ["GHI", "Temp", "Humidity", "Wind"]scaler = MinMaxScaler()df[features] = scaler.fit_transform(df[features])


    # # --- code cell ---

    # features = ["GHI", "Temp", "Humidity", "Wind"]class TimeSeriesDataset(Dataset):def __init__(self, df, seq_len = 30):self.X, self.y, self.r = [], [], []for i in range(len(df) - seq_len):self.X.append(df.iloc[i : i + seq_len][features].values)self.y.append(df.iloc[i + seq_len]["GHI"])self.r.append(df.iloc[i + seq_len]["Regime"])self.X = torch.tensor(self.X, dtype = torch.float32)self.y = torch.tensor(self.y, dtype = torch.float32).unsqueeze(1)self.r = torch.tensor(self.r, dtype = torch.long)def __len__(self):return len(self.X)def __getitem__(self, idx):return self.X[idx], self.r[idx], self.y[idx]dataset = TimeSeriesDataset(df, seq_len = 30)train_loader = DataLoader(dataset, batch_size = 32, shuffle = True)class VanillaLSTM(nn.Module):def __init__(self, input_dim = 4, hidden_dim = 64, num_layers = 1):super().__init__()self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)self.fc = nn.Linear(hidden_dim, 1)def forward(self, x, *_):out, _ = self.lstm(x)return self.fc(out[:, -1, :])class RegimeLSTM(nn.Module):def __init__(self, input_dim = 4, hidden_dim = 64, num_layers = 1, num_regimes = 4):super().__init__()self.embedding = nn.Embedding(num_regimes, 4)self.lstm = nn.LSTM(input_dim + 4, hidden_dim, num_layers, batch_first = True)self.fc = nn.Linear(hidden_dim, 1)def forward(self, x, regime_id):regime_embed = self.embedding(regime_id)regime_expanded = regime_embed.unsqueeze(1).expand(-1, x.size(1), -1)x_augmented = torch.cat([x, regime_expanded], dim = 2)out, _ = self.lstm(x_augmented)return self.fc(out[:, -1, :])def train_model(model, loader, epochs = 10):optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)loss_fn = nn.MSELoss()model.train()for epoch in range(epochs):total_loss = 0for x, r, y in loader:optimizer.zero_grad()pred = model(x, r)loss = loss_fn(pred, y)loss.backward()optimizer.step()total_loss += loss.item()print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")# Train vanilla LSTMvanilla = VanillaLSTM()train_model(vanilla, train_loader, epochs = 10)# Train regime - aware LSTMregime_model = RegimeLSTM()train_model(regime_model, train_loader, epochs = 10)# Evaluate bothvanilla.eval()regime_model.eval()y_true, y_vanilla, y_regime = [], [], []with torch.no_grad():for x, r, y in train_loader:y_true.append(y)y_vanilla.append(vanilla(x))y_regime.append(regime_model(x, r))# Concatenate and calculate MSEy_true = torch.cat(y_true).numpy()y_vanilla = torch.cat(y_vanilla).numpy()y_regime = torch.cat(y_regime).numpy()print(f"\nVanilla LSTM MSE: {mean_squared_error(y_true, y_vanilla):.4f}")print(f"Regime - Aware LSTM MSE: {mean_squared_error(y_true, y_regime):.4f}")


    # # --- code cell ---

    # # Plot first 100 predictionsplt.figure(figsize=(12, 4))plt.plot(y_true[:100], label="True", linewidth = 2)plt.plot(y_vanilla[:100], label="Vanilla LSTM", linestyle="--")plt.plot(y_regime[:100], label="Regime - Aware LSTM", linestyle=":")plt.title("Predicted vs. True GHI (Normalized) (First 100 Days)")plt.legend()# Minimalist Tufte - style axesax = plt.gca()ax.spines["top"].set_visible(False)ax.spines["right"].set_visible(False)plt.tight_layout()plt.savefig("ghi_lstm_comparison.png")plt.show()

if __name__ == "__main__":
    main()
