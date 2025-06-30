# evaluate_lstm.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_FP    = "merged_data/merged_timeseries_full.csv"
MODEL_FP   = "models/lstm_best.pth"
WINDOW     = 10
BATCH_SIZE = 32
SPLIT_DATE = datetime(2025,6,1).date()

FEATURE_COLS = [
    "open","high","low","close","volume",
    "ma5","ma10","ma20","rsi14","vol_ma5",
    "news_count","avg_positive_score","avg_negative_score","avg_neutral_score",
    "total_price_up_signal","total_price_down_signal","total_price_neutral_signal"
]
TARGET_REG = "return1"

# ─── 데이터 로드 및 라벨 생성 ─────────────────────────────────────────────────
df = pd.read_csv(DATA_FP, parse_dates=["date"])
df["date"] = df["date"].dt.date
df["next_close"] = df.groupby("code")["close"].shift(-1)
df["return1"]    = (df["next_close"] - df["close"]) / df["close"]
df = df.dropna(subset=["return1"]).reset_index(drop=True)

# ─── 스케일러 학습 & 적용 ─────────────────────────────────────────────────────
scaler = StandardScaler()
train_mask = df["date"] < SPLIT_DATE
scaler.fit(df.loc[train_mask, FEATURE_COLS].values)
scaled_vals = scaler.transform(df[FEATURE_COLS].values)
scaled_vals = np.where(np.isfinite(scaled_vals), scaled_vals, np.nan)
scaled_vals = np.nan_to_num(scaled_vals, nan=0.0)
df[FEATURE_COLS] = scaled_vals

# ─── 시퀀스 데이터셋 정의 ─────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, df, feat_cols, target, window):
        X_list, y_list, d_list = [], [], []
        for code in df["code"].unique():
            sub = df[df["code"]==code].sort_values("date")
            vals   = sub[feat_cols].values.astype(np.float32)
            labels = sub[target].values.astype(np.float32)
            dates  = sub["date"].values
            for i in range(len(vals)-window):
                X_list.append(vals[i:i+window])
                y_list.append(labels[i+window])
                d_list.append(dates[i+window])
        self.X = torch.from_numpy(np.stack(X_list))
        self.y = torch.from_numpy(np.array(y_list))
        self.dates = np.array(d_list)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

full_ds = SequenceDataset(df, FEATURE_COLS, TARGET_REG, WINDOW)
idxs       = np.arange(len(full_ds))
train_idxs = idxs[ full_ds.dates < SPLIT_DATE ]
test_idxs  = idxs[ full_ds.dates >= SPLIT_DATE ]
test_ds    = Subset(full_ds, test_idxs)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ─── 모델 정의 & 로드 ──────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]
        return self.fc(last).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim=len(FEATURE_COLS)).to(device)
model.load_state_dict(torch.load(MODEL_FP, map_location=device))
model.eval()

# ─── 예측 & 평가 ───────────────────────────────────────────────────────────────
preds, trues = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        out = model(Xb).cpu().numpy()
        preds.append(out)
        trues.append(yb.numpy())
preds = np.concatenate(preds)
trues = np.concatenate(trues)

mse = mean_squared_error(trues, preds)
mae = mean_absolute_error(trues, preds)
r2  = r2_score(trues, preds)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R2: {r2:.3f}")

# ─── 시각화: 실제 vs 예측 ───────────────────────────────────────────────────────
plt.figure()
plt.scatter(trues, preds)
plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()])
plt.xlabel("Actual return1")
plt.ylabel("Predicted return1")
plt.title("Actual vs Predicted Returns")
plt.show()
