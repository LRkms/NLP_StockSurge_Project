# train_lstm.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_FP    = "merged_data/merged_timeseries_full.csv"
WINDOW     = 10
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
SPLIT_DATE = datetime(2025,6,1).date()   # 시퀀스의 target 날짜 기준 분할

FEATURE_COLS = [
    "open","high","low","close","volume",
    "ma5","ma10","ma20","rsi14","vol_ma5",
    "news_count","avg_positive_score","avg_negative_score","avg_neutral_score",
    "total_price_up_signal","total_price_down_signal","total_price_neutral_signal"
]
TARGET_REG = "return1"   # 회귀용 라벨

# ─── 0) 데이터 로드 & 라벨 생성 ─────────────────────────────────────────────────
df = pd.read_csv(DATA_FP, parse_dates=["date"])
df["date"] = df["date"].dt.date

# 다음날 종가 & 수익률
df["next_close"] = df.groupby("code")["close"].shift(-1)
df["return1"]    = (df["next_close"] - df["close"]) / df["close"]
# 결측 제거
df = df.dropna(subset=["return1"]).reset_index(drop=True)

# ─── 1) 스케일러 학습 & 적용 ────────────────────────────────────────────────────
scaler = StandardScaler()
train_mask = df["date"] < SPLIT_DATE
scaler.fit(df.loc[train_mask, FEATURE_COLS].values)

# transform 전체
scaled_vals = scaler.transform(df[FEATURE_COLS].values)

# Inf → NaN → 0
scaled_vals = np.where(np.isfinite(scaled_vals), scaled_vals, np.nan)
scaled_vals = np.nan_to_num(scaled_vals, nan=0.0)

# DataFrame에 다시 대입
df[FEATURE_COLS] = scaled_vals


# ─── 2) 전체 시퀀스 생성 ────────────────────────────────────────────────────────
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
                d_list.append(dates[i+window])  # 이 시퀀스의 target 날짜
        self.X = torch.from_numpy(np.stack(X_list))  # (N, W, D)
        self.y = torch.from_numpy(np.array(y_list))
        self.dates = np.array(d_list)                # numpy array of dates

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

full_ds = SequenceDataset(df, FEATURE_COLS, TARGET_REG, WINDOW)
print(f"▶ 전체 시퀀스 수: {len(full_ds)}")

# ─── 3) train/test 인덱스 분리 & Subset 생성 ────────────────────────────────────
idxs = np.arange(len(full_ds))
train_idxs = idxs[ full_ds.dates < SPLIT_DATE ]
test_idxs  = idxs[ full_ds.dates >= SPLIT_DATE ]

train_ds = Subset(full_ds, train_idxs)
test_ds  = Subset(full_ds, test_idxs)
print(f"▶ train 시퀀스: {len(train_ds)}, test 시퀀스: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ─── 4) LSTM 모델 정의 ───────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)            # (B, W, hidden_dim)
        last   = out[:, -1, :]           # (B, hidden_dim)
        return self.fc(last).squeeze()   # (B,)

# ─── 5) 학습 루프 ───────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = LSTMModel(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += criterion(model(Xb), yb).item() * Xb.size(0)
        val_loss = val_loss / len(test_ds)

        print(f"Epoch {epoch:02d}/{EPOCHS}  ▶  train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_baseline.pth")
    print("✅ 모델 저장: models/lstm_baseline.pth")

if __name__ == "__main__":
    train()
