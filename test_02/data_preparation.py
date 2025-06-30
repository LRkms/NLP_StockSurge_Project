# data_preparation.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

# --- 설정 ---
DATA_FP    = "merged_data/merged_timeseries_full.csv"
WINDOW     = 10
BATCH_SIZE = 32

FEATURE_COLS = [
    "open","high","low","close","volume",
    "ma5","ma10","ma20","rsi14","vol_ma5",
    "news_count","avg_positive_score","avg_negative_score","avg_neutral_score",
    "total_price_up_signal","total_price_down_signal","total_price_neutral_signal"
]
TARGET_REG = "return1"
TARGET_CLF = "label_dir"

# --- Dataset ---
class StockDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window=WINDOW):
        X_list, y_list = [], []
        for code in df["code"].unique():
            sub    = df[df["code"]==code].sort_values("date")
            vals   = sub[feature_cols].values.astype(np.float32)
            labels = sub[target_col].values
            if target_col == TARGET_CLF:
                labels = labels.astype(np.int64)
            for i in range(len(vals) - window):
                X_list.append(vals[i:i+window])
                y_list.append(labels[i+window])
        X_arr = np.stack(X_list)
        y_arr = np.array(y_list)
        self.X = torch.from_numpy(X_arr)
        if target_col == TARGET_REG:
            self.y = torch.from_numpy(y_arr.astype(np.float32))
        else:
            self.y = torch.from_numpy(y_arr.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    # 1) 로드
    df = pd.read_csv(DATA_FP, parse_dates=["date"])

    # 2) 라벨 생성
    df["next_close"] = df.groupby("code")["close"].shift(-1)
    df["return1"]    = (df["next_close"] - df["close"]) / df["close"]
    up, down = 0.005, -0.005
    df["label_dir"] = df["return1"].apply(
        lambda x: 2 if x>up else (0 if x<down else 1) if pd.notna(x) else np.nan
    )
    df = df.dropna(subset=["return1","label_dir"]).reset_index(drop=True)

    # 3) Train/Test 날짜 기준 split (데이터 범위 내에서)
    unique_dates = sorted(df["date"].dt.date.unique())
    split_idx    = int(len(unique_dates)*0.8)
    split_date   = unique_dates[split_idx]
    print("▶ split_date:", split_date)
    train_df = df[df["date"].dt.date < split_date].copy()
    test_df  = df[df["date"].dt.date >= split_date].copy()

    # 4) 스케일링
    scaler     = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)
    train_df[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS].values)
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS].values)

    # 5) Dataset & DataLoader
    train_reg_ds = StockDataset(train_df, FEATURE_COLS, TARGET_REG)
    test_reg_ds  = StockDataset(test_df,  FEATURE_COLS, TARGET_REG)
    train_reg_loader = DataLoader(train_reg_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_reg_loader  = DataLoader(test_reg_ds,  batch_size=BATCH_SIZE, shuffle=False)

    train_clf_ds = StockDataset(train_df, FEATURE_COLS, TARGET_CLF)
    test_clf_ds  = StockDataset(test_df,  FEATURE_COLS, TARGET_CLF)
    train_clf_loader = DataLoader(train_clf_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_clf_loader  = DataLoader(test_clf_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # 6) 결과 확인
    print(f"▶ Train reg batches: {len(train_reg_loader)}, test reg: {len(test_reg_loader)}")
    print(f"▶ Train clf batches: {len(train_clf_loader)}, test clf: {len(test_clf_loader)}")

if __name__ == "__main__":
    main()
