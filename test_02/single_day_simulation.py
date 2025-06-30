# single_day_simulation.py

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 설정
DATA_FP    = "merged_data/merged_timeseries_full.csv"
MODEL_FP   = "models/lstm_best.pth"
WINDOW     = 10
FEATURE_COLS = [
    "open","high","low","close","volume",
    "ma5","ma10","ma20","rsi14","vol_ma5",
    "news_count","avg_positive_score","avg_negative_score","avg_neutral_score",
    "total_price_up_signal","total_price_down_signal","total_price_neutral_signal"
]

# 1) 데이터 로드
df = pd.read_csv(DATA_FP, parse_dates=["date"])
df["date"] = df["date"].dt.date

# 오늘 & 어제 날짜 정의
today   = df["date"].max()                    # 예: 2025-06-17
# 어제 거래일 찾기 (단순히 -1일이 비거래일이면 전날로)
yesterday = today - timedelta(days=1)
while yesterday not in set(df["date"]):
    yesterday -= timedelta(days=1)

# 2) 라벨 생성 (실제 return 계산용)
# (이미 merged 파일에 next_close, return1 있을 수도 있지만, 안전하게 다시 계산)
sub_df = df.copy()
sub_df["next_close"]  = sub_df.groupby("code")["close"].shift(-1)
sub_df["return1"]     = (sub_df["next_close"] - sub_df["close"]) / sub_df["close"]

# 3) 스케일러 학습 & 적용
scaler = StandardScaler()
mask_train = sub_df["date"] <= yesterday   # 어제까지 데이터로 fit
scaler.fit(sub_df.loc[mask_train, FEATURE_COLS].values)

# transform 전체 (어제+오늘)
scaled = scaler.transform(sub_df[FEATURE_COLS].values)
# Inf/NaN 처리
scaled = np.nan_to_num(np.where(np.isfinite(scaled), scaled, np.nan), nan=0.0)
sub_df[FEATURE_COLS] = scaled

# 4) 모델 로드
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

# 5) “어제”까지의 마지막 10일치 시퀀스로 예측 & 실제 수익률 수집
pred_returns   = {}
actual_returns = {}
for code in sub_df["code"].unique():
    df_code = sub_df[sub_df["code"]==code].sort_values("date")
    # 어제까지 데이터만
    df_hist = df_code[df_code["date"] <= yesterday]
    if len(df_hist) < WINDOW:
        continue  # 데이터 부족 시 건너뛰기
    # 10일치 피처
    seq = df_hist.iloc[-WINDOW:][FEATURE_COLS].values.astype(np.float32)
    Xb  = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, W, D)
    with torch.no_grad():
        pred = model(Xb).item()
    # 실제 수익률
    close_y = df_hist.iloc[-1]["close"]
    # 오늘 종가
    close_t = df_code[df_code["date"]==today]["close"].values[0]
    actual = (close_t - close_y) / close_y

    pred_returns[code]   = pred
    actual_returns[code] = actual

# 6) 모의 포트폴리오 수익률 계산
n = len(pred_returns)  # 정상 처리된 종목 수 (<=150)
weight = 1.0 / n
# 예상 vs 실제 포트폴리오 수익률
port_pred   = sum(pred_returns[c] * weight for c in pred_returns)
port_actual = sum(actual_returns[c] * weight for c in actual_returns)

print(f"▶ 테스트 대상 종목 수: {n}")
print(f"▶ 예상 포트폴리오 수익률: {port_pred:.4%}")
print(f"▶ 실제 포트폴리오 수익률: {port_actual:.4%}")
