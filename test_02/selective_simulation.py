import pandas as pd
import numpy as np
from datetime import timedelta
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_FP      = "merged_data/merged_timeseries_full.csv"
MODEL_FP     = "models/lstm_best.pth"
WINDOW       = 10
FEATURE_COLS = [
    "open","high","low","close","volume",
    "ma5","ma10","ma20","rsi14","vol_ma5",
    "news_count","avg_positive_score","avg_negative_score","avg_neutral_score",
    "total_price_up_signal","total_price_down_signal","total_price_neutral_signal"
]

# ─── 1) 데이터 로드 & 날짜 설정 ─────────────────────────────────────────────────
df = pd.read_csv(DATA_FP, parse_dates=["date"])
df["date"] = df["date"].dt.date
today     = df["date"].max()
yesterday = today - timedelta(days=1)
while yesterday not in df["date"].unique():
    yesterday -= timedelta(days=1)

# ─── 2) 스케일러 학습 & 적용 ─────────────────────────────────────────────────────
scaler = StandardScaler()
train_mask = df["date"] <= yesterday
scaler.fit(df.loc[train_mask, FEATURE_COLS].values)
scaled = scaler.transform(df[FEATURE_COLS].values)
scaled = np.nan_to_num(np.where(np.isfinite(scaled), scaled, np.nan), nan=0.0)
df[FEATURE_COLS] = scaled

# ─── 3) 모델 정의 & 로드 ─────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim=len(FEATURE_COLS)).to(device)
model.load_state_dict(torch.load(MODEL_FP, map_location=device))
model.eval()

# ─── 4) 예측 수익률(pred_returns) 계산 ───────────────────────────────────────────
pred_returns = {}
for code, grp in df[df["date"] <= yesterday].groupby("code"):
    grp = grp.sort_values("date")
    if len(grp) >= WINDOW:
        seq = grp.iloc[-WINDOW:][FEATURE_COLS].values.astype(np.float32)
        Xb  = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, W, D)
        with torch.no_grad():
            pred = model(Xb).item()
        pred_returns[code] = pred

# ─── 5) “오를 것” 예측 종목만 골라서 예상 포트폴리오 수익률 계산 ────────────────
selected = [c for c, p in pred_returns.items() if p > 0]
n_sel    = len(selected)

if n_sel == 0:
    print("▶ 예측 수익률이 양수인 종목이 없습니다.")
else:
    weight    = 1.0 / n_sel
    port_pred = sum(pred_returns[c] * weight for c in selected)
    print(f"▶ 선택 종목 수: {n_sel}")
    print(f"▶ 예상 포트폴리오 수익률: {port_pred:.2%}")
