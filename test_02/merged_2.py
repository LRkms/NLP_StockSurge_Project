import os
import pandas as pd
import yfinance as yf
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    FDR_AVAILABLE = False
import pandas_ta as ta
from datetime import datetime

# 1) 경로 설정
merged_dir     = "merged_data"
stock_list_fp  = os.path.join(merged_dir, "stock_list.csv")
news_feat_fp   = os.path.join(merged_dir, "daily_news_features.csv")
output_fp      = os.path.join(merged_dir, "merged_timeseries_full.csv")

# 2) 종목 리스트 로드
df_stock = pd.read_csv(stock_list_fp, dtype=str)
# 편의용 dict
stock_info = df_stock.set_index("code")[["name","market","sector"]].to_dict(orient="index")

# 3) 뉴스 feature 로드
df_news = pd.read_csv(news_feat_fp, parse_dates=["date"])
# 없으면 빈 DF
if df_news.empty:
    df_news = pd.DataFrame(columns=["date","code","news_count","avg_positive_score", 
                                    "avg_negative_score","avg_neutral_score",
                                    "total_price_up_signal","total_price_down_signal",
                                    "total_price_neutral_signal"])

# 4) 주가 데이터 수집 함수
def get_price_data(code, start_date, end_date):
    market = stock_info[code]["market"]
    suffix = ".KS" if market=="KOSPI" else ".KQ"
    ticker = yf.Ticker(code+suffix)
    df = ticker.history(start=start_date, end=end_date)
    if df.empty and FDR_AVAILABLE:
        df = fdr.DataReader(code, start_date, end_date)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Date":"date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # 컬럼명 소문자 통일
    df.columns = [c.lower().replace(" ","_") for c in df.columns]
    # 필수컬럼 체크
    return df[["date","open","high","low","close","volume"]].assign(code=code)

# 5) 기술적 지표 계산 함수
def calculate_technical_indicators(df):
    df = df.sort_values("date").set_index("date")
    # 예: 5,10,20일 이동평균
    df["ma5"]  = ta.sma(df["close"], length=5)
    df["ma10"] = ta.sma(df["close"], length=10)
    df["ma20"] = ta.sma(df["close"], length=20)
    # 예: RSI(14)
    df["rsi14"] = ta.rsi(df["close"], length=14)
    # 예: 거래량 이동평균
    df["vol_ma5"] = ta.sma(df["volume"], length=5)
    df = df.reset_index()
    df["code"] = df.get("code", df_stock["code"])
    return df

# 6) merge_data 실행
all_merged = []
# 분석 기간: 뉴스 feature 범위 기준으로 잡거나 직접 지정
start_date = df_news["date"].min().strftime("%Y-%m-%d")
end_date   = df_news["date"].max().strftime("%Y-%m-%d")

for code in stock_info:
    print(f"▶ 처리중: {code} {stock_info[code]['name']}")
    price_df = get_price_data(code, start_date, end_date)
    if price_df.empty:
        print(f"   주가 데이터 없음: {code}")
        continue
    price_df = calculate_technical_indicators(price_df)
    # 뉴스 feature merge
    news_df = df_news[df_news["code"]==code]
    merged = pd.merge(price_df, news_df, on=["date","code"], how="left")
    # 뉴스가 없는 날은 0으로 채우기
    news_cols = ["news_count","avg_positive_score","avg_negative_score","avg_neutral_score",
                 "total_price_up_signal","total_price_down_signal","total_price_neutral_signal"]
    for col in news_cols:
        if col in merged:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0
    # 종목 정보 추가
    info = stock_info[code]
    merged["company_name"] = info["name"]
    merged["market"]       = info["market"]
    merged["sector"]       = info["sector"]
    all_merged.append(merged)

# 7) 최종 concat & 저장
if all_merged:
    df_final = pd.concat(all_merged, ignore_index=True)
    os.makedirs(merged_dir, exist_ok=True)
    df_final.to_csv(output_fp, index=False, encoding="utf-8-sig")
    print(f"\n✅ 최종 병합 완료: {len(df_final)}행 -> {output_fp}")
else:
    print("⚠️ 병합할 데이터가 없습니다.")
