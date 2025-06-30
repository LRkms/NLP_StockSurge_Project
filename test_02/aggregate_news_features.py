import os
import json
import pandas as pd

# 1) 경로 설정
preprocessed_dir = "preprocessed"
merged_dir      = "merged_data"
os.makedirs(merged_dir, exist_ok=True)

news_full_fp   = os.path.join(preprocessed_dir, "processed_news_full.csv")
stock_list_fp  = os.path.join(merged_dir, "stock_list.csv")
output_fp      = os.path.join(merged_dir, "daily_news_features.csv")

# 2) 매핑 로드
df_stock = pd.read_csv(stock_list_fp, dtype=str)
# 'name' 컬럼이 회사명이므로, dict 로 변환
name_to_code = dict(zip(df_stock['name'], df_stock['code']))

# 3) 전체 뉴스 로드
df_news = pd.read_csv(news_full_fp, encoding="utf-8-sig", dtype=str)
# JSON 문자열로 저장된 칼럼 파싱
df_news['companies_list'] = df_news['companies'].apply(json.loads)

# 4) explode 해서 한 행에 회사명 하나씩
df_exp = df_news.explode('companies_list') \
            .rename(columns={'companies_list': 'company_name'})

# 5) 회사명 → code 매핑
df_exp['code'] = df_exp['company_name'].map(name_to_code)
# 매핑된 것만 남기기
df_exp = df_exp[df_exp['code'].notna()]

# ■ 6) 날짜형 변환
df_exp['date'] = pd.to_datetime(df_exp['date']).dt.date

# ■ 6.1) 숫자 칼럼 타입 변환
num_cols = [
    'positive_score', 'negative_score', 'neutral_score',
    'price_up_signal', 'price_down_signal', 'price_neutral_signal'
]
for col in num_cols:
    df_exp[col] = pd.to_numeric(df_exp[col], errors='coerce')  # 변환 불가 값은 NaN 처리

# 필요하면 NaN을 0으로 채우기
df_exp[num_cols] = df_exp[num_cols].fillna(0)


# 7) 집계 함수 정의
agg_dict = {
    'link'               : 'count',   # 뉴스 개수
    'positive_score'     : 'mean',
    'negative_score'     : 'mean',
    'neutral_score'      : 'mean',
    'price_up_signal'    : 'sum',
    'price_down_signal'  : 'sum',
    'price_neutral_signal':'sum'
}

# 8) 그룹화 & 집계
df_daily = df_exp.groupby(['date','code']) \
                 .agg(agg_dict) \
                 .reset_index()

# 칼럼명 정리
df_daily = df_daily.rename(columns={
    'link': 'news_count',
    'positive_score': 'avg_positive_score',
    'negative_score': 'avg_negative_score',
    'neutral_score': 'avg_neutral_score',
    'price_up_signal': 'total_price_up_signal',
    'price_down_signal': 'total_price_down_signal',
    'price_neutral_signal': 'total_price_neutral_signal'
})

# 9) 결과 저장
df_daily.to_csv(output_fp, index=False, encoding="utf-8-sig")
print(f"✅ 일별 뉴스 feature 집계 완료: {len(df_daily)}행 -> {output_fp}")
