import os
import pandas as pd
from glob import glob

# 1) 디렉토리 설정
preprocessed_dir = "preprocessed"
output_path = os.path.join(preprocessed_dir, "processed_news_full.csv")

# 2) 파일 목록 수집
pattern = os.path.join(preprocessed_dir, "processed_news_*.csv")
files = glob(pattern)
if not files:
    raise FileNotFoundError(f"'{pattern}'에 해당하는 파일이 없습니다.")

print(f"▶ 합칠 파일 개수: {len(files)}개")

# 3) 파일별 DataFrame 로드
df_list = []
for fp in files:
    df = pd.read_csv(fp, encoding="utf-8-sig")
    df_list.append(df)

# 4) concat
df_all = pd.concat(df_list, ignore_index=True)

# 5) (선택) 중복 제거 — 같은 title+link 기준
df_all.drop_duplicates(subset=["title", "link"], keep="first", inplace=True)

# 6) 저장
df_all.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 전체 데이터 합치기 완료: {len(df_all)}행 -> {output_path}")