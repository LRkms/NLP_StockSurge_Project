import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import yfinance as yf
# import FinanceDataReader as fdr  # 설치 문제시 주석 처리
try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except ImportError:
    logger.warning("FinanceDataReader를 사용할 수 없습니다. yfinance만 사용합니다.")
    FDR_AVAILABLE = False
from datetime import datetime, timedelta
import warnings
import logging
import json
from typing import Dict, List, Tuple
# import talib  # TA-Lib 대신 pandas로 기술적 지표 계산
import pandas_ta as ta

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# --- 설정 ---
preprocessed_dir = "preprocessed"
output_dir = "merged_data"
os.makedirs(output_dir, exist_ok=True)

class TimeSeriesMerger:
    def __init__(self):
        self.stock_codes = {}
        self.price_data = {}
        self.news_data = None
        
    def get_stock_codes(self) -> Dict[str, pd.DataFrame]:
        """종목코드 수집"""
        logger.info("종목코드 수집 시작...")
        
        if not FDR_AVAILABLE:
            logger.info("FinanceDataReader가 없으므로 수동 종목코드 사용...")
            return self.get_manual_stock_codes()
        
        try:
            # FinanceDataReader 사용
            kospi_stocks = fdr.StockListing('KOSPI')
            kosdaq_stocks = fdr.StockListing('KOSDAQ')
            
            # 시가총액 기준 상위 종목 선택
            kospi_top100 = kospi_stocks.nlargest(100, 'Marcap').reset_index(drop=True)
            kosdaq_top50 = kosdaq_stocks.nlargest(50, 'Marcap').reset_index(drop=True)
            
            # 종목코드와 이름 매핑 생성
            stock_codes = {}
            
            # KOSPI 종목
            for _, row in kospi_top100.iterrows():
                code = str(row['Code']).zfill(6)
                stock_codes[code] = {
                    'name': row['Name'],
                    'market': 'KOSPI',
                    'sector': row.get('Sector', 'Unknown'),
                    'industry': row.get('Industry', 'Unknown')
                }
            
            # KOSDAQ 종목
            for _, row in kosdaq_top50.iterrows():
                code = str(row['Code']).zfill(6)
                stock_codes[code] = {
                    'name': row['Name'],
                    'market': 'KOSDAQ',
                    'sector': row.get('Sector', 'Unknown'),
                    'industry': row.get('Industry', 'Unknown')
                }
            
            self.stock_codes = stock_codes
            logger.info(f"총 {len(stock_codes)}개 종목코드 수집 완료")
            
            # 종목 리스트 저장
            stock_df = pd.DataFrame([
                {'code': code, **info} for code, info in stock_codes.items()
            ])
            stock_df.to_csv(os.path.join(output_dir, 'stock_list.csv'), 
                          index=False, encoding='utf-8-sig')
            
            return stock_codes
            
        except Exception as e:
            logger.error(f"종목코드 수집 실패: {e}")
            # 백업: 수동으로 주요 종목들 정의
            return self.get_manual_stock_codes()
    
    def get_manual_stock_codes(self) -> Dict[str, Dict]:
        """수동 종목코드 (백업용)"""
        logger.info("수동 종목코드 사용...")
        
        manual_codes = {
            # 주요 대형주들
            '005930': {'name': '삼성전자', 'market': 'KOSPI', 'sector': 'Technology'},
            '000660': {'name': 'SK하이닉스', 'market': 'KOSPI', 'sector': 'Technology'},
            '035420': {'name': 'NAVER', 'market': 'KOSPI', 'sector': 'Technology'},
            '005380': {'name': '현대차', 'market': 'KOSPI', 'sector': 'Automotive'},
            '006400': {'name': '삼성SDI', 'market': 'KOSPI', 'sector': 'Battery'},
            '051910': {'name': 'LG화학', 'market': 'KOSPI', 'sector': 'Chemical'},
            '105560': {'name': 'KB금융', 'market': 'KOSPI', 'sector': 'Financial'},
            '055550': {'name': '신한지주', 'market': 'KOSPI', 'sector': 'Financial'},
            '035720': {'name': '카카오', 'market': 'KOSPI', 'sector': 'Technology'},
            '068270': {'name': '셀트리온', 'market': 'KOSPI', 'sector': 'Bio'},
            # 코스닥 주요 종목들
            '086520': {'name': '에코프로', 'market': 'KOSDAQ', 'sector': 'Battery'},
            '247540': {'name': '에코프로비엠', 'market': 'KOSDAQ', 'sector': 'Battery'},
            '091990': {'name': '셀트리온헬스케어', 'market': 'KOSDAQ', 'sector': 'Bio'},
            '096770': {'name': 'SK이노베이션', 'market': 'KOSDAQ', 'sector': 'Energy'},
            '293490': {'name': '카카오게임즈', 'market': 'KOSDAQ', 'sector': 'Game'}
        }
        
        self.stock_codes = manual_codes
        return manual_codes
    
    def get_stock_price_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            # yfinance용 코드 변환 (한국 주식은 .KS 또는 .KQ 접미사)
            market = self.stock_codes[code]['market']
            suffix = '.KS' if market == 'KOSPI' else '.KQ'
            yf_code = code + suffix
            
            # yfinance로 데이터 수집
            stock = yf.Ticker(yf_code)
            df = stock.history(start=start_date, end=end_date)
            
            logger.info(f"yfinance 데이터 수집 ({code}): {len(df)}개 레코드")
            
            if df.empty:
                logger.warning(f"yfinance 데이터 없음, FDR 시도: {code}")
                # yfinance 실패시 FinanceDataReader 시도 (가능한 경우)
                if FDR_AVAILABLE:
                    df = fdr.DataReader(code, start_date, end_date)
                    logger.info(f"FDR 데이터 수집 ({code}): {len(df)}개 레코드")
            
            if df.empty:
                logger.warning(f"주가 데이터 없음: {code}")
                return pd.DataFrame()
            
            # 디버깅: 원본 데이터 구조 확인
            logger.debug(f"원본 컬럼명 ({code}): {list(df.columns)}")
            logger.debug(f"원본 인덱스명 ({code}): {df.index.name}")
            logger.debug(f"원본 인덱스 타입 ({code}): {type(df.index)}")
            
            # 인덱스를 컬럼으로 변환
            df = df.reset_index()
            
            # 디버깅: reset_index 후 컬럼명 확인
            logger.debug(f"reset_index 후 컬럼명 ({code}): {list(df.columns)}")
            
            # 날짜 컬럼 찾기 및 정규화
            date_column = None
            possible_date_columns = ['Date', 'date', 'Datetime', 'datetime', 'index']
            
            for col in possible_date_columns:
                if col in df.columns:
                    date_column = col
                    break
            
            if date_column is None:
                # 첫 번째 컬럼이 날짜일 가능성 체크
                first_col = df.columns[0]
                if pd.api.types.is_datetime64_any_dtype(df[first_col]) or \
                isinstance(df[first_col].iloc[0], (pd.Timestamp, datetime)):
                    date_column = first_col
            
            if date_column is None:
                logger.error(f"날짜 컬럼을 찾을 수 없습니다 ({code}): {list(df.columns)}")
                return pd.DataFrame()
            
            # 날짜 컬럼명을 'date'로 통일
            if date_column != 'date':
                df = df.rename(columns={date_column: 'date'})
                logger.debug(f"날짜 컬럼 이름 변경: {date_column} -> date")
            
            # 다른 컬럼명들도 정규화
            column_mapping = {}
            for col in df.columns:
                if col != 'date':  # 날짜 컬럼은 이미 처리됨
                    new_col = col.lower().replace(' ', '_')
                    if new_col != col:
                        column_mapping[col] = new_col
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
                logger.debug(f"컬럼명 정규화 ({code}): {column_mapping}")
            
            # 날짜 형식 통일
            try:
                df['date'] = pd.to_datetime(df['date']).dt.date
            except Exception as e:
                logger.error(f"날짜 변환 실패 ({code}): {e}")
                return pd.DataFrame()
            
            # 필수 컬럼 확인
            required_columns = ['close', 'volume', 'open', 'high', 'low']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"필수 컬럼 누락 ({code}): {missing_columns}")
                logger.error(f"사용 가능한 컬럼: {list(df.columns)}")
                return pd.DataFrame()
            
            # 종목코드 추가
            df['code'] = code
            
            # 결과 확인
            logger.info(f"데이터 수집 완료 ({code}): {len(df)}개 레코드, 기간: {df['date'].min()} ~ {df['date'].max()}")
            
            return df
        
        except Exception as e:
            logger.error(f"주가 데이터 수집 실패 {code}: {e}")
            logger.error(f"오류 상세: {type(e).__name__}: {str(e)}")
            return pd.DataFrame()

    def debug_stock_data_structure(self, code: str = "005930"):
        """주가 데이터 구조 디버깅 함수"""
        logger.info(f"=== 주가 데이터 구조 디버깅 ({code}) ===")
        
        try:
            # yfinance 테스트
            market = 'KOSPI'  # 삼성전자는 KOSPI
            yf_code = code + '.KS'
            
            stock = yf.Ticker(yf_code)
            df = stock.history(start="2025-06-01", end="2025-06-19")
            
            logger.info(f"yfinance 원본 데이터:")
            logger.info(f"  - 데이터 개수: {len(df)}")
            logger.info(f"  - 컬럼명: {list(df.columns)}")
            logger.info(f"  - 인덱스명: {df.index.name}")
            logger.info(f"  - 인덱스 타입: {type(df.index)}")
            logger.info(f"  - 첫 번째 행:")
            if not df.empty:
                logger.info(f"    {df.iloc[0].to_dict()}")
            
            # reset_index 후
            df_reset = df.reset_index()
            logger.info(f"reset_index 후:")
            logger.info(f"  - 컬럼명: {list(df_reset.columns)}")
            logger.info(f"  - 첫 번째 행:")
            if not df_reset.empty:
                logger.info(f"    {df_reset.iloc[0].to_dict()}")
            
            # FDR 테스트 (가능한 경우)
            if FDR_AVAILABLE:
                logger.info(f"FinanceDataReader 테스트:")
                fdr_df = fdr.DataReader(code, "2025-06-01", "2025-06-19")
                logger.info(f"  - 데이터 개수: {len(fdr_df)}")
                logger.info(f"  - 컬럼명: {list(fdr_df.columns)}")
                logger.info(f"  - 인덱스명: {fdr_df.index.name}")
                if not fdr_df.empty:
                    logger.info(f"  - 첫 번째 행: {fdr_df.iloc[0].to_dict()}")
        
        except Exception as e:
            logger.error(f"디버깅 실패: {e}")

    # 메인 클래스에 디버깅 함수 추가
    def add_debug_method_to_merger():
        """TimeSeriesMerger 클래스에 디버깅 메서드 추가"""
        
        # 기존 merge_data 함수 시작 부분에 디버깅 추가
        def enhanced_merge_data(self, start_date: str, end_date: str):
            """전체 데이터 병합 (디버깅 강화 버전)"""
            logger.info("=== 시계열 데이터 병합 시작 ===")
            
            # 디버깅: 주가 데이터 구조 확인
            logger.info("주가 데이터 구조 디버깅...")
            self.debug_stock_data_structure("005930")  # 삼성전자로 테스트
            
            # 1. 종목코드 수집
            stock_codes = self.get_stock_codes()
            if not stock_codes:
                logger.error("종목코드 수집 실패")
                return
            
            # 테스트용으로 일부 종목만 처리
            test_codes = list(stock_codes.keys())[:3]  # 처음 3개 종목만
            logger.info(f"테스트 종목: {test_codes}")
            
            # 2. 뉴스 데이터 로드 및 매칭
            news_df = self.load_preprocessed_news()
            if news_df.empty:
                logger.warning("뉴스 데이터 없음, 주가 데이터만 처리")
                daily_news_features = pd.DataFrame()
            else:
                matched_news = self.extract_company_from_news(news_df)
                if matched_news.empty:
                    logger.warning("뉴스-종목 매칭 실패")
                    daily_news_features = pd.DataFrame()
                else:
                    daily_news_features = self.aggregate_daily_news_features(matched_news)
            
            # 3. 각 종목별로 주가 데이터 수집 및 병합
            all_merged_data = []
            
            for code in tqdm(test_codes, desc="종목별 데이터 병합"):
                try:
                    logger.info(f"처리 중: {code} ({stock_codes[code]['name']})")
                    
                    # 주가 데이터 수집
                    price_df = self.get_stock_price_data(code, start_date, end_date)
                    if price_df.empty:
                        logger.warning(f"주가 데이터 없음: {code}")
                        continue
                    
                    logger.info(f"주가 데이터 수집 완료: {code}, {len(price_df)}개 레코드")
                    
                    # 기술적 지표 계산
                    price_df = self.calculate_technical_indicators(price_df)
                    logger.info(f"기술적 지표 계산 완료: {code}")
                    
                    # 뉴스 특성과 병합 (뉴스가 있는 경우만)
                    if not daily_news_features.empty:
                        code_news = daily_news_features[daily_news_features['code'] == code]
                        merged = pd.merge(price_df, code_news, on=['date', 'code'], how='left')
                        
                        # 뉴스가 없는 날은 0으로 채우기
                        news_columns = [col for col in merged.columns if col.startswith(('news_', 'avg_', 'max_', 'total_', 'positive_', 'negative_', 'neutral_'))]
                        merged[news_columns] = merged[news_columns].fillna(0)
                    else:
                        merged = price_df.copy()
                    
                    # 종목 정보 추가
                    merged['company_name'] = stock_codes[code]['name']
                    merged['market'] = stock_codes[code]['market']
                    merged['sector'] = stock_codes[code].get('sector', 'Unknown')
                    
                    all_merged_data.append(merged)
                    logger.info(f"병합 완료: {code}, 최종 {len(merged)}개 레코드")
                    
                except Exception as e:
                    logger.error(f"종목 {code} 병합 실패: {e}")
                    import traceback
                    logger.error(f"상세 오류: {traceback.format_exc()}")
                    continue
            
            # 4. 전체 데이터 결합 및 저장
            if all_merged_data:
                final_df = pd.concat(all_merged_data, ignore_index=True)
                
                # 결측치 처리
                final_df = final_df.dropna(subset=['close', 'volume'])  # 필수 컬럼
                
                # 최종 저장
                output_path = os.path.join(output_dir, f'merged_timeseries_{start_date}_{end_date}_debug.csv')
                final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"=== 병합 완료 ===")
                logger.info(f"총 레코드 수: {len(final_df)}")
                logger.info(f"종목 수: {final_df['code'].nunique()}")
                logger.info(f"기간: {final_df['date'].min()} ~ {final_df['date'].max()}")
                logger.info(f"저장 위치: {output_path}")
                
                # 통계 요약
                self.print_merge_statistics(final_df)
                
            else:
                logger.error("병합할 데이터가 없습니다.")
    
    def print_merge_statistics(self, df: pd.DataFrame):
        """병합 결과 통계"""
        logger.info("=== 병합 데이터 통계 ===")
        
        # 기본 통계
        logger.info(f"전체 레코드: {len(df):,}개")
        logger.info(f"종목 수: {df['code'].nunique()}개")
        logger.info(f"날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
        
        # 뉴스 커버리지
        news_coverage = (df['news_count'] > 0).mean() * 100
        logger.info(f"뉴스 커버리지: {news_coverage:.1f}%")
        
        # 시장별 분포
        market_dist = df['market'].value_counts()
        logger.info(f"시장별 분포: {dict(market_dist)}")
        
        # 평균 뉴스 감성
        avg_sentiment = df[df['news_count'] > 0]['avg_positive_score'].mean()
        logger.info(f"평균 긍정 감성 점수: {avg_sentiment:.3f}")

def main():
    """메인 실행 함수"""
    merger = TimeSeriesMerger()
    
    # 기간 설정 (뉴스 크롤링 기간과 맞춰야 함)
    start_date = "2025-01-01"
    end_date = "2025-06-17"
    
    # 데이터 병합 실행
    merger.merge_data(start_date, end_date)

if __name__ == '__main__':
    main()