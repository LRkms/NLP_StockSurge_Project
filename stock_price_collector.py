import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockPriceCollector:
    def __init__(self, matched_news_path: str = "news_stock_matched.csv", 
                 stock_master_path: str = "stock_master.csv"):
        """
        주가 데이터 수집기 초기화
        
        Args:
            matched_news_path: 매칭된 뉴스 데이터 경로
            stock_master_path: 종목 마스터 데이터 경로
        """
        self.matched_news_path = matched_news_path
        self.stock_master_path = stock_master_path
        
        # 데이터 로드
        self.matched_news = None
        self.stock_master = None
        self.price_cache = {}  # 주가 데이터 캐시
        
        self.load_data()
        
    def load_data(self):
        """데이터 로드"""
        try:
            # 매칭된 뉴스 데이터 로드
            self.matched_news = pd.read_csv(self.matched_news_path, encoding='utf-8-sig')
            # 매칭된 뉴스만 필터링
            self.matched_news = self.matched_news[self.matched_news['matched_stock_code'].notna()].copy()
            
            # 날짜 형식 변환
            self.matched_news['date'] = pd.to_datetime(self.matched_news['date'])
            
            # 종목 마스터 로드
            self.stock_master = pd.read_csv(self.stock_master_path, encoding='utf-8-sig')
            
            logger.info(f"데이터 로드 완료:")
            logger.info(f"  - 매칭된 뉴스: {len(self.matched_news):,}개")
            logger.info(f"  - 종목 마스터: {len(self.stock_master)}개")
            logger.info(f"  - 뉴스 기간: {self.matched_news['date'].min()} ~ {self.matched_news['date'].max()}")
            
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            raise
    
    def get_ticker_symbol(self, stock_code: str, market: str) -> str:
        """yfinance 티커 심볼 생성 (수정된 버전)"""
        # float 형태의 종목코드를 정수로 변환 후 6자리로 패딩
        try:
            # float을 int로 변환하여 소수점 제거
            code_int = int(float(stock_code))
            code_str = str(code_int).zfill(6)
        except (ValueError, TypeError):
            # 이미 문자열이거나 변환 실패시
            code_str = str(stock_code).replace('.0', '').zfill(6)
        
        suffix = "KS" if market == "코스피" else "KQ"
        return f"{code_str}.{suffix}"
    
    def fetch_stock_price(self, stock_code: str, market: str, 
                         start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """개별 종목 주가 데이터 수집 (수정된 버전)"""
        ticker_symbol = self.get_ticker_symbol(stock_code, market)
        
        # 캐시 확인
        cache_key = f"{ticker_symbol}_{start_date}_{end_date}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        try:
            # 디버깅 로그 (처음 5번만)
            if len(self.price_cache) < 5:
                logger.info(f"주가 수집: {ticker_symbol} ({start_date} ~ {end_date})")
            
            ticker = yf.Ticker(ticker_symbol)
            
            # 주가 데이터 수집 (auto_adjust=True로 수정주가 사용)
            price_data = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=False)
            
            if not price_data.empty:
                # 타임존 정보 제거 및 날짜 변환
                price_data.index = price_data.index.tz_localize(None)
                price_data = price_data.reset_index()
                price_data['Date'] = pd.to_datetime(price_data['Date']).dt.date
                
                # 필요한 컬럼 확인 및 선택
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                available_cols = [col for col in required_cols if col in price_data.columns]
                
                if 'Close' not in available_cols:
                    logger.warning(f"종가 데이터 없음: {ticker_symbol}")
                    return None
                
                price_data = price_data[available_cols].copy()
                
                # 디버깅 로그 (처음 5번만)
                if len(self.price_cache) < 5:
                    logger.info(f"  성공: {len(price_data)}일, 최신종가 {price_data['Close'].iloc[-1]:.0f}원")
                
                # 캐시에 저장
                self.price_cache[cache_key] = price_data
                return price_data
            else:
                if len(self.price_cache) < 5:
                    logger.warning(f"  실패: 데이터 없음 - {ticker_symbol}")
                return None
                
        except Exception as e:
            if len(self.price_cache) < 5:
                logger.warning(f"  오류: {ticker_symbol} - {e}")
            return None
    
    def calculate_returns(self, news_date: pd.Timestamp, stock_code: str, market: str,
                         periods: List[int] = [1, 3, 5]) -> Dict:
        """뉴스 발생일 기준 수익률 계산 (수정된 버전)"""
        
        # 주가 데이터 수집 기간 설정 (뉴스일 전후 충분한 기간)
        start_date = (news_date - timedelta(days=15)).strftime('%Y-%m-%d')
        end_date = (news_date + timedelta(days=max(periods) + 10)).strftime('%Y-%m-%d')
        
        # 디버깅 로그 (처음 3번만)
        if len(self.price_cache) < 3:
            logger.info(f"수익률 계산: {stock_code} ({news_date.strftime('%Y-%m-%d')})")
            logger.info(f"  데이터 수집 기간: {start_date} ~ {end_date}")
        
        price_data = self.fetch_stock_price(stock_code, market, start_date, end_date)
        
        if price_data is None or len(price_data) < 2:
            if len(self.price_cache) < 3:
                logger.warning(f"  주가 데이터 부족: {stock_code}")
            return {f'return_{p}d': np.nan for p in periods}
        
        # 뉴스 발생일
        news_date_only = news_date.date()
        
        if len(self.price_cache) < 3:
            logger.info(f"  뉴스 발생일: {news_date_only}")
            logger.info(f"  주가 데이터 날짜 범위: {price_data['Date'].min()} ~ {price_data['Date'].max()}")
        
        # 뉴스 발생일 또는 그 이후 첫 거래일 찾기
        base_price_row = None
        base_idx = None
        
        for i, row in price_data.iterrows():
            if row['Date'] >= news_date_only:
                base_price_row = row
                base_idx = i
                break
        
        if base_price_row is None:
            if len(self.price_cache) < 3:
                logger.warning(f"  기준일 찾기 실패: {news_date_only}")
            return {f'return_{p}d': np.nan for p in periods}
        
        base_price = base_price_row['Close']
        base_date = base_price_row['Date']
        
        if len(self.price_cache) < 3:
            logger.info(f"  기준일: {base_date}, 기준가: {base_price:.0f}원")
        
        returns = {}
        
        # 각 기간별 수익률 계산
        for period in periods:
            target_idx = base_idx + period
            
            if target_idx < len(price_data):
                target_row = price_data.iloc[target_idx]
                target_price = target_row['Close']
                target_date = target_row['Date']
                return_rate = (target_price - base_price) / base_price * 100
                returns[f'return_{period}d'] = round(return_rate, 4)
                
                if len(self.price_cache) < 3:
                    logger.info(f"  {period}일후: {target_date}, {target_price:.0f}원, 수익률: {return_rate:.2f}%")
            else:
                returns[f'return_{period}d'] = np.nan
                if len(self.price_cache) < 3:
                    logger.warning(f"  {period}일후: 데이터 부족")
        
        # 추가 정보
        returns['base_price'] = base_price
        returns['base_date'] = base_date
        
        return returns
    
    def create_labels(self, returns: Dict, thresholds: Dict = None) -> Dict:
        """수익률 기반 라벨 생성"""
        if thresholds is None:
            thresholds = {
                'strong_up': 3.0,    # 강한 상승
                'up': 1.0,           # 상승
                'down': -1.0,        # 하락
                'strong_down': -3.0  # 강한 하락
            }
        
        labels = {}
        
        for period in [1, 3, 5]:
            return_key = f'return_{period}d'
            label_key = f'label_{period}d'
            
            if return_key in returns and not pd.isna(returns[return_key]):
                return_value = returns[return_key]
                
                if return_value >= thresholds['strong_up']:
                    labels[label_key] = 'strong_up'
                elif return_value >= thresholds['up']:
                    labels[label_key] = 'up'
                elif return_value <= thresholds['strong_down']:
                    labels[label_key] = 'strong_down'
                elif return_value <= thresholds['down']:
                    labels[label_key] = 'down'
                else:
                    labels[label_key] = 'neutral'
            else:
                labels[label_key] = 'unknown'
        
        return labels
    
    def process_all_news(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """모든 뉴스에 대해 주가 데이터 처리"""
        
        # 샘플링 (테스트용)
        if sample_size:
            news_sample = self.matched_news.sample(min(sample_size, len(self.matched_news))).copy()
            logger.info(f"샘플링: {len(news_sample)}개 뉴스")
        else:
            news_sample = self.matched_news.copy()
        
        logger.info(f"주가 데이터 처리 시작: {len(news_sample)}개 뉴스")
        
        # 종목별로 그룹화하여 처리 (API 호출 최적화)
        unique_stocks = news_sample[['matched_stock_code', 'matched_market']].drop_duplicates()
        logger.info(f"처리할 종목 수: {len(unique_stocks)}개")
        
        results = []
        processed_count = 0
        
        # 진행률 표시를 위한 tqdm 사용
        for idx, row in tqdm(news_sample.iterrows(), total=len(news_sample), desc="뉴스 처리"):
            try:
                # 수익률 계산 (종목코드 형식 수정)
                stock_code_clean = str(int(float(row['matched_stock_code']))).zfill(6)
                returns = self.calculate_returns(
                    row['date'], 
                    stock_code_clean, 
                    row['matched_market']
                )
                
                # 라벨 생성
                labels = self.create_labels(returns)
                
                # 결과 병합
                result_row = row.copy()
                for key, value in returns.items():
                    result_row[key] = value
                for key, value in labels.items():
                    result_row[key] = value
                
                results.append(result_row)
                processed_count += 1
                
                # API 호출 간격 (과도한 요청 방지)
                if processed_count % 50 == 0:
                    time.sleep(1)
                    logger.info(f"처리 완료: {processed_count}/{len(news_sample)} ({processed_count/len(news_sample)*100:.1f}%)")
                
            except Exception as e:
                logger.warning(f"뉴스 처리 실패 (인덱스 {idx}): {e}")
                # 실패한 경우에도 원본 데이터는 유지
                result_row = row.copy()
                for period in [1, 3, 5]:
                    result_row[f'return_{period}d'] = np.nan
                    result_row[f'label_{period}d'] = 'unknown'
                result_row['base_price'] = np.nan
                result_row['base_date'] = None
                results.append(result_row)
        
        result_df = pd.DataFrame(results)
        logger.info(f"주가 데이터 처리 완료: {len(result_df)}개")
        
        return result_df
    
    def analyze_results(self, labeled_df: pd.DataFrame) -> Dict:
        """라벨링 결과 분석"""
        analysis = {}
        
        # 기본 통계
        total_news = len(labeled_df)
        valid_returns = {}
        label_distributions = {}
        
        for period in [1, 3, 5]:
            return_col = f'return_{period}d'
            label_col = f'label_{period}d'
            
            # 유효한 수익률 데이터 비율
            valid_count = labeled_df[return_col].notna().sum()
            valid_returns[f'{period}d'] = {
                'count': valid_count,
                'ratio': round(valid_count / total_news * 100, 2)
            }
            
            # 라벨 분포
            if valid_count > 0:
                label_dist = labeled_df[labeled_df[return_col].notna()][label_col].value_counts()
                label_distributions[f'{period}d'] = label_dist.to_dict()
        
        analysis['basic_stats'] = {
            'total_news': total_news,
            'valid_returns': valid_returns
        }
        analysis['label_distributions'] = label_distributions
        
        # 종목별 통계 (JSON 직렬화 가능하도록 수정)
        stock_stats_raw = labeled_df.groupby('matched_stock_name').agg({
            'return_1d': ['count', 'mean', 'std'],
            'return_3d': ['count', 'mean', 'std'],
            'return_5d': ['count', 'mean', 'std']
        }).round(4)
        
        # MultiIndex 컬럼을 단순화
        stock_stats_dict = {}
        for stock_name in stock_stats_raw.index[:10]:  # 상위 10개만
            stock_data = {}
            for period in ['1d', '3d', '5d']:
                stock_data[f'return_{period}_count'] = int(stock_stats_raw.loc[stock_name, ('return_' + period, 'count')])
                stock_data[f'return_{period}_mean'] = float(stock_stats_raw.loc[stock_name, ('return_' + period, 'mean')])
                stock_data[f'return_{period}_std'] = float(stock_stats_raw.loc[stock_name, ('return_' + period, 'std')])
            stock_stats_dict[stock_name] = stock_data
        
        analysis['top_stocks_stats'] = stock_stats_dict
        
        return analysis
    
    def save_results(self, labeled_df: pd.DataFrame, filename: str = "news_with_labels.csv"):
        """결과 저장 및 요약"""
        try:
            # CSV 저장
            labeled_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"라벨링된 데이터 저장: {filename}")
            
            # 분석 결과
            analysis = self.analyze_results(labeled_df)
            
            # 분석 결과 JSON 저장
            analysis_filename = filename.replace('.csv', '_analysis.json')
            import json
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
            
            # 요약 출력
            self.print_summary(analysis)
            
            return True
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            return False
    
    def print_summary(self, analysis: Dict):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("주가 라벨링 결과 요약")
        print("="*60)
        
        stats = analysis['basic_stats']
        print(f"총 뉴스 수: {stats['total_news']:,}개")
        
        print(f"\n기간별 유효 데이터:")
        for period, data in stats['valid_returns'].items():
            print(f"  - {period}: {data['count']:,}개 ({data['ratio']}%)")
        
        print(f"\n기간별 라벨 분포:")
        for period, labels in analysis['label_distributions'].items():
            print(f"  {period}:")
            for label, count in labels.items():
                print(f"    - {label}: {count:,}개")
        
        print("="*60)

def test_yfinance_connection():
    """yfinance 연결 테스트 (간단 버전)"""
    print("yfinance 한국 주식 연결 테스트...")
    
    test_stocks = [
        ("005930", "KS", "삼성전자"),
        ("000660", "KS", "SK하이닉스"), 
        ("035420", "KS", "NAVER")
    ]
    
    success_count = 0
    for code, suffix, name in test_stocks:
        ticker_symbol = f"{code}.{suffix}"
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d", auto_adjust=True)
            
            if not hist.empty:
                latest_price = hist['Close'].iloc[-1]
                print(f"✅ {name}: {latest_price:.0f}원")
                success_count += 1
            else:
                print(f"❌ {name}: 데이터 없음")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    if success_count >= 2:
        print(f"✅ yfinance 연결 성공 ({success_count}/3)")
        return True
    else:
        print(f"❌ yfinance 연결 실패 ({success_count}/3)")
        return False
    
    # 필요한 파일 확인
    required_files = ["news_stock_matched.csv", "stock_master.csv"]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"필요한 파일이 없습니다: {file_path}")
            return
    
    try:
        # 수집기 초기화
        collector = StockPriceCollector()
        
        # 테스트 실행 여부 확인
        test_mode = input("테스트 모드로 실행하시겠습니까? (y/n): ").lower() == 'y'
        
        if test_mode:
            sample_size = 1000
            print(f"테스트 모드: {sample_size}개 뉴스만 처리합니다.")
            labeled_df = collector.process_all_news(sample_size=sample_size)
            filename = "news_with_labels_test.csv"
        else:
            print("전체 데이터를 처리합니다. 시간이 오래 걸릴 수 있습니다.")
            labeled_df = collector.process_all_news()
            filename = "news_with_labels.csv"
        
        # 결과 저장
        success = collector.save_results(labeled_df, filename)
        
        if success:
            print(f"\n라벨링 완료!")
            print(f"생성된 파일:")
            print(f"  - {filename}: 라벨링된 뉴스 데이터")
            print(f"  - {filename.replace('.csv', '_analysis.json')}: 분석 결과")
            print(f"\n다음 단계: 모델 학습 데이터셋 구축")
        
    except Exception as e:
        logger.error(f"주가 라벨링 프로세스 실패: {e}")
        print("라벨링에 실패했습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    # 직접 실행
    print("주가 데이터 수집 및 라벨링을 시작합니다.")
    
    # yfinance 연결 테스트
    test_yfinance_connection()
    
    # 필요한 파일 확인
    required_files = ["news_stock_matched.csv", "stock_master.csv"]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"필요한 파일이 없습니다: {file_path}")
            exit()
    
    try:
        # 수집기 초기화
        collector = StockPriceCollector()
        
        # 테스트 실행 여부 확인
        test_mode = input("테스트 모드로 실행하시겠습니까? (y/n): ").lower() == 'y'
        
        if test_mode:
            sample_size = 100  # 더 작은 샘플로 테스트
            print(f"테스트 모드: {sample_size}개 뉴스만 처리합니다.")
            labeled_df = collector.process_all_news(sample_size=sample_size)
            filename = "news_with_labels_test.csv"
        else:
            print("전체 데이터를 처리합니다. 시간이 오래 걸릴 수 있습니다.")
            labeled_df = collector.process_all_news()
            filename = "news_with_labels.csv"
        
        # 결과 저장
        success = collector.save_results(labeled_df, filename)
        
        if success:
            print(f"\n라벨링 완료!")
            print(f"생성된 파일:")
            print(f"  - {filename}: 라벨링된 뉴스 데이터")
            print(f"  - {filename.replace('.csv', '_analysis.json')}: 분석 결과")
        
    except Exception as e:
        logger.error(f"주가 라벨링 프로세스 실패: {e}")
        print("라벨링에 실패했습니다. 로그를 확인해주세요.")