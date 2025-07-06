import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def debug_step_by_step():
    """단계별 문제 진단"""
    print("=== 단계별 문제 진단 시작 ===\n")
    
    # 1. 매칭된 뉴스 데이터 확인
    print("1. 매칭된 뉴스 데이터 로드 및 확인")
    try:
        matched_news = pd.read_csv("news_stock_matched.csv", encoding='utf-8-sig')
        matched_news = matched_news[matched_news['matched_stock_code'].notna()].copy()
        print(f"   총 매칭된 뉴스: {len(matched_news)}개")
        
        # 날짜 변환
        matched_news['date'] = pd.to_datetime(matched_news['date'])
        print(f"   날짜 범위: {matched_news['date'].min()} ~ {matched_news['date'].max()}")
        
        # 샘플 5개 확인
        sample = matched_news.head(5)[['date', 'matched_stock_code', 'matched_stock_name', 'matched_market']]
        print(f"\n   샘플 데이터:")
        for _, row in sample.iterrows():
            print(f"     {row['date'].strftime('%Y-%m-%d')} | {row['matched_stock_code']} | {row['matched_stock_name']} | {row['matched_market']}")
        
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return
    
    # 2. 개별 뉴스로 테스트
    print(f"\n2. 개별 뉴스로 주가 수집 테스트")
    test_news = matched_news.iloc[0]
    
    print(f"   테스트 뉴스:")
    print(f"     날짜: {test_news['date']}")
    print(f"     종목: {test_news['matched_stock_name']} ({test_news['matched_stock_code']})")
    print(f"     시장: {test_news['matched_market']}")
    
    # 3. 티커 심볼 생성 테스트
    print(f"\n3. 티커 심볼 생성 테스트")
    stock_code = str(test_news['matched_stock_code']).zfill(6)
    market = test_news['matched_market']
    suffix = "KS" if market == "코스피" else "KQ"
    ticker_symbol = f"{stock_code}.{suffix}"
    
    print(f"   종목코드: {test_news['matched_stock_code']} → {stock_code}")
    print(f"   시장구분: {market} → .{suffix}")
    print(f"   티커심볼: {ticker_symbol}")
    
    # 4. yfinance로 주가 데이터 수집 테스트
    print(f"\n4. yfinance 주가 데이터 수집 테스트")
    news_date = test_news['date']
    start_date = (news_date - timedelta(days=15)).strftime('%Y-%m-%d')
    end_date = (news_date + timedelta(days=10)).strftime('%Y-%m-%d')
    
    print(f"   데이터 수집 기간: {start_date} ~ {end_date}")
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        price_data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if not price_data.empty:
            print(f"   ✅ 주가 데이터 수집 성공: {len(price_data)}일")
            
            # 타임존 제거 및 변환
            price_data.index = price_data.index.tz_localize(None)
            price_data = price_data.reset_index()
            price_data['Date'] = pd.to_datetime(price_data['Date']).dt.date
            
            print(f"   날짜 범위: {price_data['Date'].min()} ~ {price_data['Date'].max()}")
            print(f"   컬럼: {price_data.columns.tolist()}")
            
            # 최근 5일 샘플 출력
            print(f"\n   최근 5일 데이터:")
            for _, row in price_data.tail(5).iterrows():
                print(f"     {row['Date']}: {row['Close']:.0f}원")
                
        else:
            print(f"   ❌ 주가 데이터 없음")
            return
            
    except Exception as e:
        print(f"   ❌ 주가 수집 오류: {e}")
        return
    
    # 5. 수익률 계산 테스트
    print(f"\n5. 수익률 계산 테스트")
    news_date_only = news_date.date()
    print(f"   뉴스 발생일: {news_date_only}")
    
    # 기준일 찾기
    base_price_row = None
    base_idx = None
    
    for i, row in price_data.iterrows():
        if row['Date'] >= news_date_only:
            base_price_row = row
            base_idx = i
            break
    
    if base_price_row is None:
        print(f"   ❌ 기준일을 찾을 수 없음")
        print(f"   뉴스일: {news_date_only}")
        print(f"   주가 날짜들: {price_data['Date'].tolist()}")
        return
    
    base_price = base_price_row['Close']
    base_date = base_price_row['Date']
    print(f"   ✅ 기준일: {base_date}, 기준가: {base_price:.0f}원")
    
    # 1, 3, 5일 후 수익률 계산
    periods = [1, 3, 5]
    for period in periods:
        target_idx = base_idx + period
        
        if target_idx < len(price_data):
            target_row = price_data.iloc[target_idx]
            target_price = target_row['Close']
            target_date = target_row['Date']
            return_rate = (target_price - base_price) / base_price * 100
            
            print(f"   ✅ {period}일후: {target_date}, {target_price:.0f}원, 수익률: {return_rate:.2f}%")
        else:
            print(f"   ❌ {period}일후: 데이터 부족 (인덱스 {target_idx} >= 길이 {len(price_data)})")
    
    # 6. stock_price_collector 코드 문제점 확인
    print(f"\n6. stock_price_collector 코드 문제점 확인")
    
    # 실제 코드에서 사용하는 방식으로 테스트
    print("   실제 코드 방식으로 재테스트...")
    
    try:
        from stock_price_collector import StockPriceCollector
        
        collector = StockPriceCollector()
        
        # 첫 번째 뉴스로 테스트
        returns = collector.calculate_returns(
            test_news['date'], 
            str(test_news['matched_stock_code']).zfill(6), 
            test_news['matched_market']
        )
        
        print(f"   calculate_returns 결과: {returns}")
        
        if any(not pd.isna(v) for k, v in returns.items() if 'return_' in k):
            print(f"   ✅ calculate_returns 정상 작동")
        else:
            print(f"   ❌ calculate_returns 모든 값이 NaN")
            
    except Exception as e:
        print(f"   ❌ stock_price_collector 오류: {e}")
    
    print(f"\n=== 진단 완료 ===")

def test_specific_stock():
    """특정 종목으로 정확한 테스트"""
    print("\n=== 특정 종목 상세 테스트 ===")
    
    # 삼성전자로 고정 테스트
    stock_code = "005930"
    ticker_symbol = f"{stock_code}.KS"
    test_date = datetime(2025, 1, 15)  # 확실한 거래일
    
    print(f"테스트 종목: {ticker_symbol}")
    print(f"테스트 날짜: {test_date.strftime('%Y-%m-%d')}")
    
    # 주가 데이터 수집
    start_date = (test_date - timedelta(days=15)).strftime('%Y-%m-%d')
    end_date = (test_date + timedelta(days=10)).strftime('%Y-%m-%d')
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        price_data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if not price_data.empty:
            price_data.index = price_data.index.tz_localize(None)
            price_data = price_data.reset_index()
            price_data['Date'] = pd.to_datetime(price_data['Date']).dt.date
            
            print(f"주가 데이터: {len(price_data)}일")
            print(f"날짜 범위: {price_data['Date'].min()} ~ {price_data['Date'].max()}")
            
            # 전체 데이터 출력
            print(f"\n전체 주가 데이터:")
            for _, row in price_data.iterrows():
                print(f"  {row['Date']}: {row['Close']:.0f}원")
            
            # 수익률 계산
            test_date_only = test_date.date()
            base_idx = None
            
            for i, row in price_data.iterrows():
                if row['Date'] >= test_date_only:
                    base_idx = i
                    break
            
            if base_idx is not None:
                print(f"\n기준일 인덱스: {base_idx}")
                base_price = price_data.iloc[base_idx]['Close']
                
                for period in [1, 3, 5]:
                    target_idx = base_idx + period
                    if target_idx < len(price_data):
                        target_price = price_data.iloc[target_idx]['Close']
                        return_rate = (target_price - base_price) / base_price * 100
                        print(f"{period}일 수익률: {return_rate:.2f}%")
            
        else:
            print("주가 데이터 없음")
            
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    debug_step_by_step()
    test_specific_stock()
    