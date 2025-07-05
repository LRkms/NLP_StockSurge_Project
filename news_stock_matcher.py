import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsStockMatcher:
    def __init__(self, stock_master_path: str = "stock_master.csv"):
        """
        뉴스-종목 매칭기 초기화
        
        Args:
            stock_master_path: 종목 마스터 CSV 파일 경로
        """
        self.stock_master_path = stock_master_path
        self.stock_master = None
        self.matching_dict = {}
        self.company_patterns = {}
        
        # 매칭 로드
        self.load_stock_master()
        self.build_matching_dict()
        
    def load_stock_master(self):
        """종목 마스터 데이터 로드"""
        try:
            self.stock_master = pd.read_csv(self.stock_master_path, encoding='utf-8-sig')
            logger.info(f"종목 마스터 로드 완료: {len(self.stock_master)}개 종목")
        except Exception as e:
            logger.error(f"종목 마스터 로드 실패: {e}")
            raise
    
    def build_matching_dict(self):
        """종목명 매칭 사전 구축"""
        logger.info("종목명 매칭 사전 구축 중...")
        
        self.matching_dict = {}
        
        for _, row in self.stock_master.iterrows():
            stock_code = str(row['종목코드']).zfill(6)
            stock_name = row['종목명']
            
            # 1. 정확한 종목명
            self.matching_dict[stock_name] = {
                'code': stock_code,
                'name': stock_name,
                'market': row['시장구분'],
                'sector': row['섹터'],
                'match_type': 'exact'
            }
            
            # 2. 별칭들
            try:
                aliases = json.loads(row['별칭'])
                for alias in aliases:
                    if alias != stock_name:  # 중복 방지
                        self.matching_dict[alias] = {
                            'code': stock_code,
                            'name': stock_name,
                            'market': row['시장구분'],
                            'sector': row['섹터'],
                            'match_type': 'alias'
                        }
            except:
                pass
            
            # 3. 회사명 패턴 (정규식)
            patterns = self._generate_company_patterns(stock_name)
            for pattern in patterns:
                self.company_patterns[pattern] = {
                    'code': stock_code,
                    'name': stock_name,
                    'market': row['시장구분'],
                    'sector': row['섹터'],
                    'match_type': 'pattern'
                }
        
        logger.info(f"매칭 사전 구축 완료: {len(self.matching_dict)}개 키워드, {len(self.company_patterns)}개 패턴")
    
    def _generate_company_patterns(self, stock_name: str) -> List[str]:
        """회사명 정규식 패턴 생성"""
        patterns = []
        
        # 기본 패턴
        base_name = stock_name
        
        # 1. 기본 회사명 패턴
        patterns.append(base_name)
        
        # 2. 회사 형태 제거 패턴
        if any(suffix in base_name for suffix in ['주식회사', '㈜', 'Corp', 'Inc']):
            clean_name = re.sub(r'(주식회사|㈜|Corp|Inc)', '', base_name).strip()
            if clean_name:
                patterns.append(clean_name)
        
        # 3. 특수 패턴들
        special_patterns = {
            '삼성전자': ['삼성전자.*', '삼성(?!.*생명)(?!.*증권)(?!.*카드)'],
            'SK하이닉스': ['SK하이닉스.*', 'SK.*하이닉스', '하이닉스'],
            'LG에너지솔루션': ['LG에너지.*', 'LG.*에너지', 'LGES'],
            'NAVER': ['네이버.*', 'NAVER.*', '네이버'],
            '카카오': ['카카오(?!.*뱅크)(?!.*게임즈)', 'KAKAO'],
            '현대차': ['현대자동차.*', '현대차.*', '현대.*자동차'],
            '셀트리온': ['셀트리온(?!.*헬스케어)(?!.*제약)'],
            'POSCO홀딩스': ['포스코.*', 'POSCO.*', '포스코'],
            '하이브': ['하이브.*', 'HYBE.*', '방탄소년단', 'BTS']
        }
        
        if stock_name in special_patterns:
            patterns.extend(special_patterns[stock_name])
        
        return patterns
    
    def extract_companies_from_text(self, text: str) -> List[Dict]:
        """텍스트에서 회사명 추출"""
        if not text:
            return []
        
        matched_companies = []
        text_clean = re.sub(r'[^\w가-힣\s]', ' ', text)
        
        # 1. 정확한 매칭 (우선순위 높음)
        for keyword, info in self.matching_dict.items():
            if keyword in text_clean:
                matched_companies.append({
                    **info,
                    'matched_text': keyword,
                    'confidence': 0.9 if info['match_type'] == 'exact' else 0.8
                })
        
        # 2. 패턴 매칭
        for pattern, info in self.company_patterns.items():
            try:
                matches = re.findall(pattern, text_clean, re.IGNORECASE)
                if matches:
                    for match in matches:
                        matched_companies.append({
                            **info,
                            'matched_text': match,
                            'confidence': 0.7
                        })
            except:
                continue
        
        # 3. 중복 제거 및 정렬
        unique_companies = {}
        for company in matched_companies:
            key = company['code']
            if key not in unique_companies or company['confidence'] > unique_companies[key]['confidence']:
                unique_companies[key] = company
        
        # 신뢰도 순으로 정렬
        result = list(unique_companies.values())
        result.sort(key=lambda x: x['confidence'], reverse=True)
        
        return result
    
    def match_news_to_stocks(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """뉴스 데이터에 종목 매칭 정보 추가"""
        logger.info(f"뉴스-종목 매칭 시작: {len(news_df)}개 뉴스")
        
        matched_results = []
        no_match_count = 0
        
        for idx, row in news_df.iterrows():
            # 제목과 본문 합쳐서 매칭
            full_text = f"{row.get('title', '')} {row.get('body', '')}"
            
            # 회사명 추출
            matched_companies = self.extract_companies_from_text(full_text)
            
            if matched_companies:
                # 매칭된 회사가 있는 경우
                for company in matched_companies:
                    result_row = row.copy()
                    result_row['matched_stock_code'] = company['code']
                    result_row['matched_stock_name'] = company['name']
                    result_row['matched_market'] = company['market']
                    result_row['matched_sector'] = company['sector']
                    result_row['match_type'] = company['match_type']
                    result_row['matched_text'] = company['matched_text']
                    result_row['match_confidence'] = company['confidence']
                    matched_results.append(result_row)
            else:
                # 매칭되지 않은 경우
                result_row = row.copy()
                result_row['matched_stock_code'] = None
                result_row['matched_stock_name'] = None
                result_row['matched_market'] = None
                result_row['matched_sector'] = None
                result_row['match_type'] = None
                result_row['matched_text'] = None
                result_row['match_confidence'] = 0.0
                matched_results.append(result_row)
                no_match_count += 1
            
            # 진행률 출력
            if (idx + 1) % 1000 == 0:
                logger.info(f"매칭 진행률: {idx + 1}/{len(news_df)} ({(idx + 1)/len(news_df)*100:.1f}%)")
        
        result_df = pd.DataFrame(matched_results)
        
        # 매칭 통계
        total_news = len(news_df)
        matched_news = len(result_df[result_df['matched_stock_code'].notna()])
        match_rate = (matched_news / len(result_df)) * 100
        
        logger.info(f"매칭 완료:")
        logger.info(f"  - 원본 뉴스: {total_news}개")
        logger.info(f"  - 매칭 결과: {len(result_df)}개 (중복 포함)")
        logger.info(f"  - 매칭된 뉴스: {matched_news}개")
        logger.info(f"  - 매칭 실패: {len(result_df) - matched_news}개")
        logger.info(f"  - 매칭률: {match_rate:.1f}%")
        
        return result_df
    
    def analyze_matching_results(self, matched_df: pd.DataFrame) -> Dict:
        """매칭 결과 분석"""
        analysis = {}
        
        # 1. 기본 통계
        total_rows = len(matched_df)
        matched_rows = len(matched_df[matched_df['matched_stock_code'].notna()])
        
        analysis['basic_stats'] = {
            'total_news': total_rows,
            'matched_news': matched_rows,
            'unmatched_news': total_rows - matched_rows,
            'match_rate': round((matched_rows / total_rows) * 100, 2)
        }
        
        # 2. 종목별 뉴스 수
        if matched_rows > 0:
            stock_counts = matched_df[matched_df['matched_stock_code'].notna()]['matched_stock_name'].value_counts()
            analysis['top_stocks'] = stock_counts.head(10).to_dict()
        
        # 3. 섹터별 뉴스 수
        if matched_rows > 0:
            sector_counts = matched_df[matched_df['matched_stock_code'].notna()]['matched_sector'].value_counts()
            analysis['sector_distribution'] = sector_counts.to_dict()
        
        # 4. 매칭 타입별 분포
        if matched_rows > 0:
            match_type_counts = matched_df[matched_df['matched_stock_code'].notna()]['match_type'].value_counts()
            analysis['match_type_distribution'] = match_type_counts.to_dict()
        
        # 5. 신뢰도 분포
        if matched_rows > 0:
            confidence_stats = matched_df[matched_df['matched_stock_code'].notna()]['match_confidence'].describe()
            analysis['confidence_stats'] = {
                'mean': round(confidence_stats['mean'], 3),
                'std': round(confidence_stats['std'], 3),
                'min': round(confidence_stats['min'], 3),
                'max': round(confidence_stats['max'], 3)
            }
        
        return analysis
    
    def save_matched_results(self, matched_df: pd.DataFrame, filename: str = "news_stock_matched.csv"):
        """매칭 결과 저장"""
        try:
            matched_df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"매칭 결과 저장 완료: {filename}")
            
            # 분석 결과 저장
            analysis = self.analyze_matching_results(matched_df)
            
            analysis_filename = filename.replace('.csv', '_analysis.json')
            with open(analysis_filename, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            # 결과 출력
            self.print_matching_summary(analysis)
            
            return True
            
        except Exception as e:
            logger.error(f"매칭 결과 저장 실패: {e}")
            return False
    
    def print_matching_summary(self, analysis: Dict):
        """매칭 결과 요약 출력"""
        print("\n" + "="*60)
        print("뉴스-종목 매칭 결과 요약")
        print("="*60)
        
        stats = analysis['basic_stats']
        print(f"총 뉴스 수: {stats['total_news']:,}개")
        print(f"매칭된 뉴스: {stats['matched_news']:,}개")
        print(f"매칭 실패: {stats['unmatched_news']:,}개")
        print(f"매칭률: {stats['match_rate']}%")
        
        if 'top_stocks' in analysis:
            print(f"\n뉴스 많은 상위 10개 종목:")
            for i, (stock, count) in enumerate(analysis['top_stocks'].items(), 1):
                print(f"  {i:2d}. {stock}: {count:,}개")
        
        if 'sector_distribution' in analysis:
            print(f"\n섹터별 뉴스 분포:")
            for sector, count in analysis['sector_distribution'].items():
                print(f"  - {sector}: {count:,}개")
        
        if 'match_type_distribution' in analysis:
            print(f"\n매칭 타입별 분포:")
            for match_type, count in analysis['match_type_distribution'].items():
                print(f"  - {match_type}: {count:,}개")
        
        if 'confidence_stats' in analysis:
            print(f"\n매칭 신뢰도 통계:")
            conf = analysis['confidence_stats']
            print(f"  - 평균: {conf['mean']}")
            print(f"  - 표준편차: {conf['std']}")
            print(f"  - 최소: {conf['min']} ~ 최대: {conf['max']}")
        
        print("="*60)

def main():
    """메인 실행 함수"""
    print("뉴스-종목 매칭을 시작합니다.")
    
    # 필요한 파일 확인
    required_files = ["stock_master.csv", "integrated/integrated_news_data.csv"]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"필요한 파일이 없습니다: {file_path}")
            return
    
    try:
        # 1. 매칭기 초기화
        matcher = NewsStockMatcher("stock_master.csv")
        
        # 2. 뉴스 데이터 로드
        logger.info("뉴스 데이터 로드 중...")
        news_df = pd.read_csv("integrated/integrated_news_data.csv", encoding='utf-8-sig')
        
        # 3. 매칭 수행
        matched_df = matcher.match_news_to_stocks(news_df)
        
        # 4. 결과 저장
        success = matcher.save_matched_results(matched_df, "news_stock_matched.csv")
        
        if success:
            print(f"\n매칭 완료! 다음 단계: 주가 데이터 수집")
            print(f"생성된 파일:")
            print(f"  - news_stock_matched.csv: 매칭된 뉴스 데이터")
            print(f"  - news_stock_matched_analysis.json: 매칭 분석 결과")
        
    except Exception as e:
        logger.error(f"매칭 프로세스 실패: {e}")
        print("매칭에 실패했습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    main()