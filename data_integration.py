import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple
import warnings
import platform
import matplotlib.font_manager as fm
from collections import Counter
warnings.filterwarnings('ignore')

# 기본 설정
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsDataIntegrator:
    def __init__(self, data_dir: str = "preprocessed", output_dir: str = "integrated"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.integrated_df = None
        self.quality_report = {}
    
    def load_all_files(self) -> pd.DataFrame:
        """모든 전처리된 CSV 파일을 로드하고 통합"""
        logger.info("전처리된 파일들을 로드 중...")
        
        files = glob(os.path.join(self.data_dir, 'processed_news_*.csv'))
        
        if not files:
            raise FileNotFoundError(f"{self.data_dir} 폴더에 전처리된 파일이 없습니다.")
        
        logger.info(f"총 {len(files)}개 파일 발견")
        
        all_dataframes = []
        file_info = []
        
        for file_path in sorted(files):
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                file_name = os.path.basename(file_path)
                file_info.append({
                    'filename': file_name,
                    'rows': len(df),
                    'date_range': f"{df['date'].min()} ~ {df['date'].max()}" if 'date' in df.columns else 'N/A'
                })
                all_dataframes.append(df)
                logger.info(f"로드 완료: {file_name} ({len(df)}행)")
            except Exception as e:
                logger.error(f"파일 로드 실패: {file_path} - {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("로드된 파일이 없습니다.")
        
        integrated_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"통합 완료: 총 {len(integrated_df)}행")
        self.quality_report['file_info'] = file_info
        
        return integrated_df
    
    def data_quality_check(self, df: pd.DataFrame) -> Dict:
        """데이터 품질 검증"""
        logger.info("데이터 품질 검증 중...")
        
        quality_issues = {
            'total_rows': len(df),
            'missing_values': {},
            'duplicate_analysis': {},
            'text_quality': {},
            'date_issues': {},
            'data_distribution': {}
        }
        
        # 결측값 분석
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df) * 100
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'ratio': round(missing_ratio, 2)
                }
        quality_issues['missing_values'] = missing_info
        
        # 중복 분석
        quality_issues['duplicate_analysis'] = {
            'total_duplicates': int(df.duplicated().sum()),
            'title_duplicates': int(df.duplicated(subset=['title']).sum()),
            'link_duplicates': int(df.duplicated(subset=['link']).sum()),
            'title_date_duplicates': int(df.duplicated(subset=['title', 'date']).sum())
        }
        
        # 텍스트 품질 분석
        if 'body' in df.columns:
            failed_body = df['body'].str.contains('본문 수집 실패|수집 실패', na=False).sum()
            short_body = (df['body'].str.len() < 50).sum()
            empty_body = df['body'].isnull().sum()
            
            quality_issues['text_quality'] = {
                'failed_collection': int(failed_body),
                'failed_collection_ratio': round(failed_body / len(df) * 100, 2),
                'short_body': int(short_body),
                'short_body_ratio': round(short_body / len(df) * 100, 2),
                'empty_body': int(empty_body),
                'avg_text_length': round(df['text_length'].mean(), 2) if 'text_length' in df.columns else 0
            }
        
        # 날짜 관련 이슈
        if 'date' in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df['date'])
                date_range = {
                    'start_date': df['date_parsed'].min().strftime('%Y-%m-%d'),
                    'end_date': df['date_parsed'].max().strftime('%Y-%m-%d'),
                    'total_days': (df['date_parsed'].max() - df['date_parsed'].min()).days + 1,
                    'unique_dates': df['date_parsed'].nunique()
                }
                quality_issues['date_issues'] = date_range
            except Exception as e:
                quality_issues['date_issues'] = {'error': str(e)}
        
        # 데이터 분포 분석
        if 'category' in df.columns:
            quality_issues['data_distribution']['category'] = df['category'].value_counts().to_dict()
        if 'sentiment' in df.columns:
            quality_issues['data_distribution']['sentiment'] = df['sentiment'].value_counts().to_dict()
        
        return quality_issues
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제 (개선된 중복 제거 로직)"""
        logger.info("데이터 정제 중...")
        
        initial_count = len(df)
        
        # 날짜 형식 통일
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # 본문 수집 실패한 뉴스 제거
        if 'body' in df.columns:
            before_clean = len(df)
            df = df[~df['body'].str.contains('본문 수집 실패|수집 실패', na=False)]
            after_clean = len(df)
            logger.info(f"본문 수집 실패 뉴스 제거: {before_clean - after_clean}개")
        
        # 너무 짧은 본문 제거 (50자 미만)
        if 'body' in df.columns:
            before_clean = len(df)
            df = df[df['body'].str.len() >= 50]
            after_clean = len(df)
            logger.info(f"짧은 본문 뉴스 제거: {before_clean - after_clean}개")
        
        # 내용 중복 제거만 수행 (링크 중복 제거 삭제)
        logger.info("내용 기반 중복 제거 중...")
        
        # 본문 90% 이상 유사도 기반 중복 제거
        if len(df) > 1000:
            logger.info("본문 90% 이상 유사도 기반 중복 검사 중...")
            before_content_dedup = len(df)
            
            # 본문 전체의 80% 샘플링으로 유사도 체크
            df['body_long_sample'] = df['body'].str[:int(df['body'].str.len().mean() * 0.8)]
            df['body_normalized'] = df['body_long_sample'].str.replace(r'[^\w가-힣]', '', regex=True)
            
            # 날짜별로 그룹화하여 중복 검사 (같은 날 뉴스만 비교)
            df = df.drop_duplicates(subset=['body_normalized', 'date'], keep='first')
            content_removed = before_content_dedup - len(df)
            logger.info(f"본문 유사 중복 제거: {content_removed}개")
            
            df = df.drop(['body_long_sample', 'body_normalized'], axis=1)
        else:
            content_removed = 0
        
        # 필수 컬럼 결측값 제거
        essential_columns = ['title', 'body', 'date']
        for col in essential_columns:
            if col in df.columns:
                before_na = len(df)
                df = df.dropna(subset=[col])
                after_na = len(df)
                if before_na != after_na:
                    logger.info(f"{col} 결측값 제거: {before_na - after_na}개")
        
        df = df.reset_index(drop=True)
        
        final_count = len(df)
        total_removed = initial_count - final_count
        removal_rate = (total_removed / initial_count) * 100
        
        logger.info(f"데이터 정제 완료: {initial_count:,}개 → {final_count:,}개")
        logger.info(f"총 제거: {total_removed:,}개 ({removal_rate:.1f}%)")
        logger.info(f"  - 본문 유사 중복: {content_removed:,}개")
        
        return df
    
    def basic_eda(self, df: pd.DataFrame) -> Dict:
        """기본 탐색적 데이터 분석"""
        logger.info("기본 EDA 수행 중...")
        
        eda_results = {}
        
        # 기본 통계
        eda_results['basic_stats'] = {
            'total_news': len(df),
            'date_range': f"{df['date'].min()} ~ {df['date'].max()}",
            'categories': df['category'].nunique() if 'category' in df.columns else 0,
            'avg_text_length': round(df['text_length'].mean(), 2) if 'text_length' in df.columns else 0
        }
        
        # 일별 뉴스 수 통계
        if 'date' in df.columns:
            daily_count = df.groupby('date').size()
            eda_results['daily_stats'] = {
                'avg_daily_news': round(daily_count.mean(), 2),
                'max_daily_news': int(daily_count.max()),
                'min_daily_news': int(daily_count.min()),
                'most_active_date': daily_count.idxmax(),
                'least_active_date': daily_count.idxmin()
            }
        
        # 카테고리별 분석
        if 'category' in df.columns:
            category_stats = df['category'].value_counts()
            eda_results['category_analysis'] = {
                'distribution': category_stats.to_dict(),
                'most_common': category_stats.index[0],
                'least_common': category_stats.index[-1]
            }
        
        # 감성 분석 결과
        if 'sentiment' in df.columns:
            sentiment_stats = df['sentiment'].value_counts()
            eda_results['sentiment_analysis'] = {
                'distribution': sentiment_stats.to_dict(),
                'positive_ratio': round(sentiment_stats.get('POSITIVE', 0) / len(df) * 100, 2),
                'negative_ratio': round(sentiment_stats.get('NEGATIVE', 0) / len(df) * 100, 2),
                'neutral_ratio': round(sentiment_stats.get('NEUTRAL', 0) / len(df) * 100, 2)
            }
        
        # 이벤트 분석
        if 'event_count' in df.columns:
            eda_results['event_analysis'] = {
                'avg_events_per_news': round(df['event_count'].mean(), 2),
                'max_events': int(df['event_count'].max()),
                'news_with_events': int((df['event_count'] > 0).sum()),
                'events_ratio': round((df['event_count'] > 0).sum() / len(df) * 100, 2)
            }
        
        # 회사명 추출 분석 (개선된 버전)
        if 'companies' in df.columns:
            company_counts = []
            extracted_companies = []
            
            for companies_str in df['companies']:
                try:
                    companies_list = json.loads(companies_str) if companies_str else []
                    company_counts.append(len(companies_list))
                    extracted_companies.extend(companies_list)
                except:
                    company_counts.append(0)
            
            company_freq = Counter(extracted_companies)
            top_companies = dict(company_freq.most_common(10))
            
            eda_results['company_analysis'] = {
                'avg_companies_per_news': round(np.mean(company_counts), 2),
                'news_with_companies': int(sum(1 for c in company_counts if c > 0)),
                'company_extraction_ratio': round(sum(1 for c in company_counts if c > 0) / len(df) * 100, 2),
                'total_unique_companies': len(company_freq),
                'top_companies': top_companies
            }
        
        return eda_results
    
    def create_visualizations(self, df: pd.DataFrame):
        """데이터 시각화 (한글/영어 버전 모두 생성)"""
        logger.info("시각화 생성 중...")
        
        plt.style.use('default')
        
        # 한글/영어 라벨 매핑
        category_eng = {
            '실시간 속보': 'Breaking News',
            '기업·종목분석': 'Company Analysis', 
            '시황·전망': 'Market Outlook',
            '해외증시': 'Global Markets',
            '공시·메모': 'Disclosure',
            '채권·선물': 'Bonds & Futures',
            '환율': 'Exchange Rate'
        }
        
        sentiment_eng = {
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative', 
            'NEUTRAL': 'Neutral'
        }
        
        # 1. 일별 뉴스 수 트렌드 (한글/영어)
        if 'date' in df.columns:
            daily_count = df.groupby('date').size()
            
            for lang, title, xlabel, ylabel in [
                ('kor', '일별 뉴스 발생 수 트렌드', '날짜', '뉴스 수'),
                ('eng', 'Daily News Count Trend', 'Date', 'News Count')
            ]:
                plt.figure(figsize=(15, 6))
                daily_count.plot(kind='line', linewidth=2)
                plt.title(title, fontsize=16, fontweight='bold')
                plt.xlabel(xlabel, fontsize=12)
                plt.ylabel(ylabel, fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'daily_news_trend_{lang}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. 카테고리별 분포 (글자 겹침 방지)
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            
            for lang in ['kor', 'eng']:
                plt.figure(figsize=(14, 10))  # 크기 증가
                
                if lang == 'eng':
                    labels = [category_eng.get(cat, cat) for cat in category_counts.index]
                    
                    plt.subplot(2, 1, 1)
                    bars = plt.bar(range(len(category_counts)), category_counts.values, color='skyblue', edgecolor='black')
                    plt.xticks(range(len(category_counts)), labels, rotation=45, ha='right')  # ha='right' 추가
                    plt.title('News Count by Category', fontsize=14, fontweight='bold', pad=20)
                    plt.xlabel('Category', fontsize=12)
                    plt.ylabel('News Count', fontsize=12)
                    plt.subplots_adjust(bottom=0.2)  # 하단 여백 증가
                    
                    # 막대 위에 숫자 표시
                    for i, v in enumerate(category_counts.values):
                        plt.text(i, v + max(category_counts.values)*0.01, f'{v:,}', ha='center', fontsize=10)
                    
                    plt.subplot(2, 1, 2)
                    plt.pie(category_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
                    plt.title('Category Distribution', fontsize=14, fontweight='bold')
                else:
                    plt.subplot(2, 1, 1)
                    bars = plt.bar(range(len(category_counts)), category_counts.values, color='skyblue', edgecolor='black')
                    plt.xticks(range(len(category_counts)), category_counts.index, rotation=45, ha='right')
                    plt.title('Category News Count', fontsize=14, fontweight='bold', pad=20)
                    plt.xlabel('Category', fontsize=12)
                    plt.ylabel('News Count', fontsize=12)
                    plt.subplots_adjust(bottom=0.2)
                    
                    # 막대 위에 숫자 표시
                    for i, v in enumerate(category_counts.values):
                        plt.text(i, v + max(category_counts.values)*0.01, f'{v:,}', ha='center', fontsize=10)
                    
                    plt.subplot(2, 1, 2)
                    plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
                    plt.title('Category Distribution', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'category_distribution_{lang}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. 감성 분석 결과 (한글/영어)
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            
            for lang in ['kor', 'eng']:
                plt.figure(figsize=(10, 6))
                
                if lang == 'eng':
                    labels = [sentiment_eng.get(sent, sent) for sent in sentiment_counts.index]
                    colors = ['green' if 'Positive' in l else 'red' if 'Negative' in l else 'gray' for l in labels]
                    plt.bar(range(len(sentiment_counts)), sentiment_counts.values, color=colors, edgecolor='black')
                    plt.xticks(range(len(sentiment_counts)), labels)
                    plt.title('Sentiment Analysis Distribution', fontsize=14, fontweight='bold')
                    plt.xlabel('Sentiment', fontsize=12)
                    plt.ylabel('News Count', fontsize=12)
                else:
                    colors = ['green' if 'POSITIVE' in s else 'red' if 'NEGATIVE' in s else 'gray' for s in sentiment_counts.index]
                    sentiment_counts.plot(kind='bar', color=colors, edgecolor='black')
                    plt.title('감성 분석 결과 분포', fontsize=14, fontweight='bold')
                    plt.xlabel('감성', fontsize=12)
                    plt.ylabel('뉴스 수', fontsize=12)
                    plt.xticks(rotation=0)
                
                total = len(df)
                for i, v in enumerate(sentiment_counts.values):
                    plt.text(i, v + total*0.01, f'{v/total*100:.1f}%', ha='center', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'sentiment_distribution_{lang}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info("시각화 완료 (한글/영어 버전)")
    
    def save_results(self, df: pd.DataFrame, quality_report: Dict, eda_results: Dict):
        """결과 저장"""
        logger.info("결과 저장 중...")
        
        # 통합된 데이터 저장
        output_file = os.path.join(self.output_dir, 'integrated_news_data.csv')
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"통합 데이터 저장: {output_file} ({len(df)}행)")
        
        # 품질 보고서 저장
        quality_file = os.path.join(self.output_dir, 'data_quality_report.json')
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        # EDA 결과 저장
        eda_file = os.path.join(self.output_dir, 'eda_results.json')
        with open(eda_file, 'w', encoding='utf-8') as f:
            json.dump(eda_results, f, ensure_ascii=False, indent=2)
        
        # 요약 보고서 생성
        self.generate_summary_report(df, quality_report, eda_results)
    
    def generate_summary_report(self, df: pd.DataFrame, quality_report: Dict, eda_results: Dict):
        """요약 보고서 생성"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("뉴스 데이터 통합 및 품질 검증 보고서")
        report_lines.append("=" * 60)
        report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 기본 정보
        report_lines.append("【 기본 정보 】")
        report_lines.append(f"• 총 뉴스 수: {len(df):,}개")
        if 'basic_stats' in eda_results:
            stats = eda_results['basic_stats']
            report_lines.append(f"• 기간: {stats['date_range']}")
            report_lines.append(f"• 카테고리 수: {stats['categories']}개")
            report_lines.append(f"• 평균 텍스트 길이: {stats['avg_text_length']:,.0f}자")
        report_lines.append("")
        
        # 회사명 추출 TOP 10
        if 'company_analysis' in eda_results and 'top_companies' in eda_results['company_analysis']:
            report_lines.append("【 추출된 주요 회사명 TOP 10 】")
            for company, count in eda_results['company_analysis']['top_companies'].items():
                report_lines.append(f"• {company}: {count}회")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        # 보고서 저장
        report_file = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
    
    def run_integration(self):
        """전체 통합 프로세스 실행"""
        try:
            df = self.load_all_files()
            quality_report = self.data_quality_check(df)
            df_cleaned = self.clean_data(df)
            eda_results = self.basic_eda(df_cleaned)
            self.create_visualizations(df_cleaned)
            self.save_results(df_cleaned, quality_report, eda_results)
            self.integrated_df = df_cleaned
            logger.info("데이터 통합 프로세스 완료!")
            return df_cleaned
        except Exception as e:
            logger.error(f"통합 프로세스 실패: {e}")
            raise

def main():
    integrator = NewsDataIntegrator(data_dir="preprocessed", output_dir="integrated")
    integrated_data = integrator.run_integration()
    print(f"\n통합 완료! 최종 데이터: {len(integrated_data)}행")

if __name__ == "__main__":
    main()