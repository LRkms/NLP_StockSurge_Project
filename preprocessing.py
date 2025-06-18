import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from konlpy.tag import Mecab
from transformers import pipeline
import re
import logging
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 설정 ---
data_dir = "data"
output_dir = "preprocessed"
os.makedirs(output_dir, exist_ok=True)

class NewsPreprocessor:
    def __init__(self):
        # 형태소 분석기 초기화
        try:
            self.mecab = Mecab('C:/mecab/share/mecab-ko-dic')
        except Exception as e:
            logger.warning(f"Mecab 초기화 실패: {e}")
            self.mecab = None
        
        # 감성 분석 모델 초기화
        try:
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="snunlp/KR-FinBert-SC",
                tokenizer="snunlp/KR-FinBert-SC",
                device=-1
                # return_all_scores 옵션 제거 (모델에 따라 지원하지 않을 수 있음)
            )
            logger.info("감성 분석 모델 로드 완료")
        except Exception as e:
            logger.error(f"감성 분석 모델 로드 실패: {e}")
            # 대안 모델 시도
            try:
                logger.info("대안 감성 분석 모델 시도...")
                self.sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model="beomi/KcELECTRA-base-v2022",
                    device=-1
                )
                logger.info("대안 감성 분석 모델 로드 완료")
            except Exception as e2:
                logger.error(f"대안 모델도 로드 실패: {e2}")
                self.sentiment_pipe = None
        
        # 개체명 인식 모델 초기화
        try:
            self.ner_pipe = pipeline(
                "ner",
                model="monologg/koelectra-base-v3-finetuned-kor-ned",
                tokenizer="monologg/koelectra-base-v3-finetuned-kor-ned",
                grouped_entities=True,
                device=-1
            )
        except Exception as e:
            logger.error(f"NER 모델 로드 실패: {e}")
            self.ner_pipe = None
        
        # 확장된 이벤트 키워드 사전
        self.event_keywords = {
            "인수합병": ["인수", "합병", "M&A", "통합", "합작", "지분매각", "자회사"],
            "실적발표": ["실적", "분기실적", "연간실적", "매출", "영업이익", "순이익", "어닝", "실적발표"],
            "투자": ["투자", "시설투자", "연구개발", "R&D", "증설", "신규투자", "지분투자"],
            "임원변경": ["CEO", "사장", "회장", "임원", "교체", "사임", "선임", "연임"],
            "규제정책": ["규제", "정부정책", "법안", "제재", "승인", "허가", "금지"],
            "신제품출시": ["신제품", "출시", "런칭", "상품화", "서비스개시"],
            "파트너십": ["제휴", "협력", "계약", "MOU", "업무협약", "전략적제휴"],
            "소송분쟁": ["소송", "특허분쟁", "법정분쟁", "배상", "손해배상"],
            "배당": ["배당", "배당금", "주주환원", "자사주매입", "주식분할"],
            "재무건전성": ["부채", "차입", "신용등급", "회사채", "유상증자", "무상증자"]
        }
        
        # 주가 관련 키워드 (강도별 분류)
        self.price_keywords = {
            "상승": {
                "강": ["급등", "폭등", "상한가", "연고점", "신고가", "급상승"],
                "중": ["상승", "오름", "반등", "회복", "강세"],
                "약": ["소폭상승", "미미한오름", "약간상승"]
            },
            "하락": {
                "강": ["급락", "폭락", "하한가", "신저가", "대폭하락"],
                "중": ["하락", "내림", "약세", "부진"],
                "약": ["소폭하락", "미미한하락", "약간하락"]
            },
            "보합": ["횡보", "보합", "변동없음", "등락없음"]
        }
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text or pd.isna(text):
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 불필요한 문자 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 광고성 문구 제거
        ad_patterns = [
            r'.*?기자\s*=\s*',
            r'.*?@.*?\.com',
            r'저작권.*?',
            r'무단.*?금지',
            r'Copyright.*?',
            r'ⓒ.*?',
            r'©.*?'
        ]
        
        for pattern in ad_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_company_names(self, text: str) -> List[str]:
        """회사명 추출 (정규식 기반)"""
        # 회사명 패턴
        company_patterns = [
            r'[가-힣A-Za-z0-9]+(?:주식회사|㈜)',
            r'[가-힣A-Za-z0-9]+(?:그룹|홀딩스)',
            r'[가-힣A-Za-z0-9]+(?:코퍼레이션|Corp)',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+Inc|\s+Corp|\s+Ltd)',
        ]
        
        companies = set()
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.update(matches)
        
        return list(companies)
    
    def analyze_sentiment(self, text: str) -> Dict:
        """감성 분석 (개선된 버전)"""
        if not self.sentiment_pipe or not text:
            return {"sentiment": "NEUTRAL", "positive_score": 0.0, "negative_score": 0.0, "neutral_score": 1.0}
        
        try:
            # 텍스트가 너무 길면 앞부분만 사용
            text_sample = text[:512]
            results = self.sentiment_pipe(text_sample)
            
            # 결과 구조 확인 및 처리
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # return_all_scores=True인 경우: [[{label: score}, {label: score}]]
                    score_list = results[0]
                    scores = {item['label']: item['score'] for item in score_list}
                else:
                    # return_all_scores=False인 경우: [{label: score}]
                    if 'label' in results[0] and 'score' in results[0]:
                        # 단일 결과
                        main_result = results[0]
                        scores = {main_result['label']: main_result['score']}
                        # 다른 감성에 대해서는 기본값 설정
                        if main_result['label'] == 'positive':
                            scores['negative'] = 1 - main_result['score']
                            scores['neutral'] = 0.0
                        elif main_result['label'] == 'negative':
                            scores['positive'] = 1 - main_result['score']
                            scores['neutral'] = 0.0
                        else:
                            scores['positive'] = 0.0
                            scores['negative'] = 0.0
                            scores['neutral'] = main_result['score']
                    else:
                        # 예상치 못한 구조
                        scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            else:
                scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            
            # 라벨 이름 정규화 (모델에 따라 다를 수 있음)
            normalized_scores = {}
            for label, score in scores.items():
                if label.lower() in ['positive', 'pos', '긍정']:
                    normalized_scores['positive'] = score
                elif label.lower() in ['negative', 'neg', '부정']:
                    normalized_scores['negative'] = score
                elif label.lower() in ['neutral', 'neu', '중립']:
                    normalized_scores['neutral'] = score
                else:
                    # 알 수 없는 라벨의 경우 중립으로 처리
                    normalized_scores['neutral'] = score
            
            # 기본값 설정
            normalized_scores.setdefault('positive', 0.0)
            normalized_scores.setdefault('negative', 0.0)
            normalized_scores.setdefault('neutral', 1.0)
            
            # 주요 감성 결정
            main_sentiment = max(normalized_scores, key=normalized_scores.get)
            
            return {
                "sentiment": main_sentiment.upper(),
                "positive_score": normalized_scores['positive'],
                "negative_score": normalized_scores['negative'],
                "neutral_score": normalized_scores['neutral']
            }
            
        except Exception as e:
            logger.warning(f"감성 분석 실패: {e}")
            logger.debug(f"감성 분석 결과 구조: {results if 'results' in locals() else 'N/A'}")
            return {"sentiment": "NEUTRAL", "positive_score": 0.0, "negative_score": 0.0, "neutral_score": 1.0}
    
    def extract_entities(self, text: str) -> List[Dict]:
        """개체명 인식"""
        if not self.ner_pipe or not text:
            return []
        
        try:
            text_sample = text[:512]
            entities = self.ner_pipe(text_sample)
            
            # 결과 정리
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    "entity_type": entity.get('entity_group', 'MISC'),
                    "text": entity.get('word', ''),
                    "confidence": entity.get('score', 0.0)
                })
            
            return processed_entities
        except Exception as e:
            logger.warning(f"개체명 인식 실패: {e}")
            return []
    
    def tag_events(self, text: str) -> List[Dict]:
        """이벤트 태깅 (개선된 버전)"""
        events = []
        text_lower = text.lower()
        
        for event_type, keywords in self.event_keywords.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                events.append({
                    "event_type": event_type,
                    "matched_keywords": matched_keywords,
                    "relevance_score": len(matched_keywords) / len(keywords)
                })
        
        return events
    
    def analyze_price_sentiment(self, text: str) -> Dict:
        """주가 관련 감성 분석"""
        text_lower = text.lower()
        price_signals = {"상승": 0, "하락": 0, "보합": 0}
        
        # 상승/하락 키워드 점수 계산
        for direction, intensity_dict in self.price_keywords.items():
            if direction == "보합":
                for keyword in intensity_dict:
                    if keyword in text_lower:
                        price_signals[direction] += 1
            else:
                for intensity, keywords in intensity_dict.items():
                    weight = {"강": 3, "중": 2, "약": 1}[intensity]
                    for keyword in keywords:
                        if keyword in text_lower:
                            price_signals[direction] += weight
        
        # 정규화
        total_signals = sum(price_signals.values())
        if total_signals > 0:
            price_signals = {k: v/total_signals for k, v in price_signals.items()}
        
        return price_signals
    
    def extract_financial_numbers(self, text: str) -> List[Dict]:
        """재무 수치 추출"""
        financial_patterns = [
            (r'매출.*?([0-9,]+(?:\.[0-9]+)?)\s*(?:억|조|만|원)', '매출'),
            (r'영업이익.*?([0-9,]+(?:\.[0-9]+)?)\s*(?:억|조|만|원)', '영업이익'),
            (r'순이익.*?([0-9,]+(?:\.[0-9]+)?)\s*(?:억|조|만|원)', '순이익'),
            (r'시가총액.*?([0-9,]+(?:\.[0-9]+)?)\s*(?:억|조|만|원)', '시가총액'),
            (r'주가.*?([0-9,]+(?:\.[0-9]+)?)\s*원', '주가')
        ]
        
        financial_data = []
        for pattern, data_type in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # 숫자 정리 (쉼표 제거)
                    value = float(match.replace(',', ''))
                    financial_data.append({
                        "type": data_type,
                        "value": value,
                        "raw_text": match
                    })
                except ValueError:
                    continue
        
        return financial_data
    
    def calculate_text_features(self, text: str) -> Dict:
        """텍스트 특성 계산"""
        if not text:
            return {"length": 0, "word_count": 0, "sentence_count": 0}
        
        sentences = re.split(r'[.!?]', text)
        words = text.split()
        
        return {
            "length": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()])
        }
    
    def process_single_news(self, row: pd.Series) -> Dict:
        """단일 뉴스 처리"""
        body = self.clean_text(row.get('body', ''))
        title = self.clean_text(row.get('title', ''))
        
        # 제목과 본문 합치기
        full_text = f"{title} {body}"
        
        # 각종 분석 수행
        sentiment_analysis = self.analyze_sentiment(full_text)
        entities = self.extract_entities(full_text)
        events = self.tag_events(full_text)
        price_sentiment = self.analyze_price_sentiment(full_text)
        companies = self.extract_company_names(full_text)
        financial_numbers = self.extract_financial_numbers(full_text)
        text_features = self.calculate_text_features(body)
        
        # 결과 통합
        processed_data = {
            **row.to_dict(),
            'cleaned_title': title,
            'cleaned_body': body,
            'full_text': full_text,
            
            # 감성 분석
            'sentiment': sentiment_analysis['sentiment'],
            'positive_score': sentiment_analysis['positive_score'],
            'negative_score': sentiment_analysis['negative_score'],
            'neutral_score': sentiment_analysis['neutral_score'],
            
            # 주가 감성
            'price_up_signal': price_sentiment['상승'],
            'price_down_signal': price_sentiment['하락'],
            'price_neutral_signal': price_sentiment['보합'],
            
            # 개체명 및 회사명
            'entities': json.dumps(entities, ensure_ascii=False),
            'companies': json.dumps(companies, ensure_ascii=False),
            
            # 이벤트
            'events': json.dumps(events, ensure_ascii=False),
            'event_count': len(events),
            
            # 재무 정보
            'financial_numbers': json.dumps(financial_numbers, ensure_ascii=False),
            'financial_count': len(financial_numbers),
            
            # 텍스트 특성
            'text_length': text_features['length'],
            'word_count': text_features['word_count'],
            'sentence_count': text_features['sentence_count'],
            
            # 처리 시간
            'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return processed_data
    
    def preprocess_file(self, filepath: str):
        """파일 전처리"""
        logger.info(f"처리 시작: {filepath}")
        
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            logger.info(f"총 {len(df)}개 뉴스 로드")
            
            # 빈 본문 필터링
            df = df[df['body'].notna() & (df['body'] != '본문 수집 실패')]
            logger.info(f"유효한 뉴스: {len(df)}개")
            
            if len(df) == 0:
                logger.warning("처리할 뉴스가 없습니다.")
                return
            
            processed_records = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="뉴스 처리"):
                try:
                    processed_record = self.process_single_news(row)
                    processed_records.append(processed_record)
                except Exception as e:
                    logger.error(f"뉴스 처리 실패 (행 {idx}): {e}")
                    continue
            
            # 결과 저장
            if processed_records:
                output_df = pd.DataFrame(processed_records)
                
                # 파일명 생성
                base_name = os.path.basename(filepath)
                output_name = base_name.replace('naver_news', 'processed_news')
                output_path = os.path.join(output_dir, output_name)
                
                # CSV 저장
                output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                logger.info(f"전처리 완료: {len(output_df)}개 뉴스 저장 -> {output_path}")
                
                # 통계 출력
                self.print_statistics(output_df)
            else:
                logger.warning("처리된 뉴스가 없습니다.")
                
        except Exception as e:
            logger.error(f"파일 처리 실패: {e}")
    
    def print_statistics(self, df: pd.DataFrame):
        """처리 결과 통계 출력"""
        logger.info("=== 전처리 통계 ===")
        logger.info(f"총 뉴스 수: {len(df)}")
        
        # 감성 분포
        sentiment_dist = df['sentiment'].value_counts()
        logger.info(f"감성 분포: {dict(sentiment_dist)}")
        
        # 카테고리별 분포
        if 'category' in df.columns:
            category_dist = df['category'].value_counts()
            logger.info(f"카테고리 분포: {dict(category_dist)}")
        
        # 이벤트 통계
        avg_events = df['event_count'].mean()
        logger.info(f"평균 이벤트 수: {avg_events:.2f}")
        
        # 텍스트 길이 통계
        avg_length = df['text_length'].mean()
        logger.info(f"평균 텍스트 길이: {avg_length:.0f}자")

def main():
    """메인 실행 함수"""
    preprocessor = NewsPreprocessor()
    
    # 크롤링된 파일들 찾기
    files = glob(os.path.join(data_dir, 'naver_news_*.csv'))
    
    if not files:
        logger.error(f"{data_dir} 폴더에 크롤링 파일이 없습니다.")
        return
    
    logger.info(f"총 {len(files)}개 파일 발견")
    
    # 각 파일 처리
    for filepath in files:
        try:
            preprocessor.preprocess_file(filepath)
        except Exception as e:
            logger.error(f"파일 처리 실패 {filepath}: {e}")
            continue
    
    logger.info("모든 파일 전처리 완료!")

if __name__ == '__main__':
    main()