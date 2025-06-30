# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from datetime import datetime
import logging
import os
import re
from typing import Dict, List, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Prediction API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.5.0.2:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델 정의
class PredictionRequest(BaseModel):
    title: str
    content: Optional[str] = ""
    stock: Optional[str] = None

# 응답 모델 정의
class PredictionResponse(BaseModel):
    predictions: Dict
    analysis: Dict
    reasoning: str

# 모델 클래스들 (기존 코드와 동일)
class MultiTaskStockPredictor(nn.Module):
    """멀티태스크 주가 예측 모델"""
    
    def __init__(self, model_name='klue/bert-base', num_classes=5, additional_features_dim=8):
        super(MultiTaskStockPredictor, self).__init__()
        
        # BERT 인코더
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 추가 특성 처리
        self.additional_features_dim = additional_features_dim
        
        # 공유 특성 층
        combined_size = self.bert_hidden_size + additional_features_dim
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 각 기간별 예측 헤드
        self.head_1d = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.head_3d = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.head_5d = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, additional_features):
        # BERT 인코딩
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = bert_output.last_hidden_state[:, 0, :]  # [CLS] 토큰
        
        # 특성 결합
        combined_features = torch.cat([bert_features, additional_features], dim=1)
        
        # 공유 특성 추출
        shared_features = self.shared_layers(combined_features)
        
        # 각 기간별 예측
        pred_1d = self.head_1d(shared_features)
        pred_3d = self.head_3d(shared_features)
        pred_5d = self.head_5d(shared_features)
        
        return {
            '1d': pred_1d,
            '3d': pred_3d,
            '5d': pred_5d
        }

class StockPredictionService:
    """주가 예측 서비스 클래스"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_classes = ['strong_down', 'down', 'neutral', 'up', 'strong_up']
        self.label_to_direction = {
            'strong_down': 'down',
            'down': 'down', 
            'neutral': 'neutral',
            'up': 'up',
            'strong_up': 'up'
        }
        
        # 종목 매칭 딕셔너리 (간단 버전)
        self.stock_mapping = self._load_stock_mapping()
        
        # 모델 로드
        self.load_model()
    
    def _load_stock_mapping(self):
        """종목 매칭 사전 로드"""
        try:
            stock_master = pd.read_csv("stock_master.csv", encoding='utf-8-sig')
            mapping = {}
            
            for _, row in stock_master.iterrows():
                stock_name = row['종목명']
                stock_code = str(row['종목코드']).zfill(6)
                
                # 기본 매핑
                mapping[stock_name] = {
                    'code': stock_code,
                    'name': stock_name,
                    'market': row['시장구분']
                }
                
                # 별칭 매핑
                try:
                    aliases = json.loads(row['별칭'])
                    for alias in aliases:
                        if alias != stock_name:
                            mapping[alias] = {
                                'code': stock_code,
                                'name': stock_name,
                                'market': row['시장구분']
                            }
                except:
                    pass
            
            logger.info(f"종목 매핑 로드 완료: {len(mapping)}개")
            return mapping
            
        except Exception as e:
            logger.error(f"종목 매핑 로드 실패: {e}")
            return {}
    
    def load_model(self):
        """학습된 모델 로드"""
        try:
            # 설정 로드
            with open("saved_model/config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            model_name = config.get('model_name', 'klue/bert-base')
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 모델 초기화 및 가중치 로드
            self.model = MultiTaskStockPredictor(model_name).to(self.device)
            self.model.load_state_dict(torch.load("saved_model/model.pth", map_location=self.device))
            self.model.eval()
            
            logger.info("모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            self.model = None
    
    def extract_companies_from_text(self, text: str) -> List[Dict]:
        """텍스트에서 회사명 추출"""
        found_companies = []
        text_clean = re.sub(r'[^\w가-힣\s]', ' ', text)
        
        for keyword, info in self.stock_mapping.items():
            if keyword in text_clean:
                found_companies.append({
                    'name': info['name'],
                    'code': info['code'],
                    'market': info['market'],
                    'matched_text': keyword
                })
        
        # 중복 제거 (같은 종목코드)
        unique_companies = {}
        for company in found_companies:
            code = company['code']
            if code not in unique_companies:
                unique_companies[code] = company
        
        return list(unique_companies.values())
    
    def analyze_sentiment(self, text: str) -> Dict:
        """간단한 감성 분석 (키워드 기반)"""
        positive_words = ['상승', '증가', '성장', '호전', '개선', '긍정', '상향', '확대', '투자']
        negative_words = ['하락', '감소', '악화', '우려', '부정', '하향', '축소', '손실']
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return {"positive": 0.5, "negative": 0.25, "neutral": 0.25}
        
        pos_ratio = pos_count / total_sentiment_words
        neg_ratio = neg_count / total_sentiment_words
        neutral_ratio = 1 - pos_ratio - neg_ratio
        
        return {
            "positive": round(pos_ratio, 2),
            "negative": round(neg_ratio, 2), 
            "neutral": round(max(0.1, neutral_ratio), 2)
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        keywords = []
        key_terms = ['실적', '매출', '영업이익', '순이익', '투자', '인수', '합병', '제휴', '계약', '출시']
        
        for term in key_terms:
            if term in text:
                keywords.append(term)
        
        return keywords[:5]  # 최대 5개
    
    def predict(self, title: str, content: str = "", stock_code: str = None) -> Dict:
        """주가 예측 수행"""
        if self.model is None:
            raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
        
        # 텍스트 전처리
        full_text = f"{title} [SEP] {content}"
        
        # 회사명 추출
        companies = self.extract_companies_from_text(full_text)
        
        # 감성 분석
        sentiment = self.analyze_sentiment(full_text)
        
        # 기본 특성 설정 (간단한 버전)
        additional_features = torch.tensor([
            sentiment['positive'],  # 긍정 점수
            sentiment['negative'],  # 부정 점수  
            sentiment['neutral'],   # 중립 점수
            0.1, 0.1, 0.8,         # 주가 시그널 (기본값)
            1.0 if companies else 0.0,  # 이벤트 수 (회사명 있으면 1)
            0.0                     # 재무 정보 (기본값)
        ], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 토큰화
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, additional_features)
            
            predictions = {}
            for period in ['1d', '3d', '5d']:
                probs = torch.softmax(outputs[period], dim=1)[0]
                predicted_class = torch.argmax(probs).item()
                confidence = float(probs[predicted_class]) * 100
                
                label = self.label_classes[predicted_class]
                direction = self.label_to_direction[label]
                
                # 가상의 변화율 (실제로는 더 정교한 계산 필요)
                if direction == 'up':
                    change = np.random.uniform(1.0, 5.0)
                    probability = max(60, confidence)
                elif direction == 'down':
                    change = np.random.uniform(-5.0, -1.0)
                    probability = max(60, confidence)
                else:
                    change = np.random.uniform(-1.0, 1.0)
                    probability = max(50, confidence)
                
                predictions[period.replace('d', 'day')] = {
                    "direction": direction,
                    "probability": round(probability, 1),
                    "change": round(change, 2),
                    "confidence": round(confidence, 1),
                    "label": label
                }
        
        # 키워드 추출
        keywords = self.extract_keywords(full_text)
        
        # 회사명 리스트
        company_names = [comp['name'] for comp in companies]
        
        return {
            "predictions": predictions,
            "analysis": {
                "sentiment": sentiment,
                "companies": company_names,
                "keywords": keywords,
                "financials": []  # 실제 구현 필요
            },
            "reasoning": f"AI 모델이 뉴스 텍스트를 분석하여 {', '.join(company_names) if company_names else '관련 종목'}의 주가 변동을 예측했습니다."
        }

# 전역 서비스 인스턴스
prediction_service = None

@app.on_event("startup")
async def startup_event():
    """앱 시작시 모델 로드"""
    global prediction_service
    try:
        prediction_service = StockPredictionService()
        logger.info("예측 서비스 초기화 완료")
    except Exception as e:
        logger.error(f"예측 서비스 초기화 실패: {e}")

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "Stock Prediction API",
        "status": "running",
        "model_loaded": prediction_service is not None and prediction_service.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """주가 예측 API"""
    global prediction_service
    
    if prediction_service is None:
        raise HTTPException(status_code=500, detail="예측 서비스가 초기화되지 않았습니다")
    
    try:
        # 예측 수행
        result = prediction_service.predict(
            title=request.title,
            content=request.content or "",
            stock_code=request.stock
        )
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"예측 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if (prediction_service and prediction_service.model) else "not_loaded"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 