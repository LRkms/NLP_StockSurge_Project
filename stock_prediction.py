import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # torch에서 직접 임포트
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import json
from datetime import datetime
import logging

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"사용 디바이스: {device}")

class NewsStockDataset(Dataset):
    """뉴스-주가 데이터셋 클래스"""
    
    def __init__(self, news_data, tokenizer, max_length=512, mode='train'):
        self.news_data = news_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # 라벨 인코딩 (unknown 제외)
        self.label_encoder = LabelEncoder()
        all_labels = ['strong_down', 'down', 'neutral', 'up', 'strong_up']
        self.label_encoder.fit(all_labels)
        
        logger.info(f"데이터셋 생성: {len(self.news_data)}개 샘플")
        logger.info(f"라벨 매핑: {dict(zip(all_labels, range(len(all_labels))))}")
        
        # 라벨 분포 및 유효성 검사
        for period in ['1d', '3d', '5d']:
            label_col = f'label_{period}'
            if label_col in self.news_data.columns:
                unique_labels = self.news_data[label_col].unique()
                logger.info(f"{period} 라벨 종류: {unique_labels}")
                
                # 각 라벨별 개수
                label_counts = self.news_data[label_col].value_counts()
                logger.info(f"{period} 라벨 분포:")
                for label, count in label_counts.items():
                    logger.info(f"  {label}: {count}개")
                
                # 유효하지 않은 라벨 확인
                invalid_labels = [label for label in unique_labels if label not in all_labels and pd.notna(label)]
                if invalid_labels:
                    logger.warning(f"{period} 유효하지 않은 라벨: {invalid_labels}")
        
    def __len__(self):
        return len(self.news_data)
    
    def __getitem__(self, idx):
        row = self.news_data.iloc[idx]
        
        # 텍스트 전처리
        title = str(row.get('cleaned_title', ''))
        body = str(row.get('cleaned_body', ''))
        text = f"{title} [SEP] {body}"
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 추가 특성
        additional_features = torch.tensor([
            float(row.get('positive_score', 0)),
            float(row.get('negative_score', 0)),
            float(row.get('neutral_score', 0)),
            float(row.get('price_up_signal', 0)),
            float(row.get('price_down_signal', 0)),
            float(row.get('price_neutral_signal', 0)),
            float(row.get('event_count', 0)),
            float(row.get('financial_count', 0))
        ], dtype=torch.float32)
        
        # 라벨 (1일, 3일, 5일) - 범위 검증 추가
        labels = {}
        valid_labels = ['strong_down', 'down', 'neutral', 'up', 'strong_up']
        
        for period in ['1d', '3d', '5d']:
            label_col = f'label_{period}'
            if label_col in row and pd.notna(row[label_col]):
                label_value = row[label_col]
                # 유효한 라벨인지 확인
                if label_value in valid_labels:
                    try:
                        label = self.label_encoder.transform([label_value])[0]
                        # 라벨 범위 검증 (0-4)
                        if 0 <= label < 5:
                            labels[period] = torch.tensor(label, dtype=torch.long)
                        else:
                            labels[period] = torch.tensor(-1, dtype=torch.long)
                    except (ValueError, IndexError):
                        labels[period] = torch.tensor(-1, dtype=torch.long)
                else:
                    # unknown이나 기타 무효 라벨
                    labels[period] = torch.tensor(-1, dtype=torch.long)
            else:
                labels[period] = torch.tensor(-1, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'additional_features': additional_features,
            'labels_1d': labels['1d'],
            'labels_3d': labels['3d'],
            'labels_5d': labels['5d']
        }

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

class StockPredictionTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, model_name='klue/bert-base', max_length=512, batch_size=16, learning_rate=2e-5):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 초기화
        self.model = MultiTaskStockPredictor(model_name).to(device)
        
        # 손실 함수 (각 태스크별 가중치)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.task_weights = {'1d': 0.4, '3d': 0.3, '5d': 0.3}
        
        # 결과 저장용
        self.train_history = {'loss': [], 'accuracy': {}}
        self.val_history = {'loss': [], 'accuracy': {}}
        
    def load_and_prepare_data(self, data_path="news_with_labels.csv"):
        """데이터 로드 및 전처리"""
        logger.info("데이터 로드 중...")
        
        # 데이터 로드
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        logger.info(f"전체 데이터: {len(df)}개")
        
        # 유효한 라벨이 있는 데이터만 필터링 (unknown 제외)
        valid_data = df[
            (df['label_1d'].notna()) & (df['label_1d'] != 'unknown') &
            (df['label_3d'].notna()) & (df['label_3d'] != 'unknown') &
            (df['label_5d'].notna()) & (df['label_5d'] != 'unknown')
        ].copy()
        
        logger.info(f"유효한 데이터: {len(valid_data)}개")
        
        # 데이터 분할 (시간 순서 고려)
        # 시간순 정렬
        valid_data['date'] = pd.to_datetime(valid_data['date'])
        valid_data = valid_data.sort_values('date')
        
        # 8:1:1 비율로 분할
        train_size = int(len(valid_data) * 0.8)
        val_size = int(len(valid_data) * 0.1)
        
        train_data = valid_data[:train_size]
        val_data = valid_data[train_size:train_size+val_size]
        test_data = valid_data[train_size+val_size:]
        
        logger.info(f"데이터 분할: Train {len(train_data)}, Val {len(val_data)}, Test {len(test_data)}")
        
        # 라벨 분포 확인
        for period in ['1d', '3d', '5d']:
            logger.info(f"{period} 라벨 분포:")
            label_dist = train_data[f'label_{period}'].value_counts()
            for label, count in label_dist.items():
                logger.info(f"  {label}: {count}개 ({count/len(train_data)*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def create_dataloaders(self, train_data, val_data, test_data):
        """데이터로더 생성"""
        train_dataset = NewsStockDataset(train_data, self.tokenizer, self.max_length, 'train')
        val_dataset = NewsStockDataset(val_data, self.tokenizer, self.max_length, 'val')
        test_dataset = NewsStockDataset(test_data, self.tokenizer, self.max_length, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, train_dataset.label_encoder
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        predictions = {period: [] for period in ['1d', '3d', '5d']}
        targets = {period: [] for period in ['1d', '3d', '5d']}
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 데이터 GPU로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            
            # 예측
            outputs = self.model(input_ids, attention_mask, additional_features)
            
            # 손실 계산 - 라벨 범위 검증 추가
            batch_loss = 0
            for period in ['1d', '3d', '5d']:
                labels = batch[f'labels_{period}'].to(device)
                valid_mask = labels != -1
                
                if valid_mask.sum() > 0:
                    valid_labels = labels[valid_mask]
                    valid_outputs = outputs[period][valid_mask]
                    
                    # 라벨 범위 재검증 (0-4)
                    range_mask = (valid_labels >= 0) & (valid_labels < 5)
                    
                    if range_mask.sum() > 0:
                        final_labels = valid_labels[range_mask]
                        final_outputs = valid_outputs[range_mask]
                        
                        period_loss = self.criterion(final_outputs, final_labels)
                        batch_loss += self.task_weights[period] * period_loss
                        
                        # 예측 결과 저장
                        with torch.no_grad():
                            preds = torch.argmax(final_outputs, dim=1)
                            predictions[period].extend(preds.cpu().numpy())
                            targets[period].extend(final_labels.cpu().numpy())
                    else:
                        logger.warning(f"배치에 유효한 {period} 라벨이 없음")
            
            # 역전파
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        # 에포크 결과 계산
        avg_loss = total_loss / len(train_loader)
        accuracies = {}
        
        for period in ['1d', '3d', '5d']:
            if predictions[period]:
                accuracy = np.mean(np.array(predictions[period]) == np.array(targets[period]))
                accuracies[period] = accuracy
        
        return avg_loss, accuracies
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        predictions = {period: [] for period in ['1d', '3d', '5d']}
        targets = {period: [] for period in ['1d', '3d', '5d']}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                additional_features = batch['additional_features'].to(device)
                
                outputs = self.model(input_ids, attention_mask, additional_features)
                
                batch_loss = 0
                for period in ['1d', '3d', '5d']:
                    labels = batch[f'labels_{period}'].to(device)
                    valid_mask = labels != -1
                    
                    if valid_mask.sum() > 0:
                        period_loss = self.criterion(outputs[period][valid_mask], labels[valid_mask])
                        batch_loss += self.task_weights[period] * period_loss
                        
                        preds = torch.argmax(outputs[period][valid_mask], dim=1)
                        predictions[period].extend(preds.cpu().numpy())
                        targets[period].extend(labels[valid_mask].cpu().numpy())
                
                total_loss += batch_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        accuracies = {}
        
        for period in ['1d', '3d', '5d']:
            if predictions[period]:
                accuracy = np.mean(np.array(predictions[period]) == np.array(targets[period]))
                accuracies[period] = accuracy
        
        return avg_loss, accuracies, predictions, targets
    
    def train_model(self, epochs=5, data_path="news_with_labels.csv"):
        """모델 학습 메인 함수"""
        logger.info("모델 학습 시작...")
        
        # 데이터 준비
        train_data, val_data, test_data = self.load_and_prepare_data(data_path)
        train_loader, val_loader, test_loader, label_encoder = self.create_dataloaders(
            train_data, val_data, test_data
        )
        
        # 옵티마이저 설정
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        
        # 학습 루프
        for epoch in range(epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            
            # 검증
            val_loss, val_acc, val_predictions, val_targets = self.validate(val_loader)
            
            # 결과 출력
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            
            for period in ['1d', '3d', '5d']:
                if period in train_acc and period in val_acc:
                    logger.info(f"{period} - Train Acc: {train_acc[period]:.4f}, Val Acc: {val_acc[period]:.4f}")
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                logger.info("새로운 최고 모델 저장!")
            
            # 히스토리 저장
            self.train_history['loss'].append(train_loss)
            self.val_history['loss'].append(val_loss)
            
            for period in ['1d', '3d', '5d']:
                if period not in self.train_history['accuracy']:
                    self.train_history['accuracy'][period] = []
                    self.val_history['accuracy'][period] = []
                
                if period in train_acc:
                    self.train_history['accuracy'][period].append(train_acc[period])
                if period in val_acc:
                    self.val_history['accuracy'][period].append(val_acc[period])
        
        # 최고 모델 로드
        self.model.load_state_dict(best_model_state)
        
        # 모델 저장
        self.save_model(label_encoder)
        
        # 최종 평가
        logger.info("\n=== 최종 평가 ===")
        test_loss, test_acc, test_predictions, test_targets = self.validate(test_loader)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        for period in ['1d', '3d', '5d']:
            if period in test_acc:
                logger.info(f"{period} Test Accuracy: {test_acc[period]:.4f}")
        
        # 상세 평가 리포트
        self.generate_evaluation_report(test_predictions, test_targets, label_encoder)
        
        return self.model
    
    def save_model(self, label_encoder):
        """모델 저장"""
        save_dir = "saved_model"
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델 저장
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pth'))
        
        # 설정 저장
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'task_weights': self.task_weights,
            'label_classes': label_encoder.classes_.tolist()
        }
        
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"모델 저장 완료: {save_dir}")
    
    def generate_evaluation_report(self, predictions, targets, label_encoder):
        """평가 리포트 생성"""
        logger.info("평가 리포트 생성 중...")
        
        class_names = label_encoder.classes_
        
        for period in ['1d', '3d', '5d']:
            if period in predictions and predictions[period]:
                logger.info(f"\n=== {period} 분류 리포트 ===")
                
                y_pred = np.array(predictions[period])
                y_true = np.array(targets[period])
                
                # 분류 리포트
                report = classification_report(
                    y_true, y_pred, 
                    target_names=class_names,
                    output_dict=True,
                    zero_division=0
                )
                
                # 주요 지표 출력
                for class_name in class_names:
                    if class_name in report:
                        precision = report[class_name]['precision']
                        recall = report[class_name]['recall']
                        f1 = report[class_name]['f1-score']
                        logger.info(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
                
                # 전체 정확도
                accuracy = report['accuracy']
                macro_f1 = report['macro avg']['f1-score']
                logger.info(f"정확도: {accuracy:.3f}, Macro F1: {macro_f1:.3f}")

def main():
    """메인 실행 함수"""
    logger.info("멀티태스크 주가 예측 모델 학습 시작")
    
    # 하이퍼파라미터 설정
    config = {
        'model_name': 'klue/bert-base',  # 한국어 BERT
        'max_length': 512,
        'batch_size': 4,  # GPU 메모리에 따라 조정
        'learning_rate': 2e-5,
        'epochs': 1  # 시작은 3에포크로
    }
    
    # 트레이너 초기화
    trainer = StockPredictionTrainer(
        model_name=config['model_name'],
        max_length=config['max_length'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # 모델 학습
    try:
        model = trainer.train_model(
            epochs=config['epochs'],
            data_path="news_with_labels.csv"
        )
        
        logger.info("모델 학습 완료!")
        logger.info("saved_model 폴더에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()