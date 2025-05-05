import os
import time
import datetime
import logging
import pandas as pd
import concurrent.futures
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
import traceback
import backoff
import gc
import hashlib
import pickle
from functools import lru_cache

# 로깅 설정 개선
def setup_logger():
    """향상된 로깅 시스템 설정"""
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    os.makedirs("logs", exist_ok=True)
    
    # 로그 파일 회전 설정
    log_file = f"logs/crawler_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # 외부 라이브러리 로깅 레벨 조정
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('webdriver_manager').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# 로거 초기화
logger = setup_logger()

class NaverNewsConfig:
    """향상된 뉴스 크롤러 설정 클래스"""
    
    # Mecab 경로 설정
    MECAB_PATH = 'C:/mecab/share/mecab-ko-dic'
    
    # 검색 날짜 범위
    START_DATE = datetime.datetime(2025, 4, 1)
    END_DATE = datetime.datetime(2025, 4, 29)
    
    # 카테고리 URL 매핑
    CATEGORIES = {
        "실시간 속보": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}",
        "주요뉴스": "https://finance.naver.com/news/mainnews.naver?date={date_hyphen}",
        "시황,전망": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}",
        "기업,종목분석": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}",
        "해외증시": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}",
        "채권,선물": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}",
        "공시,메모": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}",
        "환율": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=259&date={date}",
    }
    
    # 중복 검사 관련 설정
    DUPLICATE_THRESHOLD = 0.90  # 조금 낮춤 (더 엄격한 중복 검사)
    MIN_CACHE_SIZE = 1000       # 캐시 초기화 전 최소 크기
    MAX_CACHE_SIZE = 10000      # 캐시 최대 크기
    
    # 웹드라이버 설정
    CHROME_OPTIONS = [
        "--headless=new",        # 최신 headless 모드
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--log-level=3",
        "--disable-extensions",
        "--ignore-certificate-errors",
        "--disable-notifications",
        "--disable-infobars"
    ]
    
    # 파일 저장 경로
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    CACHE_DIR = "cache"
    
    # 웹드라이버 설정
    DRIVER_TIMEOUT = 15        # 타임아웃 증가
    PAGE_LOAD_WAIT = 1.0       # 페이지 로딩 대기시간 조정
    WEBDRIVER_COOL_DOWN = 0.5  # 웹드라이버 재설정 후 대기시간
    
    # 최대 재시도 횟수 및 백오프 설정
    MAX_RETRIES = 5            # 재시도 횟수 증가
    MAX_BACKOFF = 30           # 최대 백오프 시간 (초)
    BASE_BACKOFF = 2           # 기본 백오프 시간 (초)
    
    # 병렬 처리 설정
    MAX_WORKERS = 3            # 동시 처리 작업 수 (카테고리당)
    
    # 데이터 배치 처리 설정
    BATCH_SIZE = 100           # 한 번에 저장할 데이터 크기
    
    # 중간 저장 설정
    SAVE_INTERVAL = 1          # 날짜 단위 크롤링마다 저장


class NaverFinanceCrawler:
    """개선된 네이버 파이낸스 뉴스 크롤러 클래스"""
    
    def __init__(self, config=NaverNewsConfig):
        """
        크롤러 초기화
        
        Args:
            config: 크롤러 설정 객체
        """
        self.config = config
        self.initialize_directories()
        
        # Mecab 초기화 (메모리 관리를 위해 필요할 때만 로드)
        self._mecab = None
        
        # 중복 검사 관련 변수 최적화
        self.title_hashes = set()  # 해시 기반 빠른 중복 검사
        self.collected_titles = []
        self.vectorizer = None
        self.title_vectors = None
        
        # 웹드라이버 관련 변수
        self.driver = None
        self.wait = None
        
        # 캐시 관리 변수
        self.cache_path = os.path.join(self.config.CACHE_DIR, "titles_cache.pkl")
        self.load_cache()
        
        # 성능 통계 추적
        self.stats = {
            "pages_processed": 0,
            "news_collected": 0,
            "duplicates_found": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        # 웹드라이버 초기화
        self.initialize_webdriver()
    
    @property
    def mecab(self):
        """지연 로딩 방식의 Mecab 인스턴스 접근자"""
        if self._mecab is None:
            try:
                self._mecab = Mecab(self.config.MECAB_PATH)
                logger.info("Mecab 형태소 분석기가 초기화되었습니다.")
            except Exception as e:
                logger.error(f"Mecab 초기화 실패: {str(e)}")
                # 대체 방법 준비
                self._mecab = None
        return self._mecab
    
    def initialize_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
    
    def load_cache(self):
        """기존 캐시 데이터 로드"""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.title_hashes = cache_data.get('hashes', set())
                    self.collected_titles = cache_data.get('titles', [])
                    logger.info(f"캐시에서 {len(self.collected_titles)}개의 제목과 {len(self.title_hashes)}개의 해시 로드됨")
        except Exception as e:
            logger.warning(f"캐시 로드 실패, 새로운 캐시를 시작합니다: {str(e)}")
            self.title_hashes = set()
            self.collected_titles = []
    
    def save_cache(self):
        """현재 캐시 데이터 저장"""
        try:
            # 캐시 크기 제한
            if len(self.collected_titles) > self.config.MAX_CACHE_SIZE:
                self.collected_titles = self.collected_titles[-self.config.MIN_CACHE_SIZE:]
                # 해시 재생성
                self.title_hashes = {self._compute_hash(title) for title in self.collected_titles}
            
            cache_data = {
                'hashes': self.title_hashes,
                'titles': self.collected_titles
            }
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"캐시 저장 완료: {len(self.collected_titles)}개 제목, {len(self.title_hashes)}개 해시")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {str(e)}")
    
    def initialize_webdriver(self):
        """향상된 웹드라이버 초기화"""
        try:
            chrome_options = Options()
            
            for option in self.config.CHROME_OPTIONS:
                chrome_options.add_argument(option)
            
            # 추가 성능 개선 옵션
            chrome_options.add_argument("--disable-features=NetworkService")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            
            # 페이지 로드 전략 변경 (속도 향상)
            chrome_options.page_load_strategy = 'eager'
            
            # 다운로드 기능 비활성화
            prefs = {
                "download.default_directory": "/dev/null",
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # 암시적 대기 설정
            self.driver.implicitly_wait(2)
            self.wait = WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT)
            
            logger.info("웹드라이버가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"웹드라이버 초기화 실패: {str(e)}")
            raise
    
    def reset_webdriver(self):
        """안전한 웹드라이버 재시작"""
        logger.warning("웹드라이버 재설정 중...")
        try:
            if self.driver:
                self.driver.quit()
        except Exception as e:
            logger.warning(f"웹드라이버 종료 중 오류: {str(e)}")
        
        # 메모리 정리
        self.driver = None
        self.wait = None
        gc.collect()
        
        # 잠시 대기 후 재시작
        time.sleep(self.config.WEBDRIVER_COOL_DOWN)
        self.initialize_webdriver()
    
    def _compute_hash(self, text):
        """텍스트의 SHA-256 해시 계산"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @lru_cache(maxsize=1024)
    def tokenize_text(self, text):
        """
        텍스트를 토큰화하여 명사와 동사만 추출 (캐싱 적용)
        
        Args:
            text: 토큰화할 텍스트
        
        Returns:
            토큰화된 단어 목록
        """
        try:
            if self.mecab:
                tokens = self.mecab.pos(text)
                return tuple(word for word, pos in tokens if pos.startswith('N') or pos.startswith('V'))
            else:
                raise ValueError("Mecab 인스턴스 없음")
        except Exception as e:
            logger.debug(f"토큰화 중 오류 발생: {str(e)}, 일반 분할 사용")
            # 형태소 분석기 실패 시 간단한 분할 사용
            return tuple(w for w in text.split() if len(w) > 1)
    
    def is_duplicate_title(self, title):
        """
        제목이 기존에 수집된 제목과 중복되는지 확인 (다중 기법 적용)
        
        Args:
            title: 확인할 뉴스 제목
        
        Returns:
            중복 여부 (True/False)
        """
        if not title:
            return False
        
        # 1. 해시 기반 빠른 검사
        title_hash = self._compute_hash(title)
        if title_hash in self.title_hashes:
            self.stats["duplicates_found"] += 1
            return True
        
        # 2. 정확한 문자열 일치 검사
        if title in self.collected_titles:
            self.title_hashes.add(title_hash)  # 해시 캐시 업데이트
            self.stats["duplicates_found"] += 1
            return True
        
        # 3. 벡터 유사도 기반 검사 (타이틀이 충분히 쌓였을 때만)
        if len(self.collected_titles) >= 10 and self.vectorizer is not None:
            try:
                # 토큰화 및 벡터화
                tokens = self.tokenize_text(title)
                new_title = ' '.join(tokens)
                
                new_vector = self.vectorizer.transform([new_title])
                similarities = cosine_similarity(new_vector, self.title_vectors)[0]
                
                if any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities):
                    self.title_hashes.add(title_hash)  # 해시 캐시 업데이트
                    self.stats["duplicates_found"] += 1
                    return True
                
            except Exception as e:
                logger.debug(f"벡터 유사도 검사 중 오류: {str(e)}")
        
        return False
    
    def update_title_vectors(self, new_title):
        """
        새 제목으로 제목 벡터 업데이트 (최적화됨)
        
        Args:
            new_title: 추가할 새 제목
        """
        if not new_title:
            return
        
        # 해시 및 제목 컬렉션 업데이트
        title_hash = self._compute_hash(new_title)
        self.title_hashes.add(title_hash)
        self.collected_titles.append(new_title)
        
        # 벡터 업데이트 로직
        try:
            # 초기 벡터 생성 또는 주기적 재생성
            if self.vectorizer is None or len(self.collected_titles) % 500 == 0:
                # 메모리 효율을 위해 최근 항목으로 제한
                titles_for_vectorizing = self.collected_titles[-5000:] if len(self.collected_titles) > 5000 else self.collected_titles
                
                all_tokens = [' '.join(self.tokenize_text(t)) for t in titles_for_vectorizing]
                self.vectorizer = TfidfVectorizer(max_features=10000)  # 차원 제한
                self.title_vectors = self.vectorizer.fit_transform(all_tokens)
            
            # 증분 업데이트가 필요한 경우 (매번 전체 재계산은 비효율적)
            elif self.vectorizer is not None and self.title_vectors is not None:
                # 주기적으로만 개별 벡터 추가 (성능을 위해)
                if len(self.collected_titles) % 50 == 0:
                    recent_titles = self.collected_titles[-50:]
                    recent_tokens = [' '.join(self.tokenize_text(t)) for t in recent_titles]
                    recent_vectors = self.vectorizer.transform(recent_tokens)
                    
                    if self.title_vectors.shape[0] > 5000:
                        # 오래된 벡터 제거
                        self.title_vectors = self.title_vectors[-4950:]
                    
                    # 새 벡터 추가
                    self.title_vectors = np.vstack((self.title_vectors.toarray(), recent_vectors.toarray()))
                    self.title_vectors = self.vectorizer.transform(self.title_vectors)
                    
        except Exception as e:
            logger.warning(f"벡터 업데이트 중 오류: {str(e)}")
            # 오류 발생 시 초기화
            if len(self.collected_titles) > 1000 and len(self.collected_titles) % 1000 == 0:
                self.vectorizer = None
                self.title_vectors = None
    
    @backoff.on_exception(
        backoff.expo,
        (WebDriverException, TimeoutException, StaleElementReferenceException),
        max_tries=NaverNewsConfig.MAX_RETRIES,
        max_value=NaverNewsConfig.MAX_BACKOFF,
        base=NaverNewsConfig.BASE_BACKOFF
    )
    def get_page(self, url):
        """
        지정된 URL 페이지 로드 (확장된 재시도 로직)
        
        Args:
            url: 로드할 페이지 URL
        """
        try:
            self.driver.get(url)
            # 명시적 대기로 변경
            WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            
            # 적응형 대기 시간 (페이지 로딩 상태에 따라 조정)
            if "contentarea_left" in self.driver.page_source:
                # 데이터가 이미 있는 경우 대기 시간 단축
                time.sleep(max(0.5, self.config.PAGE_LOAD_WAIT / 2))
            else:
                time.sleep(self.config.PAGE_LOAD_WAIT)
                
            self.stats["pages_processed"] += 1
            
        except (WebDriverException, TimeoutException, StaleElementReferenceException) as e:
            logger.warning(f"페이지 로드 중 오류 발생: {str(e)} - URL: {url}")
            # 심각한 오류인 경우에만 웹드라이버 재설정
            if "crashed" in str(e).lower() or "session" in str(e).lower():
                self.reset_webdriver()
            raise
    
    def parse_news_item(self, li_element, category, date_hyphen):
        """
        뉴스 항목 파싱 (향상된 안정성)
        
        Args:
            li_element: 뉴스 항목 요소
            category: 뉴스 카테고리
            date_hyphen: 날짜 (yyyy-mm-dd)
        
        Returns:
            파싱된 뉴스 항목 딕셔너리 또는 None
        """
        try:
            # 유효한 뉴스 항목인지 빠르게 검증
            try:
                title_element = WebDriverWait(li_element, 2).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "dl dd:nth-of-type(1) a"))
                )
            except (TimeoutException, NoSuchElementException):
                return None  # 유효한 뉴스 항목이 아님
            
            title = title_element.text.strip()
            if not title:  # 빈 제목 건너뛰기
                return None
                
            news_url = title_element.get_attribute("href")
            
            # 요약 추출
            try:
                summary_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(2)")
                summary_text = summary_element.text.strip()
            except NoSuchElementException:
                summary_text = ""
            
            # 시간 정보 추출
            try:
                time_element = summary_element.find_element(By.TAG_NAME, "span")
                news_time = time_element.text.strip()
                # 시간 정보에서 뉴스 내용과 분리
                summary_text = summary_text.replace(news_time, "").strip()
            except (NoSuchElementException, AttributeError):
                news_time = None
            except Exception as e:
                logger.debug(f"시간 추출 중 오류: {str(e)}")
                news_time = None
            
            # 중복 검사 (최적화됨)
            if not self.is_duplicate_title(title):
                self.update_title_vectors(title)
                
                # 키워드 추출
                try:
                    keywords = list(self.tokenize_text(title + " " + summary_text))
                    # 출현 빈도에 따라 정렬
                    keyword_freq = {}
                    for word in keywords:
                        keyword_freq[word] = keyword_freq.get(word, 0) + 1
                    
                    # 빈도 기준 상위 키워드만 선택
                    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                    keywords_str = ", ".join(word for word, _ in top_keywords)
                except Exception as e:
                    logger.debug(f"키워드 추출 중 오류: {str(e)}")
                    keywords_str = ""
                
                self.stats["news_collected"] += 1
                
                return {
                    "date": date_hyphen,
                    "time": news_time if news_time else None,
                    "category": category,
                    "title": title,
                    "summary": summary_text,
                    "keywords": keywords_str,
                    "url": news_url,
                    "collected_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
        except StaleElementReferenceException:
            logger.debug("요소가 더 이상 DOM에 존재하지 않음")
        except Exception as e:
            self.stats["errors"] += 1
            logger.debug(f"뉴스 항목 파싱 중 오류: {str(e)}")
            
        return None
    
    def extract_news_data(self, category, current_date_str, current_date_hyphen):
        """
        지정된 카테고리 및 날짜의 뉴스 데이터 추출 (안정성 개선)
        
        Args:
            category: 뉴스 카테고리
            current_date_str: 날짜 문자열 (yyyymmdd)
            current_date_hyphen: 날짜 문자열 (yyyy-mm-dd)
        
        Returns:
            추출된 뉴스 데이터 및 오류 데이터
        """
        news_data = []
        error_data = []
        
        # URL 형식 선택
        if "주요뉴스" in category:
            url = self.config.CATEGORIES[category].format(date_hyphen=current_date_hyphen)
        else:
            url = self.config.CATEGORIES[category].format(date=current_date_str)
        
        logger.info(f"크롤링 시작: {category}, 날짜: {current_date_hyphen}")
        
        page_num = 1
        max_empty_pages = 2  # 연속 빈 페이지 최대 허용 수
        empty_page_count = 0
        
        try:
            self.get_page(url)
            
            while empty_page_count < max_empty_pages:
                try:
                    # 더 안정적인 요소 탐지
                    try:
                        content_area = self.wait.until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "#contentarea_left"))
                        )
                        li_elements = content_area.find_elements(By.CSS_SELECTOR, "ul li")
                    except TimeoutException:
                        # 컨텐츠 영역을 찾을 수 없음
                        logger.warning(f"컨텐츠 영역을 찾을 수 없음: {category}, 페이지 {page_num}")
                        break
                    
                    if not li_elements:
                        logger.info(f"뉴스 항목이 없음: {category}, 페이지 {page_num}")
                        empty_page_count += 1
                        
                        # 주요뉴스 카테고리는 다음 페이지가 없을 수 있음
                        if "주요뉴스" in category:
                            break
                        
                        # 다음 페이지 시도
                        try:
                            next_page = self.driver.find_element(By.LINK_TEXT, "다음")
                            if not next_page.is_displayed() or not next_page.is_enabled():
                                logger.info("다음 페이지 버튼이 비활성화되어 있음")
                                break
                            next_page.click()
                            page_num += 1
                            time.sleep(self.config.PAGE_LOAD_WAIT)
                            continue
                        except NoSuchElementException:
                            logger.info("다음 페이지가 없음")
                            break
                    else:
                        # 뉴스 항목이 있으면 empty_page_count 초기화
                        empty_page_count = 0
                    
                    current_url = self.driver.current_url
                    logger.debug(f"{len(li_elements)}개의 뉴스 항목 발견, 페이지 {page_num}, URL: {current_url}")
                    
                    for li in li_elements:
                        try:
                            news_item = self.parse_news_item(li, category, current_date_hyphen)
                            if news_item:
                                news_data.append(news_item)
                                
                                # 배치 저장 (메모리 관리)
                                if len(news_data) % self.config.BATCH_SIZE == 0:
                                    logger.info(f"{len(news_data)}개 뉴스 항목 수집됨 (현재 진행 상황)")