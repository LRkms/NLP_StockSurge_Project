import os
import time
import datetime
import logging
import pandas as pd
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
from typing import List, Dict, Optional, Any, Union

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NaverNewsConfig:
    """네이버 뉴스 크롤러 설정 클래스"""
    MECAB_PATH = 'C:/mecab/share/mecab-ko-dic'
    START_DATE = datetime.datetime(2025, 4, 1)
    END_DATE = datetime.datetime(2025, 4, 29)
    CATEGORIES = {
        "실시간 속보": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}",
        "주요뉴스": "https://finance.naver.com/news/mainnews.naver?date={date_hyphen}",
        "시황,전망": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}",
        "기업,종목분석": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}",
        "해외증시": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}",
        "채권,선물": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}",
        "공시,메모": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}",
        "환율": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=259&date={date}"
    }
    DUPLICATE_THRESHOLD = 0.95
    CHROME_OPTIONS = [
        "--headless=new",  # 최신 헤드리스 모드 사용
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--log-level=3",
        "--disable-extensions",
        "--disable-notifications"
    ]
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    DRIVER_TIMEOUT = 10  # 웹드라이버 타임아웃(초)
    PAGE_LOAD_WAIT = 1.0  # 페이지 로딩 대기 시간(초)
    MAX_RETRIES = 3       # 최대 재시도 횟수
    VECTORIZER_REBUILD_FREQUENCY = 200  # 벡터라이저 재구축 빈도
    BATCH_SIZE = 500      # 메모리 관리를 위한 배치 크기


class NaverFinanceCrawler:
    """네이버 금융 뉴스 크롤러 클래스"""
    
    def __init__(self, config=NaverNewsConfig):
        """크롤러 초기화"""
        self.config = config
        self.initialize_directories()
        
        try:
            self.mecab = Mecab(config.MECAB_PATH)
        except Exception as e:
            logger.error(f"Mecab 초기화 오류: {e}")
            logger.info("기본 토큰화 방법으로 대체합니다.")
            self.mecab = None
            
        self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
        self.collected_titles: List[str] = []
        self.title_vectors = None
        self.driver = None
        self.wait = None
        self.initialize_webdriver()

    def initialize_directories(self) -> None:
        """필요한 디렉토리 생성"""
        for directory in [self.config.DATA_DIR, self.config.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"디렉토리 확인: {directory}")

    def initialize_webdriver(self) -> None:
        """웹드라이버 초기화"""
        try:
            chrome_options = Options()
            for option in self.config.CHROME_OPTIONS:
                chrome_options.add_argument(option)
                
            # 성능 향상을 위한 추가 설정
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            chrome_options.add_experimental_option('prefs', {
                'profile.default_content_setting_values.notifications': 2,
                'profile.managed_default_content_settings.images': 2  # 이미지 로딩 차단
            })
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT)
            logger.info("웹드라이버 초기화 완료")
        except Exception as e:
            logger.error(f"웹드라이버 초기화 오류: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def reset_webdriver(self) -> None:
        """웹드라이버 재설정"""
        logger.info("웹드라이버 재설정 중...")
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.warning(f"웹드라이버 종료 오류: {e}")
        time.sleep(2)  # 자원 해제 대기
        self.initialize_webdriver()

    def tokenize_text(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        if not text:
            return []
            
        try:
            if self.mecab:
                tokens = self.mecab.pos(text)
                return [word for word, pos in tokens if pos.startswith('N') or pos.startswith('V')]
            else:
                # Mecab이 없는 경우 대체 토큰화
                return [w for w in text.split() if len(w) > 1]
        except Exception as e:
            logger.warning(f"토큰화 오류: {e}, 기본 방식으로 대체")
            return [w for w in text.split() if len(w) > 1]

    def is_duplicate_title(self, title: str) -> bool:
        """중복 제목 확인"""
        if not title or not self.collected_titles:
            return False
            
        try:
            tokens = self.tokenize_text(title)
            if not tokens:
                return False
                
            new_title = ' '.join(tokens)
            
            if self.title_vectors is None:
                return False
                
            try:
                new_vector = self.vectorizer.transform([new_title])
                similarities = cosine_similarity(new_vector, self.title_vectors)[0]
                return any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities)
            except ValueError:
                # 벡터라이저 재구축
                self._rebuild_vectorizer(new_title)
                new_vector = self.vectorizer.transform([new_title])
                similarities = cosine_similarity(new_vector, self.title_vectors)[0]
                return any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities)
        except Exception as e:
            logger.warning(f"중복 확인 오류: {e}")
            return False

    def _rebuild_vectorizer(self, new_title: Optional[str] = None) -> None:
        """벡터라이저 재구축"""
        try:
            all_titles = [' '.join(self.tokenize_text(t)) for t in self.collected_titles]
            if new_title:
                all_titles.append(' '.join(self.tokenize_text(new_title)))
                
            self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
            if new_title:
                self.title_vectors = self.vectorizer.fit_transform(all_titles[:-1])
            else:
                self.title_vectors = self.vectorizer.fit_transform(all_titles)
        except Exception as e:
            logger.warning(f"벡터라이저 재구축 오류: {e}")
            self.title_vectors = None

    def update_title_vectors(self, new_title: str) -> None:
        """제목 벡터 업데이트"""
        if not new_title:
            return
            
        try:
            self.collected_titles.append(new_title)
            
            # 주기적으로 벡터라이저 재구축
            if self.title_vectors is None or len(self.collected_titles) % self.config.VECTORIZER_REBUILD_FREQUENCY == 0:
                self._rebuild_vectorizer()
            else:
                # 단일 제목 추가
                new_title_tokens = ' '.join(self.tokenize_text(new_title))
                try:
                    new_vector = self.vectorizer.transform([new_title_tokens])
                    if self.title_vectors is not None:
                        self.title_vectors = pd.sparse.vstack([self.title_vectors, new_vector])
                except:
                    # 실패 시 전체 재구축
                    self._rebuild_vectorizer()
        except Exception as e:
            logger.warning(f"제목 벡터 업데이트 오류: {e}")

    @backoff.on_exception(
        backoff.expo, 
        (WebDriverException, TimeoutException, StaleElementReferenceException), 
        max_tries=NaverNewsConfig.MAX_RETRIES,
        on_backoff=lambda details: logger.warning(f"페이지 로딩 재시도: {details['tries']}/{NaverNewsConfig.MAX_RETRIES}")
    )
    def get_page(self, url: str) -> None:
        """페이지 로드"""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            time.sleep(self.config.PAGE_LOAD_WAIT)
        except Exception as e:
            logger.error(f"페이지 로드 오류 ({url}): {e}")
            raise

    def parse_news_item(self, li_element, category: str, date_hyphen: str) -> Optional[Dict[str, Any]]:
        """뉴스 항목 파싱"""
        try:
            title_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(1) a")
            title = title_element.text.strip()
            
            if not title:
                return None
                
            news_url = title_element.get_attribute("href")
            
            try:
                summary_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(2)")
                summary_text = summary_element.text.strip()
                
                try:
                    time_element = summary_element.find_element(By.TAG_NAME, "span")
                    news_time = time_element.text.strip()
                    summary_text = summary_text.replace(news_time, '').strip()
                except:
                    news_time = None
            except:
                summary_text = ""
                news_time = None
            
            if not self.is_duplicate_title(title):
                self.update_title_vectors(title)
                keywords = self.tokenize_text(title + " " + summary_text)
                
                return {
                    "date": date_hyphen,
                    "time": news_time,
                    "category": category,
                    "title": title,
                    "summary": summary_text,
                    "keywords": ", ".join(set(keywords)),
                    "url": news_url,
                    "collected_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except StaleElementReferenceException:
            logger.warning("페이지 요소가 변경되었습니다. 건너뜁니다.")
            return None
        except Exception as e:
            logger.warning(f"뉴스 항목 파싱 오류: {e}")
            return None
        
        return None

    def extract_news_data(self, category: str, date_str: str, date_hyphen: str) -> List[Dict[str, Any]]:
        """카테고리별 뉴스 데이터 추출"""
        url = self.config.CATEGORIES[category].format(date=date_str, date_hyphen=date_hyphen)
        logger.info(f"{category} ({date_hyphen}) 크롤링 중...")
        news_data = []
        
        try:
            self.get_page(url)
            
            # 페이지네이션 처리 개선
            page_num = 1
            while True:
                try:
                    li_elements = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#contentarea_left ul li"))
                    )
                    
                    if not li_elements:
                        logger.info(f"{category} 페이지 {page_num}에 뉴스가 없습니다.")
                        break
                        
                    for li in li_elements:
                        news = self.parse_news_item(li, category, date_hyphen)
                        if news:
                            news_data.append(news)
                    
                    # 다음 페이지 확인
                    try:
                        next_page = self.driver.find_element(By.XPATH, f"//a[contains(text(), '{page_num + 1}')]")
                        next_page.click()
                        page_num += 1
                        time.sleep(self.config.PAGE_LOAD_WAIT)
                    except NoSuchElementException:
                        # 다음 페이지 없음
                        break
                    except Exception as e:
                        logger.warning(f"다음 페이지 이동 오류: {e}")
                        break
                        
                except TimeoutException:
                    logger.warning(f"{category} 뉴스 목록을 찾을 수 없습니다.")
                    break
                except Exception as e:
                    logger.warning(f"{category} 뉴스 추출 오류: {e}")
                    break
        except Exception as e:
            logger.error(f"{category} 카테고리 처리 오류: {e}")
            
        logger.info(f"{category} ({date_hyphen}): {len(news_data)}개 뉴스 추출 완료")
        return news_data

    def save_data(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """데이터 저장"""
        if not data:
            logger.warning(f"저장할 데이터가 없습니다: {file_path}")
            return
            
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            pd.DataFrame(data).to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"{len(data)}개 뉴스 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"데이터 저장 오류 ({file_path}): {e}")

    def crawl_by_date(self, date: datetime.datetime) -> List[Dict[str, Any]]:
        """날짜별 크롤링"""
        date_str = date.strftime("%Y%m%d")
        date_hyphen = date.strftime("%Y-%m-%d")
        all_news = []
        
        logger.info(f"===== {date_hyphen} 크롤링 시작 =====")
        
        try:
            for category in self.config.CATEGORIES:
                try:
                    news = self.extract_news_data(category, date_str, date_hyphen)
                    all_news.extend(news)
                except Exception as e:
                    logger.error(f"{category} 처리 중 오류: {e}")
                
                # 카테고리 간 간격
                time.sleep(1.0)
                
                # 메모리 관리: 배치 크기에 도달하면 중간 저장
                if len(all_news) >= self.config.BATCH_SIZE:
                    self.save_data(all_news, f"{self.config.DATA_DIR}/news_{date_str}_batch.csv")
                    # 배치 처리 후 GC 실행
                    gc.collect()
            
            if all_news:
                output_path = f"{self.config.DATA_DIR}/news_{date_str}.csv"
                self.save_data(all_news, output_path)
            else:
                logger.warning(f"{date_hyphen}: 저장할 뉴스가 없습니다.")
        except Exception as e:
            logger.error(f"{date_hyphen} 크롤링 오류: {e}")
            
        logger.info(f"===== {date_hyphen} 크롤링 완료 ({len(all_news)}개) =====")
        return all_news

    def save_combined_data(self) -> None:
        """수집된 모든 뉴스 데이터 통합"""
        try:
            logger.info("모든 뉴스 데이터 통합 중...")
            
            # 메모리 효율적인 방식으로 파일 읽기
            files = [os.path.join(self.config.DATA_DIR, f) for f in os.listdir(self.config.DATA_DIR) 
                    if f.startswith("news_") and f.endswith(".csv")]
            
            if not files:
                logger.warning("통합할 뉴스 파일이 없습니다.")
                return
                
            # 데이터프레임 청크 처리
            chunks = []
            for file in files:
                try:
                    if not os.path.isfile(file):
                        continue
                    df = pd.read_csv(file, encoding='utf-8-sig')
                    if not df.empty:
                        chunks.append(df)
                except Exception as e:
                    logger.error(f"파일 읽기 오류 ({file}): {e}")
            
            if not chunks:
                logger.warning("유효한 데이터가 없습니다.")
                return
                
            # 청크 병합 및 중복 제거
            combined = pd.concat(chunks, ignore_index=True)
            combined.drop_duplicates(subset=['title', 'date'], inplace=True)
            
            # 날짜 정렬
            try:
                combined['date'] = pd.to_datetime(combined['date'])
                combined.sort_values('date', ascending=False, inplace=True)
            except:
                logger.warning("날짜 정렬 실패")
            
            # 최종 저장
            start = self.config.START_DATE.strftime("%Y%m%d")
            end = self.config.END_DATE.strftime("%Y%m%d")
            path = f"{self.config.DATA_DIR}/all_news_{start}_to_{end}.csv"
            
            combined.to_csv(path, index=False, encoding='utf-8-sig')
            logger.info(f"총 {len(combined)}개 뉴스 통합 저장 완료: {path}")
            
            # 배치 파일 정리
            batch_files = [f for f in files if '_batch' in f]
            for bf in batch_files:
                try:
                    os.remove(bf)
                except:
                    pass
        except Exception as e:
            logger.error(f"데이터 통합 오류: {e}")
            logger.error(traceback.format_exc())

    def run(self) -> None:
        """크롤러 실행"""
        start_time = time.time()
        success = False
        
        try:
            logger.info(f"네이버 금융 뉴스 크롤링 시작: {self.config.START_DATE.strftime('%Y-%m-%d')} ~ {self.config.END_DATE.strftime('%Y-%m-%d')}")
            
            # 날짜 범위 생성
            date_range = [
                self.config.START_DATE + datetime.timedelta(days=i) 
                for i in range((self.config.END_DATE - self.config.START_DATE).days + 1)
            ]
            
            for idx, date in enumerate(date_range):
                try:
                    logger.info(f"[{idx+1}/{len(date_range)}] 날짜 처리 중: {date.strftime('%Y-%m-%d')}")
                    self.crawl_by_date(date)
                    
                    # 3일마다 웹드라이버 재설정
                    if (idx + 1) % 3 == 0:
                        self.reset_webdriver()
                except Exception as e:
                    logger.error(f"날짜 처리 오류 ({date.strftime('%Y-%m-%d')}): {e}")
                    # 오류 발생 시 웹드라이버 재설정
                    self.reset_webdriver()
            
            # 모든 데이터 통합
            self.save_combined_data()
            
            elapsed_time = time.time() - start_time
            logger.info(f"크롤링 완료! 총 소요 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")
            success = True
            
        except Exception as e:
            logger.error(f"크롤러 실행 중 오류 발생: {e}")
            logger.error(traceback.format_exc())
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("웹드라이버 종료됨")
                except:
                    pass
            
            if success:
                logger.info("크롤링이 성공적으로 완료되었습니다!")
            else:
                logger.error("크롤링 중 오류가 발생했습니다.")


if __name__ == '__main__':
    try:
        crawler = NaverFinanceCrawler()
        crawler.run()
    except Exception as e:
        logger.critical(f"프로그램 실행 오류: {e}")
        logger.critical(traceback.format_exc())