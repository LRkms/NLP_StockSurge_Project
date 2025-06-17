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
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
import traceback
import backoff

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
    """뉴스 크롤러 설정 클래스"""
    
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
    
    # 중복 검사 임계값
    DUPLICATE_THRESHOLD = 0.95
    
    # 웹드라이버 설정
    CHROME_OPTIONS = [
        "--headless",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--log-level=3"
    ]
    
    # 파일 저장 경로
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    
    # 웹드라이버 대기 시간 (초)
    DRIVER_TIMEOUT = 10
    
    # 페이지 로딩 대기 시간 (초)
    PAGE_LOAD_WAIT = 1.5
    
    # 최대 재시도 횟수
    MAX_RETRIES = 3


class NaverFinanceCrawler:
    """네이버 파이낸스 뉴스 크롤러 클래스"""
    
    def __init__(self, config=NaverNewsConfig):
        """
        크롤러 초기화
        
        Args:
            config: 크롤러 설정 객체
        """
        self.config = config
        self.initialize_directories()
        self.mecab = Mecab(config.MECAB_PATH)
        self.vectorizer = TfidfVectorizer()
        self.collected_titles = []
        self.title_vectors = None
        self.driver = None
        self.wait = None
        self.initialize_webdriver()
    
    def initialize_directories(self):
        """필요한 디렉토리 생성"""
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)
    
    def initialize_webdriver(self):
        """웹드라이버 초기화"""
        try:
            chrome_options = Options()
            
            for option in self.config.CHROME_OPTIONS:
                chrome_options.add_argument(option)
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT)
            logger.info("웹드라이버가 성공적으로 초기화되었습니다.")
        except Exception as e:
            logger.error(f"웹드라이버 초기화 실패: {str(e)}")
            raise
    
    def reset_webdriver(self):
        """웹드라이버 재시작"""
        logger.warning("웹드라이버 재설정 중...")
        try:
            if self.driver:
                self.driver.quit()
        except:
            pass
        finally:
            self.initialize_webdriver()
    
    def tokenize_text(self, text):
        """
        텍스트를 토큰화하여 명사와 동사만 추출
        
        Args:
            text: 토큰화할 텍스트
        
        Returns:
            토큰화된 단어 목록
        """
        try:
            tokens = self.mecab.pos(text)
            return [word for word, pos in tokens if pos.startswith('N') or pos.startswith('V')]
        except Exception as e:
            logger.warning(f"토큰화 중 오류 발생: {str(e)}, 일반 분할 사용")
            # 형태소 분석기 실패 시 간단한 분할 사용
            return [w for w in text.split() if len(w) > 1]
    
    def is_duplicate_title(self, title):
        """
        제목이 기존에 수집된 제목과 중복되는지 확인
        
        Args:
            title: 확인할 뉴스 제목
        
        Returns:
            중복 여부 (True/False)
        """
        if not self.collected_titles:
            return False
        
        tokens = self.tokenize_text(title)
        new_title = ' '.join(tokens)
        
        try:
            new_vector = self.vectorizer.transform([new_title])
            similarities = cosine_similarity(new_vector, self.title_vectors)[0]
            return any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities)
        except ValueError:
            # 벡터라이저 재학습
            all_titles = [' '.join(self.tokenize_text(t)) for t in self.collected_titles] + [new_title]
            self.vectorizer = TfidfVectorizer()
            self.title_vectors = self.vectorizer.fit_transform(all_titles[:-1])
            new_vector = self.vectorizer.transform([all_titles[-1]])
            similarities = cosine_similarity(new_vector, self.title_vectors)[0]
            return any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities)
        except Exception as e:
            logger.warning(f"중복 체크 중 오류 발생: {str(e)}, 중복 아님으로 처리")
            return False
    
    def update_title_vectors(self, new_title):
        """
        새 제목으로 제목 벡터 업데이트
        
        Args:
            new_title: 추가할 새 제목
        """
        self.collected_titles.append(new_title)
        
        if len(self.collected_titles) % 100 == 0:  # 100개마다 벡터 재구성
            try:
                all_titles = [' '.join(self.tokenize_text(t)) for t in self.collected_titles]
                self.vectorizer = TfidfVectorizer()
                self.title_vectors = self.vectorizer.fit_transform(all_titles)
            except Exception as e:
                logger.warning(f"벡터 업데이트 중 오류: {str(e)}")
        
        # 최초 벡터 생성
        if self.title_vectors is None and len(self.collected_titles) == 1:
            try:
                all_titles = [' '.join(self.tokenize_text(t)) for t in self.collected_titles]
                self.vectorizer = TfidfVectorizer()
                self.title_vectors = self.vectorizer.fit_transform(all_titles)
            except Exception as e:
                logger.warning(f"초기 벡터 생성 중 오류: {str(e)}")
    
    @backoff.on_exception(
        backoff.expo,
        (WebDriverException, TimeoutException),
        max_tries=NaverNewsConfig.MAX_RETRIES
    )
    def get_page(self, url):
        """
        지정된 URL 페이지 로드 (재시도 로직 포함)
        
        Args:
            url: 로드할 페이지 URL
        """
        try:
            self.driver.get(url)
            # 명시적 대기로 변경
            WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
            )
            time.sleep(self.config.PAGE_LOAD_WAIT)
        except (WebDriverException, TimeoutException) as e:
            logger.warning(f"페이지 로드 중 오류 발생: {str(e)}")
            self.reset_webdriver()
            raise
    
    def parse_news_item(self, li_element, category, date_hyphen):
        """
        뉴스 항목 파싱
        
        Args:
            li_element: 뉴스 항목 요소
            category: 뉴스 카테고리
            date_hyphen: 날짜 (yyyy-mm-dd)
        
        Returns:
            파싱된 뉴스 항목 딕셔너리 또는 None
        """
        try:
            title_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(1) a")
            title = title_element.text.strip()
            news_url = title_element.get_attribute("href")
            
            summary_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(2)")
            summary_text = summary_element.text.strip()
            
            # 시간 정보 추출
            try:
                time_element = summary_element.find_element(By.TAG_NAME, "span")
                news_time = time_element.text.strip()
                # 시간 정보에서 뉴스 내용과 분리
                summary_text = summary_text.replace(news_time, "").strip()
            except NoSuchElementException:
                news_time = None
            except Exception as e:
                logger.warning(f"시간 추출 중 오류: {str(e)}")
                news_time = None
            
            # 중복 검사
            if not self.is_duplicate_title(title):
                self.update_title_vectors(title)
                
                # 키워드 추출
                keywords = self.tokenize_text(title + " " + summary_text)
                keywords_str = ", ".join(set(keywords))
                
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
                
        except NoSuchElementException as e:
            logger.debug(f"뉴스 항목 요소를 찾을 수 없음: {str(e)}")
        except Exception as e:
            logger.warning(f"뉴스 항목 파싱 중 오류: {str(e)}")
            
        return None
    
    def extract_news_data(self, category, current_date_str, current_date_hyphen):
        """
        지정된 카테고리 및 날짜의 뉴스 데이터 추출
        
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
        
        try:
            self.get_page(url)
            
            while True:
                try:
                    li_elements = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#contentarea_left ul li"))
                    )
                    
                    if not li_elements:
                        logger.info(f"뉴스 항목이 없음: {category}, 페이지 {page_num}")
                        break
                    
                    logger.debug(f"{len(li_elements)}개의 뉴스 항목 발견, 페이지 {page_num}")
                    
                    for li in li_elements:
                        try:
                            news_item = self.parse_news_item(li, category, current_date_hyphen)
                            if news_item:
                                news_data.append(news_item)
                        except Exception as e:
                            error_info = {
                                "date": current_date_hyphen,
                                "url": url,
                                "category": category,
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            error_data.append(error_info)
                            logger.warning(f"뉴스 항목 처리 중 오류: {str(e)}")
                    
                    # 다음 페이지 처리
                    try:
                        next_page = self.driver.find_element(By.LINK_TEXT, "다음")
                        next_page.click()
                        WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT).until(
                            EC.staleness_of(li_elements[0])
                        )
                        page_num += 1
                    except NoSuchElementException:
                        logger.debug(f"다음 페이지 없음, 크롤링 완료: {category}")
                        break
                    except Exception as e:
                        logger.warning(f"다음 페이지 처리 중 오류: {str(e)}")
                        break
                        
                except TimeoutException:
                    logger.warning(f"페이지 로딩 타임아웃: {category}, 페이지 {page_num}")
                    break
                except Exception as e:
                    error_info = {
                        "date": current_date_hyphen,
                        "url": url,
                        "category": category,
                        "page": page_num,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    error_data.append(error_info)
                    logger.error(f"뉴스 추출 중 오류: {str(e)}")
                    break
                    
        except Exception as e:
            error_info = {
                "date": current_date_hyphen,
                "url": url,
                "category": category,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            error_data.append(error_info)
            logger.error(f"페이지 처리 중 치명적 오류: {str(e)}")
        
        logger.info(f"크롤링 완료: {category}, 날짜: {current_date_hyphen}, 뉴스: {len(news_data)}개, 오류: {len(error_data)}개")
        
        return news_data, error_data
    
    def save_data(self, data, file_path, encoding='utf-8-sig'):
        """
        데이터를 CSV 파일로 저장
        
        Args:
            data: 저장할 데이터
            file_path: 저장 경로
            encoding: 파일 인코딩 (기본: utf-8-sig)
        """
        try:
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding=encoding)
            logger.info(f"데이터 저장 완료: {file_path}, {len(data)}개 항목")
            return True
        except Exception as e:
            logger.error(f"데이터 저장 중 오류: {file_path}, {str(e)}")
            return False
    
    def crawl_by_date(self, current_date):
        """
        지정된 날짜의 모든 카테고리 크롤링
        
        Args:
            current_date: 크롤링할 날짜
        
        Returns:
            뉴스 데이터 및 오류 데이터
        """
        current_date_str = current_date.strftime("%Y%m%d")
        current_date_hyphen = current_date.strftime("%Y-%m-%d")
        
        all_news_data = []
        all_error_data = []
        
        logger.info(f"==== {current_date_hyphen} 뉴스 크롤링 시작 ====")
        
        for category in self.config.CATEGORIES.keys():
            try:
                news_data, error_data = self.extract_news_data(category, current_date_str, current_date_hyphen)
                all_news_data.extend(news_data)
                all_error_data.extend(error_data)
                
                # 부하 방지를 위한 카테고리 간 대기
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"카테고리 크롤링 중 오류: {category}, {str(e)}")
                error_info = {
                    "date": current_date_hyphen,
                    "category": category,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_error_data.append(error_info)
        
        # 데이터 저장
        if all_news_data:
            self.save_data(
                all_news_data,
                f"{self.config.DATA_DIR}/news_{current_date_str}.csv"
            )
        
        if all_error_data:
            self.save_data(
                all_error_data,
                f"{self.config.LOGS_DIR}/errors_{current_date_str}.csv"
            )
        
        logger.info(f"==== {current_date_hyphen} 뉴스 크롤링 완료: 뉴스 {len(all_news_data)}개, 에러 {len(all_error_data)}개 ====")
        return all_news_data, all_error_data
    
    def run(self):
        """크롤러 실행"""
        try:
            start_time = time.time()
            
            # 날짜 범위 계산
            date_range = [
                self.config.START_DATE + datetime.timedelta(days=x)
                for x in range((self.config.END_DATE - self.config.START_DATE).days + 1)
            ]
            
            total_news = 0
            total_errors = 0
            
            for current_date in date_range:
                news_data, error_data = self.crawl_by_date(current_date)
                total_news += len(news_data)
                total_errors += len(error_data)
            
            # 최종 결과 저장 (전체 기간 통합)
            self.save_combined_data()
            
            elapsed_time = time.time() - start_time
            logger.info(f"크롤링 작업 완료! 총 뉴스: {total_news}개, 총 오류: {total_errors}개, 수행 시간: {elapsed_time:.2f}초")
            
        except Exception as e:
            logger.critical(f"크롤러 실행 중 치명적 오류: {str(e)}\n{traceback.format_exc()}")
        finally:
            # 웹드라이버 종료
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("웹드라이버가 정상적으로 종료되었습니다.")
                except:
                    logger.warning("웹드라이버 종료 중 오류가 발생했습니다.")
    
    def save_combined_data(self):
        """전체 기간의 데이터를 통합 파일로 저장"""
        try:
            all_files = [os.path.join(self.config.DATA_DIR, f) for f in os.listdir(self.config.DATA_DIR) if f.startswith('news_') and f.endswith('.csv')]
            
            if not all_files:
                logger.warning("통합할 데이터 파일이 없습니다.")
                return
            
            combined_data = []
            
            for file_path in all_files:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    combined_data.append(df)
                except Exception as e:
                    logger.error(f"파일 읽기 중 오류: {file_path}, {str(e)}")
            
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                
                # 중복 제거
                combined_df.drop_duplicates(subset=['title', 'date'], keep='first', inplace=True)
                
                # 결과 저장
                start_date = self.config.START_DATE.strftime("%Y%m%d")
                end_date = self.config.END_DATE.strftime("%Y%m%d")
                
                combined_file_path = f"{self.config.DATA_DIR}/all_news_{start_date}_to_{end_date}.csv"
                combined_df.to_csv(combined_file_path, index=False, encoding='utf-8-sig')
                
                logger.info(f"통합 데이터 저장 완료: {combined_file_path}, {len(combined_df)}개 항목")
                
                # 카테고리별 통계
                category_stats = combined_df['category'].value_counts().to_dict()
                for category, count in category_stats.items():
                    logger.info(f"카테고리 '{category}': {count}개 뉴스")
                
        except Exception as e:
            logger.error(f"통합 데이터 저장 중 오류: {str(e)}")


if __name__ == "__main__":
    try:
        logger.info("네이버 파이낸스 뉴스 크롤러를 시작합니다...")
        crawler = NaverFinanceCrawler()
        crawler.run()
    except Exception as e:
        logger.critical(f"프로그램 실행 중 치명적 오류: {str(e)}\n{traceback.format_exc()}")