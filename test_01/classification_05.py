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
    CHROME_OPTIONS = ["--headless", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu", "--log-level=3"]
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    DRIVER_TIMEOUT = 10
    PAGE_LOAD_WAIT = 1.5
    MAX_RETRIES = 3

class NaverFinanceCrawler:
    def __init__(self, config=NaverNewsConfig):
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
        os.makedirs(self.config.DATA_DIR, exist_ok=True)
        os.makedirs(self.config.LOGS_DIR, exist_ok=True)

    def initialize_webdriver(self):
        try:
            chrome_options = Options()
            for option in self.config.CHROME_OPTIONS:
                chrome_options.add_argument(option)
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.wait = WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT)
            logger.info("웹드라이버 초기화 완료")
        except Exception as e:
            logger.error(f"웹드라이버 오류: {e}")
            raise

    def reset_webdriver(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        self.initialize_webdriver()

    def tokenize_text(self, text):
        try:
            tokens = self.mecab.pos(text)
            return [word for word, pos in tokens if pos.startswith('N') or pos.startswith('V')]
        except:
            return [w for w in text.split() if len(w) > 1]

    def is_duplicate_title(self, title):
        if not self.collected_titles:
            return False
        tokens = self.tokenize_text(title)
        new_title = ' '.join(tokens)
        try:
            new_vector = self.vectorizer.transform([new_title])
            similarities = cosine_similarity(new_vector, self.title_vectors)[0]
            return any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities)
        except ValueError:
            all_titles = [' '.join(self.tokenize_text(t)) for t in self.collected_titles] + [new_title]
            self.vectorizer = TfidfVectorizer()
            self.title_vectors = self.vectorizer.fit_transform(all_titles[:-1])
            new_vector = self.vectorizer.transform([all_titles[-1]])
            similarities = cosine_similarity(new_vector, self.title_vectors)[0]
            return any(sim >= self.config.DUPLICATE_THRESHOLD for sim in similarities)

    def update_title_vectors(self, new_title):
        self.collected_titles.append(new_title)
        if self.title_vectors is None or len(self.collected_titles) % 100 == 0:
            all_titles = [' '.join(self.tokenize_text(t)) for t in self.collected_titles]
            self.vectorizer = TfidfVectorizer()
            self.title_vectors = self.vectorizer.fit_transform(all_titles)

    @backoff.on_exception(backoff.expo, (WebDriverException, TimeoutException), max_tries=NaverNewsConfig.MAX_RETRIES)
    def get_page(self, url):
        self.driver.get(url)
        WebDriverWait(self.driver, self.config.DRIVER_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )
        time.sleep(self.config.PAGE_LOAD_WAIT)

    def parse_news_item(self, li_element, category, date_hyphen):
        try:
            title_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(1) a")
            title = title_element.text.strip()
            news_url = title_element.get_attribute("href")
            summary_element = li_element.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(2)")
            summary_text = summary_element.text.strip()
            try:
                time_element = summary_element.find_element(By.TAG_NAME, "span")
                news_time = time_element.text.strip()
                summary_text = summary_text.replace(news_time, '').strip()
            except:
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
        except:
            return None

    def extract_news_data(self, category, date_str, date_hyphen):
        url = self.config.CATEGORIES[category].format(date=date_str, date_hyphen=date_hyphen)
        logger.info(f"{category} 크롤링 중...")
        news_data = []
        self.get_page(url)
        try:
            li_elements = self.wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#contentarea_left ul li")))
            for li in li_elements:
                news = self.parse_news_item(li, category, date_hyphen)
                if news:
                    news_data.append(news)
        except:
            logger.warning(f"{category} 뉴스 없음")
        return news_data

    def save_data(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pd.DataFrame(data).to_csv(file_path, index=False, encoding='utf-8-sig')

    def crawl_by_date(self, date):
        date_str = date.strftime("%Y%m%d")
        date_hyphen = date.strftime("%Y-%m-%d")
        all_news = []
        for category in self.config.CATEGORIES:
            news = self.extract_news_data(category, date_str, date_hyphen)
            all_news.extend(news)
            time.sleep(1.0)
        if all_news:
            self.save_data(all_news, f"{self.config.DATA_DIR}/news_{date_str}.csv")
        return all_news

    def save_combined_data(self):
        files = [os.path.join(self.config.DATA_DIR, f) for f in os.listdir(self.config.DATA_DIR) if f.startswith("news_")]
        dfs = [pd.read_csv(f, encoding='utf-8-sig') for f in files if os.path.isfile(f)]
        combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['title', 'date'])
        start = self.config.START_DATE.strftime("%Y%m%d")
        end = self.config.END_DATE.strftime("%Y%m%d")
        path = f"{self.config.DATA_DIR}/all_news_{start}_to_{end}.csv"
        combined.to_csv(path, index=False, encoding='utf-8-sig')
        logger.info(f"총 {len(combined)}개 뉴스 저장 완료")

    def run(self):
        try:
            start_time = time.time()
            date_range = [self.config.START_DATE + datetime.timedelta(days=i) for i in range((self.config.END_DATE - self.config.START_DATE).days + 1)]
            for date in date_range:
                self.crawl_by_date(date)
            self.save_combined_data()
            logger.info(f"총 소요 시간: {time.time() - start_time:.2f}초")
        finally:
            if self.driver:
                self.driver.quit()

if __name__ == '__main__':
    crawler = NaverFinanceCrawler()
    crawler.run()