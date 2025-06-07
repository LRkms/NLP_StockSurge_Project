import os
import time
import datetime
import logging
import pickle
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException, WebDriverException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
import backoff
import gc


# ---------- Configuration ----------
@dataclass
class NaverNewsConfig:
    mecab_path: str = 'C:/mecab/share/mecab-ko-dic'
    start_date: datetime.datetime = datetime.datetime(2025, 4, 1)
    end_date: datetime.datetime = datetime.datetime(2025, 4, 29)
    categories: dict = field(default_factory=lambda: {
        # ... same mapping ...
    })
    duplicate_threshold: float = 0.90
    cache_dir: str = 'cache'
    data_dir: str = 'data'
    logs_dir: str = 'logs'
    driver_timeout: int = 15
    page_load_wait: float = 1.0
    max_workers: int = 3
    batch_size: int = 100
    save_interval: int = 1
    max_retries: int = 5
    base_backoff: int = 2
    max_backoff: int = 30
    chrome_options: list = field(default_factory=lambda: [
        '--headless=new', '--no-sandbox', '--disable-dev-shm-usage',
        '--disable-gpu', '--ignore-certificate-errors'
    ])


# ---------- Logger Setup ----------
def setup_logger(logs_dir: str) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"crawler_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    fmt = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    for lib in ['selenium', 'urllib3', 'webdriver_manager']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


class NaverFinanceCrawler:
    def __init__(self, config: NaverNewsConfig):
        self.config = config
        self.logger = setup_logger(config.logs_dir)
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)

        # Caching
        self.cache_path = os.path.join(config.cache_dir, 'titles_cache.pkl')
        self.title_hashes = set()
        self.collected_titles = deque(maxlen=5000)
        self._load_cache()

        # TF-IDF
        self.vectorizer = None
        self.title_vectors = None

        # Selenium init
        self.driver = None
        self.wait = None
        self._init_driver()

        # Stats
        self.stats = {
            'pages': 0, 'collected': 0,
            'duplicates': 0, 'errors': 0
        }

    def _load_cache(self):
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.collected_titles.extend(data.get('titles', []))
                self.title_hashes = set(data.get('hashes', []))
                self.logger.info(f"Loaded {len(self.collected_titles)} titles from cache.")
        except Exception as e:
            self.logger.warning(f"Cache load failed: {e}")

    def _save_cache(self):
        try:
            data = {'titles': list(self.collected_titles), 'hashes': list(self.title_hashes)}
            with open(self.cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info("Cache saved.")
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")

    def _init_driver(self):
        try:
            opts = Options()
            for opt in self.config.chrome_options:
                opts.add_argument(opt)
            opts.page_load_strategy = 'eager'
            prefs = {"download.prompt_for_download": False}
            opts.add_experimental_option("prefs", prefs)

            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=opts
            )
            self.driver.implicitly_wait(2)
            self.wait = WebDriverWait(self.driver, self.config.driver_timeout)
            self.logger.info("WebDriver initialized.")
        except Exception as e:
            self.logger.error(f"Driver init failed: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.quit()
        self._save_cache()

    def __del__(self):
        self.close()

    def _hash(self, text: str) -> str:
        return sha256(text.encode()).hexdigest()

    @lru_cache(maxsize=1024)
    def _tokenize(self, text: str):
        try:
            mecab = Mecab(self.config.mecab_path)
            tokens = mecab.pos(text)
            return [w for w, p in tokens if p.startswith(('N', 'V'))]
        except Exception:
            return text.split()

    def _is_duplicate(self, title: str) -> bool:
        h = self._hash(title)
        if h in self.title_hashes:
            self.stats['duplicates'] += 1
            return True
        if self.vectorizer and len(self.collected_titles) >= 10:
            vec = self.vectorizer.transform([' '.join(self._tokenize(title))])
            sims = cosine_similarity(vec, self.title_vectors)[0]
            if np.max(sims) >= self.config.duplicate_threshold:
                self.title_hashes.add(h)
                self.stats['duplicates'] += 1
                return True
        return False

    def _update_vectors(self, title: str):
        h = self._hash(title)
        self.title_hashes.add(h)
        self.collected_titles.append(title)

        tokens = [' '.join(self._tokenize(t)) for t in self.collected_titles]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.title_vectors = self.vectorizer.fit_transform(tokens)

    @backoff.on_exception(
        backoff.expo,
        (WebDriverException, TimeoutException, StaleElementReferenceException),
        max_tries=lambda self: self.config.max_retries,
        base=lambda self: self.config.base_backoff,
        max_time=lambda self: self.config.max_backoff
    )
    def _load_page(self, url: str):
        self.driver.get(url)
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'body')))
        time.sleep(self.config.page_load_wait)
        self.stats['pages'] += 1

    def _parse_item(self, li, cat, date_str):
        try:
            a = li.find_element(By.CSS_SELECTOR, 'dl dd:nth-of-type(1) a')
            title = a.text.strip()
            if not title or self._is_duplicate(title):
                return None
            self._update_vectors(title)

            link = a.get_attribute('href')
            summary = ''
            try:
                summary = li.find_element(By.CSS_SELECTOR, 'dl dd:nth-of-type(2)').text
            except NoSuchElementException:
                pass

            time_el = None
            try:
                time_el = li.find_element(By.TAG_NAME, 'span').text
            except Exception:
                pass

            self.stats['collected'] += 1
            return {
                'date': date_str,
                'category': cat,
                'title': title,
                'summary': summary.replace(time_el or '', '').strip(),
                'url': link,
                'collected_at': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.debug(f"Parse error: {e}")
            return None

    def extract(self):
        results = []
        date = self.config.start_date
        while date <= self.config.end_date:
            ymd = date.strftime('%Y%m%d')
            y_m_d = date.strftime('%Y-%m-%d')
            for cat, tmpl in self.config.categories.items():
                url = tmpl.format(date=ymd, date_hyphen=y_m_d)
                self.logger.info(f"Crawling {cat} - {y_m_d}")
                try:
                    self._load_page(url)
                    lis = self.driver.find_elements(By.CSS_SELECTOR, '#contentarea_left ul li')
                    for li in lis:
                        item = self._parse_item(li, cat, y_m_d)
                        if item:
                            results.append(item)
                            if len(results) % self.config.batch_size == 0:
                                pd.DataFrame(results).to_csv(
                                    os.path.join(self.config.data_dir, f'{y_m_d}_{cat}.csv'), index=False)
                    if date.day % self.config.save_interval == 0:
                        pd.DataFrame(results).to_csv(
                            os.path.join(self.config.data_dir, f'interim_{y_m_d}.csv'), index=False)
                except Exception as e:
                    self.logger.warning(f"Failed {cat} on {y_m_d}: {e}")
            date += datetime.timedelta(days=1)

        # final save
        pd.DataFrame(results).to_csv(
            os.path.join(self.config.data_dir, 'all_news.csv'), index=False)
        self.logger.info(f"Done: {len(results)} items, stats={self.stats}")
        return results


if __name__ == '__main__':
    cfg = NaverNewsConfig()
    crawler = NaverFinanceCrawler(cfg)
    try:
        crawler.extract()
    finally:
        crawler.close()
