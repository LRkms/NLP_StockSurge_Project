import os
import time
import random
import pandas as pd
import datetime
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab


class NaverFinanceNewsCrawler:
    def __init__(self, save_dir='./data'):
        self.save_dir = save_dir
        self.categories = {
            '실시간 속보': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}',
                'priority': 4},
            '주요뉴스': {'url': 'https://finance.naver.com/news/mainnews.naver?date={date}', 'priority': 5},
            '시황,전망': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}',
                'priority': 5},
            '기업,종목분석': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}',
                'priority': 2},
            '해외증시': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}',
                'priority': 1},
            '채권,선물': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}',
                'priority': 7},
            '공시,메모': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}',
                'priority': 3},
            '환율': {
                'url': 'https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=405&date={date}',
                'priority': 6}
        }
        self.title_xpaths = {
            '실시간 속보': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/a'],
            '주요뉴스': ['//*[@id="contentarea_left"]/div[2]/ul/li[1]/dl/dd[1]/a',
                     '//*[@id="contentarea_left"]/div[2]/ul/li[2]/dl/dd[1]/a'],
            '시황,전망': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a',
                      '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/a'],
            '기업,종목분석': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a',
                        '//*[@id="contentarea_left"]/ul/li[1]/dl/dt[2]/a'],
            '해외증시': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a',
                     '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/a'],
            '채권,선물': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a',
                      '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/a'],
            '공시,메모': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/a'],
            '환율': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[1]/a', '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/a']
        }
        self.summary_xpaths = {
            '실시간 속보': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]'],
            '주요뉴스': ['//*[@id="contentarea_left"]/div[2]/ul/li[1]/dl/dd[2]',
                     '//*[@id="contentarea_left"]/div[2]/ul/li[2]/dl/dd[2]'],
            '시황,전망': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]', '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]'],
            '기업,종목분석': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]',
                        '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]'],
            '해외증시': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]', '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]'],
            '채권,선물': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]', '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]'],
            '공시,메모1': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]'],
            '공시,메모2': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]'],
            '환율': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]', '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]']
        }
        self.time_xpaths = {
            '실시간 속보': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]/span[3]'],
            '주요뉴스': ['//*[@id="contentarea_left"]/div[2]/ul/li[1]/dl/dd[2]/span[3]',
                     '//*[@id="contentarea_left"]/div[2]/ul/li[2]/dl/dd[2]/span[3]'],
            '시황,전망': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                      '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]/span[3]'],
            '기업,종목분석': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                        '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[3]/span[3]'],
            '해외증시': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                     '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]/span[3]'],
            '채권,선물': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                      '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]/span[3]'],
            '공시,메모1': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                       '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]/span[3]'],
            '환율': ['//*[@id="contentarea_left"]/ul/li[1]/dl/dd[2]/span[3]',
                   '//*[@id="contentarea_left"]/ul/li[1]/dl/dd[4]/span[3]']
        }
        self.next_page_xpaths = {
            '실시간 속보': '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[2]/a',
            '주요뉴스': '//*[@id="contentarea_left"]/table/tbody/tr/td[2]/a',
            '시황,전망': '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[2]/a',
            '기업,종목분석': '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[2]/a',
            '해외증시': '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[3]/a',
            '채권,선물': None,  # 1페이지만 있음
            '공시,메모1': '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[3]/a',
            '환율': '//*[@id="contentarea_left"]/table/tbody/tr/td/table/tbody/tr/td[3]/a'
        }

        # 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # Mecab 초기화
        try:
            self.mecab = Mecab()
            print("Mecab 초기화 성공")
        except Exception as e:
            print(f"Mecab 초기화 실패: {e}")
            print("Mecab이 설치되어 있는지 확인하세요. (예: apt-get install mecab mecab-ko-dic)")
            self.mecab = None

        # TF-IDF 벡터라이저 초기화
        self.vectorizer = TfidfVectorizer()

        # 뉴스 데이터 저장용 데이터프레임
        self.news_data = pd.DataFrame(columns=['date', 'time', 'category', 'title', 'summary', 'url', 'keywords'])

        # 오류 로그 데이터프레임
        self.error_logs = pd.DataFrame(
            columns=['date', 'url', 'category', 'xpath', 'exception', 'details', 'title', 'page_number', 'timestamp'])

        # Selenium 설정
        self.setup_driver()

    def setup_driver(self):
        """Selenium 웹드라이버 설정"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 헤드리스 모드
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-popup-blocking')

        # User-Agent 설정
        chrome_options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36')

        # 웹드라이버 초기화
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.implicitly_wait(10)
        print("웹드라이버 초기화 완료")

    def extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        if self.mecab is None or not text:
            return []

        try:
            # 명사와 동사만 추출
            pos_tagged = self.mecab.pos(text)
            keywords = [word for word, pos in pos_tagged if pos.startswith('N') or pos.startswith('V')]

            # 출처 및 불필요한 단어 제거
            stop_words = ['연합뉴스', '뉴스', '기자', '특파원', '이데일리', '한국경제', '매일경제',
                          '헤럴드경제', '파이낸셜뉴스', '머니투데이', '서울경제', '아시아경제',
                          '조선비즈', '디지털타임스', '뉴스1', '뉴시스', '이투데이', '한국', '미국', '중국']

            keywords = [word for word in keywords if word not in stop_words and len(word) > 1]

            # 빈도수 기준 상위 10개 키워드 반환
            keyword_counts = {}
            for word in keywords:
                if word in keyword_counts:
                    keyword_counts[word] += 1
                else:
                    keyword_counts[word] = 1

            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [word for word, count in sorted_keywords[:10]]

            return top_keywords
        except Exception as e:
            print(f"키워드 추출 오류: {e}")
            return []

    def check_duplicate_news(self, new_title):
        """코사인 유사도를 이용한 중복 뉴스 확인"""
        if len(self.news_data) == 0 or not new_title:
            return False

        existing_titles = self.news_data['title'].tolist()

        # 제목이 정확히 일치하는 경우
        if new_title in existing_titles:
            return True

        try:
            # TF-IDF 벡터화 및 코사인 유사도 계산
            if len(existing_titles) > 0:
                all_titles = existing_titles + [new_title]
                tfidf_matrix = self.vectorizer.fit_transform(all_titles)
                cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

                # 유사도가 0.95 이상인 뉴스가 있으면 중복으로 처리
                if max(cosine_similarities) >= 0.95:
                    return True
        except Exception as e:
            print(f"유사도 계산 오류: {e}")

        return False

    def log_error(self, date, url, category, xpath, exception, details='', title='', page_number=0):
        """오류 로깅"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_data = {
            'date': date,
            'url': url,
            'category': category,
            'xpath': xpath,
            'exception': str(exception),
            'details': details,
            'title': title,
            'page_number': page_number,
            'timestamp': now
        }
        self.error_logs = pd.concat([self.error_logs, pd.DataFrame([error_data])], ignore_index=True)

    def save_error_logs(self, date_str):
        """오류 로그 저장"""
        if not self.error_logs.empty:
            error_log_path = os.path.join(self.save_dir, f"errors_{date_str}.csv")
            self.error_logs.to_csv(error_log_path, index=False, encoding='utf-8-sig')
            print(f"오류 로그 저장 완료: {error_log_path}")

    def save_news_data(self, date_str):
        """뉴스 데이터 저장"""
        if not self.news_data.empty:
            news_log_path = os.path.join(self.save_dir, f"news_log_{date_str}.csv")
            self.news_data.to_csv(news_log_path, index=False, encoding='utf-8-sig')
            print(f"뉴스 데이터 저장 완료: {news_log_path}")

    def crawl_news_by_category(self, date_str, category):
        """카테고리별 뉴스 크롤링"""
        try:
            formatted_date = date_str.replace('-', '')  # YYYYMMDD 형식으로 변환
            url = self.categories[category]['url'].format(date=formatted_date)
            self.driver.get(url)

            # 페이지 로딩 대기
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, self.title_xpaths[category][0]))
            )

            page_number = 1
            news_count = 0

            # 모든 페이지 순회
            while True:
                print(f"{category} 카테고리 {page_number}페이지 크롤링 중...")

                # 뉴스 아이템 크롤링
                for i, (title_xpath, summary_xpath, time_xpath) in enumerate(zip(
                        self.title_xpaths[category],
                        self.summary_xpaths[category],
                        self.time_xpaths[category]
                )):
                    try:
                        # 뉴스 제목 추출
                        title_element = self.driver.find_element(By.XPATH, title_xpath)
                        title = title_element.text.strip() if title_element else ""

                        # 빈 제목은 건너뛰기
                        if not title:
                            continue

                        # 뉴스 URL 추출
                        news_url = title_element.get_attribute('href') if title_element else ""

                        # 요약 추출
                        summary_element = self.driver.find_element(By.XPATH, summary_xpath)
                        summary = summary_element.text.strip() if summary_element else ""

                        # 시간 추출 (필수)
                        try:
                            time_element = self.driver.find_element(By.XPATH, time_xpath)
                            news_time = time_element.text.strip() if time_element else ""

                            # 시간이 없으면 건너뛰기
                            if not news_time:
                                self.log_error(date_str, url, category, time_xpath, "시간 정보 없음", title=title,
                                               page_number=page_number)
                                continue
                        except (NoSuchElementException, StaleElementReferenceException) as e:
                            self.log_error(date_str, url, category, time_xpath, e, title=title, page_number=page_number)
                            continue

                        # 중복 확인
                        if self.check_duplicate_news(title):
                            print(f"중복 뉴스 제외: {title}")
                            continue

                        # 키워드 추출
                        keywords = self.extract_keywords(title + " " + summary)
                        keywords_str = ", ".join(keywords)

                        # 뉴스 데이터 저장
                        news_item = {
                            'date': date_str,
                            'time': news_time,
                            'category': category,
                            'title': title,
                            'summary': summary,
                            'url': news_url,
                            'keywords': keywords_str
                        }

                        self.news_data = pd.concat([self.news_data, pd.DataFrame([news_item])], ignore_index=True)
                        news_count += 1

                    except (NoSuchElementException, StaleElementReferenceException) as e:
                        self.log_error(date_str, url, category, f"{title_xpath} or {summary_xpath}", e,
                                       page_number=page_number)
                        continue

                # 다음 페이지 확인
                if self.next_page_xpaths[category] is None:
                    break

                try:
                    next_page_elements = self.driver.find_elements(By.XPATH, self.next_page_xpaths[category])

                    # 다음 페이지가 없거나 현재 페이지가 마지막인 경우
                    if not next_page_elements:
                        break

                    # 다음 페이지 버튼 클릭
                    found_next = False
                    for element in next_page_elements:
                        if element.text == "다음페이지":
                            element.click()
                            found_next = True
                            page_number += 1
                            time.sleep(random.uniform(1.0, 2.0))  # 랜덤 대기
                            break

                    if not found_next:
                        break

                except Exception as e:
                    self.log_error(date_str, url, category, self.next_page_xpaths[category], e, "다음 페이지 이동 오류",
                                   page_number=page_number)
                    break

            print(f"{category} 카테고리 크롤링 완료: {news_count}개 뉴스 수집")
            return news_count

        except Exception as e:
            self.log_error(date_str, url, category, "전체 카테고리", e, "카테고리 크롤링 중 오류 발생")
            print(f"{category} 카테고리 크롤링 오류: {e}")
            return 0

    def crawl_news_by_date(self, date_str):
        """날짜별 뉴스 크롤링"""
        print(f"=== {date_str} 뉴스 크롤링 시작 ===")

        total_news_count = 0

        # 카테고리별 크롤링
        for category in tqdm(self.categories.keys()):
            try:
                news_count = self.crawl_news_by_category(date_str, category)
                total_news_count += news_count

                # 카테고리간 크롤링 간격
                time.sleep(random.uniform(2.0, 3.0))

            except Exception as e:
                print(f"{category} 카테고리 크롤링 중 오류 발생: {e}")

        # 결과 저장
        self.save_news_data(date_str)
        self.save_error_logs(date_str)

        print(f"=== {date_str} 뉴스 크롤링 완료: 총 {total_news_count}개 뉴스 수집 ===")
        return total_news_count

    def crawl_date_range(self, start_date, end_date):
        """날짜 범위 크롤링"""
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        date_list = [(start + datetime.timedelta(days=x)).strftime("%Y-%m-%d")
                     for x in range((end - start).days + 1)]

        total_count = 0

        for date_str in date_list:
            # 데이터프레임 초기화
            self.news_data = pd.DataFrame(columns=['date', 'time', 'category', 'title', 'summary', 'url', 'keywords'])
            self.error_logs = pd.DataFrame(
                columns=['date', 'url', 'category', 'xpath', 'exception', 'details', 'title', 'page_number',
                         'timestamp'])

            # 날짜별 크롤링
            count = self.crawl_news_by_date(date_str)
            total_count += count

            # 날짜간 크롤링 간격
            time.sleep(random.uniform(5.0, 10.0))

        print(f"=== 전체 크롤링 완료: {start_date} ~ {end_date}, 총 {total_count}개 뉴스 수집 ===")

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'driver'):
            self.driver.quit()
            print("웹드라이버 종료")


# 크롤러 사용 예시
if __name__ == "__main__":
    # 크롤러 초기화
    crawler = NaverFinanceNewsCrawler(save_dir='./crawled_data')

    # 특정 날짜 범위 크롤링
    try:
        # 2025년 1월 1일부터 2025년 4월 22일까지
        crawler.crawl_date_range("2025-01-01", "2025-04-22")
    except KeyboardInterrupt:
        print("사용자에 의해 크롤링이 중단되었습니다.")
    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")
    finally:
        # 리소스 정리
        crawler.close()