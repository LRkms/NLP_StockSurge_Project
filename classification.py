import os
import time
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
import traceback


# Mecab 초기화
mecab = Mecab('C:/mecab/share/mecab-ko-dic')

# 날짜 범위 설정 (2025년 4월 1일 ~ 2025년 4월 27일)
start_date = datetime.datetime(2025, 4, 1)
end_date = datetime.datetime(2025, 4, 27)
date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# 카테고리별 URL 및 XPath 정보
categories = {
    "실시간 속보": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "// interdisciplinary[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 4,
        "max_rows": 20,
    },
    "주요뉴스": {
        "url_format": "https://finance.naver.com/news/mainnews.naver?date={date_hyphen}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/div[2]/ul/li[{row}]/dl/dd[1]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/div[2]/ul/li[{row}]/dl/dd[2]",
        "time_xpath_format": "//*[@id='contentarea_left']/div[2]/ul/li[{row}]/dl/dd[2]/span[3]",
        "priority": 3,
        "max_rows": 20,
    },
    "시황,전망": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 5,
        "max_rows": 20,
    },
    "기업,종목분석": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dt[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]",
        "time_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/span[3]",
        "priority": 2,
        "max_rows": 20,
    },
    "해외증시": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td/table/tbody/tr/td[3]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 1,
        "max_rows": 20,
    },
    "채권,선물": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}",
        "page_xpath": None,
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 7,
        "max_rows": 20,
    },
    "공시,메모": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td/table/tbody/tr/td[3]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 3,
        "max_rows": 20,
    },
    "환율": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=259&section_id3=&date={date}",
        "page_xpath": "//*[@id='contentarea_left']/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id='contentarea_left']/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id='contentarea_left']//ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 6,
        "max_rows": 20,
    },
}

# 데이터 폴더 생성
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 중복 뉴스 감지를 위한 TF-IDF 벡터라이저 초기화
vectorizer = TfidfVectorizer()
collected_titles = []
title_vectors = None

# Selenium 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# WebDriver 초기화
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def tokenize_text(text):
    """텍스트를 토큰화하여 명사와 동사만 추출"""
    tokens = mecab.pos(text)
    return [word for word, pos in tokens if pos.startswith('N') or pos.startswith('V')]

def is_duplicate_title(title, threshold=0.95):
    """중복 제목 확인 (코사인 유사도 기반)"""
    global title_vectors, vectorizer
    
    if not collected_titles:
        return False
    
    tokens = tokenize_text(title)
    new_title = ' '.join(tokens)
    
    try:
        new_vector = vectorizer.transform([new_title])
        if title_vectors is None:
            title_vectors = vectorizer.fit_transform([' '.join(tokenize_text(t)) for t in collected_titles])
        similarities = cosine_similarity(new_vector, title_vectors)[0]
        return any(sim >= threshold for sim in similarities)
    except ValueError:
        # 단어 집합 불일치 시 벡터라이저 재학습
        all_titles = [' '.join(tokenize_text(t)) for t in collected_titles] + [new_title]
        vectorizer = TfidfVectorizer()
        title_vectors = vectorizer.fit_transform(all_titles[:-1])
        new_vector = vectorizer.transform([all_titles[-1]])
        similarities = cosine_similarity(new_vector, title_vectors)[0]
        return any(sim >= threshold for sim in similarities)

def extract_news_data(category, current_date_str, current_date_hyphen):
    """특정 카테고리와 날짜에 대한 뉴스 데이터 수집"""
    news_data = []
    error_data = []

    cat_info = categories[category]

    if "date_hyphen" in cat_info["url_format"]:
        url = cat_info["url_format"].format(date_hyphen=current_date_hyphen)
    else:
        url = cat_info["url_format"].format(date=current_date_str)

    print(f"크롤링 시작: {category} - {current_date_hyphen}")

    try:
        driver.get(url)
        time.sleep(1)

        page_num = 1
        while True:
            try:
                # li 전체 찾기
                li_elements = driver.find_elements(By.XPATH, "//*[@id='contentarea_left']/ul/li")

                for li in li_elements:
                    dd_elements = li.find_elements(By.TAG_NAME, "dd")
                    dd_count = len(dd_elements)

                    idx = 0
                    while idx < dd_count:
                        try:
                            # 제목
                            title_element = dd_elements[idx].find_element(By.TAG_NAME, "a")
                            title = title_element.text.strip()
                            news_url = title_element.get_attribute("href")

                            # 요약 및 시간
                            if idx + 1 < dd_count:
                                summary_element = dd_elements[idx + 1]
                                summary_text = summary_element.text.strip()

                                try:
                                    time_element = summary_element.find_element(By.TAG_NAME, "span")
                                    news_time = time_element.text.strip()
                                except NoSuchElementException:
                                    news_time = ""
                            else:
                                summary_text = ""
                                news_time = ""

                            if not is_duplicate_title(title):
                                collected_titles.append(title)
                                keywords = tokenize_text(title + " " + summary_text)
                                keywords_str = ", ".join(set(keywords))

                                news_data.append({
                                    "date": current_date_hyphen,
                                    "time": news_time,
                                    "category": category,
                                    "title": title,
                                    "summary": summary_text,
                                    "keywords": keywords_str,
                                    "url": news_url,
                                    "page": page_num
                                })
                        except Exception as e:
                            error_data.append({
                                "date": current_date_hyphen,
                                "url": url,
                                "category": category,
                                "xpath": "",
                                "exception": str(type(e).__name__),
                                "details": str(e),
                                "title": "",
                                "page_number": page_num,
                                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        idx += 2  # 다음 뉴스로 넘어가기 (2칸 이동)

            except Exception as e:
                error_data.append({
                    "date": current_date_hyphen,
                    "url": url,
                    "category": category,
                    "xpath": "",
                    "exception": str(type(e).__name__),
                    "details": str(e),
                    "title": "",
                    "page_number": page_num,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            if cat_info["page_xpath"] is None:
                break

            try:
                next_page = driver.find_element(By.XPATH, cat_info["page_xpath"])
                if "다음" in next_page.text:
                    next_page.click()
                    time.sleep(1)
                    page_num += 1
                else:
                    break
            except NoSuchElementException:
                break

    except Exception as e:
        error_data.append({
            "date": current_date_hyphen,
            "url": url,
            "category": category,
            "xpath": "",
            "exception": str(type(e).__name__),
            "details": str(e),
            "title": "",
            "page_number": 1,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    print(f"완료: {category} - {current_date_hyphen}, 수집된 뉴스: {len(news_data)}, 오류: {len(error_data)}")
    return news_data, error_data



# 메인 크롤링 프로세스
for current_date in date_range:
    current_date_str = current_date.strftime("%Y%m%d")
    current_date_hyphen = current_date.strftime("%Y-%m-%d")
    
    all_news_data = []
    all_error_data = []
    
    for category in categories:
        news_data, error_data = extract_news_data(category, current_date_str, current_date_hyphen)
        all_news_data.extend(news_data)
        all_error_data.extend(error_data)
        time.sleep(0.5)
    
    if all_news_data:
        news_df = pd.DataFrame(all_news_data)
        news_df.to_csv(f"data/news_{current_date_str}.csv", index=False, encoding='utf-8-sig')
        
        news_log_df = news_df[['date', 'time', 'category', 'title', 'summary', 'url']]
        news_log_df.to_csv(f"logs/news_log_{current_date_str}.csv", index=False, encoding='utf-8-sig')
    
    if all_error_data:
        error_df = pd.DataFrame(all_error_data)
        error_df.to_csv(f"logs/errors_{current_date_str}.csv", index=False, encoding='utf-8-sig')
    
    print(f"날짜 완료: {current_date_hyphen}, 총 뉴스 수집: {len(all_news_data)}, 총 오류: {len(all_error_data)}")
    time.sleep(1)

# WebDriver 종료
driver.quit()

print("뉴스 크롤링 작업 완료!")