import os
import time
import datetime
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab
import traceback
from datetime import datetime, timedelta
import financedatareader as fdr

# Mecab 초기화
mecab = Mecab()

# 날짜 범위 설정 (2025년 1월 1일부터 오늘까지)
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 4, 25)
date_range = []
current_date = start_date

while current_date <= end_date:
    date_range.append(current_date)
    current_date += timedelta(days=1)

# 카테고리별 URL 및 XPath 정보
categories = {
    "실시간 속보": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 4,
        "max_rows": 20,
    },
    "주요뉴스": {
        "url_format": "https://finance.naver.com/news/mainnews.naver?date={date_hyphen}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/div[2]/ul/li[{row}]/dl/dd[1]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/div[2]/ul/li[{row}]/dl/dd[2]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/div[2]/ul/li[{row}]/dl/dd[2]/span[3]",
        "priority": 3,
        "max_rows": 20,
    },
    "시황,전망": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 5,
        "max_rows": 20,
    },
    "기업,종목분석": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dt[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/span[3]",
        "priority": 2,
        "max_rows": 20,
    },
    "해외증시": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td/table/tbody/tr/td[3]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 1,
        "max_rows": 20,
    },
    "채권,선물": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}",
        "page_xpath": None,
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 7,
        "max_rows": 20,
    },
    "공시,메모": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td/table/tbody/tr/td[3]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
        "priority": 3,
        "max_rows": 20,
    },
    "환율": {
        "url_format": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=259&section_id3=&date={date}",
        "page_xpath": "//*[@id=\"contentarea_left\"]/table/tbody/tr/td/table/tbody/tr/td[2]/a",
        "title_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index}]/a",
        "summary_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]",
        "time_xpath_format": "//*[@id=\"contentarea_left\"]/ul/li[{row}]/dl/dd[{index+1}]/span[3]",
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
    global title_vectors
    
    if not collected_titles:
        return False
    
    tokens = tokenize_text(title)
    new_title = ' '.join(tokens)
    
    if title_vectors is None:
        title_vectors = vectorizer.fit_transform([' '.join(tokenize_text(t)) for t in collected_titles])
        new_vector = vectorizer.transform([new_title])
    else:
        try:
            new_vector = vectorizer.transform([new_title])
        except ValueError:
            all_titles = [' '.join(tokenize_text(t)) for t in collected_titles]
            all_titles.append(new_title)
            vectorizer_new = TfidfVectorizer()
            title_vectors = vectorizer_new.fit_transform(all_titles[:-1])
            new_vector = vectorizer_new.transform([all_titles[-1]])
    
    similarities = cosine_similarity(new_vector, title_vectors)[0]
    return any(sim >= threshold for sim in similarities)

def is_within_time_window(news_time, current_date):
    """뉴스가 지정된 시간 범위(이전 날 8:00 AM ~ 해당 날 8:00 AM)에 있는지 확인"""
    try:
        news_datetime = datetime.strptime(news_time, "%Y-%m-%d %H:%M")
        start_time = current_date - timedelta(days=1) + timedelta(hours=8)
        end_time = current_date + timedelta(hours=8)
        return start_time <= news_datetime < end_time
    except ValueError:
        return False

def extract_news_data(category, current_date_str, current_date_hyphen, current_date):
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
            for row in range(1, cat_info["max_rows"] + 1):
                for index in range(1, 4, 2):
                    try:
                        title_xpath = cat_info["title_xpath_format"].format(row=row, index=index)
                        title_element = driver.find_element(By.XPATH, title_xpath)
                        title = title_element.text.strip()
                        news_url = title_element.get_attribute("href")
                        
                        summary_xpath = cat_info["summary_xpath_format"].format(row=row, index=index+1)
                        summary_element = driver.find_element(By.XPATH, summary_xpath)
                        summary_text = summary_element.text.strip()
                        
                        try:
                            time_xpath = cat_info["time_xpath_format"].format(row=row, index=index+1)
                            time_element = driver.find_element(By.XPATH, time_xpath)
                            news_time = time_element.text.strip()
                            
                            # 시간 범위 체크
                            if not is_within_time_window(news_time, current_date):
                                continue
                            
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
                        except NoSuchElementException:
                            error_data.append({
                                "date": current_date_hyphen,
                                "url": url,
                                "category": category,
                                "xpath": time_xpath,
                                "exception":                                "details": "시간 정보 없음",
                                "title": title if 'title' in locals() else "",
                                "page_number": page_num,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    except Exception as e:
                        error_info = traceback.format_exc()
                        error_data.append({
                            "date": current_date_hyphen,
                            "url": url,
                            "category": category,
                            "xpath": title_xpath if 'title_xpath' in locals() else "",
                            "exception": str(type(e).__name__),
                            "details": str(e),
                            "title": title if 'title' in locals() else "",
                            "page_number": page_num,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        error_info = traceback.format_exc()
        error_data.append({
            "date": current_date_hyphen,
            "url": url,
            "category": category,
            "xpath": "",
            "exception": str(type(e).__name__),
            "details": str(e),
            "title": "",
            "page_number": page_num if 'page_num' in locals() else 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    print(f"완료: {category} - {current_date_hyphen}, 수집된 뉴스: {len(news_data)}, 오류: {len(error_data)}")
    
    return news_data, error_data

# 환율 민감 주식 라벨링
def label_sensitive_stocks():
    tickers = ['005930.KS', '000660.KS']  # 예시: 코스피 Top 20, 코스닥 Top 10 추가
    start_date = '2025-01-01'
    end_date = '2025-04-22'
    
    usd_krw = fdr.DataReader('USD/KRW', start_date, end_date)['Close']
    correlations = []
    
    for ticker in tickers:
        stock_data = fdr.DataReader(ticker, start_date, end_date)['Close']
        if len(stock_data) == len(usd_krw):
            correlation = stock_data.corr(usd_krw)
            label = "민감" if correlation >= 0.5 else "비민감"
            correlations.append({
                "ticker": ticker,
                "correlation": correlation,
                "label": label
            })
    
    correlation_df = pd.DataFrame(correlations)
    correlation_df.to_csv("data/correlation.csv", index=False, encoding='utf-8-sig')
    print("환율 민감 주식 라벨링 완료")

# 메인 크롤링 프로세스
for current_date in date_range:
    current_date_str = current_date.strftime("%Y%m%d")
    current_date_hyphen = current_date.strftime("%Y-%m-%d")
    
    all_news_data = []
    all_error_data = []
    
    for category in categories:
        news_data, error_data = extract_news_data(category, current_date_str, current_date_hyphen, current_date)
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

# 환율 민감 주식 라벨링 실행
label_sensitive_stocks()

# WebDriver 종료
driver.quit()

print("전체 크롤링 작업 완료!")