import os
import time
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Mecab

# Mecab 초기화
mecab = Mecab('C:/mecab/share/mecab-ko-dic')

# 날짜 범위 설정
start_date = datetime.datetime(2025, 4, 1)
end_date = datetime.datetime(2025, 4, 27)
date_range = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# 카테고리 URL 매핑
categories = {
    "실시간 속보": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}",
    "주요뉴스": "https://finance.naver.com/news/mainnews.naver?date={date_hyphen}",
    "시황,전망": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}",
    "기업,종목분석": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}",
    "해외증시": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}",
    "채권,선물": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}",
    "공시,메모": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}",
    "환율": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=259&date={date}",
}

# 폴더 준비
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# TF-IDF 벡터라이저 초기화
vectorizer = TfidfVectorizer()
collected_titles = []
title_vectors = None

# Selenium 설정
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
wait = WebDriverWait(driver, 10)

def tokenize_text(text):
    tokens = mecab.pos(text)
    return [word for word, pos in tokens if pos.startswith('N') or pos.startswith('V')]

def is_duplicate_title(title, threshold=0.95):
    global title_vectors, vectorizer
    
    if not collected_titles:
        return False
    
    tokens = tokenize_text(title)
    new_title = ' '.join(tokens)
    
    try:
        new_vector = vectorizer.transform([new_title])
        similarities = cosine_similarity(new_vector, title_vectors)[0]
        return any(sim >= threshold for sim in similarities)
    except ValueError:
        all_titles = [' '.join(tokenize_text(t)) for t in collected_titles] + [new_title]
        vectorizer = TfidfVectorizer()
        title_vectors = vectorizer.fit_transform(all_titles[:-1])
        new_vector = vectorizer.transform([all_titles[-1]])
        similarities = cosine_similarity(new_vector, title_vectors)[0]
        return any(sim >= threshold for sim in similarities)

def extract_news_data(category, current_date_str, current_date_hyphen):
    news_data = []
    error_data = []
    
    if "주요뉴스" in category:
        url = categories[category].format(date_hyphen=current_date_hyphen)
    else:
        url = categories[category].format(date=current_date_str)

    driver.get(url)
    time.sleep(1)

    while True:
        try:
            li_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#contentarea_left ul li")))
            for li in li_elements:
                try:
                    title_element = li.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(1) a")
                    title = title_element.text.strip()
                    news_url = title_element.get_attribute("href")

                    summary_element = li.find_element(By.CSS_SELECTOR, "dl dd:nth-of-type(2)")
                    summary_text = summary_element.text.strip()

                    try:
                        time_element = summary_element.find_element(By.TAG_NAME, "span")
                        news_time = time_element.text.strip()
                    except NoSuchElementException:
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
                            "url": news_url
                        })
                except Exception as e:
                    error_data.append({
                        "date": current_date_hyphen,
                        "url": url,
                        "category": category,
                        "error": str(e),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        except TimeoutException:
            break

        # 다음 버튼 처리
        try:
            next_page = driver.find_element(By.LINK_TEXT, "다음")
            next_page.click()
            time.sleep(1)
        except NoSuchElementException:
            break

    return news_data, error_data

# 메인 프로세스
for current_date in date_range:
    current_date_str = current_date.strftime("%Y%m%d")
    current_date_hyphen = current_date.strftime("%Y-%m-%d")

    all_news_data = []
    all_error_data = []

    for category in categories.keys():
        news_data, error_data = extract_news_data(category, current_date_str, current_date_hyphen)
        all_news_data.extend(news_data)
        all_error_data.extend(error_data)
        time.sleep(0.5)

    if all_news_data:
        news_df = pd.DataFrame(all_news_data)
        news_df.to_csv(f"data/news_{current_date_str}.csv", index=False, encoding='utf-8-sig')

    if all_error_data:
        error_df = pd.DataFrame(all_error_data)
        error_df.to_csv(f"logs/errors_{current_date_str}.csv", index=False, encoding='utf-8-sig')

    print(f"완료: {current_date_hyphen} - 뉴스 {len(all_news_data)}개, 에러 {len(all_error_data)}개")

# 마무리
driver.quit()
print("전체 뉴스 크롤링 완료!")