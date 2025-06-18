import os
import time
import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import logging
import requests
from bs4 import BeautifulSoup
import urllib.parse

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 저장 디렉토리 생성
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 날짜 설정
start_date = datetime.datetime(2025, 5, 1)
end_date = datetime.datetime(2025, 5, 31)
date_list = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# 크롤링 대상 카테고리
categories = {
    "실시간 속보": "https://finance.naver.com/news/news_list.naver?mode=LSS2D&section_id=101&section_id2=258&date={date}",
    "시황·전망": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401&date={date}",
    "기업·종목분석": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=402&date={date}",
    "해외증시": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=403&date={date}",
    "채권·선물": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=404&date={date}",
    "공시·메모": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=406&date={date}",
    "환율": "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=429&date={date}"
}

# 드라이버 설정
def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # 더 빠른 로딩을 위한 설정
    options.add_argument("--disable-images")
    options.add_argument("--disable-javascript")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=VizDisplayCompositor")
    
    # 타임아웃 설정
    options.add_argument("--page-load-strategy=eager")  # DOM 로드 후 바로 제어권 반환
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(10)  # 페이지 로드 타임아웃 10초
    driver.implicitly_wait(3)  # 암묵적 대기 3초
    
    return driver

# 개선된 본문 수집 (네이버 뉴스 구조 반영)
def collect_body_fast(link):
    """requests를 사용한 빠른 본문 수집 - 네이버 뉴스 구조 최적화"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://finance.naver.com/'
        }
        
        # 세션 사용으로 쿠키 유지
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(link, timeout=10, allow_redirects=True)
        response.raise_for_status()
        
        # 인코딩 설정
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 뉴스 구조별 본문 셀렉터 (우선순위별로 정렬)
        selectors = [
            # 새로운 네이버 뉴스 구조 (n.news.naver.com)
            "#dic_area",
            ".go_trans._article_content", 
            "#articleBodyContents",
            ".news_article .article_body",
            "#newsEndContents",
            
            # 기존 구조
            ".articleCont", 
            "#news_read",
            ".article_body",
            "#content",
            ".news_article",
            ".article-body",
            
            # 추가 구조
            "#main_content",
            ".article_wrap .article_body",
            "#articeBody",
            ".article_txt",
            ".article_view",
            "#CmAdContent"
        ]
        
        content = ""
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    # 불필요한 태그 제거
                    for tag in element.find_all(['script', 'style', 'iframe', 'ins', 'div.ad', '.ad', 'table']):
                        tag.decompose()
                    
                    # 광고 관련 텍스트 제거
                    for tag in element.find_all(text=True):
                        if any(ad_text in str(tag) for ad_text in ['광고', '©', 'ⓒ', '저작권', '무단전재', '재배포금지']):
                            tag.extract()
                    
                    content = element.get_text(strip=True)
                    content = ' '.join(content.split())  # 공백 정리
                    
                    if content and len(content) > 50:
                        logger.debug(f"본문 수집 성공: {selector}")
                        return content
            except Exception as e:
                continue
        
        # 모든 셀렉터가 실패한 경우, 페이지 전체에서 본문 추출 시도
        try:
            # 제목 다음의 텍스트 블록 찾기
            article_paragraphs = soup.find_all('p')
            if article_paragraphs:
                full_text = ""
                for p in article_paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 20:  # 의미있는 길이의 텍스트만
                        full_text += text + " "
                
                if len(full_text) > 100:
                    return full_text.strip()
        except:
            pass
        
        return "본문 수집 실패"
        
    except requests.exceptions.RequestException as e:
        logger.debug(f"네트워크 오류: {str(e)}")
        return "본문 수집 실패 - 네트워크 오류"
    except Exception as e:
        logger.debug(f"본문 수집 오류: {str(e)}")
        return "본문 수집 실패"

# Selenium을 사용한 백업 본문 수집
def collect_body_selenium_backup(driver, link):
    """Selenium을 사용한 백업 본문 수집"""
    try:
        original_url = driver.current_url
        driver.get(link)
        
        # 페이지 로드 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # 네이버 뉴스 구조별 본문 셀렉터
        selectors = [
            "#dic_area",
            ".go_trans._article_content", 
            "#articleBodyContents",
            ".news_article .article_body",
            "#newsEndContents",
            ".articleCont", 
            "#news_read",
            ".article_body",
            "#content",
            ".news_article",
            ".article-body"
        ]
        
        for selector in selectors:
            try:
                element = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                content = element.text.strip()
                if content and len(content) > 50:
                    # 원래 페이지로 돌아가기
                    driver.get(original_url)
                    return content
            except TimeoutException:
                continue
        
        # 모든 셀렉터 실패시 페이지 전체 텍스트 수집
        try:
            body_element = driver.find_element(By.TAG_NAME, "body")
            full_text = body_element.text
            if len(full_text) > 200:
                # 원래 페이지로 돌아가기
                driver.get(original_url)
                return full_text[:2000]  # 처음 2000자만
        except:
            pass
        
        # 원래 페이지로 돌아가기
        driver.get(original_url)
        return "본문 수집 실패"
        
    except Exception as e:
        logger.debug(f"Selenium 백업 본문 수집 오류: {str(e)}")
        try:
            driver.get(original_url)
        except:
            pass
        return "본문 수집 실패"
def get_all_news_links(driver, category):
    """페이지의 모든 뉴스 링크를 수집"""
    news_links = []
    
    try:
        # 페이지 로드 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "contentarea_left"))
        )
    except TimeoutException:
        logger.warning("페이지 로드 타임아웃")
        return []
    
    # 실시간속보의 경우 테이블 뉴스도 수집
    if category == "실시간 속보":
        table_selectors = [
            "#contentarea_left table.type_1 tr td.title a",
            "#contentarea_left table tr td.title a",
        ]
        
        for selector in table_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.info(f"  [테이블] {len(elements)}개 뉴스 발견")
                    for elem in elements:
                        try:
                            title = elem.text.strip()
                            link = elem.get_attribute("href")
                            if title and link and "http" in link:
                                news_links.append({
                                    "title": title,
                                    "link": link,
                                    "source": "table"
                                })
                        except:
                            continue
                    break
            except:
                continue
    
    # 리스트 뉴스 수집
    list_selectors = [
        "#contentarea_left ul li dl dd a",
        "#contentarea_left ul li dl dt a", 
        "#contentarea_left ul li a",
        "#contentarea li a[href*='news.naver.com']",
    ]
    
    for selector in list_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                logger.info(f"  [리스트] {len(elements)}개 뉴스 발견")
                
                for elem in elements:
                    try:
                        title = elem.text.strip()
                        link = elem.get_attribute("href")
                        
                        if (title and link and 
                            "http" in link and 
                            ("news.naver.com" in link or "n.news.naver.com" in link) and
                            len(title) > 5):
                            
                            # 중복 체크
                            if not any(existing["link"] == link for existing in news_links):
                                news_links.append({
                                    "title": title,
                                    "link": link,
                                    "source": "list"
                                })
                    except:
                        continue
                break
        except:
            continue
    
    return news_links

# 개선된 다음 페이지 확인 함수
def has_next_page(driver, current_page):
    """다음 페이지 존재 여부 확인"""
    try:
        # 방법 1: 다음 페이지 번호 링크 확인
        next_page_num = current_page + 1
        next_page_link = driver.find_elements(By.LINK_TEXT, str(next_page_num))
        if next_page_link and next_page_link[0].is_displayed():
            return True
        
        # 방법 2: '다음' 링크 확인
        next_links = driver.find_elements(By.PARTIAL_LINK_TEXT, "다음")
        for link in next_links:
            if link.is_displayed() and link.is_enabled():
                href = link.get_attribute("href")
                if href and "javascript:" not in href:  # javascript:void(0) 제외
                    return True
        
        # 방법 3: 페이지네이션 영역에서 확인
        pagination_links = driver.find_elements(By.CSS_SELECTOR, ".pgRR a, .paging a")
        for link in pagination_links:
            if link.is_displayed() and "다음" in link.text:
                return True
        
        return False
        
    except Exception as e:
        logger.debug(f"다음 페이지 확인 중 오류: {str(e)}")
        return False

# 개선된 단일 페이지 크롤링
def crawl_single_page_fast(driver, category, url):
    """빠른 단일 페이지 크롤링"""
    try:
        logger.info(f"  페이지 로딩: {url}")
        driver.get(url)
        time.sleep(1)  # 로딩 대기 시간 단축
        
        # 뉴스 링크 수집
        news_links = get_all_news_links(driver, category)
        
        if not news_links:
            logger.warning(f"  뉴스 링크를 찾을 수 없음")
            return []
        
        logger.info(f"  총 {len(news_links)}개 뉴스 링크 발견")
        
        # 뉴스 데이터 수집 (멀티스레딩으로 본문 수집 가능)
        news_items = []
        for idx, news_info in enumerate(news_links, 1):
            try:
                title = news_info["title"]
                link = news_info["link"]
                source = news_info["source"]
                
                # 본문 수집 (requests 우선, 실패시 Selenium 백업)
                body = collect_body_fast(link)
                
                # requests가 실패하면 Selenium으로 시도
                if body == "본문 수집 실패" or body == "본문 수집 실패 - 네트워크 오류":
                    logger.debug(f"   requests 실패, Selenium으로 재시도: {title[:30]}...")
                    body = collect_body_selenium_backup(driver, link)
                
                news_items.append({
                    "category": category,
                    "title": title,
                    "link": link,
                    "body": body,
                    "source": source
                })
                
                # 진행률 표시
                if idx % 10 == 0:
                    logger.info(f"   진행률: {idx}/{len(news_links)} ({idx/len(news_links)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"   뉴스 수집 실패 [{idx}]: {str(e)}")
                continue
        
        return news_items
        
    except Exception as e:
        logger.error(f"페이지 크롤링 실패: {url} - {str(e)}")
        return []

# 개선된 카테고리별 전체 페이지 크롤링
def crawl_category_all_pages_smart(driver, category, url_template, date_str, max_pages=20):
    """스마트한 카테고리별 전체 페이지 크롤링"""
    all_news = []
    consecutive_empty_pages = 0  # 연속 빈 페이지 카운터
    
    for page in range(1, max_pages + 1):
        try:
            # URL 생성
            if page == 1:
                url = url_template.format(date=date_str)
            else:
                url = url_template.format(date=date_str) + f"&page={page}"
            
            logger.info(f"[{category}] 페이지 {page} 크롤링 시작")
            
            # 페이지 크롤링
            page_news = crawl_single_page_fast(driver, category, url)
            
            if not page_news:
                consecutive_empty_pages += 1
                logger.info(f"[{category}] 페이지 {page}에서 뉴스가 없음 (연속 빈 페이지: {consecutive_empty_pages})")
                
                # 연속으로 2페이지가 비어있으면 종료
                if consecutive_empty_pages >= 2:
                    logger.info(f"[{category}] 연속 빈 페이지로 인해 크롤링 종료")
                    break
                    
                # 다음 페이지 존재 확인
                if not has_next_page(driver, page):
                    logger.info(f"[{category}] 다음 페이지 없음 - 크롤링 종료")
                    break
                    
                continue
            else:
                consecutive_empty_pages = 0  # 뉴스가 있으면 카운터 리셋
            
            all_news.extend(page_news)
            logger.info(f"[{category}] 페이지 {page}에서 {len(page_news)}개 뉴스 수집 완료 (총 {len(all_news)}개)")
            
            # 다음 페이지 존재 확인
            if not has_next_page(driver, page):
                logger.info(f"[{category}] 다음 페이지 없음 - 크롤링 종료")
                break
            
            time.sleep(0.5)  # 페이지간 딜레이 단축
            
        except Exception as e:
            logger.error(f"[{category}] 페이지 {page} 크롤링 실패: {str(e)}")
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= 3:  # 연속 실패시 종료
                break
            continue
    
    return all_news

# 메인 실행 함수
def main():
    driver = None
    
    try:
        # requests 세션 설정 (본문 수집용)
        
        driver = create_driver()
        
        for current_date in date_list:
            date_str = current_date.strftime("%Y%m%d")
            date_hyphen = current_date.strftime("%Y-%m-%d")
            
            logger.info(f"=== {date_hyphen} 크롤링 시작 ===")
            daily_news = []
            
            for category, url_template in categories.items():
                logger.info(f"[{category}] 크롤링 시작")
                
                try:
                    # max_pages를 20으로 증가 (10페이지 이상도 처리)
                    category_news = crawl_category_all_pages_smart(
                        driver, category, url_template, date_str, max_pages=20
                    )
                    
                    # 날짜 정보 추가
                    for news in category_news:
                        news['date'] = date_hyphen
                        news['crawl_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    daily_news.extend(category_news)
                    logger.info(f"[{category}] 총 {len(category_news)}개 뉴스 수집 완료")
                    
                except Exception as e:
                    logger.error(f"[{category}] 크롤링 실패: {str(e)}")
                
                time.sleep(1)  # 카테고리간 딜레이 단축
            
            # 데이터 저장
            if daily_news:
                df = pd.DataFrame(daily_news)
                
                # 중복 제거
                df = df.drop_duplicates(subset=['title', 'link'], keep='first')
                df = df[df['title'].str.len() > 5]
                
                # 컬럼 순서 정리
                column_order = ['date', 'category', 'title', 'link', 'body', 'source', 'crawl_time']
                df = df[column_order]
                
                # CSV 저장
                filename = os.path.join(data_dir, f"naver_news_{date_str}.csv")
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                
                logger.info(f"▶ {date_hyphen} 총 {len(df)}개 뉴스 저장 완료 ({filename})")
                
                # 카테고리별 통계
                category_stats = df['category'].value_counts()
                for cat, count in category_stats.items():
                    logger.info(f"   - {cat}: {count}개")
                    
            else:
                logger.warning(f"▶ {date_hyphen} 수집된 뉴스가 없음")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"메인 실행 오류: {str(e)}")
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("드라이버 종료 완료")
            except:
                logger.warning("드라이버 종료 실패")

if __name__ == "__main__":
    main()