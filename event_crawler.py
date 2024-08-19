# backlog
# datahub 연결 
# 효율적으로 db 업데이트(Locality Sensitive Hashing (LSH)?)
# 크롤링 자동화 (aws lambda 사용, https://alsxor5.tistory.com/118)
# 네이버 api로 이벤트 lat lon 크롤링
# event place와 geo data 내 건물명의 mismatch 문제
# string type nan 고치기



import re
import pandas as pd
from datetime import date
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def get_event_info_from_naver(keyword, page_limit):
    
    def extract_url_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        if keyword == '축제' :
            a_tags = soup.find_all('a', class_='img_box')
        if keyword == '콘서트' :
            a_tags = soup.find_all('a', class_='inner')
        hrefs = ['https://search.naver.com/search.naver' + a_tag['href'] for a_tag in a_tags if 'href' in a_tag.attrs]
        return hrefs
    
    def extract_name_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        card_areas = soup.find_all('strong', class_='_text')
        texts = [card.get_text(separator=' ', strip=True) for card in card_areas]
        return texts

    def extract_info_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        card_areas = soup.find_all('div', class_='info_group')
        texts = [card.get_text(separator=' ', strip=True) for card in card_areas]
        return texts
    
    def extract_map_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        a_tags = soup.find_all('a', class_='place')
        hrefs = [a_tag['href'] for a_tag in a_tags if 'href' in a_tag.attrs]
        return hrefs[0]
    
    def extract_juso_from_page(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        juso = soup.find_all('span', class_='LDgIH')
        return juso
    
    def add_info(texts):
        info_keys = ['start', 'end', 'period', 'place', 'category']
        my_dict = {key: None for key in info_keys}
        
        for tt in texts :
            if tt.startswith('기간'):
                date_match = re.findall(r'\d{4}\.\d{2}\.\d{2}', tt)

                if len(date_match) == 1:    #하루만 진행하는 공연이라면 시작일=종료일
                    date_str = date_match[0]
                    my_dict['start'] = date_str.replace('.', '')
                    my_dict['end'] = date_str.replace('.', '')
                    my_dict['period'] = date_str.replace('.', '')
                elif len(date_match) == 2:
                    start_date_str, end_date_str = date_match
                    my_dict['start'] = start_date_str.replace('.', '')
                    my_dict['end'] = end_date_str.replace('.', '')
                    my_dict['period'] = start_date_str.replace('.', '' ) + ' ~ ' + end_date_str.replace('.', '')             
                else: 
                    raise Exception("Error: Could not find exactly two dates in the text.") 

            if tt.startswith('장소'):
                tt = tt.replace('장소 ', '')
                tt = tt.replace(' 좌석배치도', '')
                tt = tt.replace('"', '')
                my_dict['place'] = tt

            if tt.startswith('개요'):
                tt = tt.replace('개요 ', '')
                tt = re.sub(r'\d+분', '', tt).strip()
                my_dict['category'] = tt

        event_period_start.append(my_dict['start'])
        event_period_end.append(my_dict['end'])
        event_period.append(my_dict['period']) 
        event_place.append(my_dict['place'])
        event_category.append(my_dict['category'])
        
    

    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=" + keyword

    event_names = []
    event_period = []
    event_period_start = []
    event_period_end = []
    event_place = []
    event_category = []
    event_url_links = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        page.goto(url)
        page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초
        
        # 공연 하이퍼링크 가져오기
        for page_index in range(page_limit): # 네이버 공연 정보 70개 크롤링
            html = page.content()
            texts = extract_url_from_page(html)
            
            # 다음 버튼 클릭
            next_button = page.query_selector('a.pg_next.on[data-kgs-page-action-next]')
            
            if next_button:
                next_button.click()
                page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초
            else:
                print(f"Next button not found on page {page_index}")
                break
            
        event_urls = texts
        
        # 공연정보 가져오기
        for event_url in event_urls : 
            page.goto(event_url)
            page.wait_for_timeout(100)  # 페이지 로딩 대기시간 조절 : 0.1초
            html = page.content()

            names = extract_name_from_page(html)
            texts = extract_info_from_page(html)
            #map_url = extract_map_from_page(html)

            event_names.append(names[0])
            event_url_links.append(event_url)
            
            add_info(texts)            


        # 네이버 지도 크롤링은 naver api로 구현 필요
        #page.goto(map_url)
        #page.wait_for_timeout(30000)  # 페이지 로딩 대기시간 조절 : 30초 (네이버지도는 오래줘야함)
        #html = page.content()
        #juso = extract_juso_from_page(html)
            
        browser.close()

    print(len(event_period), len(event_names), len(event_place), len(event_url_links))

    event_df = pd.DataFrame({
            "k_date": event_period,
            "title": event_names,
            "place": event_place,
            "lat": None,
            "lon": None,
            "url_link": event_url_links,
            "start_date": event_period_start,
            "end_date": event_period_end,
            "area": None,
            "category": event_category,
            "dt": str(date.today().strftime('%Y%m%d'))
            })
    
    event_df = event_df.astype(str)
    #print(event_df.head())
    
    return event_df


def merge_data():

    # 기존 데이터와 새 데이터 로드
    origin_data = pd.read_csv('event_crawling.csv')
    festival_data = get_event_info_from_naver('축제', 19)   #80개
    show_data = get_event_info_from_naver('콘서트', 7)  #80개

    # 중복 제거
    combined_data = pd.concat([origin_data, festival_data, show_data]).drop_duplicates(subset=['k_date', 'title', 'place'], keep='first')

    # 데이터베이스에 업데이트
    combined_data.to_csv('event_crawling.csv', index=False)


merge_data()
