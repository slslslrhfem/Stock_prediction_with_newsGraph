import sys, os
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import pickle


def news_crawling(query, day,sector, today_datetime, to_file = True): # 이 부분 코드 작동하는지 확인 좀 해야하긴 할듯..
    
    #print('브라우저를 실행시킵니다(자동 제어)\n')

    news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={0}&sort=1&photo=0&field=0&pd=3&ds={1}&de={2}'.format(query, day, day)
    # HTTP GET 요청을 보내고 응답을 받아옴
    response = requests.get(news_url)
    news_dict ={}

    # BeautifulSoup을 사용하여 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 뉴스 기사의 제목과 본문 일부를 추출하는 코드
    articles = soup.select(".type01 > li")

    for article in articles: 
        title = article.select_one("a._sp_each_title").text.strip()
        summary = article.select_one("dd").text.strip()
        new_row = {
                        'title' : title,
                        'article' : summary}
        filename = 'dataset/{}/{}/{}_{}_{}_top{}.pickle'.format(sector, query, query, day, today_datetime.strftime("%A"),i+1)
        if to_file:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename,'wb') as fw:
                pickle.dump(new_row,fw)
        else:
            news_dict[filename] = new_row
        #news_df.loc[len(news_df)] = new_row
        #news_df = news_df.reindex(sorted(news_df.columns), axis=1)
        
        
        #news_df.to_excel(filename, index=False) #엑셀보다 dict가 여러모로 나은 것 같음..
    return news_dict
    
    
def start_crawling(data):
    news_dict={}
    query,sector, to_file = data
    for i in range(30):
        today_datetime = datetime.today() - timedelta(i+1)
        day = (today_datetime).strftime("%Y.%m.%d")
        news_dict.update(news_crawling(query, day,sector, today_datetime, to_file))
    return news_dict