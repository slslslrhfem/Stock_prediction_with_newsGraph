import sys, os
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta
import pickle


def news_crawling(query, day,sector, today_datetime):
    
    #print('브라우저를 실행시킵니다(자동 제어)\n')

    news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_pge&query={0}&sort=1&photo=0&field=0&pd=3&ds={1}&de={2}'.format(query, day, day)
    # HTTP GET 요청을 보내고 응답을 받아옴
    response = requests.get(news_url)

    # BeautifulSoup을 사용하여 HTML 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # 뉴스 기사의 제목과 본문 일부를 추출하는 코드
    news_titles = soup.select(".news_tit")
    news_contents = soup.select(".api_txt_lines")
    
    #news_df = DataFrame({'title':[], 'day' : [], 'article' : []})

    for i, x in enumerate(zip(news_titles, news_contents)):
        n,m = x
        new_row = {
                        'title' : n.text,
                        'article' : m.text}
        #news_df.loc[len(news_df)] = new_row
        #news_df = news_df.reindex(sorted(news_df.columns), axis=1)
        filename = 'dataset/{}/{}/{}_{}_{}_top{}.pickle'.format(sector, query, query, day, today_datetime.strftime("%A"),i+1)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename,'wb') as fw:
            pickle.dump(new_row,fw)
        #news_df.to_excel(filename, index=False)
        

        
    ################# 데이터 전처리 #################

    #print('데이터프레임 변환\n')
    

    #print('엑셀 저장 완료 | 경로 : {}\\{}\n'.format(folder_path, xlsx_file_name))
    
def start_crawling(data):
    query,sector = data
    for i in range(30):
        today_datetime = datetime.today() - timedelta(i+1)
        day = (today_datetime).strftime("%Y.%m.%d")
        news_crawling(query, day,sector, today_datetime) # 본문의 경우 크롤링이 클린하게 되지 않거나, 경우에 따라 아예 안되는 경우도 있습니다.
        # 이게 뉴스가 있는 사이트가 다 다르다보니 생기는 현상.. 다만 일부가 Mask된 Data여도 사용할 수 있는 모델이라 일단은 씁니다
    return 0