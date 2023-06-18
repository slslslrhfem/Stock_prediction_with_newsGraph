from pykrx import stock
import pandas as pd
import matplotlib.pyplot as plt
from urllib.error import HTTPError
import urllib.request
import pandas as pd
from ast import literal_eval
from tqdm import tqdm  
import numpy as np
import os, datetime
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from utils import check_outlier, Label_changer, get_dates
import FinanceDataReader as fdr
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from transformers import (PreTrainedTokenizerFast, 
                          set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification, BertModel)
import re
from kobert_tokenizer import KoBERTTokenizer
import dgl
import logging
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def graph_construct(date):

    folder_name = 'dataset{}'.format(date)

    all_sectors = ['software','mechanic','automobile','electronic', 'medical', 'economy', 'chemistry', 'entertainment', 'science', 'food', 'clothes', 'transport', 'construction', 'wholesale_retail', 'other_service', 'other_manufacture']
    stocks = fdr.StockListing('KRX')
    ticker_list = list(stocks["Symbol"])
    name_list = list(stocks["Name"])
    Sector_Label = list(stocks["Sector"])
    Sector_Label = Label_changer(Sector_Label) # 많은 섹터들을 묶는 과정
    

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1',padding=True, truncation=True, max_length=128)
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    model.to(device)

    with open('{}/trade_data.pickle'.format(folder_name),'rb') as f:
        trade_data = pickle.load(f) 

    count=0
    not_available_idx=[]
    for i,ticker in enumerate(ticker_list):
        if ticker not in trade_data.keys():
            not_available_idx.append(i)
            count+=1
            continue
        if trade_data[ticker][-1][0]==0 or trade_data[ticker][0][0]==0: # 이 경우는 일부 데이터가 누락된 경우. 보통 상장폐지된 종목을 어찌저찌 조회하게되면 이렇게 된다.
            not_available_idx.append(i)
            count+=1

        
    print(count,'company can not be used by issue with KRX crawling module..')

    avail_ticker_list=[]
    avail_name_list=[]
    avail_Sector_Label=[]

    for i, (ticker, name, Sector) in enumerate(zip(ticker_list, name_list,Sector_Label)):
        if i not in not_available_idx:
            avail_ticker_list.append(ticker)
            avail_name_list.append(name)
            avail_Sector_Label.append(Sector)

    print(len(avail_Sector_Label),len(avail_name_list),len(avail_ticker_list))

    u=[]#from node
    v=[]#to node
    e_feature = [] # edge_feature
    maxs=[]
    mins=[]
    company_idx=0
    sectors=[]
    volumes=[]
    end_prices=[]
    profits=[]
    dateinfos=[]
    titles = []
    articles = []
    tickers=[]
    up_ratios=[]
    print("training용 그래프를 구성합니다")

    data_days = len(trade_data[ticker_list[0]]) - 1 #데이터가 30일치 있으면, 29일의 Data는 뉴스와 주가가 있고, 1일은 뉴스 데이터만 있다. -> Node 갯수가 회사 당 29임
    pbar = tqdm(zip(avail_ticker_list, avail_name_list, avail_Sector_Label))
    for ticker, name, Sector in pbar: #각 종목마다
        pbar.set_description("processing %s" % name)
        max_volume = 0
        min_volume = 25000000000000
        max_volume_index = None
        min_volume_index = None
        sector_node = all_sectors.index(Sector)
        company_end_prices=[]


        for i in range(len(trade_data[ticker])-1): # 각 날짜마다
            last_day = trade_data[ticker][i][0] # 주가 기록이 있는 마지막 날짜. 주말이나 공휴일 등이 끼어있을 경우 today와 1일 이상 차이날 수 있다.
            today = trade_data[ticker][i+1][0]
            dates=get_dates(last_day,today)


            
            title, article = get_news(folder_name, Sector, name, dates)
            titles.append(title) # 이려면 3000개 회사의 top1~top10(없으면 그냥 '') article이 들어감. titles[:,0] -> 모든 회사의 top 1 기사
            articles.append(article)
            news_edges = get_news_edges(title,article,avail_name_list)
            
            end_price = trade_data[ticker][i+1][4]
            open_price = trade_data[ticker][i+1][1]
            profit = (end_price/open_price - 1)*100 # 단위를 %로 하려고
            volume = np.log(trade_data[ticker][i+1][5])
            date_info = i # 날짜 정보를 따로 encoding. 순서가 있는 데이터이기때문에

            if volume>max_volume:
                max_volume_index = i + data_days * company_idx
                max_volume = volume
            if volume<min_volume:
                min_volume_index = i + data_days * company_idx
                min_volume = volume

            sectors.append(sector_node)
            volumes.append(volume)
            company_end_prices.append(end_price)
            profits.append(profit)
            tickers.append(int(ticker))
            dateinfos.append(date_info)
            up_ratios.append(trade_data[ticker][i+1][7])
            #print('현재 회사는', name,'이고, 노드 정보는', sector_node, volume, end_price, date_info, title, article, '이 들어가는 중입니다.')
            for news in news_edges:
                if company_idx != avail_name_list.index(news): # news 내용과 관련된 엣지.
                    u.append(int(i+data_days * company_idx))
                    v.append(int(i + data_days * avail_name_list.index(news))) 
                    e_feature.append(0)
                    v.append(int(i+data_days * company_idx))
                    u.append(int(i + data_days * avail_name_list.index(news)))  
                    e_feature.append(1)# 단방향 edge 2개를 사용. A회사 뉴스에서 에서 B 회사가 나온거랑 B회사 뉴스에서 A회사가 나온건 동치가 아님
            if i!=0:
                u.append(int(i+data_days * company_idx-1))#전날에서
                v.append(int(i+data_days * company_idx))#오늘로 연결되는 엣지.
                e_feature.append(2)
                v.append(int(i+data_days * company_idx-1))#전날로
                u.append(int(i+data_days * company_idx))#오늘에서 연결되는 엣지. 역시나 동치는 아니므로 엣지 종류는 다르게 설정.
                e_feature.append(3)

        for i in range(len(trade_data[ticker])-1): # 회사별로 가장 거래량이 낮은 날짜와 높은 날짜 Node에 전부 연결.
            if max_volume_index != i+data_days*company_idx:
                u.append(max_volume_index)
                v.append(i+data_days*company_idx)
                e_feature.append(4)
                v.append(max_volume_index)
                u.append(i+data_days*company_idx)
                e_feature.append(5)
            if min_volume_index != i+data_days*company_idx:
                u.append(min_volume_index)
                v.append(i+data_days*company_idx)
                e_feature.append(6)
                v.append(min_volume_index)
                u.append(i+data_days*company_idx) #양방향으로 다 넣어주되, 역이 동치는 아마..도 아니므로( 두 Node 중 하나만 최고/최저 거래니까? ) Edge 종류는 다르게 줌. 
                e_feature.append(7)
            # 원래는 Max 종가나 Max 등락도 있었는데, 이를 연결하면 예측해야하는 요소와 너무 종속이라 일단은 뺌

        max_price = max(company_end_prices)
        min_price = min(company_end_prices)
        for i in range(len(trade_data[ticker])-1):
            maxs.append(max_price)
            mins.append(min_price)
        company_end_prices = np.array(company_end_prices)
        company_end_prices = (company_end_prices-min_price) / (max_price - min_price) # min-max scaling to 0~1. 0은 masked value라 의미가 겹치긴 한데, 학습시 별도의 Masking 토큰을 사용한다.
        end_prices.extend(company_end_prices)
        #print('현재 Graph 구성은', u,v,'입니다. max가격과 min가격은', max_price, min_price,'node feature들은 다음과 같습니다')
        #print(sectors, volumes, end_prices, maxs, mins, dateinfos)
        gc.collect()
        company_idx+=1
    
    
    g = dgl.graph((u,v))
    g.edata['edge_feature'] = torch.tensor(e_feature)
    g.ndata['ticker'] = torch.tensor(tickers)
    g.ndata['sector'] = torch.tensor(sectors)
    g.ndata['volume'] = torch.tensor(volumes)
    g.ndata['end_price'] = torch.tensor(end_prices)
    g.ndata['max_value'] = torch.tensor(maxs)
    g.ndata['min_value'] = torch.tensor(mins)
    g.ndata['profit'] = torch.tensor(profits)
    g.ndata['up_ratio']= torch.tensor(up_ratios)
    titles = np.array(titles)
    articles = np.array(articles)
    article_embeddings , title_embeddings = get_news_embedding(titles, articles,model,tokenizer)
    g.ndata['article_embedding'] = torch.tensor(article_embeddings)
    g.ndata['title_embedding'] = torch.tensor(title_embeddings)
    g.ndata['date'] = torch.tensor(dateinfos)

    dgl.save_graphs('training{}.bin'.format(date), g)


    del g, e_feature, sectors, volumes, end_prices, article_embeddings, title_embeddings, dateinfos
    gc.collect()


    print("inference용 그래프를 구성합니다")

    inf_u=[]#from node
    inf_v=[]#to node
    inf_e_feature = [] # edge_feature

    company_idx=0
    inf_sectors=[]
    inf_volumes=[]
    inf_end_prices=[]
    inf_dateinfos=[]
    inf_titles=[]
    inf_articles=[]
    inf_maxs=[]
    inf_mins=[]
    inf_tickers=[]
    inf_profits=[]
    inf_up_ratios = []


    #이거는 Inference용. training용과 일단은 node갯수가 같다. 그래프 모양 자체는 좀 다르긴 함
    pbar = tqdm(zip(avail_ticker_list, avail_name_list, avail_Sector_Label))
    for ticker, name, Sector in pbar: #각 종목마다
        pbar.set_description("processing %s" % name)
        max_volume = 0
        min_volume = 25000000000000
        max_volume_index = None
        min_volume_index = None
        sector_node = all_sectors.index(Sector)
        inf_company_end_prices=[]

        for i in range(len(trade_data[ticker])-1): # 각 날짜마다
            if i!=len(trade_data[ticker])-2: # 마지막날은 주가정보가 없어서(코드를 일요일에 돌린다 치면, 예측할 월요일의 데이터에 해당)
                last_day = trade_data[ticker][i+1][0] # 주가 기록이 있는 마지막 날짜. 주말이나 공휴일 등이 끼어있을 경우 today와 1일 이상 차이날 수 있다.
                today = trade_data[ticker][i+2][0]
                dates=get_dates(last_day,today)

                
                
                title, article = get_news(folder_name, Sector, name, dates)
                news_edges = get_news_edges(title,article,avail_name_list)
                inf_titles.append(title) # 이려면 3000개 회사의 top1~top10(없으면 그냥 '') article이 들어감. titles[:,0] -> 모든 회사의 top 1 기사
                inf_articles.append(article)
                
                volume = np.log(trade_data[ticker][i+2][5])
                end_price = trade_data[ticker][i+2][4]
                open_price = trade_data[ticker][i+2][1]
                upratio = trade_data[ticker][i+2][7]
                profit = (end_price/open_price - 1)*100 # 단위를 %로 하려고 
                date_info = i # 날짜 정보를 따로 encoding. 순서가 있는 데이터이기때문에. 데이터의 첫 날을 0으로 본다. 즉 Inference용과 Training용의 같은날짜가 dateinfo가 1 차이나는게 맞다.
                if volume>max_volume:
                    max_volume_index = i + data_days * company_idx
                    max_volume = volume
                if volume<min_volume:
                    min_volume_index = i + data_days * company_idx
                    min_volume = volume

                inf_sectors.append(sector_node)
                inf_volumes.append(volume)
                inf_company_end_prices.append(end_price)
                inf_dateinfos.append(date_info)
                inf_tickers.append(int(ticker))
                inf_profits.append(profit)
                inf_up_ratios.append(upratio)
                for news in news_edges:
                    if company_idx != avail_name_list.index(news): # news 내용과 관련된 엣지.
                        inf_u.append(int(i+data_days * company_idx))
                        inf_v.append(int(i + data_days * avail_name_list.index(news)))
                        inf_e_feature.append(0)
                        inf_v.append(int(i+data_days * company_idx))
                        inf_u.append(int(i + data_days * avail_name_list.index(news)))  
                        inf_e_feature.append(1)# 단방향 edge 2개를 사용. A회사 뉴스에서 에서 B 회사가 나온거랑 B회사 뉴스에서 A회사가 나온건 동치가 아님
                if i!=0:
                    inf_u.append(int(i+data_days * company_idx-1))#전날에서
                    inf_v.append(int(i+data_days * company_idx))# 오늘로
                    inf_e_feature.append(2)
                    inf_v.append(int(i+data_days * company_idx-1))#오늘에서
                    inf_u.append(int(i+data_days * company_idx))#전날로. 다만 앞날의 데이터를 전날로 넘겨주는것이 상식적으로는 애매하긴 하다. 굳이 이유를 찾자면 예시로 16일의 데이터를 예측하기 위해 13일의 데이터를 보는데, 13일의 데이터가 14일의 데이터에 영향을 받는? 느낌
                    inf_e_feature.append(3)
                print(today,date_info, title)
            else: #마지막날
                last_day = trade_data[ticker][i+1][0]
                lastday = datetime.strptime(str(last_day),'%Y-%m-%d %H:%M:%S')
                to = lastday + timedelta(4) # 다음 개장일 모르니까 4일 이후로. 날짜가 넘어가도 괜찮다. 어차피 뉴스 파일이 있는지 없는지 확인한 후에 있으면 가져오는 형태
                today = to.strftime('%Y-%m-%d %H:%M:%S')
                dates = get_dates(last_day,today)

                
                title, article = get_news(folder_name, Sector, name, dates)
                news_edges = get_news_edges(title,article,avail_name_list)


                inf_sectors.append(sector_node)
                inf_volumes.append(0.0) # Mask는 하되 예측할 필요는 없게 구성할 듯
                inf_end_prices.append(0.0) # Mask하게될 지표. 모델 1
                inf_dateinfos.append(date_info+2)
                inf_titles.append(title)
                inf_articles.append(article)
                inf_tickers.append(int(ticker))
                inf_up_ratios.append(0) # Mask하게될 지표. 모델 2
                inf_profits.append(0) # Mask하게될 지표. 모델 3
                #print('현재 회사는', name,'이고, 노드 정보는', sector_node, volume, end_price, date_info, title, article, '이 들어가는 중입니다.')
        
        for i in range(len(trade_data[ticker])-1): # 회사별로 가장 거래량이 낮은 날짜와 높은 날짜 Node에 전부 연결. 종가나 등락을 연결하면 예측해야하는 요소와 너무 종속이라 일단은 뺌
            if max_volume_index != i+data_days*company_idx:#selfloop은 필요 없으니..
                inf_u.append(max_volume_index)
                inf_v.append(i+data_days*company_idx)
                inf_e_feature.append(4)
                inf_v.append(max_volume_index)
                inf_u.append(i+data_days*company_idx)
                inf_e_feature.append(5)
            if min_volume_index != i+data_days*company_idx:
                inf_u.append(min_volume_index)
                inf_v.append(i+data_days*company_idx)
                inf_e_feature.append(6)
                inf_v.append(min_volume_index)
                inf_u.append(i+data_days*company_idx) #양방향으로 다 넣어주되, 역이 동치는 아마..도 아니므로( 두 Node 중 하나만 최고/최저 거래니까? ) Edge 종류는 다르게 줌. 
                inf_e_feature.append(7)
        
        inf_max_price = max(inf_company_end_prices)
        inf_min_price = min(inf_company_end_prices)
        for i in range(len(trade_data[ticker])-1):
            inf_maxs.append(inf_max_price)
            inf_mins.append(inf_min_price)
        inf_company_end_prices = np.array(inf_company_end_prices)
        inf_company_end_prices = (inf_company_end_prices-inf_min_price) / (inf_max_price - inf_min_price) # min-max scaling to 0~1. 0은 masked value라 의미가 겹치긴 한데, 학습시 별도의 Masking 토큰을 사용한다.
        inf_end_prices.extend(inf_company_end_prices) 
        company_idx+=1
        #print('현재 Graph 구성은', inf_u,inf_v,'입니다. max가격과 min가격은', inf_max_price, inf_min_price,'node feature들은 다음과 같습니다')
        #print(inf_sectors, inf_volumes, inf_end_prices, inf_maxs, inf_mins, inf_dateinfos)
        
    g2 = dgl.graph((inf_u,inf_v))
    g2.edata['edge_feature'] = torch.tensor(inf_e_feature)
    g2.ndata['sector'] = torch.tensor(inf_sectors)
    g2.ndata['volume'] = torch.tensor(inf_volumes)
    g2.ndata['end_price'] = torch.tensor(inf_end_prices)
    g2.ndata['max_value'] = torch.tensor(inf_maxs)
    g2.ndata['min_value'] = torch.tensor(inf_mins)
    g2.ndata['profit'] = torch.tensor(inf_profits)
    g2.ndata['up_ratio'] = torch.tensor(inf_up_ratios)

    inf_titles = np.array(inf_titles)
    inf_articles = np.array(inf_articles)
    g2.ndata['ticker'] = torch.tensor(inf_tickers)
    inf_article_embeddings , inf_title_embeddings = get_news_embedding(inf_titles, inf_articles,model, tokenizer)
    g2.ndata['article_embedding'] = inf_article_embeddings.cpu()
    g2.ndata['title_embedding'] = inf_title_embeddings.cpu()
    g2.ndata['date'] = torch.tensor(inf_dateinfos)
    

    dgl.save_graphs('inference{}.bin'.format(date), g2)
    return g,g2

def get_news(folder_name, sector,name,dates):
    title = []
    article = []
    dates.reverse() # 가까운 날짜일수록 우선순위가 높도록
    for date in dates:
        day = date.strftime("%Y.%m.%d")
        for i in range(10):

            filename = '{}/{}/{}/{}_{}_{}_top{}.pickle'.format(folder_name, sector, name, name, day, date.strftime("%A"),i+1)
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    news = pickle.load(f)
                title.append(str(news['title']))
                article.append(str(news['article']))

    while (len(title)<10): # 뉴스가 10개 이하인 경우 채워줌
        title.append('')
        article.append('')

    ret_title=''
    ret_article=''

    for i in range(10):
        ret_title = ret_title + 'top' + str(i+1) +' ' + title[i] + ' ' # top1 무언가... top2 무언가... top3 무언가... 의 형태로 하나의 string으로 합침. 중간 top1, top2... 이 친구들이 뉴스를 끊는 token 역할을 해줄 것으로 기대
        ret_article = ret_article + 'top' + str(i+1) + ' ' + article[i]+ ' '

    
    return ret_title, ret_article

class NewsDataset(torch.utils.data.Dataset): # 기사 내용과 제목에서 BERT Embedding가져오기 위한 데이터셋

    def __init__(self, title_inputs, article_inputs):

        self.article_input_ids = torch.tensor(article_inputs['input_ids']).to(device)
        self.article_attention = torch.tensor(article_inputs['attention_mask']).to(device)
        self.title_input_ids = torch.tensor(title_inputs['input_ids']).to(device)
        self.title_attention = torch.tensor(title_inputs['attention_mask']).to(device)
        

    def __getitem__(self, i):

        return self.article_input_ids[i], self.article_attention[i], self.title_input_ids[i], self.title_attention[i]

    def __len__(self):
        return len(self.article_input_ids)
    
def get_news_embedding(title, article,model, tokenizer): # title은 (회사수 , 10)차원으로 string을 담고있는 형태.
    
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        top_title = title
        top_article = article
        top_article_emb=[]
        top_title_emb=[]

        title_inputs = tokenizer.batch_encode_plus(top_title,max_length = 128, padding='max_length', truncation = True)
        article_inputs = tokenizer.batch_encode_plus(top_article, max_length = 128, padding='max_length', truncation = True)
        print(torch.tensor(article_inputs['input_ids']).shape, torch.tensor(article_inputs['attention_mask']).shape)
        Dataset = NewsDataset(title_inputs, article_inputs)
        news_dataloader = DataLoader(Dataset,batch_size=128)
        for article_input_id, article_attention, title_input_id, title_attention in tqdm(news_dataloader):
            article_embedding = model(input_ids = article_input_id,  attention_mask = article_attention) #(회사 수*날짜 수,768)
            title_embedding = model(input_ids = title_input_id,  attention_mask = title_attention)
            if len(top_article_emb)==0:
                top_article_emb = article_embedding.pooler_output.cpu()
                top_title_emb = title_embedding.pooler_output.cpu()
            else:
                top_article_emb = torch.cat((top_article_emb,article_embedding.pooler_output.cpu()), dim=0) #(다 돌면 회사 수 * 768이 되도록)
                top_title_emb = torch.cat((top_title_emb, title_embedding.pooler_output.cpu()), dim=0)
    return top_article_emb, top_title_emb


def get_news_edges(title,article,name_list):#news 10개(혹은 이하)
    companys = []
    tit_res = re.findall('|'.join(name_list), title)
    art_res = re.findall('|'.join(name_list), article)
    for titres in tit_res:
        if titres not in companys:
            if titres in name_list: # 분명 re.findall로 name_list에 있는걸 찾아오긴 하는데.. 공백 관련 이슈가 조금 있는 듯 하다. ('JYP Ent ' is not in list 에러가 뜸..) 따라서 다시 확인 후 Name_list에 있는 경우만 넣음.
                companys.append(titres)
    for artres in art_res:
        if artres not in companys:
            if artres in name_list:
                companys.append(artres)
    return companys

