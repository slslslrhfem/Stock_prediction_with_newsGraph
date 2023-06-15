import numpy as np
import torch
from pykrx import stock
from ast import literal_eval
from tqdm import tqdm  
import os, datetime
from datetime import datetime, timedelta
from utils import check_outlier, Label_changer
import FinanceDataReader as fdr
from news_crawling import start_crawling
import pickle
import time
import torch

from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, random_split
from torch.nn import CrossEntropyLoss
from pytorch_lightning.core.lightning import LightningModule
#from transformers.modeling_bert import BertModel
from transformers import AdamW

def data_preprocessing():
    
    stocks = fdr.StockListing('KRX')
    ticker_list = list(stocks["Symbol"])
    name_list = list(stocks["Name"])
    Sector_Label = list(stocks["Sector"])
    end_date = datetime.now() - timedelta(1) # 예를 들어 6/13 새벽 1시에 코드를 돌리면. 6월 12일까지의 뉴스와 주가 정보를 들고옵니다. 그리고 13일의 data를 예측하도록 하면 되겠죠! 
    start_date = end_date - timedelta(30) # 일단은 데이터를 1달분량을 들고옵니다.
    end_date = end_date.strftime('%Y%m%d')#'20230531'과 같은식
    start_date = start_date.strftime('%Y%m%d')
    Sector_Label = Label_changer(Sector_Label) # 많은 섹터들을 묶는 과정
    tot_dict={}

    trade_data = stock.get_market_ohlcv(start_date, end_date, ticker_list[0])
    adding_list=trade_data.reset_index().values.tolist()
    minlen=len(adding_list)

    
    
    for ticker, name, Sector in tqdm(zip(ticker_list, name_list, Sector_Label)): # 뉴스 크롤링
        meta_dict = {}
        meta_dict['sector'] = Sector
        meta_dict['name'] = name
        meta_dict['ticker'] = ticker

        filename = 'dataset{}/{}/{}/meta.pickle'.format(end_date,Sector,name)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename,'wb') as fw:
            pickle.dump(meta_dict, fw)
        
        start_crawling(name, Sector)

    for tickers in tqdm(ticker_list,position=0): # 주가 크롤링
        try:
            time.sleep(0.5)
            trade_data = stock.get_market_ohlcv(start_date, end_date, str(tickers))
            # expecting value: line 1 column 1 (char 0) -> krx쪽에서 ip 차단당하면 이 에러가 뜹니다. 한달치 데이터를 모든 주식에 대해 진행하니 한번에 너무 많이 요청하는거같기도.. 다만 krx는 공식 API도 안만들고, 굳이 다른 분이 노력해서 만든 Pykrx 크롤링 모듈까지 차단하는게 참..
            if len(trade_data.reset_index().values.tolist())>minlen:
                adding_list=trade_data.reset_index().values.tolist()[(len(trade_data.values.tolist())-minlen):]
                adding_list=np.array(adding_list)
                if(check_outlier(adding_list)):
                    tot_dict[tickers]=adding_list
            elif len(trade_data.reset_index().values.tolist())==minlen:
                adding_list=trade_data.reset_index().values.tolist()
                adding_list=np.array(adding_list)
                if(check_outlier(adding_list)):

                    tot_dict[tickers]=adding_list
            else:
                tot_dict[tickers]=np.zeros_like(adding_list) # 설마 처음부터 이쪽 이슈가 뜨겠어.. 조회가 안 되거나, 일부 날짜가 생략된 경우에는 그냥 all_zero data를 넣어줌.
                adding_list = np.array(adding_list)
                continue
        except Exception as e:
            print(e)
            continue
    with open('dataset{}/trade_data.pickle'.format(end_date),'wb') as fw:
            pickle.dump(tot_dict, fw)
