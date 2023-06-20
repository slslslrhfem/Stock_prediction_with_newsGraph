import torch_geometric
import torch
import numpy as np
import dgl
from pathlib import Path
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Subset
from tqdm import tqdm
import copy
from pykrx import stock
import random
import pandas as pd
import FinanceDataReader as fdr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphDataset(dgl.data.DGLDataset):
    def __init__(self):
        ROOT = Path("processed_pattern_with_order/conformed_lpd_17/lpd_17_cleansed")
        self.files = sorted(ROOT.glob('[A-Z]/[A-Z]/[A-Z]/TR*/*.bin'))
        self.graph_list = []

        # Data object is from PYG, It can be replaced with DGL.graph object in the same way. 
        # However, when N is large, this will case OOM.
    def __getitem__(self,index):
        graph_list, _ = dgl.load_graphs(str(self.files[index]))
        graph = graph_list[0]
        
        label = graph.ndata['sector'][0]
        
         # genre label
        return graph, [label]
    
    def __len__(self):
        return len(self.files)


class GraphDataset(dgl.data.DGLDataset): # 그래프가 1개긴 한데 pytorch lightning활용을 위해 형식상.. Validation은 마지막 날 가격 예측
    def __init__(self,date):
        self.graph, _ = dgl.load_graphs('training{}.bin'.format(date))

    def __getitem__(self,index):
        return self.graph[index]
    
    def __len__(self):
        return 1

class InfGraphDataset(dgl.data.DGLDataset): # 이 Dataset으로 학습을 하는 전략도 시도해볼 수 있음. 장점은 그래프의 구조가 동일하다는 것
    #단점은 마지막날을 예측하는 Task가 Training과정에 주어질 수 없다는 것.
    def __init__(self,date):
        self.graph, _ = dgl.load_graphs('inference{}.bin'.format(date))

    def __getitem__(self,index):
        return self.graph[index]
    
    def __len__(self):
        return 1
    
def train_gnn(date, strategy):

    train_dataset = GraphDataset(date)
    train_loader = GraphDataLoader(train_dataset, batch_size=1)
    num_date = max(train_dataset[0].ndata['date'])+1
    classifier_model = graph_price_predictor(1536 + 420,128,num_date, strategy)
    
   
    wandb_logger = WandbLogger()
    checkpoint_callback = ModelCheckpoint(dirpath = 'graph_checkpoints/', save_top_k = 10, filename=str(strategy)+'_'+str(date)+'_bn_2layer_graph-{epoch:02d}-{mask_loss:.6f}-{val_mask_loss:.6f}', monitor="val_mask_loss")
    trainer = pl.Trainer(logger = wandb_logger, accelerator='auto', devices=1 if torch.cuda.is_available() else None,
                          max_epochs=2000,detect_anomaly=True,callbacks=[checkpoint_callback])
    trainer.fit(classifier_model, train_loader, train_loader) # epoch이 사실상 10000이 넘어갈때 까지 계속 loss가 줄어서 validation을 50 epoch당 한번만
    # validation때 맞춰야하는 값들 분명 training때 mask해서 all 0로 넣어주는걸 print까지해서 확인했는데 수상할정도로 val_mask_loss가 잘 떨어짐.. 잘 되는거라 좋은거긴 한데 의심이 됨..
    return 0

class graph_price_predictor(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, num_date, strategy):
        super(graph_price_predictor, self).__init__()
        self.strategy = strategy
        self.sector_emb = nn.Embedding(16,128)
        self.masking_emb = nn.Embedding(2,32)
        self.position_emb = nn.Embedding(128, 256) # 128 == max(num_date)
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, 8,regularizer='basis', num_bases=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, 8,regularizer='basis', num_bases=2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = dglnn.RelGraphConv(hidden_dim, hidden_dim, 8,regularizer='basis', num_bases=2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = dglnn.RelGraphConv(hidden_dim, hidden_dim, 8,regularizer='basis', num_bases=2)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,1)
        self.activation = nn.Sigmoid()
        
    def forward(self, g, h, efeat):
        # Apply graph convolution and activation.zx
        h = self.bn1(F.relu(self.conv1(g, h, efeat)))
        h = self.bn2(F.relu(self.conv2(g, h, efeat)))
        #h = self.bn3(F.relu(self.conv3(g, h, efeat)))
        #h = self.bn4(F.relu(self.conv4(g, h, efeat)))

        h = self.linear1(h)
        pred = self.linear2(h)
        return pred

    
    def training_step(self, batch, batch_idx):
        graph = batch
        max_date = max(graph.ndata['date'])+1 #0~18까지 값이 있으면 19일차 까지 있는거
        last_day_idx = [i*max_date+max_date-1 for i in range(int(len(graph.ndata['date'])/max_date))] #마지막날 데이터를 validation으로 사용하기위해 아예 가림. 
        random_idx = np.random.choice(len(graph.ndata[self.strategy]), int(0.3*len(graph.ndata[self.strategy]))) # 30%를 Masking하되, 이미 가려진 last_day는 제외
        mask_idx = [i for i in random_idx if i % max_date != max_date-1]# 학습은 Node들을 Masking하고, 그 Node를 예측하는 Label tricking을 사용한다.
        target = copy.deepcopy(graph.ndata[self.strategy][mask_idx]) 
        graph.ndata['Masking'] = torch.zeros(len(graph.ndata[self.strategy])).to(device)
        for idx in mask_idx: # 직접 가린 데이터들
            graph.ndata[self.strategy][idx] = 0
            graph.ndata['volume'][idx] = 0 #거래량은 예측할 필요는 없지만 inference시 주어지지 않는 정보이므로 masking 함. 예측에서는 안씀
            graph.ndata['Masking'][idx] = 1
        for idx in last_day_idx: # 이미 안주어져있는(이라고 가정하는) 마지막 날 데이터. Validation에서 성능 확인을 위해 사용한다.
            graph.ndata[self.strategy][idx] = 0
            graph.ndata['volume'][idx] = 0 #거래량은 예측할 필요는 없지만 inference시 주어지지 않는 정보이므로 masking 함. 예측에서는 안씀
            graph.ndata['Masking'][idx] = 1
        masked_graph = graph
        to_predict = masked_graph.ndata[self.strategy].to(torch.int).unsqueeze(1).to(torch.float32) # 1~2 or 0. 0은 masking된 경우이다. 다만 이게 Label 형태가 아니라 feature에 0을 그냥 집어넣는게 Label embedding 형태에서는 괜찮은데(증명이 있음),
        # Feature에서는 Model에서 학습을 제대로 할지 걱정이다. 따로 Masking Label Feature를 넣어서 일단은 처리.
        volumes = masked_graph.ndata['volume'].unsqueeze(1) # 1~log(max(volume)) or 0. 거래량이 100억쯤 되면 10이니 아주 커지진 않음
        title_embedding = masked_graph.ndata['title_embedding'].to(torch.float32) #-1~1. 기사 or 제목이 완전히 주어지지 않는, 즉 하루동안 뉴스가 아예 없는 경우에 들어가는 vector가 0-vector가 아니고 BERT('')값임에 유의하자. 0-vector가 더 좋을수도 있음
        article_embedding = masked_graph.ndata['article_embedding'].to(torch.float32) # -1~1
        date_info = masked_graph.ndata['date'].to(torch.int) # int, will feed in embedding layer
        sector = masked_graph.ndata['sector'].to(torch.int) # int, will feed in embedding layer
        max_price = torch.log(masked_graph.ndata['max_value']).unsqueeze(1).to(torch.float32) # 1~6쯤
        min_price = torch.log(masked_graph.ndata['min_value']).unsqueeze(1).to(torch.float32) # 1~6쯤 Feature Range가 조금씩 다르긴 한데, 큰 격차가 있지는 않아서 concat해서 사용해도 무리는 없다!

        masking_embedding = self.masking_emb(graph.ndata['Masking'].to(torch.int))
        date_embedding = self.position_emb(date_info).to(torch.float32)
        sector_embedding = self.sector_emb(sector).to(torch.float32)
        feature = torch.cat([to_predict, volumes, masking_embedding, max_price, min_price, date_embedding, sector_embedding, title_embedding, article_embedding],dim=1).to(torch.float32)# 아마 16000쯤..? 근데 15000이상이 text긴함. text embedding 줄여도 될지도..
        efeat = masked_graph.edata['edge_feature'].to(torch.int)
    
        prediction = self(masked_graph, feature, efeat).to(torch.float32)
        if self.strategy == 'end_price':
            prediction = self.activation(prediction) # 나머지 전략들은 0~1이 아니라 activation(특히 sigmoid)를 못씀
        masking_prediction_loss = F.mse_loss(prediction[mask_idx].to(torch.float32), (target).unsqueeze(1).to(torch.float32))
        #print('training',prediction[mask_idx],target)
        self.log('mask_loss', masking_prediction_loss)

        return masking_prediction_loss

    def validation_step(self, batch, batch_idx):#이는 실제 Inference와 같이 가장 마지막 날 만을 Masking해서 예측합니다.
        graph = batch
        max_date = max(graph.ndata['date'])+1 #0~18까지 값이 있으면 19일차 까지 있는거
        mask_idx = np.array([int(i*max_date+max_date-1) for i in range(int(len(graph.ndata['date'])/max_date))]) # validation은 마지막 날만 가림
        target = copy.deepcopy(graph.ndata[self.strategy][mask_idx]) # training 중에는 Masking하여 주지 않았던 예측해야하는 값들

        graph.ndata['Masking'] = torch.zeros(len(graph.ndata[self.strategy])).to(device)
        for idx in mask_idx:
            graph.ndata[self.strategy][idx] = 0 # 예측할 값
            graph.ndata['volume'][idx] = 0  # 주어져있지 않은 값
            graph.ndata['Masking'][idx] = 1
        masked_graph = graph
        end_prices = masked_graph.ndata[self.strategy].to(torch.int).unsqueeze(1).to(torch.float32) # 1~2 or 0. 0은 masking된 경우이다. 다만 이게 Label 형태가 아니라 feature에 0을 그냥 집어넣는게 Label embedding 형태에서는 괜찮은데(증명이 있음),
        # Feature에서는 Model에서 학습을 제대로 할지 애매하므로 Masking Label Feature를 넣어서 일단은 처리.
        volumes = masked_graph.ndata['volume'].unsqueeze(1) # 1~log(max(volume)) or 0. 
        title_embedding = masked_graph.ndata['title_embedding'].to(torch.float32) #-1~1. 기사 or 제목이 완전히 주어지지 않는, 즉 하루동안 뉴스가 아예 없는 경우에 들어가는 vector가 0-vector가 아니고 BERT('')값임에 유의하자. 0-vector가 더 좋을수도 있음
        article_embedding = masked_graph.ndata['article_embedding'].to(torch.float32) # -1~1
        date_info = masked_graph.ndata['date'].to(torch.int) # int, will feed in embedding layer
        sector = masked_graph.ndata['sector'].to(torch.int) # int, will feed in embedding layer
        max_price = torch.log(masked_graph.ndata['max_value']).unsqueeze(1).to(torch.float32) # 1~6쯤
        min_price = torch.log(masked_graph.ndata['min_value']).unsqueeze(1).to(torch.float32) # 1~6쯤 Feature Range가 조금씩 다르긴 한데, 큰 격차가 있지는 않아서 concat해서 사용

        masking_embedding = self.masking_emb(graph.ndata['Masking'].to(torch.int))
        date_embedding = self.position_emb(date_info).to(torch.float32)
        sector_embedding = self.sector_emb(sector).to(torch.float32)
        feature = torch.cat([end_prices, volumes, masking_embedding, max_price, min_price, date_embedding, sector_embedding, title_embedding, article_embedding],dim=1).to(torch.float32)
        efeat = masked_graph.edata['edge_feature'].to(torch.int)
    
        prediction = self(masked_graph, feature, efeat).to(torch.float32)
        if self.strategy == 'end_price':
            prediction = self.activation(prediction)
        masking_prediction_loss = F.mse_loss(prediction[mask_idx].to(torch.float32), (target).unsqueeze(1).to(torch.float32))
        #end_price 학습시에, 초반 loss가 60가까이 뛰는데 이게 수학적으로 가능한건지 모르겠음. prediction도 sigmoid라 0~1이고 target도 0~1인데..
        
        #print('validation',prediction[mask_idx],target)
        self.log('val_mask_loss', masking_prediction_loss, batch_size=1) 
        return masking_prediction_loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = 0.001)
        return opt
    

def predict_price(model_path,date, strategy):
    name_list=[]
    ticker_list=[]
    price_list=[]
    up_ratio_list=[]
    sector_list=[]
    model_output=[]
    test_dataset = InfGraphDataset(date)
    datas=test_dataset[0]

    num_date = max(datas.ndata['date'])+1
    classifier_model = graph_price_predictor(1536 + 420,128,num_date, strategy) # 1536은 text, 420은 나머지 embedding
    checkpoint = torch.load(model_path)
    classifier_model.load_state_dict(checkpoint["state_dict"])

    
    prediction = get_prediction(datas,classifier_model)
    for i in range(len(prediction)):
        last_price = (datas.ndata['end_price'][i+1])*(datas.ndata['max_value'][i+1]-datas.ndata['min_value'][i+1]) + datas.ndata['min_value'][i+1]
        ticker = datas.ndata['ticker'][i+1]
        dateinfo = datas.ndata['date'][i+1]
        print(last_price,ticker,i, dateinfo)
        if datas.ndata['date'][i] ==num_date-1:
            ticker = datas.ndata['ticker'][i]
            ticker_list.append(str(ticker.item()).zfill(6))

            #minmax, minmax scaling된 가격을 prediction
            if strategy == 'end_price':
                pred_price = float(prediction[i])*(float(datas.ndata['max_value'][i])-float(datas.ndata['min_value'][i])) + datas.ndata['min_value'][i]
                last_price = (datas.ndata['end_price'][i-1])*(datas.ndata['max_value'][i-1]-datas.ndata['min_value'][i-1]) + datas.ndata['min_value'][i-1]
                up_ratio_list.append((float(pred_price/last_price)-1)*100)
                price_list.append(float(pred_price))

            elif strategy == 'up_ratio':
                pred_price = prediction[i] * np.exp(datas.ndata['end_price'][i-1])
                last_price = (datas.ndata['end_price'][i-1])*(datas.ndata['max_value'][i-1]-datas.ndata['min_value'][i-1]) + datas.ndata['min_value'][i-1]
                up_ratio_list.append(float(prediction[i])) # up_ratio는 전일 종가대비, profit은 당일 시가대비 비율이 들어갑니다.
                price_list.append(float((1+prediction[i]/100)*last_price)) # 이거는 엄밀히는 up_ratio에서는 맞고, profit에서는 조금 다르긴 합니다. (당일 시가를 가져오기 힘든 경우가 꽤 있음. 특히 마지막 날)

            #profit, 당일 시가 대비 얼마나 오를지가 prediction
            elif strategy == 'profit':
                pred_price = prediction[i] # 이 경우는 비율을 예측한거고, 예측값의 open_price가 안주어져있어서 엄밀히는 가격이 아니긴 한데 변수명을 그대로 둡니다
                last_price = (datas.ndata['end_price'][i-1])*(datas.ndata['max_value'][i-1]-datas.ndata['min_value'][i-1]) + datas.ndata['min_value'][i-1]
                up_ratio_list.append(float(prediction[i])) # up_ratio는 전일 종가대비, profit은 당일 시가대비 비율이 들어갑니다.
                price_list.append(float((1+prediction[i]/100)*last_price)) 

                
            model_output.append([float(prediction[i]),float(datas.ndata['max_value'][i]),float(datas.ndata['min_value'][i])])
            asdf
    up_ratio_list = np.array(up_ratio_list)
    price_list = np.array(price_list)
    ticker_list = np.array(ticker_list)
    model_output = np.array(model_output)

    stocks = fdr.StockListing('KRX')
    all_ticker_list = list(stocks["Symbol"])
    all_name_list = list(stocks["Name"])
    Sector_Label = list(stocks["Sector"])

    for ticker in ticker_list:
        name_list.append(all_name_list[all_ticker_list.index(ticker)])
        sector_list.append(Sector_Label[all_ticker_list.index(ticker)])

    sector_list = np.array(sector_list)
    name_list = np.array(name_list)
    inds = (-up_ratio_list).argsort()

    sorted_up_ratio = up_ratio_list[inds]
    sorted_price = price_list[inds]
    sorted_name = name_list[inds]
    sorted_ticker = ticker_list[inds]
    sorted_last_price = sorted_price/(1+sorted_up_ratio/100)
    sorted_model_output = model_output[inds]
    sorted_sector_list = sector_list[inds]

    fin_list=[]
    df = stock.get_market_ohlcv(str(int(date)+1),market="ALL") # 이거는 토,일요일 고려를 안한 구현입니다. dataset이 12일이면, 13일의 주가 데이터를 가져옵니다. 만약 15일을 가져오고 싶다면 1을 3으로 바꿔야하긴 합니다.
    # 이미 열린 개장일은 가져올 수 있고, 따라서 공휴일 등이 끼여있어도 돌아가도록 짰지만, 다음 개장일이 무엇인지 가져오기는 꽤 까다롭습니다(공휴일 등)..
    portfolio_val = 0 # 당일 시가로 매수를 했다면 과연..?
    for i in range(len(sorted_name)):
        if df.loc[sorted_ticker[i]].values.tolist()[0]!=0:
            true_data = [df.loc[sorted_ticker[i]].values.tolist()[0],df.loc[sorted_ticker[i]].values.tolist()[3], df.loc[sorted_ticker[i]].values.tolist()[3]/df.loc[sorted_ticker[i]].values.tolist()[0]]
        else:
            true_data=['Not given yet!',0,1]
        data = [sorted_name[i], sorted_up_ratio[i], sorted_price[i], sorted_ticker[i], sorted_last_price[i], sorted_sector_list[i],sorted_model_output[i], true_data]
        if i<20:
            portfolio_val += (true_data[2]-1)*5  # %단위로 만들기 위해 100을 곱하고 20개 항목을 사므로 20을 나눔
        fin_list.append(data)

    fin_list = np.array(fin_list)

    df = pd.DataFrame(data = fin_list,columns = ['name', 'up_ratio', 'predicted price', 'ticker', 'last price','sector','model output','actual_price(open, end, ratio if you buy this with open price)'])
    print("위 모델이 추천하는 상위 20개 종목을 균일 배분하여 시가로 매수했다면 ",portfolio_val,"% 만큼 변한 상태로 종가에 판매할 수 있었습니다.")
    df.to_excel('{}result{}.xlsx'.format(strategy, date), index=False)

    return 0

def get_prediction(graph, model):
    end_prices = graph.ndata['end_price'].to(torch.int).unsqueeze(1).to(torch.float32) # 1~2 or 0. 0은 masking된 경우이다. 다만 이게 Label 형태가 아니라 feature에 0을 그냥 집어넣는게 Label embedding 형태에서는 괜찮은데(증명이 있음),
    # Feature에서는 Model에서 학습을 제대로 할지 걱정이다. 따로 Masking Label Feature를 넣어서 일단은 처리.
    volumes = graph.ndata['volume'].unsqueeze(1) # 1~log(max(volume)) or 0. 거래량이 100억쯤 되면 10이니 아주 커지진 않음
    title_embedding = graph.ndata['title_embedding'].to(torch.float32) #-1~1. 기사 or 제목이 완전히 주어지지 않는, 즉 하루동안 뉴스가 아예 없는 경우에 들어가는 vector가 0-vector가 아니고 BERT('')값임에 유의하자. 0-vector가 더 좋을수도 있음
    article_embedding = graph.ndata['article_embedding'].to(torch.float32) # -1~1
    date_info = graph.ndata['date'].to(torch.int) # int, will feed in embedding layer
    sector = graph.ndata['sector'].to(torch.int) # int, will feed in embedding layer
    max_price = torch.log(graph.ndata['max_value']).unsqueeze(1).to(torch.float32) # 1~6쯤
    min_price = torch.log(graph.ndata['min_value']).unsqueeze(1).to(torch.float32) # 1~6쯤 Feature Range가 조금씩 다르긴 한데, 큰 격차가 있지는 않아서 concat해서 사용해도 무리는 없다!
    graph.ndata['Masking'] = torch.zeros(len(graph.ndata['end_price']))
    max_date = max(graph.ndata['date'])+1 #0~18까지 값이 있으면 19일차 까지 있는거
    mask_idx = [i*max_date+max_date-1 for i in range(int(len(graph.ndata['date'])/max_date))] # validation/inference는 마지막 날만 가림. 애초에 inference는 이미 값이 없다.
    for idx in mask_idx:
        graph.ndata['Masking'][idx] = 1

    masking_embedding = model.masking_emb(graph.ndata['Masking'].to(torch.int))
    date_embedding = model.position_emb(date_info).to(torch.float32)
    sector_embedding = model.sector_emb(sector).to(torch.float32)
    feature = torch.cat([end_prices, volumes, masking_embedding, max_price, min_price, date_embedding, sector_embedding, title_embedding, article_embedding],dim=1).to(torch.float32)
    efeat = graph.edata['edge_feature'].to(torch.int)
    prediction = model(graph, feature, efeat).to(torch.float32)
    return prediction