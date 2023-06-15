# Korea Stock Prediction with Naver news embedding Graph

## Description


본 모델은 Graph Neural Network와 KOBERT 등의 모델을 활용하여 뉴스데이터를 정제하여 주가를 예측하는 모델입니다.


KOSPI와 KOSDAQ 데이터를 pykrx모듈로 수집하고, 네이버 뉴스의 제목과 본문(썸네일 본문으로, 본문 전체는 아닙니다)을 bs4 모듈로 수집하여 활용합니다.


제목과 본문은 Top10까지 모아서 Pretrain된 KOBERT로 Length 768의 Embedding을 구합니다.


위 데이터들로 Graph를 구성하게 되는데, 각 Node의 Feature로는 다음 데이터들이 Embedding되어 사용됩니다.

---


- **Node의 상대적인 날짜.** 예를 들어 15일 ~ 19일의 데이터를 사용한다면 0(15일)~4(19일)의 값이 각 날짜에 할당됩니다.
- **Top10 본문 Embedding**
- **Top10 제목 Embedding**
- **종가** 종목별로 Minmax scaling되어 들어갑니다
- **등락율** 전일 종가대비 금일 종가의 비율을 %로 나타낸 것입니다.
- **이익율** 금일 시가대비 금일 종가의 비율을 %로 나타낸 것입니다.
- **거래량** 값이 Log scale되어 들어갑니다.
- **섹터** 회사의 Sector 정보가 들어갑니다. 회사별로 일정 Label이 주어지고(utils.py의 Label Changer 함수를 참고해주세요), one hot encoding되어서 들어갑니다.
- **Ticker** 회사의 Ticker 정보입니다. 학습 때는 안 쓰고, 마지막 Inference때 씁니다.
- **최고/최저가격** Minmax scaling에 사용한 Max price와 Min Price가 들어갑니다. 

---

각 Edge의 Feature는 다음과 같습니다.

---

- **날짜 Edge** 전날에서 다음날로 넘아가는 Node 둘을 연결합니다. 양방향 모두 연결하되, 다른 Edge Feature로써 인식되게끔 사용했습니다.
- **뉴스 Edge** A회사의 뉴스나 뉴스 제목에서 B회사가 언급되면 A에서 B로 연결합니다. 이 역시 역방향도 연결하나, 다른 Edge Feature로써 인식되도록 사용했습니다.
- **Max Volume Edge** 각 종목별로, 가장 거래량이 높은 날짜에 모든 노드를 연결합니다. 양방향 모두 연결하되, 다른 Edge Feature로써 인식되게끔 사용했습니다.
- **Min Volume Edge** 각 종목별로, 가장 거래량이 낮은 날짜에 모든 노드를 연결합니다. 양방향 모두 연결하되, 다른 Edge Feature로써 인식되게끔 사용했습니다. 

---

## How to use this model

**1. 라이브러리 설치**

환경에 맞게 [Pytorch](https://pytorch.org/get-started/locally/)와 [Deep Graph Library](https://www.dgl.ai/pages/start.html)를 설치해주세요.


```
pip install -r requirements.txt
```

를 통해 라이브러리들을 설치해주세요. 또한 본 프로젝트는 [Wandb](https://wandb.ai/home) logging을 하도록 구현되어 있습니다. 따라서 계정이 필요하거나, 코드를 수정할 필요가 있을 수도 있습니다.

**2. Preprocessing**


```
python main.py data_preprocessing
```

위 코드를 실행하면, 실행 날짜의 전날까지의 주가 데이터와 뉴스 데이터를 불러옵니다. 코드 실행 중 날짜가 변해도 괜찮습니다.



1달 분량 기준으로 
뉴스 데이터는 Multiprocess사용시 15분내외, 아니면 3시간 내외쯤, 주가 데이터는 30분 내외쯤 걸립니다.


 KRX에서 IP 차단을 가끔 먹입니다. ```(expecting value: line 1 column 1 (char 0)``` 이 에러가 뜹니다)


코드 실행이 끝나면 dataset{날짜}의 폴더가 생기고, 내부에 sector와 회사명 별로 정리된 폴더들이 생깁니다.


```
python main.py graph_construct {date}
```


코드를 실행하면 dataset{date}의 데이터를 활용해 그래프를 구성합니다. Training용과 Inference용 2개를 구성하며, Inference는 아직 장이 열리지 않은 다음날의 Node도 포함하고 있습니다.
각 그래프는 training{date}.bin, inference{date}.bin으로 저장됩니다.


**3. Training**


```
python main.py train_gnn {date} {strategy}
```


코드를 실행하면 training{date}.bin 데이터를 활용해 학습을 진행합니다. 학습은 {strategy}에 해당하는 지표를 예측하도록 학습됩니다.


{strategy}는 up_ratio, profit, end_price 3개가 있습니다.


일단 50000epoch으로 설정해두고, 50epoch마다 저장되도록 합니다. 1epoch당 5초쯤 걸립니다. 


**4. Inference**
   
```
python main.py predict_price {checkpoint_path} {date} {strategy}
```


코드를 실행하면 inferece{date}.bin과 {checkpoint_path} 체크포인트를 활용하여 마지막 날의 {strategy} 지표를 예측하고, 지표가 좋은 순서대로 정리하여 excel파일을 생성합니다.
checkpoint와 strategy는 동일한 strategy여야합니다!


## Sample



**Sample** 폴더 안에 있는 excel들은 2023년 06월 12일까지의 Data를 기반으로, 13일 Data를 예측한 엑셀입니다.


지표에서 상위 20개 종목을 당일 시가로 균일하게 매수했다고 하면,

```
End-Price : 0.344%
Profit : 0.28%
Up_ratio : 0.922%
```

위 만큼 변화한 가격으로 종가에 매도할 수 있었습니다.
2023년 06월 13일 KOSPI는 0.3%, KOSDAQ은 1.2% 가량 상승했으니, 돈을 많이 벌게 해주는지는 의문이 있군요


종가의 경우 한 데이터(골드엔에스)가 상당히 이상하게 예측 되어있는데요, 모델 이상이라기보단 MinMax Scaling 알고리즘의 문제입니다.


10000원에서 20일동안 1000원까지 급락한 데이터를 1.0에서 0.0이 되도록 수치를 바꾸는데, 모델에서 0.8이라는 수치를 뱉으면 이게 8000원을 의미하게 되고, 하루만에 8배로 오른다는 그런 해석이 됩니다.




## References

---

- https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf Kobert
- https://arxiv.org/pdf/2110.07190.pdf WHY PROPAGATE ALONE? PARALLEL USE OF LABELS AND FEATURES ON GRAPHS
- https://arxiv.org/pdf/2103.13355v4.pdf Bag of Tricks for Node Classification with Graph Neural Networks
- https://arxiv.org/pdf/2009.03509v5.pdf  Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification
- https://link.springer.com/article/10.1007/s11042-022-13231-1 A graph neural network-based stock forecasting method utilizing multi-source heterogeneous data fusion

---

## Todo?

해볼만한 것은..

- 현재 모델은 하루만 예측하게끔 되어있는데, GNN이 주가 예측을 하도록 하는 게 아니라 GNN은 그래프에서의 Node Embedding만을 주도록 하고, Transformer같은거에다 Cross Attention으로 Node Embedding주면서 하면 좀 더 길게 예측할 수 있을겁니다!
- 솔직히 그냥 GNN 안쓰고 Transformer에 News Embedding만 넣는게 더 잘할 것 같습니다.
- 데이터의 날짜 길이가 고정이 안되어있어서 date embedding 가져오는게 좀 골치아픕니다.(특히 Pretraining에서)
- 그래프 구성에서 Edge나 Node Feature를 더 넣거나, 더 줄여볼 수 있습니다. 경제학적인 해석이 들어갈 수도 있겠네요!
- 종가/등락율/이익율 3가지 전략 말고도 다른 방식도 고려해볼 수도 있긴 합니다.
- GNN 모델을 다양하게 시도해볼 수 있을 것 같습니다.
