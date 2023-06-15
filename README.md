# Korea Stock Prediction with Naver news embedding Graph
---
## Description


본 모델은 Graph Neural Network와 KOBERT 등의 모델을 활용하여 뉴스데이터를 정제하여 주가를 예측하는 모델입니다.
KOSPI와 KOSDAC 데이터를 pykrx모듈로 수집하고, 네이버 뉴스의 제목과 본문(썸네일 본문으로, 본문 전체는 아닙니다)을 bs4 모듈로 수집하여 활용합니다.
제목과 본문은 Pretrain된 KOBERT를 활용하여 Top10까지 모아서 Length 768의 Embedding을 구해둡니다.

위 데이터들로 Graph를 구성하게 되는데, 각 Node의 Feature로는 다음 데이터들이 Embedding되어 사용됩니다.

-Node의 상대적인 날짜. 예를 들어 15일 ~ 19일의 데이터를 사용한다면 0(15일)~4(19일)의 값이 각 날짜에 할당됩니다.
-Top10 본문 Embedding.
-Top10 제목 Embedding.
-종가. 종목별로 Minmax scaling되어 들어갑니다
-등락율. 전일 종가대비 금일 종가의 비율을 %로 나타낸 것입니다.
-이익율. 실제로 거래를 한다고 생각했을 때를 생각한 지표입니다. 당일 시가대비 금일 종가의 비율을 %로 나타낸 것입니다.
-거래량. 값이 Log scale되어 들어갑니다.
-섹터. 회사의 Sector 정보가 들어갑니다. 회사별로 일정 Label이 주어지고(utils.py의 Label Changer 함수를 참고해주세요), one hot encoding되어서 들어갑니다.
-Ticker. 회사의 Ticker 정보입니다. 학습 때는 안 씁니다.
-Max Price & Min Price. Minmax scaling에 사용한 Max price와 Min Price가 들어갑니다. 

각 Edge의 Feature는 다음과 같습니다.

-날짜 Edge. 전날에서 다음날로 넘아가는 Node 둘을 연결합니다. 양방향 모두 연결하되, 다른 Edge Feature로써 인식되게끔 사용했습니다.
-뉴스 Edge. A회사의 뉴스나 뉴스 제목에서 B회사가 언급되면 A에서 B로 연결합니다. 이 역시 역방향도 연결하나, 다른 Edge Feature로써 인식되도록 사용했습니다.
-Max Volume Edge. 각 종목별로, 가장 거래량이 높은 날짜에 모든 노드를 연결합니다. 양방향 모두 연결하되, 다른 Edge Feature로써 인식되게끔 사용했습니다.
-Min Volume Edge. 각 종목별로, 가장 거래량이 낮은 날짜에 모든 노드를 연결합니다. 양방향 모두 연결하되, 다른 Edge Feature로써 인식되게끔 사용했습니다.

*Edge의 경우 특히나 고민을 많이한게 시간 순서가 반대로 된 Edge를 포함해야하나 싶었는데(18일을 예측할때 20일 데이터를 사용할 수 있는 구조), Hidden State Update를 굳이 못하게 할 이유도 없는 것 같아 그냥 사용합니다.

그 후, 위 그래프를 활용하여 마지막 날의 데이터를 예측하게 됩니다. 예측하는 데이터는 종가/등락율/이익율 중 하나입니다.(구현 상 3개 모드 중 하나를 하도록 되어있습니다.)

## How to use this model

1. 라이브러리 설치

pip install -r Requirements.txt

를 통해 라이브러리들을 받아주시고, 환경에 맞게 Pytorch(https://pytorch.org/get-started/locally/)와 Deep Graph Library(https://www.dgl.ai/pages/start.html)를 설치해주세요.
본 프로젝트는 Wandb logging을 하도록 구현되어 있습니다. 

2. Preprocessing

먼저


python main.py data_preprocessing

위 코드를 실행하면, 실행 날짜의 어제까지의 주가 데이터와 뉴스 데이터를 불러옵니다.
만약 오후 11시에 코드를 돌려서 오늘까지의 데이터를 얻고싶다면 data_preprocessing.py 파일의 end_date = datetime.now() - timedelta(1) 부분에서 - timedelta(1)를 지워주시면 됩니다.
1달 분량의 데이터를 가져오게 됩니다. 이를 늘리거나 줄이고 싶다면 역시나  data_preprocessing.py 파일의 start_date = end_date - timedelta(30) 부분을 수정해주세요.

1달 분량 기준으로 
뉴스 데이터는 Multiprocess사용시 15분내외, 아니면 3시간 내외쯤 걸리는 것 같습니다.
주가 데이터는 30분 내외쯤 걸립니다. Sleep없이는 5분이면 되기는 한데 KRX에서 IP 차단을 자주 먹입니다. 물론 Sleep 걸어도 가끔 먹입니다..(expecting value: line 1 column 1 (char 0) 이 에러가 뜹니다)

코드 실행이 끝나면 dataset{날짜}의 폴더가 생기고, 내부에 sector와 회사명 별로 정리된 폴더들이 생깁니다.


다음

python main.py graph_construct {date}

코드를 실행하면 dataset{date}의 데이터를 활용해 그래프를 구성합니다. Training용과 Inference용 2개를 구성하며, Inference는 아직 장이 열리지 않은 다음날의 Node도 포함하고 있습니다.
각 그래프는 training{date}.bin, inference{date}.bin으로 저장됩니다.


3. Training

python main.py train_gnn {date} {strategy}
코드를 실행하면 training{date}.bin 데이터를 활용해 학습을 진행합니다. 학습은 {strategy}에 해당하는 지표를 예측하도록 학습됩니다.
{strategy}는 up_ratio, profit, end_price 3개가 있습니다.
일단 50000epoch으로 설정해두고, 50epoch마다 저장되도록 합니다. 1epoch당 5초쯤 걸립니다. 
신기하게 빠르게 수렴하지않고, 30000epoch이 넘어가도 학습 하면서 더 loss가 줄어듭니다(validation도!). 다만 그게 미래의 데이터를 꼭 잘 맞춘다는 것을 보장하지는 아마 않을겁니다..?


4. Inference
python main.py predict_price {checkpoint_path} {date} {strategy}
코드를 실행하면 inferece{date}.bin과 {checkpoint_path} 체크포인트를 활용하여 마지막 날의 {strategy} 지표를 예측하고, 지표가 좋은 순서대로 정리하여 excel파일을 생성합니다.
checkpoint와 strategy는 동일한 strategy여야합니다!


##Sample

2023년 06월 12일까지의 Data를 기반으로, 13일 Data를 예측한 엑셀입니다.
종가의 경우 한 데이터(골드엔에스)가 상당히 이상하게 예측 되어있는데요, 모델 이상이라기보단 MinMax Scaling 알고리즘의 문제입니다.
10000원에서 20일동안 1000원까지 급락한 데이터를 1.0에서 0.0이 되도록 수치를 바꾸는데, 모델에서 0.8이라는 수치를 뱉으면 이게 8000원을 의미하게 되고, 하루만에 8배로 오른다는 그런.. 해석이 됩니다.
문제는 이게 다른 종목에서는 0.8이라는 수치가 일반적으로 급변까지는 아니기 때문에, 감안을 좀 못하는 것 같습니다.

이것 때문에 등락율과 이익율 strategy를 새로 고려했습니다...


##Todo?

해볼만한 것은..

현재 모델은 하루만 예측하게끔 되어있는데, GNN이 주가 예측을 하도록 하는 게 아니라 GNN은 그래프에서의 Node Embedding만을 주도록 하고, Transformer같은거에다 Cross Attention으로 Node Embedding주면서 하면 좀 더 길게 예측할 수 있을겁니다!
데이터의 날짜 길이가 고정이 안되어있어서 date embedding 가져오는게 좀 골치아픕니다.(특히 Pretraining) 아마 이건 금방 고쳐볼 것 같습니다. (고치고나서 학습 제대로 하면 ckpt올려보는 쪽으로!)
그래프 구성에서 Edge나 Node Feature를 더 넣거나, 더 줄여볼 수 있습니다. 경제학적인 해석이 들어갈 수도 있겠네요!
종가/등락율/이익율 3가지 전략 말고도 다른 방식도 고려해볼 수도 있긴 합니다.