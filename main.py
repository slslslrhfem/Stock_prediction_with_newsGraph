from data_preprocessing import data_preprocessing
from news_crawling import start_crawling
from graph_construct import graph_construct
from graph_training import train_gnn, predict_price
import sys
import os
def main():
    if sys.argv[1]=='data_preprocessing':
        data_preprocessing()
    if sys.argv[1]=='graph_construct': # use with date, ex) main.py graph_construct 20230612 -> dataset20230612를 불러옴
        graph_construct(sys.argv[2])
    if sys.argv[1]=='train_gnn': # use with date, and strategy ex) main.py train_gnn 20230612 up_ratio
        #strategy 3개는 up_ratio profit end_price입니다.
        train_gnn(sys.argv[2], sys.argv[3])
    if sys.argv[1]=='predict_price': # use with model path, and date, strategy  ex) main.py predict_price graph_checkpoints\2layer_graph=00-mask_loss=0.37.ckpt 20230612 up_ratio
        predict_price(sys.argv[2], sys.argv[3], sys.argv[4])


if __name__=='__main__':
    main()