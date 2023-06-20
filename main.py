from data_preprocessing import data_preprocessing
from graph_construct import graph_construct
from graph_training import train_gnn, predict_price,get_all_prediction
import sys

def main():
    #여기서 date 쓰실 때 코드 돌리는 date가 아니라, data_preprocessing한 date를 쓰셔야합니다. dataset20230612 라면 15일에 돌려도 date를 20230612로!
    if sys.argv[1]=='data_preprocessing':
        data_preprocessing(to_file=True) # to_file False를 하게 되면 RAM에 dataset을 저장합니다. Colab환경용! 일단은 데이터셋이 너무 커서 colab용은 미뤄뒀습니다.
    if sys.argv[1]=='construct_graph': # use with date, ex) main.py graph_construct 20230612 -> dataset20230612를 불러옴
        graph_construct(sys.argv[2])
    if sys.argv[1]=='train_gnn': # use with date, and strategy ex) main.py train_gnn 20230612 up_ratio
        #strategy 3개는 up_ratio profit end_price입니다.
        train_gnn(sys.argv[2], sys.argv[3])
    if sys.argv[1]=='predict_price': # use with model path, and date, strategy  ex) main.py predict_price graph_checkpoints\2layer_graph=00-mask_loss=0.37.ckpt 20230612 up_ratio
        predict_price(sys.argv[2], sys.argv[3], sys.argv[4])
        
    if sys.argv[1]=='predict_all' : #use with ckpt directory. get all prediction from all checkpoints. ex) main.py predict_all graph_checkpoints
        get_all_prediction(sys.argv[2])

if __name__=='__main__':
    main()