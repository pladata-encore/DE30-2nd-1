
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch
import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

"""# 데이터셋 정리

## 학습에 필요한 컬럼만 선택 후 새로운 데이터 프레임 생성
"""

lotto_df = pd.read_csv('app/models/로또번호.csv')
lotto_df.reset_index(drop=True, inplace=True)
lotto_df = lotto_df.drop(columns='Unnamed: 0')

"""## 학습 데이터 X, y 생성"""

X = []
y = []
data = lotto_df.values
window_size = 5
for i in range(len(data)-window_size):
  X.append(data[i : i + window_size ])
  y.append(data[i+window_size])
X = np.array(X)
y = np.array(y)

"""## 모델 학습에 필요한 하이퍼파라미터 정의"""

# 하이퍼 파라메터
input_size = X.shape[-1]
hidden_size = 64
num_layers = 5
output_size = input_size
epochs = 100
lr = 1e-4

"""## 모델 학습에 필요한 디바이스 정의"""

device = "cuda" if torch.cuda.is_available() else 'cpu'

"""## 데이터셋 클래스 정의"""

# 데이터셋
class LottoDataset(Dataset):
  def __init__(self):
    lotto_df = pd.read_csv('app/models/로또번호.csv')
    self.data = lotto_df.values

  def __len__(self):
    return len(self.data) - 5 # 사용가능한 배치

  def __getitem__(self, index):
    data = self.data[index: index+5, 1:]
    label = self.data[index+5, 1:]
    return data, label

"""## 데이터셋 객체 및 데이터 로더 정의"""

lottodataset = LottoDataset()
data,label = next(iter(lottodataset))
loader = DataLoader(lottodataset, batch_size=32)
data, label =  next(iter(loader))

"""# RNN"""

class LOTTO_RNN(nn.Module):
  def __init__(self):
    super(LOTTO_RNN,self).__init__()
    # 시계열 특징 추출
    self.rnn = nn.RNN(input_size = input_size,hidden_size=hidden_size,
                      num_layers=num_layers, batch_first=True)
     # (batch_size, 5,6) --> (batch_size, 5,64)

    # MLP 층 정의(분류기)  5*64 = 320
    self.fc1 = nn.Linear(in_features=5*64,out_features=64)
    self.fc2 = nn.Linear(in_features=64,out_features=6)
    self.relu = nn.ReLU()

  def forward(self,x, h):
    x, hn = self.rnn(x,h)  # x 마지막 RNN 층의 은닉 상태, hn 모든 RNN 층의 은닉 상태
    x = torch.reshape(x, (x.shape[0],-1))  # mlp층에 사용하기 위해서 모양 변경

    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x

  def predict(self):
    final_preds = torch.round(pred).type(torch.int)
    predict_RNN = final_preds.cpu().numpy()
    return predict_RNN[0]

# 학습
device = "cuda" if torch.cuda.is_available() else 'cpu'
rnn_model = LOTTO_RNN().to(device)
optim = Adam(rnn_model.parameters(),lr=1e-4)

iterator = tqdm.tqdm(range(50))
for epoch in iterator:
  for data, label in loader:
    optim.zero_grad()
    # 초기 은닉상태  (은닉층개수, 배치크기, 출력차원) -->0
    h0 = torch.zeros(5,data.shape[0],64).to(device)
    # 모델 forward - 예측
    pred = rnn_model(data.type(torch.FloatTensor).to(device), h0)
    # 손실
    loss = nn.MSELoss()(pred,label.type(torch.FloatTensor).to(device) )
    # 역전파
    loss.backward()
    # 가중치 업데이트(최적화)
    optim.step()
    iterator.set_description(f"epoch{epoch+1} loss:{loss.item()}")

# 평가용 데이터 로더
loader = DataLoader(lottodataset, batch_size=1)
preds = []
total_loss = 0

# 평가 수행
with torch.no_grad():
    for data, label in loader:
        # 초기 은닉 상태 생성
        h0 = torch.zeros(5, data.shape[0], 64).to(device)
        pred = rnn_model(data.type(torch.FloatTensor).to(device), h0)

        # 예측 값 반올림 및 정수형 변환
        rounded_pred = torch.round(pred).type(torch.int)

        # 필요한 경우 numpy 배열로 변환
        preds.append(rounded_pred.cpu().numpy())

        # 손실 계산
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
        total_loss += loss / len(loader)

# 전체 손실 출력
print(f"Total Loss: {total_loss.item()}")

# 예시로 첫 번째 배치의 예측 값과 실제 값을 확인
data, label = next(iter(loader))
h0 = torch.zeros(5, data.shape[0], 64).to(device)
pred = rnn_model(data.type(torch.FloatTensor).to(device), h0)

# 예측 값 반올림 및 정수형 변환
final_preds = torch.round(pred).type(torch.int)
# print("RNN:", final_preds.cpu().numpy())

# predict_RNN = final_preds.cpu().numpy()
# torch.save(rnn_model.state_dict(), '/content/drive/MyDrive/LOTTO_MODEL/model_RNN')
# pickle.dump(rnn_model, open('/content/drive/MyDrive/LOTTO_MODEL/model_RNN.pkl','wb'))

# rnn_model.predict()
