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

lotto_df = pd.read_csv('로또번호.csv')
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
    lotto_df = pd.read_csv('로또번호.csv')
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


# RNN 모델 정의
class LOTTO_RNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=6):
        super(LOTTO_RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, hn = self.rnn(x, h)
        out = out[:, -1, :]  # 시계열의 마지막 시점
        out = self.fc(out)
        return out

# RNN 모델 학습 및 번호 예측
def train_rnn_model(epochs=50, batch_size=32):
    # 데이터 로더 준비
    dataset = LottoDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rnn_model = LOTTO_RNN().to(device)
    optimizer = Adam(rnn_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 모델 학습
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            h0 = torch.zeros(2, inputs.size(0), 64).to(device)
            outputs = rnn_model(inputs, h0)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
    return rnn_model

def predict(self):
  # 예측용 로또 번호 생성 예시 (예측에 적합한 이전 데이터 필요)
  initial_sequence = torch.tensor([[3, 15, 21, 30, 33, 42], [4, 9, 14, 18, 27, 31], [6, 8, 10, 19, 23, 25], [11, 12, 16, 20, 24, 34], [1, 5, 7, 13, 29, 37]], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
  h0 = torch.zeros(2, initial_sequence.size(0), 64).to("cuda" if torch.cuda.is_available() else "cpu")
  predicted_numbers = model(initial_sequence, h0)
  predicted_numbers = torch.round(predicted_numbers).type(torch.int)
  print("TRANSFORMER RNN:", predicted_numbers.cpu().numpy())
  return predicted_numbers.cpu().numpy()

# 모델 학습
rnn_model = train_rnn_model()

# 예측용 로또 번호 생성 예시 (예측에 적합한 이전 데이터 필요)
initial_sequence = torch.tensor([[3, 15, 21, 30, 33, 42], [4, 9, 14, 18, 27, 31], [6, 8, 10, 19, 23, 25], [11, 12, 16, 20, 24, 34], [1, 5, 7, 13, 29, 37]], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
h0 = torch.zeros(2, initial_sequence.size(0), 64).to("cuda" if torch.cuda.is_available() else "cpu")
predicted_numbers = rnn_model(initial_sequence, h0)
predicted_numbers = torch.round(predicted_numbers).type(torch.int)
# print("TRANSFORMER RNN:", predicted_numbers.cpu().numpy())

# RNN = predicted_numbers.cpu().numpy()
# # torch.save(pca_model.state_dict(), '/content/drive/MyDrive/LOTTO_MODEL/model_PCA')
# pickle.dump(rnn_model, open('/content/drive/MyDrive/LOTTO_MODEL/model_RNN.pkl','wb'))

class LottoTransformer(nn.Module):
    def __init__(self, input_size=6, d_model=64, num_heads=8, num_layers=4, output_size=6):
        super(LottoTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Use only the last timestep's output for prediction
        return self.fc(x)
    
    def predict(self):
        # Prediction example
        initial_sequence = torch.tensor([[3, 15, 21, 30, 33, 42], [4, 9, 14, 18, 27, 31], [6, 8, 10, 19, 23, 25], [11, 12, 16, 20, 24, 34], [1, 5, 7, 13, 29, 37]], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        predicted_numbers = trans_model(initial_sequence)
        predicted_numbers = torch.round(predicted_numbers).type(torch.int)
        print("Predicted Lotto Numbers:", predicted_numbers.cpu().numpy())
        return predicted_numbers[0].cpu().numpy()

def train_transformer_model(epochs=50, batch_size=32):
    dataset = LottoDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trans_model = LottoTransformer().to(device)
    optimizer = Adam(trans_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            outputs = trans_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}")
    return trans_model

def predict(self):
  # Prediction example
  initial_sequence = torch.tensor([[3, 15, 21, 30, 33, 42], [4, 9, 14, 18, 27, 31], [6, 8, 10, 19, 23, 25], [11, 12, 16, 20, 24, 34], [1, 5, 7, 13, 29, 37]], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
  predicted_numbers = trans_model(initial_sequence)
  predicted_numbers = torch.round(predicted_numbers).type(torch.int)
  print("Predicted Lotto Numbers:", predicted_numbers.cpu().numpy())
  return predicted_numbers.cpu().numpy()

# Train the transformer model
trans_model = train_transformer_model()

# Prediction example
initial_sequence = torch.tensor([[3, 15, 21, 30, 33, 42], [4, 9, 14, 18, 27, 31], [6, 8, 10, 19, 23, 25], [11, 12, 16, 20, 24, 34], [1, 5, 7, 13, 29, 37]], dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
predicted_numbers = trans_model(initial_sequence)
predicted_numbers = torch.round(predicted_numbers).type(torch.int)
# print("Predicted Lotto Numbers:", predicted_numbers.cpu().numpy())

# # Save the trained transformer model
# pickle.dump(trans_model, open('/content/drive/MyDrive/LOTTO_MODEL/model_TRANSFORMER.pkl', 'wb'))
