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
# 다층 퍼셉트론 모델 정의
class LOTTO_MLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=6):
        super(LOTTO_MLP, self).__init__()
        # MLP 레이어 구성
        self.layer1 = nn.Linear(input_size * 5, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.layer1(x))
        return self.layer2(x)
    def predict(self):
        predicted_lotto_numbers = generate_lotto_numbers(mlp_model, initial_sequence, num_predictions=1)
        for idx, numbers in enumerate(predicted_lotto_numbers, 1):
            print(f"MLP : {numbers[0]}")
        return numbers[0]
# 학습 함수 정의
def train_mlp_model( batch_size=32, epochs=50, lr=1e-3):
    # 데이터 로드
    dataset = LottoDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 모델 초기화 및 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LOTTO_MLP().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    # 모델 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, label in dataloader:
            data, label = data.float().to(device), label.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    return model
# 로또 번호 생성 함수
def generate_lotto_numbers(model, initial_sequence, num_predictions=1):
    """주어진 시퀀스를 바탕으로 다음 로또 번호를 생성"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        generated_numbers = []
        current_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        for _ in range(num_predictions):
            predicted = model(current_sequence)
            # 1 ~ 45 범위로 제한하여 정수 변환
            predicted_rounded = torch.clamp(torch.round(predicted), min=1, max=45).cpu().numpy().astype(int)
            generated_numbers.append(predicted_rounded)
            # 새로운 번호를 시퀀스에 추가하고 맨 앞의 번호 제거
            new_sequence = np.vstack([current_sequence.cpu().numpy()[0, 1:], predicted_rounded])
            current_sequence = torch.tensor(new_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    return generated_numbers
# 모델 학습 및 새로운 로또 번호 생성
mlp_model = train_mlp_model()
initial_sequence = np.array([
    [3, 15, 21, 30, 33, 42],
    [7, 12, 25, 31, 35, 40],
    [5, 14, 18, 29, 32, 39],
    [9, 19, 22, 28, 37, 41],
    [4, 13, 20, 24, 26, 36]
])
predicted_lotto_numbers = generate_lotto_numbers(mlp_model, initial_sequence, num_predictions=1)
# for idx, numbers in enumerate(predicted_lotto_numbers, 1):
#     print(f"MLP : {numbers[0]}")
# MLP = numbers[0]
# # torch.save(pca_model.state_dict(), '/content/drive/MyDrive/LOTTO_MODEL/model_PCA')
# pickle.dump(mlp_model, open('/content/drive/MyDrive/LOTTO_MODEL/model_MLP.pkl','wb'))
