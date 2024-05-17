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


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class LOTTO_RANDOMFOREST:
    def __init__(self, max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=300):
        self.rf_model = RandomForestRegressor(max_depth=max_depth, max_features=max_features,
                                              min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                              n_estimators=n_estimators)

    def fit(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Reshape data for training
        X_train_2d = self.X_train.reshape(self.X_train.shape[0], -1)
        X_test_2d = self.X_test.reshape(self.X_test.shape[0], -1)

        # Train the model
        self.rf_model.fit(X_train_2d, self.y_train)

        # Evaluate the model
        self.accuracy = self.rf_model.score(X_test_2d, self.y_test)

    def predict(self):
        # Reshape test data for prediction
        X_test_2d = self.X_test.reshape(self.X_test.shape[0], -1)

        # Make predictions
        predictions = self.rf_model.predict(X_test_2d)
        predicted_numbers = np.round(predictions).astype(int)

        # Cap numbers to max lotto value (e.g., 45)
        predicted_numbers = np.where(predicted_numbers > 45, 45, predicted_numbers)

        return predicted_numbers[0]

# 사용 예시
X = np.random.rand(100, 6)  # 임의의 입력 데이터
y = np.random.randint(1, 46, size=(100, 6))  # 임의의 출력 데이터 (로또 번호)

rnf_model = LOTTO_RANDOMFOREST()
rnf_model.fit(X, y)
