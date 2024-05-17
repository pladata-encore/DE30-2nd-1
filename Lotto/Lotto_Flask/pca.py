
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

class LOTTO_PCA:
    def __init__(self, n_clusters=5, n_components=2):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.lotto_df = None
        self.cluster_centers = None
    def load_and_preprocess_data(self):
        self.lotto_df = lotto_df
        return self.scaler.fit_transform(lotto_df)
    def fit(self):
        lotto_scaled = self.load_and_preprocess_data()
        pca_result = self.pca.fit_transform(lotto_scaled)
        self.kmeans.fit(pca_result)
        centers_pca = self.scaler.inverse_transform(self.pca.inverse_transform(self.kmeans.cluster_centers_))
        # Ensure the number of DataFrame columns matches the actual data columns
        self.cluster_centers = pd.DataFrame(np.round(centers_pca).astype(int), columns=lotto_df.columns)
        self.cluster_centers['Cluster'] = range(1, 1 + self.kmeans.n_clusters)
    def predict(self):
        if self.lotto_df is None:
            raise ValueError("You must load and preprocess the data before prediction.")
        lotto_scaled = self.scaler.transform(self.lotto_df)
        pca_result = self.pca.transform(lotto_scaled)
        labels = self.kmeans.predict(pca_result)
        self.lotto_df['Cluster'] = labels
        pca_model = LOTTO_PCA(n_clusters=1, n_components=2)
        pca_model.fit()
        cluster_centers = pca_model.get_cluster_centers()
        cluster_array = np.array(cluster_centers.iloc[:, :-1])  # Exclude the 'Cluster' column
        return cluster_array.tolist()
    def get_cluster_centers(self):
        if self.cluster_centers is None:
            raise ValueError("Cluster centers are not available. Please call fit() before getting centers.")
        return self.cluster_centers
# Example usage of the class
pca_model = LOTTO_PCA(n_clusters=1, n_components=2)
pca_model.fit()
# Retrieve and format the cluster centers as per the requested output
cluster_centers = pca_model.get_cluster_centers()
cluster_array = np.array(cluster_centers.iloc[:, :-1])  # Exclude the 'Cluster' column
# Printing the outputs in the desired format
print(f"CLUSTER + PCA: {cluster_array.tolist()}")
PCA = pca_model.get_cluster_centers()
# torch.save(pca_model.state_dict(), '/content/drive/MyDrive/LOTTO_MODEL/model_PCA')
# pickle.dump(pca_model, open('/content/drive/MyDrive/LOTTO_MODEL/model_PCA.pkl','wb'))