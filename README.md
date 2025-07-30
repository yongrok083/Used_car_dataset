# Used_car_dataset

## 🚗 Used Car Dataset 분석 코드 요약

### 📦 사용된 주요 패키지

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

### 🧹 데이터 전처리 요약

단계	주요 내용
- 결측치 확인	df.isnull().sum() 또는 df.isnull().mean()
- 결측치 채우기	df['col'] = df['col'].fillna(0.0)
- 결측치 제거	df.dropna(axis=1) 또는 df.dropna(subset=['col'])
- 문자열 소문자화	df['col'].str.lower()
- 브랜드 추출	df['brand'] = df['title'].str.split().str[0].str.lower()
- 병합	car_df = car.merge(brand, on='brand', how='left')
- 중복 확인	df.duplicated().sum()

### 🧬 범주형 처리 (인코딩)

변환 대상:
'Fuel type', 'Body type', 'Gearbox', 'Emission Class', 'Service history'

처리 방식: pd.get_dummies() 사용

car_df = pd.get_dummies(car_df, columns=['Fuel type', 'Body type', 'Gearbox', 'Emission Class', 'Service history'], drop_first=True)

### ⚖️ 정규화 (Scaling)
스케일링 함수: StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(car_df.select_dtypes(include='number'))

### 🧪 주성분 분석 (PCA)
목표: 누적 설명 분산 70% 이상

사용 함수: PCA(n_components=0.7)

pca = PCA(n_components=0.7)
pca_result = pca.fit_transform(scaled_data)
print(pca.n_components_)  # 선택된 주성분 수

### 📊 상관관계 분석
① 전체 상관관계 히트맵

sns.heatmap(car_df.corr(), annot=True, cmap='coolwarm')
② 상관계수 높은 변수쌍 추출

corr_matrix = car_df.select_dtypes(include='number').corr()
corr_pairs = corr_matrix.unstack()
sorted_corr = corr_pairs[corr_pairs.index[0] != corr_pairs.index[1]].drop_duplicates()
sorted_corr = sorted_corr.reindex(sorted_corr.abs().sort_values(ascending=False).index)
print(sorted_corr.head(10))

### 🌍 국가별 브랜드 개수

car_df.groupby('country')['brand'].nunique()
