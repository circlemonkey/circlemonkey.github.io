---
layout: single
title:  "머신러닝의 모든 것 (추가중)"
categories: machinelearning
tag: [python, machinelearning]
toc: true
author_profile: false
use_math: true # mathjax-support.html을 사용할건지(라텍스 수식 사용 유무)
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## 1. 패키지 불러오기

```python
import pandas as pd
pd.set_option('display.max_columns',50) # 컬럼을 50개까지 볼 수 있도록
pd.set_option('display.max_rows',50) # row를 50개까지 볼 수 있도록
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # Warnings 제거
warnings.filterwarnings('ignore')
```

## 2. 데이터 가져오기

```python
# ▶ 데이터 불러오기
data = pd.read_excel('url')
data = pd.read_csv('url')
```
```python
# ▶ openml API를 사용하여 데이터 읽어오기
from sklearn.datasets import fetch_openml
X_orig, y = fetch_openml(data_id=43874, as_frame=True, return_X_y=True) # 당뇨 환자 데이터
```

```python
# ▶ 데이터 확인하기
data.shape()
data.info()
data.dtypes()

data.columns
data.rows

data['col'].value_counts()
data['col'].unique()

data.sample(n=5) # 무작위로 5개 데이터 보기
data.head(10)
data.tail(10)

data.describe()
data.describe().T.round(2)
```

## 3. 데이터 전처리

```python
# ▶ 데이터 변형
data = data[data['col']=='val'] # data 컬럼의 모든 값을 'val'으로 바꿈
```

```python
# ▶ 데이터 groupby
data = data.groupby('col').sum()
```

```python
# ▶ 데이터 concat
pd.concat([data1, data2])
```

```python
# ▶ 데이터 Transpose
data = data.T
```

```python
# ▶ index를 날짜형 index로 바꾸기
data.index = pd.to_datetime(data.index)
```

```python
# ▶ NaN 처리
data[data['col'].isna()].head() # Target이 nan인 데이터 탐색
data = data.dropna() # nan 데이터 삭제
```

```python
# ▶ object(dict) 타입으로 저장된 'data' column에 있는 dict 데이터를 풀어서 사용
df['col1'], df['col2'], df['col3'] = zip(*df['data'].apply(lambda x: [x['col1'], x['col2'], x['col3']]))
```

## 4. 데이터 시각화

```python
# ▶ plot 그래프
data.plot(figsize=(a,b))
```

```python
# ▶ hist 그래프 -> 모든 컬럼 별 bar 그래프를 보기
data.hist(figsize=(a,b))
```

```python
# ▶ distplot 그래프

# 1. 모든 X들에 대해서 이진분류 된 값들의 분포 보기
# -> 두 그래프가 많이 겹쳐있을 수록 해당 X에 대해서 분류가 어려울 것으로 예상
# -> 하지만 feature는 여러개가 같이 모델에 들어가므로 겹쳐있어도 다른 feature와 같이 예측되면 유의미할 수 있다.
f, axs = plt.subplots(15, 2, figsize=(20,70))

for i, feat in enumerate(X_train.T):
    sns.distplot(feat[y_train==0], ax=axs.flat[i], label='{}: {}'.format(np.unique(df['diagnosis'])[0], len(y_train[y_train==0]))) # label이 0일 때
    sns.distplot(feat[y_train==1], ax=axs.flat[i], label='{}: {}'.format(np.unique(df['diagnosis'])[1], len(y_train[y_train==1]))) # label이 1일 때
    axs.flat[i].set_title('{}:  mean: {}  std: {}'.format(list(df.iloc[:, 2:].columns)[i], abs(feat.mean().round(2)), feat.std().round(2))) # 제목에 평균, 표준편차 추가
    axs.flat[i].legend()
plt.tight_layout()
```

```python
# ▶ scatter 그래프
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data.1, date.2, color='색')
```

```python
# ▶ heatmap 그래프

# 1. 기본 heatmap
sns.set(rc={'figure.figsize':(14, 9)})
sns.heatmap(data.corr(), annot=True, linewidths=.4)
plt.show()

# 2. 반쪽에만 값있는 heatmap
plt.figure(figsize=(30, 30))

matrix = np.triu(data.corr())
sns.heatmap(data.corr(),
            annot=True, fmt='.2g',
            mask=matrix,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=256));
```

```python
# ▶ confusion_matrix 그래프
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A','B'])
disp.plot()
```

```python
# ▶ tree plot
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names = ['Dead', 'indicator'])
plt.rcParams['figure.figsize'] = [40,20]
```

```python
# ▶ randomforest에 사용되는 그래프
# 1. randomforest를 permutation_importance로 특성 중요도를 평가하여 boxplot으로 나타내기
from sklearn.inspection import permutation_importance
'''
  [permutation_importance]
  # 특성 중요도를 평가하는 방법 중 하나.
  # 특정 특성의 값을 무작위로 섞어서 (즉, 원래 데이터의 순서를 '치환'해서) 모델 성능에 어떤 영향을 미치는지 측정.
'''
'''
  # gs_rf : RandomForestClassifier 모델
  # X_test_std : X_train 데이터를 StandardScaler 처리한 것
  # y_test : y의 test split
'''
result = permutation_importance(gs_rf, X_test_std, y_test, n_repeats=10,
                                random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

X_test_df = pd.DataFrame(X_test_std, columns=list(df.iloc[:, 2:].columns))

f, ax = plt.subplots(figsize=(8,16))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_test_df.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
plt.tight_layout();
```

```python
# ▶ tree 계열 특성 중요도를 평가하여 barplot으로 나타내기
feature_map = pd.DataFrame(sorted(zip(best_model.feature_importances_, X.columns), reverse=True), columns=['Score', 'Feature'])
print(feature_map)

# Importance Score Top 10
feature_map_20 = feature_map.iloc[:10]
plt.figure(figsize=(20, 10))
sns.barplot(x="Score", y="Feature", data=feature_map_20.sort_values(by="Score", ascending=False), errwidth=40)
plt.title('Random Forest Importance Features')
plt.tight_layout()
plt.show()
```

## 5. 피처 엔지니어링

```python
# ▶ 시계열 데이터 단변량 -> 다변량으로 바꾸기
from scipy.stats import linregress # 데이터 두개를 주면 데이터 두개 간의 기울기, 절편 .. 등의 정보를 줌
def get_slope(array): # 기울기를 구하기 위한 함수
    y = np.array(array)
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope

data['slope7'] = data['sales'].rolling(7).apply(get_slope, raw=True) # 7일간 기울기 (get_slope는 아래에 정의된 함수)
data['std7'] = data['sales'].rolling(7).std(raw=True) # 7일간 표준편차
data['mean7'] = data['sales'].rolling(7).mean(raw=True) # 7일간 평균
data['skew7'] = data['sales'].rolling(7).skew() # 7일간 왜도
data['kurt7'] = data['sales'].rolling(7).kurt() # 7일간 첨도
data['min7'] = data['sales'].rolling(7).min() # 7일간 최소값
data['max7'] = data['sales'].rolling(7).max() # 7일간 최대값
```

```python
# ▶ Binary 인코딩
# 1. Boolean을 이용한 Binary 인코딩
def binary_encoding(X):
  bool_cols_l = X.select_dtypes(include=["category"]).columns.tolist()
  X[bool_cols_l] = X[bool_cols_l].astype(str).replace({"True":1, "False":0})
  return X

# 2. Map을 이용한 Binary 인코딩
class_mapping = {'M': 'malignant', 'B': 'benign'}
df['diagnosis'] = df['diagnosis'].map(class_mapping)

# 3. numpy를 이용한 Binary 인코딩
y = np.where(df['diagnosis'] == 'malignant', 1, 0)
```

```python
# ▶ OneHot 인코딩
# 1. OneHotEncoder를 이용한 OneHot 인코딩
from sklearn.preprocessing import OneHotEncoder
def onehot_encoding(X):
  cat_cols_l = X.select_dtypes(include=["object"]).columns.tolist() # object 타입의 컬럼들을 리스트로 가져옴
  '''
    ▷ categories: 각 feature에 대한 범주를 지정. 'auto'로 설정하면, 데이터에서 자동으로 범주를 결정.
    ▷ drop: 첫 번째 범주를 삭제하여 완전한 다중 공선성을 피할 수 있음. None, 'first', 또는 'if_binary' 중 하나가 될 수 있음.
    ▷ sparse: 출력을 scipy sparse matrix로 반환할지 여부를 결정. False로 설정하면 array가 반환.
    ▷ dtype: 원-핫 인코딩 후의 데이터 유형을 지정.
    ▷ handle_unknown: 학습 데이터에 없는 범주에 대해 어떻게 처리할지 결정.
      - 'error': 알 수 없는 범주에 대해 오류를 발생.
      - 'ignore': 알 수 없는 범주에 대해 무시, 해당 feature의 모든 결과 열이 0이 됨.
    ▷ transformers : tuple 리스트 (name, transformer, columns) 로 구성.
      - 'name' : 변환기 이름(임의).
      - 'transformer' : 추정기 객체 or ‘drop.
      - 'columns' : 변환기 적용될 열 이름 리스트나 인덱스.
  '''
  ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
  ohe_np = ohe.fit_transform(X[cat_cols_l].astype("category"))
  X[ohe.get_feature_names(cat_cols_l)] = ohe_np.astype(int)

  # Drop original categorical columns
  X.drop(cat_cols_l, axis=1, inplace=True)

  return X

# 2. get_dummies를 이용한 OneHot 인코딩
'''
  # "색상"이라는 특성이 "빨강", "파랑", "녹색"의 세 가지 값을 가진다면
  #  "색상_빨강", "색상_파랑", "색상_녹색". 각 더미 변수는 원래의 특성 값이 해당 색인 경우 1, 그렇지 않으면 0을 가짐
'''
# 1) 기본 인코딩 방법
data = data.join(pd.get_dummies(data['col'], prefix='col'))

# 2) 구간으로 나눠서 인코딩하기
ptile_labels = ['ptile1', 'ptile2', 'ptile3', 'ptile4', 'ptile5']
data = data.join(pd.get_dummies(pd.qcut(data['col'], q=[0, .2, .4, .6, .8, 1], labels=ptile_labels), prefix='col')) # 0 ~ 1까지 6구간으로 나누어서 인코딩

# 3) n개 중 a개만 OneHot인코딩 하고 나머지는 Other로 빼는 인코딩
from collections import defaultdict

will_be_encoded = ['column_value1', 'column_value2', 'column_value3', 'column_value4',
               'column_value5', 'column_value6', 'column_value7', 'column_value8']

dd = defaultdict(lambda: 'Other') # 딕셔너리의 해당 키가 없으면 'Other'를 반환

for _, column_value in enumerate(will_be_encoded):
    dd[column_value] = column_value

data = data.join(pd.get_dummies(data['col'].map(dd), prefix='col'))
```

```python
# ▶ StandardScaler
'''
  # 특성의 스케일을 평균이 0이고 표준편차가 1인 정규 분포에 맞게 조정.
  # z = (x - u) / s
  # x는 각 샘플의 특성 값, u는 해당 특성의 평균 값, s는 해당 특성의 표준편차

'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
# ▶ MinMaxScaler
'''
  # 모든 특성이 주어진 범위(기본적으로 0과 1 사이)에 있도록 만드는 방법
  # 최소-최대 정규화(Min-Max Normalization)
'''
from sklearn.preprocessing import MinMaxScaler

scarler = MinMaxScaler().fit(X.iloc[train_idx])
X_scal = scarler.transform(X)
```


```python
# ▶ 다중공선성 측정

# 1. VIF
# => 책에서 배운대로 생각하면 VIF가 5가 넘는 Feature는 사용할 수 없다.
# => 하지만, 대부분의 Feature가 공선성이 높은 경우에는 모든 Feature를 이용해서 1차로 모델링을 진행하고 모델링의 결과를 이용하여 Feature Selection을 하는 방법도 있다.
#   1) ---> Feature Importance를 통해 Feature Selection 시도하는 방법
from statsmodels.stats.outliers_influence import variance_inflation_factor
def compute_vif(df, considered_features):
    X = df[considered_features]
    X['intercept'] = 1

    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif
```

```python
# ▶ 실제값 y와 예측을 위한 X 만들기
X = data.drop('col').fillna(0)
y = data['col']

# Split data into train/test
from sklearn.model_selection import train_test_split # 데이터 split
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle=False)
```

