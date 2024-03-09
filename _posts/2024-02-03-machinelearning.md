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

## 6. 성능 테스트

```python
# ▶ 이진 분류의 성능평가 지표 (Accuracy, confusion matrix, precision, recall, F1 score, ROC_AUC)
from sklearn import metrics
'''
  ▷ Accuracy : 예측 결과가 동일한 데이터 건수 / 전체 예측 데이터 건수
  ▷ confusion matrix : 오차행렬
    - TP : 실제 클래스가 '양성'이고, 예측도 '양성'으로 한 경우
    - FP : 실제 클래스는 '음성'인데, 예측을 '양성'으로 잘못한 경우
    - TN : 실제 클래스가 '음성'이고, 예측도 '음성'
    - FN : 실제 클래스는 '양성'인데, 예측을 '음성'으로 잘못한 경우
  ▷ precision : 정밀도, TP/(FP+TP), 양성 클래스로 예측된 결과 중 실제로 양성 클래스인 비율
  ▷ recall : 재현율, TP/(FN+TP), 실제 양성 클래스 중 얼마나 많은 비율이 양성 클래스로 올바르게 예측되었는지
  ▷ F1 score : 정밀도(Precision)와 재현율(Recall)의 조화 평균, 정밀도와 재현율이 모두 높을 때만 높아짐
  ▷ ROC_AUC : 이진 분류 모델의 성능을 평가하는 도구
    - ROC : 곡선, y축에 True Positive Rate(TPR), x축에 False Positive Rate(FPR)
    - AUC : ROC 곡선 아래의 면적, 클수록 분류 모델의 성능이 좋음(max = 1)
    - TPR : Recall(재현율), TP / (TP + FN)
    - FPR : FP / (FP + TN)
    - TNR : TN / (TN + FP)
    - FNR : 1-재현율, FN / (FN + TP)
  ▷ MCC : 매튜상관계수(Matthews correlatio coefficient)
'''
def evaluate_class_mdl(fitted_model, X_train, X_test, y_train, y_test, plot=True, pct=True, thresh=0.5):
    y_train_pred = fitted_model.predict(X_train).squeeze() # squeeze() 결과를 1차원으로 만들어줌
    if len(np.unique(y_train_pred)) > 2:
        y_train_pred = np.where(y_train_pred > thresh, 1, 0)
        y_test_prob = fitted_model.predict(X_test).squeeze()
        y_test_pred = np.where(y_test_prob > thresh, 1, 0)
    else:
        y_test_prob = fitted_model.predict_proba(X_test)[:,1]
        y_test_pred = np.where(y_test_prob > thresh, 1, 0)

    acc_tr = metrics.accuracy_score(y_train, y_train_pred) # train Accuracy
    acc_te = metrics.accuracy_score(y_test, y_test_pred) # test Accuracy
    cf_matrix = metrics.confusion_matrix(y_test, y_test_pred) # confusion matrix
    tn, fp, fn, tp = cf_matrix.ravel() # confusion matrix - TN, TP, FN, FP
    fnr = (fn/(tp+fn)) * 100
    fpr = (fp/(tn+fp)) * 100
    pre_te = metrics.precision_score(y_test, y_test_pred) # precision
    rec_te = metrics.recall_score(y_test, y_test_pred) # recall
    f1_te = metrics.f1_score(y_test, y_test_pred) # F1 score
    roc_auc_te = metrics.roc_auc_score(y_test, y_test_prob) # ROC_AUC
    mcc_te = metrics.matthews_corrcoef(y_test, y_test_pred) # MCC

    # 시각화 하기
    if plot:
        print(f"Accuracy_train: {acc_tr:.4f} \t\tAccuracy_test:   {acc_te:.4f}")
        print(f"Precision_test: {pre_te:.4f} \t\tRecall_test:     {rec_te:.4f}")
        print(f"ROC-AUC_test:   {roc_auc_te:.4f} \t\tF1_test:     {f1_te:.4f} \t\tMCC_test: {mcc_te:.4f}")
        print(f"fnr:            {fnr:.4f}    \t\tfpr:              {fpr:.4f}")
        plt.figure(figsize=(6, 5))

        if pct:
            ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,\
                        fmt='.2%', cmap='Blues', annot_kws={'size':16})
        else:
            ax = sns.heatmap(cf_matrix, annot=True,\
                        fmt='d',cmap='Blues', annot_kws={'size':16})
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Observed', fontsize=12)
        plt.show()
        return y_train_pred, y_test_prob, y_test_pred

    # 시각화 안하고 값만 출력해보기
    else:
        t = cf_matrix.sum()
        metrics_dict = {'Accuracy_train':acc_tr , 'Accuracy_test':acc_te, 'Precision_test':pre_te, 'Recall_test':rec_te,\
                        'ROC-AUC_test':roc_auc_te,  'F1_test':f1_te, 'MCC_test': mcc_te, 'fnr':fnr, 'fpr':fpr, 'tn%':tn/t, 'fp%':fp/t, 'fn%':fn/t, 'tp%':tp/t }
        return metrics_dict
```


```python
# ▶ 교차 검증
# 1. k-fold
# => 단계별로 처리하기 때문에 상황에 따른 세밀한 조정이 가능하지만 코드가 좀 더 복잡해질 수 있음
'''
  # 전체 데이터셋을 k개의 동일한 크기를 가진 부분집합(fold)으로 나누는 것으로 시작
  # 한 개의 부분집합을 테스트 데이터로 설정하고, 나머지 k-1개의 부분집합들을 합쳐서 훈련 데이터로 사용
  # 선택된 훈련 데이터에서 모델을 학습시키고, 선택된 테스트 데이터로 모델의 성능을 평가
  # 모든 데이터가 최소 한 번은 테스트셋으로 사용되므로 과적합(overfitting) 문제를 줄이는 데 도움.
  # 모든 데이터가 최소 한 번은 학습에 사용되므로 보다 안정적인 성능 추정이 가능
'''
from sklearn.model_selection import KFold
'''
  n_splits : int, default=5, 데이터를 나눌 fold의 개수를 결정.
  shuffle : bool, default=False, 데이터를 분할하기 전에 데이터를 섞을지 여부를 결정.
'''
def k_fold(model, X, y):
  kfold = KFold(n_splits=5, shuffle=True, random_state=123)
  for k, (train, test) in enumerate(kfold.split(X, y)):
    res = model.fit(y[train], X.loc[train:]) # fit
    pred = res.predict(X.loc[test:]) # predict

# 2. Cross-Validation
# => cross_val_score 함수는 교차 검증 과정 전체를 한 번에 처리하므로 간단한 경우에 유용
# => 고급 설정이나 복잡한 경우에는 유연성이 부족
'''
  # 머신러닝 모델의 일반화 성능을 평가하기 위한 통계적 방법.
  # 데이터를 여러 부분(폴드)으로 나누고, 이 중 일부를 훈련 데이터로, 나머지를 테스트 데이터로 사용하여 모델을 평가하는 과정을 반복.
  # k-겹 교차 검증(cross-validation)을 수행하여 모델의 성능을 평가.
  # scores = cross_val_score(model, X, y, cv=5)
  # => model은 평가하려는 Scikit-learn 추정기(estimator), X는 독립 변수 데이터, y는 종속 변수 데이터이며, cv는 교차 검증에서 생성할 폴드(fold) 수를 지정
  # 함수가 반환하는 scores는 각 폴드에 대한 성능 점수(score)들의 배열
'''
from sklearn.model_selection import cross_val_score

def get_cross_val(clf, X, y, model_name, cv_num=5, metric='f1'):
    scores = cross_val_score(clf, X, y, cv=cv_num, scoring=metric)
    mean = scores.mean()
    std  = scores.std()
    p025 = np.quantile(scores, 0.025)
    p975 = np.quantile(scores, 0.975)
    metrics = ['mean', 'standard deviation', 'p025', 'p975']
    s = pd.Series([mean, std, p025, p975], index=metrics) # np.where(lb < 0, 0, lb), np.where(ub > 1, 1, ub)
    s.name = model_name
    return s
```


```python
# ▶ β(계수) 추정법 검증
'''
  [가설 검정(hypothesis testing)]
  # 귀무가설 (Null Hypothesis, H0): β = 0 (해당 독립변수는 종속변수에 영향을 주지 않음)
  # 대립가설 (Alternative Hypothesis, Ha): β ≠ 0 (해당 독립변수는 종속변수에 영향을 줌)
'''

# standard error
def coef_se(clf, X, y):
    n = X.shape[0]
    X1 = np.hstack((np.ones((n, 1)), np.matrix(X)))
    se_matrix = scipy.linalg.sqrtm(
        metrics.mean_squared_error(y, clf.predict(X)) *
        np.linalg.inv(X1.T * X1)
    )
    return np.diagonal(se_matrix)

# t-통계량
# => t-통계량이 크면 클수록 H0 (귀무가설)을 기각할 확률이 높아짐.
def coef_tval(clf, X, y):
    a = np.array(clf.intercept_ / coef_se(clf, X, y)[0])
    b = np.array(clf.coef_ / coef_se(clf, X, y)[1:])
    return np.append(a, b)

# p-value
# => p-value가 0.05 이하면  H0 (귀무가설)은 기각 되며  H1 (대립가설)이 채택.
def coef_pval(clf, X, y):
    n = X.shape[0]
    t = coef_tval(clf, X, y)
    p = 2 * (1 - scipy.stats.t.cdf(abs(t), n - 1))
    return p
```


```python
# ▶ R2_score
from sklearn.metrics import r2_score
'''
  # 회귀 모델의 성능을 평가하는 지표 중 하나.
  # 0에서 1 사이의 값을 가짐.
  # RSS(Residual Sum of Squares) = 실제 값(yi)와 예측 값(f(xi)) 사이의 차이(잔차)를 제곱하여 합함.
  # TSS(Total Sum of Squares) = 실제 값(yi)와 전체 데이터의 평균 값(y_mean) 사이의 차이를 제곱하여 합함.
  # R² = 1 - (RSS/TSS)
  # 1에 가까울 수록 모델은 데이터를 잘 설명.
  # 모델의 성능을 완전히 판단하기 어려움. 과적합된 모델도 높은 R²값을 가질 수 있음.
'''
def r2_score(label, pred):
  score = r2_score(label, y_pred)
  return score

def adj_r2_score(clf, X, y):
    n = X.shape[0]  # Number of observations
    p = X.shape[1]  # Number of features
    r_squared = metrics.r2_score(y, clf.predict(X))
    return 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))
```


```python
# ▶ SSE(Standard Squared Error)
def sse(clf, X, y):
'''
  # 예측값과 실제값의 차이(오차)를 제곱한 값들의 합
  # SE 값이 작다는 것은 모델이 데이터를 잘 적합하고 있다는 것을 나타내며, 반대로 SSE 값이 크다면 모델이 데이터를 잘 적합하지 못하고 있다는 것을 나타냄.
  # SSE = Σ(y_i - f(x_i))^2
'''
    y_hat = clf.predict(X)
    sse = np.sum((y_hat - y) ** 2)
    return sse / X.shape[0]
```


```python
# ▶ MSE(Mean Squared Error)
from sklearn.metrics import mean_squared_error
'''
  # 실제 값과 예측 값의 차이를 제곱한 값들의 평균.
  # 이 지표는 오차의 제곱을 계산하기 때문에, 실제 값과 예측 값이 크게 다를수록 그 차이가 더욱 커짐.
  # MSE = 1/n * Σ(yi - ŷi)^2
'''
def mean_squared_error(label, pred):
  mse = mean_squared_error(label, pred)
  return score
```


```python
# ▶ MAE(Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
'''
  # 실제 값과 예측 값의 절대적인 차이들의 평균
  # 오차를 제곱하지 않음. 따라서 큰 오차값에 대해 상대적으로 덜 민감
  # MAE = 1/n * Σ|yi - ŷi|
'''
def mean_absolute_error(label, pred):
  mae = mean_absolute_error(label, pred)
  return score
```


```python
# ▶ MAPE(Mean Absolute Percentage Error - 평균 절대 비율 오차)
def evaluate_mape(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    return mape
```

## 7. 모델링



[최소자승법(Ordinary Least Squares, OLS)](url)



```python
# -----▶ 최소자승법(Ordinary Least Squares, OLS) ◀-----
'''
  # 회귀 분석에서 가장 일반적으로 사용되는 방법.
  # 실제 값과 모델이 예측한 값 사이의 잔차(residuals)의 제곱합을 최소화하는 파라미터를 찾는 것을 목표.
  # y = βX + ε --> β는 우리가 추정하려는 파라미터들, ε는 오차항.
  # OLS의 목표는 ∑ε² = ∑(y - βX)² 를 최소화하는 β 값을 찾는 것.
  # 데이터가 특정 가정 (독립성, 등분산성, 정상성 등) 을 만족하지 않으면 잘못된 추정치나 과적합을 초래할 수 있음.
'''
import statsmodels.api as sm # statsmodels 사용, R 기반임

y = data['col_y'] # y
X = data[['col_x1', 'col_x2', 'col_x3']] # X
X = sm.add_constant(X) # 1

model1 = sm.OLS(y,X)
res1 = model1.fit()
print(res1.summary())

features = res1.params.index
coefs = [round(val, 4) for val in res1.params.values]
dict(zip(features, coefs)) # 잔차(residuals, ε)의 제곱합을 최소화하는 파라미터 β

# R-squared
print("R-squared: %.3f" % round(res1.rsquared, 3))

# RMSE
print("RMSE: %.3f" % round(np.mean((y - res1.fittedvalues)**2)**0.5, 3))
```

[Linear Regression](url)



```python
# -----▶ Linear Regression ◀-----
from sklearn.linear_model import LinearRegression

y = data['col_y'] # y
X = data[['col_x1', 'col_x2', 'col_x3']] # X

# Linear Regression 모델 생성
lm = LinearRegression().fit(X, y)

# 예측
pred = lm.predict(X_test)
```

[Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)



```python
# -----▶ Ridge Regression ◀-----
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
'''
  # alpha : L2-norm Penalty Term
    - alpha : 0 일 때, Just Linear Regression
  # fit_intercept : Centering to zero
    - 베타0를 0로 보내는 것 (베타0는 상수이기 때문에)
  # max_iter : Maximum number of interation
    - Loss Function의 Ridge Penalty Term은 Closed Form 값이기는 하지만 값을 찾아 나감
    - Penalty Term : (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_2
'''

# 1. Ridge
penelty = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1, 10]

for a in penelty:
    model = Ridge(alpha=a).fit(X_train, y_train)
    score = model.score(X_val, y_val)
    pred_y = model.predict(X_val)
    mse = mean_squared_error(y_val, pred_y)
    print("Alpha:{0:.5f}, R2:{1:.7f}, MSE:{2:.7f}, RMSE:{3:.7f}".format(a, score, mse, np.sqrt(mse)))

### --------------------------------------------------------------- ###
# 2. RidgeCV
ridge_cv = RidgeCV(alphas=penelty, cv=5)
model = ridge_cv.fit(X_train, y_train)
print("Best Alpha:{0:.5f}, R2:{1:.4f}".format(model.alpha_, model.best_score_))

model_best = Ridge(alpha=model.alpha_).fit(X_train, y_train)
score = model_best.score(X_val, y_val)
pred_y = model_best.predict(X_val)
mse = np.sqrt(mean_squared_error(y_val, pred_y))
print("Alpha:{0:.5f}, R2:{1:.7f}, MSE:{2:.7f}, RMSE:{3:.7f}".format(0.01, score, mse, np.sqrt(mse)))
```

[LASSO Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)



```python
# -----▶ LASSO Regression ◀-----
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
'''
  # alpha : L1-norm Penalty Term
    - alpha : 0 일 때, Just Linear Regression
  # fit_intercept : Centering to zero
    - 베타0를 0로 보내는 것 (베타0는 상수이기 때문에)
  # max_iter : Maximum number of interation
    - Loss Function의 LASSO Penalty Term은 절대 값이기 때문에 Gradient Descent와 같은 최적화가 필요함
    - Penalty Term : ||y - Xw||^2_2 + alpha * ||w||_1
'''

# 1. LASSO Regression
penelty = [0.0000001, 0.0000005, 0.000001, 0.000005,0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 100000]

for a in penelty:
    model = Lasso(alpha=a).fit(X_train, y_train)
    score = model.score(X_val, y_val)
    pred_y = model.predict(X_val)
    mse = mean_squared_error(y_val, pred_y)
    print("Alpha:{0:.7f}, R2:{1:.7f}, MSE:{2:.7f}, RMSE:{3:.7f}".format(a, score, mse, np.sqrt(mse))) # select alpha by checking R2, MSE, RMSE

### --------------------------------------------------------------- ###
# 2. Cross Validation for LASSO
lasso_cv=LassoCV(alphas=penelty, cv=5)
model = lasso_cv.fit(X_train, Y_train)
print("Best Alpha : {:.7f}".format(model.alpha_))

model_best = Lasso(alpha=model.alpha_).fit(X_train, Y_train)
score = model_best.score(X_val, y_val)
pred_y = model_best.predict(X_val)
mse = mean_squared_error(y_val, pred_y)
print("Alpha:{0:.7f}, R2:{1:.7f}, MSE:{2:.7f}, RMSE:{3:.7f}".format(model.alpha_, score, mse, np.sqrt(mse)))
```

[ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)



```python
# -----▶ ElasticNet ◀-----
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
'''
  # ElasticNet은 Ridge의 L_2-norm과 Lasso의 L_1-norm을 섞어 놓았음 (두 개의 장점 사용 가능)
  # λ_1 : Lasso Penalty Term (Feature Selection)
  # λ_2 : Ridge Penalty Term (다중공선성 방지)
  # ElasticNet은 Correlation이 강한 변수를 동시에 선택/배제하는 특성을 가지고 있음
'''
'''
  # alpha : L2-norm Penalty Term
    - alpha : 0 일 때, Just Linear Regression
  # l1_ratio : L1-norm Penalty Term
    - 0 <= l1_ratio <= 1
    - l1_ratio : 1 일 때, Just Ridge Regression
  # fit_intercept : Centering to zero
    - 베타0를 0로 보내는 것 (베타0는 상수이기 때문에)
  # max_iter : Maximum number of interation
    - Loss Function의 LASSO Penalty Term은 절대 값이기 때문에 Gradient Descent와 같은 최적화가 필요함
  # Penalty Term
    - 1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
'''

# 1. ElasticNet Regression
alphas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.001, 0.005, 0.01, 0.05]
l1_ratio = [0.9, 0.7, 0.5, 0.3, 0.1]

for a in alphas:
    for b in l1_ratio:
        model = ElasticNet(alpha=a, l1_ratio=b).fit(X_train, y_train])
        score = model.score(X_val, y_val)
        pred_y = model.predict(X_val)
        mse = mean_squared_error(y_val, pred_y)
        print("Alpha:{0:.7f}, l1_ratio: {1:.7f}, R2:{2:.7f}, MSE:{3:.7f}, RMSE:{4:.7f}".format(a, b, score, mse, np.sqrt(mse))) # select alpha and beta by checking R2, MSE, RMSE

### --------------------------------------------------------------- ###
# 2. Cross Validation for ElasticNet
grid = dict()
grid['alpha'] = alphas
grid['l1_ratio'] = l1_ratio

model = ElasticNet()
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
results = search.fit(X_val, y_val)
print('RMSE: {:.4f}'.format(-results.best_score_))
print('Config: {}'.format(results.best_params_))

model_best = ElasticNet(alpha=results.best_params_['alpha'],
                        l1_ratio=results.best_params_['l1_ratio']).fit(X_train, y_train)
score = model_best.score(X_val, y_val)
pred_y = model_best.predict(X_val)
mse = mean_squared_error(y_val, pred_y)
print("Alpha:{0:.7f}, l1_ratio: {1:.7f}, R2:{2:.7f}, MSE:{3:.7f}, RMSE:{4:.7f}".format(results.best_params_['alpha'],
                                                                                   results.best_params_['l1_ratio'],
                                                                                   score, mse, np.sqrt(mse)))
```

[Logistic Regression](url)



```python
# -----▶ Logistic Regression ◀-----
from sklearn.linear_model import LogisticRegression

# 1. Logistic Regression 기본
model = LogisticRegression()
model.fit(X_train,training_y) # 학습
prediction = model.predict(X_test) # 예측

### --------------------------------------------------------------- ###
# 2. Logistic Regression Cross-Validation 교차 검증
from sklearn.linear_model import LogisticRegressionCV
'''
  # 로지스틱 회귀를 수행하면서 동시에 교차 검증(Cross Validation)을 사용하여 최적의 정규화 파라미터를 찾음.
  # cv: 교차 검증을 수행하는 데 사용되는 폴드의 개수를 지정. 예를 들어, cv=5는 5-겹 교차 검증을 의미.
  # penalty: 사용할 정규화 유형을 지정. 'l1', 'l2', 'elasticnet', 'none' 중 하나를 선택. L1과 L2 정규화는 각각 Lasso와 Ridge 회귀에 해당하며, 'elasticnet'은 이 둘의 조합.
  # solver: 최적화 문제를 해결하는 알고리즘을 지정. 가능한 값은 ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, 'saga'.
    - liblinear : 이 솔버는 작은 데이터셋에 적합하며, L1 정규화, L2 정규화 또는 둘 다 없이 사용할 수 있음.
    ('multinomial' 다중 클래스 문제를 해결할 수 없음)
    - newton-cg : 이 알고리즘은 '뉴턴 메서드(Newton Method)'의 변형이며, 비선형 최적화 문제를 해결하는 데 사용.
    (L2 정규화나 없음만 지원)
    - sag: Stochastic Average Gradient descent (SAG) 알고리즘이며 대용량 데이터에 적합.
    (큰 데이터셋에 적합,  'multinomial' loss를 지원, L2 정규화나 없음만 지원)
    - saga: SAGA(Scaled Average Gradient descent) 알고리즘은 SAG의 변형으로 대용량 데이터에 적합.
    (큰 데이터셋에 적합,  'multinomial' loss를 지원,  L1 정규화, L2 정규화, Elasticnet 혹은 없음을 모두 지원)
  # Cs: Inverse of regularization strength 값을 설정하는데 사용되는 리스트입니다 (C = 1/λ). Cs가 int일 경우, 그만큼의 갯수로 np.logspace(-4, 4) 범위에서 동등하게 나누어진 리스트가 생성.
  # l1_ratios: Elastic-net mixing parameter(0<= l1_ratio <= 1). l1_ratio=0 이면 penalty='l2'와 같아지며 l1_ratio=1 이면 penalty='l1'과 같아짐.
  # max_iter: solver가 수렴하기 위해 필요한 최대 반복 횟수입니다.
'''

lr_clf = LogisticRegressionCV(cv=5,
                              penalty='elasticnet', solver='saga',
                              Cs=np.power(10, np.arange(-3, 1, dtype=float)), # [0.001, 0.01 , 0.1  , 1.   ]
                              l1_ratios=np.linspace(0, 1, num=6, dtype=float), # [0. , 0.2, 0.4, 0.6, 0.8, 1. ] : 0~1범위를 6등분
                              max_iter=1000,
                              random_state=0)

start = time() # 시작 시간
lr_clf.fit(X_train_std, y_train) # 학습
lr_duration = time() - start # 학습에 소요되는 시간
prediction = lr_clf.predict(X_test) # 예측

print("LogisticRegressionCV took {:.2f} seconds for {} cv iterations with {} parameter settings.".format(lr_duration,
                                                                                                         lr_clf.n_iter_.shape[1],
                                                                                                         lr_clf.n_iter_.shape[2] * lr_clf.n_iter_.shape[3]))
print('Optimal regularization strength: {}  Optimal L1 Ratio: {}'.format(lr_clf.C_[0], lr_clf.l1_ratio_[0]))
print('Accuracy (train): {:.2f}'.format(lr_clf.score(X_train_std, y_train)))
print('Accuracy  (test): {:.2f}'.format(lr_clf.score(X_test_std,  y_test)))

```

[Support Vector Machine](url)



```python
# -----▶ Support Vector Machine ◀-----
from sklearn.svm import SVR

# 1. Support Vector Machine 기본
model = SVR()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

### --------------------------------------------------------------- ###
# 2. Support Vector Machine Cross-Validation 교차 검증
from sklearn.model_selection import GridSearchCV # 하이퍼파라미터 튜닝을 자동화하는 역할, 하이퍼파라미터 값들의 모든 가능한 조합에 대해 교차 검증을 수행
from sklearn.svm import SVC
'''
  # 지도 학습 알고리즘 중 하나로, 주로 분류와 회귀 문제에 사용.
  # 선형 또는 비선형 분류, 회귀, 이상치 탐지 등에 사용.
  # 데이터를 고차원 공간으로 매핑한 후, 서로 다른 클래스 간의 최대 마진을 가지는 초평면(hyperplane)을 찾는 것.
  # 선형 분리가 불가능한 경우를 위해 커널 트릭(kernel trick)이라는 방법론을 사용.
  # C : 오차 항목에 대한 패널티
    - 값이 크면, 모델은 훈련 데이터에 가능한 한 가깝게 맞추려고 하며, 이로 인해 복잡한 모델이 될 수 있음.
    - 값이 작으면, 모델은 잘못 분류된 데이터 포인트를 더 허용하게 되어, 좀 더 단순한 모델을 만들어냄.
  # kernel : 사용할 커널 함수, 원본 데이터를 고차원 공간으로 변환하는 방법을 결정.
    - 'linear': 선형 커널은 원래의 특성 공간에서 선형 분리가 가능할 때 사용.
    - 'poly': 다항식 커널은 원본 특성의 다항식 조합을 새로운 특성으로 추가.
    - 'rbf': RBF(Radial basis function) 또는 가우시안(Gaussian) 커널은 각 데이터 포인트를 중심으로 하는 방사 기저 함수를 적용하여 비선형 매핑을 수행.
  # gamma : 커널의 영향력 범위
  # degree : 다항식 커널에서 사용되는 차수
'''
param_grid = {'C': np.power(10, np.arange(0, 3, dtype=float)),
              'kernel': ['linear', 'sigmoid', 'rbf'],
              'gamma': ['auto', 'scale']}

svc_clf = SVC(random_state=0)
gs_svc = GridSearchCV(svc_clf, param_grid=param_grid)

start = time() # 시작 시간
gs_svc.fit(X_train_std, y_train) # 모델 학습
svc_duration = time() - start # 모델 학습 경과 시간

print("GridSearchCV of SVC took {:.2f} seconds for {} candidate parameter settings.".format(svc_duration,
                                                                                            len(gs_svc.cv_results_['params'])))
# report(gs_svc.cv_results_)
print('Optimal C: {}  Optimal kernel: {}  Optimal gamma: {}'.format(gs_svc.best_params_['C'], gs_svc.best_params_['kernel'], gs_svc.best_params_['gamma']))
print('Accuracy (train): {:.2f}'.format(gs_svc.score(X_train_std, y_train)))
print('Accuracy  (test): {:.2f}'.format(gs_svc.score(X_test_std,  y_test)))
```

[Decision Tree](url)



```python
# -----▶ Decision Tree ◀-----
# Tree 계열은 Scaling이 필요한가? ➡ 필요없음!

# 1. DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

### --------------------------------------------------------------- ###
# 2. DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz # DecisionTree
'''
  # Max_Depth는 5 초과를 넘지 않아야함, 5를 초과하게 되면 Rule Extraction Plotting의 가독성이 매우 떨어짐.
  # 정확도와 설명력은 Trade-off가 존재하기 때문에 자기만의 기준으로 적절한 선을 선택하면 됨.
  # Rule Extraction 할때 GINI INDEX 뿐만 아니라 Sample 개수도 중요한 척도가 됨.
    - GINI INDEX가 아주 낮지만(불순도가 낮음, 좋음) Sample의 개수가 너무 적으면 의미가 없음(Overfitting이라고 생각됨).
'''

for i in range(2,11,1):
    print(">>>> Depth {}".format(i))

    model = DecisionTreeClassifier(max_depth=i, criterion='gini')
    model.fit(X_train, y_train)

    # Train Acc
    y_pre_train = model.predict(X_train)
    cm_train = confusion_matrix(y_train, y_pre_train)
    print("Train Confusion Matrix")
    print(cm_train)
    print("Train Acc : {}".format((cm_train[0,0] + cm_train[1,1])/cm_train.sum()))
    print("Train F1-Score : {}".format(f1_score(Y.iloc[train_idx], y_pre_train)))

    # Test Acc
    y_pre_test = model.predict(X_val)
    cm_test = confusion_matrix(y_val, y_pre_test)
    print("Test Confusion Matrix")
    print(cm_test)
    print("TesT Acc : {}".format((cm_test[0,0] + cm_test[1,1])/cm_test.sum()))
    print("Test F1-Score : {}".format(f1_score(y_val, y_pre_test)))
    print("-----------------------------------------------------------------------")
```

[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)



```python
# -----▶ Random Forest ◀-----
# 1. RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor # Random Forest
'''
  ▷ random_state : Random Forest에서 사용되는 부트스트랩 샘풀들과 feature들이 랜덤하게 선택되기 때문에, 이 파라미터로 난수 생성기의 seed 값을 지정.
  ▷ n_estimators: 생성할 결정 트리의 개수. 더 많은 트리는 모델 성능을 향상. 계산 시간이 증가.
  ▷ max_features: 각 분할에서 고려해야 하는 feature(특성)의 최대 개수.
'''
rf = RandomForestRegressor(random_state = 0) # 모델(1)
rf = RandomForestRegressor(random_state = 0, n_estimators=200, max_features=4) # 모델(2)

# 학습, 예측
rf.fit(학습features, 학습labels) # fit
predicted = rf.predict(테스트features) # predict

### --------------------------------------------------------------- ###
# 2. RandomForestClassifier
from sklearn.model_selection import GridSearchCV # 하이퍼파라미터 튜닝을 자동화하는 역할, 하이퍼파라미터 값들의 모든 가능한 조합에 대해 교차 검증을 수행
from sklearn.ensemble import RandomForestClassifier
'''
  # n_estimators: 생성할 결정 트리의 개수를 지정.
    - 이 값이 클수록 더 많은 결정 트리를 학습하므로 예측 성능이 일반적으로 좋아짐.
    - 너무 큰 값은 과적합을 유발하거나, 계산 비용과 시간을 증가시킬 수 있음.
  # max_features: 각 결정 트리에서 분할에 사용할 특성의 최대 개수를 지정(auto, sqrt, log2).
    - 이 값을 작게 설정하면 랜덤성이 증가하여 모델의 다양성이 증가.
    - 너무 작으면 각 트리가 너무 단순해져서 성능이 저하될 수 있음.
  # criterion: 분할 품질을 측정하는 기능.
    - 'gini'는 지니 불순도(Gini impurity).
    - 'entropy'는 정보 이득(Information gain)을 사용하여 분할.
  # max_depth: 각 결정 트리의 최대 깊이를 지정. 깊이가 깊어질수록 모델은 복잡해지며 과적합될 가능성이 커짐.
  # min_samples_split : 2개로 Split 하는게 아니라 N개로 Split 가능
  # bootstrap : Bagging 중 Boostrap 기법
  # oob_score : out-of-bag Score
  # class_weight : Label Imbalance 데이터 학습시 weight를 주는 것
    - {0: 1, 1: 1}
'''

# 1) RandomForestClassifier 방법 1
param_grid = {'n_estimators': np.arange(100, 1000, 200, dtype=int),
              'max_features': [None, 'sqrt', 'log2'],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 3, 5, 7]}

rf_clf = RandomForestClassifier(oob_score=True, random_state=0)
gs_rf = GridSearchCV(rf_clf, param_grid=param_grid)

start = time() # 시작 시간
gs_rf.fit(X_train_std, y_train) # 모델 학습
rf_duration = time() - start # 모델 학습 경과 시간

print("GridSearchCV of RF took {:.2f} seconds for {} candidate parameter settings.".format(rf_duration,
                                                                                           len(gs_rf.cv_results_['params'])))
# report(gs_rf.cv_results_)
print('Optimal n_estimators: {}  Optimal max_features: {}  Optimal max_depth: {}  Optimal criterion: {}'.format(gs_rf.best_params_['n_estimators'],
                                                                                                                gs_rf.best_params_['max_features'],
                                                                                                                gs_rf.best_params_['max_depth'],
                                                                                                                gs_rf.best_params_['criterion']))
print('Accuracy (train): {:.2f}'.format(gs_rf.score(X_train_std, y_train)))
print('Accuracy  (test): {:.2f}'.format(gs_rf.score(X_test_std,  y_test)))

### --------------------------------------------------------------- ###
# 2) RandomForestClassifier 방법 2
estimators = [10, 30, 40, 50, 60]
depth = [4 , 5, 10, 15]
save_est = []
save_dep = []
f1_score_ = []

for est in estimators:
    for dep in depth:
        print("Number of Estimators : {}, Max Depth : {}".format(est, dep))
        model = RandomForestClassifier(n_estimators=est, max_depth=dep, random_state=119,
                                       criterion='gini', max_features='auto',
                                       bootstrap=True, oob_score=False) # if you use "oob_score=True", get long time for training
        model.fit(X_train, y_train)

        # Train Acc
        y_pre_train = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_pre_train)
        print("Train Confusion Matrix")
        print(cm_train)
        print("Train Acc : {}".format((cm_train[0,0] + cm_train[1,1])/cm_train.sum()))
        print("Train F1-Score : {}".format(f1_score(y_train, y_pre_train)))

        # Test Acc
        y_pre_test = model.predict(X_val)
        cm_test = confusion_matrix(y_val, y_pre_test)
        print("Test Confusion Matrix")
        print(cm_test)
        print("TesT Acc : {}".format((cm_test[0,0] + cm_test[1,1])/cm_test.sum()))
        print("Test F1-Score : {}".format(f1_score(y_val, y_pre_test)))
        print("-----------------------------------------------------------------------")
        save_est.append(est)
        save_dep.append(dep)
        f1_score_.append(f1_score(Y.iloc[valid_idx], y_pre_test))

best_model = RandomForestClassifier(n_estimators=save_est[np.argmax(f1_score_)], max_depth=save_dep[np.argmax(f1_score_)], random_state=119,
                               criterion='gini', max_features='auto',
                               bootstrap=True, oob_score=False) # if you use "oob_score=True", get long time for training
best_model.fit(X_train, y_train)
```

[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)



```python
# -----▶ AdaBoost ◀-----
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
'''
  # n_estimators : # of Tree
  # learning_rate : learning_rate과 n_estimator와 Trade-off 관계가 있음
    - Weight applied to each classifier at each boosting iteration
'''
estimators = [70, 90, 100]
learning = [0.01, 0.03, 0.05, 0.1, 0.5]
save_est = []
save_lr = []
f1_score_ = []

for est in estimators:
    for lr in learning:
        print("Number of Estimators : {}, Learning Rate : {}".format(est, lr))

        model = AdaBoostClassifier(n_estimators=est, learning_rate=lr, random_state=119)
        model.fit(X_train, y_train)

        # Train Acc
        y_pre_train = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_pre_train)
        print("Train Confusion Matrix")
        print(cm_train)
        print("Train Acc : {}".format((cm_train[0,0] + cm_train[1,1])/cm_train.sum()))
        print("Train F1-Score : {}".format(f1_score(y_train, y_pre_train)))

        # Test Acc
        y_pre_test = model.predict(X_val)
        cm_test = confusion_matrix(y_val, y_pre_test)
        print("Test Confusion Matrix")
        print(cm_test)
        print("TesT Acc : {}".format((cm_test[0,0] + cm_test[1,1])/cm_test.sum()))
        print("Test F1-Score : {}".format(f1_score(y_val, y_pre_test)))
        print("-----------------------------------------------------------------------")
        save_est.append(est)
        save_lr.append(lr)
        f1_score_.append(f1_score(y_val, y_pre_test))

best_model = AdaBoostClassifier(n_estimators=save_est[np.argmax(f1_score_)], learning_rate=save_lr[np.argmax(f1_score_)], random_state=119)
best_model.fit(X_train, y_train)
```

[Gradient Boosting Machine](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)



```python
# -----▶ GBM ◀-----
from sklearn.ensemble import GradientBoostingClassifier # GBM
'''
  # n_estimators : # of Tree
  # learning_rate : learning_rate과 n_estimator와 Trade-off 관계가 있음
    - Weight applied to each classifier at each boosting iteration
    - 0~1사이의 값을 가질 수 있으며, 1로 설정할 경우 Overfitting 발생
  # max_features : Feature 수 sampling (Overfitting 방지)
  # subsample : Data Subsample (Overfitting 방지, Bootstrap X)
  # max_depth : Tree의 최대 깊이 제한
'''
estimators = [10, 20, 50]
learning = [0.05, 0.1, 0.5]
subsam = [0.5, 0.75, 1]
save_est = []
save_lr = []
save_sub = []
f1_score_ = []

for est in estimators:
    for lr in learning:
        for sub in subsam:
            print("Number of Estimators : {}, Learning Rate : {}, Subsample : {}".format(est, lr, sub))
            model = GradientBoostingClassifier(n_estimators=est,
                                               learning_rate=lr,
                                               subsample=sub,
                                               random_state=119)
            model.fit(X_train, y_train)

            # Train Acc
            y_pre_train = model.predict(X_train)
            cm_train = confusion_matrix(y_train, y_pre_train)
            print("Train Confusion Matrix")
            print(cm_train)
            print("Train Acc : {}".format((cm_train[0,0] + cm_train[1,1])/cm_train.sum()))
            print("Train F1-Score : {}".format(f1_score(y_train, y_pre_train)))

            # Test Acc
            y_pre_test = model.predict(X_val)
            cm_test = confusion_matrix(y_val, y_pre_test)
            print("Test Confusion Matrix")
            print(cm_test)
            print("TesT Acc : {}".format((cm_test[0,0] + cm_test[1,1])/cm_test.sum()))
            print("Test F1-Score : {}".format(f1_score(y_val, y_pre_test)))
            print("-----------------------------------------------------------------------")
            save_est.append(est)
            save_lr.append(lr)
            save_sub.append(sub)
            f1_score_.append(f1_score(y_val, y_pre_test))

best_model = GradientBoostingClassifier(n_estimators=save_est[np.argmax(f1_score_)],
                                        learning_rate=save_lr[np.argmax(f1_score_)],
                                        subsample = save_sub[np.argmax(f1_score_)],
                                        random_state=119)
best_model.fit(X_train, y_train)
```

[XGBoost](https://xgboost.readthedocs.io/en/stable/)



```python
# -----▶ XGBoost ◀-----
from xgboost import XGBClassifier, XGBRegressor # XGB
'''
  # booster : Iteration 마다의 Model Run Type을 고를수 있음 (2가지)
    - gbtree : tree-based models
    - gblinear : linear models
  # silent : 학습하면서 running message를 프린트해줌 (Parameter 실험 시 안좋음)
    - 0은 프린트 안해주고, 1은 프린트해줌
  # nthread : 병렬처리 할때 core를 몇개 잡을 것인지
    - default로 잡을 수 있는 모든 core를 잡을 수 있도록 해줌
  # learning_rate : GBM에서 shrinking 하는 것과 같은 것
  # reg_lambda : L2 regularization term on weights (analogous to Ridge regression)
  # reg_alpha : L1 regularization term on weight (analogous to Lasso regression)
  # objective [default=reg:linear]
    - binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
    - multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities) you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
    - multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
  # eval_metric [ default according to objective ]
    - rmse – root mean square error
    - mae – mean absolute error
    - logloss – negative log-likelihood
    - error – Binary classification error rate (0.5 threshold)
    - merror – Multiclass classification error rate
    - mlogloss – Multiclass logloss
    - auc: Area under the curve
'''

n_tree = [5, 10, 20] # n_estimators
l_rate = [0.1, 0.3] # learning_rate
m_depth = [3, 5] # max_depth
L1_norm = [0.1, 0.3, 0.5] # reg_alpha
save_n = []
save_l = []
save_m = []
save_L1 = []
f1_score_ = []

for n in n_tree:
    for l in l_rate:
        for m in m_depth:
            for L1 in L1_norm:
                print("n_estimators : {}, learning_rate : {}, max_depth : {}, reg_alpha : {}".format(n, l, m, L1))
                model = XGBClassifier(n_estimators=n, learning_rate=l,
                                      max_depth=m, reg_alpha=L1, objective='binary:logistic', random_state=119)
                model.fit(X_train, y_train)

                # Train Acc
                y_pre_train = model.predict(X_train)
                cm_train = confusion_matrix(y_train, y_pre_train)
                print("Train Confusion Matrix")
                print(cm_train)
                print("Train Acc : {}".format((cm_train[0,0] + cm_train[1,1])/cm_train.sum()))
                print("Train F1-Score : {}".format(f1_score(y_train, y_pre_train)))

                # Test Acc
                y_pre_test = model.predict(X_val)
                cm_test = confusion_matrix(y_val, y_pre_test)
                print("Test Confusion Matrix")
                print(cm_test)
                print("TesT Acc : {}".format((cm_test[0,0] + cm_test[1,1])/cm_test.sum()))
                print("Test F1-Score : {}".format(f1_score(y_val, y_pre_test)))
                print("-----------------------------------------------------------------------")
                save_n.append(n)
                save_l.append(l)
                save_m.append(m)
                save_L1.append(L1)
                f1_score_.append(f1_score(y_val, y_pre_test))

best_model = XGBClassifier(n_estimators=save_n[np.argmax(f1_score_)], learning_rate=save_l[np.argmax(f1_score_)],
                           max_depth=save_m[np.argmax(f1_score_)], reg_alpha=save_L1[np.argmax(f1_score_)], objective='binary:logistic',
                           random_state=119)
best_model.fit(X.iloc[train_idx], Y.iloc[train_idx])
```

[LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)



```python
# -----▶ LightGBM ◀-----
import lightgbm as lgb
import optuna # 하이퍼파라미터 튜닝

'''
  ▷ n_jobs : LGBMClassifier와 같은 Scikit-learn 모델에서 병렬 처리를 위해 사용되는 CPU 코어의 수를 지정.
    - n_jobs = -1: 사용 가능한 모든 코어를 사용하도록 지시.
    - n_jobs = 1: 병렬 처리 없이 하나의 CPU 코어만을 사용하도록 지시.
    - n_jobs = N (N > 1): N개의 CPU 코어를 사용하도록 지시.
  ▷ random_state(seed): Random number seed for reproducibility of results.
  ▷ scale_pos_weight : LightGBM에서 클래스 불균형을 다루기 위해 사용. 양성 샘플과 음성 샘플의 균형을 맞추기 위한 가중치.
  ▷ max_depth : 결정 트리의 최대 깊이를 제한하는 데 사용. 결정 트리가 얼마나 깊게 성장할 수 있는지 제한하여 모델의 복잡성을 조절.
    - max_depth=None: 트리에 깊이 제한이 없음을 의미. 각 리프 노드가 순수해질 때까지 계속 분할될 수 있음.
    - max_depth=n (n은 양의 정수): 트리의 최대 깊이를 n으로 제한. 트리가 n 단계까지만 성장하고, 그 이후로는 추가적인 분할이 일어나지 않음.
    ** max_depth 값은 과적합과 모델 복잡성 사이의 trade-off 관계를 조절하는 중요한 하이퍼파라미터. 너무 큰 값은 과적합 위험, 너무 작은 값은 모델이 충분한 학습을 할 수 없음.
  ▷ reg_lambda : L2 정규화(regularization)를 조절하는 데 사용. reg_lambda는 L2 정규화 항에 대한 가중치를 조절하는 역할.
                : L2 정규화는 모델의 가중치를 제한하여 과적합을 방지하고 일반화 성능을 향상.
    - reg_lambda=0: L2 정규화가 없음을 의미. 가중치에 대한 제약이 없으므로 모델은 데이터에 완벽하게 적합.
    - reg_lambda>0: L2 정규화가 적용. 값이 커질수록 L2 정규화의 강도가 강해지며, 가중치 값들이 작아지게 됨.
    ** reg_lambda 값은 과적합과 일반화 사이의 trade-off 관계를 조절. 작은 값은 모델의 복잡성을 증가, 큰 값은 모델의 복잡성을 줄임. 이 값을 조정하여 최적의 균형점을 찾아야 함.
  ▷ reg_alpha : L1 정규화(regularization)를 조절하는 데 사용. reg_alpha는 L1 정규화 항에 대한 가중치를 조절하는 역할.
               : L1 정규화는 모델의 가중치를 제한하여 과적합을 방지하고 일반화 성능을 향상.
    - reg_alpha=0: L1 정규화가 없음을 의미. 가중치에 대한 제약이 없으므로 모델은 데이터에 완벽하게 적합.
    - reg_alpha>0: L1 정규화가 적용. 값이 커질수록 L1 정규화의 강도가 강해지며, 가중치 값들이 0으로 수렴.
    ** reg_alpha 값은 과적합과 일반화 사이의 trade-off 관계를 조절. 작은 값은 모델의 복잡성을 증가, 큰 값은 모델의 복잡성을 줄임. 이 값을 조정하여 최적의 균형점을 찾아야 함.
  ▷ eval_metric [ default according to objective ]
    - rmse – root mean square error
    - mae – mean absolute error
    - logloss – negative log-likelihood
    - error – Binary classification error rate (0.5 threshold)
    - merror – Multiclass classification error rate
    - mlogloss – Multiclass logloss
    - auc: Area under the curve
'''
# ▶ 1. Default
clf = lgb.LGBMClassifier(random_state=rand, n_jobs=-1)

### --------------------------------------------------------------- ###
# ▶ 2. scale_pos_weight로 하이퍼파라미터 튜닝
# Class Weights - 1) scale_pos_weight
# Positive 클래스를 예측할 때 보수적 접근방법으로 모델을 생성 -> Positive 클래스에 더 많은 비중을 두도록 강제하는 Hyperparameter를 활용
# 가중치 = number of negative samples/number of positive samples
def_scale_pos_weight = len(y[y==0]) / len(y[y==1])
print(f"default scale pos weight: {def_scale_pos_weight:.2f}")
clf = lgb.LGBMClassifier(random_state=rand, n_jobs=-1, scale_pos_weight=def_scale_pos_weight)

### --------------------------------------------------------------- ###
# ▶ 3. max_depth(트리 깊이를 제한하기 위해), reg_lambda 및 reg_alpha를 사용한 L1/L2 정규화 하이퍼파라미터 튜닝
def lambda_alpha_tuning(trial):
params = {
    'max_depth': trial.suggest_int('max_depth', 2, 11),
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', def_scale_pos_weight/2, def_scale_pos_weight*2),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
}
if params['max_depth'] == 11:
    params['max_depth'] = -1

clf = lgb.LGBMClassifier(random_state=rand, n_jobs=-1, **params)

# optuna를 이용해서 trial을 통해 범위 내에 하이퍼파라미터 값을 제안
opt_study = optuna.create_study(direction='maximize')
opt_study.optimize(lambda_alpha_tuning, n_trials=100)

# 최적의 파라미터 값 찾기
best_params = opt_study.best_params
print(best_params)

# 최적의 파라미터로 모델 생성
best_params = {'max_depth': 4, 'scale_pos_weight': 7.687847660573682, 'reg_lambda': 1.2481827996825933e-07,\
               'reg_alpha': 6.217363714063122e-06}
clf = lgb.LGBMClassifier(random_state=rand, n_jobs=-1,**best_params)

# 학습
clf.fit(X_train, y_train)

### --------------------------------------------------------------- ###
# 4. for문을 이용한 직접 하이퍼파라미터 튜닝
from lightgbm import LGBMClassifier, LGBMRegressor # LightGBM
n_tree = [5, 10, 20] # n_estimators
l_rate = [0.1, 0.3] # learning_rate
m_depth = [3, 5] # max_depth
L1_norm = [0.1, 0.3, 0.5] # reg_alpha
save_n = []
save_l = []
save_m = []
save_L1 = []
f1_score_ = []
for n in n_tree:
    for l in l_rate:
        for m in m_depth:
            for L1 in L1_norm:
                print("n_estimators : {}, learning_rate : {}, max_depth : {}, reg_alpha : {}".format(n, l, m, L1))
                model = LGBMClassifier(n_estimators=n, learning_rate=l,
                                       max_depth=m, reg_alpha=L1,
                                       n_jobs=-1, objective='cross_entropy')
                model.fit(X_train, y_train)

                # Train Acc
                y_pre_train = model.predict(X_train)
                cm_train = confusion_matrix(y_train, y_pre_train)
                print("Train Confusion Matrix")
                print(cm_train)
                print("Train Acc : {}".format((cm_train[0,0] + cm_train[1,1])/cm_train.sum()))
                print("Train F1-Score : {}".format(f1_score(y_train, y_pre_train)))

                # Test Acc
                y_pre_test = model.predict(X_val)
                cm_test = confusion_matrix(y_val, y_pre_test)
                print("Test Confusion Matrix")
                print(cm_test)
                print("TesT Acc : {}".format((cm_test[0,0] + cm_test[1,1])/cm_test.sum()))
                print("Test F1-Score : {}".format(f1_score(y_val, y_pre_test)))
                print("-----------------------------------------------------------------------")
                save_n.append(n)
                save_l.append(l)
                save_m.append(m)
                save_L1.append(L1)
                f1_score_.append(f1_score(y_val, y_pre_test))

best_model = LGBMClassifier(n_estimators=save_n[np.argmax(f1_score_)], learning_rate=save_l[np.argmax(f1_score_)],
                           max_depth=save_m[np.argmax(f1_score_)], reg_alpha=save_L1[np.argmax(f1_score_)], objective='cross_entropy',
                           random_state=119)
best_model.fit(X.iloc[train_idx], Y.iloc[train_idx])
```

[K-Nearest Neighbors](url)



```python
# -----▶ K-Nearest Neighbors ◀-----
from sklearn.model_selection import GridSearchCV # 하이퍼파라미터 튜닝을 자동화하는 역할, 하이퍼파라미터 값들의 모든 가능한 조합에 대해 교차 검증을 수행
from sklearn.neighbors import KNeighborsClassifier
'''
  # 효과적인 분류 및 회귀 알고리즘 중 하나.
  # 가장 가까운 'k' 개의 이웃을 찾음.
  # n_neighbors : 이 파라미터는 KNN 알고리즘에서 고려할 이웃의 수를 지정. 기본값은 5.
    - 'k'가 작으면 모델은 복잡해지고 훈련 데이터에 과적합될 가능성이 높아짐.
    - 'k'가 크면 모델은 단순해지지만 훈련 데이터에 대한 정보를 충분히 활용하지 못할 수 있음.
  # weights : 이 파라미터는 예측에 사용되는 가중치 함수를 지정. 기본값은 'uniform'
    - 'uniform': 모든 최근접 이웃들의 가중치를 동일하게 취급.
    - 'distance': 거리의 역수로 가중치를 부여하여, 멀리 있는 이웃보다 가까운 이웃이 더 큰 영향을 미침.
'''
param_grid = {'weights': ['uniform', 'distance'],
              'n_neighbors': np.arange(1,16)}
knn_clf = KNeighborsClassifier()
gs_knn = GridSearchCV(knn_clf, param_grid=param_grid)

start = time() # 시작 시간
gs_knn.fit(X_train_std, y_train) # 모델 학습
knn_duration = time() - start # 학습 경과 시간

print("GridSearchCV of KNN took {:.2f} seconds for {} candidate parameter settings.".format(knn_duration,
                                                                                            len(gs_knn.cv_results_['params'])))
# report(gs_knn.cv_results_)
print('Optimal weights: {}  Optimal n_neighbors: {}'.format(gs_knn.best_params_['weights'], gs_knn.best_params_['n_neighbors']))
print('Accuracy (train): {:.2f}'.format(gs_knn.score(X_train_std, y_train))) # 훈련 정확도
print('Accuracy  (test): {:.2f}'.format(gs_knn.score(X_test_std,  y_test))) # 테스트 정확도
# best estimator
print(gs_knn.best_estimator_.get_params())
```

[K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)



```python
# -----▶ K-means ◀-----
# => Distance 기반의 Clustering의 경우 Scaling이 필수
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
'''
  # n_clusters : Cluster 개수 (K)
  # n_init : Number of times the k-means algorithm is run with different centroid seeds
    - K-means는 Step2에서 '초기 중심점 설정'이라는 작업을 하는데, 초기 중심점을 셋팅하는 것에 따라 군집의 Quality가 달라짐
    - 따라서 여러번 시도해 보는것
    - default = 10
  # max_iter : 몇번 Round를 진행할 것 인지
    - default = 300
    - 300번 안에 중심점 움직임이 멈추지 않으면 그냥 STOP
'''
# 차원축소
pca = PCA(n_components=2).fit(X)
X_PCA = pca.fit_transform(X)
X_EMM = pd.DataFrame(X_PCA, columns=['AXIS1','AXIS2'])
print(">>>> PCA Variance : {}".format(pca.explained_variance_ratio_))

# K-means Modeling
for cluster in list(range(2, 6)):
    Cluster = KMeans(n_clusters=cluster).fit(X_scal)
    labels = Cluster.predict(X_scal)

    # label Add to DataFrame
    data['{} label'.format(cluster)] = labels
    labels = pd.DataFrame(labels, columns=['labels'])
    # Plot Data Setting
    plot_data = pd.concat([X_EMM, labels], axis=1)
    groups = plot_data.groupby('labels')

    mar = ['o', '+', '*', 'D', ',', 'h', '1', '2', '3', '4', 's', '<', '>']
    colo = ['red', 'orange', 'green', 'blue', 'cyan', 'magenta', 'black', 'yellow', 'grey', 'orchid', 'lightpink']

    fig, ax = plt.subplots(figsize=(10,10))
    for j, (name, group) in enumerate(groups):
        ax.plot(group['AXIS1'],
                group['AXIS2'],
                marker=mar[j],
                linestyle='',
                label=name,
                c = colo[j],
                ms=10)
        ax.legend(fontsize=12, loc='upper right') # legend position
    plt.title('Scatter Plot', fontsize=20)
    plt.xlabel('AXIS1', fontsize=14)
    plt.ylabel('AXIS2', fontsize=14)
    plt.show()
    print("---------------------------------------------------------------------------------------------------")

    gc.collect()

# Confusion Matrix 확인
cm = confusion_matrix(data['Target'], data['3 label'])
print(cm)
```

[Hierachical Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)



```python
# -----▶ Hierachical Clustering ◀-----
# => Distance 기반의 Clustering의 경우 Scaling이 필수
# => K-means와 달리 군집 수(K)를 사전에 정하지 않아도 학습을 수행
# 계산 복잡성은  O(n3)
from scipy.cluster.hierarchy import dendrogram, linkage

meth = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

HC = linkage(X_scal,method=meth[-1])
plt.figure(figsize=(20,10))
dendrogram(HC,
            leaf_rotation=90,
            leaf_font_size=20)
plt.show()
```

[Spectral Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)



```python
# -----▶ Spectral Clustering ◀-----
# => Distance 기반의 Clustering의 경우 Scaling이 필수
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
'''
  # n_clusters : Cluster 개수 (K)
  # affinity : 유사도 행렬 만드는 방법
    - ‘nearest_neighbors’: construct the affinity matrix by computing a graph of nearest neighbors.
    - ‘rbf’: construct the affinity matrix using a radial basis function (RBF) kernel.
    - ‘precomputed’: interpret X as a precomputed affinity matrix, where larger values indicate greater similarity between instances.
    - ‘precomputed_nearest_neighbors’: interpret X as a sparse graph of precomputed distances, and construct a binary affinity matrix from the n_neighbors nearest neighbors of each instance.
  # n_neighbors : 유사도 계산시 주변 n개를 보고 판단할 것 인지
    - Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity='rbf'
'''

# 차원축소
pca = PCA(n_components=2).fit(X)
X_PCA = pca.fit_transform(X)
X_EMM = pd.DataFrame(X_PCA, columns=['AXIS1','AXIS2'])
print(">>>> PCA Variance : {}".format(pca.explained_variance_ratio_))

# Spectral Clustering Modeling
for cluster in list(range(2, 6)):
    Cluster = SpectralClustering(n_clusters=cluster).fit(X_scal)
    labels = Cluster.labels_

    # label Add to DataFrame
    data['{} label'.format(cluster)] = labels
    labels = pd.DataFrame(labels, columns=['labels'])
    # Plot Data Setting
    plot_data = pd.concat([X_EMM, labels], axis=1)
    groups = plot_data.groupby('labels')

    mar = ['o', '+', '*', 'D', ',', 'h', '1', '2', '3', '4', 's', '<', '>']
    colo = ['red', 'orange', 'green', 'blue', 'cyan', 'magenta', 'black', 'yellow', 'grey', 'orchid', 'lightpink']

    fig, ax = plt.subplots(figsize=(10,10))
    for j, (name, group) in enumerate(groups):
        ax.plot(group['AXIS1'],
                group['AXIS2'],
                marker=mar[j],
                linestyle='',
                label=name,
                c = colo[j],
                ms=4)
        ax.legend(fontsize=12, loc='upper right') # legend position
    plt.title('Scatter Plot', fontsize=20)
    plt.xlabel('AXIS1', fontsize=14)
    plt.ylabel('AXIS2', fontsize=14)
    plt.show()
    print("---------------------------------------------------------------------------------------------------")

    gc.collect()

# Confusion Matrix 확인
cm = confusion_matrix(data['censor'], data['2 label'])
print(cm)

# ACC & F1-Score
print("TesT Acc : {}".format((cm[0,0] + cm[1,1])/cm.sum()))
print("F1-Score : {}".format(f1_score(data['censor'], data['2 label'])))
```

[DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)



```python
# -----▶ DBSCAN ◀-----
# => 밀도 기반의 기법
'''
  # eps : 이웃을 판단하는 거리
  # metric : 거리를 계산할 때 사용하는 방법
    - default : euclidean
  # min_samples : eps안에 적어도 몇개 들어와야 하는지 이웃의 숫자
'''
from sklearn.cluster import DBSCAN

epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
minPls = [5, 10, 15, 20]

for e in epsilon:
    for m in minPls:
        print("epsilon : {}, minPls : {}".format(e, m))
        db = DBSCAN(eps=e, min_samples=m).fit(test_data)
        palette = sns.color_palette()
        cluster_colors = [palette[col]
                        if col >= 0 else (0.5, 0.5, 0.5) for col in
                        db.labels_]
        plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
        plt.show()
```

[HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)



```python
# -----▶ HDBSCAN ◀-----
# => 밀도 기반의 기법
'''
  # min_cluster : Cluster 안에 적어도 몇개가 있어야 하는지
  # cluster_selection_epsilon : combining HDBSACN with DBSCAN
'''
!pip install hdbscan
import hdbscan

minsize = [3, 5, 10, 15, 20, 30]
for m in minsize:
    print("min_cluster_size : {}".format(m))
    db = hdbscan.HDBSCAN(min_cluster_size=m).fit(test_data)
    palette = sns.color_palette()
    cluster_colors = [palette[col]
                    if col >= 0 else (0.5, 0.5, 0.5) for col in
                    db.labels_]
    plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
    plt.show()
```

## 8. 모델 비교



```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, cohen_kappa_score, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred, duration, model_name, *args):
    acc     = accuracy_score(y_true, y_pred)
    pre     = precision_score(y_true, y_pred)
    rec     = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred)
    ck      = cohen_kappa_score(y_true, y_pred)
    p, r, fbeta, support = precision_recall_fscore_support(y_true, y_pred)
    metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1_score', 'cohen_kappa',
               'precision_both', 'recall_both', 'fbeta_both', 'support_both', 'time_to_fit (seconds)']
    s = pd.Series([acc, pre, rec, roc_auc, f1, ck, p, r, fbeta, support, duration], index=metrics)
    s.name = model_name
    return s

lr_metrics = calculate_metrics(y_test, y_pred, lr_duration, 'logistic_regression')
knn_metrics = calculate_metrics(y_test, y_pred, knn_duration, 'k-nearest neighbors')
svc_metrics = calculate_metrics(y_test, y_pred, svc_duration, 'support vector machines')
rf_metrics = calculate_metrics(y_test, y_pred, rf_duration, 'random forest')

model_metrics = pd.concat([lr_metrics, knn_metrics, svc_metrics, rf_metrics], axis=1).T
model_metrics.apply(lambda elem: [np.round(val, 2) for val in elem]).sort_values(by='f1_score', ascending=False)
```

설명 방법 1 : [SHAP](https://shap.readthedocs.io/en/latest/)



```python
# ▶ explainable method : (1) SHAP
!pip install shap
import shap
'''
  # force_plot : 특정 데이터 하나 또는 전체 데이터에 대해 Shapley value를 1차원 평면에 정렬하여 보여주는 Plot
  # dependence_plot : 각 특성의 Shapely Value를 확인할 수 있음
    - y축에 나온 특성은 선택한 x와의 관계(상호작용 효과)를 나타냄, 그래프 상에서 색깔이 수직 패턴이 나오는 경우 관계가 존재한다고 판단할 수 잇음
  # Summary_plot : Global하게 Shapely value를 보여주는 plot
    - shapely value 분포에 어떤 영향을 미치는지 시각화 해줌
    - shap_interaction_values : X's 간 관계 (상관관계)를 파악할 수 있는 Plot
'''

# SHAP Explainer 만들기
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train) # Shap values 계산

# 1) force_plot : 특정 데이터 하나 또는 전체 데이터에 대해 Shapley value를 1차원 평면에 정렬하여 보여주는 Plot
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[2, :], X_train['col'], link="logit")

# 2) 누적 시각화
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_train)

# 3) 영향력 Top 3에 대한 Dependence plot
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
for i in range(3):
    shap.dependence_plot(top_inds[i], shap_values, X_train)

# 4) Global Importance Score
shap.summary_plot(shap_values, X_train)

# 5) Interaction Plot
shap_interaction_values = explainer.shap_interaction_values(X_train)
shap.summary_plot(shap_interaction_values, X_train)
```
