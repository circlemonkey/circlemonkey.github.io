---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
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


# 머신러닝 시작 전 미리 알아두기!

[목차]

1. Vector, Vector Space의 정의

2. Machine Learning model과 Statistical Model의 차이점

3. Machine Learning model과 Deep Learning model의 차이점

4. Machine Learning task의 정의(classification vs regression vs clustering)

5. Multivariate function(다변수 함수)의 개념

6. partial derivative(편미분)의 개념

7. feature engineering이란?

8. Linear Classifier, Linear Regression 모델의 학습원리

9. Decision Tree(CART) 모델의 학습원리

10. sklearn의 fit, predict 함수 사용법과 의미 이해하기


## Vector, Vector Space



### Vector



#### 벡터란?

- 벡터란? 크기와 방향을 가진 물리량.

- 스칼라란? 크기만을 가진 물리량.

- 두 벡터의 시작점이 달라도 크기와 방향이 같을 경우 두 벡터는 서로 같다(또는 동치이다).



#### 벡터의 합

- $\vec{AB}$와 $\vec{BC}$의 합은 이동한 거리와 방향 $\vec{AC}$로 정의한다.

<br> ✅ $\vec{AB}+\vec{BC} = \vec{AC}$

- 벡터 $\vec{u}$와 벡터 $\vec{v}$의 차이 $\vec{u}-\vec{v} = \vec{u} + (-\vec{v})$로 정의된다.

- $\vec{v}$와 $-\vec{v}$는 방향이 정반대이고 크기가 같은 벡터이다.



#### 벡터와 스칼라의 곱

- 벡터 $\vec{v}$에 크기만을 가진 값 스칼라 c를 곱하면 방향은 그대로이고 크기는 c배가 된 벡터이다.

- 벡터 $\vec{v}$의 크기는 벡터 $|\vec{v}|$으로 표기한다.

- c가 0이거나 벡터 $\vec{v}$의 크기가 0이었다면 둘을 곱하면 크기가 0인 벡터가 되고 이를 ***영벡터***라고 한다.



#### 벡터의 성질

- 벡터 $\vec{v}$, $\vec{u}$, $\vec{w}$와 스칼라 c,d에 대해서 다음이 성립

<br> 1) $\vec{v} + \vec{u} = \vec{u} + \vec{v}$

<br> 2) $\vec{v} + (\vec{u} + \vec{w}) = (\vec{v} + \vec{u}) + \vec{w}$

<br> 3) $\vec{v} + \vec{0} = \vec{v}$, $\vec{v} + (-\vec{v}) = \vec{0}$

<br> 4) $c(\vec{v}+\vec{u}) = c\vec{v}+c\vec{u}$, $(c+d)\vec{v} = c\vec{v} + d\vec{v}$

<br> 5) $cd\vec{v}=c(\vec{v})$



### Vector Space



#### 차원이란?

- 1차원 : 선

- 2차원 : 면

- 3차원 : 공간



#### 벡터공간

    ⚡ 벡터공간이란, 벡터들의 집합으로 벡터들 사이에 합과 스칼라 곱이 정의된 공간을 뜻한다.

    ⚡ 벡터들의 선형결합에 대하여 닫히 공간이어야 한다.

    ⚡ 닫힌 공간이란 벡터공간에 속한 임의의 벡터들의 선형결합도 역시 주어진 벡터공간에 속하는 것을 뜻한다.

> 🔥 실수들의 집합을 R이라고 하자. $R^n$이란 요소 n개로 이루어진 열(벡터)(n × 1행렬)들의 집합을 말한다.

<br>🔥 이때, n을 차원이라고 하고 n=1(선), n=2(면), n=3(공간)으로 표현한다.

<br>🔥 실수는 또 다른 말로 스칼라라고 한다.

<br>🔥 스칼라 c와 n차원 벡터 $\vec{x} = (x_1,...,x_n)$에 대해 그 곱을 $c\vec{x} = (cx_1,...,cx_n)$ 으로 정의한다.



##### [선형결합]

- 스칼라와 벡터들의 곱들을 합한 형태를 말함.

<br> 스칼라 $c_1, c_2, ... , c_n$과 벡터 $\vec{x}_1,\vec{x}_2, ... , \vec{x}_n$

<br> ➡ $c_1\vec{x}_1 + c_2\vec{x}_2 + ... + c_n\vec{x}_n$을 말함.



##### [부분공간]

- 벡터공간의 부분공간이란 벡터공간의 공집합이 아닌 부분집합으로 원소들 사이의 선형결합이 여전히 부분집합에 속하는 경우.

- 아래의 두 성질을 만족하는 공간을 뜻함.

<br>1) $\vec{x}$와 $\vec{y}$가 부분공간에 속하면 그들의 합 $\vec{x}+\vec{y}$도 부분공간에 속한다.

<br>2) 부분공간에 속하는 $\vec{x}$와 임의의 스칼라 c를 곱한 $c\vec{x}$도 여전히 부분공간에 속한다.



##### [열공간, null공간]

- 행렬 A의 ***열공간***이란 행렬 A의 각 열들의 선형 결합으로 이루어진 공간

- 행렬 방정식 $A\vec{x} = \vec{0}$의 해들의 집합은 벡터 공간을 이루며 이를 ***null공간***이라고 부른다.


## Machine Learning model과 Statistical Model의 차이점

|구분|머신러닝 모델|통계 모델|

|--|--|--|

|목적|- 데이터에서 패턴을 학습하고 새로운 데이터에 대한 예측을 수행하는 것이 주된 목적<br><br>- 분류(Classification), 회귀(Regression), 군집화(Clustering) 등의 작업에 사용|- 주어진 데이터로부터 인과관계나 변수 간의 관련성 등을 추론<br><br>- 현상에 대한 이해와 설명력을 제공하는 것이 주된 목적|

|가정|- 가정 없이 데이터에서 패턴을 학습<br><br>- 비선형성 및 상호작용 등 복잡한 패턴을 잘 처리|- 변수간의 관련성 및 인관관계에 대한 가정을 설정하고 이를 기반으로 추론<br><br>- 독립변수와 종속변수 간 선형 관계, 오차항의 정규성|

|모수추정|- 많은 수의 파라미터(가중치)를 포함<br><br>- 손실 함수 최소화와 같은 최적화 문제를 해결하여 파라미터 값을 추정<br><br>- 파라미터는 최소화되는 손실함수를 기반으로 경사 하강법 등 최적화 알고리즘을 사용하여 추정|- 작은 수의 파라미터(회귀 계수 등)를 포함<br><br>- 최대 가능도 추정(MLE), 최소제곱법(OLS) 등 전통적인 방법으로 파라미터 값을 추정|

|일반화및해석가능성|- "블랙박스"로 볼 수 있는 알고리즘이 많아서 해석하기 어려울 수 있음|- 과학 연구나 실험에서 현재 데이터셋에 대해 해석 가능한 결과와 결론을 도출하는 것이 중요<br><br>- 통계적 모델은 파라미터의 추정치와 그 불확실성을 제공하며, 이를 통해 변수 간의 관계를 해석|


## Machine Learning model과 Deep Learning model의 차이점

|구분|머신러닝 모델|딥러닝 모델|

|--|--|--|

|특징추출|- 대부분 특징 추출이 수동<br><br>- 도메인 전문가가 데이터를 분석하고 어떤 특징이 문제 해결에 유용할지 결정|- 특징 추출 과정이 자동화<br><br>- 네트워크는 원시 데이터에서 직접 유용한 특징을 학습할 수 있음|

|모델구조|- 선형회귀, 로지스틱회귀, SVM 등 비교적 단순한 알고리즘 사용|- 컨볼루션 신경망(CNN), 순환신경망(RNN), 변환자(Transformer)등 복잡한 신경망 구조|

|데이터요구량|- 작거나 중간 규모의 데이터 셋에서도 잘 동작|- 일반적으로 대량의 데이터가 필요<br><br>- 충분한 양의 학습 데이터 없이는 복잡한 네트워크 구조를 제대로 학습시키기 어려움|

|학습시간과리소스|- 상대적으로 적은 계산 리소스와 시간을 필요로 함<br><br>- 개인용 컴퓨터에서도 충분히 학습 가능|- 대량의 데이터와 복잡한 네트워크 구조 때문에 GPU같은 고성능 하드웨어와 많은 시간 필요|

|해석가능성|- 결과를 해석하는 것이 비교적 쉬움|- 내부 작동원리가 복잡하고 해석하기 어려움(블랙박스로 간주..)|



## Machine Learning task의 정의(classification vs regression vs clustering)



### Machine Learning task

![Machine Learning task](https://miro.medium.com/v2/resize:fit:720/0*UsO539Yis8JPIVLa)

<br>[이미지 출처) https://miro.medium.com/v2/resize:fit:720/0*UsO539Yis8JPIVLa](https://miro.medium.com/v2/resize:fit:720/0*UsO539Yis8JPIVLa)



#### 1) 지도학습(Supervised Learning)

- 입력 변수(X)와 출력 변수(Y)의 관계를 모델링하는 작업.

- Y는 사전에 알려진 정답 레이블(label).

- 회귀(Regression)와 분류(Classification)가 이 범주에 속합니다.



#### 2) 비지도학습(Unsupervised Learning)

- 입력 변수(X)만 사용하여 데이터의 구조나 패턴을 찾는 작업.

- 출력 변수(Y), 즉 정답 레이블은 제공되지 않습니다.

- 클러스터링(Clustering), 차원 축소(Dimensionality Reduction), 연관 규칙 학습(Association Rule Learning) 등이 여기에 속합니다.



#### 3) 강화학습(Reinforcement Learning)

- 에이전트(agent)가 환경과 상호작용하며 보상(reward)을 최대화하는 행동(action)을 학습하는 과제입니다.

- 시행착오를 통해 학습하며, 어떤 행동이 장기적으로 가장 큰 보상을 가져올지를 배우는 것에 중점을 둡니다.

>🔥 에이전트(Agent): 환경에서 행동을 수행하고 그 결과로서 보상과 새로운 상태 정보를 받는 주체입니다.

<br>🔥 환경(Environment): 에이전트가 상호작용하는 공간입니다. 에이전트의 행동에 따라 상태가 변하고 보상이 제공됩니다.

<br>🔥 행동(Action): 에이전트가 각 시점에서 선택할 수 있는 선택지입니다.

<br>🔥 보상(Reward): 각 행동 후에 환경에서 제공되는 신호로, 좋은 행동은 긍정적인 보상으로, 나쁜 행동은 부정적인 보상(또는 패널티)로 반영됩니다.

<br>🔥 정책(Policy): 주어진 상태에서 어떤 행동을 취할지 결정하는 전략 또는 규칙입니다. 이것은 강화학습의 목표 중 하나로, 최적의 정책을 찾아내려 하는 것입니다.

<br>🔥 상태(State): 환경의 현재 조건 또는 에이전트의 관찰 결과를 의미합니다.

- 게임(예: 알파고), 로봇 제어, 자원 관리, 자율 주행 등 다양한 분야에서 강화학습이 사용.



### Classification vs Regression vs Clustering



#### 1) Classification(분류)

- 지도 학습(Supervised Learning)의 한 유형으로, 입력 데이터를 두 개 이상의 고정된 범주나 클래스로 분류하는 작업입니다.

- 스팸 메일 필터링(스팸인지 아닌지), 이미지 인식(이미지가 어떤 객체를 나타내는지) 등이 있습니다.

- [다음과 같은 상황에서 사용]

      ⚡ 이진 분류(Binary Classification): 출력 클래스가 두 개인 경우를 말합니다. 예를 들어, 스팸 메일 필터링에서 메일이 스팸인지 아닌지를 결정하는 것은 이진 분류 문제입니다.

      ⚡ 다중 클래스 분류(Multiclass Classification): 출력 클래스가 세 개 이상인 경우를 말합니다. 예를 들어, 손으로 쓴 숫자 이미지(0~9)을 인식하여 해당 숫자를 결정하는 것은 다중 클래스 분류 문제입니다.

      ⚡ 다중 레이블 분류(Multilabel Classification): 각 입력 샘플이 여러개의 클래스 레이블을 가질 수 있는 경우입니다. 예를 들어, 한 영화가 여러 장르(액션, 코미디, 드라마 등)에 속할 수 있습니다.

- [알고리즘 종류]

> 로지스틱 회귀(Logistic Regression), 나이브 베이즈(Naive Bayes), K-최근접 이웃(K-Nearest Neighbors), 서포트 벡터 머신(Support Vector Machines), 결정 트리(Decision Trees), 랜덤 포레스트(Random Forests), 그래디언트 부스팅(Gradient Boosting), 심층 학습(Deep Learning)



#### 2) Regression(회귀)

- 지도 학습의 한 유형으로, 입력 데이터에 대한 연속적인 출력 값을 예측하는 작업입니다.

- 주택 가격 예측(주어진 특성에 따른 집값), 주식 가격 예측 등이 회귀 문제에 해당합니다.

- [다음과 같은 상황에서 사용]

      ⚡ 단일 회귀(Simple Regression): 하나의 독립 변수(입력 특성)를 사용하여 종속 변수(출력 값)를 예측합니다. 예를 들어, 집의 크기에 따른 집값을 예측하는 것이 단일 회귀 문제가 될 수 있습니다.

      ⚡ 다중 회귀(Multiple Regression): 두 개 이상의 독립 변수를 사용하여 종속 변수를 예측합니다. 예를 들어, 집의 크기와 위치에 따른 집값을 예측하는 것이 다중 회귀 문제가 될 수 있습니다.

- [알고리즘 종류]

> 선형 회귀(Linear Regression), 로지스틱 회귀(Logistic Regression), 다항 회귀(Polynomial Regression), 릿지(Ridge Regression), 라소(Lasso Regression), 엘라스틱넷(ElasticNet), 서포트 벡터 머신(Support Vector Machines for regression, called SVR), 결정 트리(Decision Trees for regression), 랜덤 포레스트(Random Forests for regression) 등



#### 3) Clustering(군집화)

- 군집화는 비지도 학습(Unsupervised Learning)의 한 유형으로, 레이블이 없는 데이터 세트를 서로 다른 그룹 혹은 클러스터로 분할하는 것을 목표로 합니다.

- 비슷한 특성을 갖는 데이터 포인트들을 같은 클러스터로 그룹핑합니다.

- 고객 세분화(Customer Segmentation), 이미지 분할(Image Segmentation) 등에서 사용됩니다.

- [다음과 같은 상황에서 사용]

      ⚡ 시장 세분화(Market Segmentation): 고객들을 그들의 구매 패턴, 소비 행동, 개인적 특성 등에 따라 여러 그룹으로 나눕니다. 이를 통해 각 세그먼트에 맞춤형 마케팅 전략을 개발할 수 있습니다.

      ⚡ 사회 네트워크 분석(Social Network Analysis): 소셜 네트워크에서 커뮤니티를 탐지하는 데 사용됩니다.

      ⚡ 이미지 분할(Image Segmentation): 이미지 내에서 비슷한 픽셀을 그룹핑하여 이미지를 여러 부분으로 나눕니다.

      ⚡ 이상치 탐지(Anomaly Detection): 이상치 탐지에서도 군집화가 활용됩니다. 정상적인 데이터 포인트들이 형성하는 클러스터와 크게 벗어난 위치에 있는 데이터 포인트를 이상치로 간주할 수 있습니다.

- [알고리즘 종류]

> K-평균(K-Means), 계층적 군집 분석(Hierarchical Clustering), DBSCAN(Density-Based Spatial Clustering of Applications with Noise), 스펙트럴 군집(Spectral Clustering), 평균-시프트(Mean-Shift) 등


## Multivariate function(다변수 함수)의 개념

- 두 개 이상의 독립 변수를 입력으로 받는 함수를 말합니다.

- 종종 고차원 데이터를 모델링하는 데 사용되며, 각 변수간의 복잡한 상호작용을 포착할 수 있습니다.

- 다변수 함수는 여러 변수들 사이의 관계를 설명하는 데 사용됩니다.

> "$f(x, y) = x^2 + y^2$" 는 2개의 독립 변수 x와 y를 가진 다변수 함수

<br> ➡ 이 함수는 2차원 공간에서 각 점 (x, y)에서 원점까지의 거리의 제곱을 계산

<br> "$g(x_1, x_2,... x_n) = Σ(x_i^2)$" 은 n개의 독립 변수 $x_1$부터 $x_n$까지를 가진 다변수 함수

<br> ➡ 이 함수는 n차원 공간에서 각 점 $g(x_1, x_2,... x_n)$에서 원점까지의 거리 제곱을 계산





## partial derivative(편미분)의 개념

    ⚡ 다변수 함수에서 한 변수에 대한 미분을 의미.

    ⚡ 다른 변수들은 상수로 간주되며 그 값이 변하지 않는다고 가정.

    ⚡ 편미분은 기울기(gradient), 자코비안(Jacobian), 해시안(Hessian) 등과 같이 벡터나 매트릭스 형태의 도함수(derivative)를 구하는데 사용.

    ⚡ 최적화 문제나 딥러닝에서 경사 하강법(gradient descent) 등을 구현할 때 중요한 개념.

- 두 변수 x와 y에 대한 함수 f(x, y)

<br> - x에 대한 f의 편미분은 y를 상수로 간주하고 f를 x만의 함수로 생각하여 미분.

<br> $∂f \over ∂x$ 또는 $f_x$

<br> - y에 대한 f의 편미분은 x를 상수로 간주하고 f를 y만의 함수로 생각하여 미분.

<br> $∂f \over ∂y$ 또는 $f_y$

- [예시]

<br> $f(x, y) = 3x^2 + 2xy + y^3$

<br> ✅ x에 대한 편미분:

<br> 여기서는, y를 상수로 취급하고 x만 미분합니다. ${∂f \over ∂x} = 6x + 2y$

<br> ✅ y에 대한 편미분:

<br> 여기서는 x를 상수로 취급하고 y만 미분합니다. ${∂f \over ∂y} = 2x + 3y^2$


## feature engineering이란?

    ⚡ 머신러닝 모델을 위한 데이터를 준비하는 과정.

    ⚡ 입력 데이터의 특성(Features)을 선택, 추출, 생성하여 머신러닝 알고리즘이 이해하고 사용할 수 있는 형태로 변환하는 작업을 포함.

- Feature Engineering의 여러 가지

>🔥 Feature Selection: 가장 유익한 특성들을 선택하는 과정입니다. 모든 특성이 유용하지 않으며, 일부는 노이즈를 추가하거나 오버피팅(Overfitting)을 초래할 수 있습니다.

<br>🔥 Feature Extraction: 기존의 원시 데이터에서 새로운 특성을 추출하는 과정입니다. 예를 들어, 날짜에서 요일이나 월 등의 정보를 추출하거나, 텍스트 데이터에서 주요 단어나 구문을 추출할 수 있습니다.

<br>🔥 Feature Creation: 새로운 특성을 생성하는 과정입니다. 종종 여러 개의 기존 특성들을 결합하여 새로운 의미있는 특성을 만드는데 사용됩니다.

<br>🔥 Feature Transformation: 기존의 특성들을 변환하는 과정입니다. 로그 변환(Log Transformation), 스케일링(Scaling), 정규화(Normalization) 등이 이에 해당합니다.

<br>🔥 Dimensionality Reduction: 많은 양의 차원(특성)으로 인해 발생할 수 있는 문제점('차원의 저주')를 해결하기 위해 차원 축소 방법이 사용됩니다.


## Linear Classifier, Linear Regression 모델의 학습원리



### Linear Classifier

    ⚡ 입력 특성의 선형 조합을 기반으로 클래스를 예측하는 분류 모델.

    ⚡ 선형 분류기는 결정 경계(Decision Boundary)가 선, 평면 또는 초평면 형태로 표현되며, 이를 기준으로 데이터를 두 개 이상의 클래스로 분리



- 알고리즘 종류

> 🔥 로지스틱 회귀(Logistic Regression): 로지스틱 회귀는 선형 함수를 사용하여 클래스에 속할 확률을 계산하고, 그 확률에 따라 적절한 클래스로 분류합니다. 로지스틱 회귀의 출력은 시그모이드 함수(Sigmoid Function)를 거치게 되어 0과 1 사이의 값으로 변환되며, 이 값을 확률로 해석할 수 있습니다.

<br>🔥 서포트 벡터 머신(Support Vector Machines, SVM): SVM은 각 클래스 사이의 마진(Margin), 즉 결정 경계와 가장 가까운 데이터 포인트들 사이의 거리가 최대가 되도록 학습합니다. 이렇게 하면 모델이 일반화(generalization)능력을 향상시킬 수 있습니다.



- 모델의 학습원리

<br><br> 1) 초기화: 모델 파라미터인 가중치와 편향을 임의의 값으로 초기화합니다.

<br><br> 2) 예측: 현재 파라미터를 사용하여 학습 데이터에 대한 예측을 수행합니다. 예측은 각 입력 특성에 가중치를 곱한 후, 모두 더하고 편향을 추가함으로써 얻어집니다.

<br><br> 3) 손실 계산: 예측된 클래스와 실제 클래스 간의 차이를 계산하여 손실(Loss)을 산출합니다. 손실 함수로는 주로 로지스틱 손실 함수나 힌지 손실 함수 등이 사용됩니다.

<br><br> 4) 업데이트: 경사 하강법(Gradient Descent)과 같은 최적화 알고리즘을 사용하여 손실 함수를 최소화하는 방향으로 파라미터를 업데이트합니다.

<br><br> 5) 반복: 위 과정들을 반복하여 최적의 가중치와 편향 값을 찾습니다.

<br><br> 6) 검증 및 조정: 검증 데이터셋에 대해 모델 성능을 평가하고, 필요한 경우 학습률, 정규화 파라미터 등 하이퍼파라미터를 조정합니다.



### Linear Regression

    ⚡ 통계학과 머신러닝에서 널리 사용되는 예측 모델.

    ⚡ 종속 변수 y와 하나 이상의 독립 변수 (또는 설명 변수) X 간의 선형 관계를 모델링.

- $y = β_0 + β_1X_1 + β_2X_2 + ... + β_nX_n + ε$

<br> ☝ $y$ : 종속 변수, 예측하려는 대상

<br> ☝ $X_1, X_2, ..., X_n$ : 독립 변수, 예측에 사용되는 특성

<br> ☝ $β_0$ : 절편(intercept), y축과 선이 만나는 지점

<br> ☝ $β_1, ..., β_n$ : 계수(coefficients), 각 독립 변수의 가중치를 나타냄

<br> ☝ $ε$ : 오차 항(error term), 모델이 설명하지 못하는 부분



- 알고리즘 종류

> 🔥 Ordinary Least Squares (OLS): OLS는 가장 기본적인 선형 회귀 알고리즘으로, 종속 변수와 독립 변수 사이의 잔차 제곱합을 최소화하는 파라미터를 찾습니다. 이 방법은 계산이 간단하며 직관적인 해석을 제공하지만, 데이터에 이상치가 있거나 독립 변수 간에 공선성(multicollinearity)이 있는 경우에 취약합니다.

<br>🔥 Ridge Regression (L2 regularization): Ridge 회귀는 OLS에 L2 정규화 항을 추가한 것입니다. 이 방법은 모델의 복잡성을 제한하여 과적합(overfitting)을 방지하며, 독립 변수간의 공선성 문제를 완화할 수 있습니다.

<br>🔥 Lasso Regression (L1 regularization): Lasso 회귀는 OLS에 L1 정규화 항을 추가한 것입니다. 이 방법도 Ridge와 마찬가지로 모델의 복잡성을 제한하지만, 불필요한 특성의 계수를 0으로 만들어서 특성 선택(feature selection) 역할도 수행합니다.

<br>🔥 Elastic Net Regression: Elastic Net은 Ridge와 Lasso를 결합한 형태로, L1 및 L2 정규화 항이 모두 포함되어 있습니다.

<br>🔥 Robust Regression: Robust regression은 이상치(outliers) 또는 높은 leverage points로 인해 일반 선형회귀에서 발생할 수 있는 문제를 완화하기 위해 설계되었습니다.

<br>🔥 Bayesian Linear Regression: Bayesian linear regression은 선형회귀 계수에 대해 사전 분포(prior distribution)를 설정하고 베이지안 추론(Bayesian inference)을 사용하여 계수를 추정합니다.



- 모델의 학습원리

<br> ✅ 주어진 데이터에 가장 잘 맞는 선형 함수를 찾는 것.

<br> ✅ 실제 값과 모델이 예측한 값 사이의 차이(오차)가 최소가 되도록 하는 것.

<br><br> 1) 모델 정의: 선형 회귀 모델은 $y = β_0 + β_1X_1 + β_2X_2 + ... + β_nX_n + ε$와 같은 형태를 가집니다. 여기서 $y$는 종속 변수, $X_n$들은 독립 변수, $β_n$들은 각 독립 변수에 대한 계수, ε는 오차항을 나타냅니다.

<br><br> 2) 손실 함수 정의: 손실 함수(Loss Function) 또는 비용 함수(Cost Function)를 설정합니다. 선형 회귀에서 일반적으로 사용되는 손실 함수로는 **평균 제곱 오차(Mean Squared Error, MSE)**가 있습니다.

<br><br> 3 ) 최적화 알고리즘 선택: 손실 함수를 최소화하는 파라미터($β_0, ..., β_n$)을 찾기 위해 최적화 알고리즘이 필요합니다. 선형회귀에서 널리 사용되는 최적화 방법인 **경사하강법(Gradient Descent)**, **확률적 경사 하강법(Stochastic Gradient Descent)**, **미니 배치 경사 하강법(Mini-Batch Gradient Descent)**, 그리고 **정규 방정식(Normal Equation)** 등이 있습니다.

<br><br> 4) 모델 학습: 선택된 최적화 알고리즘을 사용하여 손실함수를 최소화하는 파라미터 값을 찾습니다. 이 과정에서 데이터셋 내 각 관측치에 대해 예측값을 계산하고 실제값과 비교하여 오차를 구하며, 이 오차를 줄이도록 파라미터 값을 조정해 나갑니다.

<br><br> 5) 모델 검증 및 튜닝: 학습된 모델을 검증 데이터셋에 적용하여 예측 성능을 평가합니다. 필요한 경우 하이퍼파라미터를 조정하거나, 모델의 복잡도를 변경하여 모델을 튜닝합니다.


## Decision Tree(CART) 모델의 학습원리



### Decision Tree란?

    ⚡ 지도 학습 알고리즘 중 하나로, 데이터를 분석하여 이들 사이에 존재하는 패턴을 예측 가능한 규칙들의 조합으로 나타내는데 사용.

    ⚡  규칙들의 구조가 트리 구조와 유사하기 때문에 결정 트리라고 부름.

- 결정 트리 학습 과정

> 🔥 트리 생성: 전체 데이터셋을 대상으로 최적의 분할 기준을 찾아서 첫 번째 노드(루트 노드)를 생성합니다.

<br> 🔥 분할(Division): 선택된 기준에 따라 데이터셋을 두 개 이상의 서브셋(subset)으로 분할합니다.

<br> 🔥 재귀적 반복: 위와 같은 과정을 서브셋에서도 재귀적으로 반복하여 결정 트리를 생성합니다.

<br> 🔥 가지치기(Pruning): 오버피팅 방지를 위해 복잡한 트리에서 일부 가지를 제거하는 과정입니다.

<br> 🔥 예측: 새로운 입력 값이 주어지면, 해당 데이터가 결정 트리 상에서 어떤 경로를 따르는지 확인하고, 그 경로의 말단 노드에 도달했을 때 해당 노드의 값을 예측값으로 사용합니다.



### CART란?

    ⚡ 결정 트리 알고리즘의 한 종류로, 분류(Classification)와 회귀(Regression) 문제 모두에 적용 가능.

    ⚡ 이진 트리(binary tree) 구조를 가지며, 각 노드에서의 분할은 하나의 피처에 대해 이진 분할을 수행.

- CART의 학습원리

> 🔥 데이터 준비: CART 모델은 입력 데이터와 해당하는 출력 값을 필요로 합니다. 입력 데이터는 여러 개의 독립 변수(피처)로 구성되어 있으며, 출력 값은 예측하려는 종속 변수(타겟)입니다.

<br> 🔥 트리 구조 생성: CART 알고리즘은 재귀적 분할을 사용하여 트리를 구성합니다. 초기에는 전체 데이터셋이 루트 노드로 사용됩니다.

<br> 🔥 손실 함수 계산: 각 분할 후에 손실 함수를 계산하여 최적의 분할을 찾습니다. 이 때, 분류 문제에서 일반적으로 Gini 지수(Gini index)를 사용하며, 회귀 문제에서는 평균 제곱 오차(Mean Squared Error)를 사용합니다.

<br> 🔥 최적 분할 선택: 가능한 모든 분할 중에서 손실 함수가 가장 작아지도록 하는 최적의 분할을 선택합니다.

<br> 🔥 재귀적 분할: 선택된 최적 분할에 따라 노드가 자식 노드로 나뉘어집니다. 이 과정은 특정 조건을 만족하거나 더 이상 적절한 분할이 없을 때까지 반복됩니다.

<br> 🔥 종료 조건 검사: 특정 종료 조건(예: 트리 깊이 제한, 노드의 최소 샘플 수 등)을 검사하여 재귀 과정을 중단하고 트리 생성을 완료합니다.

<br> 🔥 가지치기 (Pruning): 생성된 트리가 과도하게 복잡한 경우, 가지치기를 수행하여 일반화 성능을 개선합니다.

<br> 🔥 예측 및 평가: 학습된 CART 모델을 사용하여 새로운 입력값에 대한 예측 결과를 도출하고, 해당 결과와 실제 출력 값을 비교하여 모델의 성능을 평가합니다.







## sklearn의 fit, predict 함수 사용법과 의미 이해하기



### sklearn

    ⚡ 대표적인 머신러닝 라이브러리 중 하나.

    ⚡ 다양한 머신러닝 알고리즘을 제공하며, 이를 사용해 모델 학습, 검증, 예측 등의 작업을 수행.

- 회귀(Regression), 분류(Classification), 서포트 벡터 머신(SVM), 결정 트리(Decision Trees), 랜덤 포레스트(Random Forests) 등 다양한 지도 학습 알고리즘이 구현.

- 클러스터링(Clustering), 주성분 분석(PCA), 가우시안 혼합 모델(Gaussian Mixture Models) 등 다양한 비지도 학습 알고리즘이 구현.

- 피처 스케일링(Feature Scaling), 정규화(Normalization), 인코딩(Encoding) 등 데이터 전처리 작업에 필요한 도구들을 제공.

- 교차 검증(Cross-validation), 그리드 서치(Grid Search) 등 모델 성능 평가 및 하이퍼파라미터 최적화를 위한 도구들이 포함.

- 고차원 데이터에서 유용한 특성(feature)을 추출하거나 선택하는데 사용할 수 있는 도구들이 포함.


### fit 함수의 사용법

- fit 함수 : 주어진 데이터에 대해 머신러닝 모델을 학습시키는 역할.



```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target # Target: species of iris (setosa, versicolor, virginica)

# Create an instance of the model
model = LinearRegression()

# Train the model with data (X: features, y: target)
model.fit(X, y)
```

<pre>
LinearRegression()
</pre>
### predict 함수의 사용법

- predict 함수 : 학습된 모델을 사용하여 새로운 데이터에 대한 예측값을 생성하는 역할을 합니다.



```python
# Assume we have new iris samples
X_new = [[5.1, 3.5, 1.4, 0.2],
         [6.7, 3.1, 4.7, 1.5]]

# Predict with the trained model
# 붓꽃의 품종 예측
'''
'setosa': 0
'versicolor': 1
'virginica': 2
'''
predictions = model.predict(X_new)

print(predictions)
```

<pre>
[-0.08254936  1.30098933]
</pre>
