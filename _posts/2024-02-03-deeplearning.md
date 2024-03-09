---
layout: single
title:  "딥러닝의 모든 것 (추가중)"
categories: deeplearning
tag: [python, deeplearning]
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

# ⭐ 0. 딥러닝 분야 선택

- 내가 AI를 공부하는 이유 = 미래를 예측하고 싶다!
    - 1) 미래의 문제를 예측하고 싶다.
    - 2) 미래의 결과를 예측하고 싶다.
    - 3) 미래의 가능성을 예측하고 싶다.
    - 4) 미래의 수요와 공급을 예측하고 싶다.
    - 결론) 시계열 데이터 분석과 이에 맞는 AI 모델을 공부하자

1. 컴퓨터 비전 (X)
    - CNN
    - Auto Encoder
    - GAN
    - Diffusion Model

2. NLP (X)
    - RNN
    - Transformer
    - CNN(텍스트 데이터에서도 특징 추출에 활용할 수 있음)
    - Bidirectional LSTM
    - 워드2벡터 (Word2Vec), 글로브 (GloVe), FastText
    - GAN

3. 시계열 처리 (O)
    - RNN(LSTM, GRU)
    - Transformer
    - Prophet
    - DeepAR
    - Temporal Convolutional Networks (TCN)
    - Dilated Convolutional Neural Network
    - Auto Encoder(시계열 데이터에서는 주로 차원 축소, 이상 탐지, 잡음 제거)
    - Variational Autoencoder(VAE, 시계열 데이터에서 시간적인 변동성을 반영하여 데이터를 생성하는 데 사용)

4. 강화학습 (O)
    - Q-Learning, Deep Q-Network (DQN)
    - 정책 그래디언트 (Policy Gradient)
    - Proximal Policy Optimization (PPO)
    - 액터-크리틱 (Actor-Critic)
    - 딥 강화학습 알고리즘 (Deep Reinforcement Learning from Human Demonstrations, DQfD)

# ⭐ 1. 딥러닝 이란?
- Machine Learning + Deep Neural Network
- Machine Learning의 train/test, parametric learning, weight update의 개념을 그대로 가지고 감

- DL이 ML보다 좋은 이유는 단! 성능이 좋아서이다!
    - Feature Extraction에서 차이가 많이나기 때문이다.
    - ML은 Feature Engineering을 사람이 직접하지만, DL은 Model이 직접한다.

- DL은 ML보다 더 많은 학습데이터를 필요로 한다.
    - 데이터가 충분하지 않으면 Overfitting되는 문제가 발생

- DL은 ML보다 더 많은 Computing Resource를 필요로 한다.
    - GPU, TPU 하드웨어 사용이 필요하다.

- 정형데이터 분석 vs 비정형데이터 분석(이미지, 음성, 텍스트)
    - 정형데이터는 ML을 사용하는 것이 더 파워풀한 성능을 낼 수도 있다.
    - 비정형데이터는 무조건 DL이 좋다.

## 1) Perceptron
- 인공신경망의 가장 기본적인 형태
- 입력 특성 벡터와 이에 대응하는 가중치 벡터의 선형 조합을 계산
- 그 결과를 임계값과 비교하여 출력을 결정
- Neuron (신경망)
<figure style="text-align: center;">
    <img src="https://miro.medium.com/v2/resize:fit:2902/format:webp/1*hkYlTODpjJgo32DoCOWN5w.png" width="500" height="300">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc</figcaption>
</figure>

- Artificial Neuron (인공신경망)
<figure style="text-align: center;">
    <img src="https://lanstonchu.files.wordpress.com/2021/03/cell.jpeg" width="500" height="300">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://lanstonchu.wordpress.com/2021/09/06/human-neuron-vs-artificial-neuron-similarities-and-discrepancies/</figcaption>
</figure>
    - ✅ **Inputs** : $x_1 ... x_n$, 입력값들.
    - ✅ **Weights** : $w_1 ... w_n$, 가중치값들.
    - ✅ **Bias** : $b$, 편향값.
        - 인공 뉴런의 출력을 조정하며, 특히 인공 뉴런이 얼마나 쉽게 활성화되는지를 결정하는 요소
        - 인공 뉴런의 활성화 임계값을 조정하는 역할을 함
        - 편향이 크면 인공 뉴런은 입력에 대해 덜 민감해지고, 편향이 작으면 더 민감해짐
        - 인공신경망이 데이터의 복잡한 패턴을 더 잘 잡아낼 수 있게 도와주는 역할
        -  역전파 알고리즘을 통해 가중치와 함께 학습됨
    - ✅ **Sum** : $x_1w_1 + ... + x_nw_n + b$, Linear Model.
    - ✅ **Activation Function** : $f(x)$, Sum값이 x값으로 들어옴, 활성함수.
        - 주요 목적은 비선형성(non-linearity)을 인공 신경망에 도입하는 것.
        - 활성화 함수는 또한 신경망의 출력 값을 특정 범위로 조절.
        - 인공 신경망의 각 뉴런에서 입력 값과 가중치의 곱을 합산한 후(가중 합), 그 결과에 적용.
        - 대표적인 활성화 함수로는 시그모이드, ReLU, 하이퍼볼릭 탄젠트
        - Sigmoid
        <figure style="text-align: center;">
            <img src="https://mlnotebook.github.io/img/transferFunctions/sigmoid.png" width="300" height="300">
            <figcaption style="font-size: 0.1em; color: gray;">출처) https://dacon.io/en/forum/406091</figcaption>
        </figure>
        - ReLU
        <figure style="text-align: center;">
            <img src="https://blog.kakaocdn.net/dn/vgJna/btqQzRGmwcO/TK3KTMlz4CYag8rBTKfYkK/img.png" width="300" height="300">
            <figcaption style="font-size: 0.1em; color: gray;">출처) https://limitsinx.tistory.com/40</figcaption>
        </figure>
        - 하이퍼볼릭 탄젠트(tanh)
        <figure style="text-align: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/2560px-Hyperbolic_Tangent.svg.png" width="300" height="300">
            <figcaption style="font-size: 0.1em; color: gray;">출처) https://ko.m.wikipedia.org/wiki/%ED%8C%8C%EC%9D%BC:Hyperbolic_Tangent.svg</figcaption>
        </figure>

    - ✅ **Output** : $\hat{y}$, 예측값
-$...$

## 2) Multi-Layer Perceptron
- 훌륭한 non-linear 모델
- 인공 신경망의 한 종류로, 입력층, 은닉층, 출력층의 세 부분으로 구성
- 세 부분은 모두 인공 뉴런(또는 노드)으로 이루어져 있으며, 각 뉴런은 다른 뉴런과 연결
<figure style="text-align: center;">
    <img src="https://www.dtreg.com/uploaded/pageimg/MLFNwithWeights.jpg" width="500" height="300">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://ailephant.com/glossary/multilayer-perceptron/</figcaption>
</figure>
- 입력층(Input Layer): 입력 데이터가 신경망으로 들어오는 부분
- 은닉층(Hidden Layer)
    - 입력층과 출력층 사이에 위치한 층으로, 하나 이상의 은닉층이 존재할 수 있음
    - 은닉층의 노드는 입력층의 데이터를 받아 처리하고, 그 결과를 다음 층으로 전달
- 출력층(Output Layer)
    - 신경망이 최종적으로 예측하거나 분류를 수행하는 부분
    - 출력층의 노드 수는 문제의 종류(회귀, 이진 분류, 다중 클래스 분류 등)에 따라 달라짐

## 3) Feed-Forward
- = inference
- 인공 신경망에서 정보가 입력층에서 출력층으로 단방향으로 전달되는 과정

## 4) output, loss funtion
1. 회귀
- 🔥 평균 제곱 오차(Mean Squared Error, MSE)
    - 회귀 문제에서는 평균 제곱 오차를 주로 사용
<figure style="text-align: center;">
    <img src="https://images.velog.io/images/rcchun/post/ac220735-2d93-46e0-8812-d9772b191c85/image.png" width="200" height="100">
</figure>

2. 분류
- 🔥 Softmax Funtion
<figure style="text-align: center;">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQGFKh%2FbtqPQtew8NG%2FP5e54TRwt9fZqmXi55866k%2Fimg.jpg" width="300" height="200">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://limitsinx.tistory.com/36</figcaption>
</figure>
    - 딥러닝에서 **다중 클래스 분류 문제**에서 주로 사용되는 활성화 함수(Activation Function)
    - 모델의 출력을 클래스별 확률 분포로 변환하는 역할
    - Softmax function은 **입력값을 지수 함수(exp)로 변환**하고, **모든 클래스에 대한 지수 함수 값의 합으로 나누어** 각 클래스의 확률을 계산
    - Softmax function을 통해 각 클래스의 확률을 계산하면, **가장 확률이 높은 클래스를 선택할 수 있음**
    - Softmax function을 통과한 모든 output 값들의 총합은 1이됨

- 🔥 Cross Entropy
    - 딥러닝에서 **주로 분류 문제**에서 사용되는 손실 함수(Loss Funtion)
    - 모델의 출력과 실제 레이블 사이의 차이를 측정하여 모델을 학습시키는 데 사용
    - **출력 확률 분포와 실제 레이블의 분포 사이의 차이를 측정**
<figure style="text-align: center;">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbDAn5H%2FbtrvlyHcw8i%2FTVK9WlZemCBN85qKKOIR21%2Fimg.png" width="300" height="200">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://lcyking.tistory.com/70</figcaption>
</figure>
<figure style="text-align: center;">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBnF3j%2Fbtrvpa0SZK4%2FhajdvXgTuzFyl1YGHaoVx0%2Fimg.png" width="300" height="200">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://lcyking.tistory.com/70</figcaption>
</figure>
    - 공식 : $H(p,q) = -∑_{i=1}^{n}p(x_i)log(q(x_i))$
    - $p(x_i)$ = 실제 예측 확률 분포, Softmax Function를 사용하여 나온 확률 분포 값을 one-hot encoding한 값
        - [0, 1, 0, 0, 0]
    - $q(x_i)$ = 계산한 예측 확률 분포, Softmax Function를 사용하여 나온 확률 분포 값
        - [0.02, 0.90, 0.05, 0.01, 0.02]
    - $CE = -(0ˣlog(0.02) + 1ˣlog(0.90) + 0ˣlog(0.05) + 0ˣlog(0.01) + 0ˣlog(0.02)) = 0.1053$
    <figure style="text-align: center;">
        <img src="https://i.namu.wiki/i/NE-z84UY1NUg_ASi6ExcveUWxeoWJJHpybv2gY5frfww_fqLLwjYnh_fPvgRXgAcVptAek0PX15SkexkW5e7OA.webp" width="300" height="200">
    </figure>
    - 실제 레이블 확률 = 1, 예측 확률 = 0 ----> CE = -inf 값이 출력
    - 실제 레이블 확률 = 1, 예측 확률 = 1 ----> CE = 0 (아주 이상적인 값)
    - 두 분포의 차이가 클수록 크로스 엔트로피 값은 크게 나타나며, **두 분포가 일치할 때 최소값**을 가짐

- 🔥 이진 교차 엔트로피(Binary Cross-Entropy)
    - 이진 분류 문제에서는 BCE를 주로 사용
    - $BCELoss(\hat{y},y) = -(y𝖷log(\hat{y}) + (1-y)𝖷log(1-\hat{y}))$

- 🔥 범주형 교차 엔트로피(Categorical Cross-Entropy)
    - 다중 클래스 분류 문제에서는 CCE를 주로 사용
    - Cross-Entropy Loss를 사용

- $...$

## 5) back propagation
- 손실 함수의 결과를 이용하여 각 층의 가중치와 편향을 업데이트
- 오차를 역으로 전파하여 각 층의 가중치에 대한 변화량을 계산하고, 경사 하강법(Gradient Descent)을 사용하여 가중치를 조정
- 체인룰(chain rule)을 사용하여 각 층의 가중치와 편향에 대한 기울기를 계산
    - 체인룰 이란, $F=f(g(x))=f∘g$ 에서
    - 1) x가 변화했을 때 함수 g(x)가 얼마나 변하는 지 알 수 있음 (미분이 가능하므로)
    - 2) 함수 g(x)의 변화로 인해 함수 f(x)가 얼마나 변하는 지를 알 수 있음 (미분이 가능하므로)
    - 3) 함수 f(x)의 인자가 함수 g(x)이면
    - ➡ F의 변화량에 기여하는 각 함수 f(x)와 g(x)의 기여도를 알 수 있다는 것

- Learning Rate
    - 가중치와 편향을 업데이트할 때 얼마나 크게 조정할지를 결정하는 하이퍼파라미터
    - 각 매개변수 업데이트 단계에서 변화율(기울기)에 곱해져서 실제 업데이트되는 양을 결정
    - 즉, 학습률은 기울기를 얼마나 반영하여 가중치와 편향을 업데이트할지를 제어

- [설명]
<figure style="text-align: center;">
    <img src="https://i0.wp.com/analyticsarora.com/wp-content/uploads/2021/09/Understand-The-Backpropagation-Algorithm-Interview-Question.png?resize=800%2C600&ssl=1" width="500" height="300">
    <figcaption style="font-size: 0.1em; color: gray;">출처) https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/</figcaption>
</figure>
    - 각 가중치의 Error 값에 대한 기여도 = $ {∂E} \over {∂W}$ = Error를 가중치로 편미분한 값
    - 가중치(W)의 업데이트는 기여도에 Learning Rate 값을 곱한 값을 빼준다.
    - $W_{new} = W_{old} - (LR)𝖷{{∂E} \over {∂W}}$
- $...$

## 6) Training
- Gradient Descent
  - full-batch
  - **모든 데이터**를 학습에 사용
  - 메모리가 아주 많이 소요
- Stochastic Gradient Descent
   - mini-batch
   - **일부 데이터**를 학습에 사용
   - 메모리 절약 가능
- Epoch
  - 전체 훈련 데이터셋을 한 번 모두 사용하여 학습하는 단위를 의미
  - 에포크가 너무 크면 모델이 훈련 데이터에 과적합(overfitting)될 수 있으므로 주의해야 함
- Batch Size
  - 한 번의 업데이트 단계에서 처리되는 데이터 샘플의 수
  - Batch Size는 일반적으로 2^n을 사용 : GPU의 가성비를 높여주는 숫자
- e.g.
  - input 데이터 개수 : 10,000이라고 가정
  - batch_size : 100이라고 가정
  - 그렇다면, 1epoch = 10,000 / 100 = 100 iterations = weight update 횟수