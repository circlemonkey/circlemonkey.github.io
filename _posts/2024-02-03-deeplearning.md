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
    - re) 시계열 데이터 분석과 이에 맞는 AI 모델을 공부!

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
    <figcaption style="font-size: 10px; color: gray;">출처) https://towardsdatascience.com/the-concept-of-artificial-neurons-perceptrons-in-neural-networks-fab22249cbfc</figcaption>
</figure>

- Artificial Neuron (인공신경망)
<figure style="text-align: center;">
    <img src="https://lanstonchu.files.wordpress.com/2021/03/cell.jpeg" width="500" height="300">
    <figcaption style="font-size: 10px; color: gray;">출처) https://lanstonchu.wordpress.com/2021/09/06/human-neuron-vs-artificial-neuron-similarities-and-discrepancies/</figcaption>
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
            <figcaption style="font-size: 10px; color: gray;">출처) https://dacon.io/en/forum/406091</figcaption>
        </figure>
        
        - ReLU
        <figure style="text-align: center;">
            <img src="https://blog.kakaocdn.net/dn/vgJna/btqQzRGmwcO/TK3KTMlz4CYag8rBTKfYkK/img.png" width="300" height="300">
            <figcaption style="font-size: 10px; color: gray;">출처) https://limitsinx.tistory.com/40</figcaption>
        </figure>
        
        - 하이퍼볼릭 탄젠트(tanh)
        <figure style="text-align: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/2560px-Hyperbolic_Tangent.svg.png" width="300" height="300">
            <figcaption style="font-size: 10px; color: gray;">출처) https://ko.m.wikipedia.org/wiki/%ED%8C%8C%EC%9D%BC:Hyperbolic_Tangent.svg</figcaption>
        </figure>

    - ✅ **Output** : $\hat{y}$, 예측값

## 2) Multi-Layer Perceptron

- 훌륭한 non-linear 모델

- 인공 신경망의 한 종류로, 입력층, 은닉층, 출력층의 세 부분으로 구성

- 세 부분은 모두 인공 뉴런(또는 노드)으로 이루어져 있으며, 각 뉴런은 다른 뉴런과 연결

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
    <figcaption style="font-size: 10px; color: gray;">출처) https://limitsinx.tistory.com/36</figcaption>
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
    <figcaption style="font-size: 10px; color: gray;">출처) https://lcyking.tistory.com/70</figcaption>
</figure>

<figure style="text-align: center;">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBnF3j%2Fbtrvpa0SZK4%2FhajdvXgTuzFyl1YGHaoVx0%2Fimg.png" width="300" height="200">
    <figcaption style="font-size: 10px; color: gray;">출처) https://lcyking.tistory.com/70</figcaption>
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
    <figcaption style="font-size: 10px; color: gray;">출처) https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/</figcaption>
</figure>

    - 각 가중치의 Error 값에 대한 기여도 = $ {∂E} \over {∂W}$ = Error를 가중치로 편미분한 값

    - 가중치(W)의 업데이트는 기여도에 Learning Rate 값을 곱한 값을 빼준다.

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

# ⭐ 2. 딥러닝 알고리즘

## 0) MLP
### ① TabNet

## 1) CNN
- CNN (Convolutional Neural Network)
- 이미지 인식, 컴퓨터 비전 등에서 주로 사용되는 신경망 구조
- 합성곱층(Convolutional Layer), 풀링층(Pooling Layer), 완전 연결층(Fully Connected Layer)으로 구성
<figure style="text-align: center;">
    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FclZ8Xf%2FbtrRBKTo9QN%2FnWkmTcac5HA2JE4SsQ6Ow1%2Fimg.png" width="500" height="300">
    <figcaption style="font-size: 10px; color: gray;">출처) https://medium.com/swlh/fully-connected-vs-convolutional-neural-networks-813ca7bc6ee5</figcaption>
</figure>

- 🔥 Convolutional Layer
    - Feature를 추출하는 Layer이기 때문에 Embedding Layer라고 할 수 있음
    - Filter(= Kernel)
        - 필터는 합성곱층에서 입력 이미지에 적용되는 작은 크기의 행렬
        - 일반적으로 정사각형으로 설정
        - 합성곱 연산을 통해 입력 이미지와 곱셈 및 합산되어 특징 맵을 생성
    - Stride
        - 필터가 입력 이미지 위를 이동하는 간격을 의미
        - 큰 스트라이드는 특징 맵의 크기를 줄이고 계산량을 감소시킴
    - Padding
        - 패딩은 입력 이미지 주변에 추가적인 값(일반적으로 0)을 채우는 기법
        - 패딩을 사용하면 입력 이미지의 가장자리에 있는 픽셀들도 충분한 고려를 받을 수 있음
        - [사용목적]
        <br> 1) 출력 특징 맵의 크기를 유지하기 위해 사용
        <br> 2) 경계 픽셀의 정보를 보존하기 위해 사용
    - Feature Map
        - 합성곱층(Convolutional Layer)을 통과한 결과로 생성되는 2D 배열
        - H = ((입력 이미지 높이 + 2 * Padding - Filter의 높이) / Stride) + 1
        - W = ((입력 이미지 너비 + 2 * Padding - Filter의 너비) / Stride) + 1
        - Feature Map의 채널 수 = 사용하는 필터의 개수.
        - (중요) 소수점 아래는 버림
    - 합성곱 연산 과정
        <figure style="text-align: center;">
            <img src="https://www.insilicogen.com/blog/attach/1/1379830475.png" width="350" height="250">
            <figcaption style="font-size: 10px; color: gray;">출처) https://indoml.com/2018/03/07/student-notes-convolutional-neural-networks-cnn-introduction/</figcaption>
        </figure>

        - 입력 이미지와 필터(Filter)를 겹쳐가며 요소별 곱셈을 수행.
        - 필터를 일정한 간격(Stride)만큼 이동시켜 전체 입력 이미지에 대해 합성곱 연산을 반복.
        - 필터의 이동에 따라 새로운 위치에서 새로운 특징 맵(Feature Map)이 생성.
        
- 🔥 Pooling Layer

    - Feature를 추출하는 Layer이기 때문에 Embedding Layer라고 할 수 있음

    - [사용목적]

    <br> 1) 풀링 연산을 통해 특징 맵의 크기를 줄여 계산량을 감소

    <br> 2) 특징 맵 내에서 특정 패턴이 나타나는 위치를 고려하지 않고 가장 중요한 특징만 강조하여 위치 이동에 덜 민감한 특징을 추출

    <br> 3) 작은 변화나 잡음에 민감하지 않도록 하여 특징의 일반화 능력을 향상

    - **최대 풀링(Max Pooling)**이나 **평균 풀링(Average Pooling)**을 사용하여 특징 맵의 부분 영역에서 가장 큰 값이나 평균 값을 추출

    <figure style="text-align: center;">

        <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*BwNCDOi_0BF5Isrt.png" width="350" height="250">

        <figcaption style="font-size: 10px; color: gray;">출처) https://medium.com/@miyachan84/%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-convolutional-neural-networks-5db5c7cb91dc</figcaption>

    </figure>

    - 출력 특징 맵의 크기 = ((입력 특징 맵의 크기 - 풀링 영역의 크기) / 스트라이드) + 1

    - (중요) 소수점 아래는 버림



- 🔥 FC(Fully-connected Layer)

    - 풀링층(Pooling Layer)의 출력을 받아들여 분류 등의 작업을 수행

    <br> (풀링층의 출력을 Flatten하여 쭉~ 펴줌)

    - 이전 층의 모든 뉴런과 현재 층의 모든 뉴런이 연결되어 있는 구조

    - 모든 입력 뉴런과 출력 뉴런 사이에 가중치가 존재하고, 모든 입력에 대해 가중치와 곱셈 연산을 수행한 후, 모든 결과를 더하여 출력을 계산

    <figure style="text-align: center;">

        <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*VHOUViL8dHGfvxCsswPv-Q.png" width="350" height="250">

        <figcaption style="font-size: 10px; color: gray;">출처) https://medium.com/swlh/fully-connected-vs-convolutional-neural-networks-813ca7bc6ee5</figcaption>

    </figure>



- 🔥 e.g. (계산해보기)

<figure style="text-align: center;">

    <img src="https://miro.medium.com/v2/resize:fit:1400/0*u8uSPvkagjmnxiJZ.jpeg" width="500" height="300">

    <figcaption style="font-size: 10px; color: gray;">출처) https://medium.com/@miyachan84/%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-convolutional-neural-networks-5db5c7cb91dc</figcaption>

</figure>



    - 1) Input

        - 입력 이미지(Input)가 흑백이미지 : 채널(Chennel) = 1

        - 입력 이미지(Input)의 크기가 28x28x1(H x W x C)

    - 2) Convolution - (1)

        - 필터(Filter = Kernel)의 크기가 5

        - 필터(Filter = Kernel)의 개수가 n1

        - stride = 1, padding = 0

        - ✅ 출력 Feature Map의 크기 = 24x24xn1(H x W x n1)

            - H = ((28 + 2x0 - 5) / 1) + 1 = 23 + 1 = 24

            - W = ((28 + 2x0 - 5) / 1) + 1 = 23 + 1 = 24

            - C = Filter의 개수 = n1

    - 3) Max-Pooling - (1)

        - (2x2) Max-Pooling, stride = 2

        - ✅ 출력 Feature Map의 크기 = 12x12xn1(H x W x n1)

            - H = (24 - 2 / 2) + 1 = 11 + 1 = 12

            - W = (24 - 2 / 2) + 1 = 11 + 1 = 12

            - ((입력 특징 맵의 크기 - 풀링 영역의 크기) / 스트라이드) + 1

    - 4) Convolution - (2)

        - 필터(Filter = Kernel)의 크기가 5

        - 필터(Filter = Kernel)의 개수가 n2

        - stride = 1, padding = 0

        - ✅ 출력 Feature Map의 크기 = 8x8xn2(H x W x n2)

            - H = ((12 + 2x0 - 5) / 1) + 1 = 7 + 1 = 8

            - W = ((12 + 2x0 - 5) / 1) + 1 = 7 + 1 = 8

            - C = Filter의 개수 = n2

    - 5) Max-Pooling - (1)

        - (2x2) Max-Pooling, stride = 2

        - ✅ 출력 Feature Map의 크기 = 4x4xn2(H x W x n2)

            - H = (8 - 2 / 2) + 1 = 3 + 1 = 4

            - W = (8 - 2 / 2) + 1 = 3 + 1 = 4

    - 6) FC

        - Flatten 해주기

        - fully connected.

        - Activation Function 적용.

    - 7) Output

        - 얻으려는 결과의 개수를 설정하여 얻음.

    


### ① ResNet

- Residual Network

- 2015년에 Kaiming He 등의 연구자들에 의해 소개

- 이미지 분류, 객체 검출, 객체 분할 등 다양한 컴퓨터 비전 작업에 사용

- Human Error(5.4%)를 돌파!

- 주요 아이디어는 잔차 블록(Residual Block)

    - Gradient Update가 잘 안되는 문제 발견!

        - $W_2 ⬅ W_1 - ∇W×α$

        - 가중치가 업데이트 될 때, $∇W$의 값은 계속 작아짐

        - 게다가, α(Leaning Rate)를 계속 곱해주니까 W의 업데이트는 점점 잘 안됨

        - ***Gradient Vanishing*** 문제 발생!!!

    - Gradient 정보를 앞쪽까지 "잘" 전달해주자!

        - Gradient가 skip connection을 통해 바로 전달

        - H(x) - x를 잔차(residual) 라고 함

        - input x에 대한 결과 H(x)에 x를 더한 형태로 구성

        - 동일한 연산을 하고 나서 Input인 x를 더하는 것(Residual Block)과 더하지 않는 것(Plane layer)

        <figure style="text-align: center;">

            <img src="https://i.imgur.com/mHfZYPQ.png" width="350" height="200">

            <figcaption style="font-size: 10px; color: gray;">출처) https://velog.io/@lighthouse97/ResNet%EC%9D%98-%EC%9D%B4%ED%95%B4</figcaption>

        </figure>



- ResNet34, ResNet50, ResNet101, ResNet152(3.5% 달성)

    - 이것들은 Pretrained Model로 사용됨 = "Transfer Learning"

    <figure style="text-align: center;">

        <img src="https://pytorch.kr/assets/images/resnet.png" width="500" height="300">

        <figcaption style="font-size: 10px; color: gray;">출처) https://pytorch.kr/hub/pytorch_vision_resnet/</figcaption>

    </figure>


## 2) RNN

- 🔥 RNN 이란?

<figure style="text-align: center;">

    <img src="https://velog.velcdn.com/images%2Fyuns_u%2Fpost%2Fccbb28ea-fa08-4d23-804e-419e6f578e4b%2Fimage.png" width="700" height="200">

    <figcaption style="font-size: 10px; color: gray;">출처) https://velog.io/@yuns_u/%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9DRNN-Recurrent-Neural-Network</figcaption>

</figure>

    - RNN = Recurrent Neural Network

    - 시퀀스 데이터(예: 문장, 시계열 데이터 등)를 처리하기 위해 설계된 신경망 구조

    - 이전의 정보를 현재의 작업에 활용할 수 있는 능력을 가지고 있음

    - 구성

        - 타임 스탭(Time step, t), 입력층(input layer, $x_t$), 은닉층(hidden layer, $h_t$), 셀(cell), 출력층(output layer, $o_t$)

    - 입력층(input layer)

        - 주어지는 입력 데이터를 받는 층

        - 각 시간 단위에서의 입력이 이 층으로 들어옴

    - 은닉층(hidden layer)

        - RNN에서 시퀀스 데이터의 특징을 추출

        - 이전 시간 단위의 정보를 저장하는 역할

        - ***은닉상태(hidden state)*** : 메모리 셀이 출력층 방향 또는 다음 시점인 t+1의 자신에게 보내는 값

    - 셀(cell)

        - 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 셀(cell)이라고 함

        - 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수행하므로 ***메모리 셀(RNN 셀)***이라고 표현

    - 출력층(ouput layer)

        - 최종적으로 출력되는 값을 제공하는 층

        - 분류 작업에서는 소프트맥스 함수를 사용하여 클래스별 확률을 출력하는 층을 사용할 수 있음



- 🔥 현재 시점(t)에서의 RNN 원리

    - RNN은 은닉층(hidden layer)의 노드에서 활성화 함수(activation function, 'tanh 사용')를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 ***다음 계산의 입력***으로 보냄.

    - 현재 시점 t에서의 메모리 셀이 갖고있는 값은 과거의 메모리 셀들의 값에 영향을 받은 것임

    - t 시점의 메모리 셀은 t-1 시점의 메모리 셀이 보낸 ***은닉 상태값***을 t 시점의 은닉 상태 계산을 위한 입력값으로 사용    

    - (1) 현재시점(t)에서의 은닉 상태값($h_t$), 출력값($y_t$)계산

        <figure style="text-align: center;">

            <img src="http://i.imgur.com/TIdBDTJ.png" width="500" height="300">

            <figcaption style="font-size: 10px; color: gray;">출처) https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/</figcaption>

        </figure>



        - 현재시점(t)에서 입력층의 가중치 $W_{xh}$

        - 이전시점(t-1)의 은닉 상태값 $h_{t-1}$

        - 이전시점(t-1)의 은닉 상태값을 위한 가중치 $W_{hh}$

        - $h_t = tanh(W_{xh}×x_{t} + W_{hh}×h_{t-1} + b_{h})$

        - $y_t = f(W_{hy}×h_{t} + b_{y})$



    - (2) 현재시점(t)의 Input Vector($x_t$)의 차원을 d, 은닉상태의 크기를 $D_h$라고 하면 각 크기는?

        <figure style="text-align: center;">

            <img src="https://wikidocs.net/images/page/22886/rnn_image4_ver2.PNG" width="150" height="100">

        </figure>

        <figure style="text-align: center;">

            <img src="https://wikidocs.net/images/page/22886/rnn_images4-5.PNG" width="500" height="180">

            <figcaption style="font-size: 10px; color: gray;">출처) https://wikidocs.net/22886</figcaption>

        </figure>



        - $x_t$ : (d×1)

        - $W_x$ : ($D_h$×d)

        - $W_h$ : ($D_h$×$D_h$)

        - $h_{t-1}$ : ($D_h$×1)

        - $b$ : ($D_h$×1)

        - $h_{t}$ : ($D_h$×1)



- 🔥 BPTT(오차역전파, BackPropagation Through Time)

    <figure style="text-align: center;">

        <img src="http://i.imgur.com/XYDxsNs.png" width="600" height="400">

    </figure>

    <figure style="text-align: center;">

        <img src="http://i.imgur.com/hEtvXnN.png" width="600" height="400">

        <figcaption style="font-size: 10px; color: gray;">출처) https://www.goldenplanet.co.kr/our_contents/blog?number=857&pn=</figcaption>

    </figure>

        

    - (최종) $dy_t$ = $ {σL} \over {σy_t}$

    - $dW_{hy}$ = $h_t × dy_t$

    - $dh_t$ = $W_{hy} × dy_t$

    - $dh_{raw}$ = $(1-tanh^2(h_{raw})) × dh_t$, ($dh_t$ = ${σL} \over {σh_t}$)

    - [행렬곱]

        - ***$x_t$, $W_{xh}$, $h_{t-1}$, $W_{hh}$에서 일어남***

        - $W_{xh} × dh_{raw}$

        - $x_t × dh_{raw}$

        - $W_{hh} × dh_{raw}$

        - $h_{t-1} × dh_{raw}$

    - ⭐ $h_{t-1}$

        - (t-1시점 Loss) "$W_{hy} × dy_{t-1}$"과 함께

        - (t시점에서 역전파된 값) "★" 이 역전파됨

    - ⭐ 역전파 과정

        - tanh 연산, + 연산, 행렬곱 연산이 일어남

        - 행렬곱 연산 때문에! 기울기 소실, 기울기 폭발이 발생



- 🔥 RNN의 한계점

    - 1) 병렬화 불가능
    <br> - 벡터가 순차적으로 입력
    <br> - sequential 데이터 처리는 가능하게 해주지만,
    <br> - GPU 연산의 장점인 병렬화가 불가능

    - 2) 기울기 폭발(exploding Gradient), 기울기 소실(Vanashing Gradient
    <br> - 역전파 과정에서 치명적인 문제 발생
    <br> - 역전파 과정에서 곱해주는 값이 1미만일 때, n제곱이 된다면 역전파 정보가 거의 전달되지않음
    <br> **= 기울기 소실**
    <br> - 역전파 과정에서 곱해주는 값이 1초과일 때, n제곱이 된다면 역전파 정보가 거의 과하게전달
    <br> **= 기울기 폭발**

- 🔥 One to Many

- 🔥 Many to One

- 🔥 Many to Many

    - seq2seq

    - sos : start of sentense

    - eos : end of sentense


### ① LSTM

- LSTM = Long Short-Term Memory

- 장기 의존성(Long-Term Dependencies)을 학습하고 처리하는 능력을 강화시킨 신경망 구조

- RNN의 그래디언트 소실(Vanishing Gradient) 문제를 해결하고, 긴 시퀀스 데이터에서도 장기 의존성을 효과적으로 학습할 수 있도록 설계

<figure style="text-align: center;">

    <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F999F603E5ACB86A005" width="500" height="300">

    <figcaption style="font-size: 10px; color: gray;">출처) https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr</figcaption>

</figure>



- 🔥 '셀 상태(cell state)'와 '게이트(gate)' 메커니즘

    - 셀 상태(Cell State)

    <figure style="text-align: center;">
        <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99CB87505ACB86A00F" width="500" height="200">
    </figure>

        - 셀 상태는 이전 시간 단위에서 현재 시간 단위로 전달되며, 장기적인 정보를 저장하는 역할

        - [Cell State는 마치 컨베이어 벨트]
            - $C_{t-1}$ ---------------------$C_t$---------------------$> $C_{t+1}$

        - [Cell State는 두 번의 변화를 겪음]
            <figure style="text-align: center;">
                <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F997589405ACB86A00C" width="500" height="200">
            </figure>

            - 망각게이트의 출력값($f_t$)은 곱하기

            - 입력게이트의 출력값($i_t×\tilde{C_t}$)은 더하기

            - $C_t = C_{t-1}×f_t + i_t×\tilde{C_t}$



    - 망각 게이트(Forget Gate)

    <figure style="text-align: center;">
        <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9957DB445ACB86A021" width="500" height="200">
    </figure>

        - 셀 상태의 정보를 지울 것인지 말 것인지를 결정

        - [Activation Function]
            - Sigmoid 함수를 사용(0 ~ 1사이의 값을 가지게 됨)
            - 0 : 정보를 지워버려라, 1 : 모든 정보를 보존해라

        - [입력값]
            - $h_{t-1}$ = (t-1) 시점의 출력값
            - $x_t$ = (t) 시점의 입력값

        - [출력값]
        <br> ✅ $f_t = σ(W_{hf}h_{t-1} + W_{xf}x_t)$

    - 입력 게이트(Input Gate)

    <figure style="text-align: center;">
        <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99D969495ACB86A00B" width="500" height="200">
    </figure>

        - 들어오는 새로운 정보 중 어떤 것을 Cell State에 저장할 것인지 결정.
            - 새롭게 기억해야 할 정보를 추가하자 ➡ tanh
            - 새로운 기억 셀 안에서도 적절히 취사선택하자 ➡ sigmoid

        - [Activation Fucntion]
            - Sigmoid 함수 : 어떤 값을 업데이트할 지 정함
            - tanh 함수 : 새로운 vector를 만듬

        - [입력값]
            - $h_{t-1}$ = (t-1) 시점의 출력값
            - $x_t$ = (t) 시점의 입력값

        - [출력값]
        <br> ✅ $i_t = σ(W_{hi}h_{t-1} + W_{xi}x_t)$
        <br> ✅ $\tilde{C_t} = tanh(W_{hc}h_{t-1} + W_{xc}x_t)$


    - 출력 게이트(Output Gate)

    <figure style="text-align: center;">
        <img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99FB824C5ACB86A10D" width="500" height="200">
    </figure>

        - 현재 시간 단위의 출력을 얼마나 셀 상태와 관련시킬지를 결정하는 역할

        - [Activation Function]
            - Sigmoid 함수
            - tanh 함수

        - [입력값]
            - $h_{t-1}$ = (t-1) 시점의 출력값
            - $x_t$ = (t) 시점의 입력값
            - $c_t$ = (t) 시점의 Cell State값

        - [출력값]
        <br> ✅  $O_t = σ(W_{ho}h_{t-1} + W_{xo}x_t)$
        <br> ✅  (최종) $h_t = O_t×tanh(c_t)$


- 🔥 LSTM의 역전파

<figure style="text-align: center;">
    <img src="https://qph.cf2.quoracdn.net/main-qimg-9c5235ffc1faf177cf155f4601836c74-pjlq" width="500" height="300">
    <figcaption style="font-size: 10px; color: gray;">출처) https://www.quora.com/How-do-LSTMs-solve-the-vanishing-gradient-problem</figcaption>
</figure>

    - Cell State의 역전파
        - ⨁ : 미분값을 건드리지 않고 그대로 흘려보냄
        - ⨂ : 원소별 곱 ➡ (중요) 행렬곱이 아니다!


- 🔥 Multi-Layer LSTM


- 🔥 Bidirectional LSTM



### ② Transformer

## 3) Auto Encoder

- 비지도학습.

- 라벨링 없이도 입력 데이터의 밀집 표현을 학습할 수 있는 신경망.

- 일반적으로 입력 데이터보다 훨씬 낮은 차원을 가지므로 차원 축소 혹은 시각화에 사용됨.

- 밀집 표현을 학습하고 생성 모델로 활용.

- 반드시 입력 변수의 수보다 은닉 노드의 수가 더 적은 은닉 층이 있어야함.

    - 이층에서 정보의 축약이 이루어짐.



- 🔥 활용방식

    - 1) ***차원 축소***의 목적으로 AE를 학습시킨 뒤 Latent Vector를 다른 Machine Learning 모형의 Input으로 사용

    - 2) 입력 정보와 AE의 출력 정보간 차이를 이용한 분석 (***이상상태 탐지***)


- 🔥 Encoder, Decoder

    - Encoder

        - $h(x)$ = $g(a(x))$ = $sigm(b+Wx)$

    - Decoder

        - $\hat{x}$ = $o(\hat{a}(x))$ = $sigm(c + W×h(x))$


- 🔥 한계점

    - 입력에 대한 약간의 변형(Small Perturbations)에도 모델이 민감하게 반응함



- 🔥 Convolutional Auto-Encoder

    - 고려사항 : Decoder 학습 시 feature map의 크기를 어떻게 증가시킬 것 인가?

    - Unpooling

        - Max pooling 을 사용할 경우 해당 위치를 기억해 두었다가 그 "위치 정보"를 사용

        - 기억해둔 위치에 값을 넣고 나머지 부분은 "0"으로 채운다.

    - Transpose convolution



- 🔥 RNN Auto-Encoder

    - 순차 데이터를 복원하는 오토 인코더



## 4) GAN

- GAN = Generative Adversarial Network, 생성적 적대 신경망

- 비지도학습

- 초해상도, 이미지를 컬러로 바꾸기, 강력한 이미지 편집(간단한 스케치 등)

- 밀집 표현을 학습하고 생성 모델로 활용


## 5) 강화학습

- 사전 정보가 전혀 없는, 즉 데이터가 전혀 주어지지 않는 제로베이스 상태에서 학습을 시작해서 스스로 최적의 알고리즘을 찾아내는 학습



- 에이전트 (Agent)

    - ex) 로봇

    - 학습하는 시스템, 그 자체

- 환경 (Enviorment)

    - ex) 여러 지형

    - 로봇이 학습해 나가야 하는 주변 환경

    - 초기에 로봇은 환경에 대한 정보가 아무것도 없고, 점차 알아나가야한다.

- 행동 (Action)

    - ex) 로봇이 취하는 여러 행동

    - 처음에는 아무 사전 학습이 되어있지 않을 때는 랜덤으로 매 순간의 행동을 선택할 수 있다.

- 보상 (Reward)

    - ex) 로봇이 한 행동에 대해서 받는 피드백

    - 안전한 지형에 발을 딛었을 경우 보상으로 양(+)의 점수를 주고,

    - 위험한 지형에 발을 딛었을 경우 음(-)의 패널티를 받도록 알고리즘을 짬

- 정책 (Policy)

    - ex) 로봇이 학습하는 전략

    - '행동선택 -> 보상받기 -> 보상에 따라 행동 수정'의 과정을 거치면서, 보상이 최대화되는 행동의 일련 과정을 학습하게 됨

    - 신경망 정책

        - 관측을 입력으로 받고 실행할 행동을 출력함

        - 각 행동에 대한 확률을 추정하고, 추정된 확률에 따라 랜덤하게 행동을 선택

    - OpenAI짐

        - 다양한 종류의 시뮬레이션 환경을 제공하는 툴킷

    - 신용 할당 문제

        - 에이전트가 보상을 받았을 때 어떤 행동 때문에 보상을 받은 것인지 정확히 알 수 없음


### ① 정책 그레디언트

- 높은 보상을 얻는 방향의 그래디언트를 따르도록 정책 파라미터를 최적화하는 알고리즘

- 가장 있기 있는 것은 REINFORCE 알고리즘


### ② 마르코프 연쇄

- = Markov Chain

- 메모리가 없는 확률 과정 (Stochastic Process)

- 마르코프 결정 과정 (MDP : Markov Decision Process)

- 벨만 최적 방정식 (Bellman Optimality Equation)





- 시간차 학습 (TD 학습 : Temporal Difference Learning)


### ③ Q-Learning

- 전이 확률과 보상을 초기에 알지 못한 상황에서 Q-가치 반복 알고리즘을 적용한 것

- 가치 반복 알고리즘 (Value Iteration Algorithm)

- Q-가치 반복 알고리즘 (Q-Value Iteration Algorithm)

- Q-Learning 알고리즘

    - off-policy vs on-policy


### ④ 심층 Q-Learning

- Q-러닝의 주요 문제

    - 많은 상태와 행동을 가진 대규모 (또는 중간규모)의 MDP에 적용하기 어렵다

- Q-가치를 추정하기 위해 사용하는 DNN을 심층 Q-네트워크(DQN)라 함

- 근사 Q-러닝을 위해 DQN을 사용하는 것을 심층 Q-러닝이라 함





# ⭐ 3. 실습

- tensorflow : https://www.tensorflow.org/guide?hl=ko

- keras : https://keras.io/about/

- pytorch : https://pytorch.org/docs/stable/index.html


## 0) 텐서 개념 및 연산



```python
import torch
```


```python
# 5x3 matrix 생성하기
matrix = torch.empty((5,3))
matrix
```

<pre>
tensor([[ 7.0065e-44,  6.9757e-42, -9.1240e+15],
        [ 1.1362e+30,  7.1547e+22,  4.5828e+30],
        [ 9.2065e-43,  0.0000e+00, -6.9878e-12],
        [ 4.5200e-41, -6.9878e-12,  4.5200e-41],
        [ 2.5353e+30,  3.6434e-44,  3.3491e-43]])
</pre>

```python
# 랜덤하게 초기화된 5x3 matrix 생성하기
matrix = torch.rand(5,3)
matrix
```

<pre>
tensor([[0.5051, 0.1222, 0.0023],
        [0.2738, 0.3151, 0.5304],
        [0.6350, 0.2897, 0.3284],
        [0.3748, 0.5360, 0.4069],
        [0.4091, 0.4340, 0.7416]])
</pre>

```python
# 0으로 채워진 5x3 matrix 생성
default_matrix = torch.zeros(5,3)
print(default_matrix)
print(default_matrix.dtype)
```

<pre>
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32
</pre>

```python
# matrix의 type을 지정하여 생성
long_matrix = torch.zeros(5,3, dtype=torch.long)
print(long_matrix)
print(long_matrix.dtype)
```

<pre>
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
torch.int64
</pre>

```python
# list ----> tensor
data = list([3,4])
print(data)
print(type(data))

tensor = torch.tensor(data)
print(tensor)
print(type(tensor))
```

<pre>
[3, 4]
<class 'list'>
tensor([3, 4])
<class 'torch.Tensor'>
</pre>

```python
# numpy ----> tensor
import numpy as np
array = np.array([1,2,3])
print(array)
print(type(array))

tensor = torch.tensor(array)
print(tensor)
print(type(tensor))
```

<pre>
[1 2 3]
<class 'numpy.ndarray'>
tensor([1, 2, 3])
<class 'torch.Tensor'>
</pre>
🦣 Tensor의 연산



```python
torch.manual_seed(0)
x = torch.rand(5,3)
y = torch.rand(5,3)
```


```python
print(torch.add(x,y)) # 더하기
print(torch.sub(x,y)) # 빼기
print(torch.mul(x,y)) # 곱하기
print(torch.div(x,y)) # 나누기
```

<pre>
tensor([[1.0148, 1.4659, 0.8885],
        [0.2931, 0.5897, 1.3157],
        [1.4053, 1.2935, 1.3298],
        [1.0517, 0.9018, 1.3545],
        [0.0585, 0.3541, 0.6673]])
tensor([[-0.0223,  0.0706, -0.7115],
        [-0.0290,  0.0252, -0.0475],
        [-0.4251,  0.4993, -0.4185],
        [ 0.2129, -0.2040, -0.5510],
        [-0.0138, -0.0164, -0.0795]])
tensor([[0.2573, 0.5360, 0.0708],
        [0.0213, 0.0868, 0.4322],
        [0.4485, 0.3560, 0.3983],
        [0.2652, 0.1929, 0.3827],
        [0.0008, 0.0313, 0.1097]])
tensor([[0.9571, 1.1011, 0.1106],
        [0.8199, 1.0891, 0.9303],
        [0.5355, 2.2575, 0.5212],
        [1.5076, 0.6310, 0.4216],
        [0.6173, 0.9116, 0.7870]])
</pre>
## 1) MLP, CNN (1)



```python
!pip install torchinfo
```

**1. 필요한 라이브러리 가져오기**



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
from time import time
import os
from tqdm.auto import tqdm

# 파이토치 라이브러리와 필요한 모듈들을 불러옵니다.
import torch
import torchvision
from torchinfo import summary
```


```python
seed = 2023
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU 사용시 필요한 코드
torch.cuda.manual_seed_all(seed) # cuda : GPU와 연관
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
```


```python
!nvidia-smi
```


```python
# 버전 체크하기
print(torch.__version__)
print(torchvision.__version__)
print(os.cpu_count()) # number of CPU cores
```

**2. 전처리**



```python
# 이미지 전처리에 필요한 transformation 함수를 정의 (pipeline 형태)
from torchvision.transforms import transforms

# transform = transforms.Compose([
#     transforms.ToTensor(), # numpy.array 등을 torch.FloatTensor로 변환 & 흑백 사진 [0, 255] -> [0.0, 1.0]으로 정규화
#     transforms.Normalize(mean=0.5, std=0.5) # standard scaling [0.0, 1.0] - 0.5 --> [-0.5, +0.5] / 0.5 --> [-1.0, 1.0]
# ])

# for CIFAR10 (RGB channel)
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.5,0.5,0.5),
                       std=(0.5,0.5,0.5))
])

# for ImageNet (RGB channel)
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.485, 0.456, 0.406), # ImageNet 데이터의 RGB channel별 mean, std.
                       std=(0.229, 0.224, 0.225))
])
```


```python
# torch에서 학습에 사용할 config variable들을 세팅합니다.
num_workers = 2
batch_size = 256
learning_rate = 1e-3 # 0.1 ~ 0.000001 (1e-1 ~ 1e-6)
epochs = 10
```

**3. 데이터 나누기**



```python
# Data Load (DataSet, DataLoader)
# trainset = torchvision.datasets.MNIST(root='./',
#                                       train=True, # trainset을 가져옴. (60000장)
#                                       download=True,
#                                       transform=transform)
# testset = torchvision.datasets.MNIST(root='./',
#                                      train=False, # testset (10000장)
#                                      download=True,
#                                      transform=transform
#                                      )
trainset = torchvision.datasets.CIFAR10(root='./',
                                        train=True,
                                        download=True,
                                        transform=transform)
testset = torchvision.datasets.CIFAR10(root='./',
                                       train=False,
                                       download=True,
                                       transform=transform)

# iterator의 역할을 할 수 있음.
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=batch_size, # for문을 던지면 데이터가 batch_size 단위로 뽑혀나옴
                                          shuffle=True, # batch_size 단위로 뽑힐 때 데이터가 섞여서 뽑힘
                                          num_workers=num_workers
                                          )
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=False, # 테스트는 섞을 필요가 없다!
                                         num_workers=2
                                         )
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```


```python
# show image
def imshow(img):
  img = img/2 + 0.5 # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

# iterator를 이용해서 데이터를 불러오게 만듭니다. (batch processing을 위해서)
dataiter = iter(trainloader) # trainloader를 iterator로 선언.
images, labels = next(dataiter) # next 함수는 iterator의 반복 수행함.

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))
```


```python
# shape을 확인
print(images.shape, labels.shape)
# torch.Tensor (batch_size, channel, H, W) ## (image 한정) 4차원 tensor.
```

**4. 모델 구현**



```python
import torch.nn as nn

# 모델 구현에 필요한 레이더들을 정의
class MLP(nn.Module):
  def __init__(self): # class constructor ## 모델 정의 (define layers)
    super().__init__()
    # 3 Layer-NN : (input, hidden1, hidden2)
    # nn.Linear의 in_features는 input node 개수, out_features는 output node 개수
    self.fc1 = nn.Linear(in_features=1*28*28, out_features=512) # out_features는 10~28*28 범위 내에서 2의 거듭제곱으로 정하기 # input_layer -> hidden_layer1
    self.fc2 = nn.Linear(512, 128) # hidden_layer1 -> hidden_layer2
    self.fc3 = nn.Linear(128, 10) # hidden_layer2 -> output_layer(0~9의 숫자이므로 10개)
    self.relu = nn.ReLU() # activation layer
    self.softmax = nn.Softmax(dim=1) # output function # dim의 의미 : 결과가 (batch_size, output_layer_size) 크기로 들어올 텐데, 각 벡터 별로 softmax를 적용하기 위해서 dim=1로 적용

  def forward(self, x):
    # feed-forward 연산을 구현
    # x = (bs, 1, 28, 28) --> (bs, 1*28*28)
    x = torch.flatten(x, 1) # = torch.flatten(input=x, start_dim=1, end_dim=-1) # => 784차원 vector (bias 포함하면 785차원)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.softmax(x)
    return x

# Conv -> ReLU -> Conv -> ReLU -> MaxPool -> fc1 -> fc2 -> fc3(output)
class CNN(nn.Module):
  def __init__(self):
    # X = (N-F) / S+1

    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3,
                           out_channels=6, # number of filters
                           kernel_size=3, # = (3,3)
                           stride=1,
                           padding=0) # (3,32,32) ---> (6,30,30)
    self.conv2 = nn.Conv2d(6, 12, 3) # (6,30,30) --> (12,28,28)
    self.maxpool = nn.MaxPool2d(kernel_size=2,
                                stride=2) # (12,28,28) --> (12,14,14)
    self.fc1 = nn.Linear(in_features=12*14*14, # 2352
                         out_features=1024)
    self.fc2 = nn.Linear(1024, 128)
    self.fc3 = nn.Linear(128, 10)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = torch.flatten(x, 1) # (batch_size, 12,14,14) --> (batch_size, 12*14*14)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.softmax(self.fc3(x))
    return x
```


```python
# device를 지정하기 위함
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```


```python
#model = MLP() # 메인메모리에 실행
#model = MLP().to(device) # GPU에 복사
model = CNN().to(device)
model
```


```python
'''
1290 -> 1290개의 weight ((128+1) * 10) : 1은 bias
65,664 -> 512 * 128 = 65,536 -> (512+1) * 128 = 65,664
401,920 -> (784 + 1) * 512
'''
summary(model)
```


```python
# show params
for x in model.parameters():
  print(x.shape) # real parameters
```

**5. Optimizer와 Loss function 정의**



```python
import torch.optim as optim

# optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
```

**6. 학습**



```python
start = time()

# for문을 이용하여 epoch마다 학습을 수행하는 코드를 작성
# 1 epoch : 전체 데이터를 다 학습시킨 경우
# 1 iteration : 1 weight update
# 1 epoch : batch_size * iterations
# total iterations = epoch(5) * iterations(469)


# epoch, iterations, step
# 데이터를 직접 로드해서, 직접 모델에 넣고, 직접 loss를 계산 (자동으로 loss update)
# mini-batch training!!!!
for epoch in tqdm(range(epochs)):
  n_correct = 0
  total_loss = 0.0
  for idx, data in enumerate(trainloader):
    optimizer.zero_grad() # gradient 초기화

    #### feed forward ####
    #images, labels = data[0], data[1] # CPU version
    images, labels = data[0].to(device), data[1].to(device) # GPU version => 무조건 tensor 형식으로 보내야됨

    outputs = model(images) # forward(x)와 같음
    loss = criterion(outputs, labels) # compute loss
    #print(f'Epoch {epoch} : {idx:4d} iteration --> \t{loss.item():.4f}') # loss value
    n_correct = n_correct + (torch.max(model(images), dim=1)[1] == labels).sum() # batch 당 맞은 개수
    total_loss += loss.item()

    #### backprop ####
    loss.backward() # loss를 가지고 backprop
    optimizer.step() # SGD를 이용해서 weight update를 수행함
  print(f'Epoch {epoch}, Train Accuracy : {n_correct/len(trainset):4f} \t Train Loss : {total_loss/len(trainloader):4f}') # epoch 당 맞은 정확도, epoch당 loss 평균
end = time()

print("Training Done.")
print(f'Elasped Time : {end-start:.4f} secs.')
```

**7. 예측**



```python
# make prediction for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# weight update를 하지 않는 모드 (inference only)
with torch.no_grad():  # test time 때, local gradient를 따로 저장하지 않음. (속도, 메모리) => feed forward를 저장하지 않음(back_propagation을 안할거니까)
    ## TO-DO ##
    ## testloader를 이용해서 학습이 완료된 모델에 대해 test accuracy를 계산해보세요.
    n_correct = 0
    total_loss = 0.0

    for idx, data in enumerate(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = torch.max(outputs, dim=1)[1]
        n_correct += (preds == labels).sum()
        total_loss += loss.item()

    print(f"Test Accuracy : {n_correct/len(testset):4f} | Test (average)Loss : {total_loss/len(testloader):4f}")


# for classname, correct_count in correct_pred.items():
#     accuracy = 100 * float(correct_count) / total_pred[classname]
#     print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
```

## 2) LSTM (1)


**1. 라이브러리 불러오기 및 설치, 데이터 수집**



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
```


```python
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```


```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```


```python
!pip install finance-datareader torchinfo
```


```python
# 삼성전자(005930) 전체
index = '005930'
samsung = fdr.DataReader(symbol=index, start='1990-01-01', end='2023-12-31')
samsung
```

**2. 데이터 전처리**



```python
###### 분류 시 변경 ######
# # label 만들기
# samsung['Label'] = (samsung['Change'] >= 0) * 1
###### 분류 시 변경 ######
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
# 스케일 후 columns
df = samsung.loc[samsung.index > '20200101', scale_cols]
scaled = scaler.fit_transform(df)
df = pd.DataFrame(scaled, columns=scale_cols, index=df.index)
```

**3. train/test 분할**



```python
train = df.loc[df.index < '20230101']
test = df.loc[df.index > '20230101']

###### 분류 모델 시 ######
# X_train, y_train = train.drop(["Close","Label"], axis=1), train.Close
# X_test, y_test = test.drop("Label", axis=1), test.Close

###### 회귀 모델 시 ######
X_train, y_train = train.drop("Close", axis=1), train.Close
X_test, y_test = test.drop("Close", axis=1), test.Close

# 출력
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
```

**4. Data Preparation**



```python
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

seq_length = 30
batch_size = 32

# 데이터셋 생성 함수
def build_dataset(X, y, seq_length):

    X_data = []
    y_data = []

    for idx in range(0, len(X)-seq_length):
        _X = X[idx:idx+seq_length] # 30개 feature vectors
        _y = y[idx+seq_length] # 1개 (31번째 target value, close)
        X_data.append(_X.values)
        y_data.append(_y)
        #print(_X, '--->', _y)

    X_data = torch.FloatTensor(np.array(X_data))
    y_data = torch.FloatTensor(np.array(y_data))
    return X_data, y_data

trainX, trainY = build_dataset(X_train, y_train, seq_length)
testX, testY = build_dataset(X_test, y_test, seq_length)

# 1) 데이터셋 정의
trainset = TensorDataset(trainX, trainY)
testset = TensorDataset(testX, testY)

# 2) 데이터로더정의
# => 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True)#,
                         #drop_last=True) # 데이터로더를 가져올 때 마지막에 배치사이즈랑 맞지않은 것은 drop 하겠다.

testloader = DataLoader(testset,
                        batch_size=batch_size,
                        shuffle=False)#,
                        #drop_last=True)
```

**5. Model 구현**



```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```


```python
class LSTM(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim      # input feature vector input_dim                         #
        self.hidden_dim = hidden_dim    # hidden layer node 개수 (= hidden state dim)            #
        self.seq_len = seq_len          # hidden state 개수 (=input sequence length)             #
        self.output_dim = output_dim    # output layer의 node 개수 (=output dim)                 #
        self.n_layers = n_layers        # multi-layer로 구성할 때, (hidden, LSTM)layer 수        # 멀티 레이어로 구성할 때 LSTM을 몇 개 쓸 것 이냐?

        self.lstm = nn.LSTM(input_size=input_dim, # Wxh   (input_dim x hidden_dim)
                            hidden_size=hidden_dim, # Whh (hidden_dim x hidden_dim)
                            num_layers=n_layers,
                            batch_first=True, # 맨 앞에 있는 숫자가 batch_size구나 라고 알려줌
                            dropout=0.1, # hidden layer의 node중에 일부를 deactivate 시킴. (1 layer일 때는 안쓰는게 좋음.) # 0.1일 땐, hidden layer 10개 중에서 1개를 iterator마다 학습에서 제외함
                            bidirectional=False)  # (batch_size, ~~~~) # NLP할 때 사용할 듯

        # multi layer를 사용한다면! (: input_dim = 4, hidden_dim = 10)
        # LSTM1 = Wxh (4x10)
        # LSTM2 = Wxh (10x10)
        # LSTM3 = Wxh (10x10)
        # ...

        # fc는 한개, 두개 모두 사용할 수 있다.
        self.fc = nn.Linear(in_features=hidden_dim,
                            # out_features=output_dim) # fc 한개 사용 시.
                            out_features=5) # fc 두개 사용 시.
        self.fc2 = nn.Linear(5, output_dim)
        self.relu = nn.ReLU() # non-linear하게 예측하겠다.

        ###### 분류 시 변경 ######
        #self.output = nn.Sigmoid()   # if, output_dim = 1
        #self.output = nn.Softmax()   # if, output_dim = 2
        ###### 분류 시 변경 ######

    # 예측을 위한 함수
    def forward(self, x):  # (N, L, H_in)
        # h0, c0를 zero벡터로 쓸 필요 X -> 알아서 해줌 -> h0, c0를 제로벡터가 아닌 다른 것으로 선언 하려면 이렇게 하기
        # (n_layers, batch_size, hidden_dim)
        # h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        # c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        # x, (hn, cn) = self.lstm(x, (h0, c0)) # (N, L, H_in) ---(LSTM)---> (N, L, H_out) # (32, 30, 4) ----> (32, 30, 10) = (batch_size, seq_len, hidden_dim)

        # input : (x, (h0, c0)) ---> output : (x, (hn, cn))
        x, _ = self.lstm(x)
        x = x[:, -1, :] # (32, 10) = (batch_size, hidden_dim)
        x = self.fc(x) # (32, 10) -> (32,5)= (batch_size, output_dim)  # Linear Regression
        x = self.fc2(self.relu(x)) # (32,5) -> (32,1) # non-Linear Regression
        ## fc2에 relu를 안씌우는 이유 => 그러면 예측값으로 음수를 가질 수 없다.

        ###### 분류 시 변경 ######
        # x = self.sigmoid(x)  # 1차원, Logistic Regression
        # x = self.softmax(x)  # 2차원 이상, Logistic Regression
        # x = x.view(-1, ) # make 2d tensor to 1d tensor.
        ###### 분류 시 변경 ######

        return x
        # return x, (hn, cn)
```


```python
tX, ty = next(iter(trainloader))
tX.shape, ty.shape # (batch_size, seq_len, input_dim) # (N, L, H_in)
```


```python
# 설정값
input_dim = 4
hidden_dim = 10
output_dim = 1
learning_rate = 1e-5
n_epochs = 10000
n_layers = 1
```


```python
# define model
model = LSTM(input_dim=input_dim,
             hidden_dim=hidden_dim,
             seq_len=seq_length,
             output_dim=output_dim,
             n_layers=n_layers).to(device)
model
```


```python
from torchinfo import summary
summary(model)
```

**6. Loss 줄이기**



```python
# regression
criterion = nn.MSELoss()   ## 회귀 시 Loss
#criterion = nn.BCELoss()  ## 분류 시 Loss  # for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```


```python
# title 기본 제목 텍스트
from tqdm.auto import tqdm

## training 함수 구현 ##
# 1. epoch당 avg_loss
# 2. early stopping 구현 --> validation loss를 계산해야됨 --> validation loss가 얼마 이상 떨어지지 않을 때 stop하기.
#                                                         --> 모델이 작을 때는 epoch 단위로 (epoch 당 validation의 avg_loss가 "tol(e.g. 1e-6 : 얼마나 참을거냐) * n번(patience : 몇번 참을거냐)" 이상 떨어지지 않을 때)
#                                                         --> 모델이 클 때는 iteration 단위로
# 3. loss graph 출력

best_val_loss = 123456789.0 # worst value를 넣어놔야 됨
tol = 1e-6 # 얼마나 참을거냐
patience = 5 # 몇 번 참을거냐
patience_count = 0 # 참은 횟수 => patience보다 커지면 early stopping

# loss graph 출력을 위함
train_losses = []
val_losses = []
epoch_end = 0

for epoch in tqdm(range(n_epochs)):
    train_loss = 0.0 # train_loss의 평균을 구하기 위함
    val_loss = 0.0 # val_loss의 평균을 구하기 위함

    # training
    model.train() #  model.eval()에서 다시 training mode로 바꿔줌
    for idx, data in enumerate(trainloader):
        X, y = data[0].to(device), data[1].to(device)
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad() # backprop할 때, gradient를 초기화 해줌
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

    train_loss = train_loss / len(trainloader) # epoch 당 train_loss의 평균
    train_losses.append(train_loss)

    # 1000번 마다 찍기
    if (epoch % 1000) == 0:
        print(f"Epoch : {epoch} ---> Train Loss : {train_loss:.4f}")


    # validation
    model.eval() # predict mode로 바뀜 ==> backprop을 하지 않는 모드
                # dropout, batchnorm 같은 train/predict에서 다르게 동작하는 모듈을 전환해주는 함수.)
    with torch.no_grad(): # locally disabling gradient computation = 이걸 써줘야 gradient update를 안해준다!
                            # 즉, model.eval()과 with torch.no_grad() 둘 다 써줘야됨
        for idx, data in enumerate(testloader):
            # 원래는 train 데이터에서 validation을 잘라서 써야되는데... 여기서는 그냥 test 데이터에서 해봄
            # (시계열데이터에서 정석) test의 기간만큼 validation을 train에서 잘라서 validation loss를 계산하기
            X, y = data[0].to(device), data[1].to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss = val_loss + loss.item()
        val_loss = val_loss / len(testloader)
        val_losses.append(val_loss)

        # 1000번 마다 찍기
        if (epoch % 1000) == 0:
            print(f"Epoch : {epoch} ---> Validation Loss : {val_loss:.4f}")

    # check best : best인지 아닌지 확인
    if best_val_loss - val_loss > tol:
        best_val_loss = val_loss
        print(f"Epoch : {epoch} ---> Best Validation Loss : {best_val_loss:.4f}")
        patience_count = 0
    else:
        patience_count += 1

    if patience_count > patience:
        print(f"Early stopping at {epoch:4d} epoch.")
        epoch_end = epoch
        break
```


```python
plt.figure(figsize=(10, 4))
sns.lineplot(x=list(range(epoch_end+1)), y=train_losses)
plt.figure(figsize=(10, 4))
sns.lineplot(x=list(range(epoch_end+1)), y=val_losses)
```

**7. 예측 데이터 시각화**



```python
# scaler한 것을 원본으로 만드는 방법
original_df = scaler.inverse_transform(df)
original_df = pd.DataFrame(data=original_df, columns=df.columns, index=df.index)
original_df
```


```python
# test데이터 예측 하기
predictions = []

model.eval()
with torch.no_grad():
    for data in testloader:
        X, y = data[0].to(device), data[1].to(device)
        outputs = model(X)
        predictions = predictions + outputs.to('cpu').view(-1, ).tolist()

predictions
```


```python
## 예측한 test 데이터를 inverse scaling 하기
# close의 min, max
y_min, y_max = scaler.data_min_[-1], scaler.data_max_[-1]

# min-max scaling
# x' = x - min / max-min
# x = x' * (max-min) + min
y = [int(_y *(y_max-y_min)+y_min) for _y in predictions]
y
```


```python
# plot predictions
plt.figure(figsize=(12, 4))
sns.lineplot(data=original_df, x=original_df.index, y='Close', errorbar=None, label="True") # 실제 데이터 plot
sns.lineplot(x=X_test[seq_length:].index, y=y, errorbar=None, label="Pred")
plt.show()
```
