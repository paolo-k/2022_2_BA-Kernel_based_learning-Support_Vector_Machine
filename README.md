# 2022_2_BA-Kernel_based_learning-Support_Vector_Machine
Tutorial Homework 2(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)


# Overview of Support Vector Machine
Support Vector Machine(이하 SVM)은 분류경계면을 형성하여 이진 범주를 가지는 표본을 분류하는 모델이다. 그리고 이 Machine의 분류경계면을 지지해주는(Support) 요소들이 바로 Support Vector 이다.

1. Margin의 최대화를 통한 구조적 위험 감소

예측 모델은 크게 두 가지 성능에 기반하여 평가된다. 하나는 예측의 정확도이고, 다른 하나는 메커니즘의 복잡도이다. 같은 정확도면 복잡도가 낮은 모델이 연산시간 측면에서도 당연히 효율적이고, 일반화 측면에서도 성능이 좋다. 복잡한 모델은 training error를 최소화할 수 있지만, overfit이 발생하므로 그 모델의 일반화가 어렵다. 

![structural_risk](https://user-images.githubusercontent.com/106015570/198205512-0146f627-3790-4e66-8fa5-1a1ead44554e.png)

SVM은 Loss function에 의하여 측정되는 정확도만 고려하지 않고, 복잡한 구조로 인한 구조적 위험까지 같이 고려하는 특성을 가진다. SVM은 hyperplane을 활용하여 집단을 분류한다. hyperplane은 선형함수(wx + b)로 정의 가능하다. 즉, SVM은 기본적으로 선형 분류기에 속한다. SVM은 두 집단을 분류하는 것 외에 Margin의 최대화라는 목적이 존재한다. Margin이란 hyperplane으로부터 각 집단의 경계면에 있는 인스턴스, 즉 Support vector까지의 거리를 의미한다. 

![image](https://user-images.githubusercontent.com/106015570/199633310-d358d30f-78f5-4921-ab48-1eee2fe3aa77.png)

위 그림은 고려대학교 강필성 교수님 Business Analytics 수업 자료 일부를 발췌한 것이다. 그림에서 알 수 있듯, hyperplane의 Margin이 넓을수록, 고려 가능한 경계면의 수가 줄어든다. 즉, Margin의 최대화는 모델의 복잡도를 줄임으로써, 구조적 위험을 최소화한다. 

2. Global optimal 보장

![tutorial_2_loc_glob_final](https://user-images.githubusercontent.com/106015570/198035012-b53e1d10-0864-4975-a2f6-327b003c2be6.png)

Support Vector Machine의 가장 큰 특징 중 하나는, global optimum이 확실하게 보장된다는 것이다. 보통의 Neural Network 기반의 방법은 local optimum이 곧 global optimum임을 보장할 수 없다. 그에 비해, Support Vector Machine은 후술하겠지만 목적식이 2차식인 만큼, global optimum을 보장할 수 있다.

<img src="https://user-images.githubusercontent.com/106015570/199642662-f51cec66-45df-4c0f-bb97-c02492573b2b.png.png" width="200" height="400"/>

(그러나 연구가 더 진행되면서, local optimum의 경우 대체로 다른 차원을 통해 빠져나감으로써, 결국 global optimum에 수렴한다는 사실이 밝혀졌다. )

3. Case of SVM

SVM은 선형 분류면으로 분류가 가능한지 여부(linearly separable, linearly non-separable)와, 오분류를 감수해야 하는지 여부(hard margin, soft margin)에 따라 4가지 case를 가진다. linearly non-separable인 경우는 Kernel Trick을 통해 고차원 공간 상에서 각 vector 간 내적을 구한 후, 이에 기반하여 비선형 분류기를 생성한다. 오분류가 필연적으로 발생하여 soft margin을 적용해야하는 경우는 오분류에 대하여 penalty를 부과하여 해결한다.

(분류표 삽입 예정)

SVM의 기본 형태이자, 가장 이상적인 상황은 linearly separable & hard margin 이다. 하지만 이런 경우는 현실에서 찾기 어렵다. 따라서 대부분의 경우는 linearly non-separable & soft margin을 적용한다.



# Tutorial of Support Vector Machine

## 코드 및 데이터 개요
본 tutorial에 사용된 데이터는 sklearn의 classification dataset 중 하나인 breast cancer dataset이다. 본 데이터는 y값으로 사용될 class 1개를 포함한 31개 feature, 569개 instance로 구성되어 있다. class는 0 group에 212개 instance, 1 group에 357개 instance가 존재한다.

![image](https://user-images.githubusercontent.com/106015570/199642855-b99652d5-9f40-42aa-89d1-a19a25b8e7a6.png)


## SVM 모델링 과정
모델링 과정은 크게 데이터셋 전처리, 분류경계면 구성 및 결과도출, 차원축소 후 분류경계면 구성 및 결과도출, 다른 모델(로지스틱 분석)과의 비교로 구성된다.

1. 데이터셋 전처리

전처리 과정은 다음과 같이 진행되었다. 가장 먼저, 독립변수(X)들의 결측여부와, 변수특성(명목, 연속 등)을 파악하였다. 분석 결과 모든 X는 결측값 없는 연속변수들로 구성되어 있었다. 따라서, 명목변수에 대한 별도의 dummy variable 생성 과정은 필요하지 않다.

![image](https://user-images.githubusercontent.com/106015570/199642915-ed0a385f-b01c-4054-8616-dd88b5fbe73d.png)

그 다음, 모든 X에 대하여 min-max scaling을 적용하여, 단위 차이로 인한 영향을 배제한다.

![image](https://user-images.githubusercontent.com/106015570/199642991-7e0d51cb-0c9e-411e-8c54-61922d4739f3.png)

마지막으로, 6:2:2의 비율로 training set(341개 instance), validation set(114개 instance), testing set(114개 instance)을 나누어 전처리를 마무리한다.


2. 분류경계면 구성 및 결과도출

 - linear SVM
 
선형 SVM의 hyperparameter는 비용함수, 즉 오분류에 대한 penalty가 있다. (설명 작성 예정)

3. 차원축소 후 분류경계면 구성 및 결과도출

4. 다른 모델(로지스틱 분석)과의 성능 비교


## SVM 결과 분석 및 해석
