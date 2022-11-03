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

그러나 연구가 더 진행되면서, local optimum의 경우 대체로 다른 차원을 통해 빠져나감으로써, 결국 global optimum에 수렴한다는 사실이 밝혀졌다. 

![tutorial_2_ML_escape](https://user-images.githubusercontent.com/106015570/199726561-8cc03c8c-4999-4e3c-a4cd-476523abdca2.png)


3. Case of SVM

SVM은 선형 분류면으로 분류가 가능한지 여부(linearly separable, linearly non-separable)와, 오분류 감수 여부(hard margin, soft margin)에 따라 4가지 case를 가진다. linearly non-separable인 경우는 Kernel Trick을 통해 고차원 공간 상에서 각 vector 간 내적을 구한 후, 이에 기반하여 비선형 분류기를 생성한다. soft margin을 적용해야하는 경우는 오분류에 대하여 penalty를 부과하여 해결한다.

|분류|hard margin|soft margin|
|------|---|---|
|linear|basic|오분류비용 도입|
|nonlinear|커널 트릭 적용|오분류비용 + 커널 트릭|

SVM의 기본 형태이자, 가장 이상적인 상황은 linearly separable & hard margin 이다. 하지만 이런 경우는 현실에서 찾기 어렵다. 따라서 대부분의 경우는 linearly non-separable & soft margin을 적용한다.



# Tutorial of Support Vector Machine

## 코드 및 데이터 개요

본 tutorial에 사용된 데이터는 sklearn의 classification dataset 중 하나인 breast cancer dataset이다. 본 데이터는 y값으로 사용될 class 1개를 포함한 31개 feature, 569개 instance로 구성되어 있다. class는 0 group에 212개 instance, 1 group에 357개 instance가 존재한다.

![image](https://user-images.githubusercontent.com/106015570/199642855-b99652d5-9f40-42aa-89d1-a19a25b8e7a6.png)


## SVM 모델링 과정
모델링 과정은 크게 데이터셋 전처리, 분류경계면 구성 및 결과도출, 데이터 분포 분석, 다른 모델(로지스틱 분석)과의 성능 비교의 순서로 이루어졌다.

1. 데이터셋 전처리

전처리 과정은 다음과 같이 진행되었다. 가장 먼저, 독립변수(X)들의 결측여부와, 변수특성(명목, 연속 등)을 파악하였다. 분석 결과 모든 X는 결측값 없는 연속변수들로 구성되어 있었다. 따라서, 명목변수에 대한 별도의 dummy variable 생성 과정은 필요하지 않다.

![image](https://user-images.githubusercontent.com/106015570/199642915-ed0a385f-b01c-4054-8616-dd88b5fbe73d.png)

그 다음, 모든 X에 대하여 min-max scaling을 적용하여, 단위 차이로 인한 영향을 배제한다.

![image](https://user-images.githubusercontent.com/106015570/199642991-7e0d51cb-0c9e-411e-8c54-61922d4739f3.png)

마지막으로, 6:2:2의 비율로 training set(341개 instance), validation set(114개 instance), testing set(114개 instance)을 나누어 전처리를 마무리한다.


2. 분류경계면 구성 및 결과도출

 - linear SVM
 
linear SVM의 hyperparameter는 오분류에 대한 penalty c가 있다. c와, margin의 넓이는 반비례한다. c가 넓어지면 margin은 축소되고, 반대로 c가 낮아지면 margin은 확장된다. 아래는 c = 1에서 linear SVM을 모델링한 후, testing set에 적용한 결과이다. 본 튜토리얼의 데이터셋은 class 0과 class 1의 instance 수가 각각 212개, 357개로 균형이 잡혀있기에, confusion matrix를 통해 성능을 평가하였다.

![confusion_linear_basic](https://user-images.githubusercontent.com/106015570/199727422-cc5576d5-8721-4f3d-a405-096d25d82841.png)

아래는 차례로 c=1000, c=0.01에서 testing set에 적용한 결과이다.

![confusion_linear_hard](https://user-images.githubusercontent.com/106015570/199728657-eb9fe888-2a79-4eea-b623-7186f1c4958a.png)
![confusion_linear_soft](https://user-images.githubusercontent.com/106015570/199728669-32bddcae-1ed9-468e-bb0a-c80723a8fc97.png)

c가 높을수록 정분류율이 높아지는 것을 확인할 수 있다. 일반적으로 c가 높아져 마진이 축소되면, 그만큼 구조적 위험성이 증가하므로, overfit이 발생할 가능성이 커지기 때문에, testing set에 대한 정분류율이 무조건 높아지지 않는다. 그럼에도 이와 같이 c와 정분류율이 비례하는 결과가 나타난 것은, 그만큼 본 튜토리얼에서 사용한 데이터셋이 분류하기 쉬운 데이터셋이기 때문인 것으로 추정된다.
그럼에도 비슷한 성능을 보이는 더 작은 parameter를 통해 구조적 위험성을 최소화하는 연습은 필요하다. 이러한 이유로 validation set을 이용하여 c를 100 미만의 범위에서 튜닝하였다. 일일이 조정하면서 탐색한 결과, c=39~87에서의 성능이 가장 적절하다고 판단되었다. 이에 따라 c=50에서 모델링 후, testing한 결과는 아래와 같다. 

![confusion_linear_optimal](https://user-images.githubusercontent.com/106015570/199731136-e9d6aa82-dd4d-444b-ada9-4a1e946c7823.png)

 - nonlinear SVM

본 튜토리얼에서는 gaussian kernel을 적용한 nonlinear SVM을 모델링하였다. gaussian kernel SVM의 hyperparameter는 선형 SVM과 같은 오분류 비용 c와 Margin의 넓이를 결정하는 gamma의 2가지가 있다. gamma의 경우, 표준편차에 반비례하며, gamma가 클수록 Margin은 좁아진다. 아래는 c = 1, gamma = 1에서 gaussian kernel SVM을 모델링한 후 validation set에 적용한 결과이다.

![confusion_gaussian_basic](https://user-images.githubusercontent.com/106015570/199733232-edf71714-277b-436c-a598-09fbc26f7599.png)

아래는 차례로 c=10, c=0.1, gamma = 10, gamma = 0.1에서 validation set에 적용한 결과이다.

![confusion_linear_hard](https://user-images.githubusercontent.com/106015570/199733800-c799599f-03d8-401d-82c3-7550f3038463.png)
![confusion_gaussian_soft](https://user-images.githubusercontent.com/106015570/199733821-9e562c67-207a-4c03-ae71-88300bff56b8.png)
![confusion_gaussian_naro](https://user-images.githubusercontent.com/106015570/199733915-9b4a49ce-66ff-4dab-b410-95aaa1c24f22.png)
![confusion_gaussian_wide](https://user-images.githubusercontent.com/106015570/199733954-4ed165d9-9be5-41a2-9ce2-44b49b0141a7.png)

앞서 선형 SVM에 비해서는 결과가 비교적 다채로운 편이다. 먼저, c와 gamma 모두 class 0의 정분류율과 비례하며, 반대로 class 1의 정분류율에는 반비례한다. gaussian kernel SVM의 경우 hyperparameter가 두 개이므로, manual한 방법을 적용하는 것이 매우 비효율적이다. 따라서, 다음의 범위에서 grid search를 통해 최적의 파라미터를 찾았다. 

|파라미터|최솟값|최댓값|단위|
|------|---|---|---|
|c|0.1|10|0.05|
|gamma|0.1|10|0.05|

분석 결과, c = 1.05, gamma = 3.5이 가장 좋은 성능을 보임을 도출할 수 있었다. 이에 따라 최적의 SVM 결과를 도출하면, 아래와 같다.

![confusion_gaussian_optimal](https://user-images.githubusercontent.com/106015570/199735721-140eeb72-e269-464d-aec8-28383d76f245.png)


3. 데이터 분포 파악

앞서 선형 SVM에서 언급한 바와 같이, 모델링에 사용된 데이터셋은 굉장히 분류가 쉽다고 판단되었다. 이를 확인하기 위해 본 데이터셋(30차원)을 3차원으로 축소한 후, 분포를 시각화하였다. 2차원이 아닌 3차원으로 축소한 이유는, 정보의 손실을 최소화하기 위함이다. 각 클래스 데이터 분포를 확인한 결과는 아래와 같다.

![data_distribution](https://user-images.githubusercontent.com/106015570/199736502-d38c4a80-2ebe-4034-aac1-fa9490dedf14.png)

시각화된 자료에서 알 수 있듯, 두 데이터는 상당히 명확하게 구분된다. 앞서서 linear & hard margin SVM의 성능이 좋았던 것은, 이와 같이 class에 따른 데이터 분포가 명확한 것이 그 원인으로 파악된다.


4. 다른 모델(로지스틱 분석)과의 성능 비교

최적의 hyperparameter를 적용한 nonlinear SVM의 성능을 비교하기 위하여, 널리 쓰이는 분류 방법론인 logistic regression와 비교해보았다. 아래 사진은 두 방법론의 결과를 비교한 것으로, 위가 SVM, 밑이 logistic regression 적용 결과이다. 전반적인 정분류율은 SVM 쪽이 더 뛰어나며, class 1에 대한 정분류율에서만 logistic regression이 미세하게 좋은 성능을 보인다.

![confusion_gaussian_optimal](https://user-images.githubusercontent.com/106015570/199742735-85b95efa-9103-42b1-ac50-560e6ab76f87.png)
![confusion_logistic](https://user-images.githubusercontent.com/106015570/199742798-9c628386-392a-4578-a1b5-db7ae1600da6.png)


# conclusion

## 결과의 분석 및 해석
본 튜토리얼의 결과, 모델링된 gaussian kernel SVM은 testing set에 대하여 거의 완벽에 가까운 정분류율을 보인다. 그 원인으로는 아래의 2가지를 꼽을 수 있다.
(1) 최적의 파라미터 탐색 : gaussian kernel의 경우 grid search를 이용하여 최적의 파라미터를 자동으로 구했다. 단순히 manual한 방법으로 일관했다면, 최적의 파라미터를 찾기 어려웠을 것이라 판단된다.
(2) 데이터셋의 편리성 : 3차원에 시각화된 분포에서 파악할 수 있듯, 분석에 사용된 데이터셋은 집단 구분이 명확했다. 이 때문에 모델 성능이 특히 잘 나온 것으로 추정된다.
다만, 같은 데이터셋을 적용한 로지스틱 회귀분석에서 성능이 비교적 낮았던 점을 고려하면, SVM의 성능이 로지스틱 회귀분석보다 더 좋게 나타난 것이라 판단할 수 있다.

## 의의 및 보완점

1. 의의
본 튜토리얼은 데이터의 특성을 면밀히 파악하고, 그에 따른 모델링 및 적절한 평가 방법을 적용하였다. 우선, 명목변수가 없다는 독립변수 set 특성에 따라 dummy variable를 생성하지 않기로 결정하였다. 분류 대상이 되는 두 집단 간 균형이 맞음을 근거로, 간단하면서도 명쾌한 confusion matrix를 통해 직관적인 성능지표를 시각적으로 표현했다. grid search 등 자동화된 방법을 통해 최적의 파라미터를 찾아내었고, 그 성능을 비교군을 통하여 명쾌하게 보였다. 덧붙여, 모델 성능이 잘 나온것을 그냥 넘어가지 않고, 일반적이지 않은 상황임을 식별하였다. 또한 그 일반적이지 않은 상황의 원인을 데이터 특성을 통해 명확히 규명했다.

2. 보완점
본 튜토리얼의 데이터셋은 굉장히 분류가 쉬운 데이터셋이었다. 모델 성능을 좀 더 냉정하게 평가하기 위해서는, 현실의 어려운 데이터셋 대상의 모델링이 필요하다. 비교군으로서 적용한 logistic regression의 경우, SVM에 사용된 grid search 등을 통해 최적의 parameter를 적용했다면, 결과가 달라졌을 수 있다. 향후 튜토리얼 등에서 비교군을 설정할 시에는 이 부분을 반드시 유념하여, 비교군 또한 최적의 파라미터를 맞추는 등의 노력을 견지할 필요가 있다.
