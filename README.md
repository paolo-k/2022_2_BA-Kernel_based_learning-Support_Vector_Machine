# 2022_2_BA-Kernel_based_learning-Support_Vector_Machine
Tutorial Homework 2(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)


# Support Vector Machine
Support Vector Machine(이하 SVM)은 이름이 참 직관적이지 않다. Machine이라고 하니 무엇인가 작동해서 결과물을 만들어 내는 것 같은데, 앞에 support vector를 만드는 기계를 말하는 것인지. SVM은 실제로 작동해서 결과물을 만들어낸다. 단, 통상적으로 생각하는 Machine과 다르게 그 결과물이 물리적인 실체로서 나타나지 않는다. SVM의 결과물은 바로 두 가지 범주를 가지는 표본을 분류하는 분류경계면이다. 그리고 이 Machine의 분류경계면을 지지해주는(Support) 요소들이 바로 Support Vector 이다. 

## SVM의 원리
1. 구조적 위험의 최소화
예측 모델은 크게 두 가지 성능에 기반하여 평가된다. 하나는 예측의 정확도이고, 다른 하나는 메커니즘의 복잡도이다. 같은 정확도면 복잡도가 낮은 모델이 연산시간 측면에서도 당연히 효율적이고, 일반화 측면에서도 성능이 좋다. 복잡한 모델은 training error를 최소화할 수 있지만, 그 모델의 일반화가 어렵다. 모델을 학습시킨 데이터를 지나치게 상세하게 반영하므로, 다른 데이터의 특성을 제대로 파악하지 못하는, 쉽게 말해 overfit이 발생하기 쉽다. 파라미터 등 모델의 구성요소에 민감하여, 약간의 조정만으로도 결과가 확연히 크게 달라지기도 한다.

![structural_risk](https://user-images.githubusercontent.com/106015570/198205512-0146f627-3790-4e66-8fa5-1a1ead44554e.png)


인공신경망, 의사결정나무, 로지스틱 회귀분석 등의 예측 및 분류 기법들은 (아직 미완성)

## 분류기로서 SVM

![tutorial_2_loc_glob_final](https://user-images.githubusercontent.com/106015570/198035012-b53e1d10-0864-4975-a2f6-327b003c2be6.png)

Support Vector Machine의 가장 큰 특징은, global optimum이 확실하게 보장된다는 것이다. 보통의 Machine learning 함수는 Local optimum이 곧 global optimum임을 보장할 수 없다.


# SVM의 구현


## 코드 및 데이터 개요


## SVM 모델링 및 코드 과정
모델링 과정은 크게 데이터셋 전처리, 분류경계면 구성 및 결과도출, 차원축소 후 분류경계면 구성 및 결과도출로 이어진다.

1 데이터셋 전처리
사용된 데이터는 sklearn의 classfication dataset 중 하나인 breast cancer dataset이다. 전처리 과정은 다음과 같이 진행되었다. 가장 먼저, 



## SVM 결과 분석 및 해석
