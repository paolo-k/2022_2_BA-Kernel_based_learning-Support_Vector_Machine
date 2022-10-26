# 2022_2_BA-Kernel_based_learning-Support_Vector_Machine
Tutorial Homework 2(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)


# Support Vector Machine
Support Vector Machine(이하 SVM)은 이름이 참 직관적이지 않다. Machine이라고 하니 무엇인가 작동해서 결과물을 만들어 내는 것 같은데, 앞에 support vector를 만드는 기계를 말하는 것인지... SVM은 실제로 작동해서 결과물을 만들어낸다. 단, 통상적으로 생각하는 Machine과 다르게 그 결과물이 물리적인 실체로서 나타나지 않는다. SVM의 결과물은 바로 두 가지 범주를 가지는 표본을 분류하는 분류경계면이다. 그리고 이 Machine의 분류경계면을 지지해주는(Support) 요소들이 바로 Support Vector 이다. 

# SVM의 background : Shatter

# 분류기로서 SVM

![tutorial_2_loc_glob_final](https://user-images.githubusercontent.com/106015570/198026838-c8fbb867-9d49-4371-bd7f-992606ba7fbb.png)

Support Vector Machine의 가장 큰 특징은, global optimum이 확실하게 보장된다는 것이다. 보통의 Machine learning 함수는 Local optimum이 곧 global optimum임을 보장할 수 없다.

# SVM 모델링 과정
모델링 과정은 크게 데이터셋 전처리, 분류경계면 구성 및 결과도출, 차원축소 후 분류경계면 구성 및 결과도출로 이어진다.

1 데이터셋 전처리
사용된 데이터는 sklearn의 classfication dataset 중 하나인 breast cancer dataset이다. 전처리 과정은 다음과 같이 진행되었다. 가장 먼저, 



# SVM 결과 분석 및 해석
