

[이슈]머리 염색 어플 
=============

## 목차
 #### 1. 딥러닝

  #### 2. 안드로이드
***
## 1. 딥러닝 이슈
## 1.1 hair image 데이터 모으는 방법

+ 젤 중요한 부분. 충분한 데이터를 모아야한다. 

+ 논문에서는 돈 주고 사거나 클라우드 펀딩을 받음 

+ 데이터를 모으지 못한다면 주제를 변경할 수도 있다.



## 1.2 Resolution multiplier ρ  

 + MobileNet에서 나오는 ρ는 해상도를 줄여 속도를 높이는 값이다.

 + ρ = 0.95 로 한다면 속도가 줄어들지 알아보자
 
 -------------
 
## 1.3 learning constant ε

+ optimization중 하나인 Adadelta의 hyparameter중 하나로 learning constant이다.

+ Adadelta의 설명은 [여기](http://incredible.ai/artificial-intelligence/2017/04/10/Optimizer-Adadelta/)에 들어가서 확인 할 수 있다.

+ 논문에서는 ε=1e-7을 사용하였다.


## 1.4 upsampling filter 값


+ skip connection 할 때 encoder부분과 upsampling의 filter 값이 다르다. (대칭X)

+ 논문에서 한번에 1024 -> 64로 됨

-------------
## 1.5 Skip connection 하는 방법 

+ concat or add 중 어떤 방법인지 알아보자

 - concat이라면 어떤 순서로 할 것인가 

## 1.6 Pixel-wise softmax 구현방법


## 1.7  Gradient magnitude
Second loss에 나오는 gradient magnitude의 의미를 찾고 구현 방법을 찾아보자  
Mask magnitude는 Mask라인에 겹쳐진 윤곽만 고려하도록 넣은 것으로 보임  
M, I는 normalize됨  
Mask edge와 Image edge가 같으면 IM == 1임으로 Loss는 0이고 Mask edge이외의 pixel은 M=0이므로 Loss가 0임  

추가적인 설명은 [여기](https://donghwa-kim.github.io/hog.html)서 확인






## 1.8 Transfer learning 주의할 점
 + MobileNet과 feature extraction 부분이 완전히 일치하는가
 
 + 일치하지 않는다면  특정 부분의 weight만 가져올 수 있을까?
  
    성능의 문제는 없을까? 
    
## 1.9 Tensorflow quantization의 속도 향상

+ Tensorflow quantization하면 float32 -> 8bit integer 가 되는데, 속도는 빨라질 것인가


***
## 2. 안드로이드 이슈
