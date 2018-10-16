[이슈]머리 염색 어플 
=====

***
#  딥러닝 이슈
- [ ] hair image 데이터셋 모으기
- [x] Resolution multiplier ρ?
- [x] learning constant ε?
- [x] encoder/decoder filter 갯수가 비대칭적이다
- [x] Skip connection 하는 방법. concatenate? add?
- [ ] Pixel-wise softmax 구현방법 + loss function
- [x] Gradient magnitude 뭐지?
- [ ] Mobilenet transfer learning 가능한가?
- [ ] Tensorflow quantization하면 속도 향상 될까?
- [ ] Pytorch dataloader 다시 만들기

### 데이터셋
- Dropbox: images/h0001.png, masks/h0001.png
- 224x224. 가로세로 비율 깨지지 않게
- 여기 아래 annotation tool site 링크 걸어주셈.

### ρ
- MobileNet에서 나오는 ρ는 해상도를 줄여 속도를 높이는 값이다.
- ρ = 0.95 로 한다면 속도가 줄어들지 알아보자

### ε
- optimization중 하나인 Adadelta의 hyper parameter중 하나로 learning constant이다.
- Adadelta의 설명은 [여기](http://incredible.ai/artificial-intelligence/2017/04/10/Optimizer-Adadelta/)에 들어가서 확인 할 수 있다.
- 논문에서는 ε=1e-7을 사용하였다.

### filter 갯수 비대칭
- skip connection 할 때 encoder부분과 upsampling의 filter 값이 다르다. (대칭X)
- 논문에서 한번에 1024 -> 64로 됨

### skip connections
- add였다!! 원범이 짱

### Pixelwise-softmax
- pixelwise classification, pixelwise softmax 등의 키워드로 검색

### gradient magnitude
- Second loss에 나오는 gradient magnitude의 의미를 찾고 구현 방법을 찾아보자  
- Mask magnitude는 Mask라인에 겹쳐진 윤곽만 고려하도록 넣은 것으로 보임  
- M, I는 normalize됨  
- Mask edge와 Image edge가 같으면 IM == 1임으로 Loss는 0이고 Mask edge이외의 pixel은 M=0이므로 Loss가 0임  
- 추가적인 설명은 [여기](https://donghwa-kim.github.io/hog.html)서 확인

### transfer learning
- MobileNet과 우리 논문의 feature extractor 부분이 완전히 일치하는가
- 일치하지 않는다면 특정 부분의 weight만 가져올 수 있을까? 그렇게 하면 성능의 문제는 없을까? 

### quantization
- Tensorflow quantization하면 float32 -> 8bit integer 가 되는데, 속도는 빨라질 것인가



***
#  안드로이드 이슈

## 1. 스토리보드

### 1.0 앱 진입 대기 화면(로딩)
까만 바탕 로딩화면을 할지, 카카오톡처럼 진입 대기화면을 만들지 논의. 
<br><img width="180" height="320" src="/Hoon/ref_00.jpg"></img><br>
매우 짧은 시간이지만 사용자에게 앱의 이미지를 각인시키는 효과가 있는 단계. UI/UX 디자이너분께 부탁

### 1.1 앱 진입 직후(카메라)
카메라 구동. 그 방법은 다음과 같이 3가지로 나뉨.
사진을 찍은 뒤에 **1.2**로 넘어간다.
만일 여기서 되돌아가기 실행시 어플 종료.
+ 기본 카메라
+ 카메라 선택(폰에 설치된 어플)
	+ 푸디, 캔디캠, B612 등 선택
+ 카메라 구동 AP 이용 (가장 깔끔한 화면)

### 1.2 사진 편집(Main)
<img width="270" height="480" src="/Hoon/ref_01.jpg"></img><br>
기본적인 구성은 이와 비슷하므로, 이미지를 참고하면서 읽으면 됨.
화면 가운데 가장 크게 **1.1**에서 찍은 사진이 나오고
하단에 클릭이벤트들이 다음과 같이 있음.
하위 항목은 각 클릭이벤트마다 새롭게 나오는 activity에 대한 설명임. *추후 스토리보드 구체화할때 각각 자세히 더 추가함*
+ 염색 선택(Palette)
	+ 색깔 지정(색 범위: discrete / continuous도 결정해야함)
	+ 염색 정도 지정(투명도를 어느정도로 할지)
+ 사진 편집(Edit)
	+ Crop (기본적인 잘라내기 + 비율 지정 잘라내기)
	+ 여백 만들기(사진 내에 Margin 추가 + Margin 색도 지정가능하게)
	+ 콜라주 작업(다음 사진 참고) - 한 장을 찍은 뒤 2x2 등으로 화면분할하여 각각 사진에 머리염색을 각각 적용한다.
	<img src = "https://is2-ssl.mzstatic.com/image/thumb/Purple111/v4/e6/dd/79/e6dd791c-4c10-701a-eb61-1efbd436b200/pr_source.jpg/300x0w.jpg"></img>
+ 사진 저장(따로 지정된 저장 전용폴더)
+ 공유(SNS 등. API 갖다쓰기)
+ 취소(ImageButton을 따로 만들지 안만들지 고민중. 되돌아가기 버튼으로도 충분하고 둘다 해도 되고. 여튼 이 이벤트 발생시 **1.1**로 되돌아감)
