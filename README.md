🚗 운전자 상태 인식을 위한 CNN 기반 멀티모달 시스템
![alt text](https://img.shields.io/badge/AI-Deep%20Learning-blue.svg)

![alt text](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange.svg)

![alt text](https://img.shields.io/badge/Python-3.8+-yellow.svg)

![alt text](https://img.shields.io/badge/Publication-KJAI-brightgreen.svg)

<br>
운전자의 안전을 위한 새로운 눈, 멀티모달 AI가 운전자의 미세한 상태 변화까지 감지하여 사고를 예방합니다.
본 프로젝트는 운전자의 얼굴, 자세, 그리고 주변 소리 데이터를 동시에 분석하는 멀티모달 딥러닝 모델을 통해 **99.9%**의 정확도로 운전자 상태를 인식하는 시스템을 제안합니다. On-Device AI 적용을 목표로, 간단한 장비만으로도 높은 성능을 낼 수 있도록 설계되었습니다.

<br>
🛠️ 기술 스택 (Tech Stack)
Category	Stack
Environment	
![alt text](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
 ![alt text](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)
Deep Learning	
![alt text](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
 ![alt text](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
Data Processing	
![alt text](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
 ![alt text](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
 ![alt text](https://img.shields.io/badge/Librosa-FF69B4?style=for-the-badge)
Visualization	
![alt text](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
🚀 모델 아키텍처 (Model Architecture)
본 프로젝트는 3개의 입력을 병렬로 처리 후 결합하는 멀티-입력 CNN 모델입니다.

Layer	Specifications
Input	Input 1: Driver Face Image (227x227x3)<br>Input 2: Driver State Image (227x227x3)<br>Input 3: Sound Spectrogram (227x227x3)
Conv 1	Filter: 96, Kernel: (11x11), Stride: 4 -> Output: (55x55x96)
Max Pooling 1	Pool size: (3x3), Stride: 2 -> Output: (27x27x96)
Conv 2	Filter: 256, Kernel: (5x5), Padding: same -> Output: (27x27x256)
Max Pooling 2	Pool size: (3x3), Stride: 2 -> Output: (13x13x256)
Concatenate	3개 입력 스트림의 출력을 결합 -> Output: (13x13x768)
Conv 3	Filter: 640, Kernel: (3x3), Padding: same -> Output: (13x13x640)
Conv 4	Filter: 640, Kernel: (3x3), Padding: same -> Output: (13x13x640)
Conv 5	Filter: 384, Kernel: (3x3), Padding: same -> Output: (13x13x384)
Max Pooling 3	Pool size: (3x3), Stride: 2 -> Output: (6x6x384)
Flatten	Output: 13824
FC Layer 1	Neurons: 4096, Dropout: 0.5
FC Layer 2	Neurons: 4096, Dropout: 0.5
Output Layer	Neurons: 6, Activation: Softmax
📊 데이터셋 및 전처리 (Dataset & Preprocessing)
항목 (Item)	사양 (Specification)
총 데이터 수	18,900개 (Train: 12,600 / Test: 6,300)
클래스	6가지 상태 (정상, 기절, 졸음, 놀람, 분노, 불안)
입력 데이터	(1) 운전자 얼굴 이미지, (2) 운전자 자세 이미지, (3) 소리 스펙트로그램
이미지 전처리	227x227 크기로 리사이즈 및 정규화
사운드 전처리	2초 길이의 음성 파일을 Mel Spectrogram으로 변환 후 이미지화
📈 실험 결과 (Experimental Results)
모델 (Model)	입력 데이터 (Input Data)	정확도 (Accuracy)
ResNet50	Driver face + state (이중 입력)	81.0% ~ 85.7%
Xception	Driver face + state (이중 입력)	76.8% ~ 87.8%
Our CNN (제안 모델)	Driver face + state (이중 입력)	75.8%
Our CNN (제안 모델)	Driver face + state + sound (삼중 입력)	🏆 99.9%
<p align="center">
<img src="path/to/your/graph-image.png" width="800">
<br>
<em>Figure: 멀티-입력 CNN의 학습 결과 그래프. 삼중 입력에서 99.9% 정확도를 달성했다.</em>
</p>
🧑‍💻 팀원 (Our Team)
이름 (Name)	역할 (Role)
강지수, 김수아, 한기준	레퍼런스 조사 및 보고서 작성
신수아, 여환서	비교 모델(ResNet50, Xception) 연구 및 실험
이동연, 이정현	데이터셋 생성 및 전처리
📜 논문 및 인용 (Citation)
본 프로젝트는 한국성서대학교의 지원을 받아 수행되었으며, 연구 결과는 KJAI (Korean Journal of Artificial Intelligence) 에 게재되었습니다.

code
Code
Sooah SHIN, Jisu KANG, Sooah KIM, Hwanseo YEO, Dongyeon LEE, Jeongjyun LEE, Gijun HAN, Jinho HAN. (2025). "A study on driver state recognition using CNN-based multimodal multi-input learning". Korean Journal of Artificial Intelligence, xx(xx), xx-xx.
