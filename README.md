🚗 운전자 상태 인식을 위한 CNN 기반 멀티모달 멀티-입력 시스템
![alt text](https://img.shields.io/badge/AI-Deep%20Learning-blue.svg)

![alt text](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange.svg)

![alt text](https://img.shields.io/badge/Python-3.8+-yellow.svg)

![alt text](https://img.shields.io/badge/Publication-KJAI-brightgreen.svg)

<br>
운전자의 안전을 위한 새로운 눈, 멀티모달 AI가 운전자의 미세한 상태 변화까지 감지하여 사고를 예방합니다.
본 프로젝트는 운전자의 얼굴, 자세, 그리고 주변 소리 데이터를 동시에 분석하는 멀티모달 딥러닝 모델을 통해 **99.9%**의 정확도로 운전자 상태를 인식하는 시스템을 제안합니다. On-Device AI 적용을 목표로, 간단한 장비만으로도 높은 성능을 낼 수 있도록 설계되었습니다.

<br>
🌟 프로젝트 소개 (Introduction)
운전 중 발생하는 기절, 졸음, 분노와 같은 이상 상태는 대형 사고의 주요 원인입니다. 저희는 이러한 문제를 해결하기 위해 **세 가지 다른 종류의 데이터(얼굴, 자세, 소리)**를 함께 학습하는 멀티-입력 CNN 모델을 개발했습니다. 이 접근법을 통해 단일 데이터만 사용했을 때보다 압도적으로 높은 정확도를 달성하며, 운전자 상태 모니터링 시스템의 새로운 가능성을 제시합니다.

✨ 핵심 기여 (Key Contributions)
초고정확도 멀티모달 모델: 운전자의 얼굴, 자세 이미지와 소리 스펙트로그램을 동시에 입력받아 **99.9%**의 분류 정확도를 달성했습니다.
포괄적인 운전자 상태 분류: '정상', '기절', '졸음'과 같은 신체적 상태뿐만 아니라, '놀람', '분노', '불안'과 같이 이상 운전을 유발할 수 있는 3가지 감정 상태까지 총 6가지 클래스를 분류합니다.
On-Device AI 적용 가능성: 복잡한 센서 없이 카메라와 마이크 데이터만 사용하며, 비교적 간단한 CNN 구조로 설계되어 모바일 기기나 임베디드 시스템에 탑재하기 용이합니다.
🚀 모델 아키텍처 (Model Architecture)
본 프로젝트는 3개의 입력(얼굴 이미지, 자세 이미지, 소리 스펙트로그램)을 병렬로 처리하는 멀티-입력 CNN 모델을 사용합니다.

입력: 3개의 227x227x3 크기의 이미지를 각각 입력받습니다.
특징 추출: 각 입력 스트림은 2개의 Convolutional Layer를 거쳐 독립적으로 특징을 추출합니다.
특징 결합: 추출된 특징 맵(Feature Map)들은 하나로 결합(Concatenate)됩니다.
심층 학습: 결합된 특징은 3개의 추가 Convolutional Layer와 3개의 Affine Layer(FC Layer)를 거쳐 최종적으로 6개의 클래스로 분류됩니다.
Figure 1: 제안된 시스템의 전체 구조도

📊 데이터셋 (Dataset)
실험을 위해 총 6개의 클래스에 대한 멀티모달 데이터를 자체적으로 구축했습니다.

총 데이터 수: 18,900개 (학습 데이터: 12,600개 / 테스트 데이터: 6,300개)
입력 데이터 종류:
운전자 얼굴 이미지 (Driver Face)
운전자 상태 이미지 (Driver State / Posture)
소리 스펙트로그램 (Sound Spectrograms)
6가지 분류 클래스 (6 Classes)
클래스	상태 (State)	설명
c0	정상 (Normal)	정면을 주시하며, 주변 소음이 조용함
c1	기절 (Faint)	고개를 숙이고 핸들에 기댐, 충돌음 발생
c2	졸음 (Drowsy)	눈을 감거나 하품, 하품 소리 발생
c3	놀람 (Surprise)	눈과 입을 크게 벌림, 경적 소리 발생
c4	분노 (Anger)	창밖을 보며 화를 냄, 경적과 분노 음성
c5	불안 (Anxiety)	긴장된 표정, 짧은 경적과 불안 음성
Figure 2: 실험에 사용된 데이터 샘플

📈 실험 결과 (Experimental Results)
다양한 입력 조합에 따른 모델 성능을 ResNet50, Xception과 비교하여 제안 모델의 우수성을 입증했습니다.

모델 (Model)	입력 데이터 (Input Data)	정확도 (Accuracy)
ResNet50	Driver face (단일)	81.0%
Driver state (단일)	85.7%
Xception	Driver face (단일)	87.8%
Driver state (단일)	76.8%
Our CNN (제안 모델)	Driver face (단일)	80.7%
Driver state (단일)	68.6%
Driver face + state (이중)	75.8%
Driver face + state + sound (삼중)	🏆 99.9%
단일 입력이나 이중 입력 방식에 비해, 세 가지 데이터를 모두 활용한 멀티모달 접근법이 압도적으로 높은 성능을 보임을 확인했습니다.

Figure 3: 멀티-입력 CNN의 학습 결과 그래프. 삼중 입력(우측 하단)에서 훈련 및 테스트 정확도가 모두 99.9%에 수렴했다.

🛠️ 시작하기 (Getting Started)
사전 요구사항
Python 3.8+
TensorFlow 2.x
Keras
NumPy
Matplotlib
설치
code
Bash
# 1. 저장소를 클론합니다.
git clone https://github.com/your-username/driver-state-recognition.git
cd driver-state-recognition

# 2. 필요한 패키지를 설치합니다.
pip install -r requirements.txt
실행 (예시)
code
Bash
# 모델 학습
python train.py --data_path ./dataset

# 예측 수행
python predict.py --face_img_path ./face.jpg --state_img_path ./state.jpg --sound_path ./sound.wav
(위 명령어는 예시이며, 실제 코드 구조에 맞게 수정이 필요합니다.)

🧑‍💻 팀원 (Our Team)
이름 (Name)	역할 (Role)
강지수	레퍼런스 조사 및 보고서 작성
김수아	레퍼런스 조사 및 보고서 작성
신수아	ResNet50 모델 연구 및 실험
여환서	Xception 모델 연구 및 실험
이동연	데이터셋 생성 및 전처리
이정현	데이터셋 생성 및 전처리
한기준	레퍼런스 조사 및 보고서 작성
📜 논문 및 인용 (Citation)
본 프로젝트는 한국성서대학교의 지원을 받아 수행되었으며, 연구 결과는 KJAI (Korean Journal of Artificial Intelligence) 에 게재되었습니다.

연구를 인용하시려면 아래 정보를 사용해주세요.

code
Code
Sooah SHIN, Jisu KANG, Sooah KIM, Hwanseo YEO, Dongyeon LEE, Jeongjyun LEE, Gijun HAN, Jinho HAN. (2025). "A study on driver state recognition using CNN-based multimodal multi-input learning". Korean Journal of Artificial Intelligence, xx(xx), xx-xx.
🙏 감사의 말 (Acknowledgments)
본 연구는 2024년 한국성서대학교 대학혁신지원사업의 지원을 받아 수행되었습니다.
