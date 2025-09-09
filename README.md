# 🚗 CNN 기반 멀티모달 멀티-입력 운전자 상태 인식 시스템

<div align="center">
  <!-- 여기에 데모 GIF나 프로젝트 대표 이미지를 삽입하세요 -->
  <!-- 예: <img src="https://path/to/your/image.png" width="800"> -->
  <br>
  <strong>얼굴 표정, 운전 자세, 소리 데이터를 활용한 운전자 상태 인식 딥러닝 시스템</strong>
  <br><br>
  <a href="#-논문-및-인용">📖 논문</a> •
  <a href="#">🎯 데모</a> •
  <a href="#-결과">📊 결과</a> •
  <a href="#-시작하기">🚀 시작하기</a>
</div>

## 🌟 개요 (Overview)

본 프로젝트는 다음과 같은 세 종류의 입력 데이터를 결합하여 **99.9%**의 정확도를 달성하는 새로운 CNN 기반 멀티모달 운전자 상태 인식 접근법을 제시합니다.

-   **👤 운전자 얼굴 표정**
-   **🚗 운전 자세 이미지**
-   **🔊 소리 스펙트로그램** (운전자 음성 + 주변 환경음)

저희가 제안하는 경량화된 멀티-입력 CNN 아키텍처는 스마트폰 등에서 흔히 볼 수 있는 간단한 센서(카메라, 마이크)만을 사용하여 **온디바이스 AI(On-Device AI)** 환경에 최적화되도록 설계되었습니다.

## 🎯 주요 특징 (Key Features)

#### 🔍 6가지 클래스 분류
-   **운전자 상태**: 정상, 기절, 졸음
-   **운전자 감정**: 놀람, 분노, 불안

#### 📱 온디바이스 AI 최적화
-   최소한의 컴퓨팅 자원을 요구하는 간단한 CNN 아키텍처
-   카메라와 마이크 데이터만을 사용
-   모바일 및 임베디드 시스템에 탑재 용이

#### 🎨 멀티모달 접근법
-   3중 입력 처리 (얼굴 + 자세 + 소리)
-   고도화된 결합(Concatenate) 전략
-   단일 모달(Single-modal) 방식 대비 뛰어난 성능 향상

## 📊 결과 (Results)

| 입력 종류 | 정확도 | 모델 |
| :--- | :---: | :--- |
| 운전자 얼굴 (단일) | 80.7% | 제안 모델 (Multi-input CNN) |
| 운전자 상태 (단일) | 68.6% | 제안 모델 (Multi-input CNN) |
| 이중 입력 (얼굴 + 상태) | 75.8% | 제안 모델 (Multi-input CNN) |
| **삼중 입력 (얼굴 + 상태 + 소리)** | **🏆 99.9%** | **제안 모델 (Multi-input CNN)** |

### 📈 기존 최고 성능 연구(SOTA)와 비교

| 연구 | 클래스 수 | 데이터 종류 | 멀티모달 | 정확도 |
| :--- | :---: | :--- | :---: | :---: |
| **Ours (본 연구)** | **6** | **얼굴 + 자세 + 소리** | **✅** | **99.9%** |
| Kim et al. (2024) | 2 | 얼굴 (눈, 입) | ❌ | 99.84% |
| Das et al. (2024) | 2 | 이미지 + IoT 센서 | ✅ | 98.8% |
| Qin et al. (2022) | 10 | 운전자 이미지 (HOG) | ❌ | 99.87% |

## 🏗️ 아키텍처 (Architecture)

<div align="center">

```mermaid
graph TB
    A[운전자 얼굴 이미지<br/>227×227×3] --> D[Conv 계층 1<br/>96 필터]
    B[운전자 상태 이미지<br/>227×227×3] --> E[Conv 계층 1<br/>96 필터] 
    C[소리 스펙트로그램<br/>227×227×3] --> F[Conv 계층 1<br/>96 필터]
    
    D --> G[최대 풀링 1]
    E --> H[최대 풀링 1]
    F --> I[최대 풀링 1]
    
    G --> J[Conv 계층 2<br/>256 필터]
    H --> K[Conv 계층 2<br/>256 필터]
    I --> L[Conv 계층 2<br/>256 필터]
    
    J --> M[최대 풀링 2]
    K --> N[최대 풀링 2]
    L --> O[최대 풀링 2]
    
    M --> P[결합 (Concatenate)<br/>13×13×768]
    N --> P
    O --> P
    
    P --> Q[Conv 계층 3<br/>640 필터]
    Q --> R[Conv 계층 4<br/>640 필터]
    R --> S[Conv 계층 5<br/>384 필터]
    S --> T[완전 연결 계층<br/>4096 뉴런]
    T --> U[출력 계층<br/>6 클래스]
</div>
🚀 시작하기 (Getting Started)
사전 요구사항
code
Bash
Python 3.8+
TensorFlow 2.x
OpenCV 4.x
NumPy
Matplotlib
Adobe Audition (소리 스펙트로그램 생성용)
설치
리포지토리 클론
code
Bash
git clone https://github.com/your-username/driver-state-recognition.git
cd driver-state-recognition
의존성 설치
code
Bash
pip install -r requirements.txt
데이터셋 다운로드
code
Bash
# 데이터셋은 18,900개의 이미지로 구성되어 있습니다 (6개 클래스 × 3개 입력 × 1,050개 이미지)
python download_dataset.py
🔧 사용법 (Usage)
학습
code
Bash
python train.py --epochs 200 --learning_rate 0.001 --batch_size 32
추론
code
Bash
python predict.py --face_image path/to/face.jpg --state_image path/to/state.jpg --sound_file path/to/sound.wav
평가
code
Bash
python evaluate.py --model_path models/best_model.h5 --test_data data/test/
📁 데이터셋 구조 (Dataset Structure)
code
Code
data/
└── train/
│   ├── c0_정상/
│   │   ├── face/
│   │   ├── state/
│   │   └── sound/
│   ├── c1_기절/
│   ├── c2_졸음/
│   ├── c3_놀람/
│   ├── c4_분노/
│   └── c5_불안/
└── test/
    └── [train과 동일한 구조]
🎨 데이터 예시 (Data Examples)
<div align="center">
클래스	얼굴	자세	소리 스펙트로그램
정상	😐 정면 주시	🚗 올바른 자세	🔇 조용한 주변 소음
기절	😵 고개 숙임	💤 핸들에 기댐	💥 충돌음
졸음	😴 감은 눈/하품	😪 피곤한 자세	🥱 하품 소리
놀람	😲 크게 뜬 눈/입	😱 갑작스러운 움직임	📯 경적 소리
분노	😠 곁눈질	🤬 긴장된 자세	😤 분노 음성 + 경적
불안	😰 긴장된 표정	😟 불안한 자세	⚠️ 짧은 경적 + 감탄사
</div>
📈 성능 지표 (Performance Metrics)
학습 곡선
훈련 정확도: 약 150 에포크에서 99.9%로 수렴
검증 정확도: 99.9%에서 안정적으로 유지
손실: 과적합 없이 꾸준히 감소
모델 비교
본 연구의 접근법을 다음과 같은 모델들과 비교했습니다.

ResNet50: 81.0% (얼굴), 85.7% (자세)
Xception: 87.8% (얼굴), 76.8% (자세)
제안 모델 (CNN): 99.9% (멀티모달)
🔬 실험 환경 (Experimental Setup)
파라미터	값
학습률 (Learning Rate)	0.001
에포크 (Epochs)	200
배치 크기 (Batch Size)	32
최적화 함수 (Optimizer)	Adam
손실 함수 (Loss Function)	Categorical Crossentropy
학습/테스트 분할 (Train/Test Split)	70/30
총 샘플 수 (Total Samples)	18,900
🏆 핵심 기여 (Key Contributions)
🔬 새로운 멀티모달 접근법: 운전자 상태 인식을 위해 얼굴 표정, 운전 자세, 소리 데이터를 최초로 결합
📱 온디바이스 AI 최적화: 모바일 배포에 적합한 경량 CNN 아키텍처
🎯 포괄적인 분류 시스템: 신체적 상태와 감정 상태를 모두 포함하는 6가지 클래스 분류
📊 최고 수준의 성능: 기존 단일 모달 접근법을 뛰어넘는 **99.9%**의 정확도 달성
🔮 향후 연구 (Future Work)
비접촉 센서(EEG, ECG, EOG) 데이터 통합
실시간 처리 최적화
더욱 다양한 조건의 데이터셋 확장
모바일 애플리케이션 배포
차량 시스템과의 연동
👥 팀원 (Team)
한국성서대학교 컴퓨터소프트웨어학과

역할	담당자
연구 총괄	한진호 (교수)
데이터팀	이동연, 이정현
코딩팀	신수아, 여환서
레퍼런스팀	강지수, 김수아, 한기준
📚 논문 및 인용 (Citation)
본 연구 결과를 인용하실 경우 아래 정보를 사용해 주세요.

code
Bibtex
@article{shin2025driver,
  title={A study on driver state recognition using CNN-based multimodal multi-input learning},
  author={Shin, Sooah and Kang, Jisu and Kim, Sooah and Yeo, Hwanseo and Lee, Dongyeon and Lee, Jeongjyun and Han, Gijun and Han, Jinho},
  journal={Korean Journal of Artificial Intelligence},
  year={2025},
  publisher={KODISA \& KAIA}
}
📄 라이선스 (License)
본 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참고하세요.

🤝 기여하기 (Contributing)
기여를 환영합니다! Pull Request를 자유롭게 제출해주세요.

프로젝트를 Fork 하세요.
기능 브랜치를 생성하세요 (git checkout -b feature/AmazingFeature).
변경 사항을 커밋하세요 (git commit -m 'Add some AmazingFeature').
브랜치에 푸시하세요 (git push origin feature/AmazingFeature).
Pull Request를 열어주세요.
🙏 감사의 글 (Acknowledgments)
2024년 한국성서대학교 대학혁신지원사업
헌신적으로 기여해주신 모든 팀원
귀중한 피드백을 주신 연구 커뮤니티
<br>
<div align="center">
<strong>이 프로젝트가 도움이 되셨다면 ⭐ Star를 눌러주세요!</strong>
<br>
<strong>❤️ 한국성서대학교 팀 제작</strong>
</div>
```
