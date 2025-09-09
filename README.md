CNN 기반 멀티모달 다중입력 학습을 이용한 운전자 상태 인식 연구
Show Image
Show Image
Show Image
Show Image
📚 목차

📋 프로젝트 개요
🏆 연구 성과
🔍 연구 배경
🏗️ 시스템 구조
📊 분류 클래스
🔬 실험 결과
💻 설치 및 실행
📁 프로젝트 구조
📈 연구 기여도
👥 팀 구성
📚 참고문헌
📄 라이선스
📞 문의
🙏 감사의 말

📋 프로젝트 개요
본 프로젝트는 CNN 기반 다중입력 학습을 통해 99.9%의 정확도로 운전자 상태를 인식하는 멀티모달 운전자 상태 인식 시스템을 제안합니다. 온디바이스 AI 적용을 고려하여 일반적인 모바일 기기에서 사용 가능한 간단한 센서(카메라, 마이크)만을 활용합니다.
🎯 주요 특징

99.9% 정확도 달성
멀티모달 학습 (시각 + 청각 데이터 융합)
온디바이스 AI 호환 가능한 경량 CNN 구조
6가지 클래스 분류 (운전자 상태 3가지 + 감정 3가지)
실시간 처리 가능한 안전 시스템

🏆 연구 성과

학술지 게재: 한국인공지능학회 논문지 (KJAI)
연구팀: 학부생 7명 + 지도교수 1명
연구기간: 10개월 (2024.05 - 2025.01)
지원사업: 2024년 한국성서대학교 대학혁신지원사업

🔍 연구 배경
운전자 상태 모니터링은 도로 안전을 위해 매우 중요합니다. 기존 연구들의 한계점:

복잡한 센서 사용 (EEG, ECG, EOG 등) - 일상적 사용 어려움
단일 모달리티 중심 - 제한적인 정확도
고가의 장비 필요 - 일반 차량 적용 어려움

본 연구는 카메라와 마이크만을 사용하여 온디바이스 배포에 최적화된 단순한 CNN 구조로 이러한 한계를 해결합니다.
🏗️ 시스템 구조
다중입력 CNN 아키텍처
입력 1: 운전자 얼굴 이미지 (227×227×3)
입력 2: 운전자 상태 이미지 (227×227×3)  
입력 3: 소리 스펙트로그램 (227×227×3)
           ↓
    [Conv Layer 1] × 3
           ↓
    [Max Pooling 1] × 3
           ↓
    [Conv Layer 2] × 3
           ↓
    [Max Pooling 2] × 3
           ↓
      [Concatenation]
           ↓
  [Conv Layers 3-5] + [Affine Layers 1-3]
           ↓
    [Softmax - 6 Classes]
📊 분류 클래스
운전자 상태 (3가지)

c0: 정상 - 전방 주시, 조용한 환경음
c1: 기절 - 고개 숙임, 핸들에 머리 기댄 상태, 충격음
c2: 졸림 - 눈 감음/하품, 하품 소리

운전자 감정 (3가지)

c3: 놀람 - 눈/입 크게 벌림, 경적 소리
c4: 분노 - 측면 응시/화난 표정, 경적+화난 목소리
c5: 불안 - 긴장된 표정, 가벼운 경적+불안한 감탄사

🔬 실험 결과
데이터셋

총 데이터: 18,900장 (6클래스 × 3입력 × 1,050장)
훈련 데이터: 12,600장 (70%)
테스트 데이터: 6,300장 (30%)
이미지 크기: 227×227 픽셀
소리 길이: 약 2초

성능 비교
모델데이터셋정확도 (%)ResNet50Driver face81.0Driver state85.7XceptionDriver face87.8Driver state76.8Our Multi-input CNNDriver face80.7Driver state68.6Dual input (face+state)75.8Triple input (face+state+sound)99.9
핵심 발견
✅ 멀티모달 다중입력 학습 시 정확도가 현저히 향상됨
✅ 소리 데이터 추가가 성능 향상의 핵심 요인
✅ 단순한 CNN 구조로도 복잡한 모델과 경쟁 가능한 성능
💻 설치 및 실행
요구사항
bashPython >= 3.8
TensorFlow >= 2.0
OpenCV >= 4.0
NumPy
Matplotlib
Adobe Audition (스펙트로그램 생성용)
설치
bashgit clone https://github.com/yourusername/driver-state-recognition.git
cd driver-state-recognition
pip install -r requirements.txt
학습 실행
bashpython train.py --epochs 200 --lr 0.001 --batch_size 32
추론 실행
bashpython inference.py --model_path models/best_model.h5 --input_data test_samples/
📁 프로젝트 구조
driver-state-recognition/
├── data/
│   ├── driver_face/          # 운전자 얼굴 이미지
│   ├── driver_state/         # 운전자 상태 이미지  
│   └── sound_spectrogram/    # 소리 스펙트로그램
├── models/
│   ├── multi_input_cnn.py    # 다중입력 CNN 모델
│   ├── resnet50_model.py     # ResNet50 비교 모델
│   └── xception_model.py     # Xception 비교 모델
├── utils/
│   ├── data_preprocessing.py # 데이터 전처리
│   ├── spectrogram_utils.py  # 스펙트로그램 생성
│   └── visualization.py      # 결과 시각화
├── train.py                  # 훈련 스크립트
├── inference.py              # 추론 스크립트
└── requirements.txt          # 의존성 목록
📈 연구 기여도
학술적 기여

멀티모달 CNN 제안: 얼굴, 운전자세, 음성 데이터를 활용한 99.9% 정확도 달성
6가지 상태 분류: 기존 연구 대비 더 포괄적인 운전자 상태 및 감정 인식
온디바이스 AI 최적화: 단순한 센서와 경량 구조로 실용적 적용 가능성 제시

실용적 가치

🚗 자율주행 전환기 운전자 안전 시스템에 활용 가능
📱 모바일 기기 호환 일반 소비자도 쉽게 사용 가능
⚡ 실시간 처리 즉각적인 위험 상황 대응 가능

👥 팀 구성
이름역할강지수레퍼런스 조사 및 논문 작성김수아레퍼런스 조사 및 논문 작성신수아ResNet50 코드 연구 및 실행여환서Xception 코드 연구 및 실행이동연데이터 생성 및 테이블 제작이정현데이터 생성 및 테이블 제작한기준레퍼런스 조사 및 논문 작성한진호지도교수 (책임교수)
📚 참고문헌
주요 참고 논문들은 references/ 폴더에서 확인할 수 있습니다.
📄 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
📞 문의

논문 관련: hjinob@bible.ac.kr (한진호 교수)
코드 관련: Issues 페이지에 문의해주세요

🙏 감사의 말
본 연구는 2024년 한국성서대학교 대학혁신지원사업의 지원을 받아 수행되었습니다. 10개월간 함께 연구에 참여한 모든 학부생 연구원들과 지원해주신 대학에 감사드립니다.
