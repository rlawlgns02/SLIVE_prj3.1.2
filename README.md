# SLIVE

수어 실시간 통역 시스템

## 📋 프로젝트 소개

SLIVE는 AI 기반 한국 수어 인식 및 실시간 통역 시스템입니다. MediaPipe와 TensorFlow.js를 활용하여 웹캠을 통해 수어를 인식하고 텍스트로 변환합니다.

이 프로젝트는 **두 가지 주요 시스템**으로 구성되어 있습니다:

### 1. SLIVE 기본 시스템 (`ksl_project Up/`)
- TensorFlow.js 기반 브라우저 내 수어 인식
- 실시간 통역 및 대화 번역
- TTS(음성 출력) 기능
- 단일 모델 학습 및 관리

### 2. 모델 비교 시스템 (`model_comparison/`)
- 4가지 CNN 모델 (Baseline, ResNet, DenseNet, EfficientNet) 성능 비교
- Python/TensorFlow 기반 서버 사이드 학습
- 리더보드 및 실시간 모델 성능 비교
- 모델별 정확도, 학습 시간, 추론 속도 분석

## ✨ 주요 기능

### SLIVE 기본 시스템

#### 1. 데이터 수집 📊
- 웹캠을 통한 실시간 손 동작 캡처
- 30개 이상의 수어 제스처 지원
- 자동 저장 및 진행률 추적
- 목표 데이터셋 수량 설정 가능

#### 2. 모델 학습 🧠
- TensorFlow.js 기반 브라우저 내 딥러닝 모델
- 실시간 학습 진행률 및 정확도 표시
- 다양한 학습 프리셋 제공
- 학습 중 실시간 그래프 표시

#### 3. 실시간 통역 🤟
- 웹캠 기반 실시간 수어 인식
- 높은 정확도의 인식 결과 (신뢰도 93% 이상)
- 상위 5개 예측 결과 표시
- 인식 기록 및 신뢰도 표시
- 실시간 음성 출력(TTS) 기능: 인식된 결과를 브라우저 음성으로 재생 (번역/대화 탭 별도 토글 제공)

#### 4. 대화 번역 💬
- 연속적인 수어 제스처를 문장으로 변환
- 낮은 신뢰도 제스처 자동 필터링
- 음성 출력 지원

#### 5. 모델 관리 ⚙️
- 여러 모델 저장 및 관리
- 모델 이름 변경 및 삭제
- 모델 성능 비교
- 모델 경쟁 모드

### 모델 비교 시스템

#### 1. 다중 모델 지원 🔬
- **Baseline**: 기본 Dense Layer 모델
- **ResNet**: Residual Network 기반 모델
- **DenseNet**: Dense Connection 기반 모델
- **EfficientNet**: 효율적인 스케일링 모델

#### 2. 성능 비교 📊
- 학습 정확도 및 검증 정확도 비교
- 학습 시간 및 에폭당 시간 측정
- 추론 속도 (ms) 비교
- 모델 파라미터 수 분석

#### 3. 리더보드 🏆
- 모든 학습 결과 저장 및 비교
- 정렬 기능 (정확도, 속도, 학습 시간)
- 학습 기록 관리 (삭제, 초기화)
- **실시간 학습 로그**: 터미널 스타일의 로그 창에서 에포크별 진행 상황 확인
- **실시간 학습 그래프**: 에포크 1부터 N까지 한 에포크 완료될 때마다 즉시 그래프 갱신
- **그래프 다운로드**: 학습 완료 후 버튼 클릭으로 정확도 및 손실 그래프를 PNG 이미지로 저장

#### 4. 실시간 비교 ⚡
- 여러 모델 동시 추론
- 모델별 예측 결과 및 신뢰도 실시간 표시
- 추론 속도 실시간 측정

## 🚀 설치 및 실행

### 필요 사항
- Python 3.8 이상
- 웹캠
- 최신 웹 브라우저 (Chrome, Firefox, Edge 등)

### 설치

```bash
# 저장소 클론
git clone https://github.com/rlawlgns02/SLIVE_prj

# 프로젝트 디렉토리로 이동
cd SLIVE_prj

# SLIVE 기본 시스템 패키지 설치
pip install flask flask-cors

# 모델 비교 시스템 패키지 설치 (선택사항)
pip install tensorflow scikit-learn numpy
```

### 실행

#### SLIVE 기본 시스템

```bash
# ksl_project Up 디렉토리로 이동
cd "ksl_project Up"

# Flask 서버 시작
python app_flask.py
```

브라우저에서 `http://localhost:5000` 접속

**사용 가능한 기능:**
- 데이터 수집
- 모델 학습
- 실시간 통역 (TTS 음성 출력 지원)
- 대화 번역 (TTS 음성 출력 지원)
- 모델 관리

#### 모델 비교 시스템

```bash
# model_comparison 디렉토리로 이동
cd model_comparison

# Flask 서버 시작
python app_comparison.py
```

브라우저에서 `http://localhost:5001` 접속

**사용 가능한 기능:**
- 데이터 수집: `http://localhost:5001/datacollector`
- 리더보드: `http://localhost:5001/leaderboard`
- 실시간 비교: `http://localhost:5001/live`

### 주요 기능 사용 방법

#### TTS(음성 재생) - SLIVE 기본 시스템
1. 실시간 통역 탭의 `음성(TTS)` 토글을 활성화하면 인식된 레이블이 음성으로 재생됩니다.
2. 대화 번역 탭에서도 `음성(TTS)` 토글을 활성화하면 인식 단어 및 생성된 문장이 음성으로 재생됩니다.
3. 브라우저가 Web Speech API(SpeechSynthesis)를 지원하지 않을 경우 토글이 비활성화됩니다.
4. 토글 상태는 로컬 스토리지에 저장되어 다음 방문 시에도 유지됩니다.

#### 신뢰도 필터링
- 대화 번역 및 실시간 통역 모두 **최소 신뢰도 93% (0.93)** 기준 적용
- 신뢰도가 낮은 제스처는 자동으로 무시되며, UI에 `학습되지 않은 동작입니다` 메시지 표시

## 📁 프로젝트 구조

```
SLIVE/
├── ksl_project Up/              # SLIVE 기본 시스템
│   ├── app_flask.py            # Flask 백엔드 서버 (포트 5000)
│   ├── templates/
│   │   └── workspace.html      # 통합 워크스페이스
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css      # 스타일시트
│   │   ├── js/
│   │   │   ├── model.js        # TensorFlow.js 모델 클래스
│   │   │   └── viewer3d.js     # 3D 시각화
│   │   └── workspace.js        # 워크스페이스 로직
│   ├── trained-model/          # 학습된 모델 저장 (git 제외)
│   └── data/                   # 학습 데이터 (git 제외)
│
├── model_comparison/            # 모델 비교 시스템
│   ├── app_comparison.py       # Flask 백엔드 서버 (포트 5001)
│   ├── models/                 # CNN 모델 구현
│   │   ├── __init__.py
│   │   ├── baseline.py         # Baseline Dense 모델
│   │   ├── resnet.py           # ResNet 모델
│   │   ├── densenet.py         # DenseNet 모델
│   │   └── efficientnet.py     # EfficientNet 모델
│   ├── templates/
│   │   ├── datacollector.html  # 데이터 수집 페이지
│   │   ├── leaderboard.html    # 리더보드 페이지
│   │   └── live.html           # 실시간 비교 페이지
│   ├── static/
│   │   └── js/
│   │       ├── datacollector.js
│   │       ├── leaderboard.js
│   │       └── live.js
│   ├── data/                   # 비교용 학습 데이터 (git 제외)
│   ├── results/                # 리더보드 결과 (git 제외)
│   └── trained_models/         # 학습된 모델들 (git 제외)
│
└── README.md
```

## 🎯 지원되는 수어 제스처

- **인사**: 안녕하세요, 감사합니다, 미안합니다, 잘가
- **감정**: 좋아요, 싫어요, 사랑해요
- **동작**: 확인, 평화, 멈춰, 와, 가
- **숫자**: 하나~열 (1-10)
- **기타**: 주먹, 가리키기, 물, 밥, 도와주세요, 전화, 락

## 🛠️ 기술 스택

### SLIVE 기본 시스템
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**:
  - TensorFlow.js (브라우저 내 모델 학습 및 추론)
  - MediaPipe Hands (손 랜드마크 감지)
- **기타**:
  - Chart.js (데이터 시각화)
  - Web Speech API (TTS 음성 출력)

### 모델 비교 시스템
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**:
  - TensorFlow (Python, 서버 사이드 학습)
  - Scikit-learn (데이터 전처리, Label Encoding)
  - MediaPipe Hands (손 랜드마크 감지)
- **기타**: NumPy (데이터 처리)

## 📊 모델 구조

### SLIVE 기본 시스템 (TensorFlow.js)

**Input**: 21 hand landmarks × 3 coordinates (x, y, z) = 63 features

**Architecture**:
- Flatten Layer (63 → 63)
- Dense (256) + BatchNormalization + Dropout (0.3)
- Dense (128) + BatchNormalization + Dropout (0.3)
- Dense (64) + BatchNormalization + Dropout (0.3)
- Dense (num_classes) + Softmax

**Training**:
- Optimizer: Adam (학습률 조정 가능)
- Loss: Categorical Crossentropy
- Metrics: Accuracy

### 모델 비교 시스템 (TensorFlow Python)

**Input**: 21 hand landmarks × 3 coordinates (x, y, z) = Shape (21, 3)

#### 1. Baseline Model
- Dense Layer 기반 기본 모델
- 구조: Dense(256) → Dense(128) → Dense(64) → Output
- 경량화되어 빠른 추론 속도

#### 2. ResNet Model
- Residual Connections를 활용한 모델
- Skip Connection으로 Gradient Vanishing 문제 해결
- 더 깊은 네트워크 학습 가능

#### 3. DenseNet Model
- Dense Connection 구조
- 각 레이어가 이전 모든 레이어와 연결
- 특징 재사용으로 파라미터 효율성 향상

#### 4. EfficientNet Model
- Compound Scaling 기법 적용
- Width, Depth, Resolution을 균형있게 확장
- 높은 정확도와 효율성 제공

**공통 설정**:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Data Preprocessing: Wrist 기준 정규화 + 스케일 정규화

## 🎮 사용 방법

### SLIVE 기본 시스템

#### 1. 데이터 수집
1. "데이터 수집" 탭으로 이동
2. 수집할 제스처 선택
3. "카메라 시작" 클릭
4. "녹화 시작" 클릭하여 데이터 수집
5. 충분한 데이터 수집 후 저장

#### 2. 모델 학습
1. "모델 학습" 탭으로 이동
2. 학습 파라미터 설정 (또는 프리셋 선택)
3. 모델 이름 입력 (선택사항)
4. "학습 시작" 클릭
5. 브라우저 내에서 학습 진행 (실시간 그래프 확인)
6. 학습 완료 후 자동 저장

#### 3. 실시간 통역
1. "실시간 통역" 탭으로 이동
2. 사용할 모델 선택
3. "통역 시작" 클릭
4. 수어 제스처 수행
5. 인식 결과 및 신뢰도 확인
6. TTS 토글로 음성 출력 활성화 가능

#### 4. 대화 번역
1. "대화 번역" 탭으로 이동
2. 연속적인 수어 제스처 수행
3. 자동으로 문장 구성
4. 생성된 문장 확인 및 음성 출력

### 모델 비교 시스템

#### 1. 데이터 수집
1. `/datacollector` 페이지 접속
2. 수집할 제스처 선택
3. "카메라 시작" 후 데이터 수집
4. 서버에 자동 저장

#### 2. 모델 학습 및 비교
1. `/leaderboard` 페이지 접속
2. 학습할 모델 선택 (Baseline, ResNet, DenseNet, EfficientNet)
3. 학습 파라미터 설정 (Epochs, Batch Size, Learning Rate)
4. "학습 시작" 클릭
5. **실시간 학습 진행 상황 표시**:
   - 학습 시작과 동시에 진행 상황 카드가 나타남
   - 에포크 진행률을 프로그레스 바로 표시
   - 실시간 통계 (4개 박스):
     - 학습 정확도
     - 검증 정확도
     - 학습 손실
     - 검증 손실
   - 한 에포크 완료될 때마다 즉시 업데이트
6. **실시간 학습 로그 표시**:
   - 터미널 스타일의 학습 로그 창
   - 각 에포크마다 실시간으로 로그 출력:
     ```
     [14:35:12] Epoch 1/20: 정확도 87.34%
     [14:35:18] Epoch 2/20: 정확도 89.56%
     ...
     ```
   - 타임스탬프 포함으로 진행 시간 추적 가능
   - "로그 지우기" 버튼으로 언제든지 초기화
7. **실시간 그래프 표시**:
   - 학습 시작과 동시에 독립된 그래프 섹션이 나타남
   - **에포크 1부터 N까지 한 에포크 완료될 때마다 즉시 그래프에 반영**
   - 왼쪽: 정확도 그래프 (학습 vs 검증)
   - 오른쪽: 손실 그래프 (학습 vs 검증)
   - 학습 중에도 진행 상황을 가시적으로 확인 가능
8. 학습 완료 후:
   - 리더보드에 결과 자동 추가
   - **진행 상황, 로그, 그래프 모두 유지됨** (새로운 학습 시작 전까지 보존)
   - 다운로드 버튼 활성화 (상단 우측)
9. **그래프 저장**:
   - "📊 정확도 그래프 저장" 버튼 클릭 → `{모델명}_accuracy_{타임스탬프}.png`
   - "📉 손실 그래프 저장" 버튼 클릭 → `{모델명}_loss_{타임스탬프}.png`
   - 다운로드 위치: 브라우저 기본 다운로드 폴더
10. 여러 모델 학습 후 성능 비교

#### 3. 실시간 모델 비교
1. `/live` 페이지 접속
2. 비교할 모델들 선택 (여러 개 선택 가능)
3. "모델 로드" 클릭
4. "카메라 시작" 클릭
5. 수어 제스처 수행
6. 각 모델의 예측 결과 및 추론 속도 실시간 확인


## 🔍 시스템 비교

| 특성 | SLIVE 기본 시스템 | 모델 비교 시스템 |
|------|------------------|----------------|
| **실행 환경** | 브라우저 (Client-side) | 서버 (Server-side) |
| **프레임워크** | TensorFlow.js | TensorFlow (Python) |
| **포트** | 5000 | 5001 |
| **학습 속도** | 느림 (브라우저 제약) | 빠름 (서버 GPU 활용 가능) |
| **모델 종류** | 단일 Dense 모델 | 4가지 CNN 모델 |
| **주요 용도** | 실시간 통역 및 대화 | 모델 성능 비교 및 분석 |
| **배포** | 간편 (브라우저만 필요) | TensorFlow 설치 필요 |
| **TTS 지원** | ✅ (Web Speech API) | ❌ |
| **실시간 비교** | 동일 모델 동시 추론 | 여러 모델 동시 추론 |

## 💡 활용 시나리오

### 개발 및 연구
1. **모델 비교 시스템**으로 여러 모델 학습 및 성능 비교
2. 최적 모델 선택 후 **SLIVE 기본 시스템**으로 배포
3. 실제 사용자 대상 실시간 통역 서비스 제공

### 교육 및 데모
1. **SLIVE 기본 시스템**으로 빠른 프로토타입 및 데모
2. 브라우저만으로 즉시 실행 가능
3. TTS 기능으로 몰입감 있는 체험 제공

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 👨‍💻 개발자

HandCode Team

## 🙏 감사의 말

- **MediaPipe** - 손 랜드마크 감지 기술 제공
- **TensorFlow / TensorFlow.js** - 머신러닝 프레임워크
- **Flask** - 웹 프레임워크
- **Scikit-learn** - 데이터 전처리 도구

---

**Note**: 이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.



