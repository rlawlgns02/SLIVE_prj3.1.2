# AI 모델 비교 시스템

여러 딥러닝 모델(ResNet, DenseNet, EfficientNet, Baseline)의 성능을 비교하고 리더보드로 시각화하는 시스템입니다.

## 주요 기능

### 1. 데이터 수집
- 웹캠을 통한 실시간 손 랜드마크 수집
- MediaPipe Hands 기반 21개 포인트 감지
- 수동/자동 수집 모드 지원
- 제스처별 데이터 관리

### 2. 모델 학습 및 비교
- **5가지 모델 지원:**
  - Baseline (Flatten + Dense)
  - SLIVE (Original SLIVE Model)
  - ResNet (Residual Network)
  - DenseNet (Densely Connected Network)
  - EfficientNet (Efficient Neural Network)

### 3. 성능 지표
- **학습 속도**: 전체 학습 시간, 에포크당 평균 시간
- **정확도**: 학습 정확도, 검증 정확도
- **추론 속도**: 평균 추론 시간 (ms)
- **컴퓨팅 자원**:
  - 메모리 사용량 (피크 메모리, 메모리 증가량)
  - 모델 크기 (MB)
  - FLOPs (연산량)
  - GPU 메모리 사용량 (사용 가능한 경우)

### 4. 리더보드
- 실시간 성능 비교 차트
- 정렬 가능한 상세 테이블
- 모델별 파라미터 수 비교

## 설치 방법

### 1. 의존성 설치

```bash
cd model_comparison
pip install -r requirements_comparison.txt
```

### 2. 서버 실행

```bash
python app_comparison.py
```

서버가 시작되면 다음 주소로 접속:
- 데이터 수집: http://localhost:5001/datacollector
- 리더보드: http://localhost:5001/leaderboard

## 사용 방법

### 1단계: 데이터 수집

1. **데이터 수집 페이지** (http://localhost:5001/datacollector) 접속
2. "카메라 시작" 버튼 클릭
3. 제스처 이름 입력 (예: "안녕하세요")
4. "녹화 시작" 버튼 클릭하고 동작 수행
5. 충분한 데이터 수집 후 "녹화 중지"
6. "저장하기" 버튼 클릭
7. 여러 제스처에 대해 반복 (최소 3-5개 제스처, 각 100개 이상 권장)

### 2단계: 모델 학습

1. **리더보드 페이지** (http://localhost:5001/leaderboard) 접속
2. 학습할 모델 선택 (Baseline, SLIVE, ResNet, DenseNet, EfficientNet)
3. 학습 파라미터 설정:
   - 에포크 수: 20 (기본값)
   - 배치 크기: 32 (기본값)
   - 학습률: 0.001 (기본값)
4. "학습 시작" 버튼 클릭
5. 학습 진행 상황 확인
6. 다른 모델들도 동일하게 학습

### 3단계: 성능 비교

리더보드에서 자동으로 표시되는 항목:
- **정확도 비교 차트**: 각 모델의 학습/검증 정확도
- **학습 시간 비교 차트**: 각 모델의 학습 소요 시간
- **추론 속도 비교 차트**: 각 모델의 추론 속도
- **메모리 사용량 비교 차트**: 피크 메모리 및 메모리 증가량
- **모델 크기 비교 차트**: 저장된 모델 파일 크기
- **FLOPs 비교 차트**: 각 모델의 연산량 (Floating Point Operations)
- **상세 테이블**: 모든 메트릭을 포함한 상세 정보

## 프로젝트 구조

```
model_comparison/
├── app_comparison.py          # Flask 백엔드 서버
├── requirements_comparison.txt # Python 의존성
├── README.md                  # 이 파일
│
├── models/                    # 모델 클래스
│   ├── __init__.py
│   ├── baseline.py           # Baseline 모델 (Flatten + Dense)
│   ├── slive.py              # SLIVE 모델
│   ├── resnet.py             # ResNet 모델
│   ├── densenet.py           # DenseNet 모델
│   └── efficientnet.py       # EfficientNet 모델
│
├── utils/                    # 유틸리티 모듈
│   ├── __init__.py
│   └── resource_monitor.py   # 컴퓨팅 자원 모니터링
│
├── templates/                # HTML 템플릿
│   ├── datacollector.html    # 데이터 수집 페이지
│   └── leaderboard.html      # 리더보드 페이지
│
├── static/                   # 정적 파일
│   ├── css/
│   │   └── comparison.css    # 스타일시트
│   └── js/
│       ├── datacollector.js  # 데이터 수집 로직
│       └── leaderboard.js    # 리더보드 로직
│
├── data/                     # 수집된 데이터
│   └── comparison_data.json  # 손 랜드마크 데이터
│
├── results/                  # 학습 결과
│   └── leaderboard.json      # 리더보드 데이터
│
└── trained_models/           # 학습된 모델 파일
    └── *.h5                  # Keras 모델 파일
```

## API 엔드포인트

### 데이터 관리
- `POST /api/data/save` - 데이터 저장
- `GET /api/data/stats` - 데이터 통계 조회
- `POST /api/data/reset` - 데이터 초기화

### 모델 관리
- `GET /api/models/list` - 사용 가능한 모델 목록
- `POST /api/train` - 모델 학습

### 리더보드
- `GET /api/leaderboard` - 리더보드 조회 (정렬 옵션)
- `POST /api/leaderboard/clear` - 리더보드 초기화
- `DELETE /api/leaderboard/delete/<index>` - 항목 삭제

## 모델 아키텍처 설명

### Baseline (Flatten + Dense)
- 기본 비교 모델
- Flatten → Dense(256) → Dense(128) → Dense(64) → Output
- 빠른 학습, 적은 파라미터

### SLIVE (Original)
- 원본 SLIVE 프로젝트의 모델
- Flatten → Dense(256) → Dense(128) → Dense(64) → Output
- Baseline과 동일한 구조지만 원본 프로젝트 버전
- 규칙 기반 + 딥러닝 하이브리드 접근법

### ResNet (Residual Network)
- Skip Connection을 사용한 깊은 네트워크
- Conv1D 기반 Residual Block
- 그래디언트 소실 문제 해결

### DenseNet (Densely Connected Network)
- 모든 레이어가 서로 연결
- Dense Block과 Transition Block
- 효율적인 특징 재사용

### EfficientNet
- Mobile Inverted Bottleneck (MBConv) 블록
- Squeeze-and-Excitation 블록
- 효율성과 정확도의 균형

## 성능 최적화 팁

1. **데이터 수집**: 각 제스처당 최소 100-200개 샘플 수집
2. **데이터 다양성**: 다양한 조명, 각도, 속도로 수집
3. **에포크 수**: 20-50 에포크 (과적합 방지)
4. **배치 크기**: 32 또는 64 (GPU 메모리에 따라)
5. **학습률**: 0.001 시작, 필요시 조정

## 문제 해결

### 카메라가 작동하지 않음
- 브라우저 카메라 권한 확인
- HTTPS 환경에서만 작동 (로컬호스트 제외)

### 학습이 느림
- GPU 사용 확인: `tensorflow-gpu` 설치
- 배치 크기 줄이기
- 더 작은 모델 사용 (Baseline)

### 정확도가 낮음
- 더 많은 데이터 수집
- 에포크 수 증가
- 데이터 품질 확인

### 메모리 부족
- 배치 크기 줄이기 (16 또는 8)
- 더 작은 모델 사용
- 에포크 수 줄이기

## 기존 SLIVE 시스템과의 차이점

| 항목 | 기존 SLIVE | 모델 비교 시스템 |
|------|-----------|-----------------|
| 포트 | 5000 | 5001 |
| 모델 | 1개 (SLIVE) | 5개 (Baseline, SLIVE, ResNet, DenseNet, EfficientNet) |
| 목적 | 실시간 수어 통역 | 모델 성능 비교 |
| 데이터 | collected_data.json | comparison_data.json |
| UI | 통합 워크스페이스 | 데이터 수집 + 리더보드 |

## 기술 스택

**백엔드:**
- Python 3.8+
- Flask 3.0.0
- TensorFlow 2.15.0
- NumPy, scikit-learn
- psutil (리소스 모니터링)

**프론트엔드:**
- HTML5, CSS3, JavaScript (ES6+)
- MediaPipe Hands (손 감지)
- Chart.js (데이터 시각화)

**딥러닝:**
- TensorFlow/Keras
- 1D CNN 아키텍처
- 손 랜드마크 기반 분류

## 컴퓨팅 자원 비교 기능

이 시스템은 모델의 성능뿐만 아니라 **컴퓨팅 자원 소모량**도 함께 측정하여 비교합니다.

### 측정 항목

#### 1. 메모리 사용량
- **피크 메모리 (Peak Memory)**: 학습 중 최대 메모리 사용량 (MB)
- **메모리 증가량**: 학습 시작 전후의 메모리 차이 (MB)
- **평균 에포크 메모리**: 각 에포크마다의 평균 메모리 사용량 (MB)
- **추론 메모리**: 추론 시 메모리 사용량 (MB)

#### 2. 모델 크기
- 저장된 모델 파일의 크기 (MB)
- 배포 및 저장 공간 효율성 평가에 활용

#### 3. FLOPs (Floating Point Operations)
- 모델의 연산량 측정
- 레이어별 연산량 계산 (Dense, Conv1D, Conv2D, BatchNorm 등)
- 백만 단위로 표시 (예: 1.5M FLOPs)

#### 4. 추론 속도 상세 통계
- 평균 추론 시간 (ms)
- 표준편차, 최소/최대 추론 시간
- 100회 반복 측정을 통한 정확한 성능 평가

#### 5. GPU 메모리 (선택)
- GPU 사용 가능 시 GPU 메모리 사용량 측정
- pynvml 라이브러리 설치 시 활성화

### 리소스 효율성 평가

모델 선택 시 다음 요소들을 종합적으로 고려할 수 있습니다:

1. **정확도 vs 속도**: 높은 정확도를 위해 얼마나 많은 시간이 소요되는가?
2. **정확도 vs 메모리**: 높은 정확도를 위해 얼마나 많은 메모리가 필요한가?
3. **모델 크기 vs 성능**: 작은 모델로도 좋은 성능을 낼 수 있는가?
4. **연산량 vs 추론 속도**: FLOPs와 실제 추론 속도의 관계는?

### 실시간 모니터링

학습 중에도 실시간으로 리소스 사용량을 모니터링:
- 에포크마다 메모리 사용량 추적
- 학습 진행 상황과 함께 리소스 정보 표시
- 학습 완료 후 종합 리포트 제공

## 라이선스

이 프로젝트는 기존 SLIVE 프로젝트의 확장입니다.

## 문의

문제가 발생하거나 개선 제안이 있으시면 이슈를 등록해주세요.

---

**제작**: SLIVE 프로젝트 팀
**버전**: 1.1.0
**최종 업데이트**: 2024-12-05
