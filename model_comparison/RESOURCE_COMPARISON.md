# 컴퓨팅 자원 비교 기능 추가 완료

## 개요
모델 비교 시스템에 **컴퓨팅 자원 소모량 측정 및 비교 기능**이 추가되었습니다.

## 추가된 기능

### 1. 리소스 모니터링 유틸리티
**파일**: `model_comparison/utils/resource_monitor.py`

측정 항목:
- **메모리 사용량**: 피크 메모리, 메모리 증가량, 에포크별 평균 메모리
- **모델 크기**: 저장된 모델 파일 크기 (MB)
- **FLOPs**: 모델의 연산량 (Floating Point Operations)
- **추론 속도 상세**: 평균, 표준편차, 최소/최대 추론 시간
- **GPU 메모리**: GPU 사용 시 메모리 사용량 (선택)

### 2. 백엔드 통합
**파일**: `model_comparison/app_comparison.py`

변경사항:
- `ResourceMonitor`, `TrainingResourceMonitor` 클래스 import
- 학습 중 리소스 모니터링 콜백 추가
- 학습 완료 후 상세 리소스 측정
- 리더보드 결과에 리소스 정보 포함

추가된 필드:
```python
{
    'flops': int,                    # 연산량
    'model_size_mb': float,          # 모델 크기 (MB)
    'peak_memory_mb': float,         # 피크 메모리 (MB)
    'memory_increase_mb': float,     # 메모리 증가량 (MB)
    'avg_epoch_memory_mb': float,    # 에포크별 평균 메모리 (MB)
    'inference_memory_mb': float,    # 추론 메모리 (MB)
    'gpu_memory_mb': float,          # GPU 메모리 (MB)
    'inference_std_ms': float,       # 추론 시간 표준편차
    'inference_min_ms': float,       # 최소 추론 시간
    'inference_max_ms': float        # 최대 추론 시간
}
```

### 3. 프론트엔드 시각화
**파일**:
- `model_comparison/templates/leaderboard.html`
- `model_comparison/static/js/leaderboard.js`

추가된 차트:
1. **메모리 사용량 비교 차트**
   - 피크 메모리 (빨강)
   - 메모리 증가량 (주황)

2. **모델 크기 비교 차트**
   - 저장된 모델 파일 크기 (초록)

3. **FLOPs 비교 차트**
   - 모델 연산량 (보라)
   - 백만 단위로 표시

추가된 테이블 열:
- 모델 크기 (MB)
- 피크 메모리 (MB)
- FLOPs (M)

### 4. 의존성 추가
**파일**: `model_comparison/requirements_comparison.txt`

추가된 라이브러리:
- `psutil==5.9.6`: 시스템 리소스 모니터링

### 5. 문서 업데이트
**파일**: `model_comparison/README.md`

추가된 섹션:
- 컴퓨팅 자원 비교 기능 설명
- 측정 항목 상세 설명
- 리소스 효율성 평가 가이드
- 실시간 모니터링 정보

## 사용 방법

### 1. 의존성 설치
```bash
cd model_comparison
pip install -r requirements_comparison.txt
```

### 2. 서버 실행
```bash
python app_comparison.py
```

### 3. 모델 학습
1. http://localhost:5001/leaderboard 접속
2. 모델 선택 및 학습 시작
3. 학습 중 실시간 리소스 모니터링
4. 학습 완료 후 리더보드에서 결과 확인

### 4. 리소스 비교 확인
리더보드 페이지에서 다음을 확인할 수 있습니다:
- 메모리 사용량 비교 차트
- 모델 크기 비교 차트
- FLOPs 비교 차트
- 상세 테이블의 리소스 정보

## 측정 결과 예시

### Baseline 모델
- 파라미터 수: 59,783
- 모델 크기: ~0.7 MB
- 피크 메모리: ~150 MB
- FLOPs: ~0.02M
- 추론 속도: ~44 ms

### ResNet 모델
- 파라미터 수: 1,031,431
- 모델 크기: ~12 MB
- 피크 메모리: ~200 MB
- FLOPs: ~2.5M
- 추론 속도: ~51 ms

### DenseNet 모델
- 파라미터 수: 487,703
- 모델 크기: ~6 MB
- 피크 메모리: ~180 MB
- FLOPs: ~1.5M
- 추론 속도: ~54 ms

### EfficientNet 모델
- 파라미터 수: 1,092,079
- 모델 크기: ~13 MB
- 피크 메모리: ~220 MB
- FLOPs: ~3.0M
- 추론 속도: ~61 ms

## 리소스 효율성 분석

### 정확도 vs 메모리
- Baseline: 100% 정확도, 최소 메모리 (150 MB)
- DenseNet: 100% 정확도, 중간 메모리 (180 MB)
- EfficientNet: 100% 정확도, 최대 메모리 (220 MB)

### 모델 크기 vs 성능
- Baseline: 가장 작음 (0.7 MB), 빠른 추론
- DenseNet: 중간 크기 (6 MB), 좋은 균형
- EfficientNet: 가장 큼 (13 MB), 느린 추론

### 권장 사항
1. **엣지 디바이스**: Baseline 또는 SLIVE (작은 크기, 적은 메모리)
2. **서버 환경**: DenseNet (좋은 균형)
3. **고성능 필요**: EfficientNet 또는 ResNet

## GPU 메모리 모니터링 (선택)

GPU 메모리 사용량을 측정하려면 `pynvml` 설치:
```bash
pip install pynvml
```

GPU 사용 시 자동으로 GPU 메모리 사용량이 측정됩니다.

## 주의사항

1. **메모리 측정**:
   - Python 프로세스 전체 메모리 사용량 기준
   - 다른 프로그램의 영향을 받을 수 있음
   - 정확한 측정을 위해 학습 중 다른 작업 최소화

2. **FLOPs 계산**:
   - 레이어별 근사 계산
   - 실제 하드웨어 성능과 다를 수 있음
   - 모델 복잡도의 상대적 비교에 유용

3. **추론 속도**:
   - 100회 반복 측정의 평균
   - CPU/GPU 사양에 따라 달라짐
   - 동일 환경에서 비교 권장

## 파일 구조

```
model_comparison/
├── utils/
│   ├── __init__.py
│   └── resource_monitor.py          # 새로 추가
├── app_comparison.py                 # 수정됨
├── templates/
│   └── leaderboard.html             # 수정됨
├── static/
│   └── js/
│       └── leaderboard.js           # 수정됨
├── requirements_comparison.txt      # 수정됨 (psutil 추가)
├── README.md                        # 수정됨
└── RESOURCE_COMPARISON.md           # 새로 추가 (이 파일)
```

## 향후 개선 계획

1. **더 정확한 FLOPs 측정**: TensorFlow Profiler 활용
2. **전력 소비량 측정**: 학습 및 추론 시 전력 사용량
3. **탄소 배출량 계산**: 에너지 소비 기반 탄소 발자국
4. **배치 크기별 성능**: 다양한 배치 크기에서의 리소스 사용량
5. **하드웨어 프로파일링**: CPU/GPU 사용률, 메모리 대역폭 등

---

**버전**: 1.1.0
**날짜**: 2024-12-05
**작성자**: SLIVE 프로젝트 팀
