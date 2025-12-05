/**
 * 데이터 수집 스크립트
 * MediaPipe Hands를 사용하여 손 랜드마크를 수집합니다
 */

class DataCollector {
    constructor() {
        this.camera = null;
        this.hands = null;
        this.isActive = false;
        this.isRecording = false;
        this.collectedData = [];
        this.currentGestureData = [];

        // DOM 요소
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');

        // 버튼
        this.toggleCameraBtn = document.getElementById('toggleCamera');
        this.recordBtn = document.getElementById('recordBtn');
        this.saveBtn = document.getElementById('saveBtn');
        this.refreshStatsBtn = document.getElementById('refreshStats');
        this.exportDataBtn = document.getElementById('exportData');
        this.resetDataBtn = document.getElementById('resetData');

        // 입력
        this.gestureNameInput = document.getElementById('gestureName');
        this.collectionModeSelect = document.getElementById('collectionMode');
        this.autoIntervalInput = document.getElementById('autoInterval');

        // 상태
        this.cameraStatus = document.getElementById('cameraStatus');
        this.handsStatus = document.getElementById('handsStatus');
        this.currentCountSpan = document.getElementById('currentCount');
        this.totalCountSpan = document.getElementById('totalCount');

        // 자동 수집
        this.autoCollectInterval = null;

        this.init();
    }

    init() {
        // 이벤트 리스너
        this.toggleCameraBtn.addEventListener('click', () => this.toggleCamera());
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.saveBtn.addEventListener('click', () => this.saveData());
        this.refreshStatsBtn.addEventListener('click', () => this.loadStats());
        this.exportDataBtn.addEventListener('click', () => this.exportData());
        this.resetDataBtn.addEventListener('click', () => this.resetData());

        // 통계 로드
        this.loadStats();
    }

    async toggleCamera() {
        if (!this.isActive) {
            await this.startCamera();
        } else {
            this.stopCamera();
        }
    }

    async startCamera() {
        try {
            // MediaPipe Hands 초기화
            this.hands = new Hands({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
                }
            });

            this.hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });

            this.hands.onResults((results) => this.onResults(results));

            // 카메라 시작
            this.camera = new Camera(this.video, {
                onFrame: async () => {
                    await this.hands.send({ image: this.video });
                },
                width: 640,
                height: 480
            });

            await this.camera.start();

            this.isActive = true;
            this.toggleCameraBtn.textContent = '카메라 정지';
            this.toggleCameraBtn.classList.remove('btn-primary');
            this.toggleCameraBtn.classList.add('btn-danger');
            this.cameraStatus.textContent = '카메라 활성';
            this.cameraStatus.classList.remove('status-inactive');
            this.cameraStatus.classList.add('status-active');
            this.recordBtn.disabled = false;

        } catch (error) {
            console.error('카메라 시작 실패:', error);
            alert('카메라를 시작할 수 없습니다. 권한을 확인하세요.');
        }
    }

    stopCamera() {
        if (this.camera) {
            this.camera.stop();
            this.camera = null;
        }

        if (this.hands) {
            this.hands.close();
            this.hands = null;
        }

        this.isActive = false;
        this.isRecording = false;

        this.toggleCameraBtn.textContent = '카메라 시작';
        this.toggleCameraBtn.classList.remove('btn-danger');
        this.toggleCameraBtn.classList.add('btn-primary');
        this.cameraStatus.textContent = '카메라 꺼짐';
        this.cameraStatus.classList.remove('status-active');
        this.cameraStatus.classList.add('status-inactive');
        this.handsStatus.textContent = '손 미감지';
        this.handsStatus.classList.remove('status-active');
        this.handsStatus.classList.add('status-inactive');
        this.recordBtn.disabled = true;

        // Canvas 초기화
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    onResults(results) {
        // Canvas 크기 조정
        this.canvas.width = results.image.width;
        this.canvas.height = results.image.height;

        // 이미지 그리기
        this.ctx.save();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(results.image, 0, 0, this.canvas.width, this.canvas.height);

        // 손 감지 여부
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            this.handsStatus.textContent = '손 감지됨';
            this.handsStatus.classList.remove('status-inactive');
            this.handsStatus.classList.add('status-active');

            const landmarks = results.multiHandLandmarks[0];

            // 랜드마크 그리기
            this.drawLandmarks(landmarks);

            // 녹화 중이면 데이터 수집
            if (this.isRecording) {
                const landmarksArray = landmarks.map(lm => [lm.x, lm.y, lm.z]);
                this.collectSample(landmarksArray);
            }
        } else {
            this.handsStatus.textContent = '손 미감지';
            this.handsStatus.classList.remove('status-active');
            this.handsStatus.classList.add('status-inactive');
        }

        this.ctx.restore();
    }

    drawLandmarks(landmarks) {
        // 랜드마크 점 그리기
        this.ctx.fillStyle = '#00FF00';
        for (const landmark of landmarks) {
            const x = landmark.x * this.canvas.width;
            const y = landmark.y * this.canvas.height;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
            this.ctx.fill();
        }

        // 연결선 그리기
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],  // 엄지
            [0, 5], [5, 6], [6, 7], [7, 8],  // 검지
            [0, 9], [9, 10], [10, 11], [11, 12],  // 중지
            [0, 13], [13, 14], [14, 15], [15, 16],  // 약지
            [0, 17], [17, 18], [18, 19], [19, 20],  // 소지
            [5, 9], [9, 13], [13, 17]  // 손바닥
        ];

        this.ctx.strokeStyle = '#00FF00';
        this.ctx.lineWidth = 2;

        for (const [start, end] of connections) {
            const startLm = landmarks[start];
            const endLm = landmarks[end];

            this.ctx.beginPath();
            this.ctx.moveTo(startLm.x * this.canvas.width, startLm.y * this.canvas.height);
            this.ctx.lineTo(endLm.x * this.canvas.width, endLm.y * this.canvas.height);
            this.ctx.stroke();
        }
    }

    toggleRecording() {
        if (!this.isRecording) {
            this.startRecording();
        } else {
            this.stopRecording();
        }
    }

    startRecording() {
        const gestureName = this.gestureNameInput.value.trim();
        if (!gestureName) {
            alert('제스처 이름을 입력하세요.');
            return;
        }

        this.isRecording = true;
        this.currentGestureData = [];

        this.recordBtn.innerHTML = '<span id="recordIcon">■</span> 녹화 중지';
        this.recordBtn.classList.remove('btn-success');
        this.recordBtn.classList.add('btn-danger');
        this.saveBtn.disabled = true;

        // 자동 수집 모드
        const mode = this.collectionModeSelect.value;
        if (mode === 'auto') {
            const interval = parseInt(this.autoIntervalInput.value);
            this.startAutoCollect(interval);
        }
    }

    stopRecording() {
        this.isRecording = false;

        this.recordBtn.innerHTML = '<span id="recordIcon">●</span> 녹화 시작';
        this.recordBtn.classList.remove('btn-danger');
        this.recordBtn.classList.add('btn-success');
        this.saveBtn.disabled = false;

        // 자동 수집 중지
        this.stopAutoCollect();

        // 현재 카운트 업데이트
        this.currentCountSpan.textContent = this.currentGestureData.length;
    }

    startAutoCollect(interval) {
        this.autoCollectInterval = setInterval(() => {
            // 녹화 중이 아니면 중지
            if (!this.isRecording) {
                this.stopAutoCollect();
            }
        }, interval);
    }

    stopAutoCollect() {
        if (this.autoCollectInterval) {
            clearInterval(this.autoCollectInterval);
            this.autoCollectInterval = null;
        }
    }

    collectSample(landmarks) {
        const gestureName = this.gestureNameInput.value.trim();

        const sample = {
            label: gestureName,
            landmarks: landmarks,
            timestamp: new Date().toISOString()
        };

        this.currentGestureData.push(sample);
        this.currentCountSpan.textContent = this.currentGestureData.length;
    }

    async saveData() {
        if (this.currentGestureData.length === 0) {
            alert('저장할 데이터가 없습니다.');
            return;
        }

        try {
            const response = await fetch('/api/data/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ samples: this.currentGestureData })
            });

            const result = await response.json();

            if (result.success) {
                alert(`${result.added_samples}개 샘플 저장 완료!\n전체 데이터: ${result.total_samples}개`);
                this.currentGestureData = [];
                this.currentCountSpan.textContent = 0;
                this.loadStats();
            } else {
                alert('저장 실패: ' + result.error);
            }

        } catch (error) {
            console.error('저장 실패:', error);
            alert('저장 중 오류가 발생했습니다.');
        }
    }

    async loadStats() {
        try {
            const response = await fetch('/api/data/stats');
            const result = await response.json();

            if (result.success) {
                this.totalCountSpan.textContent = result.total_samples;

                // 제스처별 통계 표시
                const statsDiv = document.getElementById('gestureStats');
                if (result.num_gestures === 0) {
                    statsDiv.innerHTML = '<p class="text-muted">수집된 데이터가 없습니다.</p>';
                } else {
                    let html = '<div class="gesture-stats-grid">';
                    for (const [gesture, count] of Object.entries(result.gesture_counts)) {
                        html += `
                            <div class="gesture-stat-item">
                                <span class="gesture-name">${gesture}</span>
                                <span class="gesture-count">${count}개</span>
                            </div>
                        `;
                    }
                    html += '</div>';
                    statsDiv.innerHTML = html;
                }
            }

        } catch (error) {
            console.error('통계 로드 실패:', error);
        }
    }

    async exportData() {
        try {
            const response = await fetch('/api/data/stats');
            const result = await response.json();

            if (result.success) {
                const dataStr = JSON.stringify(result, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);

                const a = document.createElement('a');
                a.href = url;
                a.download = `comparison_data_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();

                URL.revokeObjectURL(url);
            }

        } catch (error) {
            console.error('내보내기 실패:', error);
            alert('데이터 내보내기 실패');
        }
    }

    async resetData() {
        if (!confirm('모든 수집된 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
            return;
        }

        try {
            const response = await fetch('/api/data/reset', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                alert('모든 데이터가 삭제되었습니다.');
                this.currentGestureData = [];
                this.currentCountSpan.textContent = 0;
                this.loadStats();
            } else {
                alert('초기화 실패: ' + result.error);
            }

        } catch (error) {
            console.error('초기화 실패:', error);
            alert('초기화 중 오류가 발생했습니다.');
        }
    }
}

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    new DataCollector();
});
