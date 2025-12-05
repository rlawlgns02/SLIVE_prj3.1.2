/**
 * 실시간 모델 비교 스크립트
 */

class LiveComparison {
    constructor() {
        this.camera = null;
        this.hands = null;
        this.isActive = false;
        this.isRunning = false;

        // 모델 관련
        this.availableModels = [];
        this.selectedModels = new Set();
        this.loadedModels = [];

        // 성능 추적
        this.performanceStats = {};
        this.inferenceHistory = {};

        // FPS 추적
        this.frameCount = 0;
        this.lastFpsUpdate = Date.now();
        this.currentFps = 0;

        // DOM 요소
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');

        // 버튼
        this.loadModelsBtn = document.getElementById('loadModelsBtn');
        this.startLiveBtn = document.getElementById('startLiveBtn');
        this.stopLiveBtn = document.getElementById('stopLiveBtn');

        // 상태
        this.cameraStatus = document.getElementById('cameraStatus');
        this.handsStatus = document.getElementById('handsStatus');
        this.inferenceStatus = document.getElementById('inferenceStatus');
        this.fpsCounter = document.getElementById('fpsCounter');

        // 차트
        this.speedChart = null;
        this.confidenceChart = null;

        this.init();
    }

    async init() {
        // 이벤트 리스너
        this.loadModelsBtn.addEventListener('click', () => this.loadModels());
        this.startLiveBtn.addEventListener('click', () => this.startLive());
        this.stopLiveBtn.addEventListener('click', () => this.stopLive());

        // 사용 가능한 모델 로드
        await this.loadAvailableModels();

        // 차트 초기화
        this.initCharts();
    }

    async loadAvailableModels() {
        try {
            const response = await fetch('/api/live/models');
            const result = await response.json();

            if (result.success) {
                this.availableModels = result.models;
                this.displayModelSelection();
            } else {
                alert('모델 목록을 불러올 수 없습니다: ' + result.error);
            }

        } catch (error) {
            console.error('모델 목록 로드 실패:', error);
            alert('모델 목록을 불러오는 중 오류가 발생했습니다.');
        }
    }

    displayModelSelection() {
        const container = document.getElementById('modelSelection');

        if (this.availableModels.length === 0) {
            container.innerHTML = '<p class="text-muted">학습된 모델이 없습니다. 먼저 모델을 학습시켜주세요.</p>';
            return;
        }

        let html = '';
        for (const model of this.availableModels) {
            html += `
                <label class="model-checkbox">
                    <input type="checkbox" value="${model.model_file}" data-key="${model.key}" data-name="${model.name}">
                    <div class="model-checkbox-content">
                        <div class="model-name">${model.name}</div>
                        <div class="model-info">정확도: ${(model.val_accuracy * 100).toFixed(2)}%</div>
                    </div>
                </label>
            `;
        }

        container.innerHTML = html;

        // 체크박스 이벤트
        const checkboxes = container.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(cb => {
            cb.addEventListener('change', () => {
                if (cb.checked) {
                    this.selectedModels.add(cb.value);
                } else {
                    this.selectedModels.delete(cb.value);
                }

                this.loadModelsBtn.disabled = this.selectedModels.size === 0;
            });
        });
    }

    async loadModels() {
        if (this.selectedModels.size === 0) {
            alert('최소 1개 이상의 모델을 선택하세요.');
            return;
        }

        const loadingStatus = document.getElementById('loadingStatus');
        loadingStatus.style.display = 'block';
        this.loadModelsBtn.disabled = true;

        try {
            const response = await fetch('/api/live/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_files: Array.from(this.selectedModels)
                })
            });

            const result = await response.json();

            if (result.success) {
                this.loadedModels = result.loaded_models.filter(m => m.loaded);

                // 성능 통계 초기화
                for (const model of this.loadedModels) {
                    this.performanceStats[model.model_file] = {
                        totalInferences: 0,
                        totalTime: 0,
                        totalConfidence: 0,
                        avgTime: 0,
                        avgConfidence: 0
                    };

                    this.inferenceHistory[model.model_file] = {
                        times: [],
                        confidences: []
                    };
                }

                alert(`${this.loadedModels.length}개 모델이 성공적으로 로드되었습니다!`);
                this.startLiveBtn.disabled = false;

            } else {
                alert('모델 로드 실패: ' + result.error);
            }

        } catch (error) {
            console.error('모델 로드 실패:', error);
            alert('모델 로드 중 오류가 발생했습니다.');
        } finally {
            loadingStatus.style.display = 'none';
            this.loadModelsBtn.disabled = false;
        }
    }

    async startLive() {
        if (this.loadedModels.length === 0) {
            alert('먼저 모델을 로드하세요.');
            return;
        }

        // 카메라 시작
        await this.startCamera();

        this.isRunning = true;
        this.startLiveBtn.disabled = true;
        this.stopLiveBtn.disabled = false;
        this.loadModelsBtn.disabled = true;

        this.inferenceStatus.textContent = '실시간 추론 중';
        this.inferenceStatus.classList.remove('status-inactive');
        this.inferenceStatus.classList.add('status-active');

        // 모델 결과 UI 초기화
        this.initModelResultsUI();
    }

    stopLive() {
        this.isRunning = false;
        this.stopCamera();

        this.startLiveBtn.disabled = false;
        this.stopLiveBtn.disabled = true;
        this.loadModelsBtn.disabled = false;

        this.inferenceStatus.textContent = '대기 중';
        this.inferenceStatus.classList.remove('status-active');
        this.inferenceStatus.classList.add('status-inactive');
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
                minDetectionConfidence: 0.7,
                minTrackingConfidence: 0.7
            });

            this.hands.onResults((results) => this.onResults(results));

            // 카메라 시작
            this.camera = new Camera(this.video, {
                onFrame: async () => {
                    await this.hands.send({ image: this.video });
                    this.updateFPS();
                },
                width: 640,
                height: 480
            });

            await this.camera.start();

            this.isActive = true;
            this.cameraStatus.textContent = '카메라 활성';
            this.cameraStatus.classList.remove('status-inactive');
            this.cameraStatus.classList.add('status-active');

        } catch (error) {
            console.error('카메라 시작 실패:', error);
            alert('카메라를 시작할 수 없습니다.');
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
        this.cameraStatus.textContent = '카메라 꺼짐';
        this.cameraStatus.classList.remove('status-active');
        this.cameraStatus.classList.add('status-inactive');
        this.handsStatus.textContent = '손 미감지';
        this.handsStatus.classList.remove('status-active');
        this.handsStatus.classList.add('status-inactive');

        // Canvas 초기화
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    async onResults(results) {
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

            // 실시간 추론
            if (this.isRunning) {
                const landmarksArray = landmarks.map(lm => [lm.x, lm.y, lm.z]);
                await this.performInference(landmarksArray);
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
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [0, 9], [9, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15], [15, 16],
            [0, 17], [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
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

    async performInference(landmarks) {
        try {
            const response = await fetch('/api/live/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ landmarks: landmarks })
            });

            const result = await response.json();

            if (result.success) {
                this.updateResults(result.predictions);
            }

        } catch (error) {
            console.error('추론 실패:', error);
        }
    }

    updateResults(predictions) {
        // 성능 통계 업데이트
        for (const [modelFile, prediction] of Object.entries(predictions)) {
            if (!prediction.success) continue;

            const stats = this.performanceStats[modelFile];
            const history = this.inferenceHistory[modelFile];

            stats.totalInferences++;
            stats.totalTime += prediction.inference_time_ms;
            stats.totalConfidence += prediction.top_prediction.confidence;

            stats.avgTime = stats.totalTime / stats.totalInferences;
            stats.avgConfidence = stats.totalConfidence / stats.totalInferences;

            // 히스토리 (최근 100개만 유지)
            history.times.push(prediction.inference_time_ms);
            history.confidences.push(prediction.top_prediction.confidence);

            if (history.times.length > 100) {
                history.times.shift();
                history.confidences.shift();
            }
        }

        // UI 업데이트
        this.updateLeaderboard(predictions);
        this.updateModelResults(predictions);
        this.updateCharts();
    }

    updateLeaderboard(predictions) {
        // 리더보드 정렬 (신뢰도 높은 순)
        const sortedPredictions = Object.entries(predictions)
            .filter(([_, pred]) => pred.success)
            .sort((a, b) => b[1].top_prediction.confidence - a[1].top_prediction.confidence);

        let html = '<div class="leaderboard-list">';

        sortedPredictions.forEach(([modelFile, prediction], index) => {
            const stats = this.performanceStats[modelFile];
            const modelName = this.getModelName(modelFile);
            const rank = index + 1;

            let rankIcon = rank;
            if (rank === 1) rankIcon = '🥇';
            else if (rank === 2) rankIcon = '🥈';
            else if (rank === 3) rankIcon = '🥉';

            html += `
                <div class="leaderboard-item rank-${rank}">
                    <div class="leaderboard-rank">${rankIcon}</div>
                    <div class="leaderboard-info">
                        <div class="leaderboard-model-name">${modelName}</div>
                        <div class="leaderboard-prediction">${prediction.top_prediction.label}</div>
                    </div>
                    <div class="leaderboard-stats">
                        <div class="stat-item">
                            <span class="stat-label">신뢰도</span>
                            <span class="stat-value">${(prediction.top_prediction.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">속도</span>
                            <span class="stat-value">${prediction.inference_time_ms.toFixed(1)}ms</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">평균 신뢰도</span>
                            <span class="stat-value">${(stats.avgConfidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">평균 속도</span>
                            <span class="stat-value">${stats.avgTime.toFixed(1)}ms</span>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';

        document.getElementById('performanceLeaderboard').innerHTML = html;
    }

    initModelResultsUI() {
        const container = document.getElementById('modelResults');
        let html = '';

        for (const model of this.loadedModels) {
            const modelName = this.getModelName(model.model_file);

            html += `
                <div class="model-result-card" id="result-${model.model_file}">
                    <div class="model-result-header">
                        <h4>${modelName}</h4>
                    </div>
                    <div class="model-result-body">
                        <p class="text-muted">대기 중...</p>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    updateModelResults(predictions) {
        for (const [modelFile, prediction] of Object.entries(predictions)) {
            const card = document.getElementById(`result-${modelFile}`);
            if (!card) continue;

            const body = card.querySelector('.model-result-body');

            if (!prediction.success) {
                body.innerHTML = '<p class="text-error">오류 발생</p>';
                continue;
            }

            let html = `
                <div class="top-prediction">
                    <div class="prediction-label">${prediction.top_prediction.label}</div>
                    <div class="prediction-confidence">${(prediction.top_prediction.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="prediction-list">
            `;

            for (let i = 1; i < prediction.top_5.length; i++) {
                const pred = prediction.top_5[i];
                html += `
                    <div class="prediction-item">
                        <span>${i + 1}. ${pred.label}</span>
                        <span>${(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;
            }

            html += '</div>';
            body.innerHTML = html;
        }
    }

    getModelName(modelFile) {
        const model = this.availableModels.find(m => m.model_file === modelFile);
        return model ? model.name : modelFile;
    }

    updateFPS() {
        this.frameCount++;
        const now = Date.now();
        const elapsed = now - this.lastFpsUpdate;

        if (elapsed >= 1000) {
            this.currentFps = Math.round((this.frameCount * 1000) / elapsed);
            this.fpsCounter.textContent = `${this.currentFps} FPS`;
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }
    }

    initCharts() {
        // 속도 차트
        const speedCtx = document.getElementById('speedChart').getContext('2d');
        this.speedChart = new Chart(speedCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '평균 추론 시간 (ms)',
                    data: [],
                    backgroundColor: 'rgba(59, 130, 246, 0.8)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        // 신뢰도 차트
        const confCtx = document.getElementById('confidenceChart').getContext('2d');
        this.confidenceChart = new Chart(confCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '평균 신뢰도 (%)',
                    data: [],
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    updateCharts() {
        const labels = [];
        const speeds = [];
        const confidences = [];

        for (const model of this.loadedModels) {
            const stats = this.performanceStats[model.model_file];
            const modelName = this.getModelName(model.model_file);

            labels.push(modelName);
            speeds.push(stats.avgTime);
            confidences.push(stats.avgConfidence * 100);
        }

        // 속도 차트 업데이트
        this.speedChart.data.labels = labels;
        this.speedChart.data.datasets[0].data = speeds;
        this.speedChart.update();

        // 신뢰도 차트 업데이트
        this.confidenceChart.data.labels = labels;
        this.confidenceChart.data.datasets[0].data = confidences;
        this.confidenceChart.update();
    }
}

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    new LiveComparison();
});
