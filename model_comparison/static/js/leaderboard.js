/**
 * 리더보드 스크립트
 * 모델 학습 및 성능 비교 시각화
 */

class LeaderboardManager {
    constructor() {
        // DOM 요소
        this.modelSelect = document.getElementById('modelSelect');
        this.epochsInput = document.getElementById('epochsInput');
        this.batchSizeInput = document.getElementById('batchSizeInput');
        this.learningRateInput = document.getElementById('learningRateInput');
        this.trainBtn = document.getElementById('trainBtn');

        this.trainingProgress = document.getElementById('trainingProgress');
        this.trainingStatus = document.getElementById('trainingStatus');
        this.trainingPercent = document.getElementById('trainingPercent');
        this.trainingProgressBar = document.getElementById('trainingProgressBar');
        this.trainingGraphContainer = document.getElementById('trainingGraphContainer');

        this.leaderboardBody = document.getElementById('leaderboardBody');
        this.refreshLeaderboardBtn = document.getElementById('refreshLeaderboard');
        this.clearLeaderboardBtn = document.getElementById('clearLeaderboard');

        this.sortBySelect = document.getElementById('sortBy');
        this.sortOrderSelect = document.getElementById('sortOrder');

        // 차트
        this.accuracyChart = null;
        this.trainTimeChart = null;
        this.inferenceChart = null;
        this.memoryChart = null;
        this.modelSizeChart = null;
        this.flopsChart = null;

        // 실시간 학습 차트
        this.liveAccuracyChart = null;
        this.liveLossChart = null;
        this.liveTrainingData = {
            epochs: [],
            trainAcc: [],
            valAcc: [],
            trainLoss: [],
            valLoss: []
        };

        // EventSource
        this.eventSource = null;

        this.init();
    }

    async init() {
        // 이벤트 리스너
        this.trainBtn.addEventListener('click', () => this.startTraining());
        this.refreshLeaderboardBtn.addEventListener('click', () => this.loadLeaderboard());
        this.clearLeaderboardBtn.addEventListener('click', () => this.clearLeaderboard());
        this.sortBySelect.addEventListener('change', () => this.loadLeaderboard());
        this.sortOrderSelect.addEventListener('change', () => this.loadLeaderboard());

        // 다운로드 버튼 이벤트 리스너
        document.getElementById('downloadAccuracyBtn').addEventListener('click', () => this.downloadAccuracyGraph());
        document.getElementById('downloadLossBtn').addEventListener('click', () => this.downloadLossGraph());

        // 로그 지우기 버튼
        document.getElementById('clearLogBtn').addEventListener('click', () => this.clearTrainingLog());

        // 초기 로드
        await this.loadModels();
        await this.loadLeaderboard();
        this.initCharts();
    }

    addLog(message, type = 'info') {
        // 학습 로그에 메시지 추가
        const logContainer = document.getElementById('trainingLog');
        const logLine = document.createElement('div');
        logLine.className = `terminal-line ${type}`;

        // 타임스탬프 추가
        const timestamp = new Date().toLocaleTimeString('ko-KR');
        logLine.textContent = `[${timestamp}] ${message}`;

        logContainer.appendChild(logLine);

        // 자동 스크롤 (최신 로그가 보이도록)
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    clearTrainingLog() {
        // 학습 로그 지우기
        const logContainer = document.getElementById('trainingLog');
        logContainer.innerHTML = '<div class="terminal-line system">> 로그가 초기화되었습니다.</div>';
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models/list');
            const result = await response.json();

            if (result.success) {
                this.modelSelect.innerHTML = '';
                for (const model of result.models) {
                    const option = document.createElement('option');
                    option.value = model.key;
                    option.textContent = `${model.name} (${this.formatNumber(model.parameters)} params)`;
                    this.modelSelect.appendChild(option);
                }

                // 모델 정보 표시
                this.displayModelsInfo(result.models);
            }

        } catch (error) {
            console.error('모델 목록 로드 실패:', error);
            this.modelSelect.innerHTML = '<option>로드 실패</option>';
        }
    }

    displayModelsInfo(models) {
        const modelsInfoDiv = document.getElementById('modelsInfo');
        let html = '<div class="models-grid">';

        for (const model of models) {
            html += `
                <div class="model-info-card">
                    <h4>${model.name}</h4>
                    <p>파라미터: ${this.formatNumber(model.parameters)}</p>
                    <p class="text-muted">키: ${model.key}</p>
                </div>
            `;
        }

        html += '</div>';
        modelsInfoDiv.innerHTML = html;
    }

    async startTraining() {
        const modelKey = this.modelSelect.value;
        const epochs = parseInt(this.epochsInput.value);
        const batchSize = parseInt(this.batchSizeInput.value);
        const learningRate = parseFloat(this.learningRateInput.value);

        if (!modelKey) {
            alert('모델을 선택하세요.');
            return;
        }

        // UI 초기화
        this.trainBtn.disabled = true;

        // 학습 진행 상황 카드 표시
        document.getElementById('trainingProgressCard').style.display = 'block';
        document.getElementById('epochText').textContent = `0/${epochs}`;
        this.trainingProgressBar.style.width = '0%';
        document.getElementById('trainAccuracy').textContent = '0%';
        document.getElementById('valAccuracy').textContent = '0%';
        document.getElementById('trainLoss').textContent = '0';
        document.getElementById('valLoss').textContent = '0';

        // 로그 카드 표시 및 초기화
        document.getElementById('trainingLogCard').style.display = 'block';
        this.clearTrainingLog();
        this.addLog('학습을 시작합니다...', 'system');
        this.addLog(`모델: ${modelKey.toUpperCase()}`, 'info');
        this.addLog(`에포크: ${epochs}, 배치 크기: ${batchSize}, 학습률: ${learningRate}`, 'info');

        // 그래프 카드 표시 (학습 시작하자마자 표시)
        document.getElementById('trainingGraphCard').style.display = 'block';

        // 다운로드 버튼 숨기기 (학습 중에는 다운로드 불가)
        document.getElementById('downloadAccuracyBtn').style.display = 'none';
        document.getElementById('downloadLossBtn').style.display = 'none';

        // 실시간 학습 데이터 초기화
        this.liveTrainingData = {
            epochs: [],
            trainAcc: [],
            valAcc: [],
            trainLoss: [],
            valLoss: []
        };

        // 실시간 학습 차트 초기화
        this.initLiveCharts();

        // EventSource 설정
        const params = {
            model: modelKey,
            epochs: epochs,
            batch_size: batchSize,
            learning_rate: learningRate
        };

        // POST 요청을 위해 fetch로 스트림 시작
        try {
            const response = await fetch('/api/train/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error('학습 요청 실패');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                // 스트림 데이터를 디코딩
                buffer += decoder.decode(value, { stream: true });

                // SSE 메시지 파싱 (data: 형식)
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // 마지막 불완전한 부분은 버퍼에 유지

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const jsonData = line.substring(6);
                        try {
                            const data = JSON.parse(jsonData);
                            this.handleTrainingEvent(data);
                        } catch (e) {
                            console.error('JSON 파싱 실패:', e, jsonData);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('학습 실패:', error);
            alert('학습 중 오류가 발생했습니다: ' + error.message);
            this.trainingProgress.style.display = 'none';
            this.trainBtn.disabled = false;
        }
    }

    handleTrainingEvent(data) {
        const type = data.type;

        if (type === 'status') {
            // 상태 메시지 업데이트
            this.trainingStatus.textContent = data.message;
            this.updateProgress(data.progress);
            this.addLog(data.message, 'info');

        } else if (type === 'epoch') {
            // 에포크 로그 출력 (ksl_project 스타일)
            const epochLog = `Epoch ${data.epoch}/${data.total_epochs}: 정확도 ${(data.val_accuracy * 100).toFixed(2)}%`;
            this.addLog(epochLog, 'success');

            // 진행 상황 카드 업데이트
            document.getElementById('epochText').textContent = `${data.epoch}/${data.total_epochs}`;
            document.getElementById('trainAccuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('valAccuracy').textContent = (data.val_accuracy * 100).toFixed(2) + '%';
            document.getElementById('trainLoss').textContent = data.loss.toFixed(4);
            document.getElementById('valLoss').textContent = data.val_loss.toFixed(4);

            // 진행바 업데이트
            const progress = (data.epoch / data.total_epochs) * 100;
            this.trainingProgressBar.style.width = progress + '%';

            // 에포크 데이터 추가
            this.liveTrainingData.epochs.push(data.epoch);
            this.liveTrainingData.trainAcc.push(data.accuracy * 100);
            this.liveTrainingData.valAcc.push(data.val_accuracy * 100);
            this.liveTrainingData.trainLoss.push(data.loss);
            this.liveTrainingData.valLoss.push(data.val_loss);

            // 차트 업데이트 (실시간으로 그래프가 갱신됨)
            this.updateLiveCharts();

        } else if (type === 'complete') {
            // 학습 완료
            // 현재 학습 결과 저장 (다운로드 버튼용)
            this.currentTrainingResult = data.result;

            // 완료 로그
            this.addLog('✅ 학습 완료!', 'success');
            this.addLog(`최종 검증 정확도: ${(data.result.val_accuracy * 100).toFixed(2)}%`, 'success');
            this.addLog(`학습 시간: ${data.result.train_time.toFixed(2)}초`, 'info');

            setTimeout(() => {
                alert('학습이 완료되었습니다!\n그래프 다운로드 버튼을 사용하여 저장할 수 있습니다.');

                // 다운로드 버튼 활성화
                document.getElementById('downloadAccuracyBtn').style.display = 'inline-block';
                document.getElementById('downloadLossBtn').style.display = 'inline-block';

                // 학습 시작 버튼 활성화
                this.trainBtn.disabled = false;

                // 리더보드 갱신
                this.loadLeaderboard();
            }, 1000);

        } else if (type === 'error') {
            // 오류 발생
            this.addLog('❌ 오류 발생: ' + data.message, 'error');
            alert('학습 실패: ' + data.message);
            // 그래프와 진행 상황은 유지 (에러 발생 전까지의 학습 결과 확인 가능)
            this.trainBtn.disabled = false;
        }
    }

    updateProgress(percent) {
        this.trainingPercent.textContent = percent + '%';
        this.trainingProgressBar.style.width = percent + '%';
    }

    initLiveCharts() {
        // 정확도 차트
        const accCtx = document.getElementById('liveAccuracyChart');
        if (accCtx) {
            if (this.liveAccuracyChart) {
                this.liveAccuracyChart.destroy();
            }

            this.liveAccuracyChart = new Chart(accCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: '학습 정확도',
                            data: [],
                            borderColor: 'rgba(16, 185, 129, 1)',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true
                        },
                        {
                            label: '검증 정확도',
                            data: [],
                            borderColor: 'rgba(59, 130, 246, 1)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: {
                        duration: 300
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: '정확도 (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: '에포크'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    }
                }
            });
        }

        // 손실 차트
        const lossCtx = document.getElementById('liveLossChart');
        if (lossCtx) {
            if (this.liveLossChart) {
                this.liveLossChart.destroy();
            }

            this.liveLossChart = new Chart(lossCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: '학습 손실',
                            data: [],
                            borderColor: 'rgba(239, 68, 68, 1)',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true
                        },
                        {
                            label: '검증 손실',
                            data: [],
                            borderColor: 'rgba(245, 158, 11, 1)',
                            backgroundColor: 'rgba(245, 158, 11, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: {
                        duration: 300
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: '손실 (Loss)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: '에포크'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    }
                }
            });
        }
    }

    updateLiveCharts() {
        // 정확도 차트 업데이트
        if (this.liveAccuracyChart) {
            this.liveAccuracyChart.data.labels = this.liveTrainingData.epochs;
            this.liveAccuracyChart.data.datasets[0].data = this.liveTrainingData.trainAcc;
            this.liveAccuracyChart.data.datasets[1].data = this.liveTrainingData.valAcc;
            this.liveAccuracyChart.update('none'); // 애니메이션 없이 즉시 업데이트
        }

        // 손실 차트 업데이트
        if (this.liveLossChart) {
            this.liveLossChart.data.labels = this.liveTrainingData.epochs;
            this.liveLossChart.data.datasets[0].data = this.liveTrainingData.trainLoss;
            this.liveLossChart.data.datasets[1].data = this.liveTrainingData.valLoss;
            this.liveLossChart.update('none'); // 애니메이션 없이 즉시 업데이트
        }
    }

    downloadAccuracyGraph() {
        // 정확도 그래프 다운로드
        try {
            if (!this.liveAccuracyChart) {
                alert('다운로드할 그래프가 없습니다.');
                return;
            }

            const modelName = this.currentTrainingResult?.model_name || 'model';
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            const filename = `${modelName}_accuracy_${timestamp}.png`;

            const canvas = document.getElementById('liveAccuracyChart');
            this.downloadChartAsImage(canvas, filename);

            console.log('정확도 그래프가 다운로드되었습니다:', filename);
        } catch (error) {
            console.error('정확도 그래프 다운로드 실패:', error);
            alert('그래프 다운로드에 실패했습니다.');
        }
    }

    downloadLossGraph() {
        // 손실 그래프 다운로드
        try {
            if (!this.liveLossChart) {
                alert('다운로드할 그래프가 없습니다.');
                return;
            }

            const modelName = this.currentTrainingResult?.model_name || 'model';
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
            const filename = `${modelName}_loss_${timestamp}.png`;

            const canvas = document.getElementById('liveLossChart');
            this.downloadChartAsImage(canvas, filename);

            console.log('손실 그래프가 다운로드되었습니다:', filename);
        } catch (error) {
            console.error('손실 그래프 다운로드 실패:', error);
            alert('그래프 다운로드에 실패했습니다.');
        }
    }

    downloadChartAsImage(canvas, filename) {
        // Canvas를 이미지로 변환하여 다운로드
        try {
            // Canvas를 PNG 이미지로 변환
            canvas.toBlob((blob) => {
                // Blob을 다운로드 가능한 URL로 변환
                const url = URL.createObjectURL(blob);

                // 임시 링크 생성하여 다운로드 트리거
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();

                // 정리
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            });
        } catch (error) {
            console.error('이미지 다운로드 실패:', error);
            throw error;
        }
    }

    async loadLeaderboard() {
        const sortBy = this.sortBySelect.value;
        const order = this.sortOrderSelect.value;

        try {
            const response = await fetch(`/api/leaderboard?sort_by=${sortBy}&order=${order}`);
            const result = await response.json();

            if (result.success) {
                this.displayLeaderboard(result.results);
                this.updateCharts(result.results);
            }

        } catch (error) {
            console.error('리더보드 로드 실패:', error);
        }
    }

    displayLeaderboard(results) {
        if (results.length === 0) {
            this.leaderboardBody.innerHTML = `
                <tr>
                    <td colspan="18" class="text-center text-muted">학습 결과가 없습니다. 모델을 학습시켜 보세요!</td>
                </tr>
            `;
            return;
        }

        let html = '';
        for (let i = 0; i < results.length; i++) {
            const result = results[i];
            const rank = i + 1;

            // 메달 아이콘
            let rankDisplay = rank;
            if (this.sortBySelect.value.includes('accuracy')) {
                if (rank === 1) rankDisplay = '🥇';
                else if (rank === 2) rankDisplay = '🥈';
                else if (rank === 3) rankDisplay = '🥉';
            }

            html += `
                <tr>
                    <td class="rank">${rankDisplay}</td>
                    <td><strong>${result.model_name}</strong></td>
                    <td>${(result.val_accuracy * 100).toFixed(2)}%</td>
                    <td>${(result.train_accuracy * 100).toFixed(2)}%</td>
                    <td>${this.formatTime(result.train_time)}</td>
                    <td>${this.formatTime(result.avg_epoch_time)}</td>
                    <td>${result.inference_time_ms.toFixed(2)} ms</td>
                    <td>${this.formatNumber(result.num_parameters)}</td>
                    <td>${result.model_size_mb ? result.model_size_mb.toFixed(2) : 'N/A'}</td>
                    <td>${result.peak_memory_mb ? result.peak_memory_mb.toFixed(2) : 'N/A'}</td>
                    <td>${result.flops ? (result.flops / 1000000).toFixed(2) : 'N/A'}</td>
                    <td>${result.epochs}</td>
                    <td>${result.batch_size}</td>
                    <td>${result.learning_rate}</td>
                    <td>${result.num_samples}</td>
                    <td>${result.num_classes}</td>
                    <td>${this.formatDateTime(result.timestamp)}</td>
                    <td>
                        <button class="btn btn-small btn-danger" onclick="leaderboardManager.deleteEntry(${i})">삭제</button>
                    </td>
                </tr>
            `;
        }

        this.leaderboardBody.innerHTML = html;
    }

    async deleteEntry(index) {
        if (!confirm('이 항목을 삭제하시겠습니까?')) {
            return;
        }

        try {
            const response = await fetch(`/api/leaderboard/delete/${index}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.loadLeaderboard();
            } else {
                alert('삭제 실패: ' + result.error);
            }

        } catch (error) {
            console.error('삭제 실패:', error);
            alert('삭제 중 오류가 발생했습니다.');
        }
    }

    async clearLeaderboard() {
        if (!confirm('모든 리더보드 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.')) {
            return;
        }

        try {
            const response = await fetch('/api/leaderboard/clear', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                alert('리더보드가 초기화되었습니다.');
                this.loadLeaderboard();
            } else {
                alert('초기화 실패: ' + result.error);
            }

        } catch (error) {
            console.error('초기화 실패:', error);
            alert('초기화 중 오류가 발생했습니다.');
        }
    }

    initCharts() {
        // 정확도 차트
        const accCtx = document.getElementById('accuracyChart').getContext('2d');
        this.accuracyChart = new Chart(accCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '검증 정확도',
                        data: [],
                        backgroundColor: 'rgba(59, 130, 246, 0.8)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 2
                    },
                    {
                        label: '학습 정확도',
                        data: [],
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderColor: 'rgba(16, 185, 129, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + (context.parsed.y * 100).toFixed(2) + '%';
                            }
                        }
                    }
                }
            }
        });

        // 학습 시간 차트
        const timeCtx = document.getElementById('trainTimeChart').getContext('2d');
        this.trainTimeChart = new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '학습 시간 (초)',
                    data: [],
                    backgroundColor: 'rgba(245, 158, 11, 0.8)',
                    borderColor: 'rgba(245, 158, 11, 1)',
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

        // 추론 속도 차트
        const infCtx = document.getElementById('inferenceChart').getContext('2d');
        this.inferenceChart = new Chart(infCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '추론 시간 (ms)',
                    data: [],
                    backgroundColor: 'rgba(139, 92, 246, 0.8)',
                    borderColor: 'rgba(139, 92, 246, 1)',
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

        // 메모리 사용량 차트
        const memCtx = document.getElementById('memoryChart').getContext('2d');
        this.memoryChart = new Chart(memCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '피크 메모리 (MB)',
                        data: [],
                        backgroundColor: 'rgba(239, 68, 68, 0.8)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 2
                    },
                    {
                        label: '메모리 증가량 (MB)',
                        data: [],
                        backgroundColor: 'rgba(251, 146, 60, 0.8)',
                        borderColor: 'rgba(251, 146, 60, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'MB'
                        }
                    }
                }
            }
        });

        // 모델 크기 차트
        const sizeCtx = document.getElementById('modelSizeChart').getContext('2d');
        this.modelSizeChart = new Chart(sizeCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '모델 크기 (MB)',
                    data: [],
                    backgroundColor: 'rgba(34, 197, 94, 0.8)',
                    borderColor: 'rgba(34, 197, 94, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'MB'
                        }
                    }
                }
            }
        });

        // FLOPs 차트
        const flopsCtx = document.getElementById('flopsChart').getContext('2d');
        this.flopsChart = new Chart(flopsCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'FLOPs (백만)',
                    data: [],
                    backgroundColor: 'rgba(168, 85, 247, 0.8)',
                    borderColor: 'rgba(168, 85, 247, 1)',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'FLOPs (백만)'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                return context.dataset.label + ': ' + value.toFixed(2) + 'M';
                            }
                        }
                    }
                }
            }
        });
    }

    updateCharts(results) {
        // 차트가 초기화되지 않았거나 결과가 없으면 리턴
        if (!this.accuracyChart || !this.trainTimeChart || !this.inferenceChart) {
            return;
        }

        if (results.length === 0) {
            // 빈 데이터로 차트 초기화
            this.accuracyChart.data.labels = [];
            this.accuracyChart.data.datasets[0].data = [];
            this.accuracyChart.data.datasets[1].data = [];
            this.accuracyChart.update();

            this.trainTimeChart.data.labels = [];
            this.trainTimeChart.data.datasets[0].data = [];
            this.trainTimeChart.update();

            this.inferenceChart.data.labels = [];
            this.inferenceChart.data.datasets[0].data = [];
            this.inferenceChart.update();

            // 리소스 차트 초기화
            if (this.memoryChart) {
                this.memoryChart.data.labels = [];
                this.memoryChart.data.datasets[0].data = [];
                this.memoryChart.data.datasets[1].data = [];
                this.memoryChart.update();
            }

            if (this.modelSizeChart) {
                this.modelSizeChart.data.labels = [];
                this.modelSizeChart.data.datasets[0].data = [];
                this.modelSizeChart.update();
            }

            if (this.flopsChart) {
                this.flopsChart.data.labels = [];
                this.flopsChart.data.datasets[0].data = [];
                this.flopsChart.update();
            }

            return;
        }

        // 최근 10개만 표시
        const displayResults = results.slice(0, 10);

        const labels = displayResults.map(r => r.model_name);
        const valAccuracies = displayResults.map(r => r.val_accuracy);
        const trainAccuracies = displayResults.map(r => r.train_accuracy);
        const trainTimes = displayResults.map(r => r.train_time);
        const inferenceTimes = displayResults.map(r => r.inference_time_ms);

        // 정확도 차트 업데이트
        this.accuracyChart.data.labels = labels;
        this.accuracyChart.data.datasets[0].data = valAccuracies;
        this.accuracyChart.data.datasets[1].data = trainAccuracies;
        this.accuracyChart.update();

        // 학습 시간 차트 업데이트
        this.trainTimeChart.data.labels = labels;
        this.trainTimeChart.data.datasets[0].data = trainTimes;
        this.trainTimeChart.update();

        // 추론 속도 차트 업데이트
        this.inferenceChart.data.labels = labels;
        this.inferenceChart.data.datasets[0].data = inferenceTimes;
        this.inferenceChart.update();

        // 리소스 차트 업데이트
        if (this.memoryChart) {
            const peakMemories = displayResults.map(r => r.peak_memory_mb || 0);
            const memoryIncreases = displayResults.map(r => r.memory_increase_mb || 0);

            this.memoryChart.data.labels = labels;
            this.memoryChart.data.datasets[0].data = peakMemories;
            this.memoryChart.data.datasets[1].data = memoryIncreases;
            this.memoryChart.update();
        }

        if (this.modelSizeChart) {
            const modelSizes = displayResults.map(r => r.model_size_mb || 0);

            this.modelSizeChart.data.labels = labels;
            this.modelSizeChart.data.datasets[0].data = modelSizes;
            this.modelSizeChart.update();
        }

        if (this.flopsChart) {
            const flops = displayResults.map(r => (r.flops || 0) / 1000000); // FLOPs를 백만 단위로 변환

            this.flopsChart.data.labels = labels;
            this.flopsChart.data.datasets[0].data = flops;
            this.flopsChart.update();
        }
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(2) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(2) + 'K';
        }
        return num.toString();
    }

    formatTime(seconds) {
        if (seconds < 60) {
            return seconds.toFixed(2) + 's';
        }
        const minutes = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(0);
        return `${minutes}m ${secs}s`;
    }

    formatDateTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString('ko-KR', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// 전역 변수로 저장 (삭제 버튼에서 사용)
let leaderboardManager;

// 초기화
document.addEventListener('DOMContentLoaded', () => {
    leaderboardManager = new LeaderboardManager();
});
