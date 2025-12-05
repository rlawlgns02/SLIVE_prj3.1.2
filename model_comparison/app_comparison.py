"""
모델 비교 시스템 - Flask 백엔드 서버
여러 CNN 모델의 성능을 비교하고 리더보드를 제공합니다.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
import json
import os
import time
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import queue
import threading

# TensorFlow 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 모델 import
from models.baseline import BaselineModel
from models.resnet import ResNetModel
from models.densenet import DenseNetModel
from models.efficientnet import EfficientNetModel
from models.slive import SLIVEModel

# 유틸리티 import
from utils import ResourceMonitor, TrainingResourceMonitor, measure_all_resources, get_system_info

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
app.config['JSON_AS_ASCII'] = False

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')

# 데이터 파일
COMPARISON_DATA_FILE = os.path.join(DATA_DIR, 'comparison_data.json')
LEADERBOARD_FILE = os.path.join(RESULTS_DIR, 'leaderboard.json')

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 모델 매핑
MODEL_CLASSES = {
    'baseline': BaselineModel,
    'resnet': ResNetModel,
    'densenet': DenseNetModel,
    'efficientnet': EfficientNetModel,
    'slive': SLIVEModel
}

# 로드된 모델 캐시 (실시간 추론용)
LOADED_MODELS = {}


# ============ 유틸리티 함수 ============

def load_json_file(filepath, default=None):
    """JSON 파일 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default if default is not None else {}


def save_json_file(filepath, data):
    """JSON 파일 저장"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def preprocess_landmarks(landmarks):
    """랜드마크 전처리 (Wrist 기준 정규화)"""
    landmarks = np.array(landmarks)
    if len(landmarks.shape) == 1:
        landmarks = landmarks.reshape(-1, 3)

    # Wrist 기준 정규화
    wrist = landmarks[0]
    normalized = landmarks - wrist

    # 스케일 정규화
    max_val = np.max(np.abs(normalized))
    if max_val > 0:
        normalized = normalized / max_val

    return normalized


def prepare_dataset(data_file):
    """데이터셋 준비"""
    data = load_json_file(data_file, {'dataset': []})
    dataset = data.get('dataset', [])

    if len(dataset) == 0:
        return None, None, None, None, None

    # 데이터 분리
    X = []
    y = []
    labels_list = []

    for item in dataset:
        landmarks = item['landmarks']
        label = item['label']

        # 전처리
        processed = preprocess_landmarks(landmarks)
        X.append(processed)
        labels_list.append(label)

    X = np.array(X)

    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels_list)
    y_categorical = keras.utils.to_categorical(y_encoded)

    num_classes = len(label_encoder.classes_)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return X_train, X_val, y_train, y_val, num_classes, label_encoder


# ============ 라우트 ============

@app.route('/')
def index():
    """메인 페이지 - 리더보드"""
    return render_template('leaderboard.html')


@app.route('/datacollector')
def datacollector():
    """데이터 수집 페이지"""
    return render_template('datacollector.html')


@app.route('/leaderboard')
def leaderboard():
    """리더보드 페이지"""
    return render_template('leaderboard.html')


@app.route('/live')
def live_comparison():
    """실시간 비교 페이지"""
    return render_template('live.html')


# ============ API 엔드포인트 ============

@app.route('/api/data/save', methods=['POST'])
def save_data():
    """데이터 저장 API"""
    try:
        data = request.json
        new_samples = data.get('samples', [])

        # 기존 데이터 로드
        existing_data = load_json_file(COMPARISON_DATA_FILE, {'dataset': []})
        dataset = existing_data.get('dataset', [])

        # 새 데이터 추가
        for sample in new_samples:
            dataset.append({
                'label': sample['label'],
                'landmarks': sample['landmarks'],
                'timestamp': sample.get('timestamp', datetime.now().isoformat())
            })

        # 저장
        save_json_file(COMPARISON_DATA_FILE, {'dataset': dataset})

        return jsonify({
            'success': True,
            'total_samples': len(dataset),
            'added_samples': len(new_samples)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/stats', methods=['GET'])
def get_data_stats():
    """데이터 통계 API"""
    try:
        data = load_json_file(COMPARISON_DATA_FILE, {'dataset': []})
        dataset = data.get('dataset', [])

        # 제스처별 카운트
        gesture_counts = {}
        for item in dataset:
            label = item['label']
            gesture_counts[label] = gesture_counts.get(label, 0) + 1

        return jsonify({
            'success': True,
            'total_samples': len(dataset),
            'num_gestures': len(gesture_counts),
            'gesture_counts': gesture_counts
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/data/reset', methods=['POST'])
def reset_data():
    """데이터 초기화 API"""
    try:
        save_json_file(COMPARISON_DATA_FILE, {'dataset': []})
        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """사용 가능한 모델 목록"""
    models_info = []

    for model_key, model_class in MODEL_CLASSES.items():
        try:
            model_instance = model_class()
            model_instance.build_model()
            params = model_instance.count_parameters()

            models_info.append({
                'key': model_key,
                'name': model_instance.name,
                'parameters': int(params)
            })
        except Exception as e:
            models_info.append({
                'key': model_key,
                'name': model_key.capitalize(),
                'parameters': 0,
                'error': str(e)
            })

    return jsonify({
        'success': True,
        'models': models_info
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """모델 학습 API (기존 방식 - 호환성 유지)"""
    try:
        params = request.json
        model_key = params.get('model', 'baseline')
        epochs = params.get('epochs', 20)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.001)

        # 모델 클래스 가져오기
        if model_key not in MODEL_CLASSES:
            return jsonify({'success': False, 'error': 'Invalid model'}), 400

        # 데이터셋 준비
        result = prepare_dataset(COMPARISON_DATA_FILE)
        if result[0] is None:
            return jsonify({'success': False, 'error': 'No data available'}), 400

        X_train, X_val, y_train, y_val, num_classes, label_encoder = result

        # 모델 생성
        model_class = MODEL_CLASSES[model_key]
        model_instance = model_class(num_classes=num_classes)
        model_instance.build_model()
        model_instance.compile_model(learning_rate=learning_rate)
        model = model_instance.get_model()

        # 학습 시작 시간
        start_time = time.time()

        # 콜백
        class MetricsCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_times = []
                self.epoch_start = None

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start = time.time()

            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start
                self.epoch_times.append(epoch_time)

        metrics_callback = MetricsCallback()

        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[metrics_callback],
            verbose=0
        )

        # 학습 시간 계산
        total_train_time = time.time() - start_time
        avg_epoch_time = np.mean(metrics_callback.epoch_times)

        # 최종 정확도
        train_acc = float(history.history['accuracy'][-1])
        val_acc = float(history.history['val_accuracy'][-1])

        # 추론 속도 측정
        inference_times = []
        for _ in range(100):
            sample = X_val[np.random.randint(len(X_val))].reshape(1, 21, 3)
            start = time.time()
            model.predict(sample, verbose=0)
            inference_times.append(time.time() - start)

        avg_inference_time = np.mean(inference_times) * 1000  # ms

        # 모델 저장
        model_filename = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(MODELS_DIR, f"{model_filename}.h5")
        model.save(model_path)

        # 리더보드 업데이트
        leaderboard = load_json_file(LEADERBOARD_FILE, {'results': []})
        results = leaderboard.get('results', [])

        result_entry = {
            'model_key': model_key,
            'model_name': model_instance.name,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_time': total_train_time,
            'avg_epoch_time': avg_epoch_time,
            'inference_time_ms': avg_inference_time,
            'num_parameters': int(model_instance.count_parameters()),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_samples': len(X_train) + len(X_val),
            'num_classes': num_classes,
            'timestamp': datetime.now().isoformat(),
            'model_file': model_filename
        }

        results.append(result_entry)
        save_json_file(LEADERBOARD_FILE, {'results': results})

        return jsonify({
            'success': True,
            'result': result_entry
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train/stream', methods=['POST'])
def train_model_stream():
    """모델 학습 API - 실시간 스트리밍"""

    def generate():
        try:
            params = request.json
            model_key = params.get('model', 'baseline')
            epochs = params.get('epochs', 20)
            batch_size = params.get('batch_size', 32)
            learning_rate = params.get('learning_rate', 0.001)

            # 모델 클래스 가져오기
            if model_key not in MODEL_CLASSES:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid model'})}\n\n"
                return

            # 진행 상태 전송
            yield f"data: {json.dumps({'type': 'status', 'message': '데이터셋 준비 중...', 'progress': 0})}\n\n"

            # 데이터셋 준비
            result = prepare_dataset(COMPARISON_DATA_FILE)
            if result[0] is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No data available'})}\n\n"
                return

            X_train, X_val, y_train, y_val, num_classes, label_encoder = result

            yield f"data: {json.dumps({'type': 'status', 'message': '모델 생성 중...', 'progress': 5})}\n\n"

            # 모델 생성
            model_class = MODEL_CLASSES[model_key]
            model_instance = model_class(num_classes=num_classes)
            model_instance.build_model()
            model_instance.compile_model(learning_rate=learning_rate)
            model = model_instance.get_model()

            yield f"data: {json.dumps({'type': 'status', 'message': '학습 시작!', 'progress': 10})}\n\n"

            # 학습 시작 시간
            start_time = time.time()

            # 리소스 모니터링 초기화
            resource_monitor = TrainingResourceMonitor()

            # 실시간 콜백 - 에포크마다 즉시 전송
            epoch_results = []

            class StreamingCallback(keras.callbacks.Callback):
                def __init__(self, total_epochs):
                    super().__init__()
                    self.total_epochs = total_epochs
                    self.epoch_times = []
                    self.epoch_start = None

                def on_epoch_begin(self, epoch, logs=None):
                    self.epoch_start = time.time()

                def on_epoch_end(self, epoch, logs=None):
                    epoch_time = time.time() - self.epoch_start
                    self.epoch_times.append(epoch_time)

                    # 진행률 계산 (10% ~ 90%)
                    progress = 10 + int((epoch + 1) / self.total_epochs * 80)

                    # 에포크 정보 저장 (나중에 전송)
                    epoch_data = {
                        'type': 'epoch',
                        'epoch': epoch + 1,
                        'total_epochs': self.total_epochs,
                        'progress': progress,
                        'loss': float(logs.get('loss', 0)),
                        'accuracy': float(logs.get('accuracy', 0)),
                        'val_loss': float(logs.get('val_loss', 0)),
                        'val_accuracy': float(logs.get('val_accuracy', 0)),
                        'epoch_time': epoch_time,
                        'message': f'에포크 {epoch + 1}/{self.total_epochs} 완료'
                    }
                    epoch_results.append(epoch_data)

            streaming_callback = StreamingCallback(epochs)

            # 학습을 별도 스레드에서 실행하고 에포크마다 결과 전송
            import threading
            training_complete = threading.Event()
            training_error = None

            def train_model():
                nonlocal training_error
                try:
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[streaming_callback, resource_monitor],
                        verbose=0
                    )
                except Exception as e:
                    training_error = e
                finally:
                    training_complete.set()

            # 학습 스레드 시작
            training_thread = threading.Thread(target=train_model)
            training_thread.start()

            # 에포크 결과를 실시간으로 전송
            last_sent_epoch = 0
            while not training_complete.is_set() or last_sent_epoch < len(epoch_results):
                # 새로운 에포크 결과가 있으면 전송
                while last_sent_epoch < len(epoch_results):
                    epoch_data = epoch_results[last_sent_epoch]
                    yield f"data: {json.dumps(epoch_data)}\n\n"
                    last_sent_epoch += 1

                # 학습이 완료되지 않았으면 잠시 대기
                if not training_complete.is_set():
                    time.sleep(0.1)

            # 학습 스레드가 완료될 때까지 대기
            training_thread.join()

            # 에러 발생 시 처리
            if training_error:
                raise training_error

            yield f"data: {json.dumps({'type': 'status', 'message': '리소스 통계 수집 중...', 'progress': 91})}\n\n"

            # 학습 시간 계산
            total_train_time = time.time() - start_time
            avg_epoch_time = np.mean(streaming_callback.epoch_times)

            # 최종 정확도 (마지막 에포크 결과에서 가져오기)
            if len(epoch_results) > 0:
                last_epoch = epoch_results[-1]
                train_acc = float(last_epoch['accuracy'])
                val_acc = float(last_epoch['val_accuracy'])
            else:
                train_acc = 0.0
                val_acc = 0.0

            # 리소스 모니터링 통계 가져오기
            resource_stats = resource_monitor.get_statistics()

            yield f"data: {json.dumps({'type': 'status', 'message': '모델 저장 중...', 'progress': 93})}\n\n"

            # 모델 저장
            model_filename = f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = os.path.join(MODELS_DIR, f"{model_filename}.h5")
            model.save(model_path)

            yield f"data: {json.dumps({'type': 'status', 'message': '리소스 측정 중...', 'progress': 95})}\n\n"

            # 상세 리소스 측정
            sample_data = X_val[0:1]  # 단일 샘플
            detailed_resources = measure_all_resources(
                model,
                model_path,
                sample_data,
                (21, 3)
            )

            yield f"data: {json.dumps({'type': 'status', 'message': '결과 저장 중...', 'progress': 98})}\n\n"

            # 리더보드 업데이트
            leaderboard = load_json_file(LEADERBOARD_FILE, {'results': []})
            results = leaderboard.get('results', [])

            result_entry = {
                'model_key': model_key,
                'model_name': model_instance.name,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_time': total_train_time,
                'avg_epoch_time': avg_epoch_time,
                'inference_time_ms': detailed_resources['inference_mean_time_ms'],
                'inference_std_ms': detailed_resources['inference_std_time_ms'],
                'inference_min_ms': detailed_resources['inference_min_time_ms'],
                'inference_max_ms': detailed_resources['inference_max_time_ms'],
                'num_parameters': int(model_instance.count_parameters()),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_samples': len(X_train) + len(X_val),
                'num_classes': num_classes,
                'timestamp': datetime.now().isoformat(),
                'model_file': model_filename,
                # 추가 리소스 정보
                'flops': detailed_resources['flops'],
                'model_size_mb': detailed_resources['model_size_mb'],
                'peak_memory_mb': resource_stats['peak_memory_mb'],
                'memory_increase_mb': resource_stats['memory_increase_mb'],
                'avg_epoch_memory_mb': resource_stats['avg_epoch_memory_mb'],
                'inference_memory_mb': detailed_resources['inference_memory_mb'],
                'gpu_memory_mb': detailed_resources['gpu_memory_mb']
            }

            results.append(result_entry)
            save_json_file(LEADERBOARD_FILE, {'results': results})

            # 완료 메시지
            yield f"data: {json.dumps({'type': 'complete', 'message': '학습 완료!', 'progress': 100, 'result': result_entry})}\n\n"

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """리더보드 조회 API"""
    try:
        leaderboard = load_json_file(LEADERBOARD_FILE, {'results': []})
        results = leaderboard.get('results', [])

        # 정렬 옵션
        sort_by = request.args.get('sort_by', 'val_accuracy')
        order = request.args.get('order', 'desc')

        # 정렬
        reverse = (order == 'desc')
        if sort_by in ['val_accuracy', 'train_accuracy']:
            results_sorted = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=reverse)
        elif sort_by == 'train_time':
            results_sorted = sorted(results, key=lambda x: x.get(sort_by, float('inf')), reverse=not reverse)
        elif sort_by == 'inference_time_ms':
            results_sorted = sorted(results, key=lambda x: x.get(sort_by, float('inf')), reverse=not reverse)
        else:
            results_sorted = results

        return jsonify({
            'success': True,
            'results': results_sorted,
            'total': len(results_sorted)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/leaderboard/clear', methods=['POST'])
def clear_leaderboard():
    """리더보드 초기화 API"""
    try:
        save_json_file(LEADERBOARD_FILE, {'results': []})
        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/leaderboard/delete/<int:index>', methods=['DELETE'])
def delete_leaderboard_entry(index):
    """리더보드 항목 삭제 API"""
    try:
        leaderboard = load_json_file(LEADERBOARD_FILE, {'results': []})
        results = leaderboard.get('results', [])

        if 0 <= index < len(results):
            deleted = results.pop(index)
            save_json_file(LEADERBOARD_FILE, {'results': results})

            # 모델 파일 삭제
            if 'model_file' in deleted:
                model_path = os.path.join(MODELS_DIR, f"{deleted['model_file']}.h5")
                if os.path.exists(model_path):
                    os.remove(model_path)

            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Index out of range'}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============ 실시간 비교 API ============

@app.route('/api/live/models', methods=['GET'])
def get_available_models_for_live():
    """실시간 추론 가능한 모델 목록 조회"""
    try:
        leaderboard = load_json_file(LEADERBOARD_FILE, {'results': []})
        results = leaderboard.get('results', [])

        # 학습된 모델들만 반환
        available_models = []
        for result in results:
            model_file = result.get('model_file')
            model_path = os.path.join(MODELS_DIR, f"{model_file}.h5")

            if os.path.exists(model_path):
                available_models.append({
                    'key': result['model_key'],
                    'name': result['model_name'],
                    'model_file': model_file,
                    'val_accuracy': result['val_accuracy'],
                    'num_classes': result['num_classes']
                })

        return jsonify({
            'success': True,
            'models': available_models
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def get_custom_objects():
    """커스텀 객체 딕셔너리 반환 (EfficientNet 등의 커스텀 레이어 지원)"""
    import tensorflow as tf

    # Swish activation 함수
    @tf.function
    def swish(x):
        return x * tf.nn.sigmoid(x)

    return {
        'swish': swish,
        'Swish': swish,
    }


@app.route('/api/live/load', methods=['POST'])
def load_models_for_live():
    """실시간 추론을 위해 모델들을 메모리에 로드"""
    try:
        data = request.json
        model_files = data.get('model_files', [])

        # 기존 모델 언로드
        global LOADED_MODELS
        LOADED_MODELS.clear()

        # 데이터셋에서 레이블 정보 가져오기
        dataset_data = load_json_file(COMPARISON_DATA_FILE, {'dataset': []})
        dataset = dataset_data.get('dataset', [])

        if len(dataset) == 0:
            return jsonify({'success': False, 'error': 'No dataset available'}), 400

        # Label encoder 준비
        labels_list = [item['label'] for item in dataset]
        label_encoder = LabelEncoder()
        label_encoder.fit(labels_list)
        classes = label_encoder.classes_.tolist()

        # 커스텀 객체 준비
        custom_objects = get_custom_objects()

        # 모델 로드
        loaded_info = []
        for model_file in model_files:
            try:
                model_path = os.path.join(MODELS_DIR, f"{model_file}.h5")

                if not os.path.exists(model_path):
                    loaded_info.append({
                        'model_file': model_file,
                        'loaded': False,
                        'error': 'Model file not found'
                    })
                    continue

                # Keras 모델 로드 (커스텀 객체 포함)
                model = keras.models.load_model(model_path, custom_objects=custom_objects)

                LOADED_MODELS[model_file] = {
                    'model': model,
                    'classes': classes,
                    'label_encoder': label_encoder
                }

                loaded_info.append({
                    'model_file': model_file,
                    'loaded': True
                })

            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"Error loading {model_file}: {error_msg}")
                loaded_info.append({
                    'model_file': model_file,
                    'loaded': False,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'loaded_models': loaded_info,
            'num_classes': len(classes),
            'classes': classes
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/live/predict', methods=['POST'])
def live_predict():
    """실시간 추론 - 모든 로드된 모델에 대해 예측"""
    try:
        data = request.json
        landmarks = data.get('landmarks')

        if not landmarks:
            return jsonify({'success': False, 'error': 'No landmarks provided'}), 400

        # 전처리
        processed = preprocess_landmarks(landmarks)
        input_data = np.expand_dims(processed, axis=0)  # (1, 21, 3)

        # 모든 모델에 대해 예측
        predictions = {}

        for model_file, model_data in LOADED_MODELS.items():
            try:
                model = model_data['model']
                classes = model_data['classes']

                # 추론 시작
                start_time = time.time()
                prediction = model.predict(input_data, verbose=0)
                inference_time = (time.time() - start_time) * 1000  # ms

                # 결과 처리
                top_indices = np.argsort(prediction[0])[::-1][:5]
                top_predictions = []

                for idx in top_indices:
                    top_predictions.append({
                        'label': classes[idx],
                        'confidence': float(prediction[0][idx])
                    })

                predictions[model_file] = {
                    'top_prediction': top_predictions[0],
                    'top_5': top_predictions,
                    'inference_time_ms': inference_time,
                    'success': True
                }

            except Exception as e:
                predictions[model_file] = {
                    'success': False,
                    'error': str(e)
                }

        return jsonify({
            'success': True,
            'predictions': predictions
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============ 정적 파일 서빙 ============

@app.route('/static/<path:filename>')
def serve_static(filename):
    """정적 파일 서빙"""
    return send_from_directory('static', filename)


if __name__ == '__main__':
    print("=" * 60)
    print("모델 비교 시스템 서버 시작")
    print("=" * 60)
    print(f"데이터 수집: http://localhost:5001/datacollector")
    print(f"리더보드: http://localhost:5001/leaderboard")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True)
