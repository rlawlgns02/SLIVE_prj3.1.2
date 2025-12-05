"""
컴퓨팅 자원 모니터링 유틸리티
학습 및 추론 시 메모리, FLOPs, 모델 크기 등을 측정합니다.
"""

import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback


class ResourceMonitor:
    """컴퓨팅 자원 모니터링 클래스"""

    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0

    def get_memory_usage(self):
        """현재 메모리 사용량 (MB)"""
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Bytes to MB

    def get_gpu_memory_usage(self):
        """GPU 메모리 사용량 (MB)"""
        if not self.gpu_available:
            return None

        try:
            # TensorFlow의 메모리 정보 가져오기
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                # pynvml을 사용하여 GPU 메모리 확인 (선택적)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    return info.used / (1024 * 1024)  # Bytes to MB
                except:
                    pass
            return None
        except:
            return None

    def calculate_flops(self, model, input_shape):
        """모델의 FLOPs 계산"""
        try:
            # TensorFlow 2.x에서 FLOPs 계산
            forward_pass = tf.function(
                model.call,
                input_signature=[tf.TensorSpec(shape=(1,) + input_shape)]
            )

            graph = forward_pass.get_concrete_function().graph

            # TensorFlow Profiler 사용
            from tensorflow.python.profiler.model_analyzer import profile
            from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

            opts = ProfileOptionBuilder.float_operation()
            flops = profile(graph, options=opts)

            if flops and hasattr(flops, 'total_float_ops'):
                return flops.total_float_ops

            # 대안: 간단한 추정
            return self._estimate_flops(model)
        except:
            # 에러 발생 시 간단한 추정 사용
            return self._estimate_flops(model)

    def _estimate_flops(self, model):
        """FLOPs 간단 추정 (레이어 기반)"""
        total_flops = 0

        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                # Dense layer: 2 * input_dim * output_dim (곱셈 + 덧셈)
                input_dim = layer.input_shape[-1]
                output_dim = layer.units
                total_flops += 2 * input_dim * output_dim

            elif isinstance(layer, keras.layers.Conv1D):
                # Conv1D: 2 * kernel_size * input_channels * output_channels * output_length
                kernel_size = layer.kernel_size[0]
                input_channels = layer.input_shape[-1]
                output_channels = layer.filters
                output_length = layer.output_shape[1]
                total_flops += 2 * kernel_size * input_channels * output_channels * output_length

            elif isinstance(layer, keras.layers.Conv2D):
                # Conv2D: 2 * kernel_h * kernel_w * input_channels * output_channels * output_h * output_w
                kernel_h, kernel_w = layer.kernel_size
                input_channels = layer.input_shape[-1]
                output_channels = layer.filters
                output_h, output_w = layer.output_shape[1:3]
                total_flops += 2 * kernel_h * kernel_w * input_channels * output_channels * output_h * output_w

            elif isinstance(layer, keras.layers.BatchNormalization):
                # BatchNorm: 2 * num_features (정규화 + 스케일링)
                num_features = np.prod(layer.output_shape[1:])
                total_flops += 2 * num_features

        return total_flops

    def get_model_size(self, model_path):
        """모델 파일 크기 (MB)"""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            return size_bytes / (1024 * 1024)  # Bytes to MB
        return None

    def measure_inference_time_detailed(self, model, sample_data, num_runs=100):
        """추론 시간 상세 측정"""
        times = []
        memory_before = []
        memory_after = []

        for _ in range(num_runs):
            mem_before = self.get_memory_usage()
            memory_before.append(mem_before)

            start = time.time()
            _ = model.predict(sample_data, verbose=0)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

            mem_after = self.get_memory_usage()
            memory_after.append(mem_after)

        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'mean_memory_mb': np.mean(memory_after),
            'memory_increase_mb': np.mean(np.array(memory_after) - np.array(memory_before))
        }


class TrainingResourceMonitor(Callback):
    """학습 중 자원 모니터링을 위한 Keras Callback"""

    def __init__(self):
        super().__init__()
        self.monitor = ResourceMonitor()
        self.epoch_memories = []
        self.epoch_times = []
        self.batch_times = []
        self.peak_memory = 0
        self.start_memory = 0
        self.start_time = 0
        self.total_time = 0
        self.end_memory = 0

    def on_train_begin(self, logs=None):
        """학습 시작 시"""
        self.start_memory = self.monitor.get_memory_usage()
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        """에포크 시작 시"""
        self.epoch_start_time = time.time()
        self.epoch_start_memory = self.monitor.get_memory_usage()

    def on_epoch_end(self, epoch, logs=None):
        """에포크 종료 시"""
        epoch_time = time.time() - self.epoch_start_time
        current_memory = self.monitor.get_memory_usage()

        self.epoch_times.append(epoch_time)
        self.epoch_memories.append(current_memory)

        # 피크 메모리 업데이트
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def on_batch_begin(self, batch, logs=None):
        """배치 시작 시"""
        self.batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        """배치 종료 시"""
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)

    def on_train_end(self, logs=None):
        """학습 종료 시"""
        self.total_time = time.time() - self.start_time
        self.end_memory = self.monitor.get_memory_usage()

    def get_statistics(self):
        """통계 반환"""
        return {
            'total_training_time': self.total_time,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': self.end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': self.end_memory - self.start_memory,
            'avg_epoch_memory_mb': np.mean(self.epoch_memories) if self.epoch_memories else 0
        }


def get_system_info():
    """시스템 정보 조회"""
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB

    gpu_info = []
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for i, gpu in enumerate(gpu_devices):
            gpu_info.append({
                'id': i,
                'name': gpu.name,
                'device_type': gpu.device_type
            })

    return {
        'cpu_count_physical': cpu_count,
        'cpu_count_logical': cpu_count_logical,
        'total_memory_gb': total_memory,
        'gpu_available': len(gpu_devices) > 0,
        'gpu_count': len(gpu_devices),
        'gpu_info': gpu_info
    }


def measure_all_resources(model, model_path, sample_data, input_shape):
    """모든 리소스 측정"""
    monitor = ResourceMonitor()

    # FLOPs 계산
    flops = monitor.calculate_flops(model, input_shape)

    # 모델 크기
    model_size_mb = monitor.get_model_size(model_path)

    # 추론 시간 상세 측정
    inference_stats = monitor.measure_inference_time_detailed(model, sample_data, num_runs=100)

    # GPU 메모리
    gpu_memory = monitor.get_gpu_memory_usage()

    return {
        'flops': int(flops) if flops else None,
        'model_size_mb': float(model_size_mb) if model_size_mb else None,
        'inference_mean_time_ms': float(inference_stats['mean_time_ms']),
        'inference_std_time_ms': float(inference_stats['std_time_ms']),
        'inference_min_time_ms': float(inference_stats['min_time_ms']),
        'inference_max_time_ms': float(inference_stats['max_time_ms']),
        'inference_memory_mb': float(inference_stats['mean_memory_mb']),
        'gpu_memory_mb': float(gpu_memory) if gpu_memory else None
    }
