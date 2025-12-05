"""
SLIVE 모델 - 원본 SLIVE 프로젝트의 고정확도 모델
규칙 기반 + 딥러닝 하이브리드 접근법
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class SLIVEModel:
    """SLIVE 프로젝트의 원본 모델 (Flatten + Dense)"""

    def __init__(self, num_classes=74, input_shape=(21, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.name = "SLIVE (Original)"

    def build_model(self):
        """모델 구조 생성"""
        model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Flatten
            layers.Flatten(),

            # Dense Block 1
            layers.Dense(256, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            # Dense Block 2
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.3),

            # Dense Block 3
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),

            # Output
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """모델 컴파일"""
        if self.model is None:
            self.build_model()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def get_model(self):
        """모델 반환"""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model

    def summary(self):
        """모델 요약"""
        if self.model is None:
            self.build_model()
        return self.model.summary()

    def count_parameters(self):
        """파라미터 수 계산"""
        if self.model is None:
            self.build_model()
        return self.model.count_params()
