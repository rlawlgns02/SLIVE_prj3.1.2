"""
ResNet 모델 - Residual Network 구조
손 랜드마크 데이터를 위한 1D ResNet 구현
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ResNetModel:
    """ResNet 모델 (1D Residual Blocks for hand landmarks)"""

    def __init__(self, num_classes=74, input_shape=(21, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.name = "ResNet"

    def residual_block(self, x, filters, kernel_size=3, stride=1):
        """Residual Block 구현"""
        shortcut = x

        # First conv
        x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second conv
        x = layers.Conv1D(filters, kernel_size, strides=1, padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                    kernel_initializer='he_normal')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        return x

    def build_model(self):
        """ResNet 모델 구조 생성"""
        inputs = layers.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv1D(64, 7, strides=2, padding='same',
                         kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

        # Residual blocks
        x = self.residual_block(x, 64)
        x = self.residual_block(x, 64)

        x = self.residual_block(x, 128, stride=2)
        x = self.residual_block(x, 128)

        x = self.residual_block(x, 256, stride=2)
        x = self.residual_block(x, 256)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(256, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)

        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

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
