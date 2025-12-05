"""
DenseNet 모델 - Densely Connected Network 구조
손 랜드마크 데이터를 위한 1D DenseNet 구현
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DenseNetModel:
    """DenseNet 모델 (1D Dense Blocks for hand landmarks)"""

    def __init__(self, num_classes=74, input_shape=(21, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.name = "DenseNet"

    def conv_block(self, x, growth_rate):
        """Dense Block의 기본 Convolution Block"""
        x1 = layers.BatchNormalization()(x)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv1D(growth_rate * 4, 1, padding='same',
                          kernel_initializer='he_normal')(x1)

        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv1D(growth_rate, 3, padding='same',
                          kernel_initializer='he_normal')(x1)

        # Concatenate
        x = layers.Concatenate()([x, x1])
        return x

    def dense_block(self, x, num_layers, growth_rate):
        """Dense Block 구현"""
        for _ in range(num_layers):
            x = self.conv_block(x, growth_rate)
        return x

    def transition_block(self, x, compression_factor=0.5):
        """Transition Block (downsampling)"""
        num_filters = int(x.shape[-1] * compression_factor)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv1D(num_filters, 1, padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.AveragePooling1D(2, strides=2, padding='same')(x)

        return x

    def build_model(self):
        """DenseNet 모델 구조 생성"""
        inputs = layers.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv1D(64, 7, strides=2, padding='same',
                         kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

        # Dense Block 1
        x = self.dense_block(x, num_layers=4, growth_rate=32)
        x = self.transition_block(x)

        # Dense Block 2
        x = self.dense_block(x, num_layers=4, growth_rate=32)
        x = self.transition_block(x)

        # Dense Block 3
        x = self.dense_block(x, num_layers=4, growth_rate=32)

        # Global pooling
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
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
