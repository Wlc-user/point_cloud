import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

class ModelTrainer:
    def __init__(self):
        pass
    
    def create_model(self, model_type='classification', input_shape=(224, 224, 3), num_classes=2):
        """
        创建深度学习模型
        
        Args:
            model_type: 模型类型 (classification, detection, segmentation)
            input_shape: 输入图像形状
            num_classes: 类别数量
            
        Returns:
            Keras模型
        """
        if model_type == 'classification':
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
        elif model_type == 'detection':
            # 简化的目标检测模型
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(4 + num_classes, activation='sigmoid')  # 4个坐标 + 类别
            ])
        elif model_type == 'segmentation':
            # 简化的图像分割模型
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                Flatten(),
                Dense(input_shape[0] * input_shape[1] * num_classes, activation='softmax'),
                tf.keras.layers.Reshape((input_shape[0], input_shape[1], num_classes))
            ])
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model
    
    def train_model(self, model, train_data, val_data, **kwargs):
        """
        训练模型
        
        Args:
            model: Keras模型
            train_data: 训练数据
            val_data: 验证数据
            **kwargs: 训练参数
            
        Returns:
            训练历史
        """
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练模型
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history
    
    def prepare_data(self, data_dir, **kwargs):
        """
        准备训练数据
        
        Args:
            data_dir: 数据目录
            **kwargs: 数据处理参数
            
        Returns:
            训练和验证数据生成器
        """
        target_size = kwargs.get('target_size', (224, 224))
        batch_size = kwargs.get('batch_size', 32)
        validation_split = kwargs.get('validation_split', 0.2)
        
        # 数据增强
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # 训练数据生成器
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # 验证数据生成器
        val_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def evaluate_model(self, model, test_data):
        """
        评估模型
        
        Args:
            model: Keras模型
            test_data: 测试数据
            
        Returns:
            评估结果
        """
        return model.evaluate(test_data)
    
    def save_model(self, model, model_path):
        """
        保存模型
        
        Args:
            model: Keras模型
            model_path: 保存路径
        """
        model.save(model_path)
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            Keras模型
        """
        return tf.keras.models.load_model(model_path)