"""
PCB缺陷分类器训练脚本
使用真实数据进行深度学习模型训练
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from pcb_dataset_manager import PCBDatasetManager


class PCBDefectClassifier:
    """PCB缺陷分类器"""
    
    def __init__(self, img_size=(224, 224), num_classes=6):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            'missing_hole', 'mouse_bite', 'open_circuit',
            'short', 'spur', 'spurious_copper'
        ]
        self.history = None
        
    def load_data(self, dataset_manager: PCBDatasetManager):
        """从数据集管理器加载数据"""
        print("Loading data for training...")
        
        images = []
        labels = []
        
        for defect in dataset_manager.defects:
            # 加载图像
            img = cv2.imread(defect.image_path)
            if img is None:
                continue
            
            # 预处理
            img = cv2.resize(img, self.img_size)
            img = img / 255.0  # 归一化
            
            images.append(img)
            
            # 获取标签
            label_idx = self.class_names.index(defect.defect_type)
            labels.append(label_idx)
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} samples")
        print(f"Image shape: {X.shape}")
        print(f"Number of classes: {self.num_classes}")
        
        return X, y
    
    def build_cnn_model(self):
        """构建CNN模型"""
        model = models.Sequential([
            # 第一个卷积块
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第二个卷积块
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第三个卷积块
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 第四个卷积块
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 全连接层
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model built successfully")
        model.summary()
        return model
    
    def build_transfer_learning_model(self):
        """使用预训练模型（迁移学习）"""
        # 使用MobileNetV2作为基础模型
        base_model = keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # 冻结基础模型
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Transfer learning model built successfully")
        model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """训练模型"""
        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # 数据增强
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        # 回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_pcb_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 训练
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        print("\nEvaluating model...")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("Confusion matrix saved to: confusion_matrix.png")
        plt.close()
        
        # 计算准确率
        accuracy = np.sum(y_pred_classes == y_test) / len(y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 损失
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("Training history saved to: training_history.png")
        plt.close()
    
    def save_model(self, filepath='pcb_defect_classifier.h5'):
        """保存模型"""
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
        
        # 保存类别名称
        with open('class_names.json', 'w') as f:
            json.dump(self.class_names, f)
        print("Class names saved to: class_names.json")
    
    def load_model(self, filepath='pcb_defect_classifier.h5'):
        """加载模型"""
        self.model = keras.models.load_model(filepath)
        
        # 加载类别名称
        with open('class_names.json', 'r') as f:
            self.class_names = json.load(f)
        
        print(f"Model loaded from: {filepath}")


def main():
    """主程序"""
    print("="*70)
    print("PCB Defect Classifier Training")
    print("="*70)
    
    # 1. 加载数据集
    print("\n[1/5] Loading dataset...")
    manager = PCBDatasetManager("real_images/pcb/images")
    manager.load_dataset()
    manager.print_statistics()
    
    # 2. 创建分类器
    print("\n[2/5] Creating classifier...")
    classifier = PCBDefectClassifier(img_size=(224, 224), num_classes=6)
    
    # 3. 加载数据
    print("\n[3/5] Preparing data...")
    X, y = classifier.load_data(manager)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # 4. 构建模型
    print("\n[4/5] Building model...")
    # 选择模型类型
    use_transfer_learning = True
    
    if use_transfer_learning:
        classifier.build_transfer_learning_model()
    else:
        classifier.build_cnn_model()
    
    # 5. 训练
    print("\n[5/5] Training model...")
    classifier.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=16)
    
    # 6. 评估
    print("\n" + "="*70)
    print("Evaluation")
    print("="*70)
    classifier.evaluate(X_test, y_test)
    
    # 7. 绘制训练历史
    classifier.plot_training_history()
    
    # 8. 保存模型
    classifier.save_model('pcb_defect_classifier.h5')
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - pcb_defect_classifier.h5 (trained model)")
    print("  - class_names.json (class labels)")
    print("  - confusion_matrix.png")
    print("  - training_history.png")
    print("  - best_pcb_model.h5 (best checkpoint)")


if __name__ == '__main__':
    main()
