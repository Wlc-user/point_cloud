"""
训练好的PCB缺陷分类模型使用指南
如何使用 pcb_defect_classifier.h5 进行推理
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path


class PCBDefectPredictor:
    """PCB缺陷预测器 - 使用训练好的模型"""
    
    def __init__(self, model_path='pcb_defect_classifier.h5', 
                 class_names_path='class_names.json'):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            class_names_path: 类别名称文件路径
        """
        # 加载模型
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # 加载类别名称
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"Classes: {self.class_names}")
        
        # 图像尺寸（必须与训练时一致）
        self.img_size = (224, 224)
        
    def preprocess_image(self, image_path):
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像数组
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # 调整大小
        img = cv2.resize(img, self.img_size)
        
        # 归一化（与训练时一致）
        img = img / 255.0
        
        # 添加batch维度
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path):
        """
        预测单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预测结果字典
        """
        # 预处理
        img = self.preprocess_image(image_path)
        
        # 预测
        predictions = self.model.predict(img, verbose=0)
        
        # 获取最高置信度的类别
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # 获取所有类别的置信度
        all_confidences = {
            class_name: float(conf) 
            for class_name, conf in zip(self.class_names, predictions[0])
        }
        
        # 排序
        sorted_confidences = dict(sorted(
            all_confidences.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_confidences': sorted_confidences,
            'top3': list(sorted_confidences.items())[:3]
        }
    
    def predict_batch(self, image_paths):
        """
        批量预测
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            预测结果列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
                print(f"✓ {Path(path).name}: {result['predicted_class']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ {Path(path).name}: Error - {e}")
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        可视化预测结果
        
        Args:
            image_path: 图像文件路径
            save_path: 保存路径（可选）
        """
        # 预测
        result = self.predict(image_path)
        
        # 读取原图
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建结果图像
        h, w = img.shape[:2]
        result_img = np.ones((h + 150, w, 3), dtype=np.uint8) * 255
        result_img[:h, :w] = img
        
        # 添加预测结果文字
        text = f"Prediction: {result['predicted_class']}"
        conf_text = f"Confidence: {result['confidence']:.2%}"
        
        # 根据置信度选择颜色
        if result['confidence'] > 0.8:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif result['confidence'] > 0.5:
            color = (0, 165, 255)  # 橙色 - 中等置信度
        else:
            color = (255, 0, 0)  # 红色 - 低置信度
        
        cv2.putText(result_img, text, (10, h + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(result_img, conf_text, (10, h + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 添加Top-3
        y_offset = h + 120
        for i, (class_name, conf) in enumerate(result['top3']):
            top_text = f"{i+1}. {class_name}: {conf:.2%}"
            cv2.putText(result_img, top_text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 保存或显示
        if save_path:
            result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, result_img_bgr)
            print(f"Result saved to: {save_path}")
        else:
            # 显示
            result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Prediction Result', result_img_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_img


def demo_single_prediction():
    """单张图像预测演示"""
    print("="*70)
    print("PCB Defect Prediction Demo - Single Image")
    print("="*70)
    
    # 创建预测器
    predictor = PCBDefectPredictor(
        model_path='pcb_defect_classifier.h5',
        class_names_path='class_names.json'
    )
    
    # 测试图像路径（使用数据集中的图像）
    test_image = 'real_images/pcb/images/01_open_circuit_01.jpg'
    
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path")
        return
    
    # 预测
    print(f"\nPredicting: {test_image}")
    result = predictor.predict(test_image)
    
    # 打印结果
    print("\nPrediction Result:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\nTop 3 Predictions:")
    for i, (class_name, conf) in enumerate(result['top3'], 1):
        print(f"  {i}. {class_name}: {conf:.2%}")
    
    # 可视化
    print("\nVisualizing result...")
    predictor.visualize_prediction(test_image, save_path='prediction_result.jpg')


def demo_batch_prediction():
    """批量预测演示"""
    print("="*70)
    print("PCB Defect Prediction Demo - Batch")
    print("="*70)
    
    # 创建预测器
    predictor = PCBDefectPredictor()
    
    # 获取测试图像（每种类型取2张）
    test_images = []
    image_dir = Path('real_images/pcb/images')
    
    if image_dir.exists():
        for defect_type in ['open_circuit', 'short', 'mouse_bite']:
            images = list(image_dir.glob(f'01_{defect_type}_*.jpg'))[:2]
            test_images.extend(images)
    
    if not test_images:
        print("No test images found")
        return
    
    # 批量预测
    print(f"\nBatch predicting {len(test_images)} images...\n")
    results = predictor.predict_batch([str(p) for p in test_images])
    
    # 统计
    print("\n" + "="*70)
    print("Batch Prediction Summary")
    print("="*70)
    correct = sum(1 for r in results if r['confidence'] > 0.7)
    print(f"Total: {len(results)}")
    print(f"High confidence (>70%): {correct}")
    print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.2%}")


def demo_folder_prediction():
    """文件夹批量预测并保存结果"""
    print("="*70)
    print("PCB Defect Prediction Demo - Folder")
    print("="*70)
    
    # 创建预测器
    predictor = PCBDefectPredictor()
    
    # 输入文件夹
    input_folder = Path('real_images/pcb/images')
    output_folder = Path('prediction_results')
    output_folder.mkdir(exist_ok=True)
    
    if not input_folder.exists():
        print(f"Input folder not found: {input_folder}")
        return
    
    # 获取所有图像
    image_files = list(input_folder.glob('*.jpg'))[:20]  # 只取前20张做演示
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Results will be saved to: {output_folder}\n")
    
    # 预测并保存
    for img_path in image_files:
        try:
            result = predictor.predict(str(img_path))
            save_path = output_folder / f"{img_path.stem}_result.jpg"
            predictor.visualize_prediction(str(img_path), str(save_path))
            print(f"✓ {img_path.name} -> {result['predicted_class']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"✗ {img_path.name}: Error - {e}")
    
    print(f"\nAll results saved to: {output_folder}")


def main():
    """主程序 - 选择演示模式"""
    import sys
    
    print("="*70)
    print("PCB Defect Prediction - Model Usage Demo")
    print("="*70)
    print("\nAvailable demos:")
    print("  1. Single image prediction")
    print("  2. Batch prediction (multiple images)")
    print("  3. Folder prediction (save all results)")
    print("\nUsage: python use_trained_model.py [1|2|3]")
    
    # 检查模型文件
    if not Path('pcb_defect_classifier.h5').exists():
        print("\n⚠️  Model file not found: pcb_defect_classifier.h5")
        print("Please train the model first using: python train_pcb_classifier.py")
        return
    
    # 选择模式
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("\nSelect demo mode (1/2/3): ").strip()
    
    if mode == '1':
        demo_single_prediction()
    elif mode == '2':
        demo_batch_prediction()
    elif mode == '3':
        demo_folder_prediction()
    else:
        print("Invalid mode. Please select 1, 2, or 3.")


if __name__ == '__main__':
    main()
