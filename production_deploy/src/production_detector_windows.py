"""
生产环境PCB缺陷检测器 - Windows版本
"""
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
import threading
import queue
import winsound  # Windows声音报警

# 配置日志
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'detector.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionDetector:
    """生产环境检测器 - Windows版本"""
    
    def __init__(self, config_path='config/production.yaml'):
        # 加载配置
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 使用默认配置
            self.config = self.get_default_config()
        
        # 初始化模型
        self.load_model()
        
        # 初始化相机
        self.init_camera()
        
        # 初始化统计
        self.stats = {
            'total_inspected': 0,
            'total_defects': 0,
            'start_time': datetime.now()
        }
        
        self.running = False
        
    def get_default_config(self):
        """获取默认配置"""
        # 获取脚本所在目录
        script_dir = Path(__file__).parent
        # 获取项目根目录（上两级）
        project_root = script_dir.parent.parent
        
        return {
            'system': {
                'name': 'PCB Defect Detection System',
                'version': '1.0.0',
                'debug': False
            },
            'input': {
                'source_type': 'camera',
                'camera': {
                    'id': 0,
                    'width': 1920,
                    'height': 1080,
                    'fps': 30,
                    'exposure': -1
                }
            },
            'model': {
                'path': str(project_root / 'pcb_defect_classifier.h5'),
                'class_names_path': str(project_root / 'class_names.json'),
                'confidence_threshold': 0.7,
                'input_size': [224, 224]
            },
            'output': {
                'save_images': True,
                'save_path': str(project_root / 'logs' / 'detection_results'),
                'log_level': 'INFO'
            },
            'alarm': {
                'enabled': True,
                'sound': {
                    'enabled': True
                }
            }
        }
        
    def load_model(self):
        """加载模型"""
        model_config = self.config['model']
        model_path = model_config['path']
        
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        
        class_names_path = model_config['class_names_path']
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
        
        self.confidence_threshold = model_config['confidence_threshold']
        self.input_size = tuple(model_config['input_size'])
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Classes: {self.class_names}")
        
    def init_camera(self):
        """初始化相机"""
        camera_config = self.config['input']['camera']
        camera_id = camera_config['id']
        
        logger.info(f"Initializing camera {camera_id}...")
        
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            logger.warning(f"Failed to open camera {camera_id}, switching to folder mode")
            self.config['input']['source_type'] = 'folder'
            self.init_folder_mode()
            return
        
        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])
        
        # 获取实际分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera initialized: {actual_width}x{actual_height}@{actual_fps}fps")
    
    def init_folder_mode(self):
        """初始化文件夹模式"""
        import glob
        
        # 查找测试图像
        test_patterns = [
            'real_images/pcb/images/*.jpg',
            'deeppcb_*.jpg',
            '*.jpg'
        ]
        
        self.image_list = []
        for pattern in test_patterns:
            self.image_list.extend(glob.glob(pattern))
        
        if not self.image_list:
            raise RuntimeError("No test images found")
        
        self.current_image_idx = 0
        logger.info(f"Folder mode: Found {len(self.image_list)} test images")
        
    def preprocess(self, frame):
        """预处理图像"""
        img = cv2.resize(frame, self.input_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """预测单帧"""
        img = self.preprocess(frame)
        predictions = self.model.predict(img, verbose=0)
        
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = self.class_names[predicted_idx]
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'is_defective': confidence > self.confidence_threshold,
            'all_probs': {
                name: float(prob) 
                for name, prob in zip(self.class_names, predictions[0])
            }
        }
    
    def save_result(self, frame, result):
        """保存检测结果"""
        if not self.config['output']['save_images']:
            return
        
        save_path = Path(self.config['output']['save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result['class']}_{timestamp}_{result['confidence']:.2f}.jpg"
        
        # 添加文字标注
        display_frame = frame.copy()
        text = f"{result['class']}: {result['confidence']:.2%}"
        color = (0, 0, 255) if result['is_defective'] else (0, 255, 0)
        cv2.putText(display_frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imwrite(str(save_path / filename), display_frame)
        
    def log_result(self, result):
        """记录检测结果"""
        self.stats['total_inspected'] += 1
        if result['is_defective']:
            self.stats['total_defects'] += 1
        
        status = "DEFECT" if result['is_defective'] else "OK"
        logger.info(f"[{status}] {result['class']} ({result['confidence']:.2%})")
    
    def trigger_alarm(self, result):
        """触发报警 - Windows版本"""
        if not self.config['alarm']['enabled']:
            return
        
        if result['is_defective']:
            logger.warning(f"[ALERT] DEFECT DETECTED: {result['class']}")
            
            # Windows声音报警
            if self.config['alarm']['sound']['enabled']:
                try:
                    # 播放系统蜂鸣声
                    winsound.Beep(1000, 500)  # 1000Hz, 500ms
                    winsound.Beep(800, 500)   # 800Hz, 500ms
                except Exception as e:
                    logger.error(f"Failed to play sound: {e}")
    
    def print_stats(self):
        """打印统计信息"""
        runtime = datetime.now() - self.stats['start_time']
        hours = runtime.total_seconds() / 3600
        
        defect_rate = (self.stats['total_defects'] / self.stats['total_inspected'] * 100) \
                      if self.stats['total_inspected'] > 0 else 0
        
        throughput = self.stats['total_inspected'] / hours if hours > 0 else 0
        
        logger.info("="*60)
        logger.info("Production Statistics")
        logger.info("="*60)
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Total Inspected: {self.stats['total_inspected']}")
        logger.info(f"Total Defects: {self.stats['total_defects']}")
        logger.info(f"Defect Rate: {defect_rate:.2f}%")
        logger.info(f"Throughput: {throughput:.1f} pcs/hour")
        logger.info("="*60)
    
    def run(self):
        """运行检测循环"""
        logger.info("Starting production detection system...")
        logger.info("Press 'Q' to quit, 'S' for statistics")
        self.running = True
        
        try:
            while self.running:
                # 根据输入源获取图像
                if self.config['input']['source_type'] == 'folder':
                    # 文件夹模式
                    if self.current_image_idx >= len(self.image_list):
                        logger.info("All images processed")
                        break
                    frame = cv2.imread(self.image_list[self.current_image_idx])
                    self.current_image_idx += 1
                    if frame is None:
                        continue
                else:
                    # 相机模式
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.error("Failed to capture frame")
                        time.sleep(0.1)
                        continue
                
                # 检测
                start_time = time.time()
                result = self.predict(frame)
                process_time = time.time() - start_time
                
                # 记录结果
                self.log_result(result)
                
                # 保存图像
                self.save_result(frame, result)
                
                # 触发报警
                self.trigger_alarm(result)
                
                # 显示结果
                display_frame = frame.copy()
                
                # 添加检测信息
                text = f"{result['class']}: {result['confidence']:.2%}"
                color = (0, 0, 255) if result['is_defective'] else (0, 255, 0)
                cv2.putText(display_frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 添加处理时间
                cv2.putText(display_frame, f"Time: {process_time*1000:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 添加统计信息
                cv2.putText(display_frame, f"Total: {self.stats['total_inspected']}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示窗口
                cv2.imshow('🔍 PCB Defect Detection - Production', display_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("用户请求退出")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.print_stats()
                
        except KeyboardInterrupt:
            logger.info("⛔ 用户中断")
        except Exception as e:
            logger.error(f"❌ 发生错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止检测器"""
        logger.info("Stopping detection system...")
        self.running = False
        self.print_stats()
        
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        logger.info("Detection system stopped")


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB Defect Detection - Production')
    parser.add_argument('--config', default='config/production.yaml',
                       help='配置文件路径')
    parser.add_argument('--camera', type=int, default=0,
                       help='相机ID (默认: 0)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='置信度阈值 (默认: 0.7)')
    args = parser.parse_args()
    
    # 打印系统信息
    print("="*70)
    print("PCB Defect Detection System - Production")
    print("="*70)
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print("="*70)
    
    try:
        # 启动检测器
        detector = ProductionDetector(args.config)
        detector.run()
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {e}")
        print("\n💡 提示: 请确保模型文件存在:")
        print("  - pcb_defect_classifier.h5")
        print("  - class_names.json")
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        raise


if __name__ == '__main__':
    main()
