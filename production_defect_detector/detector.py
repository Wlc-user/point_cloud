"""
工业缺陷检测器 - 实战版本
支持多种输入源、检测模式和输出方式
"""
import cv2
import numpy as np
import yaml
import json
import csv
import sqlite3
import logging
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InputSource(Enum):
    """输入源类型"""
    CAMERA = "camera"
    FOLDER = "folder"
    RTSP = "rtsp"


class DetectionMode(Enum):
    """检测模式"""
    TEMPLATE_MATCH = "template_match"
    DEEP_LEARNING = "deep_learning"
    RULE_BASED = "rule_based"


@dataclass
class Defect:
    """缺陷数据类"""
    id: int
    type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    severity: str
    timestamp: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionResult:
    """检测结果数据类"""
    image_path: str
    timestamp: str
    defects: List[Defect]
    total_defects: int
    severe_defects: int
    processing_time: float
    
    def to_dict(self):
        return {
            'image_path': self.image_path,
            'timestamp': self.timestamp,
            'total_defects': self.total_defects,
            'severe_defects': self.severe_defects,
            'processing_time': self.processing_time,
            'defects': [d.to_dict() for d in self.defects]
        }


class ImageSource:
    """图像源基类"""
    def __init__(self, config: dict):
        self.config = config
        self.running = False
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def get_frame(self) -> Optional[np.ndarray]:
        raise NotImplementedError
    
    def release(self):
        pass


class CameraSource(ImageSource):
    """USB相机图像源"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.camera_id = config.get('id', 0)
        self.width = config.get('width', 1920)
        self.height = config.get('height', 1080)
        self.fps = config.get('fps', 30)
        self.cap = None
    
    def start(self):
        super().start()
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(f"Camera {self.camera_id} started: {self.width}x{self.height}@{self.fps}fps")
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.running or self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("Camera released")


class FolderSource(ImageSource):
    """文件夹图像源"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.path = Path(config.get('path', './input_images'))
        self.extensions = config.get('extensions', ['.jpg', '.png'])
        self.recursive = config.get('recursive', False)
        self.move_after_process = config.get('move_after_process', True)
        self.image_list = []
        self.current_index = 0
        self.processed_dir = self.path / 'processed'
    
    def start(self):
        super().start()
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created input directory: {self.path}")
        
        # 获取所有图像文件
        pattern = '**/*' if self.recursive else '*'
        self.image_list = [
            f for f in self.path.glob(pattern)
            if f.suffix.lower() in self.extensions and f.is_file()
        ]
        self.image_list.sort()
        self.current_index = 0
        
        if self.move_after_process:
            self.processed_dir.mkdir(exist_ok=True)
        
        logger.info(f"Found {len(self.image_list)} images in {self.path}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        if not self.running or self.current_index >= len(self.image_list):
            return None
        
        image_path = self.image_list[self.current_index]
        frame = cv2.imread(str(image_path))
        
        if frame is not None:
            # 移动到processed文件夹
            if self.move_after_process:
                dest = self.processed_dir / image_path.name
                try:
                    image_path.rename(dest)
                except Exception as e:
                    logger.warning(f"Cannot move file: {e}")
        
        self.current_index += 1
        return frame
    
    def get_current_path(self) -> str:
        if self.current_index > 0 and self.current_index <= len(self.image_list):
            return str(self.image_list[self.current_index - 1])
        return ""


class DefectDetector:
    """缺陷检测器基类"""
    def __init__(self, config: dict):
        self.config = config
        self.product_type = config.get('product_type', 'general')
    
    def detect(self, image: np.ndarray, template: Optional[np.ndarray] = None) -> List[Defect]:
        raise NotImplementedError
    
    def classify_severity(self, defect_type: str, area: int) -> str:
        """分类缺陷严重程度"""
        severe_types = self.config.get('alert', {}).get('severe_defects', [])
        if defect_type in severe_types:
            return "HIGH"
        elif area > 1000:
            return "MEDIUM"
        else:
            return "LOW"


class TemplateMatchDetector(DefectDetector):
    """模板匹配检测器"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.template_config = config.get('template_match', {})
        self.threshold = self.template_config.get('threshold', 30)
        self.min_area = self.template_config.get('min_defect_area', 50)
        self.max_area = self.template_config.get('max_defect_area', 10000)
        self.template = None
        self.load_template()
    
    def load_template(self):
        """加载模板图像"""
        template_path = self.template_config.get('template_path', './templates/template.jpg')
        if os.path.exists(template_path):
            self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            logger.info(f"Template loaded: {template_path}")
        else:
            logger.warning(f"Template not found: {template_path}")
    
    def detect(self, image: np.ndarray, template: Optional[np.ndarray] = None) -> List[Defect]:
        if template is None:
            template = self.template
        
        if template is None:
            logger.error("No template available")
            return []
        
        # 确保尺寸一致
        if image.shape[:2] != template.shape[:2]:
            template = cv2.resize(template, (image.shape[1], image.shape[0]))
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算差异
        diff = cv2.absdiff(gray, template)
        
        # 二值化
        _, binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        timestamp = datetime.now().isoformat()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # 缺陷分类
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 3 or aspect_ratio < 0.33:
                defect_type = "scratch"
            elif area > 500:
                defect_type = "large_defect"
            else:
                defect_type = "small_defect"
            
            severity = self.classify_severity(defect_type, int(area))
            
            defect = Defect(
                id=i,
                type=defect_type,
                confidence=min(area / 1000, 1.0),
                bbox=(x, y, w, h),
                area=int(area),
                severity=severity,
                timestamp=timestamp
            )
            defects.append(defect)
        
        return defects


class RuleBasedDetector(DefectDetector):
    """基于规则的检测器"""
    def __init__(self, config: dict):
        super().__init__(config)
        self.rule_config = config.get('rule_based', {})
    
    def detect(self, image: np.ndarray, template: Optional[np.ndarray] = None) -> List[Defect]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测配置
        edge_config = self.rule_config.get('edge_detection', {})
        method = edge_config.get('method', 'canny')
        
        if method == 'canny':
            edges = cv2.Canny(gray, 
                            edge_config.get('threshold1', 50),
                            edge_config.get('threshold2', 150))
        else:
            edges = cv2.Canny(gray, 50, 150)
        
        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        timestamp = datetime.now().isoformat()
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            defect = Defect(
                id=i,
                type="edge_anomaly",
                confidence=min(area / 500, 1.0),
                bbox=(x, y, w, h),
                area=int(area),
                severity=self.classify_severity("edge_anomaly", int(area)),
                timestamp=timestamp
            )
            defects.append(defect)
        
        return defects


class ResultOutput:
    """结果输出管理器"""
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get('directory', './output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.original_dir = self.output_dir / 'original'
        self.annotated_dir = self.output_dir / 'annotated'
        self.roi_dir = self.output_dir / 'defect_rois'
        self.report_dir = self.output_dir / 'reports'
        
        for d in [self.original_dir, self.annotated_dir, self.roi_dir, self.report_dir]:
            d.mkdir(exist_ok=True)
        
        # CSV统计文件
        self.csv_file = self.report_dir / 'statistics.csv'
        self.init_csv()
        
        # 数据库
        self.db_enabled = config.get('save_to_database', False)
        if self.db_enabled:
            self.init_database()
    
    def init_csv(self):
        """初始化CSV文件"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'image_name', 'total_defects', 
                    'severe_defects', 'processing_time'
                ])
    
    def init_database(self):
        """初始化数据库"""
        db_config = self.config.get('database', {})
        db_type = db_config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            db_path = db_config.get('sqlite_path', './defects.db')
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            
            # 创建表
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS defects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    image_path TEXT,
                    defect_type TEXT,
                    confidence REAL,
                    x INTEGER,
                    y INTEGER,
                    width INTEGER,
                    height INTEGER,
                    area INTEGER,
                    severity TEXT
                )
            ''')
            self.conn.commit()
            logger.info("Database initialized")
    
    def save_results(self, image: np.ndarray, result: DetectionResult):
        """保存检测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(result.image_path).stem
        
        # 保存原始图像
        if self.config.get('save_original', True):
            original_path = self.original_dir / f"{image_name}_{timestamp}.jpg"
            cv2.imwrite(str(original_path), image)
        
        # 保存标注图像
        if self.config.get('save_annotated', True):
            annotated = self.draw_annotations(image, result.defects)
            annotated_path = self.annotated_dir / f"{image_name}_{timestamp}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated)
        
        # 保存缺陷ROI
        if self.config.get('save_defect_rois', True):
            for defect in result.defects:
                x, y, w, h = defect.bbox
                roi = image[y:y+h, x:x+w]
                roi_path = self.roi_dir / f"{image_name}_defect{defect.id}_{timestamp}.jpg"
                cv2.imwrite(str(roi_path), roi)
        
        # 保存JSON报告
        if self.config.get('generate_json_report', True):
            json_path = self.report_dir / f"{image_name}_{timestamp}_report.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 更新CSV
        if self.config.get('generate_csv_stats', True):
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.timestamp,
                    image_name,
                    result.total_defects,
                    result.severe_defects,
                    result.processing_time
                ])
        
        # 保存到数据库
        if self.db_enabled:
            for defect in result.defects:
                self.cursor.execute('''
                    INSERT INTO defects VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    defect.timestamp,
                    result.image_path,
                    defect.type,
                    defect.confidence,
                    defect.bbox[0], defect.bbox[1],
                    defect.bbox[2], defect.bbox[3],
                    defect.area,
                    defect.severity
                ))
            self.conn.commit()
    
    def draw_annotations(self, image: np.ndarray, defects: List[Defect]) -> np.ndarray:
        """绘制标注"""
        result = image.copy()
        
        for defect in defects:
            x, y, w, h = defect.bbox
            
            # 根据严重程度选择颜色
            color_map = {
                "HIGH": (0, 0, 255),
                "MEDIUM": (0, 165, 255),
                "LOW": (0, 255, 0)
            }
            color = color_map.get(defect.severity, (128, 128, 128))
            
            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label = f"{defect.type} ({defect.severity})"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x, y - text_h - 8), (x + text_w, y), color, -1)
            cv2.putText(result, label, (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 添加统计信息
        info_text = f"Total: {len(defects)} defects"
        cv2.putText(result, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result


class ProductionDefectDetector:
    """生产级缺陷检测系统主类"""
    def __init__(self, config_path: str = "config.yaml"):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.system_config = self.config.get('system', {})
        self.input_config = self.config.get('input', {})
        self.detection_config = self.config.get('detection', {})
        self.output_config = self.config.get('output', {})
        self.display_config = self.config.get('display', {})
        
        # 初始化组件
        self.source = None
        self.detector = None
        self.output = None
        self.running = False
        
        self.setup()
    
    def setup(self):
        """设置系统组件"""
        logger.info("Setting up production defect detector...")
        
        # 创建输入源
        source_type = self.input_config.get('source_type', 'folder')
        if source_type == 'camera':
            self.source = CameraSource(self.input_config.get('camera', {}))
        elif source_type == 'folder':
            self.source = FolderSource(self.input_config.get('folder', {}))
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # 创建检测器
        mode = self.detection_config.get('mode', 'template_match')
        if mode == 'template_match':
            self.detector = TemplateMatchDetector(self.detection_config)
        elif mode == 'rule_based':
            self.detector = RuleBasedDetector(self.detection_config)
        else:
            raise ValueError(f"Unsupported detection mode: {mode}")
        
        # 创建输出管理器
        self.output = ResultOutput(self.output_config)
        
        logger.info("Setup complete")
    
    def run(self):
        """运行检测系统"""
        logger.info("Starting production defect detector...")
        self.running = True
        self.source.start()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # 获取图像
                frame = self.source.get_frame()
                if frame is None:
                    if isinstance(self.source, FolderSource):
                        logger.info("All images processed")
                        break
                    continue
                
                frame_count += 1
                process_start = time.time()
                
                # 获取图像路径
                image_path = ""
                if isinstance(self.source, FolderSource):
                    image_path = self.source.get_current_path()
                
                # 执行检测
                defects = self.detector.detect(frame)
                processing_time = time.time() - process_start
                
                # 创建结果
                severe_count = sum(1 for d in defects if d.severity == "HIGH")
                result = DetectionResult(
                    image_path=image_path or f"frame_{frame_count}",
                    timestamp=datetime.now().isoformat(),
                    defects=defects,
                    total_defects=len(defects),
                    severe_defects=severe_count,
                    processing_time=processing_time
                )
                
                # 保存结果
                self.output.save_results(frame, result)
                
                # 显示结果
                if self.display_config.get('enabled', True):
                    self.display_results(frame, result)
                
                # 打印统计
                if frame_count % 10 == 0:
                    fps = frame_count / (time.time() - start_time)
                    logger.info(f"Processed {frame_count} frames, FPS: {fps:.2f}")
                
                # 检查键盘输入
                if self.display_config.get('enabled', True):
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        logger.info("Paused, press any key to continue...")
                        cv2.waitKey(0)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def display_results(self, image: np.ndarray, result: DetectionResult):
        """显示检测结果"""
        display_img = self.output.draw_annotations(image, result.defects)
        
        # 添加FPS信息
        if self.display_config.get('show_fps', True):
            fps_text = f"Defects: {result.total_defects} | Time: {result.processing_time*1000:.1f}ms"
            cv2.putText(display_img, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 调整显示大小
        window_name = self.display_config.get('window_name', 'Defect Detection')
        display_width = self.display_config.get('width', 1280)
        display_height = self.display_config.get('height', 720)
        
        h, w = display_img.shape[:2]
        scale = min(display_width / w, display_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        display_img = cv2.resize(display_img, (new_w, new_h))
        
        cv2.imshow(window_name, display_img)
    
    def stop(self):
        """停止检测系统"""
        logger.info("Stopping production defect detector...")
        self.running = False
        if self.source:
            self.source.stop()
            self.source.release()
        cv2.destroyAllWindows()
        logger.info("Stopped")


def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Defect Detector')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', '-i', help='Override input path')
    parser.add_argument('--template', '-t', help='Override template path')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please create config.yaml or specify a valid config file")
        return 1
    
    # 创建并运行检测器
    try:
        detector = ProductionDefectDetector(args.config)
        detector.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
