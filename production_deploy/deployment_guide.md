# PCB缺陷检测模型 - 上线部署与现场调试指南

## 目录
1. [部署架构](#部署架构)
2. [环境准备](#环境准备)
3. [部署步骤](#部署步骤)
4. [现场调试](#现场调试)
5. [监控与维护](#监控与维护)
6. [故障排查](#故障排查)

---

## 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                        生产现场                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ 工业相机    │───→│  检测服务器  │───→│  MES系统    │     │
│  │  (USB/GigE) │    │  (GPU/CPU)  │    │  (数据库)   │     │
│  └─────────────┘    └──────┬──────┘    └─────────────┘     │
│                            │                                │
│                            ↓                                │
│                     ┌─────────────┐                         │
│                     │  报警系统   │                         │
│                     │ (声光/IO)  │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 环境准备

### 1. 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| 内存 | 8 GB | 16 GB |
| 存储 | 128 GB SSD | 256 GB SSD |
| GPU | 可选 (GTX 1060) | RTX 3060 |
| 相机 | USB 3.0 工业相机 | GigE 工业相机 |

### 2. 软件环境

```bash
# 安装Python 3.8-3.10
python --version

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install tensorflow==2.12.0 opencv-python numpy pyyaml
```

### 3. 部署文件清单

```
deployment/
├── models/
│   ├── pcb_defect_classifier.h5    # 训练好的模型
│   └── class_names.json            # 类别标签
├── config/
│   └── production.yaml             # 生产环境配置
├── src/
│   ├── detector.py                 # 核心检测模块
│   ├── camera_interface.py         # 相机接口
│   └── alarm_system.py             # 报警系统
├── logs/                           # 日志目录
├── scripts/
│   ├── start.sh                    # 启动脚本
│   ├── stop.sh                     # 停止脚本
│   └── health_check.sh             # 健康检查
└── README.md                       # 部署文档
```

---

## 部署步骤

### Step 1: 准备部署包

```bash
# 创建部署目录
mkdir -p production_deploy/models
mkdir -p production_deploy/config
mkdir -p production_deploy/src
mkdir -p production_deploy/logs
mkdir -p production_deploy/scripts

# 复制模型文件
cp pcb_defect_classifier.h5 production_deploy/models/
cp class_names.json production_deploy/models/

# 复制源代码
cp use_trained_model.py production_deploy/src/
```

### Step 2: 配置生产环境

创建 `production_deploy/config/production.yaml`:

```yaml
system:
  name: "PCB Defect Detection System"
  version: "1.0.0"
  debug: false

# 输入配置
input:
  source_type: "camera"
  camera:
    id: 0
    width: 1920
    height: 1080
    fps: 30
    exposure: -1  # 自动曝光

# 模型配置
model:
  path: "models/pcb_defect_classifier.h5"
  class_names_path: "models/class_names.json"
  confidence_threshold: 0.7
  input_size: [224, 224]

# 输出配置
output:
  save_images: true
  save_path: "logs/detection_results"
  log_level: "INFO"

# 报警配置
alarm:
  enabled: true
  sound:
    enabled: true
    file: "sounds/alarm.wav"
  gpio:
    enabled: false
    pin: 18
    active_low: false

# 性能配置
performance:
  use_gpu: true
  batch_size: 1
  skip_frames: 0
```

### Step 3: 创建生产检测脚本

创建 `production_deploy/src/production_detector.py`:

```python
"""
生产环境PCB缺陷检测器
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDetector:
    """生产环境检测器"""
    
    def __init__(self, config_path='config/production.yaml'):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        
    def load_model(self):
        """加载模型"""
        model_config = self.config['model']
        logger.info(f"Loading model from: {model_config['path']}")
        
        self.model = keras.models.load_model(model_config['path'])
        
        with open(model_config['class_names_path'], 'r') as f:
            self.class_names = json.load(f)
        
        self.confidence_threshold = model_config['confidence_threshold']
        self.input_size = tuple(model_config['input_size'])
        
        logger.info(f"Model loaded. Classes: {self.class_names}")
        
    def init_camera(self):
        """初始化相机"""
        camera_config = self.config['input']['camera']
        logger.info(f"Initializing camera {camera_config['id']}")
        
        self.cap = cv2.VideoCapture(camera_config['id'])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        logger.info("Camera initialized successfully")
    
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
        
        logger.info(f"Detection: {result['class']} ({result['confidence']:.2%}) "
                   f"Defective: {result['is_defective']}")
    
    def trigger_alarm(self, result):
        """触发报警"""
        if not self.config['alarm']['enabled']:
            return
        
        if result['is_defective']:
            logger.warning(f"DEFECT DETECTED: {result['class']}")
            
            # 声音报警
            if self.config['alarm']['sound']['enabled']:
                self.play_sound()
    
    def play_sound(self):
        """播放报警声音"""
        try:
            import platform
            system = platform.system()
            
            if system == 'Windows':
                import winsound
                winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms
            else:
                os.system('beep')
        except Exception as e:
            logger.error(f"Failed to play sound: {e}")
    
    def print_stats(self):
        """打印统计信息"""
        runtime = datetime.now() - self.stats['start_time']
        hours = runtime.total_seconds() / 3600
        
        defect_rate = (self.stats['total_defects'] / self.stats['total_inspected'] * 100) \
                      if self.stats['total_inspected'] > 0 else 0
        
        logger.info("="*60)
        logger.info("Production Statistics")
        logger.info("="*60)
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Total Inspected: {self.stats['total_inspected']}")
        logger.info(f"Total Defects: {self.stats['total_defects']}")
        logger.info(f"Defect Rate: {defect_rate:.2f}%")
        logger.info(f"Throughput: {self.stats['total_inspected']/hours:.1f} pcs/hour")
        logger.info("="*60)
    
    def run(self):
        """运行检测循环"""
        logger.info("Starting production detector...")
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
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
                
                # 显示结果（调试用）
                display_frame = frame.copy()
                text = f"{result['class']}: {result['confidence']:.2%}"
                color = (0, 0, 255) if result['is_defective'] else (0, 255, 0)
                cv2.putText(display_frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(display_frame, f"Time: {process_time*1000:.1f}ms", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Production Detector', display_frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.print_stats()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """停止检测器"""
        logger.info("Stopping production detector...")
        self.running = False
        self.print_stats()
        
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        logger.info("Detector stopped")


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml',
                       help='Configuration file path')
    args = parser.parse_args()
    
    # 创建日志目录
    Path('logs').mkdir(exist_ok=True)
    
    # 启动检测器
    detector = ProductionDetector(args.config)
    detector.run()


if __name__ == '__main__':
    main()
```

### Step 4: 创建启动脚本

创建 `production_deploy/scripts/start.sh`:

```bash
#!/bin/bash

echo "Starting PCB Defect Detection System..."

# 激活虚拟环境
source ../venv/bin/activate

# 检查模型文件
if [ ! -f "models/pcb_defect_classifier.h5" ]; then
    echo "Error: Model file not found!"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动检测器
nohup python src/production_detector.py --config config/production.yaml > logs/console.log 2>&1 &

# 保存PID
echo $! > detector.pid

echo "Detector started with PID: $(cat detector.pid)"
echo "Logs: logs/detector.log"
```

创建 `production_deploy/scripts/stop.sh`:

```bash
#!/bin/bash

if [ -f "detector.pid" ]; then
    PID=$(cat detector.pid)
    echo "Stopping detector (PID: $PID)..."
    kill $PID
    rm detector.pid
    echo "Detector stopped"
else
    echo "Detector is not running"
fi
```

---

## 现场调试

### 调试检查清单

#### 1. 硬件检查
- [ ] 相机连接正常，图像清晰
- [ ] 光源稳定，无闪烁
- [ ] 网络连接正常（如需要）
- [ ] 报警设备测试正常

#### 2. 软件检查
- [ ] 模型文件完整
- [ ] 配置文件正确
- [ ] 依赖安装完整
- [ ] 日志系统正常

#### 3. 检测参数调优

```python
# 调整置信度阈值
confidence_threshold: 0.7  # 根据现场情况调整

# 如果误检率高，提高阈值
confidence_threshold: 0.8

# 如果漏检率高，降低阈值
confidence_threshold: 0.6
```

### 现场调试步骤

#### Phase 1: 单机测试（1-2天）

```bash
# 1. 测试模型加载
python -c "from production_detector import ProductionDetector; d = ProductionDetector(); print('OK')"

# 2. 测试相机
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read())"

# 3. 单帧测试
python src/production_detector.py
# 按 's' 查看统计信息
# 按 'q' 退出
```

#### Phase 2: 小批量测试（3-5天）

```bash
# 连续运行测试
# 观察24小时，记录：
# - 检测准确率
# - 误检率
# - 系统稳定性
# - 资源占用情况
```

#### Phase 3: 联调测试（1周）

- 与MES系统对接
- 与PLC信号对接
- 报警系统测试
- 操作员培训

### 调试工具

创建 `production_deploy/src/debug_tools.py`:

```python
"""
现场调试工具
"""
import cv2
import numpy as np
import time
from datetime import datetime


class DebugTools:
    """调试工具集"""
    
    @staticmethod
    def test_camera(camera_id=0, duration=10):
        """测试相机"""
        print(f"Testing camera {camera_id} for {duration} seconds...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Failed to open camera!")
            return False
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                cv2.imshow('Camera Test', frame)
                
                # 显示FPS
                fps = frame_count / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Test completed. Captured {frame_count} frames.")
        print(f"Average FPS: {frame_count / duration:.1f}")
        return True
    
    @staticmethod
    def test_model_speed(model_path, iterations=100):
        """测试模型推理速度"""
        import tensorflow as tf
        
        print(f"Testing model speed ({iterations} iterations)...")
        
        model = tf.keras.models.load_model(model_path)
        
        # 生成测试数据
        test_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # 预热
        for _ in range(10):
            model.predict(test_data, verbose=0)
        
        # 测试
        start_time = time.time()
        for _ in range(iterations):
            model.predict(test_data, verbose=0)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / iterations
        
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average inference time: {avg_time*1000:.1f}ms")
        print(f"Throughput: {1/avg_time:.1f} FPS")
    
    @staticmethod
    def capture_samples(camera_id=0, num_samples=10, save_dir='samples'):
        """采集样本图像"""
        import os
        from pathlib import Path
        
        Path(save_dir).mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(camera_id)
        
        print(f"Capturing {num_samples} samples...")
        print("Press SPACE to capture, 'q' to quit")
        
        captured = 0
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            cv2.imshow('Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/sample_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                captured += 1
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Captured {captured} samples")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_tools.py [camera|speed|capture]")
        sys.exit(1)
    
    tool = DebugTools()
    
    if sys.argv[1] == 'camera':
        tool.test_camera()
    elif sys.argv[1] == 'speed':
        tool.test_model_speed('models/pcb_defect_classifier.h5')
    elif sys.argv[1] == 'capture':
        tool.capture_samples()
    else:
        print("Unknown command")
```

---

## 监控与维护

### 日志监控

```bash
# 实时查看日志
tail -f logs/detector.log

# 查看错误日志
grep ERROR logs/detector.log

# 统计检测数量
grep "Detection:" logs/detector.log | wc -l
```

### 性能监控

```python
# 添加到 production_detector.py

import psutil
import GPUtil

def log_system_stats(self):
    """记录系统状态"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    logger.info(f"System Stats - CPU: {cpu_percent}%, "
               f"Memory: {memory.percent}%")
    
    # 如果有GPU
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logger.info(f"GPU {gpu.id}: {gpu.load*100:.1f}% load, "
                       f"{gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
    except:
        pass
```

### 定期维护

```bash
# 每天执行
#!/bin/bash
# daily_maintenance.sh

echo "Daily Maintenance - $(date)"

# 1. 备份日志
tar -czf logs/backup_$(date +%Y%m%d).tar.gz logs/*.log

# 2. 清理旧日志（保留30天）
find logs -name "*.log" -mtime +30 -delete

# 3. 检查磁盘空间
df -h

# 4. 检查模型文件完整性
md5sum models/pcb_defect_classifier.h5 > models/model.md5

echo "Maintenance completed"
```

---

## 故障排查

### 常见问题

#### Q1: 模型加载失败
```bash
# 检查TensorFlow版本
python -c "import tensorflow as tf; print(tf.__version__)"

# 检查模型文件
ls -lh models/pcb_defect_classifier.h5

# 重新保存模型
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/pcb_defect_classifier.h5')
model.save('models/pcb_defect_classifier.h5')
"
```

#### Q2: 相机无法打开
```bash
# 列出可用相机
ls /dev/video*

# 测试相机
python debug_tools.py camera

# 检查权限
sudo chmod 666 /dev/video0
```

#### Q3: 检测速度慢
```bash
# 检查GPU是否可用
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 测试模型速度
python debug_tools.py speed

# 降低相机分辨率
# 修改 config/production.yaml
```

#### Q4: 误检率高
```bash
# 采集现场样本
python debug_tools.py capture

# 分析预测结果
# 调整置信度阈值
# 考虑重新训练模型
```

### 紧急处理

```bash
# 如果系统崩溃，快速重启
./scripts/stop.sh
sleep 2
./scripts/start.sh

# 查看最后100行日志
tail -n 100 logs/detector.log

# 检查系统资源
top
nvidia-smi  # 如果有GPU
```

---

## 联系支持

遇到问题时的排查步骤：
1. 查看日志文件 `logs/detector.log`
2. 运行健康检查脚本
3. 使用调试工具测试
4. 联系技术支持

---

**祝您部署顺利！**
