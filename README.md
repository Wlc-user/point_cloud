<img width="658" height="646" alt="image" src="https://github.com/user-attachments/assets/0adcb13a-31e5-45c6-aa29-76fce4bcae9f" /><img width="1031" height="666" alt="image" src="https://github.com/user-attachments/assets/4248378d-c0a9-4eaf-a120-715918f0e0cf" /># 工业机器视觉算法开发平台
<img width="1815" height="904" alt="image" src="https://github.com/user-attachments/assets/b748daaf-df73-4dd9-83c6-7ecc28bf6018" />
<img width="1865" height="979" alt="image" src="https://github.com/user-attachments/assets/830ecc27-2ad1-4330-b5e5-7caf8d5db6cb" />
<img width="1883" height="1019" alt="image" src="https://github.com/user-attachments/assets/54b65427-cb8a-433c-b1e2-f6ec54ea6435" />
<img width="1667" height="752" alt="image" src="https://github.com/user-attachments/assets/26dcd610-955a-4236-82c7-6bf3ca66bb63" />

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个功能完善的工业机器视觉算法开发平台，提供从图像采集到3D点云处理的完整视觉解决方案。

## 功能特性

### 核心模块

| 模块 | 功能 | 状态 |
|------|------|------|
| 图像采集 | 相机连接、标定、图像捕捉 | ✅ |
| 图像预处理 | 滤波、边缘检测、形态学操作 | ✅ |
| 图像分割 | 分水岭、GrabCut、聚类分割 | ✅ |
| 特征提取 | ORB/SIFT、HOG、LBP、形状特征 | ✅ |
| 图像匹配 | 模板匹配、特征匹配、目标追踪 | ✅ |
| 测量模块 | 尺寸测量、孔位检测、螺纹检测 | ✅ |
| 缺陷检测 | 表面缺陷、划痕、污渍检测 | ✅ |
| 光流分析 | 运动检测、速度估计 | ✅ |
| 3D点云 | 滤波、配准、分割、聚类 | ✅ |
| 深度学习 | 模型训练、管理、部署 | ✅ |

## 安装

### 环境要求
- Python 3.8+
- Windows/Linux/macOS

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖库
- OpenCV 4.8.0
- NumPy 1.24.3
- TensorFlow 2.12.0
- PyQt5 5.15.9
- scikit-learn 1.3.0
- scikit-image 0.21.0
- scipy 1.11.1
- matplotlib 3.7.1

## 快速开始

### 1. 图像采集

```python
from src_core.image_capture import ImageCapture

capture = ImageCapture()
capture.open_camera(0)
image = capture.capture_frame()
capture.close_camera()
```

### 2. 图像处理

```python
from src_core.image_processing import ImageProcessor

processor = ImageProcessor()
filtered = processor.apply_filter(image, 'gaussian', kernel_size=(5, 5))
edges = processor.detect_edges(image, 'canny')
```

### 3. 特征提取

```python
from src_core.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
keypoints, descriptors = extractor.detect_and_compute(image, method='orb')
```

### 4. 完整流程

```python
from examples.complete_pipeline_example import CompleteVisionPipeline

pipeline = CompleteVisionPipeline()
results = pipeline.run_complete_pipeline(
    image_source='file',
    image_path='test.jpg'
)
```

## 示例程序

### 光流分析
```bash
python examples/optical_flow_example.py
```

### 缺陷检测
```bash
python examples/defect_detection_example.py
```

### 点云处理
```bash
python examples/point_cloud_example.py --mode filtering
```

### 完整流程
```bash
python examples/complete_pipeline_example.py --image test.jpg
```

## 项目结构

```
opencv/
├── src/
│   ├── core/              # 核心算法模块
│   │   ├── image_capture.py
│   │   ├── image_processing.py
│   │   ├── image_segmentation.py
│   │   ├── feature_extraction.py
│   │   ├── image_matching.py
│   │   ├── measurement.py
│   │   ├── defect_detection.py
│   │   ├── optical_flow.py
│   │   └── analysis.py
│   ├── 3d_vision/         # 3D视觉模块
│   │   ├── point_cloud.py
│   │   └── point_cloud_visualizer.py
│   ├── deep_learning/     # 深度学习模块
│   │   ├── model_trainer.py
│   │   └── model_manager.py
│   ├── deployment/        # 部署模块
│   │   └── model_deployer.py
│   └── ui/                # 用户界面
│       └── main_window.py
├── examples/              # 示例程序
│   ├── optical_flow_example.py
│   ├── defect_detection_example.py
│   ├── point_cloud_example.py
│   └── complete_pipeline_example.py
├── requirements.txt       # 依赖列表
├── README.md             # 项目说明
└── PROJECT_RESUME.md     # 项目简历
```

## 应用场景

### 电子制造
- PCB缺陷检测
- SMT元件定位
- 焊点质量检测

### 汽车工业
- 车身尺寸测量
- 焊缝质量检测
- 零部件装配验证

### 食品包装
- 包装完整性检测
- 标签印刷质量
- 条码/二维码识别

### 3D视觉
- 产品三维重建
- 体积测量
- 机器人抓取定位

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面层 (UI)                         │
├─────────────────────────────────────────────────────────────┤
│                      应用逻辑层                              │
│  图像采集 → 预处理 → 分割 → 特征 → 匹配 → 测量 → 3D        │
├─────────────────────────────────────────────────────────────┤
│                      算法核心层                              │
│  OpenCV + NumPy + scikit-image + TensorFlow                 │
├─────────────────────────────────────────────────────────────┤
│                      硬件接口层                              │
│  工业相机 + 深度相机 + 光源控制 + PLC通信                    │
└─────────────────────────────────────────────────────────────┘
```

## 性能指标

- **图像处理速度**：实时（30+ FPS）
- **测量精度**：亚像素级（0.1像素）
- **点云处理**：支持 100万+ 点云
- **检测准确率**：> 95%（取决于应用场景）

## 开发计划

### 短期目标
- [ ] 完善GUI界面
- [ ] 增加更多工业相机支持
- [ ] 优化算法性能
- [ ] 完善文档和示例

### 长期目标
- [ ] 商业化产品发布
- [ ] 行业解决方案定制
- [ ] 云服务集成
- [ ] AI模型市场

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request
<img width="658" height="646" alt="image" src="https://github.com/user-attachments/assets/19d3cb60-a85a-44cb-8e29-1f6b4728dcc8" />
<img width="1031" height="666" alt="image" src="https://github.com/user-attachments/assets/28ed9077-ee56-4d81-96ec-c872b00b8679" />
<img width="1034" height="704" alt="image" src="https://github.com/user-attachments/assets/e27ff1fe-3794-489c-8a2c-47c9b7f1299f" />
<img width="1032" height="684" alt="image" src="https://github.com/user-attachments/assets/f69c18e3-2009-4240-bcd6-fa809e248e65" />
<img width="1066" height="706" alt="image" src="https://github.com/user-attachments/assets/961e81bf-11db-4ded-a250-23ece13df89a" />
<img width="721" height="684" alt="image" src="https://github.com/user-attachments/assets/4585f346-1182-40a4-b6ab-0c7dba8fbbef" />
<img width="1042" height="681" alt="image" src="https://github.com/user-attachments/assets/fb2590c8-b21c-4148-bf8a-6027f6cfdd6c" />
<img width="911" height="674" alt="image" src="https://github.com/user-attachments/assets/5606c9e3-f6ca-4cde-a2f0-61a1f642ff6f" />

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目地址：[GitHub Repository](https://github.com/yourusername/opencv)
- 邮箱：your.email@example.com

## 致谢

- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [scikit-image](https://scikit-image.org/)

---

**关键词**：机器视觉、计算机视觉、OpenCV、深度学习、工业检测、图像处理、3D点云、Python
