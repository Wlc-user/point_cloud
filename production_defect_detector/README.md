# 工业缺陷检测系统 - 生产版本

## 系统概述

这是一个**可直接部署到生产线**的工业缺陷检测系统，支持多种输入源、检测模式和输出方式。

## 功能特点

### ✅ 输入源支持
- **USB工业相机**: 实时视频流检测
- **文件夹批量处理**: 自动检测文件夹内所有图片
- **RTSP网络流**: 支持网络摄像头

### ✅ 检测模式
- **模板匹配**: 与标准模板对比检测差异
- **基于规则**: 使用图像处理算法检测
- **深度学习**: 支持ONNX模型（预留接口）

### ✅ 输出方式
- **图像标注**: 自动绘制缺陷边界框
- **JSON报告**: 详细的检测结果数据
- **CSV统计**: 批量统计信息
- **数据库存储**: SQLite/MySQL/PostgreSQL
- **缺陷ROI**: 单独保存每个缺陷区域

### ✅ 报警系统
- **声音报警**: 检测到缺陷时播放提示音
- **邮件通知**: 发送邮件给管理人员
- **HTTP回调**: 调用外部API接口
- **GPIO控制**: 控制物理设备（如停止生产线）

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml
```

### 2. 准备数据

```
production_defect_detector/
├── input_images/          # 放入待检测图片
├── templates/             # 放入模板图片
│   └── template.jpg      # 标准无缺陷图像
└── output/               # 检测结果输出
```

### 3. 运行系统

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
python detector.py
```

### 4. 查看结果

检测结果保存在 `output/` 目录：
- `original/` - 原始图像
- `annotated/` - 标注后的图像
- `defect_rois/` - 缺陷区域裁剪
- `reports/` - JSON报告和CSV统计

## 配置说明

编辑 `config.yaml` 配置文件：

### 输入配置
```yaml
input:
  source_type: "folder"  # camera / folder / rtsp
  folder:
    path: "./input_images"
    extensions: [".jpg", ".png"]
```

### 检测配置
```yaml
detection:
  product_type: "pcb"  # pcb / metal / textile / general
  mode: "template_match"  # template_match / rule_based / deep_learning
  template_match:
    template_path: "./templates/template.jpg"
    threshold: 30
```

### 输出配置
```yaml
output:
  save_original: true
  save_annotated: true
  save_defect_rois: true
  generate_json_report: true
  generate_csv_stats: true
```

## 使用示例

### 示例1: 文件夹批量检测

1. 将待检测图片放入 `input_images/` 文件夹
2. 运行 `run.bat`
3. 系统自动处理所有图片并保存结果

### 示例2: USB相机实时检测

修改 `config.yaml`:
```yaml
input:
  source_type: "camera"
  camera:
    id: 0
    width: 1920
    height: 1080
```

### 示例3: 自定义检测参数

```yaml
detection:
  mode: "rule_based"
  rule_based:
    edge_detection:
      method: "canny"
      threshold1: 50
      threshold2: 150
```

## 快捷键

| 按键 | 功能 |
|------|------|
| `Q` | 退出程序 |
| `空格` | 暂停/继续 |
| `S` | 保存当前帧 |
| `N` | 处理下一张（文件夹模式）|

## 数据格式

### JSON报告示例
```json
{
  "image_path": "input_images/product_001.jpg",
  "timestamp": "2024-01-15T10:30:00",
  "total_defects": 3,
  "severe_defects": 1,
  "processing_time": 0.125,
  "defects": [
    {
      "id": 0,
      "type": "scratch",
      "confidence": 0.85,
      "bbox": [100, 200, 50, 30],
      "area": 1500,
      "severity": "HIGH"
    }
  ]
}
```

### CSV统计格式
```csv
timestamp,image_name,total_defects,severe_defects,processing_time
2024-01-15T10:30:00,product_001,3,1,0.125
```

## 性能优化

### GPU加速
```yaml
performance:
  use_gpu: true
```

### 多线程处理
```yaml
performance:
  num_workers: 4
```

### 跳帧处理（实时视频）
```yaml
performance:
  skip_frames: 2  # 每3帧处理1帧
```

## 集成到生产线

### 1. 物理报警
配置GPIO输出控制报警灯或蜂鸣器：
```yaml
alert:
  methods:
    gpio:
      enabled: true
      pin: 18
```

### 2. 与MES系统集成
使用HTTP回调将结果发送到MES系统：
```yaml
alert:
  methods:
    webhook:
      enabled: true
      url: "http://mes.company.com/api/defects"
```

### 3. 数据库存储
启用数据库存储便于追溯：
```yaml
output:
  save_to_database: true
  database:
    type: "mysql"
    host: "192.168.1.100"
    username: "detector"
    password: "password"
    database: "defect_detection"
```

## 故障排除

### 问题1: 无法打开相机
- 检查相机连接
- 确认相机ID正确
- 检查是否有其他程序占用相机

### 问题2: 检测效果不佳
- 调整 `threshold` 参数
- 检查模板图像质量
- 确保光照条件一致

### 问题3: 程序运行缓慢
- 降低图像分辨率
- 启用GPU加速
- 减少保存选项

## 技术支持

- OpenCV文档: https://docs.opencv.org/
- 项目GitHub: [Your Repository]
- 问题反馈: [Your Email]

## 许可证

MIT License

---

**注意**: 这是一个生产就绪的系统，可以直接部署到工业环境中使用！
