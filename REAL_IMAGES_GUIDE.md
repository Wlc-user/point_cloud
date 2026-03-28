# 真实工业缺陷图片使用指南

## 系统说明

本系统支持使用**真实的工业缺陷图片**进行检测，而不是程序生成的模拟图片。

## 使用方法

### 1. 准备真实图片

将您的真实工业缺陷图片放入对应文件夹：

```
real_images/
├── pcb/          # PCB缺陷图片
├── chip/         # 芯片缺陷图片
└── wafer/        # 晶圆缺陷图片
```

### 2. 支持的图片格式

- `.jpg` / `.jpeg`
- `.png`
- `.bmp`
- `.tiff`

### 3. 运行检测

```bash
python real_image_inspection.py
```

## 获取真实缺陷图片的途径

### 方式1：使用公开数据集（推荐）

以下是知名的工业缺陷公开数据集，可以免费下载使用：

#### PCB缺陷数据集
- **DeepPCB**: https://github.com/tangsanli5201/DeepPCB
- **PKU-Market-PCB**: 北京大学发布的PCB缺陷数据集

#### 表面缺陷数据集
- **NEU Surface Defect Database**: 热轧钢表面缺陷
  - 下载: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm
- **DAGM 2007**: 弱监督学习缺陷检测数据集
  - 下载: https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection

#### 通用工业缺陷
- **MVTec AD**: 工业异常检测数据集（需申请）
  - https://www.mvtec.com/company/research/datasets/mvtec-ad
- **KolektorSDD**: 电子换向器表面缺陷
  - https://www.vicos.si/Downloads/KolektorSDD

### 方式2：使用自己的图片

如果您有实际的工业检测图片，直接复制到对应文件夹即可。

### 方式3：从学术论文下载

许多学术论文会提供数据集下载链接，例如：
- IEEE Xplore
- ScienceDirect
- arXiv (计算机视觉论文)

## 注意事项

### 版权问题
- 商业用途请确保图片有合法授权
- 公开数据集通常有特定的使用协议，请遵守
- 建议优先使用公开数据集或自己的数据

### 图片质量建议
- 分辨率：建议不低于 640x480
- 格式：JPG或PNG
- 清晰度：避免过度模糊
- 光照：尽量均匀光照

## 检测结果

运行后会生成以下文件：

```
inspection_results/
├── *_defects.jpg      # 缺陷检测结果图
├── *_measure.jpg      # 尺寸测量结果图
└── summary_report_*.json  # JSON格式检测报告
```

## 示例工作流程

1. **下载公开数据集**（例如 DeepPCB）
2. **解压并将图片放入对应文件夹**
3. **运行检测程序**
4. **查看检测结果**

## 推荐的免费数据集下载

### 1. DeepPCB (PCB缺陷)
```bash
# 使用git下载
git clone https://github.com/tangsanli5201/DeepPCB.git

# 将图片复制到我们的文件夹
copy DeepPCB\Images\* real_images\pcb\
```

### 2. NEU Surface Defect (钢材表面)
```bash
# 手动下载后解压
# 将图片放入 real_images\steel\ (需要创建新类别)
```

## 联系与支持

如需帮助获取数据集或有其他问题，请参考：
- OpenCV官方文档: https://docs.opencv.org/
- 工业视觉论坛: 各种技术社区

---

**重要提示**: 本系统框架已准备就绪，只需添加真实图片即可进行专业的工业缺陷检测！
