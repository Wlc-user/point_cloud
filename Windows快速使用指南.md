# Windows环境快速使用指南

## 🚀 快速启动（推荐）

### 方式1: 双击启动（最简单）
```
双击运行: 启动生产检测系统.bat
```

### 方式2: 命令行启动
```bash
cd e:\pyspace\opencv
python production_deploy\src\production_detector_windows.py
```

---

## 📋 文件说明

### 启动脚本
| 文件名 | 功能 |
|--------|------|
| `启动生产检测系统.bat` | 一键启动检测系统 |
| `查看日志.bat` | 查看检测日志 |

### 核心文件
| 文件名 | 说明 |
|--------|------|
| `pcb_defect_classifier.h5` | 训练好的模型 |
| `class_names.json` | 类别标签 |
| `production_detector_windows.py` | Windows版检测器 |

---

## 🎮 操作说明

### 运行中按键
| 按键 | 功能 |
|------|------|
| `Q` | 退出程序 |
| `S` | 查看统计信息 |

### 显示信息
- **绿色文字**: 正常/OK
- **红色文字**: 缺陷/DEFECT
- **处理时间**: 每帧检测耗时
- **检测总数**: 累计检测数量

---

## 📊 查看日志

### 方式1: 双击脚本
```
双击运行: 查看日志.bat
```

### 方式2: PowerShell命令
```powershell
# 查看最新100行
Get-Content logs\detector.log -Tail 100

# 实时查看
Get-Content logs\detector.log -Wait -Tail 10

# 查看错误
Select-String -Path logs\detector.log -Pattern "ERROR|DEFECT"
```

---

## 🔧 常见问题

### Q1: 提示找不到模型文件？
**解决**: 确保以下文件在当前目录：
```
pcb_defect_classifier.h5
class_names.json
```

### Q2: 相机无法打开？
**解决**: 
1. 检查相机是否连接
2. 检查是否有其他程序占用相机
3. 修改相机ID（默认是0）

### Q3: 中文显示乱码？
**解决**: 脚本已设置UTF-8编码，如仍乱码请修改PowerShell编码：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 💡 使用流程

```
1. 双击 启动生产检测系统.bat
        ↓
2. 系统自动检查环境和文件
        ↓
3. 加载模型和启动相机
        ↓
4. 开始实时检测
        ↓
5. 按 Q 退出，查看日志分析结果
```

---

## 📁 输出文件

检测完成后会在以下目录生成文件：

```
logs/
├── detector.log              # 检测日志
└── detection_results/        # 检测结果图像
    ├── open_circuit_20240328_120000_0.95.jpg
    ├── short_20240328_120001_0.87.jpg
    └── ...
```

---

## 🎯 生产部署建议

### 1. 创建桌面快捷方式
```
右键 启动生产检测系统.bat → 发送到 → 桌面快捷方式
```

### 2. 设置开机自启
```
Win+R → shell:startup
将快捷方式复制到启动文件夹
```

### 3. 配置文件调整
编辑 `production_deploy\config\production.yaml`:
```yaml
model:
  confidence_threshold: 0.7  # 根据现场调整
  
input:
  camera:
    id: 0                    # 相机ID
    width: 1920             # 分辨率
    height: 1080
```

---

## 📞 技术支持

遇到问题时的排查步骤：
1. 查看日志文件 `logs\detector.log`
2. 检查模型文件是否存在
3. 测试相机是否正常工作
4. 检查Python环境

---

**现在可以双击 `启动生产检测系统.bat` 开始使用了！** 🎉
