@echo off
chcp 65001 >nul
echo ==========================================
echo PCB Defect Detection System - Starting
echo ==========================================
echo.

REM 检查模型文件
if not exist "models\pcb_defect_classifier.h5" (
    echo [错误] 模型文件不存在: models\pcb_defect_classifier.h5
    pause
    exit /b 1
)

REM 创建日志目录
if not exist logs mkdir logs

REM 启动检测器
echo [信息] 正在启动检测器...
python ..\..\use_trained_model.py 1

echo.
echo [信息] 检测器已启动
echo [信息] 日志文件: logs\detector.log
echo.
pause
