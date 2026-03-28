@echo off
chcp 65001 >nul
title PCB缺陷检测系统 - 生产环境
color 0A

echo ==========================================
echo    PCB缺陷检测系统 - 生产环境
echo ==========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请安装Python 3.8+
    pause
    exit /b 1
)

REM 检查模型文件
if not exist "pcb_defect_classifier.h5" (
    echo [错误] 模型文件不存在: pcb_defect_classifier.h5
    echo.
    echo 请确保以下文件存在于当前目录:
    echo   - pcb_defect_classifier.h5
    echo   - class_names.json
    pause
    exit /b 1
)

if not exist "class_names.json" (
    echo [错误] 类别标签文件不存在: class_names.json
    pause
    exit /b 1
)

REM 创建日志目录
if not exist logs mkdir logs

echo [信息] 系统检查通过！
echo [信息] 正在启动检测系统...
echo.
echo 操作说明:
echo   - 按 Q 键退出程序
echo   - 按 S 键查看统计信息
echo   - 检测到缺陷时会发出声音报警
echo.
echo ==========================================
echo.

REM 启动检测器
python production_deploy\src\production_detector_windows.py

echo.
echo ==========================================
echo [信息] 检测系统已停止
echo [信息] 日志文件: logs\detector.log
echo ==========================================
pause
