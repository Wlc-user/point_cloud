@echo off
chcp 65001 >nul
title 查看检测日志
color 0B

echo ==========================================
echo    查看检测日志
echo ==========================================
echo.

if not exist logs\detector.log (
    echo [警告] 日志文件不存在，请先运行检测系统
echo.
    pause
    exit /b 1
)

echo [信息] 正在显示最新100行日志...
echo.
echo ==========================================
echo.

REM 使用PowerShell显示日志（支持中文）
powershell -Command "Get-Content logs\detector.log -Tail 100 -Encoding UTF8"

echo.
echo ==========================================
echo.
echo 操作选项:
echo   1. 继续查看新日志（实时）
echo   2. 查看错误日志
echo   3. 查看统计信息
echo   4. 退出
echo.
set /p choice="请选择 (1-4): "

if "%choice%"=="1" (
    echo.
    echo [信息] 正在实时监控日志，按 Ctrl+C 退出...
    echo.
    powershell -Command "Get-Content logs\detector.log -Wait -Tail 10 -Encoding UTF8"
) else if "%choice%"=="2" (
    echo.
    echo [信息] 错误日志:
    echo.
    powershell -Command "Select-String -Path logs\detector.log -Pattern 'ERROR|错误|DEFECT' -Encoding UTF8 | Select-Object -Last 20"
    pause
) else if "%choice%"=="3" (
    echo.
    echo [信息] 统计信息:
    echo.
    powershell -Command "Select-String -Path logs\detector.log -Pattern '检测总数|缺陷数量|缺陷率' -Encoding UTF8 | Select-Object -Last 10"
    pause
)

echo.
echo ==========================================
