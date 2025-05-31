@echo off
chcp 65001 > nul
title 显示器色彩校准数据可视化系统

echo.
echo ========================================
echo   显示器色彩校准数据可视化系统
echo   Monitor Color Calibration GUI
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误：未检测到Python环境
    echo 请确保已安装Python 3.7或更高版本
    echo.
    pause
    exit /b 1
)

echo 🐍 Python环境检测正常
echo.

REM 检查并安装依赖包
echo 📦 正在检查依赖包...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ⚠️  警告：部分依赖包安装可能有问题
    echo 尝试继续运行...
    echo.
) else (
    echo ✅ 依赖包检查完成
    echo.
)

REM 启动GUI应用
echo 🚀 正在启动GUI界面...
echo.
python calibration_gui.py

REM 如果程序异常退出，等待用户确认
if errorlevel 1 (
    echo.
    echo ❌ 程序异常退出
    pause
) 