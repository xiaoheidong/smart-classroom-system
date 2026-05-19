@echo off
chcp 65001 >nul
echo ==========================================
echo 智慧教室系统 - 启动脚本
echo ==========================================
echo.

REM 检查虚拟环境
if not exist "venv\Scripts\python.exe" (
    echo [错误] 虚拟环境不存在，请先创建
    echo 运行: python -m venv venv
    pause
    exit /b 1
)

echo [1/3] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [2/3] 检查依赖...
python -c "import torch, cv2, ultralytics, PyQt5, dlib" 2>nul
if errorlevel 1 (
    echo [警告] 依赖未安装，正在安装...
    pip install -r requirements.txt
)

echo [3/3] 启动系统...
python main.py

if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查错误信息
    pause
)

deactivate
