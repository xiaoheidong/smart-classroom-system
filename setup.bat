@echo off
chcp 65001 >nul
echo ==========================================
echo 智慧教室系统 - 安装脚本
echo ==========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/5] Python版本:
python --version
echo.

REM 创建虚拟环境
if not exist "venv" (
    echo [2/5] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
) else (
    echo [2/5] 虚拟环境已存在
)

echo [3/5] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [4/5] 安装依赖（可能需要几分钟）...
pip install -r requirements.txt
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，尝试单独安装核心依赖...
    pip install torch torchvision ultralytics PyQt5 opencv-python numpy
)

echo [5/5] 初始化数据库...
python scripts\init_database.py --sample

echo.
echo ==========================================
echo 安装完成！
echo ==========================================
echo.
echo 使用方法:
echo   1. 运行 start.bat 启动系统
echo   2. 或手动运行: venv\Scripts\python main.py
echo.
echo 下一步:
echo   - 训练模型: python training\train_action_classifier.py
echo   - 下载模型: python scripts\download_models.py
echo.

pause
