@echo off
chcp 65001 >nul
echo ==========================================
echo 智慧教室系统 - Anaconda环境安装
echo ==========================================
echo.

REM Anaconda Python路径
set PYTHON=D:\APP\PyCharm\Anaconda3\python.exe
set PIP=D:\APP\PyCharm\Anaconda3\Scripts\pip.exe

echo Python路径: %PYTHON%
echo.

echo [1/4] 检查PyTorch...
%PYTHON% -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
echo 若显卡为 NVIDIA 但 CUDA可用为 False，说明当前是 CPU 版 torch，请运行项目根目录 install_pytorch_cuda.bat 或: scripts\check_pytorch_cuda.py
echo.

echo [2/4] 安装核心依赖（跳过torch）...
%PIP% install ultralytics opencv-python PyQt5 numpy pandas matplotlib scikit-learn tqdm scipy Pillow
if errorlevel 1 (
    echo [警告] 部分依赖安装失败
)
echo.

echo [3/4] 尝试安装dlib（可选，人脸识别需要）...
%PIP% install dlib
if errorlevel 1 (
    echo [警告] dlib安装失败，将使用备用方案
)
echo.

echo [4/4] 初始化数据库...
%PYTHON% scripts\init_database.py --sample
echo.

echo ==========================================
echo 安装完成！
echo ==========================================
echo.
echo 使用方法:
echo   双击运行 run_anaconda.bat 启动系统
echo.

pause
