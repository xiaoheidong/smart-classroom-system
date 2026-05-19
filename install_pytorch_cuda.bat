@echo off
chcp 65001 >nul
echo ==========================================
echo 安装带 CUDA 的 PyTorch（cu121，适用于 RTX 等 NVIDIA 显卡）
echo 会覆盖当前环境中的 CPU 版 torch / torchvision
echo ==========================================
echo.

set PYTHON=D:\APP\PyCharm\Anaconda3\python.exe
if not exist "%PYTHON%" (
    echo 未找到 %PYTHON%，请用记事本编辑本 bat，把 PYTHON 改成你的 python.exe 路径
    pause
    exit /b 1
)

echo 使用: %PYTHON%
echo.
"%PYTHON%" scripts\check_pytorch_cuda.py
echo.
echo [即将执行] pip 从 PyTorch 官方 cu121 源强制重装...
pause

"%PYTHON%" -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo 安装失败，可尝试 cu118: .../whl/cu118
    pause
    exit /b 1
)

echo.
"%PYTHON%" scripts\check_pytorch_cuda.py
echo.
echo 完成后请重新运行 main.py 或 run_anaconda.bat
pause
