@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到 venv。请先在本目录执行:
    echo   D:\APP\PyCharm\Anaconda3\python.exe -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    echo   venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)
echo 使用: %CD%\venv\Scripts\python.exe
venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo.
venv\Scripts\python.exe main.py
if errorlevel 1 pause
