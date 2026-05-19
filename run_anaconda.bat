@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo 智慧教室系统 - 启动（优先使用项目 venv）
echo ==========================================
echo.

REM 若存在 venv（含 CUDA 版 PyTorch），优先使用，避免与 Anaconda 全局 CPU 版冲突
if exist "venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
    echo [提示] 使用项目虚拟环境 venv（推荐）
) else (
    set "PYTHON=D:\APP\PyCharm\Anaconda3\python.exe"
    echo [提示] 未找到 venv，使用 Anaconda。若 CUDA 不可用，请运行 run_venv.bat 前先创建 venv。
)

echo [1/3] 检查环境...
echo Python路径: %PYTHON%
%PYTHON% --version
echo.

echo [2/3] 检查依赖...
%PYTHON% -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [错误] PyTorch检查失败
    pause
    exit /b 1
)
echo.

echo [3/3] 启动智慧教室系统...
%PYTHON% main.py

if errorlevel 1 (
    echo.
    echo [错误] 启动失败
    pause
)
