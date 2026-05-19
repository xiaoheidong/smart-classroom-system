"""
检查当前 Python 环境中的 PyTorch 是否为 CUDA 构建、能否使用 GPU。
用法: python scripts/check_pytorch_cuda.py
"""
import os
import sys


def main() -> None:
    print("=" * 60)
    print("PyTorch / CUDA 诊断")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    try:
        import torch
    except ImportError as e:
        print("未安装 torch:", e)
        sys.exit(1)

    print(f"torch 版本: {torch.__version__}")
    print(f"torch.version.cuda (编译时 CUDA 版本): {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.get_device_name(0)}")

    torch_dir = os.path.dirname(torch.__file__)
    lib = os.path.join(torch_dir, "lib")
    has_cuda_dll = False
    if os.path.isdir(lib):
        for name in os.listdir(lib):
            lower = name.lower()
            if "cuda" in lower or "cudnn" in lower:
                has_cuda_dll = True
                break
    print(f"torch 安装路径: {torch_dir}")
    print(f"lib 目录下是否存在 cuda/cudnn 相关库: {has_cuda_dll}")

    if "+cpu" in torch.__version__ or torch.version.cuda is None:
        print()
        print("结论: 当前是 **CPU 版** PyTorch，无法使用 GPU。")
        print("若本机有 NVIDIA 显卡，请安装 CUDA 版 wheel，例如:")
        print(
            "  python -m pip install --force-reinstall torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        print()
        print("结论: 当前为 **含 CUDA 的** PyTorch 构建。")
    print("=" * 60)


if __name__ == "__main__":
    main()
