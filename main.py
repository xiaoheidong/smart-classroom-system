"""
智慧教室系统 - 主程序入口
基于深度学习的课堂行为分析与考勤系统

运行方式:
    python main.py

环境要求:
    - Python 3.8+
    - PyTorch 2.0+ (CUDA 11.8)
    - RTX 3050 4GB显存或更高
"""
import os
import sys
import warnings

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 忽略不必要的警告
warnings.filterwarnings('ignore')


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("智慧教室系统 - 环境检查")
    print("=" * 60)
    
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch和CUDA
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("警告: PyTorch未安装")
        return False
    
    # 检查其他依赖
    required_packages = [
        ('cv2', 'opencv-python', True),
        ('ultralytics', 'ultralytics', True),
        ('dlib', 'dlib', False),
        ('PyQt5', 'PyQt5', True),
        ('numpy', 'numpy', True),
    ]
    
    print("\n依赖检查:")
    optional_missing = []
    for module, package, required in required_packages:
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            if required:
                print(f"  [MISSING] {package} (未安装)")
                return False
            print(f"  [OPTIONAL] {package} (未安装，相关功能将不可用)")
            optional_missing.append(package)
    
    if optional_missing:
        print("\n可选依赖缺失，不影响主界面启动:")
        for package in optional_missing:
            print(f"  - {package}")

    print("=" * 60)
    return True


def create_directories():
    """创建必要的目录结构"""
    from config import (
        MODELS_DIR, PRETRAINED_DIR, TRAINED_DIR,
        DATA_DIR, VIDEO_DIR, IMAGE_DIR, DB_DIR
    )
    
    dirs = [MODELS_DIR, PRETRAINED_DIR, TRAINED_DIR, 
            DATA_DIR, VIDEO_DIR, IMAGE_DIR, DB_DIR]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("目录结构检查完成")


def main():
    """主函数"""
    # 环境检查
    if not check_environment():
        print("\n环境检查失败，请安装缺失的依赖:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # 创建目录
    create_directories()
    
    # 导入并运行主窗口
    try:
        from PyQt5.QtWidgets import QApplication
        from gui.main_window import MainWindow
        
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # 设置应用字体
        font = app.font()
        font.setPointSize(10)
        app.setFont(font)
        
        print("\n启动主窗口...")
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
