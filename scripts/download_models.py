"""
模型下载脚本
自动下载所需的预训练模型

使用方法:
    python scripts/download_models.py
"""
import os
import sys
import urllib.request
import bz2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PRETRAINED_DIR


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, desc=None):
    """下载文件"""
    if os.path.exists(output_path):
        print(f"文件已存在: {output_path}")
        return True
    
    print(f"下载: {desc or url}")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
        print(f"  [OK] 下载完成: {output_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] 下载失败: {e}")
        return False


def decompress_bz2(input_path, output_path):
    """解压bz2文件"""
    if os.path.exists(output_path):
        print(f"解压文件已存在: {output_path}")
        return True
    
    print(f"解压: {input_path}")
    
    try:
        with bz2.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"  [OK] 解压完成")
        return True
    except Exception as e:
        print(f"  [FAIL] 解压失败: {e}")
        return False


def download_dlib_models():
    """下载dlib模型"""
    models = [
        {
            'name': 'dlib_face_recognition_resnet_model_v1.dat',
            'url': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
            'compressed': True
        },
        {
            'name': 'shape_predictor_68_face_landmarks.dat',
            'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            'compressed': True
        }
    ]
    
    print("=" * 60)
    print("下载dlib模型")
    print("=" * 60)
    
    for model in models:
        output_path = os.path.join(PRETRAINED_DIR, model['name'])
        
        if model.get('compressed'):
            # 下载压缩文件
            compressed_path = output_path + '.bz2'
            
            if download_file(model['url'], compressed_path, model['name']):
                # 解压
                decompress_bz2(compressed_path, output_path)
                # 删除压缩文件
                os.remove(compressed_path)
        else:
            download_file(model['url'], output_path, model['name'])


def download_yolo_models():
    """下载YOLO模型（通过ultralytics）"""
    print("\n" + "=" * 60)
    print("下载YOLO模型")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        
        models = ['yolov11n.pt']
        
        for model_name in models:
            print(f"下载: {model_name}")
            try:
                model = YOLO(model_name)
                print(f"  [OK] {model_name} 下载/加载成功")
            except Exception as e:
                print(f"  [FAIL] 失败: {e}")
    except ImportError:
        print("请先安装ultralytics: pip install ultralytics")


def print_facenet_note():
    """免 dlib 人脸方案说明（权重首次运行时由 facenet-pytorch 自动下载）"""
    print("\n" + "=" * 60)
    print("人脸特征（facenet-pytorch，可选）")
    print("=" * 60)
    print("""
若未安装 dlib，系统使用 facenet-pytorch（MTCNN + InceptionResnetV1）。
请先执行: pip install facenet-pytorch
首次运行人脸注册/点名时，会自动下载 VGGFace2 预训练权重（需联网）。
也可提前在 Python 中执行一次以触发下载:
    from facenet_pytorch import InceptionResnetV1
    InceptionResnetV1(pretrained='vggface2')
    """)


def download_liveness_model():
    """下载活体检测模型"""
    print("\n" + "=" * 60)
    print("下载活体检测模型")
    print("=" * 60)
    
    print("""
MiniFASNet活体检测模型需要从以下仓库下载:
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

下载后请将模型文件放入: models/pretrained/
需要的文件:
    - 2.7_80x80_MiniFASNetV2.pth

或者使用备用方案（简单纹理分析），无需下载此模型。
    """)


def main():
    """主函数"""
    print("=" * 60)
    print("智慧教室系统 - 模型下载工具")
    print("=" * 60)
    
    # 创建模型目录
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    
    # 下载各模型
    download_dlib_models()
    download_yolo_models()
    print_facenet_note()
    download_liveness_model()
    
    print("\n" + "=" * 60)
    print("下载完成")
    print("=" * 60)
    print(f"\n模型保存在: {PRETRAINED_DIR}")
    print("\n接下来请运行:")
    print("  1. python training/train_action_classifier.py  (训练行为分类模型)")
    print("  2. python main.py  (启动系统)")


if __name__ == '__main__':
    main()
