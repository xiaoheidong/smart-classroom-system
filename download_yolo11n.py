import os
import urllib.request

os.makedirs('models/pretrained', exist_ok=True)

url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
output = 'models/pretrained/yolo11n.pt'

print(f'正在下载YOLOv11n模型...')
print(f'URL: {url}')

urllib.request.urlretrieve(url, output)

size = os.path.getsize(output) / 1024 / 1024
print(f'下载完成！文件大小: {size:.1f} MB')
print(f'保存路径: {output}')
