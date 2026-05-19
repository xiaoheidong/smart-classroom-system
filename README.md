# 智慧教室系统 - 基于深度学习的课堂行为分析与考勤

## 项目概述

本项目是一个基于深度学习的实时智能教室监控系统，专为毕业设计开发。系统实现了以下核心功能：

- **作弊检测**: 检测传纸条、低头偷看、东张西望等作弊行为
- **动态点名**: 基于人脸识别的实时考勤系统
- **人脸注册**: 静默活体检测 + 人脸采集

## 硬件适配

针对 **RTX 3050 Laptop 4GB显存** 进行了优化：

| 硬件 | 规格 | 说明 |
|------|------|------|
| CPU | i5-11260H | 数据预处理 |
| GPU | RTX 3050 4GB | 模型推理（轻量化模型） |
| 内存 | 16GB+ | 数据加载和缓冲 |

### 优化策略

- 使用YOLOv8n（3.2M参数）替代标准版本
- 使用YOLOv8n-Pose（11MB）进行姿态估计
- 训练batch_size设为4-8
- 每2帧处理一次，降低计算量

## 项目结构

```
smart_classroom2/
├── main.py                     # 主程序入口
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
├── models/                     # 模型文件
│   ├── pretrained/             # 预训练模型
│   │   ├── yolov8n.pt         # YOLOv8 nano
│   │   ├── yolov8n-pose.pt    # YOLOv8 pose
│   │   ├── dlib_face_recognition_resnet_model_v1.dat
│   │   ├── shape_predictor_68_face_landmarks.dat
│   │   └── 2.7_80x80_MiniFASNetV2.pth
│   └── trained/                # 训练后的模型
│       └── action_classifier.pth
├── data/                       # 数据目录
│   ├── videos/                 # 视频文件
│   └── images/                 # 图像文件
├── db/                         # 数据库
│   └── smart_classroom.db
├── src/                        # 源代码
│   ├── config.py              # 配置文件
│   ├── core/                  # 核心算法
│   │   ├── person_detector.py      # 人体检测
│   │   ├── pose_estimator.py       # 姿态估计
│   │   ├── action_classifier.py    # 行为分类
│   │   ├── cheating_detector.py    # 作弊检测
│   │   ├── face_recognition.py     # 人脸识别
│   │   └── liveness_detector.py   # 活体检测
│   ├── gui/                   # GUI界面
│   │   ├── main_window.py          # 主窗口
│   │   ├── cheating_detection_tab.py # 作弊检测页
│   │   ├── attendance_tab.py       # 考勤页
│   │   └── face_registration_tab.py # 人脸注册页
│   └── utils/                 # 工具模块
│       ├── video_capture.py   # 视频捕获
│       ├── visualization.py   # 可视化
│       └── database.py        # 数据库
├── training/                   # 训练脚本
│   └── train_action_classifier.py  # 行为分类训练
└── scripts/                    # 工具脚本
    └── export_models.py       # 模型导出
```

## 安装步骤

### 1. 环境准备

```bash
# 安装Python 3.8+
python --version

# 创建虚拟环境（推荐）
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 如果需要特定版本的PyTorch（CUDA 11.8）
pip install torch==2.0.1+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 下载预训练模型

```bash
# 创建模型目录
mkdir -p models/pretrained

# 下载YOLOv8n（会自动下载，也可手动下载）
# wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -P models/pretrained/

# 下载dlib模型
# wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
# bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
# mv dlib_face_recognition_resnet_model_v1.dat models/pretrained/

# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bunzip2 shape_predictor_68_face_landmarks.dat.bz2
# mv shape_predictor_68_face_landmarks.dat models/pretrained/

# 下载MiniFASNet（静默活体检测）
# 从 https://github.com/minivision-ai/Silent-Face-Anti-Spoofing 下载
# mv 2.7_80x80_MiniFASNetV2.pth models/pretrained/
```

### 4. 训练行为分类模型

```bash
# 使用示例数据训练（自动生成）
python training/train_action_classifier.py

# 或使用自己的数据
python training/train_action_classifier.py --data_dir ./data/training --epochs 100
```

### 5. 导出模型（可选）

```bash
python scripts/export_models.py
```

## 运行系统

```bash
# 启动主程序
python main.py
```

### 功能说明

#### 1. 作弊检测

- 点击"开始检测"启动
- 系统自动检测教室中的学生行为
- 右侧显示实时统计数据和警报记录

#### 2. 动态点名

- 点击"开始点名"启动
- 系统自动识别已注册的学生人脸
- 右侧显示已签到和未到名单

#### 3. 人脸注册

- 开启预览后，填写学号和姓名
- 系统会自动进行活体检测
- 活体通过后点击"开始注册"

## 数据收集

如需收集自己的训练数据：

```bash
# 录制教室场景视频
python scripts/collect_training_data.py --source 0 --duration 300

# 或使用视频文件
python scripts/collect_training_data.py --source video.mp4 --duration 600
```

### 数据标注格式

行为分类训练数据格式：
- 正常行为: 0
- 传纸条: 1
- 低头偷看: 2
- 东张西望: 3

数据目录结构：
```
data/training/
├── normal/         # 正常行为样本
├── pass_note/      # 传纸条样本
├── look_down/      # 低头偷看样本
└── look_around/    # 东张西望样本
```

每个样本是一个 `.npy` 文件，包含17个关键点的归一化坐标（34维向量）。

## 模型说明

### 人体检测模型

- **模型**: YOLOv8n
- **参数**: 3.2M
- **输入**: 640x640
- **输出**: 人体边界框

### 姿态估计模型

- **模型**: YOLOv8n-Pose
- **参数**: 11MB
- **输入**: 640x640
- **输出**: 17个关键点（COCO格式）

### 行为分类模型

- **模型**: 轻量级MLP
- **结构**: Input(34) -> FC(64) -> FC(32) -> Output(5)
- **输入**: 归一化的关键点坐标
- **输出**: 5类行为

### 人脸识别模型

- **模型**: dlib 128维特征编码
- **特点**: 预训练模型，无需训练
- **匹配阈值**: 0.6

### 活体检测模型

- **模型**: MiniFASNet
- **输入**: 80x80人脸图像
- **输出**: 真人/照片/屏幕

## 性能指标

| 模型 | 推理时间(3050) | 显存占用 |
|------|---------------|----------|
| YOLOv8n检测 | ~8ms | ~500MB |
| YOLOv8n-Pose | ~12ms | ~700MB |
| 行为分类MLP | ~1ms | ~50MB |
| 人脸识别 | ~15ms | ~300MB |

**总体帧率**: 约20-25 FPS（开启所有功能）

## 常见问题

### 1. CUDA out of memory

- 减小 `BATCH_SIZE` 到 4
- 增大 `frame_skip` 到 3-5
- 降低 `IMAGE_SIZE` 到 480

### 2. dlib安装失败

```bash
# Windows需要Visual Studio Build Tools
# 下载地址: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 或下载预编译wheel
pip install dlib-19.24.2-cp39-cp39-win_amd64.whl
```

### 3. 模型下载失败

系统会自动从ultralytics下载YOLO模型。如需手动下载：

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # 自动下载
```

## 开发计划

| 周次 | 任务 |
|------|------|
| 1-2 | 环境搭建、数据收集 |
| 3 | 训练行为分类模型 |
| 4 | 实现基础Pipeline |
| 5 | 实现作弊检测应用 |
| 6 | 实现人脸注册功能 |
| 7 | 实现动态点名功能 |
| 8 | GUI整合、联调测试 |
| 9-10 | 性能优化、撰写论文 |

## 参考文献

1. Redmon, J., et al. "You only look once: Unified, real-time object detection." CVPR 2016.
2. Cao, Z., et al. "OpenPose: Realtime multi-person 2D pose estimation." TPAMI 2019.
3. Kazemi, V., et al. "One millisecond face alignment with an ensemble of regression trees." CVPR 2014.
4. Yu, Z., et al. "Searching central difference convolutional networks for face anti-spoofing." CVPR 2020.

## 许可

本项目仅供学习和毕业设计使用。

## 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**毕业设计项目 | 适配RTX 3050 4GB显存**
