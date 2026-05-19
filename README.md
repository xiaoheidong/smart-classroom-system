# 智慧教室系统 - 基于深度学习的课堂行为分析与考勤

## 项目概述

本项目是一个基于深度学习的智能教室监控系统，专为本科毕业设计开发。系统实现了课堂行为实时检测、人脸识别考勤、智能问答等功能，采用本地部署方式保护学生隐私。

## 核心功能

- **作弊检测**: 实时检测睡觉、玩手机、趴桌子等异常行为，自动抓拍存证
- **动态点名**: 基于人脸识别的无感考勤，支持飞书Webhook推送
- **人脸注册**: 静默活体检测 + 人脸特征采集
- **智能问答**: 基于DeepSeek API的自然语言交互

## 技术栈

| 模块 | 技术 |
|------|------|
| 人体检测 | YOLOv11n (Ultralytics) |
| 行为分类 | ResNet18 (PyTorch) |
| 人脸识别 | dlib / FaceNet 双后端 |
| 活体检测 | MiniFASNet |
| GUI界面 | PyQt5 |
| 数据库 | SQLite |
| 大模型 | DeepSeek API |

## 硬件要求

针对 **RTX 3050 Laptop 4GB显存** 优化：

| 硬件 | 规格 | 说明 |
|------|------|------|
| CPU | Intel i7-12700H | 数据预处理 |
| GPU | RTX 3050 4GB | 模型推理 |
| 内存 | 16GB DDR5 | 数据加载和缓冲 |

## 项目结构

```
smart-classroom-system/
├── main.py                     # 主程序入口
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
├── .gitignore                  # Git忽略配置
├── models/                     # 模型文件
│   ├── yolo11n.pt             # YOLOv11n预训练模型
│   └── trained/               # 训练后的模型
│       └── run_data_8cls_e20_bs8/
│           └── action_classifier.pth
├── src/                        # 源代码
│   ├── config.py              # 配置文件
│   ├── core/                  # 核心算法
│   │   ├── person_detector.py      # 人体检测 (YOLOv11)
│   │   ├── action_classifier.py    # 行为分类 (ResNet18)
│   │   ├── cheating_detector.py    # 作弊检测逻辑
│   │   ├── face_recognition.py     # 人脸识别
│   │   └── liveness_detector.py   # 活体检测
│   ├── gui/                   # GUI界面
│   │   ├── main_window.py          # 主窗口
│   │   ├── cheating_detection_tab.py # 作弊检测页
│   │   ├── attendance_tab.py       # 考勤页
│   │   ├── face_registration_tab.py # 人脸注册页
│   │   └── llm_assistant_tab.py    # 智能问答页
│   └── utils/                 # 工具模块
│       ├── database.py        # 数据库操作
│       ├── feishu_webhook.py  # 飞书推送
│       ├── deepseek_client.py # DeepSeek API客户端
│       └── visualization.py   # 可视化
├── training/                   # 训练脚本
│   └── train_action_classifier.py  # 行为分类训练
└── scripts/                    # 工具脚本
    ├── check_pytorch_cuda.py  # 环境检查
    ├── download_models.py     # 模型下载
    └── init_database.py       # 数据库初始化
```

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/xiaoheidong/smart-classroom-system.git
cd smart-classroom-system
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载模型

```bash
# 自动下载YOLOv11n
python download_yolo11n.py

# 或手动下载其他模型到 models/ 目录
```

### 5. 初始化数据库

```bash
python scripts/init_database.py
```

### 6. 配置API（可选）

编辑 `src/config.py`：

```python
# 飞书Webhook（用于考勤推送）
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx"
FEISHU_WEBHOOK_SECRET = "your_secret"

# DeepSeek API（用于智能问答）
DEEPSEEK_API_KEY = "your_api_key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
```

## 运行系统

```bash
python main.py
```

### 功能说明

#### 1. 作弊检测

- 点击"开始检测"启动
- 系统自动检测教室中的学生行为
- 对睡觉、玩手机等异常行为进行红色标注
- 持续异常30秒或连续25帧触发抓拍存证
- 右侧显示实时统计数据和警报记录

#### 2. 动态点名

- 点击"开始点名"启动
- 系统自动识别已注册的学生人脸
- 右侧显示已签到和未到名单
- 结束考勤后自动推送汇总到飞书

#### 3. 人脸注册

- 开启预览后，填写学号和姓名
- 系统会自动进行活体检测（防止照片攻击）
- 活体通过后点击"开始注册"
- 人脸特征以二进制形式存储在本地数据库

#### 4. 智能问答

- 输入自然语言问题
- 系统基于DeepSeek API回答
- 支持查询考勤统计、系统使用帮助等

## 数据集

### 行为分类数据集

包含8类课堂行为：
- 0: 举手
- 1: 玩手机
- 2: 趴桌子
- 3: 站立
- 4: 看书
- 5: 使用电脑
- 6: 写字
- 7: 听课

### 数据划分

| 数据集 | 原始图片 | 裁剪后样本 | 比例 |
|--------|----------|------------|------|
| 训练集 | 9932张 | 59174个 | 70% |
| 验证集 | 2128张 | 12687个 | 15% |
| 测试集 | 2129张 | 12665个 | 15% |

### 训练模型

```bash
python training/train_action_classifier.py
```

训练参数：
- 模型: ResNet18
- 优化器: AdamW (lr=3e-4, weight_decay=1e-4)
- 学习率调度: CosineAnnealingLR
- 损失函数: 加权交叉熵损失
- 数据增强: 随机裁剪、水平翻转、颜色抖动
- 训练轮数: 20 epochs
- 验证准确率: 93.71%

## 性能指标

| 模块 | 推理时间 | 显存占用 |
|------|----------|----------|
| YOLOv11n检测 | ~8ms | ~500MB |
| ResNet18分类 | ~2ms/框 | ~200MB |
| 人脸识别 | ~15ms | ~300MB |
| 活体检测 | ~5ms | ~100MB |

**总体帧率**: 约12 FPS（RTX 3050, 跳帧比例=1）

## 优化策略

- **跳帧处理**: 可配置跳帧比例（0-3），平衡帧率和精度
- **CPU/GPU自适应**: 自动检测CUDA可用性，无缝切换
- **时序平滑**: 最近5帧投票机制，减少单帧抖动
- **抓拍冷却**: 60秒冷却时间，避免重复存证
- **人脸缓存**: 同场次不重复签到，提高效率

## 隐私保护

- **本地部署**: 所有数据处理在本地完成，不上传云端
- **特征加密**: 人脸特征以二进制BLOB形式存储，非明文
- **无图像存储**: 只存储特征向量，不存储原始人脸图片
- **符合规范**: 遵循《个人信息保护法》要求

## 常见问题

### 1. CUDA out of memory

- 增大 `skip_frames` 到 2-3
- 降低 `batch_size` 到 4
- 使用CPU模式运行

### 2. dlib安装失败

```bash
# Windows需要Visual C++ Build Tools
# 或下载预编译wheel
pip install dlib-19.24.x-cp39-cp39-win_amd64.whl
```

### 3. 模型下载失败

```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # 自动下载
```

## 开发计划

| 阶段 | 任务 |
|------|------|
| 1-2周 | 需求分析、文献调研 |
| 3-4周 | 系统设计、数据采集 |
| 5-8周 | 模型训练、系统开发 |
| 9-12周 | 功能测试、性能优化 |
| 13-16周 | 论文撰写、答辩准备 |

## 参考文献

1. Ultralytics. YOLOv11 [EB/OL]. https://github.com/ultralytics/ultralytics
2. He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
3. King, D. E. "Dlib-ml: A machine learning toolkit." JMLR 2009.
4. Schroff, F., et al. "FaceNet: A unified embedding for face recognition." CVPR 2015.

## 许可

本项目仅供学习和毕业设计使用。

## 联系方式

- 作者: 宋宇
- 学校: 信息工程学院
- 指导教师: 周振宇

---

**本科毕业设计项目 | 2025年4月**
