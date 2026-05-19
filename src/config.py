"""
智慧教室系统配置文件
适配RTX 3050 4GB显存
"""
import os
from typing import Optional

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模型路径
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PRETRAINED_DIR = os.path.join(MODELS_DIR, 'pretrained')
TRAINED_DIR = os.path.join(MODELS_DIR, 'trained')

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'videos')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')

# 数据库路径
DB_DIR = os.path.join(BASE_DIR, 'db')
DB_PATH = os.path.join(DB_DIR, 'smart_classroom.db')

# 预训练模型文件名
YOLOV11N_MODEL = 'yolo11n.pt'
DLIB_FACE_MODEL = 'dlib_face_recognition_resnet_model_v1.dat'
DLIB_LANDMARK_MODEL = 'shape_predictor_68_face_landmarks.dat'
LIVENESS_MODEL = '2.7_80x80_MiniFASNetV2.pth'

# 训练好的模型文件名
ACTION_CLASSIFIER_MODEL = 'action_classifier_cnn_best.pth'
YOLOV11N_PERSON_MODEL = 'yolo11n_person.torchscript.pt'


def resolve_action_classifier_path() -> Optional[str]:
    """
    解析行为分类权重路径：优先 trained 根目录，其次常见训练输出子目录，
    再扫描 trained 下任意一级子目录中的同名文件（便于 run_data_xxx 结构）。
    """
    name = ACTION_CLASSIFIER_MODEL
    direct = os.path.join(TRAINED_DIR, name)
    if os.path.isfile(direct):
        return direct
    common = os.path.join(TRAINED_DIR, 'run_data_8cls_e20_bs8', name)
    if os.path.isfile(common):
        return common
    if os.path.isdir(TRAINED_DIR):
        try:
            for sub in sorted(os.listdir(TRAINED_DIR), reverse=True):
                sub_path = os.path.join(TRAINED_DIR, sub)
                if not os.path.isdir(sub_path):
                    continue
                p = os.path.join(sub_path, name)
                if os.path.isfile(p):
                    return p
        except OSError:
            pass
    return None

# 设备配置
import torch
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动检测GPU/CPU
BATCH_SIZE = 4  # 适配4GB显存
IMAGE_SIZE = 640

# 行为分类配置
ACTION_CLASSES = {
    0: 'sleep',
    1: 'raise_head',
    2: 'using_computer',
    3: 'stand',
    4: 'using_phone',
    5: 'raise_hand',
    6: 'read',
    7: 'writing',
}

ACTION_DISPLAY_NAMES = {
    'sleep': '睡觉',
    'raise_head': '抬头听课',
    'using_computer': '使用电脑',
    'stand': '站立',
    'using_phone': '使用手机',
    'raise_hand': '举手',
    'read': '阅读',
    'writing': '写字',
}

NORMAL_ACTIONS = {
    'raise_head',
    'using_computer',
    'stand',
    'raise_hand',
    'read',
    'writing',
}

ALERT_ACTIONS = {
    'sleep': '睡觉',
    'using_phone': '使用手机'
}

# 作弊检测阈值
CHEATING_CONFIG = {
    'head_pitch_threshold': -25,  # 低头角度阈值
    'head_yaw_threshold': 40,     # 头部转动阈值
    'confidence_threshold': 0.6   # 置信度阈值
}

# 异常行为抓拍存证（存磁盘 + SQLite 元数据）
CAPTURES_DIR = os.path.join(BASE_DIR, 'captures', 'behavior')
EVIDENCE_CONFIG = {
    # 睡觉：持续超过该秒数则抓拍（从首次判定为睡觉开始计时）
    'sleep_duration_sec': 30.0,
    # 睡觉：连续「已处理帧」中判定为睡觉的帧数达到该值则抓拍（与上条二选一满足即触发一次）
    'sleep_consecutive_frames': 25,
    # 玩手机：同一 track_id 两次抓拍最小间隔（秒），避免每帧一张
    'phone_cooldown_sec': 45.0,
    # 抓拍时最低分类置信度
    'min_confidence': 0.6,
}

# 人脸识别配置
FACE_CONFIG = {
    'face_match_threshold': 0.6,  # dlib 欧氏距离阈值（约 <0.6 判同一人）
    # facenet-pytorch 512 维单位向量欧氏距离（约 <0.95 判同一人，可按实验微调）
    'face_match_threshold_torch': 0.95,
    'liveness_threshold': 0.9,
    'detection_confidence': 0.5
}

# 视频处理配置
VIDEO_CONFIG = {
    'fps': 30,
    'frame_skip': 2,  # 每2帧处理一次，降低计算量
    'buffer_size': 30  # 行为分析缓冲区大小（帧数）
}

# 飞书群机器人（课堂签到结束推送未到名单）。生产环境建议用环境变量覆盖，勿提交真实密钥到公开仓库
FEISHU_WEBHOOK_URL = os.environ.get(
    "FEISHU_WEBHOOK_URL",
    "https://open.feishu.cn/open-apis/bot/v2/hook/e04c671d-cfa2-4c8c-be06-a34167bcd1b4",
)
FEISHU_WEBHOOK_SECRET = os.environ.get(
    "FEISHU_WEBHOOK_SECRET",
    "3RA9ePbXskMc1HPjhyplXc",
)

# DeepSeek 大模型（「智能问答」分页）。密钥仅通过环境变量注入，勿写入代码或提交仓库
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# 颜色配置（用于可视化）
COLORS = {
    'normal': (0, 255, 0),      # 绿色
    'cheating': (0, 0, 255),    # 红色
    'warning': (0, 255, 255),   # 黄色
    'bbox': (255, 0, 0),        # 蓝色
    'keypoint': (0, 255, 0)     # 绿色
}
