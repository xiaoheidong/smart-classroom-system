"""
作弊检测Tab页 - 玻璃拟态风格
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QTextEdit, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, VIDEO_CONFIG, DB_PATH
from utils.behavior_evidence import BehaviorEvidenceCollector
from utils.video_capture import VideoCapture
from utils.visualization import Visualizer, resize_frame
from core.person_detector import PersonDetector
from core.action_classifier import ActionClassifier
from core.cheating_detector import CheatingDetector, SimpleTracker
from gui.glass_style import (
    apply_glass_style, get_glass_card_style, get_video_panel_style,
    get_glass_button_style, get_group_box_style, get_table_style,
    get_text_edit_style, get_label_style
)


class ProcessingThread(QThread):
    """视频处理线程"""
    frame_processed = pyqtSignal(np.ndarray, dict)
    stats_updated = pyqtSignal(dict)
    alert_triggered = pyqtSignal(str, str)  # 类型, 消息
    init_failed = pyqtSignal(str)
    evidence_saved = pyqtSignal(dict)  # 抓拍存证：type, track_id, path, reason, db_id

    def __init__(self, video_source=0):
        super().__init__()
        self.video_source = video_source
        self.running = False
        
        # 初始化检测器
        self.detector = None
        self.action_classifier = None
        self.cheating_detector = None
        self.tracker = None
        self.visualizer = None
        self.evidence_collector = BehaviorEvidenceCollector(DB_PATH)

        self.frame_skip = VIDEO_CONFIG['frame_skip']
        self.frame_count = 0
        
    def init_models(self):
        """初始化模型。成功返回 (True, '')，失败返回 (False, 原因)。"""
        try:
            print("加载人体检测模型...")
            self.detector = PersonDetector()

            print("加载行为分类模型...")
            self.action_classifier = ActionClassifier()
            
            print("加载作弊检测器...")
            self.cheating_detector = CheatingDetector()
            self.tracker = SimpleTracker()
            self.visualizer = Visualizer()
            
            return True, ""
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False, str(e)
    
    def run(self):
        """主处理循环"""
        ok, err = self.init_models()
        if not ok:
            self.init_failed.emit(err or "模型加载失败，请查看控制台输出或检查 models 目录。")
            return

        video_source = self.video_source
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)

        # Windows 下 CAP_DSHOW 常能避免摄像头被占用/黑屏；其它系统用默认后端
        if sys.platform == "win32":
            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.init_failed.emit(f"无法打开视频源: {video_source}（摄像头是否被其它页面占用？）")
            return
        
        self.running = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # 循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.frame_count += 1
            
            # 跳帧处理
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # 处理帧
            result = self.process_frame(frame)
            
            # 发送结果
            self.frame_processed.emit(result['frame'], result['stats'])
            self.stats_updated.emit(result['stats'])
            
            # 检查警报
            for alert in result['alerts']:
                self.alert_triggered.emit(alert['type'], alert['message'])
        
        cap.release()
    
    def process_frame(self, frame):
        """处理单帧"""
        result = {
            'frame': frame.copy(),
            'stats': {},
            'alerts': []
        }
        
        # 1. 人体检测
        detections = self.detector.detect(frame)
        
        # 2. 跟踪
        tracked = self.tracker.update(detections)

        cheating_results = []
        bbox_tuple = lambda b: tuple(float(x) for x in b)

        # 3. 对每个跟踪目标进行行为分析
        for track_id, (bbox, conf, cls_id) in tracked:
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            classify_result = self.action_classifier.classify_with_display(person_crop)
            action_conf = classify_result['confidence']
            action_name = classify_result['action_name']
            action_display = classify_result['display_name']

            for ev in self.evidence_collector.process_track(
                track_id, action_name, action_conf, frame, bbox_tuple(bbox)
            ):
                self.evidence_saved.emit(ev)

            cheating_result = self.cheating_detector.detect_cheating(
                action_name, action_conf, track_id
            )
            cheating_results.append(cheating_result)

            color = COLORS['normal']
            label = action_display

            if cheating_result['is_cheating']:
                color = COLORS['cheating']
                label = f"⚠ {cheating_result['cheating_type']}"

                if cheating_result['confidence'] > 0.7:
                    result['alerts'].append({
                        'type': cheating_result['cheating_type'],
                        'message': f"检测到异常行为: {cheating_result['cheating_type']} (ID: {track_id})"
                    })
            
            # 绘制检测框
            result['frame'] = self.visualizer.draw_bbox(
                result['frame'], bbox, label, action_conf, color
            )

        self.evidence_collector.prune_tracks([t[0] for t in tracked])

        # 更新统计数据
        self.cheating_detector.update_stats(cheating_results)
        result['stats'] = self.cheating_detector.get_stats()
        
        # 统计信息已移至右侧栏显示，不再在视频画面上叠加
        # stats_text = {
        #     '总人数': result['stats']['total'],
        #     '正常': result['stats']['normal'],
        #     '异常': result['stats']['cheating']
        # }
        # result['frame'] = self.visualizer.draw_stats(result['frame'], stats_text)
        
        return result
    
    def stop(self):
        self.running = False
        self.wait()


class CheatingDetectionTab(QWidget):
    """作弊检测Tab页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.processing_thread = None
        self.alert_history = deque(maxlen=100)
        self.video_source = 0
        self.init_ui()
    
    def init_ui(self):
        """初始化UI - 玻璃拟态风格"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # 应用玻璃风格
        apply_glass_style(self)
        
        # 左侧：视频显示
        left_panel = self.create_video_panel()
        
        # 右侧：控制面板
        right_panel = self.create_control_panel()
        
        # 添加分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        splitter.setHandleWidth(2)
        
        layout.addWidget(splitter)
    
    def create_video_panel(self):
        """创建视频显示面板 - 深色带银色边框"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频标签 - 深色背景 + 银色边框
        self.video_label = QLabel("视频显示区")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(get_video_panel_style())
        
        layout.addWidget(self.video_label)
        return panel
    
    def create_control_panel(self):
        """创建控制面板 - 玻璃拟态风格"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # 控制按钮组 - 玻璃卡片
        control_group = QGroupBox("控制")
        control_group.setStyleSheet(get_group_box_style())
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_layout.setSpacing(12)
        
        # 视频源选择
        source_layout = QHBoxLayout()
        source_label = QLabel("视频源:")
        source_label.setStyleSheet(get_label_style('muted'))
        
        self.combo_source = QComboBox()
        self.combo_source.addItem("默认摄像头", 0)
        self.combo_source.addItem("摄像头 1", 1)
        self.combo_source.addItem("摄像头 2", 2)
        self.combo_source.addItem("本地视频文件...", "file")
        self.combo_source.setMinimumHeight(30)
        self.combo_source.currentIndexChanged.connect(self.on_source_changed)
        
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.combo_source)
        
        self.btn_start = QPushButton("▶ 开始检测")
        self.btn_start.setStyleSheet(get_glass_button_style('primary'))
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_detection)
        
        self.btn_stop = QPushButton("⏹ 停止检测")
        self.btn_stop.setStyleSheet(get_glass_button_style('danger'))
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.lbl_source = QLabel("当前视频源: 默认摄像头")
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setStyleSheet(get_label_style('muted'))
        
        control_layout.addLayout(source_layout)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.lbl_source)
        
        # 统计面板 - 玻璃卡片
        stats_group = QGroupBox("实时统计")
        stats_group.setStyleSheet(get_group_box_style())
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(16, 16, 16, 16)
        stats_layout.setSpacing(12)
        
        self.lbl_total = QLabel("总人数: 0")
        self.lbl_total.setStyleSheet(get_label_style('normal'))
        
        self.lbl_normal = QLabel("正常: 0")
        self.lbl_normal.setStyleSheet(get_label_style('stat_blue'))
        
        self.lbl_cheating = QLabel("异常: 0")
        self.lbl_cheating.setStyleSheet(get_label_style('stat_coral'))
        
        self.lbl_sleep = QLabel("睡觉: 0")
        self.lbl_sleep.setStyleSheet(get_label_style('normal'))
        
        self.lbl_phone = QLabel("使用手机: 0")
        self.lbl_phone.setStyleSheet(get_label_style('normal'))
        
        stats_layout.addWidget(self.lbl_total, 0, 0)
        stats_layout.addWidget(self.lbl_normal, 0, 1)
        stats_layout.addWidget(self.lbl_cheating, 1, 0)
        stats_layout.addWidget(self.lbl_sleep, 1, 1)
        stats_layout.addWidget(self.lbl_phone, 2, 0)
        
        # 警报记录 - 玻璃卡片
        alert_group = QGroupBox("警报记录")
        alert_group.setStyleSheet(get_group_box_style())
        alert_layout = QVBoxLayout(alert_group)
        alert_layout.setContentsMargins(16, 16, 16, 16)
        
        self.alert_table = QTableWidget()
        self.alert_table.setColumnCount(3)
        self.alert_table.setHorizontalHeaderLabels(["时间", "类型", "详情"])
        self.alert_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alert_table.setMaximumHeight(180)
        self.alert_table.setStyleSheet(get_table_style())
        
        alert_layout.addWidget(self.alert_table)
        
        # 日志区域 - 玻璃卡片
        log_group = QGroupBox("系统日志")
        log_group.setStyleSheet(get_group_box_style())
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(16, 16, 16, 16)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setStyleSheet(get_text_edit_style(read_only=True))
        
        log_layout.addWidget(self.log_text)
        
        # 添加所有组件
        layout.addWidget(control_group)
        layout.addWidget(stats_group)
        layout.addWidget(alert_group)
        layout.addWidget(log_group)
        
        return panel
    
    def start_detection(self):
        """开始检测"""
        self.log("正在启动检测...")
        source_name = "默认摄像头" if self.video_source == 0 else str(self.video_source)
        self.log(f"当前视频源: {source_name}")
        
        # 创建处理线程
        self.processing_thread = ProcessingThread(self.video_source)
        self.processing_thread.frame_processed.connect(self.on_frame_processed)
        self.processing_thread.stats_updated.connect(self.on_stats_updated)
        self.processing_thread.alert_triggered.connect(self.on_alert)
        self.processing_thread.init_failed.connect(self.on_init_failed)
        self.processing_thread.evidence_saved.connect(self.on_evidence_saved)
        
        self.processing_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.log("检测已启动")
    
    def on_init_failed(self, message: str):
        """子线程初始化失败（模型或摄像头）"""
        self.log(f"启动失败: {message}")
        QMessageBox.warning(self, "作弊检测无法启动", message)
        self.processing_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
    
    def stop_detection(self):
        """停止检测"""
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        self.log("检测已停止")

    def on_source_changed(self, index):
        """视频源下拉框切换回调"""
        source_data = self.combo_source.currentData()
        
        if source_data == "file":
            # 选择本地视频文件
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                "",
                "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*.*)"
            )
            if filename:
                self.set_video_source(filename)
                # 更新下拉框显示为文件名
                self.combo_source.setItemText(index, os.path.basename(filename))
            else:
                # 用户取消选择，恢复默认摄像头
                self.combo_source.setCurrentIndex(0)
        else:
            # 选择摄像头
            self.set_video_source(source_data)
    
    def set_video_source(self, video_source):
        """设置视频源，支持摄像头编号或视频文件路径。"""
        if self.processing_thread:
            self.stop_detection()

        self.video_source = video_source
        if video_source == 0:
            self.lbl_source.setText("当前视频源: 默认摄像头")
            self.log("已切换到默认摄像头")
        else:
            self.lbl_source.setText(f"当前视频源: {video_source}")
            self.log(f"已选择视频文件: {video_source}")
    
    def on_frame_processed(self, frame, stats):
        """帧处理完成回调"""
        # 调整大小以适应显示
        display_frame = resize_frame(frame, max_width=800, max_height=600)
        
        # 转换为QPixmap
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        
        q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)
    
    def on_stats_updated(self, stats):
        """统计更新回调"""
        self.lbl_total.setText(f"总人数: {stats['total']}")
        self.lbl_normal.setText(f"正常: {stats['normal']}")
        self.lbl_cheating.setText(f"异常: {stats['cheating']}")
        
        self.lbl_sleep.setText(f"睡觉: {stats['by_type'].get('睡觉', 0)}")
        self.lbl_phone.setText(f"使用手机: {stats['by_type'].get('使用手机', 0)}")
    
    def on_evidence_saved(self, ev: dict):
        """异常行为抓拍已写入数据库"""
        r = ev.get("reason", "")
        tid = ev.get("track_id", "")
        p = ev.get("path", "")
        self.log(f"[存证] {ev.get('type')} track={tid} {r} -> {p}")

    def on_alert(self, alert_type, message):
        """警报回调"""
        # 添加到表格
        row = self.alert_table.rowCount()
        self.alert_table.insertRow(row)
        
        time_str = datetime.now().strftime("%H:%M:%S")
        
        self.alert_table.setItem(row, 0, QTableWidgetItem(time_str))
        self.alert_table.setItem(row, 1, QTableWidgetItem(alert_type))
        self.alert_table.setItem(row, 2, QTableWidgetItem(message))
        
        # 滚动到最新
        self.alert_table.scrollToBottom()
        
        self.log(f"[警报] {message}")
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_detection()
        event.accept()
