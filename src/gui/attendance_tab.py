"""
动态点名Tab页 - 玻璃拟态风格
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QTextEdit, QComboBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, COLORS, FEISHU_WEBHOOK_URL, FEISHU_WEBHOOK_SECRET
from utils.database import Database
from utils.feishu_webhook import send_session_attendance_to_feishu
from utils.visualization import Visualizer
from core.face_recognition import FaceRecognizer
from gui.glass_style import (
    apply_glass_style, get_video_panel_style, get_glass_button_style,
    get_group_box_style, get_table_style, get_text_edit_style,
    get_label_style, get_combo_box_style, get_secondary_button_style
)


class AttendanceThread(QThread):
    """考勤处理线程"""
    frame_processed = pyqtSignal(np.ndarray, list)
    attendance_recorded = pyqtSignal(str, str, float)  # student_id, name, confidence
    
    def __init__(self, video_source=0, db_path=DB_PATH, session_id=None):
        super().__init__()
        self.video_source = video_source
        self.db_path = db_path
        self.session_id = session_id
        self.running = False
        
        self.recognizer = None
        self.db = None
        self.visualizer = None
        
        # 本场已签到人员缓存（避免重复写入）
        self.checked_in_session = set()
        
    def init_system(self):
        """初始化系统"""
        try:
            print("初始化人脸识别...")
            self.recognizer = FaceRecognizer()
            
            print("初始化数据库...")
            self.db = Database(self.db_path)
            
            self.visualizer = Visualizer()
            
            # 加载已注册学生
            self.load_known_faces()
            
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def load_known_faces(self):
        """加载已注册的人脸（仅加载与当前人脸后端维度一致的特征）"""
        students = self.db.get_all_students()
        
        self.known_encodings = []
        self.known_ids = []
        self.known_names = []
        dim = self.recognizer.embedding_dim
        for student in students:
            enc = student["face_encoding"]
            if enc is None:
                continue
            if enc.shape[0] != dim:
                print(
                    f"跳过 {student['id']}: 特征维度 {enc.shape[0]} 与当前后端 {dim} 不一致，请重新注册"
                )
                continue
            self.known_encodings.append(enc)
            self.known_ids.append(student["id"])
            self.known_names.append(student["name"])
        
        print(f"加载了 {len(self.known_encodings)} 个人脸（后端: {self.recognizer.backend}）")
    
    def run(self):
        """主循环"""
        if not self.init_system():
            return
        
        src = self.video_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"无法打开视频源: {self.video_source}")
            return
        
        self.running = True
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # 每3帧处理一次
            if frame_count % 3 != 0:
                continue
            
            # 处理帧
            result_frame, faces_info = self.process_frame(frame)
            
            # 发送结果
            self.frame_processed.emit(result_frame, faces_info)
            
            # 记录考勤
            for info in faces_info:
                if (
                    self.session_id is None
                    or not info["recognized"]
                    or info["student_id"] in self.checked_in_session
                ):
                    continue
                success = self.db.record_attendance(
                    info["student_id"],
                    info["confidence"],
                    self.session_id,
                )
                if success:
                    self.checked_in_session.add(info["student_id"])
                    self.attendance_recorded.emit(
                        info["student_id"],
                        info["name"],
                        info["confidence"],
                    )
        
        cap.release()
    
    def process_frame(self, frame):
        """处理单帧"""
        result_frame = frame.copy()
        faces_info = []
        
        # 检测人脸
        faces = self.recognizer.detect_faces(frame)
        
        for face_rect in faces:
            left, top, right, bottom = face_rect
            
            # 识别人脸
            result = self.recognizer.recognize_face(
                frame,
                face_rect,
                self.known_encodings,
                self.known_ids
            )
            
            if result['success'] and result['encoding'] is not None:
                # 找到匹配的姓名
                name = "未知"
                if result['name'] is not None:
                    idx = self.known_ids.index(result['name'])
                    name = self.known_names[idx]
                
                is_recognized = result['name'] is not None
                
                info = {
                    'face_rect': face_rect,
                    'student_id': result['name'] if result['name'] else 'unknown',
                    'name': name,
                    'confidence': result['confidence'] if result['confidence'] else 0.0,
                    'recognized': is_recognized
                }
                faces_info.append(info)
                
                # 可视化
                color = COLORS['normal'] if is_recognized else COLORS['warning']
                label = f"{name} ({result['confidence']:.2f})" if is_recognized else "未知"
                
                result_frame = self.visualizer.draw_face_box(
                    result_frame, face_rect, label, result['confidence'] if result['confidence'] else 0
                )
            else:
                # 仅绘制检测框
                result_frame = self.visualizer.draw_face_box(
                    result_frame, face_rect, "未注册"
                )
        
        return result_frame, faces_info
    
    def stop(self):
        self.running = False
        self.wait()


class AttendanceTab(QWidget):
    """考勤Tab页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.attendance_thread = None
        self._active_session_id = None
        self.init_ui()
        self._populate_session_combo()
    
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
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 500])
        splitter.setHandleWidth(2)
        
        layout.addWidget(splitter)
    
    def create_video_panel(self):
        """创建视频面板 - 深色带银色边框"""
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
        
        # 场次选择 - 玻璃卡片
        history_group = QGroupBox("场次选择")
        history_group.setStyleSheet(get_group_box_style())
        history_layout = QVBoxLayout(history_group)
        history_layout.setContentsMargins(16, 16, 16, 16)
        
        self.lbl_session_info = QLabel("未选择场次")
        self.lbl_session_info.setWordWrap(True)
        self.lbl_session_info.setStyleSheet(get_label_style('muted'))
        
        self.session_combo = QComboBox()
        self.session_combo.setToolTip("每次「开始点名」开启一场；「停止」后本场结束，未到即缺勤")
        self.session_combo.currentIndexChanged.connect(self._on_session_combo_changed)
        self.session_combo.setStyleSheet(get_combo_box_style())
        
        history_layout.addWidget(self.lbl_session_info)
        history_layout.addWidget(self.session_combo)
        
        # 控制组 - 玻璃卡片
        control_group = QGroupBox("控制")
        control_group.setStyleSheet(get_group_box_style())
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_layout.setSpacing(10)
        
        self.btn_start = QPushButton("▶ 开始点名")
        self.btn_start.setStyleSheet(get_glass_button_style('primary'))
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_attendance)
        
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setStyleSheet(get_glass_button_style('danger'))
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_attendance)
        
        self.btn_refresh = QPushButton("🔄 刷新名单")
        self.btn_refresh.setStyleSheet(get_secondary_button_style())
        self.btn_refresh.clicked.connect(self.refresh_lists)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(self.btn_refresh)
        
        # 统计组 - 玻璃卡片
        stats_group = QGroupBox("签到统计")
        stats_group.setStyleSheet(get_group_box_style())
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(16, 16, 16, 16)
        stats_layout.setSpacing(12)
        
        self.lbl_total = QLabel("应到: 0")
        self.lbl_total.setStyleSheet(get_label_style('normal'))
        
        self.lbl_present = QLabel("实到: 0")
        self.lbl_present.setStyleSheet(get_label_style('stat_blue'))
        
        self.lbl_absent = QLabel("未到: 0")
        self.lbl_absent.setStyleSheet(get_label_style('stat_coral'))
        
        stats_layout.addWidget(self.lbl_total, 0, 0)
        stats_layout.addWidget(self.lbl_present, 0, 1)
        stats_layout.addWidget(self.lbl_absent, 1, 0)
        
        # 已签到列表 - 玻璃卡片
        present_group = QGroupBox("已签到学生")
        present_group.setStyleSheet(get_group_box_style())
        present_layout = QVBoxLayout(present_group)
        present_layout.setContentsMargins(16, 16, 16, 16)
        
        self.present_table = QTableWidget()
        self.present_table.setColumnCount(3)
        self.present_table.setHorizontalHeaderLabels(["学号", "姓名", "置信度"])
        self.present_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.present_table.setMaximumHeight(160)
        self.present_table.setStyleSheet(get_table_style())
        
        present_layout.addWidget(self.present_table)
        
        # 未到列表 - 玻璃卡片
        absent_group = QGroupBox("未到学生")
        absent_group.setStyleSheet(get_group_box_style())
        absent_layout = QVBoxLayout(absent_group)
        absent_layout.setContentsMargins(16, 16, 16, 16)
        
        self.absent_table = QTableWidget()
        self.absent_table.setColumnCount(2)
        self.absent_table.setHorizontalHeaderLabels(["学号", "姓名"])
        self.absent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.absent_table.setMaximumHeight(160)
        self.absent_table.setStyleSheet(get_table_style())
        
        absent_layout.addWidget(self.absent_table)
        
        # 日志 - 玻璃卡片
        log_group = QGroupBox("系统日志")
        log_group.setStyleSheet(get_group_box_style())
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(16, 16, 16, 16)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet(get_text_edit_style(read_only=True))
        
        log_layout.addWidget(self.log_text)
        
        # 添加所有组件
        layout.addWidget(history_group)
        layout.addWidget(control_group)
        layout.addWidget(stats_group)
        layout.addWidget(present_group)
        layout.addWidget(absent_group)
        layout.addWidget(log_group)
        
        return panel
    
    def _populate_session_combo(self, select_id=None):
        """填充历史场次下拉框"""
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        db = Database(DB_PATH)
        sessions = db.list_attendance_sessions(50)
        for s in sessions:
            self.session_combo.addItem(s["label"], s["id"])
        if select_id is not None:
            idx = self.session_combo.findData(select_id)
            if idx >= 0:
                self.session_combo.setCurrentIndex(idx)
        elif self.session_combo.count() > 0:
            self.session_combo.setCurrentIndex(0)
        self.session_combo.blockSignals(False)
    
    def _on_session_combo_changed(self):
        if self._active_session_id is not None:
            return
        self.refresh_lists()
    
    def _display_session_id(self):
        """当前展示的场次：进行中优先，否则下拉所选"""
        if self._active_session_id is not None:
            return self._active_session_id
        data = self.session_combo.currentData()
        if data is not None:
            return int(data)
        db = Database(DB_PATH)
        return db.get_latest_session_id()
    
    def start_attendance(self):
        """开始考勤"""
        self.log("正在启动考勤系统...")
        
        db = Database(DB_PATH)
        self._active_session_id = db.start_attendance_session()
        self._populate_session_combo(select_id=self._active_session_id)
        self._update_session_info_label()
        
        self.attendance_thread = AttendanceThread(0, session_id=self._active_session_id)
        self.attendance_thread.frame_processed.connect(self.on_frame_processed)
        self.attendance_thread.attendance_recorded.connect(self.on_attendance_recorded)
        
        self.attendance_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.session_combo.setEnabled(False)
        
        self.refresh_lists()
        self.log(f"本场签到已开启 (session #{self._active_session_id})")
    
    def stop_attendance(self):
        """停止考勤（本场签到截止）"""
        if self.attendance_thread:
            self.attendance_thread.stop()
            self.attendance_thread = None
        
        ended_id = None
        if self._active_session_id is not None:
            db = Database(DB_PATH)
            db.end_attendance_session(self._active_session_id)
            ended_id = self._active_session_id
            self._active_session_id = None
            self.log(f"本场签到已结束 (session #{ended_id})")
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.session_combo.setEnabled(True)
        
        if ended_id is not None:
            self._populate_session_combo(select_id=ended_id)
        self._update_session_info_label()
        self.refresh_lists()
        self.log("考勤系统已停止")

        if ended_id is not None and FEISHU_WEBHOOK_URL.strip():
            ok, msg = send_session_attendance_to_feishu(
                DB_PATH,
                ended_id,
                FEISHU_WEBHOOK_URL,
                FEISHU_WEBHOOK_SECRET or None,
            )
            self.log(f"飞书推送: {'成功' if ok else '失败'} — {msg}")
    
    def _update_session_info_label(self):
        sid = self._display_session_id()
        if sid is None:
            self.lbl_session_info.setText("暂无场次记录")
            return
        db = Database(DB_PATH)
        meta = db.get_session_row(sid)
        if not meta:
            self.lbl_session_info.setText("")
            return
        started, ended = meta["started_at"], meta["ended_at"]
        if ended:
            self.lbl_session_info.setText(
                f"场次 #{sid}：{started} ～ {ended}（未到者视为缺勤）"
            )
        else:
            self.lbl_session_info.setText(
                f"场次 #{sid}：{started} 起（进行中）"
            )
    
    def refresh_lists(self, silent=False):
        """按当前选中/进行中的场次刷新名单"""
        try:
            sid = self._display_session_id()
            self._update_session_info_label()
            
            if sid is None:
                self.lbl_total.setText("应到: 0")
                self.lbl_present.setText("实到: 0")
                self.lbl_absent.setText("未到: 0")
                self.present_table.setRowCount(0)
                self.absent_table.setRowCount(0)
                if not silent:
                    self.log("名单已刷新（无场次）")
                return
            
            db = Database(DB_PATH)
            present_records, absent = db.get_session_summary(sid)
            total = len(present_records) + len(absent)
            self.lbl_total.setText(f"应到: {total}")
            self.lbl_present.setText(f"实到: {len(present_records)}")
            self.lbl_absent.setText(f"未到: {len(absent)}")
            
            self.present_table.setRowCount(0)
            for record in present_records:
                row = self.present_table.rowCount()
                self.present_table.insertRow(row)
                self.present_table.setItem(row, 0, QTableWidgetItem(record["student_id"]))
                self.present_table.setItem(row, 1, QTableWidgetItem(record["name"]))
                conf = record.get("confidence")
                self.present_table.setItem(
                    row, 2, QTableWidgetItem(f"{conf:.2f}" if conf is not None else "-")
                )
            
            self.absent_table.setRowCount(0)
            for a in absent:
                row = self.absent_table.rowCount()
                self.absent_table.insertRow(row)
                self.absent_table.setItem(row, 0, QTableWidgetItem(a["id"]))
                self.absent_table.setItem(row, 1, QTableWidgetItem(a["name"]))
            
            if not silent:
                self.log(f"名单已刷新（场次 #{sid}）")
            
        except Exception as e:
            self.log(f"刷新名单失败: {e}")
    
    def on_frame_processed(self, frame, faces_info):
        """帧处理回调"""
        # 转换显示
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        
        q_image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)
    
    def on_attendance_recorded(self, student_id, name, confidence):
        """考勤记录回调"""
        self.log(f"✓ 签到成功: {name} ({student_id}), 置信度: {confidence:.2f}")
        
        self.refresh_lists(silent=True)
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_attendance()
        event.accept()
