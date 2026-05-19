"""
人脸注册Tab页 - 玻璃拟态风格
支持静默活体检测 + 人脸采集
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QLineEdit, QMessageBox, QProgressBar,
    QTextEdit, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, PRETRAINED_DIR, FACE_CONFIG, COLORS
from utils.database import Database
from core.face_recognition import FaceRecognizer
from core.liveness_detector import LivenessDetector, SimpleLivenessDetector
from utils.visualization import Visualizer
from gui.glass_style import (
    apply_glass_style, get_video_panel_style, get_glass_button_style,
    get_group_box_style, get_table_style, get_text_edit_style,
    get_label_style, get_line_edit_style, get_progress_bar_style,
    get_secondary_button_style
)


class RegistrationThread(QThread):
    """注册处理线程"""
    frame_processed = pyqtSignal(np.ndarray, dict)
    liveness_result = pyqtSignal(bool, float)
    registration_complete = pyqtSignal(bool, str)
    
    def __init__(self, video_source=0, db_path=DB_PATH):
        super().__init__()
        self.video_source = video_source
        self.db_path = db_path
        self.running = False
        
        self.recognizer = None
        self.liveness_detector = None
        self.visualizer = None

        # 注册状态
        self.registration_mode = False
        self.student_id = None
        self.student_name = None
        
    def init_system(self):
        """初始化系统"""
        try:
            print("初始化人脸识别...")
            self.recognizer = FaceRecognizer()
            
            print("初始化活体检测...")
            # 先尝试使用MiniFASNet
            self.liveness_detector = LivenessDetector()
            if self.liveness_detector.model is None:
                # 回退到简单活体检测
                print("使用备用活体检测方案")
                self.liveness_detector = SimpleLivenessDetector()

            self.visualizer = Visualizer()
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def run(self):
        """主循环"""
        if not self.init_system():
            return
        
        src = self.video_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if sys.platform == "win32":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"无法打开视频源: {self.video_source}")
            return
        
        self.running = True
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                result = self.process_frame(frame)
                
                self.frame_processed.emit(result["frame"], result["info"])
                self.liveness_result.emit(
                    result["info"]["is_live"],
                    result["info"]["liveness_score"],
                )
                
                if (
                    self.registration_mode
                    and result["info"]["face_detected"]
                    and result["info"]["is_live"]
                ):
                    enc = result["info"]["face_encoding"]
                    if enc is None:
                        self.registration_complete.emit(
                            False, "未提取到人脸特征，请调整光线与角度后重试"
                        )
                        self.registration_mode = False
                    else:
                        success, message = self.register_student(
                            frame,
                            result["info"]["face_rect"],
                            enc,
                        )
                        self.registration_complete.emit(success, message)
                        self.registration_mode = False
            except Exception as e:
                print(f"[RegistrationThread] 帧处理异常: {e}")
                import traceback
                traceback.print_exc()
        
        cap.release()
    
    def process_frame(self, frame):
        """处理单帧"""
        result = {
            'frame': frame.copy(),
            'info': {
                'face_detected': False,
                'face_rect': None,
                'is_live': False,
                'liveness_score': 0.0,
                'face_encoding': None
            }
        }
        
        # 检测人脸
        faces = self.recognizer.detect_faces(frame)
        
        if len(faces) > 0:
            face_rect = faces[0]  # 取最大人脸
            left, top, right, bottom = face_rect
            
            result['info']['face_detected'] = True
            result['info']['face_rect'] = face_rect
            
            # 活体检测
            is_live, score = self.liveness_detector.detect(frame, face_rect)
            result['info']['is_live'] = is_live
            result['info']['liveness_score'] = score
            
            # 提取特征
            encoding = self.recognizer.get_face_encoding(frame, face_rect)
            result['info']['face_encoding'] = encoding
            
            # 可视化（人脸框 + 中文标签用 PIL，避免 OpenCV 字体显示为问号）
            color = COLORS["normal"] if is_live else COLORS["cheating"]
            cv2.rectangle(result["frame"], (left, top), (right, bottom), color, 2)

            if self.visualizer is not None:
                label = f"{'活体' if is_live else '非活体'} ({score:.2f})"
                fs = 20
                _, text_h = self.visualizer._measure_text(label, fs)
                label_y = max(0, top - text_h - 10)
                result["frame"] = self.visualizer._draw_text(
                    result["frame"],
                    label,
                    (left + 4, label_y),
                    font_size=fs,
                    color=(255, 255, 255),
                    background_color=color,
                    padding=4,
                )
                if self.registration_mode:
                    result["frame"] = self.visualizer._draw_text(
                        result["frame"],
                        "正在注册...",
                        (left + 4, bottom + 8),
                        font_size=20,
                        color=(0, 255, 255),
                        background_color=(0, 0, 0),
                        padding=4,
                    )
        
        return result
    
    def register_student(self, frame, face_rect, face_encoding):
        """注册学生"""
        if face_encoding is None:
            return False, "人脸特征为空，请重试"
        try:
            db = Database(self.db_path)
            
            # 检查学号是否已存在
            existing = db.get_student(self.student_id)
            if existing:
                # 更新人脸
                db.add_student(self.student_id, self.student_name, face_encoding)
                return True, f"更新成功: {self.student_name} ({self.student_id})"
            else:
                # 添加新学生
                db.add_student(self.student_id, self.student_name, face_encoding)
                return True, f"注册成功: {self.student_name} ({self.student_id})"
                
        except Exception as e:
            return False, f"注册失败: {str(e)}"
    
    def start_registration(self, student_id, student_name):
        """开始注册流程"""
        self.student_id = student_id
        self.student_name = student_name
        self.registration_mode = True
    
    def stop(self):
        self.running = False
        self.wait()


class FaceRegistrationTab(QWidget):
    """人脸注册Tab页"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.registration_thread = None
        self.init_ui()
    
    def init_ui(self):
        """初始化UI - 玻璃拟态风格"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # 应用玻璃风格
        apply_glass_style(self)
        
        # 左侧：视频和状态
        left_panel = self.create_video_panel()
        
        # 右侧：注册表单和学生列表
        right_panel = self.create_form_panel()
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([650, 550])
        splitter.setHandleWidth(2)
        
        layout.addWidget(splitter)
    
    def create_video_panel(self):
        """创建视频面板 - 玻璃拟态风格"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # 视频显示 - 深色背景 + 银色边框
        self.video_label = QLabel("摄像头预览")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(get_video_panel_style())
        
        # 活体检测状态 - 玻璃卡片
        status_group = QGroupBox("检测状态")
        status_group.setStyleSheet(get_group_box_style())
        status_layout = QGridLayout(status_group)
        status_layout.setContentsMargins(16, 16, 16, 16)
        status_layout.setSpacing(10)
        
        self.lbl_face_status = QLabel("人脸: 未检测")
        self.lbl_face_status.setStyleSheet(get_label_style('normal'))
        
        self.lbl_liveness_status = QLabel("活体: 未检测")
        self.lbl_liveness_status.setStyleSheet(get_label_style('normal'))
        
        self.liveness_bar = QProgressBar()
        self.liveness_bar.setRange(0, 100)
        self.liveness_bar.setValue(0)
        self.liveness_bar.setStyleSheet(get_progress_bar_style())
        
        status_layout.addWidget(QLabel("人脸状态:"), 0, 0)
        status_layout.addWidget(self.lbl_face_status, 0, 1)
        status_layout.addWidget(QLabel("活体检测:"), 1, 0)
        status_layout.addWidget(self.lbl_liveness_status, 1, 1)
        status_layout.addWidget(self.liveness_bar, 2, 0, 1, 2)
        
        layout.addWidget(self.video_label)
        layout.addWidget(status_group)
        
        return panel
    
    def create_form_panel(self):
        """创建表单面板 - 玻璃拟态风格"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # 注册表单 - 玻璃卡片
        form_group = QGroupBox("学生注册表单")
        form_group.setStyleSheet(get_group_box_style())
        form_layout = QGridLayout(form_group)
        form_layout.setContentsMargins(16, 16, 16, 16)
        form_layout.setSpacing(12)
        
        form_layout.addWidget(QLabel("学号:"), 0, 0)
        self.edit_student_id = QLineEdit()
        self.edit_student_id.setPlaceholderText("请输入学号")
        self.edit_student_id.setStyleSheet(get_line_edit_style())
        form_layout.addWidget(self.edit_student_id, 0, 1)
        
        form_layout.addWidget(QLabel("姓名:"), 1, 0)
        self.edit_student_name = QLineEdit()
        self.edit_student_name.setPlaceholderText("请输入姓名")
        self.edit_student_name.setStyleSheet(get_line_edit_style())
        form_layout.addWidget(self.edit_student_name, 1, 1)
        
        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.btn_preview = QPushButton("▶ 开启预览")
        self.btn_preview.setStyleSheet(get_glass_button_style('primary'))
        self.btn_preview.setMinimumHeight(36)
        self.btn_preview.clicked.connect(self.start_preview)
        
        self.btn_stop_preview = QPushButton("⏹ 停止预览")
        self.btn_stop_preview.setStyleSheet(get_glass_button_style('danger'))
        self.btn_stop_preview.setMinimumHeight(36)
        self.btn_stop_preview.setEnabled(False)
        self.btn_stop_preview.clicked.connect(self.stop_preview)
        
        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_stop_preview)
        
        self.btn_register = QPushButton("📷 开始注册")
        self.btn_register.setStyleSheet(get_glass_button_style('success'))
        self.btn_register.setMinimumHeight(42)
        self.btn_register.clicked.connect(self.start_registration)
        
        form_layout.addLayout(btn_layout, 2, 0, 1, 2)
        form_layout.addWidget(self.btn_register, 3, 0, 1, 2)
        
        # 已注册学生列表 - 玻璃卡片
        list_group = QGroupBox("已注册学生列表")
        list_group.setStyleSheet(get_group_box_style())
        list_layout = QVBoxLayout(list_group)
        list_layout.setContentsMargins(16, 16, 16, 16)
        
        self.student_table = QTableWidget()
        self.student_table.setColumnCount(3)
        self.student_table.setHorizontalHeaderLabels(["学号", "姓名", "注册时间"])
        self.student_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.student_table.setStyleSheet(get_table_style())
        
        self.btn_refresh = QPushButton("🔄 刷新列表")
        self.btn_refresh.setStyleSheet(get_secondary_button_style())
        self.btn_refresh.clicked.connect(self.refresh_student_list)
        
        self.btn_delete = QPushButton("🗑 删除选中")
        self.btn_delete.setStyleSheet(get_glass_button_style('danger'))
        self.btn_delete.clicked.connect(self.delete_student)
        
        list_layout.addWidget(self.student_table)
        
        btn_layout2 = QHBoxLayout()
        btn_layout2.addWidget(self.btn_refresh)
        btn_layout2.addWidget(self.btn_delete)
        list_layout.addLayout(btn_layout2)
        
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
        layout.addWidget(form_group)
        layout.addWidget(list_group)
        layout.addWidget(log_group)
        
        return panel
    
    def start_preview(self):
        """开启预览"""
        if self.registration_thread is not None and self.registration_thread.isRunning():
            self.log("预览已在运行，请先停止预览再重开")
            return
        
        self.log("正在开启摄像头预览...")
        
        self.registration_thread = RegistrationThread(0)
        self.registration_thread.frame_processed.connect(self.on_frame_processed)
        self.registration_thread.liveness_result.connect(self.on_liveness_result)
        self.registration_thread.registration_complete.connect(self.on_registration_complete)
        
        self.registration_thread.start()
        
        self.btn_preview.setEnabled(False)
        self.btn_stop_preview.setEnabled(True)
        self.btn_register.setEnabled(True)
        
        self.log("预览已开启，请面对摄像头")
    
    def stop_preview(self):
        """停止预览并释放摄像头（切换 Tab 或退出前务必调用，避免与其它功能抢摄像头）"""
        if self.registration_thread is not None:
            self.registration_thread.stop()
            self.registration_thread = None
        self.btn_preview.setEnabled(True)
        self.btn_stop_preview.setEnabled(False)
        self.btn_register.setEnabled(False)
        self.log("预览已停止")
    
    def start_registration(self):
        """开始注册"""
        student_id = self.edit_student_id.text().strip()
        student_name = self.edit_student_name.text().strip()
        
        if not student_id or not student_name:
            QMessageBox.warning(self, "输入错误", "请填写学号和姓名")
            return
        
        if self.registration_thread is None or not self.registration_thread.isRunning():
            QMessageBox.warning(self, "错误", "请先开启预览")
            return
        
        self.log(f"开始注册: {student_name} ({student_id})")
        self.log("请面对摄像头，保持自然表情...")
        
        self.registration_thread.start_registration(student_id, student_name)
    
    def refresh_student_list(self):
        """刷新学生列表"""
        try:
            db = Database(DB_PATH)
            students = db.get_all_students()
            
            self.student_table.setRowCount(0)
            for student in students:
                row = self.student_table.rowCount()
                self.student_table.insertRow(row)
                self.student_table.setItem(row, 0, QTableWidgetItem(student['id']))
                self.student_table.setItem(row, 1, QTableWidgetItem(student['name']))
                self.student_table.setItem(row, 2, QTableWidgetItem(student['register_time']))
            
            self.log(f"已加载 {len(students)} 名学生")
            
        except Exception as e:
            self.log(f"刷新列表失败: {e}")
    
    def delete_student(self):
        """删除选中学生"""
        selected = self.student_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "提示", "请先选择要删除的学生")
            return
        
        student_id = self.student_table.item(selected, 0).text()
        student_name = self.student_table.item(selected, 1).text()
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除学生 {student_name} ({student_id}) 吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                db = Database(DB_PATH)
                db.delete_student(student_id)
                self.log(f"已删除学生: {student_name}")
                self.refresh_student_list()
            except Exception as e:
                self.log(f"删除失败: {e}")
    
    def on_frame_processed(self, frame, info):
        """帧处理回调"""
        rgb_frame = np.ascontiguousarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        h, w = rgb_frame.shape[:2]
        bytes_per_line = int(rgb_frame.strides[0])
        
        q_image = QImage(
            rgb_frame.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)
        
        # 更新人脸状态
        if info['face_detected']:
            self.lbl_face_status.setText("✓ 已检测")
            self.lbl_face_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_face_status.setText("✗ 未检测")
            self.lbl_face_status.setStyleSheet("color: #f44336;")
    
    def on_liveness_result(self, is_live, score):
        """活体检测结果回调"""
        if is_live:
            self.lbl_liveness_status.setText(f"✓ 真人 ({score:.2f})")
            self.lbl_liveness_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.lbl_liveness_status.setText(f"✗ 非活体 ({score:.2f})")
            self.lbl_liveness_status.setStyleSheet("color: #f44336;")
        
        # 更新进度条
        self.liveness_bar.setValue(int(score * 100))
    
    def on_registration_complete(self, success, message):
        """注册完成回调"""
        if success:
            self.log(f"✓ {message}")
            QMessageBox.information(self, "注册成功", message)
            self.refresh_student_list()
            
            # 清空输入
            self.edit_student_id.clear()
            self.edit_student_name.clear()
        else:
            self.log(f"✗ {message}")
            QMessageBox.warning(self, "注册失败", message)
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_preview()
        event.accept()
