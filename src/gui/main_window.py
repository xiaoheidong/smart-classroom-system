"""
主窗口
集成功能 Tab：作弊检测、动态点名、人脸注册、智能问答（DeepSeek）
"""
import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QMenuBar, QMenu, QAction, QMessageBox, QStatusBar,
    QLabel, QHBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR
from .cheating_detection_tab import CheatingDetectionTab
from .attendance_tab import AttendanceTab
from .face_registration_tab import FaceRegistrationTab
from .llm_assistant_tab import LLMAssistantTab


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧教室系统 - 基于深度学习的课堂行为分析与考勤")
        self.setMinimumSize(1400, 900)
        
        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon(os.path.join(BASE_DIR, 'assets', 'icon.png')))
        
        self.init_ui()
        self.init_menu()
        self.init_statusbar()
        
        # 检查模型文件
        self.check_models()
    
    def init_ui(self):
        """初始化UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 创建Tab页
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)
        
        # 设置Tab样式
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: #f5f5f5;
            }
            QTabBar::tab {
                padding: 12px 24px;
                margin: 2px;
                background: #e0e0e0;
                border: 1px solid #ccc;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover {
                background: #f0f0f0;
            }
        """)
        
        # 功能 Tab（智能问答独立线程，不占用摄像头）
        self.tab_cheating = CheatingDetectionTab()
        self.tab_attendance = AttendanceTab()
        self.tab_registration = FaceRegistrationTab()
        self.tab_llm = LLMAssistantTab()

        self.tab_widget.addTab(self.tab_cheating, "🚨 作弊检测")
        self.tab_widget.addTab(self.tab_attendance, "📋 动态点名")
        self.tab_widget.addTab(self.tab_registration, "👤 人脸注册")
        self.tab_widget.addTab(self.tab_llm, "🤖 智能问答")
        
        layout.addWidget(self.tab_widget)
        
        # 三个 Tab 会争用同一摄像头与显存：切换时停止其它页的后台线程，避免 OpenCV/Qt 崩溃
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        """仅保留当前 Tab 的视频与推理，防止多路同时打开摄像头导致进程退出。"""
        try:
            if index != 0 and hasattr(self.tab_cheating, "stop_detection"):
                self.tab_cheating.stop_detection()
            if index != 1 and hasattr(self.tab_attendance, "stop_attendance"):
                self.tab_attendance.stop_attendance()
            if index != 2 and hasattr(self.tab_registration, "stop_preview"):
                self.tab_registration.stop_preview()
            # index 3：智能问答，无摄像头线程
        except Exception as e:
            print(f"[MainWindow] 切换 Tab 时停止后台任务: {e}")

    def init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        # 打开视频文件
        open_action = QAction('打开视频文件(&O)', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_video_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction('退出(&Q)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具(&T)')
        
        # 刷新学生列表
        refresh_action = QAction('刷新学生列表(&R)', self)
        refresh_action.triggered.connect(self.refresh_students)
        tools_menu.addAction(refresh_action)
        
        # 清理旧记录
        cleanup_action = QAction('清理旧考勤记录(&C)', self)
        cleanup_action.triggered.connect(self.cleanup_records)
        tools_menu.addAction(cleanup_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 关于
        about_action = QAction('关于(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_statusbar(self):
        """初始化状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # 状态标签
        self.lbl_status = QLabel("就绪")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_device = QLabel("设备: --")
        
        self.statusbar.addWidget(self.lbl_status, 1)
        self.statusbar.addWidget(self.lbl_fps)
        self.statusbar.addWidget(self.lbl_device)
        
        # 检查GPU状态
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.lbl_device.setText(f"GPU: {device_name}")
            else:
                self.lbl_device.setText("GPU: 不可用，使用CPU")
        except:
            self.lbl_device.setText("GPU: 检测失败")
    
    def check_models(self):
        """检查模型文件（作弊检测必需；dlib 人脸模型为可选，无则使用 facenet-pytorch）"""
        from config import (
            ACTION_CLASSIFIER_MODEL,
            PRETRAINED_DIR,
            TRAINED_DIR,
            YOLOV11N_MODEL,
        )
        
        missing_models = []
        
        # 行为检测链路必需
        yolo_path = os.path.join(PRETRAINED_DIR, YOLOV11N_MODEL)
        if not os.path.exists(yolo_path):
            missing_models.append(YOLOV11N_MODEL)

        action_model = os.path.join(TRAINED_DIR, ACTION_CLASSIFIER_MODEL)
        if not os.path.exists(action_model):
            missing_models.append(f"{ACTION_CLASSIFIER_MODEL} (需要训练)")
        
        if missing_models:
            msg = "以下模型文件缺失:\n\n"
            for m in missing_models:
                msg += f"  - {m}\n"
            msg += "\n请按 README.md 中的说明下载或训练模型。"
            
            QMessageBox.warning(self, "模型文件缺失", msg)
    
    def open_video_file(self):
        """打开视频文件"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*.*)"
        )
        
        if filename:
            self.tab_cheating.set_video_source(filename)
            self.tab_widget.setCurrentWidget(self.tab_cheating)
            self.statusbar.showMessage(f"已选择视频文件: {filename}", 5000)
            QMessageBox.information(
                self,
                "提示",
                f"已选择视频文件:\n{filename}\n\n现在可在“作弊检测”页点击“开始检测”。",
            )
    
    def refresh_students(self):
        """刷新学生列表"""
        if hasattr(self.tab_registration, "refresh_student_list"):
            self.tab_registration.refresh_student_list()
        if hasattr(self.tab_attendance, "refresh_lists"):
            self.tab_attendance.refresh_lists()
        self.statusbar.showMessage("学生列表已刷新", 3000)
    
    def cleanup_records(self):
        """清理旧记录"""
        reply = QMessageBox.question(
            self, "确认清理",
            "确定要清理30天前的考勤记录吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                from utils.database import Database
                from config import DB_PATH
                
                db = Database(DB_PATH)
                db.clear_old_attendance(days=30)
                
                QMessageBox.information(self, "完成", "旧考勤记录已清理")
                self.statusbar.showMessage("记录已清理", 3000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"清理失败: {e}")
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 智慧教室系统",
            """<h2>智慧教室系统 v1.0</h2>
            <p>基于深度学习的课堂行为分析与考勤系统</p>
            <hr>
            <p><b>核心功能：</b></p>
            <ul>
                <li>实时课堂行为检测（睡觉、使用手机等异常行为）</li>
                <li>基于人脸识别的动态点名</li>
                <li>静默活体检测 + 人脸注册</li>
                <li>智能问答（DeepSeek，基于当日考勤与抓拍文本上下文）</li>
            </ul>
            <p><b>技术栈：</b></p>
            <ul>
                <li>YOLOv11 人体检测</li>
                <li>ResNet18 行为分类</li>
                <li>dlib 人脸识别</li>
                <li>PyQt5 图形界面</li>
                <li>PyTorch 深度学习框架</li>
            </ul>
            <hr>
            <p>毕业设计项目 | 适配RTX 3050 4GB显存</p>
            """
        )
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止所有后台线程
        self.tab_cheating.stop_detection()
        if hasattr(self.tab_attendance, "stop_attendance"):
            self.tab_attendance.stop_attendance()
        if hasattr(self.tab_registration, 'registration_thread') and self.tab_registration.registration_thread:
            self.tab_registration.registration_thread.stop()
        if hasattr(self, "tab_llm") and hasattr(self.tab_llm, "stop_worker"):
            self.tab_llm.stop_worker()

        event.accept()


def main():
    """主函数"""
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
