"""
智能问答 Tab - 玻璃拟态风格
DeepSeek API + 当日考勤与异常抓拍上下文（仅文本，不传图片字节）。
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QCheckBox,
    QGroupBox,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_PATH, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
from utils.database import Database
from utils.llm_context import build_daily_context
from utils.deepseek_client import chat_completion
from gui.glass_style import (
    apply_glass_style, get_glass_button_style, get_group_box_style,
    get_text_edit_style, get_label_style, get_secondary_button_style
)


SYSTEM_PROMPT = """你是「智慧教室」本地系统的助手。用户会提供一段【当日数据】（考勤摘要与异常行为抓拍列表，含图片相对路径）。
请遵守：
1. 仅根据【当日数据】与用户问题作答；数据中没有的信息请明确说「数据中未记录」或「无法从当前数据推断」，不要编造学号、姓名或抓拍次数。
2. 抓拍由算法自动产生，可能存在误判；表述时建议用「记录显示」「系统抓拍」等措辞。
3. 若问及照片，可依据数据中的「本地路径」告诉用户在本机打开查看；你本身无法打开文件。
4. 回答简洁、分点列出为宜。"""


class _LLMWorker(QThread):
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, messages: list):
        super().__init__()
        self._messages = messages

    def run(self) -> None:
        ok, text = chat_completion(
            self._messages,
            DEEPSEEK_API_KEY,
            DEEPSEEK_API_BASE,
            DEEPSEEK_MODEL,
            timeout=120.0,
            temperature=0.35,
        )
        if ok:
            self.finished_ok.emit(text)
        else:
            self.finished_err.emit(text)


class LLMAssistantTab(QWidget):
    """DeepSeek 问答页，独立于摄像头与检测线程。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = Database(DB_PATH)
        self._worker: Optional[_LLMWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        """构建UI - 玻璃拟态风格，强调文本交互"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # 应用玻璃风格
        apply_glass_style(self)

        # 使用说明 - 精美层级感
        hint_container = QGroupBox("使用说明")
        hint_container.setStyleSheet(get_group_box_style())
        hint_layout = QVBoxLayout(hint_container)
        hint_layout.setContentsMargins(16, 12, 16, 12)
        
        hint_title = QLabel("📝 智能问答助手")
        hint_title.setStyleSheet(get_label_style('title'))
        
        hint_desc = QLabel(
            "在系统环境变量中设置 DEEPSEEK_API_KEY 后重启本程序。"
            "本页仅将「今日考勤 + 今日异常抓拍」以文字形式发给模型，不上传图片文件。"
        )
        hint_desc.setWordWrap(True)
        hint_desc.setStyleSheet(get_label_style('muted'))
        
        hint_layout.addWidget(hint_title)
        hint_layout.addWidget(hint_desc)
        layout.addWidget(hint_container)

        # 选项
        opt = QHBoxLayout()
        self.chk_today = QCheckBox("在提问中附带今日数据（考勤摘要 + 抓拍列表）")
        self.chk_today.setChecked(True)
        self.chk_today.setStyleSheet(f"color: {get_label_style('normal').split(':')[1].strip()}")
        opt.addWidget(self.chk_today)
        opt.addStretch()
        layout.addLayout(opt)

        # 输入问题区 - 大型玻璃质感文本框
        g_in = QGroupBox("输入问题")
        g_in.setStyleSheet(get_group_box_style())
        in_layout = QVBoxLayout(g_in)
        in_layout.setContentsMargins(16, 16, 16, 16)
        
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText(
            "例如：今天有多少条睡觉抓拍？玩手机的情况如何？请根据数据总结今日课堂异常概况。"
        )
        self.input_edit.setMinimumHeight(140)
        self.input_edit.setStyleSheet(get_text_edit_style(read_only=False))
        in_layout.addWidget(self.input_edit)
        layout.addWidget(g_in, stretch=1)

        # 控制按钮行 - 横向排列
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        self.btn_send = QPushButton("✉ 发送给 DeepSeek")
        self.btn_send.setStyleSheet(get_glass_button_style('primary'))
        self.btn_send.setMinimumHeight(44)
        self.btn_send.setMinimumWidth(160)
        
        self.btn_clear = QPushButton("清空回复")
        self.btn_clear.setStyleSheet(get_secondary_button_style())
        self.btn_clear.setMinimumHeight(36)
        self.btn_clear.clicked.connect(self._clear_output)
        
        btn_row.addStretch()
        btn_row.addWidget(self.btn_send)
        btn_row.addWidget(self.btn_clear)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        self.btn_send.clicked.connect(self._on_send)

        # 模型回复区 - 大型玻璃质感文本框
        g_out = QGroupBox("模型回复")
        g_out.setStyleSheet(get_group_box_style())
        out_layout = QVBoxLayout(g_out)
        out_layout.setContentsMargins(16, 16, 16, 16)
        
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setMinimumHeight(280)
        self.output_edit.setStyleSheet(get_text_edit_style(read_only=True))
        out_layout.addWidget(self.output_edit)
        layout.addWidget(g_out, stretch=2)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(get_label_style('muted'))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def _clear_output(self) -> None:
        self.output_edit.clear()

    def _on_send(self) -> None:
        if not (DEEPSEEK_API_KEY or "").strip():
            QMessageBox.warning(
                self,
                "未配置 API Key",
                "请设置环境变量 DEEPSEEK_API_KEY 后重新启动程序。\n"
                "（不要将密钥写入代码或上传到公开仓库）",
            )
            return

        q = self.input_edit.toPlainText().strip()
        if not q:
            QMessageBox.information(self, "提示", "请先输入问题。")
            return

        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "请稍候", "上一请求尚未结束。")
            return

        context = ""
        if self.chk_today.isChecked():
            try:
                context = build_daily_context(self._db)
            except Exception as e:
                QMessageBox.warning(self, "数据读取失败", str(e))
                return

        user_block = q
        if context:
            user_block = "【当日数据】\n" + context + "\n\n【用户问题】\n" + q

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ]

        self.btn_send.setEnabled(False)
        self.status_label.setText("请求中…（最长约 2 分钟）")

        self._worker = _LLMWorker(messages)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished_err.connect(self._on_err)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_ok(self, text: str) -> None:
        self.output_edit.setPlainText(text)

    def _on_err(self, text: str) -> None:
        self.output_edit.setPlainText("【错误】\n" + text)

    def _on_worker_done(self) -> None:
        self.btn_send.setEnabled(True)
        self.status_label.setText("就绪")

    def stop_worker(self) -> None:
        """主窗口退出时调用：尽量结束未完成的请求。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
