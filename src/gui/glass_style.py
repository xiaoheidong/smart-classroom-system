"""
微妙的玻璃拟态（Subtle Glassmorphism）样式模块
为智慧教室系统提供统一的现代化UI风格
"""

# 颜色配置 - 柔和专业配色
COLORS = {
    # 背景色
    'bg_primary': '#F5F7FA',        # 主背景 - 高级浅灰白
    'bg_secondary': '#E8ECF0',      # 次级背景
    'bg_card': 'rgba(255, 255, 255, 0.75)',  # 玻璃卡片背景
    
    # 强调色
    'sky_blue': '#5B9BD5',          # 柔和天蓝色 - 主要操作
    'sky_blue_light': '#7AB8E8',    # 天蓝浅色 - 悬停
    'coral': '#E07A5F',             # 柔和珊瑚色 - 停止/警告
    'coral_light': '#E89A85',       # 珊瑚浅色 - 悬停
    
    # 文字色
    'text_primary': '#2C3E50',      # 主文字 - 深蓝灰
    'text_secondary': '#5D6D7E',    # 次级文字
    'text_muted': '#95A5A6',        # 弱化文字
    
    # 状态色
    'success': '#52C4A0',           # 成功 - 柔和绿
    'warning': '#F4D03F',           # 警告 - 柔和黄
    'danger': '#E07A5F',            # 危险 - 珊瑚色
    'info': '#5B9BD5',              # 信息 - 天蓝色
    
    # 边框与分割线
    'border_light': 'rgba(255, 255, 255, 0.5)',
    'border_dark': 'rgba(0, 0, 0, 0.08)',
    'divider': 'rgba(0, 0, 0, 0.06)',
    'silver': '#C0C0C0',            # 银色边框
    
    # 视频区
    'video_bg': '#1A1D21',          # 深色视频背景
}


def get_glass_card_style():
    """玻璃卡片样式 - 磨砂玻璃质感"""
    return f"""
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border_light']};
        border-radius: 12px;
        padding: 16px;
    """


def get_main_window_style():
    """主窗口样式 - 浅灰白背景"""
    return f"""
        QMainWindow, QWidget {{
            background-color: {COLORS['bg_primary']};
        }}
    """


def get_video_panel_style():
    """视频显示区样式 - 深色带银色边框"""
    return f"""
        QLabel {{
            background-color: {COLORS['video_bg']};
            border: 2px solid {COLORS['silver']};
            border-radius: 8px;
        }}
    """


def get_glass_button_style(color_type='primary'):
    """
    玻璃质感按钮样式
    color_type: 'primary' (天蓝), 'danger' (珊瑚), 'success' (绿)
    """
    if color_type == 'primary':
        bg_color = COLORS['sky_blue']
        hover_color = COLORS['sky_blue_light']
    elif color_type == 'danger':
        bg_color = COLORS['coral']
        hover_color = COLORS['coral_light']
    elif color_type == 'success':
        bg_color = COLORS['success']
        hover_color = '#6DD4B2'
    else:
        bg_color = COLORS['sky_blue']
        hover_color = COLORS['sky_blue_light']
    
    return f"""
        QPushButton {{
            background-color: {bg_color};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 13px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: {hover_color};
        }}
        QPushButton:pressed {{
            background-color: {bg_color};
        }}
        QPushButton:disabled {{
            background-color: #BDC3C7;
            color: #ECF0F1;
        }}
    """


def get_secondary_button_style():
    """次要按钮样式 - 浅灰色"""
    return f"""
        QPushButton {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_dark']};
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: #D5DBDB;
        }}
    """


def get_group_box_style(title=""):
    """玻璃质感分组框样式"""
    return f"""
        QGroupBox {{
            background-color: {COLORS['bg_card']};
            border: 1px solid {COLORS['border_light']};
            border-radius: 12px;
            margin-top: 12px;
            padding-top: 8px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px;
            color: {COLORS['text_secondary']};
            font-size: 12px;
        }}
    """


def get_table_style():
    """现代数据表格样式"""
    return f"""
        QTableWidget {{
            background-color: transparent;
            border: none;
            gridline-color: {COLORS['divider']};
            font-size: 12px;
        }}
        QTableWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {COLORS['divider']};
        }}
        QTableWidget::item:selected {{
            background-color: {COLORS['sky_blue']};
            color: white;
        }}
        QHeaderView::section {{
            background-color: transparent;
            color: {COLORS['text_secondary']};
            padding: 10px 8px;
            border: none;
            border-bottom: 2px solid {COLORS['divider']};
            font-weight: 600;
            font-size: 11px;
        }}
        QTableCornerButton::section {{
            background-color: transparent;
            border: none;
        }}
    """


def get_text_edit_style(read_only=False):
    """文本编辑区样式 - 玻璃质感"""
    if read_only:
        return f"""
            QTextEdit {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_light']};
                border-radius: 12px;
                padding: 12px;
                color: {COLORS['text_primary']};
                font-size: 13px;
                line-height: 1.5;
            }}
        """
    else:
        return f"""
            QTextEdit {{
                background-color: white;
                border: 1px solid {COLORS['border_dark']};
                border-radius: 12px;
                padding: 12px;
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QTextEdit:focus {{
                border: 2px solid {COLORS['sky_blue']};
            }}
        """


def get_line_edit_style():
    """输入框样式 - 高级干净带内阴影效果"""
    return f"""
        QLineEdit {{
            background-color: white;
            border: 1px solid {COLORS['border_dark']};
            border-radius: 8px;
            padding: 10px 12px;
            color: {COLORS['text_primary']};
            font-size: 13px;
        }}
        QLineEdit:focus {{
            border: 2px solid {COLORS['sky_blue']};
        }}
        QLineEdit::placeholder {{
            color: {COLORS['text_muted']};
        }}
    """


def get_combo_box_style():
    """下拉框样式 - 干净扁平"""
    return f"""
        QComboBox {{
            background-color: white;
            border: 1px solid {COLORS['border_dark']};
            border-radius: 8px;
            padding: 8px 12px;
            color: {COLORS['text_primary']};
            font-size: 13px;
            min-width: 120px;
        }}
        QComboBox:hover {{
            border: 1px solid {COLORS['sky_blue']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: white;
            border: 1px solid {COLORS['border_dark']};
            border-radius: 8px;
            selection-background-color: {COLORS['sky_blue']};
        }}
    """


def get_progress_bar_style():
    """进度条样式 - 细致现代"""
    return f"""
        QProgressBar {{
            background-color: {COLORS['bg_secondary']};
            border: none;
            border-radius: 6px;
            height: 8px;
            text-align: center;
            font-size: 10px;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['sky_blue']};
            border-radius: 6px;
        }}
    """


def get_label_style(label_type='normal'):
    """标签样式"""
    if label_type == 'title':
        return f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: 600;"
    elif label_type == 'stat_blue':
        return f"color: {COLORS['sky_blue']}; font-size: 18px; font-weight: bold;"
    elif label_type == 'stat_coral':
        return f"color: {COLORS['coral']}; font-size: 18px; font-weight: bold;"
    elif label_type == 'muted':
        return f"color: {COLORS['text_muted']}; font-size: 11px;"
    else:
        return f"color: {COLORS['text_secondary']}; font-size: 12px;"


def get_splitter_style():
    """分割器样式"""
    return f"""
        QSplitter::handle {{
            background-color: {COLORS['divider']};
        }}
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        QSplitter::handle:vertical {{
            height: 2px;
        }}
    """


def get_scrollbar_style():
    """滚动条样式 - 极简"""
    return f"""
        QScrollBar:vertical {{
            background-color: transparent;
            width: 8px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {COLORS['border_dark']};
            border-radius: 4px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {COLORS['text_muted']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
    """


def apply_glass_style(widget):
    """
    为整个窗口应用玻璃拟态风格
    在窗口初始化时调用
    """
    style_sheet = f"""
        /* 基础 */
        QWidget {{
            background-color: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
        }}
        
        /* 滚动条 */
        {get_scrollbar_style()}
        
        /* 分割器 */
        {get_splitter_style()}
    """
    widget.setStyleSheet(style_sheet)


def create_glass_card(parent, title=""):
    """
    创建玻璃质感卡片容器
    返回 QGroupBox 实例
    """
    from PyQt5.QtWidgets import QGroupBox, QVBoxLayout
    
    card = QGroupBox(title, parent)
    card.setStyleSheet(get_group_box_style(title))
    
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)
    
    return card, layout
