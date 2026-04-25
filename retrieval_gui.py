import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime

import pandas as pd
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import dinov2_patch_retrieval_fast as retrieval_core


MAX_DISPLAY_COUNT = 14
THUMB_SIZE = 240
REFERENCE_PREVIEW_SIZE = 280
CARD_MIN_WIDTH = 240
CARD_GAP = 16
SIDEBAR_WIDTH = 380
LOG_PANEL_HEIGHT = 170
WINDOW_MIN_WIDTH = 1440
WINDOW_MIN_HEIGHT = 900

COLOR_BG = "#edf3f8"
COLOR_PANEL = "#ffffff"
COLOR_SOFT_PANEL = "#f7fafc"
COLOR_TEXT = "#16212b"
COLOR_MUTED = "#66768a"
COLOR_PRIMARY = "#1d4ed8"
COLOR_PRIMARY_ACTIVE = "#1e40af"
COLOR_SUCCESS = "#0f766e"
COLOR_BORDER = "#d9e3ee"
COLOR_ACCEPTED = "#dcfce7"
COLOR_ACCEPTED_TEXT = "#166534"
COLOR_REJECTED = "#fee2e2"
COLOR_REJECTED_TEXT = "#991b1b"
COLOR_UNLABELED = "#eef2f7"
COLOR_UNLABELED_TEXT = "#4b5563"
COLOR_STATUS_BG = "#e8f0ff"
COLOR_STATUS_TEXT = "#1e3a8a"

GUI_HISTORY_PATH = os.path.join(retrieval_core.output_dir, "retrieval_gui_history.json")


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"retrieval_gui_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger = logging.getLogger("retrieval_gui")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


def default_target_dir(reference_image_path):
    return os.path.dirname(os.path.abspath(reference_image_path))


def filter_display_results(
    result_df,
    excluded_paths=None,
    max_display_count=MAX_DISPLAY_COUNT,
):
    excluded_paths = {os.path.abspath(path) for path in (excluded_paths or set())}
    if excluded_paths and not result_df.empty:
        filtered_df = result_df[
            ~result_df["gallery_image_path"].map(lambda path: os.path.abspath(path) in excluded_paths)
        ].copy()
    else:
        filtered_df = result_df.copy()

    if filtered_df.empty:
        return filtered_df

    return filtered_df.head(max_display_count).copy()


def load_gui_history():
    default_history = {"copy_records": [], "last_deleted_count": 0, "feedback_by_reference": {}}
    if not os.path.isfile(GUI_HISTORY_PATH):
        return default_history
    try:
        with open(GUI_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default_history
    history.setdefault("copy_records", [])
    history.setdefault("last_deleted_count", 0)
    history.setdefault("feedback_by_reference", {})
    return history


def save_gui_history(history):
    os.makedirs(os.path.dirname(GUI_HISTORY_PATH), exist_ok=True)
    with open(GUI_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def has_copy_record(history, source_path, target_path):
    source_abs = os.path.abspath(source_path)
    target_abs = os.path.abspath(target_path)
    for record in history.get("copy_records", []):
        if (
            os.path.abspath(record.get("source_path", "")) == source_abs
            and os.path.abspath(record.get("target_path", "")) == target_abs
        ):
            return True
    return False


def make_unique_copy_path(target_dir, source_path):
    return os.path.join(target_dir, os.path.basename(source_path))


def normalize_abs_path(path):
    if not path:
        return ""
    return os.path.normcase(os.path.abspath(path))


def is_same_or_child_path(path, root):
    normalized_path = normalize_abs_path(path)
    normalized_root = normalize_abs_path(root)
    if not normalized_path or not normalized_root:
        return False
    try:
        return os.path.commonpath([normalized_path, normalized_root]) == normalized_root
    except ValueError:
        return False


def pil_to_qpixmap(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    qimage = ImageQt(image)
    return QPixmap.fromImage(qimage)


class RetrievalWorker(QThread):
    finished_ok = Signal(dict, dict)
    failed = Signal(str, str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            retrieval_core.reference_image_path = self.params["reference_path"]
            retrieval_core.gallery_dir = self.params["gallery_path"]
            retrieval_core.gallery_recursive = self.params["recursive"]
            retrieval_core.retrieval_topk = self.params["topk"]
            result = retrieval_core.retrieve_with_feedback(
                reference_path=self.params["reference_path"],
                feedback_labels=self.params["feedback_labels"],
                hidden_paths=self.params["hidden_paths"],
                topk=self.params["topk"],
                write_outputs=True,
            )
            self.finished_ok.emit(result, self.params)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


class PreviewDialog(QDialog):
    def __init__(self, image_path, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 920)
        self._image = Image.open(image_path).convert("RGB")
        self._zoom = 1.0

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.info_label = QLabel(image_path)
        self.info_label.setWordWrap(True)
        zoom_in = QPushButton("放大")
        zoom_out = QPushButton("缩小")
        zoom_reset = QPushButton("重置")
        zoom_in.clicked.connect(lambda: self.adjust_zoom(1.2))
        zoom_out.clicked.connect(lambda: self.adjust_zoom(1 / 1.2))
        zoom_reset.clicked.connect(self.reset_zoom)
        toolbar.addWidget(self.info_label, 1)
        toolbar.addWidget(zoom_out)
        toolbar.addWidget(zoom_in)
        toolbar.addWidget(zoom_reset)
        layout.addLayout(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area, 1)
        self.render_image()

    def adjust_zoom(self, factor):
        self._zoom = max(0.3, min(6.0, self._zoom * factor))
        self.render_image()

    def reset_zoom(self):
        self._zoom = 1.0
        self.render_image()

    def render_image(self):
        width = max(1, int(self._image.width * self._zoom))
        height = max(1, int(self._image.height * self._zoom))
        resized = self._image.resize((width, height), Image.BILINEAR)
        self.image_label.setPixmap(pil_to_qpixmap(resized))


class CardEventFilter(QObject):
    clicked = Signal()
    double_clicked = Signal()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.clicked.emit()
            return True
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            self.double_clicked.emit()
            return True
        return super().eventFilter(watched, event)


class ResultCard(QFrame):
    def __init__(self, row, feedback_label, thumb_size, toggle_callback, preview_callback, parent=None):
        super().__init__(parent)
        self.row = row
        self.image_path = row["gallery_image_path"]
        self.feedback_label = feedback_label
        self.toggle_callback = toggle_callback
        self.preview_callback = preview_callback
        self.thumb_size = thumb_size

        self.setObjectName("resultCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(CARD_MIN_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.rank_label = QLabel(f"#{int(row['rank'])}")
        self.rank_label.setObjectName("rankLabel")
        rerank_score = float(row["rerank_score"]) if pd.notna(row.get("rerank_score")) else float(row["similarity_score"])
        self.score_badge = QLabel(f"{rerank_score:.4f}")
        self.score_badge.setObjectName("scoreBadge")
        top_row.addWidget(self.rank_label)
        top_row.addStretch(1)
        top_row.addWidget(self.score_badge)
        layout.addLayout(top_row)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        image = Image.open(self.image_path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.BILINEAR)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(pil_to_qpixmap(image))
        self.image_label.setMinimumHeight(thumb_size)
        layout.addWidget(self.image_label)

        global_score = row.get("global_score")
        local_score = row.get("local_score")
        score_parts = [f"base {float(row['similarity_score']):.4f}", f"rerank {rerank_score:.4f}"]
        if pd.notna(global_score):
            score_parts.append(f"global {float(global_score):.4f}")
        if pd.notna(local_score):
            score_parts.append(f"local {float(local_score):.4f}")
        self.meta_label = QLabel(" | ".join(score_parts))
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        self.name_label = QLabel(os.path.basename(self.image_path))
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)

        self._click_filter = CardEventFilter(self)
        self._click_filter.clicked.connect(self.on_toggle)
        self._click_filter.double_clicked.connect(self.on_preview)
        for widget in (self, self.rank_label, self.score_badge, self.status_label, self.image_label, self.meta_label, self.name_label):
            widget.installEventFilter(self._click_filter)

        self.apply_state()

    def set_feedback_label(self, label):
        self.feedback_label = label
        self.apply_state()

    def on_toggle(self):
        self.toggle_callback(self)

    def on_preview(self):
        self.preview_callback(self)

    def apply_state(self):
        if self.feedback_label == "accepted":
            bg = COLOR_ACCEPTED
            fg = COLOR_ACCEPTED_TEXT
            status = "Accepted"
        elif self.feedback_label == "rejected":
            bg = COLOR_REJECTED
            fg = COLOR_REJECTED_TEXT
            status = "Rejected"
        else:
            bg = COLOR_UNLABELED
            fg = COLOR_UNLABELED_TEXT
            status = "Unlabeled"

        self.status_label.setText(status)
        self.setStyleSheet(
            f"""
            QFrame#resultCard {{
                background: {bg};
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
            }}
            QLabel {{
                color: {COLOR_TEXT};
                background: transparent;
            }}
            QLabel#scoreBadge {{
                background: {COLOR_STATUS_BG};
                color: {COLOR_STATUS_TEXT};
                border-radius: 10px;
                padding: 6px 10px;
                font-weight: 700;
            }}
            """
        )
        self.status_label.setStyleSheet(f"color: {fg}; font-weight: 700;")
        self.meta_label.setStyleSheet(f"color: {fg};")
        self.name_label.setStyleSheet(f"color: {fg}; font-weight: 600;")


class RetrievalMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINOv2 图像反查工作台")
        self.resize(1600, 1020)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self.logger, self.log_path = setup_logger(retrieval_core.output_dir)
        self.gui_history = load_gui_history()
        self.result_df = pd.DataFrame()
        self.display_df = pd.DataFrame()
        self.card_widgets = []
        self.worker = None
        self.pending_rerun_reason = None
        self.pending_rerun_timer = QTimer(self)
        self.pending_rerun_timer.setSingleShot(True)
        self.pending_rerun_timer.timeout.connect(self.start_retrieval)

        self.reference_path = retrieval_core.reference_image_path
        self.gallery_path = retrieval_core.gallery_dir
        self.target_path = default_target_dir(retrieval_core.reference_image_path)
        self.recursive_enabled = retrieval_core.gallery_recursive
        self.auto_rerun_enabled = True
        self.display_limit = MAX_DISPLAY_COUNT

        self._build_ui()
        self.refresh_reference_preview()
        self.update_history_summary()
        self.update_feedback_summary()
        self.log(f"日志文件: {self.log_path}")
        self.show_startup_announcement()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("panel")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)
        title_col = QVBoxLayout()
        title = QLabel("图像反查工作台")
        title.setObjectName("heroTitle")
        subtitle = QLabel("参考图检索、人工筛选、反馈收敛和历史去重。")
        subtitle.setObjectName("heroSub")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        self.status_badge = QLabel("等待检索")
        self.status_badge.setObjectName("statusBadge")
        header_layout.addLayout(title_col, 1)
        header_layout.addWidget(self.status_badge, 0, Qt.AlignTop)
        root_layout.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(12)
        root_layout.addLayout(content, 1)

        sidebar_shell = QFrame()
        sidebar_shell.setObjectName("panel")
        sidebar_shell.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(sidebar_shell)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QFrame.NoFrame)
        self.sidebar_content = QWidget()
        self.sidebar_body = QVBoxLayout(self.sidebar_content)
        self.sidebar_body.setContentsMargins(18, 18, 18, 18)
        self.sidebar_body.setSpacing(16)
        self.sidebar_scroll.setWidget(self.sidebar_content)
        sidebar_layout.addWidget(self.sidebar_scroll)
        content.addWidget(sidebar_shell, 0)

        self._build_sidebar()

        right = QVBoxLayout()
        right.setSpacing(10)
        content.addLayout(right, 1)

        result_header = QFrame()
        result_header.setObjectName("panel")
        result_header_layout = QVBoxLayout(result_header)
        result_header_layout.setContentsMargins(18, 16, 18, 16)
        result_header_layout.setSpacing(12)

        header_top = QHBoxLayout()
        left_col = QVBoxLayout()
        result_title = QLabel("结果区")
        result_title.setObjectName("sectionTitle")
        result_sub = QLabel("单击卡片切换 Accepted，双击卡片查看大图。")
        result_sub.setObjectName("mutedText")
        left_col.addWidget(result_title)
        left_col.addWidget(result_sub)
        self.selection_badge = QLabel("Accepted 0 / 0 张")
        self.selection_badge.setObjectName("statusBadge")
        header_top.addLayout(left_col, 1)
        header_top.addWidget(self.selection_badge, 0, Qt.AlignTop)
        result_header_layout.addLayout(header_top)

        stats_bar = QFrame()
        stats_bar.setObjectName("softPanel")
        stats_layout = QHBoxLayout(stats_bar)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        self.history_label = QLabel("历史已排除 0 张")
        self.history_label.setObjectName("softText")
        self.feedback_label = QLabel("反馈样本 accepted 0 | rejected 0")
        self.feedback_label.setObjectName("softText")
        hint_label = QLabel("灰=未处理，绿=Accepted，红=Rejected")
        hint_label.setObjectName("softText")
        stats_layout.addWidget(self.history_label, 1)
        stats_layout.addWidget(self.feedback_label, 1)
        stats_layout.addWidget(hint_label, 0, Qt.AlignRight)
        result_header_layout.addWidget(stats_bar)

        actions = QHBoxLayout()
        self.accept_all_btn = QPushButton("当前页全Accept")
        self.accept_all_btn.clicked.connect(lambda: self.set_feedback_for_displayed("accepted"))
        self.clear_marks_btn = QPushButton("当前页清空标记")
        self.clear_marks_btn.clicked.connect(lambda: self.set_feedback_for_displayed("unlabeled"))
        self.reject_rest_btn = QPushButton("当前页未Accept设为Rejected")
        self.reject_rest_btn.clicked.connect(self.reject_current_unaccepted)
        self.copy_btn = QPushButton("复制 Accepted 图片")
        self.copy_btn.setObjectName("successButton")
        self.copy_btn.clicked.connect(self.copy_selected)
        self.clean_btn = QPushButton("删除已复制源图")
        self.clean_btn.clicked.connect(self.delete_copied_source_files)
        for button in (self.accept_all_btn, self.clear_marks_btn, self.reject_rest_btn, self.copy_btn, self.clean_btn):
            actions.addWidget(button)
        actions.addStretch(1)
        result_header_layout.addLayout(actions)
        right.addWidget(result_header, 0)

        result_panel = QFrame()
        result_panel.setObjectName("panel")
        result_panel_layout = QVBoxLayout(result_panel)
        result_panel_layout.setContentsMargins(8, 8, 8, 8)
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setFrameShape(QFrame.NoFrame)
        self.result_content = QWidget()
        self.result_grid = QGridLayout(self.result_content)
        self.result_grid.setContentsMargins(0, 0, 0, 0)
        self.result_grid.setHorizontalSpacing(CARD_GAP)
        self.result_grid.setVerticalSpacing(CARD_GAP)
        self.result_scroll.setWidget(self.result_content)
        result_panel_layout.addWidget(self.result_scroll)
        right.addWidget(result_panel, 1)

        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_panel.setMinimumHeight(LOG_PANEL_HEIGHT)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(12, 10, 12, 10)
        log_title = QLabel("日志")
        log_title.setObjectName("sectionTitle")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_box, 1)
        right.addWidget(log_panel, 0)

        self.result_scroll.viewport().installEventFilter(self)
        self._apply_styles()

    def _build_sidebar(self):
        title = QLabel("检索设置")
        title.setObjectName("sectionTitle")
        sub = QLabel("左侧区域可滚动，避免窗口高度不足时按钮和筛选项被截断。")
        sub.setObjectName("mutedText")
        self.sidebar_body.addWidget(title)
        self.sidebar_body.addWidget(sub)

        field_card = self._make_card("路径输入", soft=True)
        field_layout = field_card.layout()
        self.reference_edit = QLineEdit(self.reference_path)
        self.gallery_edit = QLineEdit(self.gallery_path)
        self.target_edit = QLineEdit(self.target_path)
        self.reference_edit.textChanged.connect(self.on_reference_changed)
        self.gallery_edit.textChanged.connect(lambda text: setattr(self, "gallery_path", text.strip()))
        self.target_edit.textChanged.connect(lambda text: setattr(self, "target_path", text.strip()))
        field_layout.addWidget(self._row_with_picker("参考图", self.reference_edit, self.pick_reference))
        field_layout.addWidget(self._row_with_picker("图库目录", self.gallery_edit, self.pick_gallery))
        field_layout.addWidget(self._row_with_picker("目标目录", self.target_edit, self.pick_target))
        self.sidebar_body.addWidget(field_card)

        action_card = self._make_card("主要操作")
        action_layout = action_card.layout()
        self.recursive_check = QCheckBox("递归扫描图库")
        self.recursive_check.setChecked(self.recursive_enabled)
        self.recursive_check.toggled.connect(lambda checked: setattr(self, "recursive_enabled", checked))
        self.auto_rerun_check = QCheckBox("反馈后自动重新检索")
        self.auto_rerun_check.setChecked(self.auto_rerun_enabled)
        self.auto_rerun_check.toggled.connect(lambda checked: setattr(self, "auto_rerun_enabled", checked))
        action_layout.addWidget(self.recursive_check)
        action_layout.addWidget(self.auto_rerun_check)

        self.start_button = QPushButton("开始检索 / 应用反馈")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_retrieval)
        action_layout.addWidget(self.start_button)

        self.side_status = QLabel("等待检索")
        self.side_status.setObjectName("mutedText")
        action_layout.addWidget(self.side_status)

        filter_card = self._make_card("结果数量", soft=True)
        filter_layout = filter_card.layout()
        self.display_limit_edit = QLineEdit(str(self.display_limit))
        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(10)
        form_grid.setVerticalSpacing(10)
        form_grid.addWidget(QLabel("显示数量"), 0, 0)
        form_grid.addWidget(self.display_limit_edit, 0, 1)
        filter_layout.addLayout(form_grid)
        apply_filter_button = QPushButton("应用数量设置")
        apply_filter_button.clicked.connect(self.apply_display_filters)
        filter_layout.addWidget(apply_filter_button)
        action_layout.addWidget(filter_card)

        hint = QLabel("单点 Accepted 只打标不重刷；批量 Rejected 和复制 Accepted 后可自动重新检索。")
        hint.setWordWrap(True)
        hint.setObjectName("mutedText")
        action_layout.addWidget(hint)
        self.sidebar_body.addWidget(action_card)

        preview_card = self._make_card("参考图预览")
        preview_layout = preview_card.layout()
        self.reference_preview_label = QLabel("未加载参考图")
        self.reference_preview_label.setAlignment(Qt.AlignCenter)
        self.reference_preview_label.setMinimumHeight(REFERENCE_PREVIEW_SIZE)
        self.reference_preview_label.setObjectName("previewLabel")
        self.reference_preview_label.mousePressEvent = lambda event: self.open_reference_preview()
        self.reference_preview_text = QLabel("未加载参考图")
        self.reference_preview_text.setWordWrap(True)
        self.reference_preview_text.setAlignment(Qt.AlignCenter)
        self.reference_preview_text.setObjectName("mutedText")
        preview_layout.addWidget(self.reference_preview_label)
        preview_layout.addWidget(self.reference_preview_text)
        self.sidebar_body.addWidget(preview_card)
        self.sidebar_body.addStretch(1)

    def _make_card(self, title, soft=False):
        card = QFrame()
        card.setObjectName("softPanel" if soft else "panel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        label = QLabel(title)
        label.setObjectName("accentLabel")
        layout.addWidget(label)
        return card

    def _row_with_picker(self, title, line_edit, pick_callback):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setObjectName("softText")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        button = QPushButton("选择")
        button.clicked.connect(pick_callback)
        row.addWidget(line_edit, 1)
        row.addWidget(button, 0)
        layout.addWidget(label)
        layout.addLayout(row)
        return container

    def _apply_styles(self):
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {COLOR_BG};
                color: {COLOR_TEXT};
                font-size: 14px;
                font-family: "Microsoft YaHei UI";
            }}
            QFrame#panel {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 16px;
            }}
            QFrame#softPanel {{
                background: {COLOR_SOFT_PANEL};
                border: none;
                border-radius: 14px;
            }}
            QLabel#heroTitle {{
                font-size: 26px;
                font-weight: 700;
            }}
            QLabel#heroSub, QLabel#mutedText, QLabel#softText {{
                color: {COLOR_MUTED};
            }}
            QLabel#sectionTitle {{
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#accentLabel {{
                color: {COLOR_PRIMARY};
                font-weight: 700;
            }}
            QLabel#statusBadge {{
                background: {COLOR_STATUS_BG};
                color: {COLOR_STATUS_TEXT};
                padding: 8px 12px;
                border-radius: 12px;
                font-weight: 700;
            }}
            QLabel#previewLabel {{
                background: {COLOR_SOFT_PANEL};
                border: 1px dashed {COLOR_BORDER};
                border-radius: 12px;
            }}
            QPushButton {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: {COLOR_PRIMARY};
            }}
            QPushButton#primaryButton {{
                background: {COLOR_PRIMARY};
                color: white;
                border-color: {COLOR_PRIMARY};
                padding: 12px 16px;
                font-weight: 700;
            }}
            QPushButton#primaryButton:hover {{
                background: {COLOR_PRIMARY_ACTIVE};
            }}
            QPushButton#successButton {{
                background: {COLOR_SUCCESS};
                color: white;
                border-color: {COLOR_SUCCESS};
            }}
            QLineEdit {{
                background: white;
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 8px 10px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QPlainTextEdit {{
                background: {COLOR_SOFT_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
                padding: 8px;
            }}
            """
        )

    def eventFilter(self, watched, event):
        if watched is self.result_scroll.viewport() and event.type() == QEvent.Resize:
            QTimer.singleShot(0, self.render_results)
        return super().eventFilter(watched, event)

    def show_startup_announcement(self):
        QMessageBox.information(
            self,
            "功能公告",
            "欢迎使用 DINOv2 图像反查工作台。\n\n"
            "当前版本主要功能：\n"
            "1. 选择参考图，对图库执行相似图检索。\n"
            "2. Accepted / Rejected 都会参与下一轮检索反馈。\n"
            "3. 检索分数融合整图相似度与 patch 局部匹配。\n"
            "4. 支持复制 Accepted、删除已复制对应的图库源图。\n",
        )

    def log(self, message):
        self.logger.info(message)
        self.log_box.appendPlainText(message)

    def set_status(self, message):
        self.status_badge.setText(message)
        self.side_status.setText(message)

    def on_reference_changed(self, text):
        self.reference_path = text.strip()
        if self.reference_path:
            self.target_path = default_target_dir(self.reference_path)
            self.target_edit.setText(self.target_path)
        self.update_feedback_summary()

    def pick_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考图", self.reference_path, "Images (*.bmp *.jpg *.jpeg *.png)")
        if path:
            self.reference_path = path
            self.reference_edit.setText(path)
            self.result_df = pd.DataFrame()
            self.display_df = pd.DataFrame()
            self.refresh_reference_preview()
            self.update_feedback_summary()
            self.render_results()
            self.set_status("已切换参考图，等待检索")
            self.log(f"选择参考图: {path}")

    def pick_gallery(self):
        path = QFileDialog.getExistingDirectory(self, "选择图库目录", self.gallery_path)
        if path:
            self.gallery_path = path
            self.gallery_edit.setText(path)
            self.log(f"选择图库目录: {path}")

    def pick_target(self):
        path = QFileDialog.getExistingDirectory(self, "选择目标目录", self.target_path)
        if path:
            self.target_path = path
            self.target_edit.setText(path)
            self.log(f"选择目标目录: {path}")

    def refresh_reference_preview(self):
        reference_path = self.reference_edit.text().strip()
        if not reference_path or not os.path.isfile(reference_path):
            self.reference_preview_label.setText("未加载参考图")
            self.reference_preview_label.setPixmap(QPixmap())
            self.reference_preview_text.setText("未加载参考图")
            return
        try:
            image = Image.open(reference_path).convert("RGB")
            image.thumbnail((REFERENCE_PREVIEW_SIZE, REFERENCE_PREVIEW_SIZE), Image.BILINEAR)
            self.reference_preview_label.setPixmap(pil_to_qpixmap(image))
            self.reference_preview_text.setText(os.path.basename(reference_path))
        except Exception as exc:
            self.reference_preview_label.setText("预览失败")
            self.reference_preview_text.setText(f"预览失败: {exc}")

    def open_reference_preview(self):
        reference_path = self.reference_edit.text().strip()
        if not reference_path or not os.path.isfile(reference_path):
            QMessageBox.warning(self, "参考图不存在", "请先选择有效的参考图。")
            return
        dialog = PreviewDialog(reference_path, "参考图", self)
        dialog.exec()

    def get_reference_feedback(self, reference_path=None):
        reference_path = reference_path or self.reference_edit.text().strip()
        if not reference_path:
            return {}
        reference_key = os.path.abspath(reference_path)
        feedback_map = self.gui_history.get("feedback_by_reference", {}).get(reference_key, {})
        cleaned = {}
        for image_path, label in feedback_map.items():
            normalized = str(label).strip().lower()
            if normalized in ("accepted", "rejected", "unlabeled"):
                cleaned[os.path.abspath(image_path)] = normalized
        return cleaned

    def save_feedback_label(self, image_path, label):
        reference_path = self.reference_edit.text().strip()
        if not reference_path:
            return
        reference_key = os.path.abspath(reference_path)
        feedback_root = self.gui_history.setdefault("feedback_by_reference", {})
        reference_feedback = feedback_root.setdefault(reference_key, {})
        image_key = os.path.abspath(image_path)
        normalized = str(label).strip().lower()
        if normalized == "unlabeled":
            reference_feedback.pop(image_key, None)
        else:
            reference_feedback[image_key] = normalized
        if not reference_feedback:
            feedback_root.pop(reference_key, None)
        save_gui_history(self.gui_history)

    def get_excluded_source_paths(self):
        excluded = set()
        for record in self.gui_history.get("copy_records", []):
            source_path = record.get("source_path")
            if source_path:
                excluded.add(os.path.abspath(source_path))
        return excluded

    def get_current_context_paths(self):
        return {
            "reference_path": normalize_abs_path(self.reference_edit.text().strip()),
            "gallery_path": normalize_abs_path(self.gallery_edit.text().strip()),
            "target_dir": normalize_abs_path(self.target_edit.text().strip()),
        }

    def is_current_context_record(self, record):
        context = self.get_current_context_paths()
        target_path = normalize_abs_path(record.get("target_path", ""))
        record_target_dir = normalize_abs_path(record.get("target_dir", ""))
        record_reference_path = normalize_abs_path(record.get("reference_path", ""))
        record_gallery_path = normalize_abs_path(record.get("gallery_path", ""))

        if not target_path or not context["target_dir"]:
            return False
        if not is_same_or_child_path(target_path, context["target_dir"]):
            return False
        if record_target_dir and record_target_dir != context["target_dir"]:
            return False
        if record_reference_path and context["reference_path"] and record_reference_path != context["reference_path"]:
            return False
        if record_gallery_path and context["gallery_path"] and record_gallery_path != context["gallery_path"]:
            return False
        return True

    def can_safely_delete_source_record(self, record):
        context = self.get_current_context_paths()
        source_path = normalize_abs_path(record.get("source_path", ""))
        target_path = normalize_abs_path(record.get("target_path", ""))

        if not source_path:
            return False, "源文件路径为空"
        if context["reference_path"] and source_path == context["reference_path"]:
            return False, "源文件等于当前参考图"
        if context["gallery_path"] and not is_same_or_child_path(source_path, context["gallery_path"]):
            return False, "源文件不在当前图库目录下"
        if target_path and source_path == target_path:
            return False, "源文件与保存副本是同一路径"
        return True, ""

    def get_rejected_paths_for_reference(self, reference_path=None):
        feedback_map = self.get_reference_feedback(reference_path)
        return {os.path.abspath(path) for path, label in feedback_map.items() if label == "rejected"}

    def get_hidden_result_paths(self, reference_path=None):
        hidden = set()
        hidden.update(self.get_excluded_source_paths())
        hidden.update(self.get_rejected_paths_for_reference(reference_path))
        return hidden

    def update_history_summary(self):
        excluded_count = len(self.get_excluded_source_paths())
        pending_count = len([r for r in self.gui_history.get("copy_records", []) if not r.get("deleted_at")])
        last_deleted_count = int(self.gui_history.get("last_deleted_count", 0))
        self.history_label.setText(
            f"历史已排除 {excluded_count} 张 | 待删源图 {pending_count} 张 | 上次删除源图 {last_deleted_count} 张"
        )

    def update_feedback_summary(self):
        feedback_map = self.get_reference_feedback()
        accepted_count = sum(1 for label in feedback_map.values() if label == "accepted")
        rejected_count = sum(1 for label in feedback_map.values() if label == "rejected")
        self.feedback_label.setText(f"反馈样本 accepted {accepted_count} | rejected {rejected_count}")

    def update_selection_summary(self):
        accepted_count = sum(1 for card in self.card_widgets if card.feedback_label == "accepted")
        shown_count = len(self.card_widgets)
        self.selection_badge.setText(f"Accepted {accepted_count} / {shown_count} 张")

    def get_display_filter_values(self):
        try:
            max_display_count = max(1, int(self.display_limit_edit.text().strip()))
        except ValueError:
            max_display_count = MAX_DISPLAY_COUNT
            self.display_limit_edit.setText(str(max_display_count))
        self.display_limit = max_display_count
        return {
            "max_display_count": max_display_count,
        }

    def apply_display_filters(self):
        self.refresh_display_results()
        filter_values = self.get_display_filter_values()
        self.log(f"显示数量已应用 | max_display={filter_values['max_display_count']}")

    def refresh_display_results(self):
        filter_values = self.get_display_filter_values()
        self.display_df = filter_display_results(
            self.result_df,
            excluded_paths=self.get_hidden_result_paths(),
            max_display_count=filter_values["max_display_count"],
        )
        self.render_results()

    def queue_feedback_rerun(self, reason, delay_ms=450):
        self.update_feedback_summary()
        if not self.auto_rerun_check.isChecked():
            self.set_status("反馈已更新，点击“开始检索 / 应用反馈”进入下一轮")
            return
        self.pending_rerun_reason = reason
        self.set_status("反馈已更新，准备自动重新检索...")
        self.pending_rerun_timer.start(delay_ms)

    def start_retrieval(self):
        if self.pending_rerun_timer.isActive():
            self.pending_rerun_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.pending_rerun_reason = self.pending_rerun_reason or "检索进行中收到新的反馈"
            self.set_status("检索中，当前反馈会在本轮结束后自动继续应用")
            return

        reference_path = self.reference_edit.text().strip()
        gallery_path = self.gallery_edit.text().strip()
        if not reference_path or not os.path.isfile(reference_path):
            QMessageBox.warning(self, "参考图不存在", "请先选择有效的参考图。")
            return
        if not gallery_path or not os.path.isdir(gallery_path):
            QMessageBox.warning(self, "图库目录不存在", "请先选择有效的图库目录。")
            return

        filter_values = self.get_display_filter_values()
        reason = self.pending_rerun_reason or "手动触发"
        self.pending_rerun_reason = None
        params = {
            "reference_path": reference_path,
            "gallery_path": gallery_path,
            "recursive": self.recursive_check.isChecked(),
            "feedback_labels": self.get_reference_feedback(reference_path),
            "hidden_paths": self.get_hidden_result_paths(reference_path),
            "topk": filter_values["max_display_count"],
            "filter_values": filter_values,
            "reason": reason,
        }
        self.worker = RetrievalWorker(params, self)
        self.worker.finished_ok.connect(self.on_retrieval_finished)
        self.worker.failed.connect(self.on_retrieval_failed)
        self.start_button.setEnabled(False)
        self.set_status("检索中...")
        self.log(
            f"开始检索 | reason={reason} | reference={reference_path} | "
            f"gallery={gallery_path} | recursive={params['recursive']}"
        )
        self.worker.start()

    def on_retrieval_finished(self, retrieval_result, params):
        filter_values = params["filter_values"]
        hidden_paths = params["hidden_paths"]
        feedback_labels = params["feedback_labels"]
        self.result_df = retrieval_result["result_df"]
        self.display_df = filter_display_results(
            self.result_df,
            excluded_paths=hidden_paths,
            max_display_count=filter_values["max_display_count"],
        )
        accepted_count = sum(1 for label in feedback_labels.values() if label == "accepted")
        rejected_count = sum(1 for label in feedback_labels.values() if label == "rejected")
        self.log(
            f"检索完成 | 全部结果 {len(self.result_df)} 张 | 展示 {len(self.display_df)} 张 | "
            f"accepted={accepted_count} | rejected={rejected_count} | 当前隐藏 {len(hidden_paths)} 张 | "
            f"规则: 最多展示 {filter_values['max_display_count']} 张 | "
            f"Accepted 会扩展下一轮查询, Rejected 会抑制下一轮相似结果"
        )
        self.update_history_summary()
        self.update_feedback_summary()
        self.render_results()
        self.start_button.setEnabled(True)
        self.set_status("检索完成")
        self.worker = None
        if self.pending_rerun_reason:
            QTimer.singleShot(0, self.start_retrieval)

    def on_retrieval_failed(self, error_message, tb):
        self.logger.error(tb)
        self.start_button.setEnabled(True)
        self.set_status("检索失败")
        QMessageBox.critical(self, "检索失败", error_message)
        self.log(f"检索失败: {error_message}")
        self.worker = None

    def clear_result_cards(self):
        while self.result_grid.count():
            item = self.result_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.card_widgets = []

    def render_results(self):
        self.clear_result_cards()
        if self.display_df.empty:
            empty = QLabel("当前没有可展示结果。可以更换参考图、调整过滤条件，或清理历史排除策略。")
            empty.setWordWrap(True)
            empty.setAlignment(Qt.AlignCenter)
            empty.setObjectName("mutedText")
            self.result_grid.addWidget(empty, 0, 0)
            self.update_selection_summary()
            return

        viewport_width = max(400, self.result_scroll.viewport().width() - 24)
        columns = max(1, viewport_width // (CARD_MIN_WIDTH + CARD_GAP))
        thumb_size = max(180, min(260, (viewport_width - CARD_GAP * (columns - 1)) // columns - 28))
        feedback_map = self.get_reference_feedback()

        for index, (_, row) in enumerate(self.display_df.reset_index(drop=True).iterrows()):
            image_path = row["gallery_image_path"]
            feedback = feedback_map.get(os.path.abspath(image_path), "unlabeled")
            card = ResultCard(row, feedback, thumb_size, self.toggle_accept_for_card, self.open_card_preview)
            self.card_widgets.append(card)
            self.result_grid.addWidget(card, index // columns, index % columns)

        self.update_selection_summary()
        self.log("结果已渲染。灰色=未处理，绿色=Accepted，红色=Rejected。")

    def toggle_accept_for_card(self, card):
        current_label = card.feedback_label
        next_label = "unlabeled" if current_label == "accepted" else "accepted"
        self.save_feedback_label(card.image_path, next_label)
        card.set_feedback_label(next_label)
        self.update_feedback_summary()
        self.update_selection_summary()
        self.log(f"反馈标记更新 | {next_label} | {card.image_path}")

    def open_card_preview(self, card):
        title = f"排名 {int(card.row['rank'])} | 分数 {float(card.row['rerank_score']):.4f}"
        dialog = PreviewDialog(card.image_path, title, self)
        dialog.exec()

    def set_feedback_for_displayed(self, label):
        if self.display_df.empty or not self.card_widgets:
            return
        changed_count = 0
        for card in self.card_widgets:
            if card.feedback_label != label:
                changed_count += 1
            self.save_feedback_label(card.image_path, label)
            card.set_feedback_label(label)
        self.update_feedback_summary()
        self.update_selection_summary()
        self.log(f"当前展示批量设定 | label={label} | changed_count={changed_count} | displayed_count={len(self.card_widgets)}")

    def reject_current_unaccepted(self):
        if self.display_df.empty or not self.card_widgets:
            return
        changed_count = 0
        kept_accepted_count = 0
        for card in self.card_widgets:
            if card.feedback_label == "accepted":
                kept_accepted_count += 1
                continue
            if card.feedback_label != "rejected":
                changed_count += 1
            self.save_feedback_label(card.image_path, "rejected")
        self.update_feedback_summary()
        self.refresh_display_results()
        self.log(
            f"当前展示批量反馈完成 | changed_to_rejected={changed_count} | "
            f"kept_accepted={kept_accepted_count} | displayed_count={len(self.card_widgets)} | "
            f"这些 rejected 会在下一轮检索中作为负样本"
        )
        self.queue_feedback_rerun("当前页未Accept设为Rejected")

    def copy_selected(self):
        if self.display_df.empty or not self.card_widgets:
            QMessageBox.warning(self, "没有结果", "请先完成一次检索。")
            return
        gallery_dir = self.gallery_edit.text().strip()
        target_dir = self.target_edit.text().strip()
        if not target_dir:
            QMessageBox.warning(self, "目标目录为空", "请先选择目标目录。")
            return
        if gallery_dir and (
            is_same_or_child_path(target_dir, gallery_dir) or is_same_or_child_path(gallery_dir, target_dir)
        ):
            QMessageBox.warning(
                self,
                "目录配置有风险",
                "保存目录与图库目录存在重叠，已阻止复制。\n"
                "请把保存目录设置到图库目录之外，避免污染图库或误删图库文件。",
            )
            self.log(
                "阻止复制 Accepted 图片 | reason=target_gallery_overlap | "
                f"gallery={gallery_dir} | target={target_dir}"
            )
            return
        os.makedirs(target_dir, exist_ok=True)

        selected_paths = [card.image_path for card in self.card_widgets if card.feedback_label == "accepted"]
        if not selected_paths:
            QMessageBox.information(self, "没有 Accepted", "当前展示结果里没有标记为 Accepted 的图片。")
            return

        selected_rows = [
            self.display_df[self.display_df["gallery_image_path"] == path].iloc[0]
            for path in selected_paths
        ]
        self.log(f"开始复制 Accepted 图片 | count={len(selected_rows)} | target={target_dir}")
        copied_count = 0
        for row in selected_rows:
            source_path = row["gallery_image_path"]
            target_path = make_unique_copy_path(target_dir, source_path)
            overwrite_existing = os.path.exists(target_path)
            shutil.copy2(source_path, target_path)
            if not has_copy_record(self.gui_history, source_path, target_path):
                self.gui_history["copy_records"].append(
                    {
                        "source_path": os.path.abspath(source_path),
                        "target_path": os.path.abspath(target_path),
                        "reference_path": os.path.abspath(self.reference_edit.text().strip()),
                        "gallery_path": os.path.abspath(self.gallery_edit.text().strip()),
                        "target_dir": os.path.abspath(target_dir),
                        "copied_at": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                        "deleted_at": None,
                    }
                )
            copied_count += 1
            action_text = "覆盖复制" if overwrite_existing else "复制"
            self.log(f"{action_text} | score={float(row['similarity_score']):.6f} | {source_path} -> {target_path}")
        save_gui_history(self.gui_history)
        self.update_history_summary()
        QMessageBox.information(self, "复制完成", f"已复制 {copied_count} 张；同名目标已自动覆盖。")
        self.log(f"复制完成 | copied_count={copied_count} | overwrite_mode=true")
        self.refresh_display_results()
        self.queue_feedback_rerun("复制 Accepted 图片")

    def delete_copied_source_files(self):
        context = self.get_current_context_paths()
        pending_records = [record for record in self.gui_history.get("copy_records", []) if not record.get("deleted_at")]
        scoped_records = [record for record in pending_records if self.is_current_context_record(record)]
        if not scoped_records:
            QMessageBox.information(
                self,
                "没有可删除源图",
                "当前保存目录 / 当前参考图 / 当前图库组合下，没有可删除的已复制源图。\n"
                f"当前图库目录: {self.gallery_edit.text().strip() or '未设置'}\n"
                f"上次删除数量: {int(self.gui_history.get('last_deleted_count', 0))} 张",
            )
            return
        result = QMessageBox.question(
            self,
            "确认删除",
            f"将删除当前上下文下的 {len(scoped_records)} 个已复制对应源文件。\n"
            f"参考图: {self.reference_edit.text().strip() or '未设置'}\n"
            f"图库目录: {self.gallery_edit.text().strip() or '未设置'}\n"
            f"保存目录: {self.target_edit.text().strip() or '未设置'}\n"
            "说明: 只删除图库里的源文件，已复制到保存目录的副本会保留。\n"
            f"上次删除数量: {int(self.gui_history.get('last_deleted_count', 0))} 张\n\n是否继续？",
        )
        if result != QMessageBox.Yes:
            return

        deleted_count = 0
        skipped_count = 0
        deleted_at = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
        for record in scoped_records:
            source_path = record.get("source_path", "")
            target_path = record.get("target_path", "")
            can_delete, reason = self.can_safely_delete_source_record(record)
            if not can_delete:
                skipped_count += 1
                self.log(
                    "跳过删除图库源文件 | "
                    f"reason={reason} | source={source_path} | target={target_path}"
                )
                continue
            if source_path and os.path.isfile(source_path):
                os.remove(source_path)
                deleted_count += 1
                self.log(f"删除图库源文件: {source_path} | 保留副本: {target_path}")
            else:
                self.log(f"图库源文件不存在，按已清理处理: {source_path} | 副本保留: {target_path}")
            record["deleted_at"] = deleted_at
        self.gui_history["last_deleted_count"] = deleted_count
        save_gui_history(self.gui_history)
        self.update_history_summary()
        self.refresh_display_results()
        QMessageBox.information(
            self,
            "删除完成",
            f"本次删除 {deleted_count} 张图库源图，跳过 {skipped_count} 张不安全记录。\n"
            f"上次删除按钮执行删除数量已更新为 {deleted_count} 张。",
        )
        self.log(
            "删除图库源文件完成 | "
            f"deleted_count={deleted_count} | skipped_count={skipped_count} | "
            f"reference={context['reference_path']} | gallery={context['gallery_path']} | target={context['target_dir']}"
        )


def main():
    app = QApplication(sys.argv)
    window = RetrievalMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
