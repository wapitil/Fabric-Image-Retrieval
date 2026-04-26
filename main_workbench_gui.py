import json
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime

import pandas as pd
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import predict_gallery_with_classifier as predict_core
import retrieval_gui
import train_dinov2_classifier as train_core
from dinov2_classifier_common import (
    cache_training_feature_for_path,
    ensure_device,
    load_or_build_gallery_features,
)
from PIL import Image


COLOR_BG = "#edf3f8"
COLOR_PANEL = "#ffffff"
COLOR_SOFT_PANEL = "#f7fafc"
COLOR_TEXT = "#16212b"
COLOR_MUTED = "#66768a"
COLOR_PRIMARY = "#1d4ed8"
COLOR_PRIMARY_ACTIVE = "#1e40af"
COLOR_SUCCESS = "#0f766e"
COLOR_BORDER = "#d9e3ee"
COLOR_STATUS_BG = "#e8f0ff"
COLOR_STATUS_TEXT = "#1e3a8a"


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"main_workbench_{datetime.now():%Y%m%d_%H%M%S}.log")
    logger = logging.getLogger(f"main_workbench_{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


class EmbeddedRetrievalWindow(retrieval_gui.RetrievalMainWindow):
    def show_startup_announcement(self):
        return

    def export_theme_stylesheet(self):
        return self.styleSheet()


class AutomationWorker(QThread):
    finished_ok = Signal(dict)
    failed = Signal(str, str)

    def __init__(self, mode, config, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.config = config

    def run(self):
        try:
            result = {}
            if self.mode == "train":
                result["train"] = train_core.run_training(self.config)
            elif self.mode == "predict":
                result["predict"] = predict_core.run_prediction(self.config)
            elif self.mode == "train_predict":
                train_result = train_core.run_training(self.config)
                predict_result = predict_core.run_prediction(
                    {
                        **self.config,
                        "classifier_artifact_path": train_result["artifact_path"],
                        "dataset_manifest_csv_path": train_result["manifest_path"],
                    }
                )
                result["train"] = train_result
                result["predict"] = predict_result
            else:
                raise RuntimeError(f"未知任务模式: {self.mode}")
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


class PredictionReviewCard(QFrame):
    def __init__(self, row, decision, thumb_size, toggle_callback, preview_callback, parent=None):
        super().__init__(parent)
        self.row = row
        self.image_path = row["image_path"]
        self.decision = decision
        self.toggle_callback = toggle_callback
        self.preview_callback = preview_callback

        self.setObjectName("resultCard")
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        self.class_label = QLabel(str(row["predicted_class"]))
        self.class_label.setObjectName("accentLabel")
        self.score_badge = QLabel(f"{float(row.get('top1_probability', 0.0)):.4f}")
        self.score_badge.setObjectName("scoreBadge")
        top_row.addWidget(self.class_label)
        top_row.addStretch(1)
        top_row.addWidget(self.score_badge)
        layout.addLayout(top_row)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        image = Image.open(self.image_path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.BILINEAR)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setPixmap(retrieval_gui.pil_to_qpixmap(image))
        self.image_label.setMinimumHeight(thumb_size)
        layout.addWidget(self.image_label)

        self.meta_label = QLabel(
            f"tier {row.get('tier', '')} | second {row.get('second_class', '')} "
            f"{float(row.get('top2_probability', 0.0)):.4f} | margin {float(row.get('probability_margin', 0.0)):.4f}"
        )
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        self.name_label = QLabel(os.path.basename(self.image_path))
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)

        self._click_filter = retrieval_gui.CardEventFilter(self)
        self._click_filter.clicked.connect(self.on_toggle)
        self._click_filter.double_clicked.connect(self.on_preview)
        for widget in (
            self,
            self.class_label,
            self.score_badge,
            self.status_label,
            self.image_label,
            self.meta_label,
            self.name_label,
        ):
            widget.installEventFilter(self._click_filter)

        self.apply_state()

    def on_toggle(self):
        self.toggle_callback(self)

    def on_preview(self):
        self.preview_callback(self)

    def set_decision(self, decision):
        self.decision = decision
        self.apply_state()

    def apply_state(self):
        if self.decision == "accepted":
            bg = retrieval_gui.COLOR_ACCEPTED
            fg = retrieval_gui.COLOR_ACCEPTED_TEXT
            status = "Accepted"
        elif self.decision == "rejected":
            bg = retrieval_gui.COLOR_REJECTED
            fg = retrieval_gui.COLOR_REJECTED_TEXT
            status = "Rejected"
        elif self.decision == "unknown":
            bg = "#fff7ed"
            fg = "#9a3412"
            status = "Unknown"
        else:
            bg = retrieval_gui.COLOR_UNLABELED
            fg = retrieval_gui.COLOR_UNLABELED_TEXT
            status = "Pending"

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


class TrainingAutomationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.review_df = None
        self.high_confidence_df = None
        self.unknown_df = None
        self.current_review_tier = "review"
        self.current_display_df = None
        self.review_cards = []
        self.review_feedback = {}
        self.logger, self.log_path = setup_logger(train_core.output_dir)
        self._build_ui()
        self._apply_styles()
        self.log(f"日志文件: {self.log_path}")
        self.refresh_summary_panels()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)
        page_scroll.setFrameShape(QFrame.NoFrame)
        root_layout.addWidget(page_scroll, 1)

        page = QWidget()
        page_scroll.setWidget(page)

        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(18, 18, 18, 18)
        page_layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("panel")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 18, 22, 18)
        title_col = QVBoxLayout()
        title = QLabel("训练与自动化")
        title.setObjectName("heroTitle")
        subtitle = QLabel("直接使用按类别分文件夹的数据集，训练分类器并对全图库做自动分流。")
        subtitle.setObjectName("heroSub")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        self.status_badge = QLabel("等待执行")
        self.status_badge.setObjectName("statusBadge")
        header_layout.addLayout(title_col, 1)
        header_layout.addWidget(self.status_badge, 0, Qt.AlignTop)
        page_layout.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(12)
        page_layout.addLayout(content, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        content.addLayout(left, 0)

        path_card = self._make_card("路径设置", soft=False)
        path_layout = path_card.layout()
        self.dataset_edit = QLineEdit(train_core.dataset_root_dir)
        self.gallery_edit = QLineEdit(train_core.gallery_dir)
        self.output_edit = QLineEdit(train_core.output_dir)
        self.cache_edit = QLineEdit(train_core.feature_cache_dir)
        self.ckpt_edit = QLineEdit(train_core.local_pretrained_path)
        path_layout.addWidget(self._row_with_picker("训练数据集目录", self.dataset_edit, self.pick_dataset, directory_only=True))
        path_layout.addWidget(self._row_with_picker("图库目录", self.gallery_edit, self.pick_gallery, directory_only=True))
        self.recursive_check = QCheckBox("递归扫描图库")
        self.recursive_check.setChecked(train_core.gallery_recursive)
        path_layout.addWidget(self.recursive_check)
        self.advanced_toggle_btn = QPushButton("显示高级设置")
        self.advanced_toggle_btn.clicked.connect(self.toggle_advanced_settings)
        path_layout.addWidget(self.advanced_toggle_btn)
        left.addWidget(path_card)

        advanced_card = self._make_card("高级设置", soft=True)
        advanced_layout = advanced_card.layout()
        advanced_layout.addWidget(self._row_with_picker("输出目录", self.output_edit, self.pick_output, directory_only=True))
        advanced_layout.addWidget(self._row_with_picker("特征缓存目录", self.cache_edit, self.pick_cache, directory_only=True))
        advanced_layout.addWidget(self._row_with_picker("模型权重", self.ckpt_edit, self.pick_checkpoint, directory_only=False, file_filter="Weights (*.pth *.pt);;All Files (*)"))

        train_card = self._make_card("训练参数", soft=True)
        train_layout = train_card.layout()
        self.epochs_edit = QLineEdit(str(train_core.epochs))
        self.batch_size_edit = QLineEdit(str(train_core.batch_size))
        self.lr_edit = QLineEdit(str(train_core.learning_rate))
        self.val_ratio_edit = QLineEdit(str(train_core.val_ratio))
        form_grid = QGridLayout()
        form_grid.setHorizontalSpacing(10)
        form_grid.setVerticalSpacing(10)
        form_grid.addWidget(QLabel("训练轮数"), 0, 0)
        form_grid.addWidget(self.epochs_edit, 0, 1)
        form_grid.addWidget(QLabel("Batch Size"), 1, 0)
        form_grid.addWidget(self.batch_size_edit, 1, 1)
        form_grid.addWidget(QLabel("学习率"), 2, 0)
        form_grid.addWidget(self.lr_edit, 2, 1)
        form_grid.addWidget(QLabel("验证比例"), 3, 0)
        form_grid.addWidget(self.val_ratio_edit, 3, 1)
        train_layout.addLayout(form_grid)
        advanced_layout.addWidget(train_card)

        predict_card = self._make_card("预测分流阈值", soft=True)
        predict_layout = predict_card.layout()
        self.high_conf_prob_edit = QLineEdit(str(predict_core.high_confidence_probability_threshold))
        self.review_prob_edit = QLineEdit(str(predict_core.review_probability_threshold))
        self.unknown_prob_edit = QLineEdit(str(predict_core.unknown_probability_threshold))
        self.margin_edit = QLineEdit(str(predict_core.min_top1_top2_margin))
        predict_grid = QGridLayout()
        predict_grid.setHorizontalSpacing(10)
        predict_grid.setVerticalSpacing(10)
        predict_grid.addWidget(QLabel("高置信阈值"), 0, 0)
        predict_grid.addWidget(self.high_conf_prob_edit, 0, 1)
        predict_grid.addWidget(QLabel("复核阈值"), 1, 0)
        predict_grid.addWidget(self.review_prob_edit, 1, 1)
        predict_grid.addWidget(QLabel("未知阈值"), 2, 0)
        predict_grid.addWidget(self.unknown_prob_edit, 2, 1)
        predict_grid.addWidget(QLabel("Top1-Top2 margin"), 3, 0)
        predict_grid.addWidget(self.margin_edit, 3, 1)
        predict_layout.addLayout(predict_grid)
        advanced_layout.addWidget(predict_card)
        advanced_card.setVisible(False)
        self.advanced_card = advanced_card
        left.addWidget(advanced_card)

        model_card = self._make_card("模型操作", soft=False)
        model_layout = model_card.layout()
        self.model_hint_label = QLabel("训练新模型，或者直接使用已有 classifier 做预测。")
        self.model_hint_label.setObjectName("softText")
        model_layout.addWidget(self.model_hint_label)
        self.model_status_text = QPlainTextEdit()
        self.model_status_text.setReadOnly(True)
        self.model_status_text.setMinimumHeight(90)
        self.model_status_text.setMaximumHeight(120)
        model_layout.addWidget(self.model_status_text)
        self.train_btn = QPushButton("训练模型")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.clicked.connect(lambda: self.start_job("train"))
        self.predict_btn = QPushButton("使用已有模型预测")
        self.predict_btn.clicked.connect(lambda: self.start_job("predict"))
        self.run_all_btn = QPushButton("重新训练并预测")
        self.run_all_btn.setObjectName("successButton")
        self.run_all_btn.clicked.connect(lambda: self.start_job("train_predict"))
        self.refresh_btn = QPushButton("刷新结果摘要")
        self.refresh_btn.clicked.connect(self.refresh_summary_panels)
        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(10)
        action_grid.setVerticalSpacing(10)
        action_grid.addWidget(self.train_btn, 0, 0)
        action_grid.addWidget(self.predict_btn, 0, 1)
        action_grid.addWidget(self.run_all_btn, 1, 0)
        action_grid.addWidget(self.refresh_btn, 1, 1)
        model_layout.addLayout(action_grid)
        left.addWidget(model_card)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setSpacing(12)
        content.addLayout(right, 1)

        summary_card = self._make_card("结果摘要", soft=False)
        summary_layout = summary_card.layout()
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(130)
        self.summary_text.setMaximumHeight(170)
        summary_layout.addWidget(self.summary_text)
        right.addWidget(summary_card, 0)

        review_card = self._make_card("预测结果审核", soft=False)
        review_layout = review_card.layout()
        review_filter_row = QHBoxLayout()
        self.review_tier_combo = QComboBox()
        self.review_tier_combo.addItems(["review", "high_confidence_candidate", "unknown"])
        self.review_tier_combo.currentTextChanged.connect(self.on_review_tier_changed)
        self.review_class_combo = QComboBox()
        self.review_class_combo.addItem("全部类别")
        self.review_class_combo.currentTextChanged.connect(self.refresh_review_results)
        self.review_limit_edit = QLineEdit("60")
        self.review_refresh_btn = QPushButton("刷新审核区")
        self.review_refresh_btn.clicked.connect(self.refresh_review_results)
        self.save_review_btn = QPushButton("保存审核进度")
        self.save_review_btn.setObjectName("successButton")
        self.save_review_btn.clicked.connect(self.save_review_feedback)
        self.add_to_pool_btn = QPushButton("加入最终结果池")
        self.add_to_pool_btn.clicked.connect(self.add_reviewed_to_final_pool)
        self.export_pool_btn = QPushButton("导出到训练集目录")
        self.export_pool_btn.setObjectName("dangerButton")
        self.export_pool_btn.clicked.connect(self.export_final_pool_to_dataset_root)
        self.accept_page_btn = QPushButton("当前页全Accept")
        self.accept_page_btn.clicked.connect(lambda: self.set_feedback_for_displayed("accepted"))
        self.reject_page_btn = QPushButton("当前未Accept全部Reject")
        self.reject_page_btn.clicked.connect(self.reject_current_unaccepted)
        self.unknown_page_btn = QPushButton("当前页设Unknown")
        self.unknown_page_btn.clicked.connect(lambda: self.set_feedback_for_displayed("unknown"))
        review_filter_row.addWidget(QLabel("结果类型"))
        review_filter_row.addWidget(self.review_tier_combo)
        review_filter_row.addWidget(QLabel("类别"))
        review_filter_row.addWidget(self.review_class_combo)
        review_filter_row.addWidget(QLabel("展示数量"))
        review_filter_row.addWidget(self.review_limit_edit)
        review_filter_row.addWidget(self.review_refresh_btn)
        review_filter_row.addStretch(1)
        review_layout.addLayout(review_filter_row)

        review_action_row = QHBoxLayout()
        review_action_row.setSpacing(8)
        review_action_row.addWidget(self.accept_page_btn)
        review_action_row.addWidget(self.reject_page_btn)
        review_action_row.addWidget(self.unknown_page_btn)
        review_action_row.addWidget(self.add_to_pool_btn)
        review_action_row.addWidget(self.export_pool_btn)
        review_action_row.addWidget(self.save_review_btn)
        review_action_row.addStretch(1)
        review_layout.addLayout(review_action_row)

        self.review_status_label = QLabel("还没有加载预测结果。先执行预测，再在这里审核。")
        self.review_status_label.setObjectName("softText")
        review_layout.addWidget(self.review_status_label)

        self.review_scroll = QScrollArea()
        self.review_scroll.setWidgetResizable(True)
        self.review_scroll.setFrameShape(QFrame.NoFrame)
        self.review_content = QWidget()
        self.review_grid = QGridLayout(self.review_content)
        self.review_grid.setContentsMargins(0, 0, 0, 0)
        self.review_grid.setHorizontalSpacing(16)
        self.review_grid.setVerticalSpacing(16)
        self.review_scroll.setWidget(self.review_content)
        review_layout.addWidget(self.review_scroll, 1)
        right.addWidget(review_card, 1)

        log_card = self._make_card("日志", soft=False)
        log_layout = log_card.layout()
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(110)
        self.log_box.setMaximumHeight(150)
        log_layout.addWidget(self.log_box)
        right.addWidget(log_card, 0)

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

    def _row_with_picker(self, title, line_edit, pick_callback, directory_only=True, file_filter="All Files (*)"):
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
        if directory_only:
            button.clicked.connect(pick_callback)
        else:
            button.clicked.connect(lambda: pick_callback(file_filter))
        row.addWidget(line_edit, 1)
        row.addWidget(button)
        layout.addWidget(label)
        layout.addLayout(row)
        return container

    def _apply_styles(self):
        self.setStyleSheet(
            f"""
            QWidget {{
                background: {COLOR_BG};
                color: {COLOR_TEXT};
                font-size: 14px;
                font-family: "Microsoft YaHei UI";
            }}
            QFrame#panel {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 18px;
            }}
            QFrame#softPanel {{
                background: {COLOR_SOFT_PANEL};
                border: none;
                border-radius: 16px;
            }}
            QLabel#heroTitle {{
                font-size: 24px;
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
            QPushButton {{
                background: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 13px;
                padding: 11px 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                border-color: {COLOR_PRIMARY};
                background: #f8fbff;
            }}
            QPushButton#primaryButton {{
                background: {COLOR_PRIMARY};
                color: white;
                border-color: {COLOR_PRIMARY};
                padding: 13px 16px;
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
            QPushButton#dangerButton {{
                color: #9a3412;
                font-weight: 700;
            }}
            QPushButton#dangerButton:hover {{
                background: #fff7ed;
            }}
            QLineEdit, QComboBox {{
                background: white;
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
                padding: 9px 12px;
            }}
            QPlainTextEdit {{
                background: {COLOR_SOFT_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
                padding: 8px;
            }}
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            """
        )

    def log(self, message):
        self.logger.info(message)
        self.log_box.appendPlainText(message)

    def set_status(self, message):
        self.status_badge.setText(message)

    def pick_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "选择训练数据集目录", self.dataset_edit.text().strip())
        if path:
            self.dataset_edit.setText(path)
            self.log(f"选择训练数据集目录: {path}")

    def pick_gallery(self):
        path = QFileDialog.getExistingDirectory(self, "选择图库目录", self.gallery_edit.text().strip())
        if path:
            self.gallery_edit.setText(path)
            self.log(f"选择图库目录: {path}")

    def pick_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text().strip())
        if path:
            self.output_edit.setText(path)
            self.log(f"选择输出目录: {path}")

    def pick_cache(self):
        path = QFileDialog.getExistingDirectory(self, "选择特征缓存目录", self.cache_edit.text().strip())
        if path:
            self.cache_edit.setText(path)
            self.log(f"选择特征缓存目录: {path}")

    def pick_checkpoint(self, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", self.ckpt_edit.text().strip(), file_filter)
        if path:
            self.ckpt_edit.setText(path)
            self.log(f"选择模型权重: {path}")

    def toggle_advanced_settings(self):
        is_visible = self.advanced_card.isVisible()
        self.advanced_card.setVisible(not is_visible)
        self.advanced_toggle_btn.setText("隐藏高级设置" if not is_visible else "显示高级设置")

    def build_config(self):
        dataset_root_dir = self.dataset_edit.text().strip()
        gallery_dir = self.gallery_edit.text().strip()
        output_dir = self.output_edit.text().strip() or train_core.output_dir
        if not dataset_root_dir or not os.path.isdir(dataset_root_dir):
            raise RuntimeError("请先选择有效的训练数据集目录。")
        if not gallery_dir or not os.path.isdir(gallery_dir):
            raise RuntimeError("请先选择有效的图库目录。")

        return {
            "dataset_root_dir": dataset_root_dir,
            "gallery_dir": gallery_dir,
            "gallery_recursive": self.recursive_check.isChecked(),
            "output_dir": output_dir,
            "feature_cache_dir": self.cache_edit.text().strip(),
            "local_pretrained_path": self.ckpt_edit.text().strip(),
            "epochs": int(self.epochs_edit.text().strip()),
            "batch_size": int(self.batch_size_edit.text().strip()),
            "learning_rate": float(self.lr_edit.text().strip()),
            "val_ratio": float(self.val_ratio_edit.text().strip()),
            "high_confidence_probability_threshold": float(self.high_conf_prob_edit.text().strip()),
            "review_probability_threshold": float(self.review_prob_edit.text().strip()),
            "unknown_probability_threshold": float(self.unknown_prob_edit.text().strip()),
            "min_top1_top2_margin": float(self.margin_edit.text().strip()),
            "dataset_manifest_csv_path": os.path.join(output_dir, "folder_dataset_manifest.csv"),
            "classifier_artifact_path": os.path.join(output_dir, "dinov2_linear_classifier.pt"),
        }

    def set_busy(self, busy):
        for button in (self.train_btn, self.predict_btn, self.run_all_btn, self.refresh_btn):
            button.setEnabled(not busy)

    def start_job(self, mode):
        if self.worker is not None and self.worker.isRunning():
            self.set_status("任务正在执行，请稍候")
            return
        try:
            config = self.build_config()
        except Exception as exc:
            QMessageBox.warning(self, "配置无效", str(exc))
            return

        self.worker = AutomationWorker(mode, config, self)
        self.worker.finished_ok.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)
        self.set_busy(True)
        self.set_status("执行中...")
        self.log(f"开始执行任务 | mode={mode} | dataset={config['dataset_root_dir']} | gallery={config['gallery_dir']}")
        self.worker.start()

    def on_worker_finished(self, result):
        self.worker = None
        self.set_busy(False)
        self.set_status("执行完成")
        if "train" in result:
            train_result = result["train"]
            self.log(
                f"训练完成 | class_count={train_result['class_count']} | sample_count={train_result['sample_count']} | "
                f"best_val_accuracy={train_result['best_val_accuracy']}"
            )
        if "predict" in result:
            predict_result = result["predict"]
            self.log(
                f"预测完成 | gallery_count={predict_result['gallery_count']} | high_conf={predict_result['high_confidence_candidate_count']} | "
                f"review={predict_result['review_count']} | unknown={predict_result['unknown_count']}"
            )
        self.refresh_summary_panels()
        self.load_prediction_results()

    def on_worker_failed(self, error_message, tb):
        self.worker = None
        self.set_busy(False)
        self.set_status("执行失败")
        self.logger.error(tb)
        self.log(f"执行失败: {error_message}")
        QMessageBox.critical(self, "执行失败", error_message)

    def refresh_summary_panels(self):
        output_dir = self.output_edit.text().strip() or train_core.output_dir
        training_summary_path = os.path.join(output_dir, "training_summary.json")
        prediction_summary_path = os.path.join(output_dir, "prediction_summary.json")
        lines = []
        model_lines = []

        artifact_path = os.path.join(output_dir, "dinov2_linear_classifier.pt")
        if os.path.isfile(artifact_path):
            model_lines.append("已有 classifier 可直接用于预测")
            model_lines.append(f"模型文件: {os.path.basename(artifact_path)}")
        else:
            model_lines.append("当前还没有 classifier，请先训练模型")

        if os.path.isfile(training_summary_path):
            with open(training_summary_path, "r", encoding="utf-8") as f:
                training_summary = json.load(f)
            model_lines.append(f"类别数: {training_summary.get('class_count', 0)}")
            model_lines.append(f"样本数: {training_summary.get('sample_count', 0)}")
            model_lines.append(f"最佳验证精度: {training_summary.get('best_val_accuracy', None)}")
            lines.append("训练摘要")
            lines.append(f"class_count: {training_summary.get('class_count', 0)}")
            lines.append(f"sample_count: {training_summary.get('sample_count', 0)}")
            lines.append(f"best_val_accuracy: {training_summary.get('best_val_accuracy', None)}")
            lines.append("")

        if os.path.isfile(prediction_summary_path):
            with open(prediction_summary_path, "r", encoding="utf-8") as f:
                prediction_summary = json.load(f)
            lines.append("预测摘要")
            lines.append(f"gallery_count: {prediction_summary.get('gallery_count', 0)}")
            lines.append(f"labeled_skip_count: {prediction_summary.get('labeled_skip_count', 0)}")
            lines.append(f"high_confidence_candidate_count: {prediction_summary.get('high_confidence_candidate_count', 0)}")
            lines.append(f"review_count: {prediction_summary.get('review_count', 0)}")
            lines.append(f"unknown_count: {prediction_summary.get('unknown_count', 0)}")
            lines.append("")

        final_pool_path = self.get_final_pool_path()
        if os.path.isfile(final_pool_path):
            final_pool_df = pd.read_csv(final_pool_path)
            lines.append("最终结果池摘要")
            lines.append(f"final_pool_count: {len(final_pool_df)}")
            if not final_pool_df.empty and "exported" in final_pool_df.columns:
                exported_series = final_pool_df["exported"].fillna(False).astype(str).str.lower().isin(["true", "1"])
                lines.append(f"exported_count: {int(exported_series.sum())}")
            lines.append("")

        if not lines:
            lines = ["当前还没有训练或预测结果。先选择训练数据集目录和图库目录，再执行任务。"]
        else:
            lines.append("下一步：先审核高置信候选 / 待复核，再把 Accepted 加入最终结果池。")
        self.summary_text.setPlainText("\n".join(lines))
        self.model_status_text.setPlainText("\n".join(model_lines))
        self.load_prediction_results()

    def get_review_feedback_path(self):
        output_dir = self.output_edit.text().strip() or train_core.output_dir
        return os.path.join(output_dir, "prediction_review_feedback.csv")

    def get_final_pool_path(self):
        output_dir = self.output_edit.text().strip() or train_core.output_dir
        return os.path.join(output_dir, "final_result_pool.csv")

    def load_prediction_results(self):
        output_dir = self.output_edit.text().strip() or train_core.output_dir
        paths = {
            "review": os.path.join(output_dir, "prediction_review_queue.csv"),
            "high_confidence_candidate": os.path.join(output_dir, "prediction_high_confidence_candidate.csv"),
            "unknown": os.path.join(output_dir, "prediction_unknown.csv"),
        }
        hidden_paths = self.get_hidden_prediction_paths()

        self.review_df = self._filter_prediction_df(self._read_prediction_csv(paths["review"]), hidden_paths)
        self.high_confidence_df = self._filter_prediction_df(self._read_prediction_csv(paths["high_confidence_candidate"]), hidden_paths)
        self.unknown_df = self._filter_prediction_df(self._read_prediction_csv(paths["unknown"]), hidden_paths)
        self.review_feedback = self._load_review_feedback()
        self.refresh_review_class_options()
        self.refresh_review_results()

    def _read_prediction_csv(self, path):
        if not os.path.isfile(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception as exc:
            self.log(f"读取预测文件失败 | path={path} | error={exc}")
            return None

    def _filter_prediction_df(self, df, hidden_paths):
        if df is None or df.empty:
            return df
        filtered_df = df[~df["image_path"].map(lambda path: self.make_review_key(path) in hidden_paths)].copy()
        filtered_df = filtered_df[filtered_df["image_path"].map(lambda path: os.path.isfile(str(path)))].copy()
        return filtered_df.reset_index(drop=True)

    def get_hidden_prediction_paths(self):
        final_pool_path = self.get_final_pool_path()
        if not os.path.isfile(final_pool_path):
            return set()
        try:
            final_pool_df = pd.read_csv(final_pool_path)
        except Exception:
            return set()
        hidden_paths = set()
        for row in final_pool_df.to_dict("records"):
            hidden_paths.add(self.make_review_key(row.get("image_path", "")))
        return hidden_paths

    def _load_review_feedback(self):
        feedback_path = self.get_review_feedback_path()
        if not os.path.isfile(feedback_path):
            return {}
        try:
            feedback_df = pd.read_csv(feedback_path)
        except Exception as exc:
            self.log(f"读取审核反馈失败 | {exc}")
            return {}
        feedback_map = {}
        for row in feedback_df.to_dict("records"):
            key = self.make_review_key(row.get("image_path", ""))
            feedback_map[key] = str(row.get("human_decision", "pending")).strip().lower()
        return feedback_map

    def refresh_review_class_options(self):
        current_class = self.review_class_combo.currentText()
        class_names = ["全部类别"]
        tier_df = self.get_current_tier_df()
        if tier_df is not None and not tier_df.empty:
            class_names.extend(sorted(tier_df["predicted_class"].dropna().unique().tolist()))
        self.review_class_combo.blockSignals(True)
        self.review_class_combo.clear()
        self.review_class_combo.addItems(class_names)
        if current_class in class_names:
            self.review_class_combo.setCurrentText(current_class)
        self.review_class_combo.blockSignals(False)

    def on_review_tier_changed(self, tier_name):
        self.current_review_tier = tier_name
        self.refresh_review_class_options()
        self.refresh_review_results()

    def get_current_tier_df(self):
        if self.current_review_tier == "high_confidence_candidate":
            return self.high_confidence_df
        if self.current_review_tier == "unknown":
            return self.unknown_df
        return self.review_df

    def get_review_limit(self):
        try:
            return max(1, int(self.review_limit_edit.text().strip()))
        except ValueError:
            self.review_limit_edit.setText("60")
            return 60

    def refresh_review_results(self):
        tier_df = self.get_current_tier_df()
        self.clear_review_cards()
        if tier_df is None or tier_df.empty:
            self.current_display_df = pd.DataFrame()
            self.review_status_label.setText("当前类型没有可审核结果。")
            self.update_review_status()
            return

        filtered_df = tier_df.copy()
        selected_class = self.review_class_combo.currentText().strip()
        if selected_class and selected_class != "全部类别":
            filtered_df = filtered_df[filtered_df["predicted_class"] == selected_class].copy()
        self.current_display_df = filtered_df.head(self.get_review_limit()).reset_index(drop=True)

        if self.current_display_df.empty:
            self.review_status_label.setText("当前筛选条件下没有结果。")
            self.update_review_status()
            return

        viewport_width = max(480, self.review_scroll.viewport().width() - 24)
        columns = max(1, viewport_width // 290)
        thumb_size = max(180, min(240, (viewport_width - 16 * (columns - 1)) // columns - 28))

        for index, (_, row) in enumerate(self.current_display_df.iterrows()):
            decision = self.review_feedback.get(self.make_review_key(row["image_path"]), "pending")
            card = PredictionReviewCard(row, decision, thumb_size, self.toggle_review_card, self.open_review_preview)
            self.review_cards.append(card)
            self.review_grid.addWidget(card, index // columns, index % columns)
        self.update_review_status()

    def clear_review_cards(self):
        while self.review_grid.count():
            item = self.review_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.review_cards = []

    def make_review_key(self, image_path):
        return os.path.normcase(os.path.abspath(str(image_path).strip()))

    def toggle_review_card(self, card):
        order = ["pending", "accepted", "rejected", "unknown"]
        current_index = order.index(card.decision) if card.decision in order else 0
        next_decision = order[(current_index + 1) % len(order)]
        card.set_decision(next_decision)
        self.review_feedback[self.make_review_key(card.image_path)] = next_decision
        self.update_review_status()
        self.log(f"审核标记更新 | {next_decision} | tier={self.current_review_tier} | {card.image_path}")

    def open_review_preview(self, card):
        title = f"{card.row['predicted_class']} | {float(card.row.get('top1_probability', 0.0)):.4f}"
        dialog = retrieval_gui.PreviewDialog(card.image_path, title, self)
        dialog.exec()

    def set_feedback_for_displayed(self, decision):
        if not self.review_cards:
            return
        for card in self.review_cards:
            card.set_decision(decision)
            self.review_feedback[self.make_review_key(card.image_path)] = decision
        self.update_review_status()
        self.log(f"当前审核页批量设置 | decision={decision} | count={len(self.review_cards)}")

    def reject_current_unaccepted(self):
        if not self.review_cards:
            return
        changed_count = 0
        kept_accepted_count = 0
        for card in self.review_cards:
            if card.decision == "accepted":
                kept_accepted_count += 1
                continue
            if card.decision != "rejected":
                changed_count += 1
            card.set_decision("rejected")
            self.review_feedback[self.make_review_key(card.image_path)] = "rejected"
        self.update_review_status()
        self.log(
            f"当前未Accept全部Reject | changed_to_rejected={changed_count} | "
            f"kept_accepted={kept_accepted_count} | displayed_count={len(self.review_cards)}"
        )

    def update_review_status(self):
        accepted_count = sum(1 for card in self.review_cards if card.decision == "accepted")
        rejected_count = sum(1 for card in self.review_cards if card.decision == "rejected")
        unknown_count = sum(1 for card in self.review_cards if card.decision == "unknown")
        pending_count = sum(1 for card in self.review_cards if card.decision == "pending")
        total_count = 0 if self.current_display_df is None else len(self.current_display_df)
        self.review_status_label.setText(
            f"当前类型: {self.current_review_tier} | 当前展示 {total_count} 张 | "
            f"Accepted {accepted_count} | Rejected {rejected_count} | Unknown {unknown_count} | Pending {pending_count}"
        )

    def save_review_feedback(self, silent=False):
        rows = []
        for tier_name, tier_df in (
            ("review", self.review_df),
            ("high_confidence_candidate", self.high_confidence_df),
            ("unknown", self.unknown_df),
        ):
            if tier_df is None or tier_df.empty:
                continue
            for row in tier_df.to_dict("records"):
                key = self.make_review_key(row["image_path"])
                decision = self.review_feedback.get(key, "pending")
                if decision == "pending":
                    continue
                rows.append(
                    {
                        "image_path": row["image_path"],
                        "predicted_class": row["predicted_class"],
                        "source_tier": tier_name,
                        "top1_probability": row.get("top1_probability", None),
                        "second_class": row.get("second_class", ""),
                        "top2_probability": row.get("top2_probability", None),
                        "probability_margin": row.get("probability_margin", None),
                        "human_decision": decision,
                    }
                )

        feedback_df = pd.DataFrame(rows)
        feedback_path = self.get_review_feedback_path()
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        feedback_df.to_csv(feedback_path, index=False, encoding="utf-8-sig")
        self.log(f"审核进度已保存 | path={feedback_path} | count={len(feedback_df)}")
        if not silent:
            QMessageBox.information(self, "保存完成", f"已保存 {len(feedback_df)} 条审核进度。")

    def add_reviewed_to_final_pool(self):
        rows = []
        for tier_name, tier_df in (
            ("review", self.review_df),
            ("high_confidence_candidate", self.high_confidence_df),
            ("unknown", self.unknown_df),
        ):
            if tier_df is None or tier_df.empty:
                continue
            for row in tier_df.to_dict("records"):
                key = self.make_review_key(row["image_path"])
                decision = self.review_feedback.get(key, "pending")
                if decision != "accepted":
                    continue
                rows.append(
                    {
                        "image_path": os.path.abspath(row["image_path"]),
                        "final_class": row["predicted_class"],
                        "predicted_class": row["predicted_class"],
                        "source_tier": tier_name,
                        "top1_probability": row.get("top1_probability", None),
                        "second_class": row.get("second_class", ""),
                        "top2_probability": row.get("top2_probability", None),
                        "probability_margin": row.get("probability_margin", None),
                        "human_decision": "accepted",
                        "exported": False,
                        "exported_path": "",
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        if not rows:
            QMessageBox.information(self, "没有可加入结果池的内容", "当前没有被标记为 Accepted 的审核结果。")
            return

        final_pool_path = self.get_final_pool_path()
        existing_df = pd.read_csv(final_pool_path) if os.path.isfile(final_pool_path) else pd.DataFrame()
        merged_map = {}
        if not existing_df.empty:
            for row in existing_df.to_dict("records"):
                merged_map[self.make_review_key(row["image_path"])] = row
        for row in rows:
            merged_map[self.make_review_key(row["image_path"])] = row

        final_pool_df = pd.DataFrame(merged_map.values())
        final_pool_df = final_pool_df.sort_values(by=["final_class", "image_path"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(final_pool_path), exist_ok=True)
        final_pool_df.to_csv(final_pool_path, index=False, encoding="utf-8-sig")
        for row in rows:
            self.review_feedback.pop(self.make_review_key(row["image_path"]), None)
        self.save_review_feedback(silent=True)
        self.log(f"已加入最终结果池 | path={final_pool_path} | added_or_updated={len(rows)} | total={len(final_pool_df)}")
        self.refresh_summary_panels()
        self.load_prediction_results()
        QMessageBox.information(self, "已加入最终结果池", f"本次新增或更新 {len(rows)} 条最终结果，已从审核区移除。")

    def export_final_pool_to_dataset_root(self):
        final_dataset_root = self.dataset_edit.text().strip()
        if not final_dataset_root or not os.path.isdir(final_dataset_root):
            QMessageBox.warning(self, "训练数据集目录无效", "请先选择有效的训练数据集目录。")
            return

        final_pool_path = self.get_final_pool_path()
        if not os.path.isfile(final_pool_path):
            QMessageBox.warning(self, "最终结果池不存在", "请先把审核通过结果加入 final_result_pool.csv。")
            return

        final_pool_df = pd.read_csv(final_pool_path)
        if final_pool_df.empty:
            QMessageBox.information(self, "结果池为空", "final_result_pool.csv 当前为空。")
            return

        gallery_feature_lookup = {}
        current_gallery_dir = self.gallery_edit.text().strip()
        current_cache_dir = self.cache_edit.text().strip() or train_core.feature_cache_dir
        current_output_dir = self.output_edit.text().strip() or train_core.output_dir
        current_ckpt_path = self.ckpt_edit.text().strip() or train_core.local_pretrained_path
        if current_gallery_dir and os.path.isdir(current_gallery_dir):
            try:
                gallery_bundle = load_or_build_gallery_features(
                    gallery_dir=current_gallery_dir,
                    gallery_recursive=self.recursive_check.isChecked(),
                    output_dir=current_output_dir,
                    feature_cache_dir=current_cache_dir,
                    local_pretrained_path=current_ckpt_path,
                    device=ensure_device(),
                )
                gallery_feature_lookup = {
                    os.path.normcase(os.path.abspath(path)): gallery_bundle["gallery_feature_matrix"][index].float().cpu()
                    for index, path in enumerate(gallery_bundle["gallery_image_list"])
                }
                self.log(
                    f"已加载图库特征用于回流缓存 | gallery_count={len(gallery_feature_lookup)} | "
                    "后续训练/预测将优先复用图库增量特征缓存"
                )
            except Exception as exc:
                self.log(f"加载图库特征用于回流缓存失败，后续训练/预测时将按需增量补提 | error={exc}")

        moved_count = 0
        cached_feature_count = 0
        updated_rows = []
        for row in final_pool_df.to_dict("records"):
            image_path = os.path.abspath(str(row.get("image_path", "")).strip())
            final_class = str(row.get("final_class", "")).strip()
            exported = str(row.get("exported", "")).strip().lower() in {"true", "1"}
            if exported or not image_path or not final_class or not os.path.isfile(image_path):
                updated_rows.append(row)
                continue

            target_dir = os.path.join(final_dataset_root, final_class)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, os.path.basename(image_path))
            if os.path.isfile(target_path):
                os.remove(target_path)
            shutil.move(image_path, target_path)

            cached_feature = gallery_feature_lookup.get(os.path.normcase(image_path))
            if cached_feature is not None:
                cache_training_feature_for_path(current_output_dir, target_path, cached_feature)
                cached_feature_count += 1

            row["exported"] = True
            row["exported_path"] = os.path.abspath(target_path)
            row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_rows.append(row)
            moved_count += 1
            self.log(f"移动到训练数据集目录 | {image_path} -> {target_path}")

        updated_df = pd.DataFrame(updated_rows)
        updated_df.to_csv(final_pool_path, index=False, encoding="utf-8-sig")
        self.refresh_summary_panels()
        self.load_prediction_results()
        QMessageBox.information(
            self,
            "导出完成",
            f"已移动 {moved_count} 张图片到训练数据集目录。\n"
            f"其中 {cached_feature_count} 张已同步训练特征缓存；后续重训会直接复用这些特征，"
            "图库特征也会优先走增量缓存，只补提新增或变更图片。",
        )


class MainWorkbenchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINOv2 工作台")
        self.resize(1680, 1060)
        self.setMinimumSize(1480, 920)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root_layout.addWidget(self.tabs, 1)

        self.retrieval_host = EmbeddedRetrievalWindow()
        retrieval_page = self.retrieval_host.takeCentralWidget()
        retrieval_page.setStyleSheet(self.retrieval_host.export_theme_stylesheet())
        self.tabs.addTab(retrieval_page, "界面一：图片检索")

        self.training_page = TrainingAutomationPage()
        self.tabs.addTab(self.training_page, "界面二：训练与自动化")
        self._apply_shell_styles()

    def _apply_shell_styles(self):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: {COLOR_BG};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLOR_BORDER};
                background: {COLOR_PANEL};
                border-radius: 18px;
                top: -1px;
            }}
            QTabBar::tab {{
                background: {COLOR_SOFT_PANEL};
                color: {COLOR_MUTED};
                border: 1px solid {COLOR_BORDER};
                padding: 12px 18px;
                min-width: 180px;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                margin-right: 6px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {COLOR_PANEL};
                color: {COLOR_TEXT};
                border-bottom-color: {COLOR_PANEL};
            }}
            QTabBar::tab:hover {{
                color: {COLOR_PRIMARY};
            }}
            """
        )


def main():
    app = QApplication(sys.argv)
    window = MainWorkbenchWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
