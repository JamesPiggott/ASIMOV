
import os
import html
import colorsys
import csv

# Initialize TensorFlow before PyQt5 on Windows.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

import sys
import time
from pathlib import Path
from threading import Condition, Lock

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QAction,
    QFrame,
    QCheckBox,
    QSizePolicy,
    QDialog,
    QScrollArea,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QComboBox,
    QToolButton,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)


class DetectionWorker(QThread):
    """
    Process face detection asynchronously.

    Only the newest pending frame is retained. The worker returns
    coordinates rather than an annotated video frame.
    """

    detection_ready = pyqtSignal(object, int, int)
    detection_error = pyqtSignal(str)

    def __init__(self, api, api_lock):
        super().__init__()

        self.api = api
        self.api_lock = api_lock
        self.condition = Condition()
        self.pending = None
        self.running = True

    def submit(self, frame, frame_number, generation, references=(), original_size=None):
        with self.condition:
            if not self.running:
                return

            self.pending = (
                frame.copy(),
                frame_number,
                generation,
                references,
                original_size,
            )

            self.condition.notify()

    def clear_pending(self):
        with self.condition:
            self.pending = None

    def stop(self):
        with self.condition:
            self.running = False
            self.pending = None
            self.condition.notify_all()

    def run(self):
        while True:
            with self.condition:
                while self.running and self.pending is None:
                    self.condition.wait()

                if not self.running:
                    return

                frame, frame_number, generation, references, original_size = self.pending
                self.pending = None

            try:
                with self.api_lock:
                    result = self.api.detect_faces(frame, False)
                    # Embeddings are calculated from the same detected face objects.
                    # This runs off the GUI thread and shares the API lock with imports.
                    face_results = []
                    for face in result.faces:
                        vector = None
                        if references:
                            crop = face.get_crop()
                            if crop is not None and crop.size:
                                raw = self.api.recognize_faces(crop)
                                if hasattr(raw, "numpy"):
                                    raw = raw.numpy()
                                vector = np.asarray(raw, dtype=np.float32).reshape(-1)
                        face_results.append((face, vector))

                detections = []

                for face, vector in face_results:
                    box = face.bound_box

                    landmarks = []

                    points = (
                        face.landmarks.left_eye,
                        face.landmarks.right_eye,
                        face.landmarks.nose,
                        face.landmarks.left_corner_mouth,
                        face.landmarks.right_corner_mouth,
                    )

                    if all(point is not None for point in points):
                        landmarks = [
                            (point.x, point.y)
                            for point in points
                        ]

                    name = "Unknown" if references else None
                    best_name = None
                    best_score = None
                    best_reference_id = None
                    if vector is not None and np.all(np.isfinite(vector)):
                        magnitude = np.linalg.norm(vector)
                        if magnitude > 0:
                            normalized = vector / magnitude
                            candidates = []
                            for ref_id, ref_name, ref_vector in references:
                                if ref_vector.shape == normalized.shape:
                                    candidates.append((float(np.dot(normalized, ref_vector)), ref_name, ref_id))
                            # Group reference photographs by identity; multiple images
                            # of one person must not compete with each other.
                            by_identity = {}
                            for score, identity, reference_id in candidates:
                                key = identity.strip().casefold()
                                if key not in by_identity or score > by_identity[key][0]:
                                    by_identity[key] = (score, identity, reference_id)
                            ranked = sorted(by_identity.values(), key=lambda value: value[0], reverse=True)
                            if ranked:
                                best_score, best_name, best_reference_id = ranked[0]
                                if best_score >= 0.55 and (len(ranked) == 1 or best_score - ranked[1][0] >= 0.04):
                                    name = best_name

                    detections.append({
                        "name": name,
                        "best_name": best_name,
                        "best_score": best_score,
                        "best_reference_id": best_reference_id if name == best_name else None,
                        "box": (
                            box.x1,
                            box.y1,
                            box.x2,
                            box.y2,
                        ),
                        "probability": face.prob.as_text,
                        "landmarks": landmarks,
                    })

                if original_size is not None:
                    original_width, original_height = original_size
                    height, width = frame.shape[:2]
                    sx, sy = original_width / width, original_height / height
                    for item in detections:
                        item["box"] = tuple(
                            int(round(value * (sx if i % 2 == 0 else sy)))
                            for i, value in enumerate(item["box"])
                        )
                        item["landmarks"] = [
                            (int(round(x * sx)), int(round(y * sy)))
                            for x, y in item["landmarks"]
                        ]
                self.detection_ready.emit(
                    detections,
                    frame_number,
                    generation,
                )

            except Exception as exc:
                self.detection_error.emit(str(exc))


class ReferenceImageWorker(QThread):
    """Import multiple images in one background job, isolating model calls."""

    completed = pyqtSignal(object, object)

    def __init__(self, api, api_lock, paths):
        super().__init__()
        self.api = api
        self.api_lock = api_lock
        self.paths = list(paths)

    def run(self):
        faces, errors = [], []
        for path in self.paths:
            try:
                raw = np.fromfile(path, dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Cannot read image")
                with self.api_lock:
                    result = self.api.detect_faces(image, True)
                    count = 0
                    for face in result.faces:
                        crop = face.get_crop()
                        if crop is None or crop.size == 0:
                            continue
                        embedding = self.api.recognize_faces(crop)
                        if hasattr(embedding, "numpy"):
                            embedding = embedding.numpy()
                        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
                            continue
                        faces.append((path, crop.copy(), embedding.copy()))
                        count += 1
                if not count:
                    errors.append(f"{Path(path).name}: no usable faces")
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        self.completed.emit(faces, errors)


class ReferenceFacesDialog(QDialog):
    """Immediately registered faces, editable inline with native row selection."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("ASIMOV | Reference Faces")
        self.resize(780, 580)
        self.registry = parent.face_registry
        self.worker = None
        self.row_lookup = {}
        layout = QVBoxLayout(self)
        heading = QLabel("Reference faces")
        heading.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Add photographs, edit names and choose which faces are included. Ctrl/Shift-click to select multiple rows."))

        toolbar = QHBoxLayout()
        self.upload_button = QPushButton("Add Images...")
        self.upload_button.clicked.connect(self.choose_images)
        toolbar.addWidget(self.upload_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.message = QLabel("Reference faces are stored for this session only.")
        layout.addWidget(self.message)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setSpacing(3)
        self.list_widget.itemSelectionChanged.connect(self.update_selection_actions)
        layout.addWidget(self.list_widget, 1)

        bottom = QHBoxLayout()
        self.count_label = QLabel()
        bottom.addWidget(self.count_label)
        bottom.addStretch()
        self.include_button = QPushButton("Include Selected")
        self.include_button.clicked.connect(lambda: self.bulk_include(True))
        self.exclude_button = QPushButton("Exclude Selected")
        self.exclude_button.clicked.connect(lambda: self.bulk_include(False))
        self.remove_button = QPushButton("Delete Selected")
        self.remove_button.clicked.connect(self.remove_selected)
        for button in (self.include_button, self.exclude_button, self.remove_button):
            button.setVisible(False)
            bottom.addWidget(button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom.addWidget(close_button)
        layout.addLayout(bottom)
        self.render_registry()

    @staticmethod
    def make_preview(crop):
        preview = QLabel()
        preview.setFixedSize(76, 76)
        preview.setAlignment(Qt.AlignCenter)
        if crop is not None and crop.size:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            preview.setPixmap(QPixmap.fromImage(image).scaled(
                72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return preview

    def render_registry(self):
        self.list_widget.clear()
        self.row_lookup.clear()
        for entry in self.registry:
            entry.setdefault("id", id(entry))
            entry.setdefault("included", True)
            entry.setdefault("matches", 0)
            entry.setdefault("name", "")
            item = QListWidgetItem()
            item.setSizeHint(QSize(500, 88))
            self.list_widget.addItem(item)
            self.row_lookup[id(item)] = entry
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(8, 3, 8, 3)
            thumbnail = self.make_preview(entry.get("crop"))
            # Clicking the thumbnail selects its QListWidget row, including Ctrl/Shift selection.
            thumbnail.mousePressEvent = lambda event, it=item: self.select_thumbnail(it, event)
            line.addWidget(thumbnail)
            field = QLineEdit(entry["name"])
            field.setPlaceholderText("Enter name (same name groups reference images)")
            field.editingFinished.connect(lambda e=entry, f=field: self.rename(e, f))
            line.addWidget(field, 1)
            included = QCheckBox("Include")
            included.setChecked(entry["included"])
            included.toggled.connect(lambda value, e=entry: self.set_included(e, value))
            line.addWidget(included)
            self.list_widget.setItemWidget(item, row)
        self.update_selection_actions()
        self.refresh_count()

    def select_thumbnail(self, item, event):
        flags = event.modifiers()
        if flags & Qt.ControlModifier:
            item.setSelected(not item.isSelected())
        elif flags & Qt.ShiftModifier and self.list_widget.currentRow() >= 0:
            start = self.list_widget.currentRow()
            end = self.list_widget.row(item)
            for index in range(min(start, end), max(start, end) + 1):
                self.list_widget.item(index).setSelected(True)
        else:
            self.list_widget.clearSelection()
            item.setSelected(True)
        self.list_widget.setCurrentItem(item, QAbstractItemView.NoUpdate)

    def selected_entries(self):
        return [self.row_lookup[id(item)] for item in self.list_widget.selectedItems()]

    def update_selection_actions(self):
        has_selection = bool(self.list_widget.selectedItems())
        for button in (self.include_button, self.exclude_button, self.remove_button):
            button.setVisible(has_selection)
        self.refresh_count()

    def refresh_count(self):
        included = sum(e.get("included", True) for e in self.registry)
        self.count_label.setText(f"{len(self.registry)} images · {included} included · {len(self.selected_entries())} selected")
        self.parent().refresh_reference_strip()

    def set_included(self, entry, value):
        entry["included"] = value
        self.refresh_count()

    def rename(self, entry, field):
        entry["name"] = field.text().strip()
        self.refresh_count()

    def bulk_include(self, value):
        for entry in self.selected_entries():
            entry["included"] = value
        self.render_registry()

    def remove_selected(self):
        selected = self.selected_entries()
        if not selected:
            return
        response = QMessageBox.question(
            self, "Delete reference faces", f"Delete {len(selected)} selected reference image(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response != QMessageBox.Yes:
            return
        keys = {entry["id"] for entry in selected}
        self.registry[:] = [entry for entry in self.registry if entry["id"] not in keys]
        self.render_registry()
        self.message.setText(f"Deleted {len(selected)} reference image(s).")

    def choose_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Reference Images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)")
        if not paths:
            return
        self.upload_button.setEnabled(False)
        self.message.setText(f"Processing {len(paths)} image(s)...")
        self.worker = ReferenceImageWorker(self.parent().api, self.parent().api_lock, paths)
        self.worker.completed.connect(self.on_completed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def on_completed(self, faces, errors):
        self.upload_button.setEnabled(True)
        for path, crop, embedding in faces:
            entry = {"name": Path(path).stem, "crop": crop, "embedding": embedding,
                     "included": True, "source": path, "matches": 0}
            entry["id"] = id(entry)
            self.registry.append(entry)
        self.render_registry()
        self.message.setText(f"Added {len(faces)} face(s) from selected images. Names can be edited immediately.")
        if errors:
            QMessageBox.warning(self, "Some images could not be imported", "\n".join(errors[:20]))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Processing", "Please wait until image processing finishes.")
            event.ignore()
            return
        super().closeEvent(event)


class BasicVideoApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASIMOV | Face Detection Studio")
        self.resize(1100, 740)
        self.setMinimumSize(720, 500)

        # Video state.
        self.video_path = None
        self.video_capture = None
        self.video_fps = 30.0
        self.frame_count = 0
        self.frame_number = 0

        self.is_paused = True
        self.is_seeking = False

        # Playback clock.
        self.playback_start_time = 0.0
        self.playback_start_frame = 0

        # Display FPS.
        self.last_display_time = 0.0
        self.display_fps = 0.0
        self.inference_pending = False
        self.last_frame_due = 0.0

        # Detection state.
        self.detection_generation = 0
        self.latest_detections = []
        self.latest_detection_frame = -1
        self.diagnostic_events = []
        self.diagnostic_limit = 10000
        self.diagnostic_dirty = False

        # Build Qt UI before starting the worker.
        self.build_interface()
        self.apply_styles()

        # Import API after TensorFlow initialization and create it once.
        from application.core.API import API

        self.api = API()
        self.api_lock = Lock()
        self.face_registry = []
        self.identity_colors = {}

        self.detection_worker = DetectionWorker(self.api, self.api_lock)
        self.detection_worker.detection_ready.connect(
            self.on_detection_ready
        )
        self.detection_worker.detection_error.connect(
            self.on_detection_error
        )
        self.detection_worker.start()

        # Playback timer.
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)

        self.update_controls()
        self.refresh_reference_strip()

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(16)

        # Header.
        header = QHBoxLayout()
        header.setSpacing(14)

        title_group = QVBoxLayout()
        title_group.setSpacing(3)

        title = QLabel("ASIMOV")
        title.setObjectName("appTitle")

        subtitle = QLabel("FACE DETECTION STUDIO")
        subtitle.setObjectName("appSubtitle")

        title_group.addWidget(title)
        title_group.addWidget(subtitle)

        header.addLayout(title_group)
        header.addStretch()

        self.open_button = QPushButton("Open Video")
        self.open_button.setObjectName("primaryButton")
        self.open_button.setMinimumWidth(130)
        self.open_button.clicked.connect(self.load_video)

        self.reference_button = QPushButton("Reference Faces")
        self.reference_button.clicked.connect(self.open_reference_faces)
        header.addWidget(self.reference_button)
        header.addWidget(self.open_button)

        root.addLayout(header)

        # Video display.
        video_container = QFrame()
        video_container.setObjectName("videoContainer")

        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(12, 12, 12, 12)

        self.video_label = QLabel()
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )

        self.video_label.setText(
            "NO VIDEO LOADED\n\n"
            "Select Open Video to begin"
        )

        video_layout.addWidget(self.video_label)
        root.addWidget(video_container, 1)

        # Filename and playback time.
        information = QHBoxLayout()

        self.filename_label = QLabel("No video selected")
        self.filename_label.setObjectName("filenameLabel")

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("timeLabel")

        information.addWidget(self.filename_label)
        information.addStretch()
        information.addWidget(self.time_label)

        root.addLayout(information)

        # Timeline.
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 0)
        self.video_slider.sliderPressed.connect(
            self.on_seek_started
        )
        self.video_slider.sliderReleased.connect(
            self.on_seek_finished
        )
        self.video_slider.valueChanged.connect(
            self.on_slider_changed
        )

        root.addWidget(self.video_slider)

        # Player toolbar.
        toolbar = QFrame()
        toolbar.setObjectName("toolbar")

        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 12, 14, 12)
        toolbar_layout.setSpacing(10)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)

        for button in (
            self.play_button,
            self.pause_button,
            self.stop_button,
        ):
            button.setObjectName("controlButton")
            button.setMinimumWidth(80)
            button.setFixedHeight(36)

        toolbar_layout.addWidget(self.play_button)
        toolbar_layout.addWidget(self.pause_button)
        toolbar_layout.addWidget(self.stop_button)

        toolbar_layout.addStretch()

        self.detection_checkbox = QCheckBox("Face detection")
        self.detection_checkbox.setChecked(False)
        self.detection_checkbox.toggled.connect(
            self.on_detection_toggled
        )

        self.recognition_checkbox = QCheckBox("Face recognition")
        self.recognition_checkbox.setChecked(False)
        self.recognition_checkbox.toggled.connect(self.on_recognition_toggled)

        self.fps_checkbox = QCheckBox("Show FPS")
        self.fps_checkbox.setChecked(True)

        toolbar_layout.addWidget(self.detection_checkbox)
        toolbar_layout.addSpacing(14)
        toolbar_layout.addWidget(self.recognition_checkbox)
        toolbar_layout.addSpacing(14)
        toolbar_layout.addWidget(self.fps_checkbox)

        # Keep inference options with the other video controls.
        self.no_skip_checkbox = QCheckBox("No frame skipping")
        self.no_skip_checkbox.setChecked(False)
        self.no_skip_checkbox.toggled.connect(self.on_inference_options_changed)
        toolbar_layout.addSpacing(14)
        toolbar_layout.addWidget(self.no_skip_checkbox)

        toolbar_layout.addSpacing(14)
        toolbar_layout.addWidget(QLabel("Inference resolution:"))
        self.inference_scale_combo = QComboBox()
        for label, scale in (("100%", 1.0), ("75%", 0.75), ("50%", 0.5), ("25%", 0.25)):
            self.inference_scale_combo.addItem(label, scale)
        self.inference_scale_combo.currentIndexChanged.connect(self.on_inference_options_changed)
        toolbar_layout.addWidget(self.inference_scale_combo)

        root.addWidget(toolbar)

        # Collapsible reference strip: default closed to maximize video height.
        self.reference_toggle = QToolButton()
        self.reference_toggle.setCheckable(True)
        self.reference_toggle.setText("Reference Faces  ▸")
        self.reference_toggle.toggled.connect(self.toggle_reference_strip)
        root.addWidget(self.reference_toggle)
        self.reference_count_label = QLabel("0 images")
        self.reference_scroll = QScrollArea()
        self.reference_scroll.setWidgetResizable(True)
        self.reference_scroll.setFixedHeight(94)
        self.reference_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.reference_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.reference_strip_widget = QWidget()
        self.reference_strip_layout = QHBoxLayout(self.reference_strip_widget)
        self.reference_strip_layout.setContentsMargins(6, 2, 6, 2)
        self.reference_strip_layout.addStretch()
        self.reference_scroll.setWidget(self.reference_strip_widget)
        self.reference_scroll.setVisible(False)
        root.addWidget(self.reference_scroll)

        # Expandable, filterable diagnostics. Events are bounded in memory.
        self.diagnostic_toggle = QToolButton()
        self.diagnostic_toggle.setText("Diagnostics  ▸")
        self.diagnostic_toggle.setCheckable(True)
        self.diagnostic_toggle.toggled.connect(self.toggle_diagnostics)
        root.addWidget(self.diagnostic_toggle)
        self.diagnostic_panel = QWidget()
        diagnostic_layout = QVBoxLayout(self.diagnostic_panel)
        diagnostic_layout.setContentsMargins(0, 0, 0, 0)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Show:"))
        self.diagnostic_filter = QComboBox()
        self.diagnostic_filter.addItems(["All", "Detection", "Recognition"])
        self.diagnostic_filter.currentTextChanged.connect(self.refresh_diagnostics)
        self.matched_only = QCheckBox("Above threshold (accepted matches only)")
        self.matched_only.toggled.connect(self.refresh_diagnostics)
        filters.addWidget(self.matched_only)
        filters.addWidget(self.diagnostic_filter)
        filters.addStretch()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_diagnostics)
        filters.addWidget(clear_button)
        export_button = QPushButton("Export CSV")
        export_button.clicked.connect(self.export_diagnostics)
        filters.addWidget(export_button)
        reset_button = QPushButton("Reset Counters")
        reset_button.clicked.connect(self.reset_match_counters)
        filters.addWidget(reset_button)
        diagnostic_layout.addLayout(filters)
        self.diagnostic_text = QTextEdit()
        self.diagnostic_text.setReadOnly(True)
        self.diagnostic_text.setFixedHeight(105)
        diagnostic_layout.addWidget(self.diagnostic_text)
        self.diagnostic_panel.setVisible(False)
        root.addWidget(self.diagnostic_panel)
        self.diagnostic_refresh_timer = QTimer(self)
        self.diagnostic_refresh_timer.setInterval(350)
        self.diagnostic_refresh_timer.timeout.connect(self.flush_diagnostics)
        self.diagnostic_refresh_timer.start()

        # Bottom status.
        status = QHBoxLayout()

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")

        self.fps_label = QLabel("SOURCE: --   DISPLAY: --")
        self.fps_label.setObjectName("statusLabel")

        status.addWidget(self.status_label)
        status.addStretch()
        status.addWidget(self.fps_label)

        root.addLayout(status)

        # Build the menu only after all interface controls exist.
        self.build_menu()

    def refresh_reference_strip(self):
        if not hasattr(self, "reference_strip_layout") or not hasattr(self, "face_registry"):
            return
        layout = self.reference_strip_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        registry = self.face_registry
        names = {x.get("name", "").strip().casefold() for x in registry if x.get("name", "").strip()}
        self.reference_count_label.setText(f"{len(registry)} image(s) · {len(names)} named person(s)")
        self.reference_toggle.setText(
            f"Reference Faces ({len(registry)} images · {len(names)} people)  "
            + ("▾" if self.reference_toggle.isChecked() else "▸")
        )
        for entry in registry:
            cell = QWidget()
            column = QVBoxLayout(cell)
            column.setContentsMargins(3, 0, 3, 0)
            crop = entry.get("crop")
            preview = ReferenceFacesDialog.make_preview(crop)
            preview.setFixedSize(65, 65)
            if preview.pixmap() is not None:
                preview.setPixmap(preview.pixmap().scaled(62, 62, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # Overlay the accepted-match count on the image itself.
            # The badge is a child of the thumbnail, not a layout item,
            # so it cannot push the name or counter out of view.
            count = QLabel(str(entry.get("matches", 0)), preview)
            count.setAlignment(Qt.AlignCenter)
            count.setToolTip("Accepted recognition matches for this reference image")
            count.setStyleSheet(
                "QLabel { background-color: rgba(12, 19, 26, 220); "
                f"color: {self.color_for_identity(entry.get('name', ''))}; "
                "border: 1px solid rgba(255,255,255,65); "
                "border-radius: 9px; font-size: 11px; font-weight: 700; "
                "padding: 0px 4px; }"
            )
            count.adjustSize()
            count.resize(max(20, count.width()), 19)
            count.move(preview.width() - count.width() - 2, preview.height() - 21)
            count.show()
            column.addWidget(preview)
            label = QLabel(entry.get("name") or "Unlabelled")
            label.setMaximumWidth(100)
            label.setToolTip(entry.get("name") or "Unlabelled")
            label.setStyleSheet("color: #8496a5;" if not entry.get("included", True) else "")
            column.addWidget(label)
            layout.addWidget(cell)
        layout.addStretch()

    def toggle_reference_strip(self, expanded):
        self.reference_scroll.setVisible(expanded)
        self.refresh_reference_strip()

    def toggle_diagnostics(self, expanded):
        self.diagnostic_panel.setVisible(expanded)
        self.update_diagnostic_header()
        if expanded:
            self.refresh_diagnostics()

    def color_for_identity(self, name):
        """Stable, readable color per named identity for this session."""
        key = (name or "").strip().casefold()
        if not key:
            return "#91a0ad"
        if key not in self.identity_colors:
            hue = (len(self.identity_colors) * 0.61803398875 + 0.10) % 1.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.70, 0.72)
            self.identity_colors[key] = f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"
        return self.identity_colors[key]

    def update_diagnostic_header(self):
        accepted = sum(1 for event in self.diagnostic_events if event[3])
        arrow = "▾" if self.diagnostic_toggle.isChecked() else "▸"
        self.diagnostic_toggle.setText(
            f"Diagnostics ({len(self.diagnostic_events)} events · {accepted} accepted)  {arrow}"
        )

    def add_diagnostic(self, category, message, frame_number, identity=None,
                       accepted=False, metadata=None):
        seconds = frame_number / self.video_fps if self.video_fps else 0
        line = f"[{self.format_time(seconds)}] [{category}] {message}"
        self.diagnostic_events.append((category, line, identity, accepted, {
            "video_time_seconds": round(seconds, 3),
            "frame_number": frame_number,
            **(metadata or {}),
        }))
        if len(self.diagnostic_events) > self.diagnostic_limit:
            del self.diagnostic_events[:-self.diagnostic_limit]
        self.diagnostic_dirty = True

    def flush_diagnostics(self):
        if not self.diagnostic_dirty:
            return
        self.diagnostic_dirty = False
        self.update_diagnostic_header()
        if self.diagnostic_panel.isVisible():
            self.refresh_diagnostics()

    def refresh_diagnostics(self, *_):
        selected = self.diagnostic_filter.currentText()
        matched_only = self.matched_only.isChecked()
        lines = []
        for category, line, identity, accepted, metadata in self.diagnostic_events:
            if selected != "All" and category != selected:
                continue
            if matched_only and not accepted:
                continue
            color = self.color_for_identity(identity) if identity else (
                "#90a7b8" if category == "Detection" else "#a6adb5")
            lines.append(f'<span style="color:{color};">{html.escape(line)}</span>')
        self.diagnostic_text.setHtml(
            '<div style="font-family:Consolas,monospace; font-size:11px; white-space:pre;">'
            + '<br>'.join(lines) + '</div>')
        self.diagnostic_text.verticalScrollBar().setValue(
            self.diagnostic_text.verticalScrollBar().maximum())

    def clear_diagnostics(self):
        self.diagnostic_events.clear()
        self.diagnostic_dirty = False
        self.update_diagnostic_header()
        self.refresh_diagnostics()

    def reset_match_counters(self):
        for entry in self.face_registry:
            entry["matches"] = 0
        self.refresh_reference_strip()
        self.status_label.setText("Reference match counters reset")

    def export_diagnostics(self):
        if not self.diagnostic_events:
            QMessageBox.information(self, "Export Diagnostics", "No events to export.")
            return
        default_name = "asimov_diagnostics.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Diagnostics", default_name, "CSV Files (*.csv)"
        )
        if not path:
            return
        columns = ["video_time_seconds", "frame_number", "event_type", "identity",
                   "accepted", "face_width", "face_height", "detection_confidence",
                   "best_reference", "best_reference_id", "similarity", "threshold", "message"]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for category, line, identity, accepted, metadata in self.diagnostic_events:
                    row = {key: metadata.get(key, "") for key in columns}
                    row.update(event_type=category, identity=identity or "",
                               accepted=accepted, message=line)
                    writer.writerow(row)
        except OSError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
            return
        self.status_label.setText(f"Exported {len(self.diagnostic_events)} diagnostic events")

    def build_menu(self):
        # Use QMainWindow's existing menu bar.
        menu_bar = self.menuBar()
        menu_bar.clear()
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu("&File")

        open_action = QAction("Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_video)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_action)
        references_action = QAction("Reference Faces...", self)
        references_action.triggered.connect(self.open_reference_faces)
        file_menu.addAction(references_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        playback_menu = menu_bar.addMenu("&Playback")

        play_action = QAction("Play", self)
        play_action.triggered.connect(self.play_video)

        pause_action = QAction("Pause", self)
        pause_action.triggered.connect(self.pause_video)

        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_video)

        playback_menu.addAction(play_action)
        playback_menu.addAction(pause_action)
        playback_menu.addAction(stop_action)

        settings_menu = menu_bar.addMenu("&Settings")

        detection_action = QAction(
            "Face Detection",
            self,
            checkable=True,
        )

        detection_action.setChecked(False)

        # Keep menu action and checkbox synchronized.
        detection_action.toggled.connect(
            self.detection_checkbox.setChecked
        )

        self.detection_checkbox.toggled.connect(
            detection_action.setChecked
        )

        settings_menu.addAction(detection_action)

        recognition_action = QAction("Face Recognition", self, checkable=True)
        recognition_action.setChecked(False)
        recognition_action.toggled.connect(self.recognition_checkbox.setChecked)
        self.recognition_checkbox.toggled.connect(recognition_action.setChecked)
        settings_menu.addAction(recognition_action)

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow,
            QWidget {
                background-color: #151a21;
                color: #e5eaf0;
                font-family: "Segoe UI";
                font-size: 13px;
            }

            QMenuBar {
                background-color: #11161c;
                color: #d9e2eb;
                padding: 5px 10px;
                spacing: 8px;
            }

            QMenuBar::item {
                background: transparent;
                padding: 7px 12px;
                border-radius: 4px;
            }

            QMenuBar::item:selected {
                background-color: #283641;
                color: #65d8c1;
            }

            QMenu {
                background-color: #202832;
                color: #e5eaf0;
                border: 1px solid #384652;
                padding: 6px;
            }

            QMenu::item {
                padding: 8px 28px;
            }

            QMenu::item:selected {
                background-color: #31534f;
                color: white;
            }

            QLabel#appTitle {
                font-size: 25px;
                font-weight: 700;
                color: #f3f6f9;
            }

            QLabel#appSubtitle {
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 2px;
                color: #65d8c1;
            }

            QFrame#videoContainer {
                background-color: #0b0f14;
                border: 1px solid #303b46;
                border-radius: 10px;
            }

            QLabel#videoDisplay {
                background-color: #0b0f14;
                color: #778796;
                font-size: 14px;
                border: none;
            }

            QFrame#toolbar {
                background-color: #202832;
                border: 1px solid #303b46;
                border-radius: 8px;
            }

            QPushButton {
                background-color: #303b46;
                color: #e5eaf0;
                border: 1px solid #43515e;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #3b4b58;
            }

            QPushButton:pressed {
                background-color: #26333e;
            }

            QPushButton:disabled {
                color: #697783;
                background-color: #252d35;
                border-color: #303b46;
            }

            QPushButton#primaryButton {
                background-color: #42bfa8;
                color: #0c201d;
                border: none;
                font-weight: 700;
            }

            QPushButton#primaryButton:hover {
                background-color: #65d8c1;
            }

            QLabel#filenameLabel {
                color: #c5d0da;
                font-size: 12px;
            }

            QLabel#timeLabel {
                color: #a7b6c4;
                font-family: "Consolas";
                font-size: 12px;
            }

            QLabel#statusLabel {
                color: #8496a5;
                font-size: 11px;
            }

            QCheckBox {
                spacing: 8px;
                color: #d6dfe8;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }

            QCheckBox::indicator:unchecked {
                background-color: #19212a;
                border: 1px solid #61717f;
                border-radius: 3px;
            }

            QCheckBox::indicator:checked {
                background-color: #42bfa8;
                border: 1px solid #42bfa8;
                border-radius: 3px;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #34414d;
                border-radius: 3px;
            }

            QSlider::sub-page:horizontal {
                background: #42bfa8;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #65d8c1;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }

            QSlider::handle:horizontal:hover {
                background: #a0f0df;
            }
        """)

    @staticmethod
    def format_time(seconds):
        seconds = max(0, int(seconds))

        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return f"{minutes:02d}:{seconds:02d}"

    def update_time_label(self, frame_number=None):
        if frame_number is None:
            frame_number = self.frame_number

        elapsed = frame_number / self.video_fps
        duration = self.frame_count / self.video_fps

        self.time_label.setText(
            f"{self.format_time(elapsed)} / "
            f"{self.format_time(duration)}"
        )

    def update_controls(self):
        has_video = self.video_path is not None

        self.play_button.setEnabled(has_video)
        self.pause_button.setEnabled(
            has_video and not self.is_paused
        )
        self.stop_button.setEnabled(has_video)
        self.video_slider.setEnabled(has_video)

    def open_capture(self):
        if not self.video_path:
            return False

        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            capture.release()
            self.status_label.setText("Unable to open video")
            return False

        self.video_capture = capture
        return True

    def open_reference_faces(self):
        dialog = ReferenceFacesDialog(self)
        dialog.exec_()
        # Invalidate results computed against old names or inclusion settings.
        self.clear_detection()

    def load_video(self):
        default_dir = (
            Path(__file__).resolve().parents[1]
            / "test"
            / "sample_videos"
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            str(default_dir),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;"
            "All Files (*)",
        )

        if not path:
            return

        self.stop_video()

        self.video_path = path

        if not self.open_capture():
            self.video_path = None
            self.update_controls()
            return

        fps = self.video_capture.get(cv2.CAP_PROP_FPS)

        self.video_fps = (
            float(fps)
            if np.isfinite(fps) and fps > 0
            else 30.0
        )

        self.frame_count = max(
            0,
            int(
                self.video_capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            ),
        )

        self.frame_number = 0

        self.video_slider.setRange(
            0,
            max(0, self.frame_count - 1),
        )

        self.filename_label.setText(Path(path).name)

        self.clear_detection()
        self.update_time_label()

        self.status_label.setText("Video loaded")

        self.fps_label.setText(
            f"SOURCE: {self.video_fps:.1f} FPS   "
            f"DISPLAY: --"
        )

        self.update_controls()
        self.play_video()

    def play_video(self):
        if not self.video_path:
            return

        if self.video_capture is None:
            if not self.open_capture():
                return

            self.frame_number = 0

        if not self.is_paused:
            return

        self.is_paused = False

        self.playback_start_time = time.perf_counter()
        self.playback_start_frame = self.frame_number
        self.last_frame_due = 0.0

        self.last_display_time = 0.0
        self.display_fps = 0.0

        self.status_label.setText("Playing")

        # This timer checks whether the next frame is due.
        # It does not force a video frame every 5 ms.
        self.timer.start(5)

        self.update_controls()

    def pause_video(self):
        if self.video_capture is None or self.is_paused:
            return

        self.is_paused = True
        self.timer.stop()

        self.clear_detection()

        self.status_label.setText("Paused")
        self.update_controls()

    def stop_video(self):
        self.is_paused = True
        self.timer.stop() if hasattr(self, "timer") else None

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        self.frame_number = 0

        self.clear_detection()

        self.video_slider.setValue(0)
        self.update_time_label()

        self.video_label.clear()

        if self.video_path:
            self.video_label.setText(
                "PLAYBACK STOPPED\n\n"
                "Press Play to restart"
            )
        else:
            self.video_label.setText(
                "NO VIDEO LOADED\n\n"
                "Select Open Video to begin"
            )

        self.status_label.setText("Stopped")
        self.update_controls()

    def on_seek_started(self):
        self.is_seeking = True

    def on_slider_changed(self, position):
        if self.is_seeking:
            self.update_time_label(position)

    def on_seek_finished(self):
        if self.video_capture is None:
            self.is_seeking = False
            return

        position = self.video_slider.value()

        self.video_capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            position,
        )

        self.frame_number = position

        self.clear_detection()

        self.playback_start_time = time.perf_counter()
        self.playback_start_frame = position

        self.is_seeking = False

        if self.is_paused:
            self.show_current_frame()

    def clear_detection(self):
        self.detection_generation += 1

        self.latest_detections = []
        self.latest_detection_frame = -1
        self.inference_pending = False

        if hasattr(self, "detection_worker"):
            self.detection_worker.clear_pending()

    def on_inference_options_changed(self, *_):
        self.clear_detection()
        self.playback_start_time = time.perf_counter()
        self.playback_start_frame = self.frame_number
        self.last_frame_due = 0.0

    def on_recognition_toggled(self, enabled):
        if enabled and not self.detection_checkbox.isChecked():
            self.detection_checkbox.setChecked(True)
        self.clear_detection()
        self.status_label.setText("Face recognition enabled" if enabled else "Face recognition disabled")

    def on_detection_toggled(self, enabled):
        if not enabled and self.recognition_checkbox.isChecked():
            self.recognition_checkbox.setChecked(False)
        self.clear_detection()

        self.status_label.setText(
            "Face detection enabled"
            if enabled
            else "Face detection disabled"
        )

    def on_detection_error(self, message):
        self.inference_pending = False
        self.status_label.setText(
            f"Detection error: {message}"
        )
        print(f"Detection error: {message}")

    def on_detection_ready(
        self,
        detections,
        frame_number,
        generation,
    ):
        if generation != self.detection_generation:
            return

        if not self.detection_checkbox.isChecked():
            return

        if self.video_capture is None or self.is_paused:
            return

        if frame_number <= self.latest_detection_frame:
            return

        # Sequential mode accepts completed inference even if it took longer
        # than the usual quarter-second overlay age limit.
        if not self.no_skip_checkbox.isChecked():
            max_age = max(1, round(self.video_fps * 0.25))
            if self.frame_number - frame_number > max_age:
                return
        else:
            self.inference_pending = False

        self.latest_detections = detections
        self.latest_detection_frame = frame_number
        counts_changed = False
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            self.add_diagnostic("Detection", f"Face {max(0,x2-x1)}×{max(0,y2-y1)} px · confidence {detection['probability']}", frame_number,
                                metadata={"face_width": max(0,x2-x1), "face_height": max(0,y2-y1),
                                          "detection_confidence": detection["probability"]})
            if self.recognition_checkbox.isChecked():
                score = detection.get("best_score")
                score_text = f"{score:.3f}" if score is not None else "N/A"
                accepted = bool(detection.get("name") and detection["name"] != "Unknown")
                identity = detection.get("name") if accepted else detection.get("best_name")
                reference_id = detection.get("best_reference_id")
                if accepted and reference_id is not None:
                    for entry in self.face_registry:
                        if entry.get("id") == reference_id:
                            entry["matches"] = entry.get("matches", 0) + 1
                            counts_changed = True
                            break
                self.add_diagnostic("Recognition", f"{detection.get('name') or 'Unknown'} · best: {detection.get('best_name') or 'N/A'} · similarity {score_text} · threshold 0.55", frame_number, identity=identity, accepted=accepted,
                                    metadata={"face_width": max(0,x2-x1), "face_height": max(0,y2-y1),
                                              "detection_confidence": detection["probability"],
                                              "best_reference": detection.get("best_name") or "",
                                              "best_reference_id": reference_id or "",
                                              "similarity": score if score is not None else "",
                                              "threshold": 0.55})
        if counts_changed:
            self.refresh_reference_strip()

    @staticmethod
    def annotate_frame(frame, detections):
        landmark_colors = (
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 0),
            (0, 100, 255),
            (255, 0, 100),
        )

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            probability = detection["probability"]
            label = detection.get("name") or probability

            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            for point, color in zip(
                detection["landmarks"],
                landmark_colors,
            ):
                cv2.circle(
                    frame,
                    point,
                    1,
                    color,
                    2,
                )

        return frame

    def show_current_frame(self):
        """
        Display the selected frame while paused without advancing
        the playback position.
        """
        if self.video_capture is None:
            return

        position = self.frame_number

        ret, frame = self.video_capture.read()

        if ret:
            self.display_frame(frame)

        self.video_capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            position,
        )

    def display_frame(self, frame):
        if self.detection_checkbox.isChecked():
            if self.latest_detection_frame >= 0:
                age = (
                    self.frame_number
                    - self.latest_detection_frame
                )

                max_age = (
                    2 if self.no_skip_checkbox.isChecked()
                    else max(1, round(self.video_fps * 0.25))
                )

                if 0 <= age <= max_age:
                    frame = self.annotate_frame(
                        frame,
                        self.latest_detections,
                    )

        if self.fps_checkbox.isChecked():
            text = f"FPS: {self.display_fps:.1f}"

            cv2.putText(
                frame,
                text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (101, 216, 193),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.video_label.setPixmap(scaled)

    def update_frame(self):
        if self.video_capture is None or self.is_paused:
            return

        if self.is_seeking:
            return

        now = time.perf_counter()
        sequential = self.no_skip_checkbox.isChecked()
        if sequential:
            if self.detection_checkbox.isChecked() and self.inference_pending:
                return
            # Playback never runs faster than the source FPS, but can run
            # slower when inference cannot keep up.
            if now < self.last_frame_due:
                return
            self.last_frame_due = now + 1.0 / self.video_fps
        elapsed = now - self.playback_start_time

        target_frame = (
            self.frame_number if sequential else
            self.playback_start_frame + int(elapsed * self.video_fps)
        )

        if target_frame < self.frame_number:
            return

        # Skip frames when behind schedule.
        skip_count = target_frame - self.frame_number

        if skip_count > 0:
            # For larger jumps, seek directly rather than decoding
            # every intermediate frame.
            if skip_count > 5:
                self.video_capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    target_frame,
                )
            else:
                for _ in range(skip_count):
                    if not self.video_capture.grab():
                        self.stop_video()
                        return

            self.frame_number = target_frame

        ret, frame = self.video_capture.read()

        if not ret:
            self.stop_video()
            return

        current_frame_number = self.frame_number
        self.frame_number += 1

        if self.detection_checkbox.isChecked():
            references = []
            if self.recognition_checkbox.isChecked():
                for item in self.face_registry:
                    if not item.get("included", True) or not item.get("name", "").strip():
                        continue
                    vector = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
                    magnitude = np.linalg.norm(vector)
                    if magnitude > 0 and np.isfinite(magnitude):
                        references.append((item["id"], item["name"], (vector / magnitude).copy()))
            scale = float(self.inference_scale_combo.currentData())
            if scale != 1.0:
                inference_frame = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                inference_frame = frame
            self.detection_worker.submit(
                inference_frame,
                current_frame_number,
                self.detection_generation,
                tuple(references),
                # Original dimensions are needed to restore coordinates.
                (frame.shape[1], frame.shape[0]),
            )
            if sequential:
                self.inference_pending = True

        display_time = time.perf_counter()

        if self.last_display_time > 0:
            delta = display_time - self.last_display_time

            if delta > 0:
                instantaneous_fps = 1.0 / delta

                if self.display_fps == 0:
                    self.display_fps = instantaneous_fps
                else:
                    self.display_fps = (
                        0.85 * self.display_fps
                        + 0.15 * instantaneous_fps
                    )

        self.last_display_time = display_time

        self.display_frame(frame)

        if not self.is_seeking:
            self.video_slider.setValue(
                current_frame_number
            )

        self.update_time_label(current_frame_number)

        self.fps_label.setText(
            f"SOURCE: {self.video_fps:.1f} FPS   "
            f"DISPLAY: {self.display_fps:.1f} FPS"
        )

    def closeEvent(self, event):
        self.timer.stop()

        self.detection_worker.stop()

        # Wait for any inference already in progress to finish.
        self.detection_worker.wait()

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = BasicVideoApp()
    window.show()

    sys.exit(app.exec_())
