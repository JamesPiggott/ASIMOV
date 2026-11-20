
import os
import html
import colorsys
import csv
import uuid

# Initialize TensorFlow before PyQt5 on Windows.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

import sys
import time
from pathlib import Path
from threading import Condition, Lock
from collections import deque

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QColor, QPen, QFont
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
    QGridLayout,
    QScrollArea,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QComboBox,
    QInputDialog,
    QToolButton,
    QMenu,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QSplitter,
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

    def submit(self, frame, frame_number, generation, references=(), original_size=None, crop_padding=0.0, crop_mode="legacy"):
        with self.condition:
            if not self.running:
                return

            self.pending = (
                frame.copy(),
                frame_number,
                generation,
                references,
                original_size,
                crop_padding,
                crop_mode,
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

                frame, frame_number, generation, references, original_size, crop_padding, crop_mode = self.pending
                self.pending = None

            try:
                worker_started = time.perf_counter()
                detection_ms = 0.0
                embedding_ms = 0.0
                crop_ms = 0.0
                arcface_ms = 0.0
                embedding_count = 0
                with self.api_lock:
                    detection_started = time.perf_counter()
                    result = self.api.detect_faces(frame, False)
                    detection_ms = (time.perf_counter() - detection_started) * 1000.0
                    # Embeddings are calculated from the same detected face objects.
                    # This runs off the GUI thread and shares the API lock with imports.
                    face_results = []
                    for face in result.faces:
                        vector = None
                        recognition_crop = None
                        if references:
                            embedding_started = time.perf_counter()
                            crop_started = time.perf_counter()
                            crop = self.api.crop_face(frame, face, crop_padding, mode=crop_mode)
                            crop_ms += (time.perf_counter() - crop_started) * 1000.0
                            if crop is not None and crop.size:
                                arcface_started = time.perf_counter()
                                recognition_crop = crop
                                raw = self.api.recognize_faces(crop)
                                arcface_ms += (time.perf_counter() - arcface_started) * 1000.0
                                if hasattr(raw, "numpy"):
                                    raw = raw.numpy()
                                vector = np.asarray(raw, dtype=np.float32).reshape(-1)
                                embedding_count += 1
                            embedding_ms += (time.perf_counter() - embedding_started) * 1000.0
                        face_results.append((face, vector, recognition_crop))

                detections = []

                for face, vector, recognition_crop in face_results:
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
                    best_reference_label = None
                    runner_up_name = None
                    runner_up_score = None
                    score_margin = None
                    best_candidate_reference_id = None
                    if vector is not None and np.all(np.isfinite(vector)):
                        magnitude = np.linalg.norm(vector)
                        if magnitude > 0:
                            normalized = vector / magnitude
                            candidates = []
                            for ref_id, identity_id, identity_name, reference_label, ref_vector in references:
                                if ref_vector.shape == normalized.shape:
                                    candidates.append((float(np.dot(normalized, ref_vector)),
                                                       identity_id, identity_name, ref_id, reference_label))
                            # Compare each reference, but rank identities using their best image.
                            # Ungrouped references retain independent identity IDs.
                            by_identity = {}
                            for score, identity_id, identity_name, reference_id, reference_label in candidates:
                                if identity_id not in by_identity or score > by_identity[identity_id][0]:
                                    by_identity[identity_id] = (score, identity_name, reference_id, reference_label)
                            ranked = sorted(by_identity.values(), key=lambda value: value[0], reverse=True)
                            best_reference_label = None
                            if ranked:
                                best_score, best_name, best_reference_id, best_reference_label = ranked[0]
                                best_candidate_reference_id = best_reference_id
                                if len(ranked) > 1:
                                    runner_up_score, runner_up_name = ranked[1][0], ranked[1][1]
                                    score_margin = best_score - runner_up_score
                                if best_score >= 0.55 and (len(ranked) == 1 or best_score - ranked[1][0] >= 0.04):
                                    name = best_name

                    # Retain a compact crop from this exact inference frame.
                    # Never crop from a later playback frame: inference is asynchronous.
                    accepted_crop = None
                    if name is not None and name != "Unknown":
                        if recognition_crop is not None and recognition_crop.size:
                            accepted_crop = cv2.resize(recognition_crop, (72, 72), interpolation=cv2.INTER_AREA)

                    detections.append({
                        "accepted_crop": accepted_crop,
                        "name": name,
                        "best_name": best_name,
                        "best_reference_label": best_reference_label,
                        "best_candidate_reference_id": best_candidate_reference_id,
                        "runner_up_name": runner_up_name,
                        "runner_up_score": runner_up_score,
                        "score_margin": score_margin,
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
                # A separate frame-level event avoids repeating timing data for each face.
                detections.append({"_performance": {
                    "detection_ms": round(detection_ms, 2),
                    "embedding_ms": round(embedding_ms, 2),
                    "crop_ms": round(crop_ms, 2),
                    "arcface_ms": round(arcface_ms, 2),
                    "worker_total_ms": round((time.perf_counter() - worker_started) * 1000.0, 2),
                    "embedding_count": embedding_count,
                    "processed_face_count": len(face_results),
                    "inference_width": frame.shape[1],
                    "inference_height": frame.shape[0],
                }})
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

    def __init__(self, api, api_lock, paths, crop_padding=0.0, crop_mode="legacy"):
        super().__init__()
        self.api = api
        self.api_lock = api_lock
        self.paths = list(paths)
        self.crop_padding = crop_padding
        self.crop_mode = crop_mode

    def run(self):
        faces, errors = [], []
        for path in self.paths:
            try:
                raw = np.fromfile(path, dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Cannot read image")
                with self.api_lock:
                    result = self.api.detect_faces(image, False)
                    count = 0
                    for face in result.faces:
                        crop = self.api.crop_face(image, face, self.crop_padding, mode=self.crop_mode)
                        if crop is None or crop.size == 0:
                            continue
                        embedding = self.api.recognize_faces(crop)
                        if hasattr(embedding, "numpy"):
                            embedding = embedding.numpy()
                        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
                        if embedding.size == 0 or not np.all(np.isfinite(embedding)):
                            continue
                        faces.append((path, crop.copy(), embedding.copy(), self.api.assess_face_quality(image, face, crop), face))
                        count += 1
                if not count:
                    errors.append(f"{Path(path).name}: no usable faces")
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        self.completed.emit(faces, errors)



class ReferenceRecropWorker(QThread):
    """Rebuild reference embeddings from original photographs without redetection."""
    completed = pyqtSignal(object, object)

    def __init__(self, api, api_lock, entries, padding, mode="legacy"):
        super().__init__()
        self.api, self.api_lock = api, api_lock
        self.entries = [(e["id"], e["source"], e["detected_face"]) for e in entries]
        self.padding = padding
        self.mode = mode

    def run(self):
        updates, errors = {}, []
        image_cache = {}
        for ref_id, path, face in self.entries:
            try:
                if path not in image_cache:
                    raw = np.fromfile(path, dtype=np.uint8)
                    image_cache[path] = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                image = image_cache[path]
                if image is None:
                    raise ValueError("Source image cannot be read")
                crop = self.api.crop_face(image, face, self.padding, mode=self.mode)
                if crop.size == 0:
                    raise ValueError("Empty crop at this padding")
                with self.api_lock:
                    raw_vector = self.api.recognize_faces(crop)
                if hasattr(raw_vector, "numpy"):
                    raw_vector = raw_vector.numpy()
                vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
                if vector.size == 0 or not np.all(np.isfinite(vector)):
                    raise ValueError("Invalid embedding")
                updates[ref_id] = (crop, vector)
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")
        self.completed.emit(updates, errors)

class ClickableVideoLabel(QLabel):
    """Video surface that toggles playback on a left click."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
        else:
            super().mousePressEvent(event)


class ReferenceFacesDialog(QDialog):
    """Immediately registered faces, editable inline with native row selection."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("ASIMOV | Reference Faces")
        self.resize(1120, 630)
        self.registry = parent.face_registry
        self.worker = None
        self.row_lookup = {}
        self.row_widgets = {}
        self._rendering_registry = False
        layout = QVBoxLayout(self)
        heading = QLabel("Reference faces")
        heading.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Select a reference to inspect its quality. Ctrl/Shift-click to select multiple rows."))

        toolbar = QHBoxLayout()
        self.upload_button = QPushButton("+ Add images")
        self.upload_button.clicked.connect(self.choose_images)
        toolbar.addWidget(self.upload_button)
        self.group_button = QPushButton("Identity ▾")
        self.group_menu = QMenu(self)
        self.group_menu.addAction("Assign identity…", self.assign_identity)
        self.group_menu.addAction("Ungroup", self.ungroup_selected)
        self.group_button.setMenu(self.group_menu)
        toolbar.addWidget(self.group_button)
        self.include_button = QPushButton("Include")
        self.include_button.clicked.connect(lambda: self.bulk_include(True))
        self.exclude_button = QPushButton("Exclude")
        self.exclude_button.clicked.connect(lambda: self.bulk_include(False))
        toolbar.addWidget(self.include_button)
        toolbar.addWidget(self.exclude_button)
        self.remove_button = QPushButton("Delete")
        self.remove_button.clicked.connect(self.remove_selected)
        toolbar.addWidget(self.remove_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.message = QLabel("Reference faces are stored for this session only.")
        layout.addWidget(self.message)

        content = QHBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setSpacing(3)
        self.list_widget.itemSelectionChanged.connect(self.update_selection_actions)
        content.addWidget(self.list_widget, 3)

        inspector = QFrame()
        inspector.setObjectName("faceInspector")
        inspector.setStyleSheet("QFrame#faceInspector {background:#1c2530; border:1px solid #35434f; border-radius:7px;}")
        details = QVBoxLayout(inspector)
        details.setContentsMargins(16, 14, 16, 14)
        details.addWidget(QLabel("FACE INSPECTOR"))
        self.inspector_preview = self.make_preview(None)
        self.inspector_preview.setFixedSize(150, 150)
        details.addWidget(self.inspector_preview, alignment=Qt.AlignHCenter)
        self.inspector_name = QLabel("Select a reference image")
        self.inspector_name.setWordWrap(True)
        details.addWidget(self.inspector_name)
        self.inspector_metrics = QTextEdit()
        self.inspector_metrics.setReadOnly(True)
        self.inspector_metrics.setFrameShape(QFrame.NoFrame)
        self.inspector_metrics.setStyleSheet("QTextEdit {background:transparent; border:none;}")
        self.inspector_metrics.setPlaceholderText("Select a reference to inspect its metrics.")
        details.addWidget(self.inspector_metrics, 1)
        self.copy_report_button = QPushButton("Copy Report")
        self.copy_report_button.clicked.connect(self.copy_inspector_report)
        self.copy_report_button.setEnabled(False)
        details.addWidget(self.copy_report_button)
        details.addStretch()
        content.addWidget(inspector, 2)
        layout.addLayout(content, 1)

        bottom = QHBoxLayout()
        self.count_label = QLabel()
        bottom.addWidget(self.count_label)
        bottom.addStretch()
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
        # Avoid re-entrant selection notifications while Qt destroys row widgets.
        self._rendering_registry = True
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.row_lookup.clear()
        self.row_widgets.clear()
        for entry in self.registry:
            entry.setdefault("id", id(entry))
            entry.setdefault("included", True)
            entry.setdefault("matches", 0)
            entry.setdefault("name", "")
            entry.setdefault("identity_id", None)
            entry.setdefault("identity_name", None)
            item = QListWidgetItem()
            item.setSizeHint(QSize(500, 92))
            self.list_widget.addItem(item)
            self.row_lookup[id(item)] = entry
            row = QWidget()
            row.setObjectName("referenceRow")
            self.row_widgets[id(item)] = row
            line = QHBoxLayout(row)
            line.setContentsMargins(8, 3, 8, 3)
            thumbnail = self.make_preview(entry.get("crop"))
            # Passive thumbnail: do not install a mouse handler or nested button.
            # Native QListWidget selection works via the row's text/blank area.
            # Interactive children inside setItemWidget caused a Windows Qt crash.
            thumbnail.setToolTip("Select this reference by clicking the row beside the image")
            thumbnail.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            line.addWidget(thumbnail)
            field = QLineEdit(entry["name"])
            field.setPlaceholderText("Reference image label")
            field.editingFinished.connect(lambda e=entry, f=field: self.rename(e, f))
            line.addWidget(field, 1)
            identity_label = QLabel(entry.get("identity_name") or "Independent")
            identity_label.setToolTip("Assigned identity (or independent reference)")
            identity_label.setMinimumWidth(95)
            line.addWidget(identity_label)
            included = QCheckBox("Include")
            included.setChecked(entry["included"])
            included.toggled.connect(lambda value, e=entry: self.set_included(e, value))
            line.addWidget(included)
            self.list_widget.setItemWidget(item, row)
        self.list_widget.blockSignals(False)
        self._rendering_registry = False
        self.update_selection_actions()
        self.refresh_count()

    def selected_entries(self):
        return [self.row_lookup[id(item)] for item in self.list_widget.selectedItems()
                if id(item) in self.row_lookup]

    def update_selection_actions(self):
        if self._rendering_registry:
            return
        has_selection = bool(self.list_widget.selectedItems())
        # Keep toolbar geometry stable; toggling visibility from a Qt selection
        # callback can trigger nested layout/selection events on Windows.
        for button in (self.include_button, self.exclude_button, self.group_button, self.remove_button):
            button.setEnabled(has_selection)
        self.count_label.setText(self._count_text())
        self.update_inspector()
        # QListWidget's default selection highlight is obscured by setItemWidget.
        # Paint the embedded row widget directly, without rebuilding the list.
        selected_ids = {id(item) for item in self.list_widget.selectedItems()}
        for item_id, row in self.row_widgets.items():
            if item_id in selected_ids:
                row.setStyleSheet(
                    "QWidget#referenceRow { background-color: #28534e; "
                    "border: 1px solid #65d8c1; border-radius: 5px; }"
                )
            else:
                row.setStyleSheet(
                    "QWidget#referenceRow { background-color: #202832; "
                    "border: 1px solid transparent; border-radius: 5px; }"
                )

    def update_inspector(self):
        selected = self.selected_entries()
        if len(selected) != 1:
            self.inspector_preview.clear()
            self.inspector_name.setText("Select one reference" if not selected else f"{len(selected)} references selected")
            self.inspector_metrics.setPlainText("Select a single row to inspect its metrics.")
            self.copy_report_button.setEnabled(False)
            return
        entry = selected[0]
        self.inspector_name.setText(entry.get("name") or "Unlabelled")
        crop = entry.get("crop")
        if crop is not None and crop.size:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            self.inspector_preview.setPixmap(QPixmap.fromImage(qimg).scaled(
                148, 148, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        metrics = entry.get("quality") or {}
        if not metrics:
            self.inspector_metrics.setPlainText("Quality metrics unavailable for this reference. Re-import to calculate them.")
            self.copy_report_button.setEnabled(True)
            return
        def fmt(key, suffix=""):
            value = metrics.get(key)
            return "N/A" if value is None else f"{value}{suffix}"
        size = metrics.get("face_size", (0, 0))
        rows = [
            ("Detection confidence", fmt("confidence")),
            ("Face size", f"{size[0]} × {size[1]} px"),
            ("Image coverage", fmt("face_coverage", "%")),
            ("Face-region sharpness (160px)", fmt("sharpness")),
            ("Face brightness (0–255)", fmt("brightness")),
            ("Facial contrast (std. dev.)", fmt("contrast")),
            ("Dark pixels (≤35)", fmt("dark_fraction", "%")),
            ("Bright pixels (≥220)", fmt("bright_fraction", "%")),
            ("Eye distance", fmt("eye_distance", " px")),
            ("Eye-line roll", fmt("eye_tilt", "°")),
            ("Nose offset / eye distance", fmt("nose_offset")),
            ("Yaw asymmetry indicator", fmt("yaw_asymmetry")),
            ("Pitch geometry indicator", fmt("pitch_indicator")),
            ("Mouth tilt", fmt("mouth_tilt", "°")),
        ]
        body = "".join(f"<p style='margin:4px 0'><span style='color:#98aab7'>{html.escape(label)}:</span> "
                       f"<b>{html.escape(value)}</b></p>" for label, value in rows)
        assessment = html.escape(str(metrics.get("assessment", "Not assessed")))
        assessment_color = {"Good": "#65d8c1", "Review": "#e9bc76", "Poor": "#e88989"}.get(assessment, "#98aab7")
        body = f"<p style='font-size:14px;color:{assessment_color}'><b>Suitability: {assessment}</b></p>" + body
        warnings = metrics.get("warnings", [])
        if warnings:
            body += "<p style='color:#e9bc76'><b>Review:</b> " + html.escape("; ".join(warnings)) + "</p>"
        else:
            body += "<p style='color:#65d8c1'>No heuristic warnings</p>"
        body += "<p style='color:#8496a5;font-size:11px'>Pose and quality indicators are heuristic, not validated recognition-accuracy scores.</p>"
        stats = entry.get("recognition_stats") or {}
        candidate_count = stats.get("candidate_count", 0)
        accepted_count = stats.get("accepted_count", 0)
        scores = stats.get("accepted_scores", [])
        margins = stats.get("accepted_margins", [])
        body += "<hr><b>REFERENCE PERFORMANCE</b>"
        body += f"<p>Best candidate: {candidate_count} · Accepted: {accepted_count}</p>"
        if scores:
            body += (f"<p>Accepted similarity — mean: {sum(scores)/len(scores):.3f}; "
                     f"maximum: {max(scores):.3f}; minimum: {min(scores):.3f}</p>")
            near = sum(1 for score in scores if score < 0.60)
            body += f"<p>Accepted near threshold (0.55–0.60): {near} / {len(scores)}</p>"
        if margins:
            body += f"<p>Mean margin to runner-up identity: {sum(margins)/len(margins):.3f}</p>"
        body += "<p style='color:#8496a5;font-size:11px'>Observed match statistics are not accuracy or false-positive rates without labeled ground truth.</p>"
        self.inspector_metrics.setHtml(body)
        self.copy_report_button.setEnabled(True)

    def copy_inspector_report(self):
        selected = self.selected_entries()
        if len(selected) != 1:
            return
        entry = selected[0]
        report = (f"ASIMOV Reference Face Report\n"
                  f"Reference: {entry.get('name') or 'Unlabelled'}\n"
                  f"Identity: {entry.get('identity_name') or 'Independent'}\n"
                  f"Included: {entry.get('included', True)}\n\n"
                  + self.inspector_metrics.toPlainText())
        QApplication.clipboard().setText(report)
        self.message.setText("Reference report copied to clipboard.")

    def _count_text(self):
        included = sum(e.get("included", True) for e in self.registry)
        identities = {e.get("identity_id") or e.get("id") for e in self.registry}
        return f"{len(self.registry)} images · {len(identities)} identities · {included} included · {len(self.selected_entries())} selected"

    def refresh_count(self):
        self.count_label.setText(self._count_text())
        self.parent().refresh_reference_strip()

    def set_included(self, entry, value):
        entry["included"] = value
        self.refresh_count()

    def rename(self, entry, field):
        entry["name"] = field.text().strip()
        self.refresh_count()

    def assign_identity(self):
        selected = self.selected_entries()
        if not selected:
            return
        existing = sorted({e.get("identity_name") for e in self.registry if e.get("identity_name")}, key=str.casefold)
        choices = ["Create new identity..."] + existing
        choice, ok = QInputDialog.getItem(self, "Assign Identity", "Identity:", choices, 0, False)
        if not ok:
            return
        if choice == choices[0]:
            name, ok = QInputDialog.getText(self, "New Identity", "Identity name:")
            if not ok or not name.strip():
                return
            name = name.strip()
        else:
            name = choice
        # Reuse existing IDs when adding images to an identity.
        existing_entry = next((e for e in self.registry if (e.get("identity_name") or "").casefold() == name.casefold()), None)
        identity_id = existing_entry["identity_id"] if existing_entry else str(uuid.uuid4())
        canonical_name = existing_entry["identity_name"] if existing_entry else name
        for entry in selected:
            entry["identity_id"] = identity_id
            entry["identity_name"] = canonical_name
        self.render_registry()
        self.message.setText(f"Assigned {len(selected)} reference(s) to {canonical_name}.")

    def ungroup_selected(self):
        selected = self.selected_entries()
        for entry in selected:
            entry["identity_id"] = None
            entry["identity_name"] = None
        self.render_registry()
        self.message.setText(f"Made {len(selected)} reference(s) independent.")

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
        self.worker = ReferenceImageWorker(self.parent().api, self.parent().api_lock, paths,
                                           self.parent().crop_padding, self.parent().crop_mode)
        self.worker.completed.connect(self.on_completed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def on_completed(self, faces, errors):
        self.upload_button.setEnabled(True)
        for path, crop, embedding, quality, face in faces:
            entry = {"name": Path(path).stem, "crop": crop, "embedding": embedding,
                     "included": True, "source": path, "matches": 0, "quality": quality,
                     "detected_face": face, "crop_padding": self.parent().crop_padding, "crop_mode": self.parent().crop_mode}
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


class MetricsGraph(QWidget):
    """Small, dependency-free, dark Qt time-series graph."""

    def __init__(self, title, keys, colors, parent=None):
        super().__init__(parent)
        self.title = title
        self.keys = keys
        self.colors = colors
        self.samples = []
        self.setMinimumHeight(175)

    def set_samples(self, samples):
        self.samples = list(samples)[-120:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#111923'))
        painter.setPen(QColor('#dce8ed'))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(13, 21, self.title)
        left, top, right, bottom = 43, 35, self.width() - 12, self.height() - 28
        if right <= left or bottom <= top:
            return
        maximum = max((float(sample.get(key, 0) or 0)
                       for sample in self.samples for key in self.keys), default=1.0)
        maximum = max(1.0, maximum * 1.12)
        for i in range(5):
            y = int(bottom - (bottom - top) * i / 4)
            painter.setPen(QPen(QColor('#293744'), 1))
            painter.drawLine(left, y, right, y)
            painter.setPen(QColor('#8397a7'))
            painter.drawText(2, y + 4, f'{maximum * i / 4:.0f}')
        if len(self.samples) >= 2:
            for key, color in zip(self.keys, self.colors):
                painter.setPen(QPen(QColor(color), 2))
                previous = None
                for i, sample in enumerate(self.samples):
                    x = int(left + (right - left) * i / (len(self.samples) - 1))
                    value = float(sample.get(key, 0) or 0)
                    y = int(bottom - (bottom - top) * min(value / maximum, 1.0))
                    if previous is not None:
                        painter.drawLine(previous[0], previous[1], x, y)
                    previous = (x, y)
        painter.setPen(QColor('#8da0ae'))
        painter.drawText(left, self.height() - 7, 'Older')
        painter.drawText(right - 38, self.height() - 7, 'Latest')


class PerformanceMonitor(QDialog):
    """Non-modal, low-frequency performance display; no model calls."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('ASIMOV | Performance Monitor')
        self.resize(860, 700)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setStyleSheet('QDialog {background:#151a21;} QLabel {color:#dce7ed;}')
        layout = QVBoxLayout(self)
        heading = QLabel('PERFORMANCE MONITOR')
        heading.setStyleSheet('font-size:18px;font-weight:700;color:#65d8c1;')
        layout.addWidget(heading)
        self.configuration = QLabel('Waiting for frames...')
        layout.addWidget(self.configuration)
        metrics = QGridLayout()
        self.values = {}
        for i, (key, label) in enumerate((
            ('fps', 'Processing FPS'), ('det', 'RetinaFace mean'),
            ('crop', 'Crop / alignment mean'), ('arc', 'ArcFace mean'),
            ('total', 'Worker mean'), ('gui', 'GUI event-loop delay'),
            ('frames', 'Processed frames'), ('empty', 'Frames without faces'),
        )):
            panel = QFrame()
            panel.setStyleSheet('QFrame {background:#202b35;border:1px solid #344653;border-radius:7px;}')
            column = QVBoxLayout(panel)
            title = QLabel(label)
            title.setStyleSheet('color:#9eb0bd;font-size:11px;border:none;')
            value = QLabel('--')
            value.setStyleSheet('color:#e8f5f2;font-size:19px;font-weight:700;border:none;')
            column.addWidget(title)
            column.addWidget(value)
            metrics.addWidget(panel, i // 4, i % 4)
            self.values[key] = value
        layout.addLayout(metrics)
        self.latency_graph = MetricsGraph('Worker latency (ms)  —  teal: detector  |  orange: ArcFace  |  blue: crop',
                                          ('detection_ms', 'arcface_ms', 'crop_ms'),
                                          ('#65d8c1', '#f1b66a', '#76a9e8'))
        self.gui_graph = MetricsGraph('GUI event-loop delay (ms)', ('gui_delay_ms',), ('#df8cdb',))
        layout.addWidget(self.latency_graph, 1)
        layout.addWidget(self.gui_graph, 1)
        buttons = QHBoxLayout()
        reset = QPushButton('Reset Metrics')
        reset.clicked.connect(parent.reset_performance_metrics)
        buttons.addWidget(reset)
        copy = QPushButton('Copy Metrics')
        copy.clicked.connect(self.copy_metrics)
        buttons.addWidget(copy)
        buttons.addStretch()
        close = QPushButton('Close')
        close.clicked.connect(self.hide)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def closeEvent(self, event):
        self.hide()
        event.ignore()

    def refresh(self):
        if not self.isVisible():
            return
        app = self.parent()
        rows = list(app.performance_samples)
        count = len(rows)
        mean = lambda key: sum(float(row.get(key, 0) or 0) for row in rows) / count if count else 0.0
        total = mean('worker_total_ms')
        values = {
            'fps': f'{1000.0 / total:.2f}' if total > 0 else '--',
            'det': f'{mean("detection_ms"):.1f} ms' if count else '--',
            'crop': f'{mean("crop_ms"):.1f} ms' if count else '--',
            'arc': f'{mean("arcface_ms"):.1f} ms' if count else '--',
            'total': f'{total:.1f} ms' if count else '--',
            'gui': f'{app.gui_delay_ms:.1f} ms',
            'frames': str(app.performance_frame_count),
            'empty': str(app.performance_empty_count),
        }
        for key, value in values.items():
            self.values[key].setText(value)
        self.configuration.setText(
            f'{app.crop_mode} | padding {app.crop_padding:.0%} | '
            f'no frame skipping: {app.no_skip_checkbox.isChecked()} | '
            f'inference: {app.inference_scale_combo.currentText()} | '
            f'window: last {count} frames')
        self.latency_graph.set_samples(rows)
        self.gui_graph.set_samples(app.gui_delay_samples)

    def copy_metrics(self):
        self.refresh()
        text = 'ASIMOV Performance Monitor\n' + self.configuration.text() + '\n'
        text += '\n'.join(f'{key}: {label.text()}' for key, label in self.values.items())
        QApplication.clipboard().setText(text)


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
        self.recent_recognitions = deque(maxlen=100)
        self.performance_samples = deque(maxlen=300)
        self.gui_delay_samples = deque(maxlen=120)
        self.performance_frame_count = 0
        self.performance_empty_count = 0
        self.gui_delay_ms = 0.0
        self.performance_monitor = None
        self.recognized_refresh_dirty = False

        # Build Qt UI before starting the worker.
        self.build_interface()
        self.apply_styles()

        # Import API after TensorFlow initialization and create it once.
        from application.core.API import API

        self.api = API()
        self.api_lock = Lock()
        self.face_registry = []
        self.crop_padding = 0.0
        self.crop_mode = "legacy"
        self.recrop_worker = None
        self._previous_crop_padding = 0.0
        # Statistics are session-only and belong to individual reference images.
        self.identity_colors = {}

        self.detection_worker = DetectionWorker(self.api, self.api_lock)
        self.detection_worker.detection_ready.connect(
            self.on_detection_ready
        )
        self.detection_worker.detection_error.connect(
            self.on_detection_error
        )
        self.detection_worker.start()
        self.reference_refresh_timer = QTimer(self)
        self.reference_refresh_timer.setInterval(400)
        self.reference_refresh_timer.timeout.connect(self.flush_reference_refresh)
        self.reference_refresh_timer.start()
        self.gui_heartbeat_last = time.perf_counter()
        self.gui_heartbeat = QTimer(self)
        self.gui_heartbeat.setInterval(100)
        self.gui_heartbeat.timeout.connect(self.measure_gui_latency)
        self.gui_heartbeat.start()
        self.reference_refresh_dirty = False

        # Playback timer.
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)

        self.update_controls()
        self.refresh_reference_strip()

    def toggle_advanced_controls(self, expanded):
        """Show or hide inference options within the original toolbar frame."""
        self.advanced_controls.setVisible(expanded)
        self.options_button.setText("⌃" if expanded else "⌄")

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

        self.video_label = ClickableVideoLabel()
        self.video_label.setObjectName("videoDisplay")
        self.video_label.clicked.connect(self.toggle_video_playback)
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
        toolbar = QWidget()
        toolbar.setObjectName("toolbarTopRow")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 6, 14, 6)
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

        toolbar_layout.addStretch(1)

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
        self.options_button = QToolButton()
        self.options_button.setText("⌄")
        self.options_button.setObjectName("toolbarChevron")
        self.options_button.setCheckable(True)
        self.options_button.setToolTip("Show or hide inference and recognition settings")
        self.options_button.toggled.connect(self.toggle_advanced_controls)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(self.options_button)

        options_toolbar = QWidget()
        options_toolbar.setObjectName("toolbarOptionsRow")
        options_layout = QHBoxLayout(options_toolbar)
        options_layout.setContentsMargins(14, 6, 14, 6)
        options_layout.setSpacing(10)

        options_layout.addStretch(1)
        options_layout.addWidget(QLabel("Inference resolution:"))
        self.inference_scale_combo = QComboBox()
        for label, scale in (("100%", 1.0), ("75%", 0.75), ("50%", 0.5), ("25%", 0.25)):
            self.inference_scale_combo.addItem(label, scale)
        self.inference_scale_combo.currentIndexChanged.connect(self.on_inference_options_changed)
        options_layout.addWidget(self.inference_scale_combo)

        options_layout.addSpacing(10)
        options_layout.addWidget(QLabel("Crop padding:"))
        self.crop_padding_combo = QComboBox()
        self.crop_padding_combo.setToolTip("Enlarge both reference and video face crops; changing this rebuilds reference embeddings")
        for label, value in (("0%", 0.0), ("+10%", 0.10), ("+15%", 0.15)):
            self.crop_padding_combo.addItem(label, value)
        self.crop_padding_combo.currentIndexChanged.connect(self.on_crop_padding_changed)
        options_layout.addWidget(self.crop_padding_combo)

        options_layout.addSpacing(10)
        options_layout.addWidget(QLabel("Recognition preprocessing:"))
        self.crop_mode_combo = QComboBox()
        self.crop_mode_combo.addItem("Legacy crop", "legacy")
        self.crop_mode_combo.addItem("5-point aligned (112×112)", "aligned")
        self.crop_mode_combo.setToolTip("Rebuilds reference embeddings from the original images; uses RetinaFace landmarks")
        self.crop_mode_combo.currentIndexChanged.connect(self.on_crop_padding_changed)
        options_layout.addWidget(self.crop_mode_combo)

        # A single frame and ordinary Qt visibility toggle: no custom resize grip,
        # nested scroll viewport, or mouse-driven geometry changes.
        self.controls_panel = QFrame()
        self.controls_panel.setObjectName("toolbar")
        panel_layout = QVBoxLayout(self.controls_panel)
        panel_layout.setContentsMargins(8, 5, 8, 5)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(toolbar)
        self.advanced_controls = options_toolbar
        panel_layout.addWidget(self.advanced_controls)
        self.advanced_controls.hide()
        self.controls_panel.setStyleSheet(
            'QFrame#toolbar QLabel { background:transparent; border:none; }'
            'QFrame#toolbar QCheckBox { background:transparent; }'
        )
        root.addWidget(self.controls_panel)

        # Two independent collapsible strips share a single row.
        # Both are closed initially to prioritize the video preview.
        strips = QHBoxLayout()
        strips.setSpacing(12)

        reference_panel = QWidget()
        reference_panel_layout = QVBoxLayout(reference_panel)
        reference_panel_layout.setContentsMargins(0, 0, 0, 0)
        reference_panel_layout.setSpacing(4)
        self.reference_toggle = QToolButton()
        self.reference_toggle.setCheckable(True)
        self.reference_toggle.setText("Reference Faces  ▸")
        self.reference_toggle.toggled.connect(self.toggle_reference_strip)
        reference_panel_layout.addWidget(self.reference_toggle)
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
        reference_panel_layout.addWidget(self.reference_scroll)
        strips.addWidget(reference_panel, 1)

        recognized_panel = QWidget()
        recognized_panel_layout = QVBoxLayout(recognized_panel)
        recognized_panel_layout.setContentsMargins(0, 0, 0, 0)
        recognized_panel_layout.setSpacing(4)
        self.recognized_toggle = QToolButton()
        self.recognized_toggle.setCheckable(True)
        self.recognized_toggle.setText("Recognized Faces  ▸")
        self.recognized_toggle.toggled.connect(self.toggle_recognized_strip)
        recognized_panel_layout.addWidget(self.recognized_toggle)
        self.recognized_scroll = QScrollArea()
        self.recognized_scroll.setWidgetResizable(False)
        self.recognized_scroll.setFixedHeight(112)
        self.recognized_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.recognized_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recognized_strip_widget = QWidget()
        self.recognized_strip_layout = QHBoxLayout(self.recognized_strip_widget)
        self.recognized_strip_layout.setContentsMargins(6, 2, 6, 2)
        self.recognized_strip_layout.setSpacing(6)
        self.recognized_scroll.setWidget(self.recognized_strip_widget)
        self.recognized_scroll.setVisible(False)
        recognized_panel_layout.addWidget(self.recognized_scroll)
        strips.addWidget(recognized_panel, 1)
        root.addLayout(strips)

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
        self.diagnostic_filter.addItems(["All", "Detection", "Recognition", "Performance"])
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

    def measure_gui_latency(self):
        now = time.perf_counter()
        self.gui_delay_ms = max(0.0, (now - self.gui_heartbeat_last) * 1000.0 - 100.0)
        self.gui_heartbeat_last = now
        self.gui_delay_samples.append({"gui_delay_ms": self.gui_delay_ms})

    def reset_performance_metrics(self):
        self.performance_samples.clear()
        self.gui_delay_samples.clear()
        self.performance_frame_count = 0
        self.performance_empty_count = 0
        self.gui_delay_ms = 0.0
        self.gui_heartbeat_last = time.perf_counter()
        self.status_label.setText("Performance metrics reset (recognition counters preserved)")
        if self.performance_monitor is not None:
            self.performance_monitor.refresh()

    def open_performance_monitor(self):
        if self.performance_monitor is None:
            self.performance_monitor = PerformanceMonitor(self)
        self.performance_monitor.show()
        self.performance_monitor.raise_()
        self.performance_monitor.activateWindow()
        self.performance_monitor.refresh()

    def flush_reference_refresh(self):
        if self.recognized_refresh_dirty:
            self.recognized_refresh_dirty = False
            if self.recognized_toggle.isChecked():
                self.refresh_recognized_strip()
        if self.reference_refresh_dirty:
            self.reference_refresh_dirty = False
            if self.reference_toggle.isChecked():
                for entry in self.face_registry:
                    label = getattr(self, "reference_count_widgets", {}).get(entry.get("id"))
                    if label is not None:
                        label.setText(str(entry.get("matches", 0)))
                        label.adjustSize()
                        label.resize(max(17, label.width()), 16)
                        label.move(label.parentWidget().width() - label.width() - 1, label.parentWidget().height() - 17)

    def refresh_reference_strip(self):
        if not hasattr(self, "reference_strip_layout") or not hasattr(self, "face_registry"):
            return
        layout = self.reference_strip_layout
        self.reference_count_widgets = {}
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        registry = self.face_registry
        groups = {}
        for entry in registry:
            key = ("identity", entry["identity_id"]) if entry.get("identity_id") else ("independent", entry.get("id"))
            groups.setdefault(key, []).append(entry)
        self.reference_count_label.setText(f"{len(registry)} image(s) · {len(groups)} identities")
        self.reference_toggle.setText(
            f"Reference Faces ({len(registry)} images · {len(groups)} identities)  "
            + ("▾" if self.reference_toggle.isChecked() else "▸")
        )
        for key, entries in groups.items():
            grouped = key[0] == "identity"
            name = entries[0].get("identity_name") if grouped else "Independent"
            container = QFrame()
            container.setObjectName("identityGroup" if grouped else "independentGroup")
            container.setStyleSheet(
                "QFrame#identityGroup { background: #202d37; border: 1px solid #4c8f88; border-radius: 5px; }"
                "QFrame#independentGroup { background: #1b242c; border: 1px solid #35434f; border-radius: 5px; }"
            )
            vertical = QVBoxLayout(container)
            vertical.setContentsMargins(5, 3, 5, 3)
            vertical.setSpacing(2)
            header = QLabel(name or "Independent")
            header.setStyleSheet("font-size: 10px; font-weight: 600; color: #65d8c1; border: none;")
            header.setToolTip(name or "Independent reference")
            vertical.addWidget(header)
            images = QHBoxLayout()
            images.setSpacing(4)
            for entry in entries:
                cell = QWidget()
                column = QVBoxLayout(cell)
                column.setContentsMargins(0, 0, 0, 0)
                column.setSpacing(1)
                preview = ReferenceFacesDialog.make_preview(entry.get("crop"))
                preview.setFixedSize(52, 52)
                if preview.pixmap() is not None:
                    preview.setPixmap(preview.pixmap().scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                count = QLabel(str(entry.get("matches", 0)), preview)
                count.setAlignment(Qt.AlignCenter)
                count.setToolTip("Accepted matches for this reference")
                count.setStyleSheet(
                    "QLabel { background-color: rgba(12, 19, 26, 220); "
                    f"color: {self.color_for_identity(entry.get('identity_name') or entry.get('name', ''))}; "
                    "border: 1px solid rgba(255,255,255,65); border-radius: 8px; "
                    "font-size: 10px; font-weight: 700; padding: 0px 3px; }"
                )
                self.reference_count_widgets[entry.get("id")] = count
                count.adjustSize()
                count.resize(max(17, count.width()), 16)
                count.move(preview.width() - count.width() - 1, preview.height() - 17)
                count.show()
                column.addWidget(preview)
                label = QLabel(entry.get("name") or "Unlabelled")
                label.setMaximumWidth(58)
                label.setToolTip(entry.get("name") or "Unlabelled")
                label.setStyleSheet("font-size: 9px; color: #8496a5;" if not entry.get("included", True) else "font-size: 9px;")
                column.addWidget(label)
                images.addWidget(cell)
            vertical.addLayout(images)
            layout.addWidget(container)
        layout.addStretch()

    def toggle_reference_strip(self, expanded):
        self.reference_scroll.setVisible(expanded)
        self.refresh_reference_strip()

    def toggle_recognized_strip(self, expanded):
        self.recognized_scroll.setVisible(expanded)
        self.refresh_recognized_strip()

    def refresh_recognized_strip(self):
        if not hasattr(self, "recognized_strip_layout"):
            return
        arrow = "▾" if self.recognized_toggle.isChecked() else "▸"
        self.recognized_toggle.setText(
            f"Recognized Faces ({len(self.recent_recognitions)} recent)  {arrow}")
        scrollbar = self.recognized_scroll.horizontalScrollBar()
        old_position = scrollbar.value()
        follow_latest = old_position >= scrollbar.maximum() - 12
        layout = self.recognized_strip_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for event in self.recent_recognitions:
            cell = QWidget()
            cell.setFixedWidth(71)
            column = QVBoxLayout(cell)
            column.setContentsMargins(2, 0, 2, 0)
            column.setSpacing(1)
            preview = ReferenceFacesDialog.make_preview(event["crop"])
            preview.setFixedSize(65, 65)
            if preview.pixmap() is not None:
                preview.setPixmap(preview.pixmap().scaled(
                    62, 62, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            preview.setToolTip(
                f'{event["name"]} · similarity {event["score"]:.3f} · '
                f'{self.format_time(event["seconds"])}')
            column.addWidget(preview)
            label = QLabel(f'{event["score"]:.3f}')
            label.setAlignment(Qt.AlignCenter)
            label.setFixedWidth(65)
            label.setStyleSheet(f'color: {self.color_for_identity(event["name"])};')
            label.setToolTip(event["name"])
            column.addWidget(label)
            layout.addWidget(cell)
        layout.addStretch()
        # Explicit content width exposes the native horizontal scrollbar.
        count = len(self.recent_recognitions)
        content_width = 12 + count * 71 + max(0, count - 1) * 6
        self.recognized_strip_widget.setFixedWidth(max(
            self.recognized_scroll.viewport().width(), content_width))
        if follow_latest:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(old_position, scrollbar.maximum()))

    def add_recognized_crop(self, detection, frame_number):
        crop = detection.get("accepted_crop")
        score = detection.get("best_score")
        if crop is None or score is None:
            return
        self.recent_recognitions.append({
            "crop": crop,
            "score": float(score),
            "name": detection["name"],
            "seconds": frame_number / self.video_fps if self.video_fps else 0,
        })
        self.recognized_refresh_dirty = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "controls_view"):
            self.controls_content.setFixedWidth(max(1, self.controls_view.viewport().width()))
        if hasattr(self, "recognized_scroll") and hasattr(self, "recent_recognitions"):
            if self.recognized_toggle.isChecked():
                self.refresh_recognized_strip()

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
        for category, line, identity, accepted, metadata in self.diagnostic_events[-600:]:
            if selected != "All" and category != selected:
                continue
            if matched_only and not accepted:
                continue
            color = self.color_for_identity(identity) if identity else (
                "#90a7b8" if category == "Detection" else "#a6adb5")
            lines.append(f'<span style="color:{color};">{html.escape(line)}</span>')
        lines = lines[-300:]  # Keep the live text widget bounded; full history remains exportable.
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
            entry.pop("recognition_stats", None)
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
                   "best_reference", "best_reference_id", "similarity", "threshold",
                   "runner_up_identity", "runner_up_similarity", "score_margin",
                   "best_candidate_reference_id", "message",
                   "video_filename", "no_frame_skipping", "inference_scale",
                   "preprocessing_mode", "crop_padding", "recognition_enabled",
                   "reference_count", "source_video_fps", "detection_ms",
                   "embedding_ms", "worker_total_ms", "embedding_count",
                    "processed_face_count", "inference_width", "inference_height",
                    "crop_ms", "arcface_ms"]
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                for category, line, identity, accepted, metadata in self.diagnostic_events:
                    row = {key: metadata.get(key, "") for key in columns}
                    row.update(event_type=category, identity=identity or "",
                               accepted=accepted, message=line,
                               video_filename=Path(self.video_path).name if self.video_path else "",
                               no_frame_skipping=self.no_skip_checkbox.isChecked(),
                               inference_scale=self.inference_scale_combo.currentData(),
                               preprocessing_mode=self.crop_mode,
                               crop_padding=self.crop_padding,
                               recognition_enabled=self.recognition_checkbox.isChecked(),
                               reference_count=sum(bool(e.get("included", True)) for e in self.face_registry),
                               source_video_fps=self.video_fps)
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

        monitor_menu = menu_bar.addMenu("&Monitor")
        monitor_action = QAction("Performance Monitor...", self)
        monitor_action.setShortcut("Ctrl+M")
        monitor_action.triggered.connect(self.open_performance_monitor)
        monitor_menu.addAction(monitor_action)
        reset_metrics_action = QAction("Reset Performance Metrics", self)
        reset_metrics_action.triggered.connect(self.reset_performance_metrics)
        monitor_menu.addAction(reset_metrics_action)

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

            QWidget#toolbarTopRow, QWidget#toolbarOptionsRow {
                background-color: #202832;
                border: none;
            }

            QToolButton#toolbarChevron {
                background: transparent;
                border: none;
                color: #b7c7d3;
                padding: 0px 6px;
                font-size: 19px;
                font-weight: 600;
            }
            QToolButton#toolbarChevron:hover { color: #65d8c1; }

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

    def on_crop_padding_changed(self, *_):
        requested_mode = str(self.crop_mode_combo.currentData())
        requested = float(self.crop_padding_combo.currentData())
        # Alignment uses the standard 112x112 template; padding only applies to legacy mode.
        if requested == self.crop_padding and requested_mode == self.crop_mode:
            return
        if self.recrop_worker is not None and self.recrop_worker.isRunning():
            self.crop_padding_combo.blockSignals(True)
            self.crop_padding_combo.setCurrentIndex(self.crop_padding_combo.findData(self.crop_padding))
            self.crop_padding_combo.blockSignals(False)
            self.crop_mode_combo.blockSignals(True)
            self.crop_mode_combo.setCurrentIndex(self.crop_mode_combo.findData(self.crop_mode))
            self.crop_mode_combo.blockSignals(False)
            return
        # Previously imported entries must carry their source detection, so
        # embeddings can be recomputed without invoking RetinaFace again.
        if any("detected_face" not in e for e in self.face_registry):
            QMessageBox.warning(self, "Reimport required",
                "Some references were imported by an older version. Reimport them before changing crop padding.")
            self.crop_padding_combo.blockSignals(True)
            self.crop_padding_combo.setCurrentIndex(self.crop_padding_combo.findData(self.crop_padding))
            self.crop_padding_combo.blockSignals(False)
            self.crop_mode_combo.blockSignals(True)
            self.crop_mode_combo.setCurrentIndex(self.crop_mode_combo.findData(self.crop_mode))
            self.crop_mode_combo.blockSignals(False)
            return
        if not self.face_registry:
            self.crop_padding = requested
            self.crop_mode = requested_mode
            self.clear_detection()
            self.status_label.setText(f"Preprocessing: {requested_mode}; padding +{requested:.0%}")
            return
        self.pause_video()
        self.clear_detection()
        self.crop_padding_combo.setEnabled(False)
        self.crop_mode_combo.setEnabled(False)
        self.reference_button.setEnabled(False)
        self.status_label.setText("Rebuilding reference embeddings from source images…")
        self.recrop_worker = ReferenceRecropWorker(self.api, self.api_lock, self.face_registry, requested, requested_mode)
        self.recrop_worker.completed.connect(lambda updates, errors, p=requested, m=requested_mode: self.on_recrop_completed(p, m, updates, errors))
        self.recrop_worker.start()

    def on_recrop_completed(self, requested, requested_mode, updates, errors):
        # The completion signal is emitted just before the QThread exits.
        # Wait before dropping its last Python reference.
        self.recrop_worker.wait()
        # Commit only if ALL embeddings succeeded, to avoid mixing settings.
        if errors or len(updates) != len(self.face_registry):
            self.crop_padding_combo.blockSignals(True)
            self.crop_padding_combo.setCurrentIndex(self.crop_padding_combo.findData(self.crop_padding))
            self.crop_padding_combo.blockSignals(False)
            self.crop_mode_combo.blockSignals(True)
            self.crop_mode_combo.setCurrentIndex(self.crop_mode_combo.findData(self.crop_mode))
            self.crop_mode_combo.blockSignals(False)
            QMessageBox.warning(self, "Crop experiment", "Could not rebuild every reference:\n" +
                                "\n".join(errors[:15]))
            self.status_label.setText("Crop padding unchanged")
        else:
            for entry in self.face_registry:
                entry["crop"], entry["embedding"] = updates[entry["id"]]
                entry["crop_padding"] = requested
                entry["crop_mode"] = requested_mode
                entry["matches"] = 0
                entry.pop("recognition_stats", None)
            self.crop_padding = requested
            self.crop_mode = requested_mode
            self.clear_diagnostics()
            self.recent_recognitions.clear()
            self.refresh_recognized_strip()
            self.refresh_reference_strip()
            self.status_label.setText(f"Preprocessing: {requested_mode}; padding +{requested:.0%}; embeddings rebuilt; counters reset")
        self.crop_padding_combo.setEnabled(True)
        self.crop_mode_combo.setEnabled(True)
        self.reference_button.setEnabled(True)
        self.recrop_worker = None

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

    def toggle_video_playback(self):
        if not self.video_path:
            return
        if self.is_paused:
            self.play_video()
        else:
            self.pause_video()

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

        if self.no_skip_checkbox.isChecked():
            self.inference_pending = False

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

        # Strip internal timing payload before drawing boxes or counting faces.
        timing = next((d["_performance"] for d in detections if "_performance" in d), None)
        detections = [d for d in detections if "_performance" not in d]
        if timing is not None:
            self.performance_samples.append(timing)
            self.performance_frame_count += 1
            if not timing.get("processed_face_count", 0):
                self.performance_empty_count += 1
            self.add_diagnostic(
                "Performance",
                f"Detector {timing['detection_ms']:.1f} ms · crops + ArcFace {timing['embedding_ms']:.1f} ms · worker {timing['worker_total_ms']:.1f} ms",
                frame_number, metadata=timing)
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
                candidate_id = detection.get("best_candidate_reference_id")
                if candidate_id is not None:
                    candidate_entry = next((e for e in self.face_registry if e.get("id") == candidate_id), None)
                    if candidate_entry is not None:
                        stats = candidate_entry.setdefault("recognition_stats", {
                            "candidate_count": 0, "accepted_count": 0,
                            "accepted_scores": [], "accepted_margins": []})
                        stats["candidate_count"] += 1
                        if accepted and score is not None:
                            stats["accepted_count"] += 1
                            stats["accepted_scores"].append(float(score))
                            margin = detection.get("score_margin")
                            if margin is not None:
                                stats["accepted_margins"].append(float(margin))
                if accepted:
                    self.add_recognized_crop(detection, frame_number)
                if accepted and reference_id is not None:
                    for entry in self.face_registry:
                        if entry.get("id") == reference_id:
                            entry["matches"] = entry.get("matches", 0) + 1
                            counts_changed = True
                            break
                self.add_diagnostic("Recognition", f"{detection.get('name') or 'Unknown'} · best reference: {detection.get('best_reference_label') or 'N/A'} · similarity {score_text} · threshold 0.55", frame_number, identity=identity, accepted=accepted,
                                    metadata={"face_width": max(0,x2-x1), "face_height": max(0,y2-y1),
                                              "detection_confidence": detection["probability"],
                                              "best_reference": detection.get("best_reference_label") or "",
                                              "best_reference_id": reference_id or "",
                                              "similarity": score if score is not None else "",
                                              "threshold": 0.55,
                                              "runner_up_identity": detection.get("runner_up_name") or "",
                                              "runner_up_similarity": detection.get("runner_up_score") if detection.get("runner_up_score") is not None else "",
                                              "score_margin": detection.get("score_margin") if detection.get("score_margin") is not None else "",
                                              "best_candidate_reference_id": candidate_id or ""})
        if counts_changed:
            self.reference_refresh_dirty = True

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
                    if not item.get("included", True) or not (item.get("identity_name") or item.get("name", "")).strip():
                        continue
                    vector = np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
                    magnitude = np.linalg.norm(vector)
                    if magnitude > 0 and np.isfinite(magnitude):
                        references.append((item["id"], item.get("identity_id") or str(item["id"]),
                                           item.get("identity_name") or item["name"],
                                           item["name"], (vector / magnitude).copy()))
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
                self.crop_padding,
                self.crop_mode,
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

        if self.recrop_worker is not None and self.recrop_worker.isRunning():
            self.recrop_worker.wait()

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
