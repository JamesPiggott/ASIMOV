
import os

# Initialize TensorFlow before PyQt5 on Windows.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

import sys
import time
from pathlib import Path
from threading import Condition

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
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
)


class DetectionWorker(QThread):
    """
    Process face detection asynchronously.

    Only the newest pending frame is retained. The worker returns
    coordinates rather than an annotated video frame.
    """

    detection_ready = pyqtSignal(object, int, int)
    detection_error = pyqtSignal(str)

    def __init__(self, api):
        super().__init__()

        self.api = api
        self.condition = Condition()
        self.pending = None
        self.running = True

    def submit(self, frame, frame_number, generation):
        with self.condition:
            if not self.running:
                return

            self.pending = (
                frame.copy(),
                frame_number,
                generation,
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

                frame, frame_number, generation = self.pending
                self.pending = None

            try:
                result = self.api.detect_faces(frame, False)

                detections = []

                for face in result.faces:
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

                    detections.append({
                        "box": (
                            box.x1,
                            box.y1,
                            box.x2,
                            box.y2,
                        ),
                        "probability": face.prob.as_text,
                        "landmarks": landmarks,
                    })

                self.detection_ready.emit(
                    detections,
                    frame_number,
                    generation,
                )

            except Exception as exc:
                self.detection_error.emit(str(exc))


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

        # Detection state.
        self.detection_generation = 0
        self.latest_detections = []
        self.latest_detection_frame = -1

        # Build Qt UI before starting the worker.
        self.build_interface()
        self.apply_styles()

        # Import API after TensorFlow initialization and create it once.
        from application.core.API import API

        self.api = API()

        self.detection_worker = DetectionWorker(self.api)
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

        self.fps_checkbox = QCheckBox("Show FPS")
        self.fps_checkbox.setChecked(True)

        toolbar_layout.addWidget(self.detection_checkbox)
        toolbar_layout.addSpacing(14)
        toolbar_layout.addWidget(self.fps_checkbox)

        root.addWidget(toolbar)

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

        if hasattr(self, "detection_worker"):
            self.detection_worker.clear_pending()

    def on_detection_toggled(self, enabled):
        self.clear_detection()

        self.status_label.setText(
            "Face detection enabled"
            if enabled
            else "Face detection disabled"
        )

    def on_detection_error(self, message):
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

        # Avoid retaining results that are already too old.
        max_age = max(1, round(self.video_fps * 0.25))

        if self.frame_number - frame_number > max_age:
            return

        self.latest_detections = detections
        self.latest_detection_frame = frame_number

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

            if probability:
                cv2.putText(
                    frame,
                    probability,
                    (x1, y1),
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

                max_age = max(
                    1,
                    round(self.video_fps * 0.25),
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

        elapsed = now - self.playback_start_time

        target_frame = (
            self.playback_start_frame
            + int(elapsed * self.video_fps)
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
            self.detection_worker.submit(
                frame,
                current_frame_number,
                self.detection_generation,
            )

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
