import sys
import cv2
import pathlib
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QMenuBar,
    QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from application.core.API import API


class BasicVideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basic Video Application")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(300, 200)

        # Initialize API
        self.api = API()

        # Menu
        menu_bar = QMenuBar(self)
        main_menu = menu_bar.addMenu("Main")
        load_action = QAction("Load Video", self)
        load_action.triggered.connect(self.load_video)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        main_menu.addAction(load_action)
        main_menu.addAction(quit_action)
        self.setMenuBar(menu_bar)

        settings_menu = menu_bar.addMenu("Settings")
        self.show_fps_action = QAction("Show FPS", self, checkable=True)
        self.show_fps_action.setChecked(True)
        self.perform_detection_action = QAction("Perform Detection", self, checkable=True)
        self.perform_detection_action.setChecked(False)
        settings_menu.addAction(self.show_fps_action)
        settings_menu.addAction(self.perform_detection_action)

        # UI Elements
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.sliderMoved.connect(self.set_position)

        self.set_button_style([self.play_button, self.pause_button, self.stop_button])

        # Layouts
        video_controls_layout = QHBoxLayout()
        video_controls_layout.addStretch(1)
        video_controls_layout.addWidget(self.play_button)
        video_controls_layout.addWidget(self.pause_button)
        video_controls_layout.addWidget(self.stop_button)
        video_controls_layout.addStretch(1)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.video_label)
        controls_layout.addWidget(self.video_slider)
        controls_layout.addLayout(video_controls_layout)

        container = QWidget()
        container.setLayout(controls_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.video_capture = None
        self.is_paused = False
        self.last_frame_time = time.time()

    def set_button_style(self, buttons):
        for button in buttons:
            button.setStyleSheet(
                "QPushButton { background-color: rgba(50, 50, 50, 150); color: white; border: none; }"
            )
            button.setFixedHeight(30)
            button.setFixedWidth(70)
            button.setAutoFillBackground(True)
            button.setAttribute(Qt.WA_TranslucentBackground)

    def load_video(self):
        default_dir = str(pathlib.Path(__file__).parent.parent.parent / 'application' / 'test' / 'sample_videos')
        video_path, _ = QFileDialog.getOpenFileName(self, "Load Video", default_dir, "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_capture = cv2.VideoCapture(video_path)
            self.timer.start(20)
            self.video_slider.setRange(0, int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    def play_video(self):
        self.is_paused = False

    def pause_video(self):
        self.is_paused = True

    def stop_video(self):
        self.is_paused = True
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            self.video_label.clear()
            self.video_slider.setValue(0)

    def set_position(self, position):
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)

    def update_frame(self):
        if self.video_capture is not None and not self.is_paused:
            ret, frame = self.video_capture.read()
            if ret:
                if self.perform_detection_action.isChecked():
                    result = self.api.detect_faces(frame, True)
                    frame = result.annotated_image

                if self.show_fps_action.isChecked():
                    # Calculate FPS based on the time taken to process the frame
                    current_time = time.time()
                    fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time

                    # Draw FPS on the frame
                    text = f'FPS: {fps:.2f}'
                    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.putText(frame, text, (frame.shape[1] - text_width - 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(
                    self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
                )
                self.video_label.setPixmap(QPixmap.fromImage(p))
                self.video_slider.setValue(int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)))
            else:
                self.video_capture.release()
                self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BasicVideoApp()
    window.show()
    sys.exit(app.exec_())
