# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:19:57 2025

@author: minjoo
"""

import sys, os, datetime, time, logging
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QLabel, QStackedWidget, QHBoxLayout, QDialog, QGridLayout,
    QLineEdit, QCheckBox, QGroupBox, QComboBox, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial.tools import list_ports
import numpy as np
import pyqtgraph as pq

# variable setting
IMG_PATH = "image/"
IMG_GRASP = "grasp.png"
IMG_HOLDING = "holding.png"
IMG_RELEASE = "release.png"
IMG_REST = "rest.png"

class PreparationPage(QWidget):
    def __init__(self, switch_to_main_callback):
        super().__init__()

        layout = QVBoxLayout(self)

        # Top bar
        top_bar = QHBoxLayout()
        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(switch_to_main_callback)
        top_bar.addWidget(self.back_btn, alignment=Qt.AlignLeft)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Central layout (left: setup / right: graph)
        central_layout = QHBoxLayout()

        # Left side (Setup)
        self.setup_group = QGroupBox("SETUP")
        self.setup_group.setFont(self._font(bold=True, size=13))
        setup_layout = QVBoxLayout(self.setup_group)

        # Device group
        self.device_group = QGroupBox("Device")
        self.device_group.setFont(self._font(bold=True, size=12))
        device_layout = QGridLayout(self.device_group)

        self.label_5 = QLabel("Select Board:")
        self.label_5.setFont(self._font())
        device_layout.addWidget(self.label_5, 0, 0, 1, 2)

        self.cyton_CB = QCheckBox("CYTON")
        self.cyton_CB.setFont(self._font())
        device_layout.addWidget(self.cyton_CB, 1, 0)

        self.daisy_CB = QCheckBox("with DAISY")
        self.daisy_CB.setFont(self._font())
        device_layout.addWidget(self.daisy_CB, 1, 1)

        self.label_4 = QLabel("Select Port:")
        self.label_4.setFont(self._font())
        device_layout.addWidget(self.label_4, 2, 0, 1, 2)

        self.port_COMB = QComboBox()
        self.port_COMB.setFont(self._font())
        device_layout.addWidget(self.port_COMB, 3, 0, 1, 2)

        self.connect_PB = QPushButton("Connect")
        self.connect_PB.setFont(self._font())
        device_layout.addWidget(self.connect_PB, 4, 0)

        self.refresh_PB = QPushButton("Refresh")
        self.refresh_PB.setFont(self._font())
        device_layout.addWidget(self.refresh_PB, 4, 1)

        self.label_6 = QLabel("Streaming:")
        self.label_6.setFont(self._font())
        device_layout.addWidget(self.label_6, 5, 0, 1, 2)

        self.start_PB = QPushButton("Start stream")
        self.start_PB.setFont(self._font())
        device_layout.addWidget(self.start_PB, 6, 0)

        self.stop_PB = QPushButton("Stop stream")
        self.stop_PB.setFont(self._font())
        device_layout.addWidget(self.stop_PB, 6, 1)

        self.start_PB.clicked.connect(self.start_stream)
        self.stop_PB.clicked.connect(self.stop_stream)

        # Measurement group
        self.measurement_group = QGroupBox("Measurement")
        self.measurement_group.setFont(self._font(bold=True, size=12))
        measurement_layout = QGridLayout(self.measurement_group)

        self.label_sec = QLabel("Seconds:")
        self.label_sec.setFont(self._font())
        measurement_layout.addWidget(self.label_sec, 0, 0)
        self.time_LE = QLineEdit()
        measurement_layout.addWidget(self.time_LE, 0, 1)

        self.label_buf = QLabel("Buffer size:")
        self.label_buf.setFont(self._font())
        measurement_layout.addWidget(self.label_buf, 1, 0)
        self.buffer_LE = QLineEdit()
        measurement_layout.addWidget(self.buffer_LE, 1, 1)

        self.label_relfft = QLabel("Relative FFT:")
        self.label_relfft.setFont(self._font())
        measurement_layout.addWidget(self.label_relfft, 2, 0)
        self.fft_CB = QCheckBox("FFT")
        self.fft_CB.setFont(self._font())
        measurement_layout.addWidget(self.fft_CB, 2, 1)

        self.label_maxfreq = QLabel("Max Freq:")
        self.label_maxfreq.setFont(self._font())
        measurement_layout.addWidget(self.label_maxfreq, 3, 0)
        self.freq_LE = QLineEdit()
        measurement_layout.addWidget(self.freq_LE, 3, 1)

        # Recording group
        self.recording_group = QGroupBox("Recording")
        self.recording_group.setFont(self._font(bold=True, size=12))
        recording_layout = QGridLayout(self.recording_group)

        self.rec_start_PB = QPushButton("Start")
        recording_layout.addWidget(self.rec_start_PB, 0, 0)
        self.rec_stop_PB = QPushButton("Stop")
        recording_layout.addWidget(self.rec_stop_PB, 0, 1)

        self.imp_start_PB = QPushButton("Imp Start")
        recording_layout.addWidget(self.imp_start_PB, 1, 0)
        self.imp_stop_PB = QPushButton("Imp Stop")
        recording_layout.addWidget(self.imp_stop_PB, 1, 1)

        self.sound_PB = QPushButton("Play Sound")
        recording_layout.addWidget(self.sound_PB, 2, 0)
        self.null_PB = QPushButton("NULL")
        recording_layout.addWidget(self.null_PB, 2, 1)

        # Add all groups
        setup_layout.addWidget(self.device_group)
        setup_layout.addWidget(self.measurement_group)
        setup_layout.addWidget(self.recording_group)

        # Channel layout (new!)
        self.channel_widget = QWidget()
        self.channel_layout = QGridLayout(self.channel_widget)

        labels = [
            (0, 1, "F3"), (0, 2, "Fz"), (0, 3, "F4"), (1, 2, "GND"), 
            (2, 0, "T3"), (2, 1, "C3"), (2, 2, "Cz"), (2, 3, "C4"), (2, 4, "T4"),
            (3, 0, "T5"), (3, 1, "P3"), (3, 2, "Pz"), (3, 3, "P4"), (3, 4, "T6"),
            (4, 1, "O1"), (4, 3, "O2")
        ]

        for row, col, text in labels:
            label = QLabel(text)
            label.setAlignment(Qt.AlignCenter)
            label.setFont(self._font(bold=True, size=10))
            self.channel_layout.addWidget(label, row, col)

        for row_col in [(3,1), (3,3)]:
            spacer = QLabel("")
            self.channel_layout.addWidget(spacer, *row_col)

        setup_layout.addWidget(self.channel_widget)

        central_layout.addWidget(self.setup_group)

        # Right side (Graphs)
        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout(self.graph_widget)
        central_layout.addWidget(self.graph_widget)

        layout.addLayout(central_layout)

        # Bottom bar
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        self.skip_btn = QPushButton("건너뛰기")
        self.skip_btn.setFixedSize(150, 40)
        self.skip_btn.clicked.connect(switch_to_main_callback)
        bottom_bar.addWidget(self.skip_btn, alignment=Qt.AlignCenter)
        bottom_bar.addStretch()
        layout.addLayout(bottom_bar)

        # Timers and Board setup
        self.timer = QTimer()
        self.timer.timeout.connect(switch_to_main_callback)
        self.timer.setSingleShot(True)
        self.timer.start(10000)

        self.setup_board()
        self.InitGraph()
        self.RefreshSerialPort()

    def _font(self, bold=False, size=10):
        font = self.font()
        font.setFamily("Malgun Gothic")
        font.setPointSize(size)
        font.setBold(bold)
        return font

    def setup_board(self):
        self.Board = None
        self.Port = None
        self.cyton_CB.toggle()
        self.board_type = 0

        self.port_COMB.activated[str].connect(self.SelectSerialPort)
        self.connect_PB.clicked.connect(self.btnConnect)
        self.refresh_PB.clicked.connect(self.RefreshSerialPort)

    def SelectSerialPort(self, text):
        self.Port = text

    def RefreshSerialPort(self):
        # 보드 세션이 열려 있으면 해제하지 말고 그대로 둠
        serial_objs = list_ports.comports()
        ports = [p.device for p in serial_objs]
        self.port_COMB.clear()
        for port in ports:
            self.port_COMB.addItem(port)

    def btnConnect(self):
        if not self.Port:
            QMessageBox.warning(self, "Warning", "COM 포트를 먼저 선택하세요.")
            return
        try:
            BoardShim.enable_dev_board_logger()
            params = BrainFlowInputParams()
            params.serial_port = self.Port
            self.Board = BoardShim(self.board_type, params)
            self.Board.prepare_session()
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            return
    
        self.exg_channels = BoardShim.get_exg_channels(self.board_type)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_type)
        QMessageBox.information(self, "Success", "Board connected and session prepared!")


    def InitGraph(self):
        self.plots = []
        self.curves = []
        for i in range(16):
            p = pq.PlotWidget(labels={"left": f"CH{i+1}"})
            p.setYRange(-2200, 2200)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setMenuEnabled('left', False)
            p.setMouseEnabled(x=False, y=False)
            self.graph_layout.addWidget(p)
            self.plots.append(p)
            self.curves.append(p.plot())

    # ① 스트림 시작
    def start_stream(self):
        if not self.Board:                       # 보드가 없는 경우
            QMessageBox.warning(self, "Warning", "먼저 Connect 를 눌러 보드 세션을 열어주세요.")
            return
        # 기존 타이머가 살아 있으면 중지
        if hasattr(self, "graph_timer") and self.graph_timer.isActive():
            self.graph_timer.stop()
    
        try:
            ring_size = int(self.buffer_LE.text()) if self.buffer_LE.text() else 450000
            self.Board.start_stream(ring_size)
        except Exception as e:
            QMessageBox.critical(self, "Stream Error", str(e))
            return
    
        # 50 ms 간격으로 그래프 업데이트
        self.graph_timer = QTimer(self)
        self.graph_timer.timeout.connect(self.update_graph)
        self.graph_timer.start(50)
    
    # ② 스트림 정지
    def stop_stream(self):
        if not self.Board:
            return
        try:
            self.Board.stop_stream()
        except Exception:
            pass
        if hasattr(self, "graph_timer"):
            self.graph_timer.stop()
    
    # ③ 그래프 갱신
    def update_graph(self):
        """
        sampling_rate 의 1/5 구간(0.2 s)만 가져와 실시간 플롯을 갱신
        """
        try:
            window_points = max(1, self.sampling_rate // 5)
            data = self.Board.get_current_board_data(window_points)
            for idx, ch in enumerate(self.exg_channels):
                # data shape: (num_channels, N)
                if data.shape[1] >= window_points:
                    self.curves[idx].setData(data[ch])
        except Exception:
            pass



# ----- Main Page -----
class MainPage(QWidget):
    def __init__(self, switch_page_callback, open_test_setting_callback, back_callback):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn, alignment=Qt.AlignLeft)

        layout.addStretch()

        # 버튼 생성
        self.buttons = []
        names = ["연습", "웹캠 연결", "오프라인 수집", "온라인 수집", "영상 보기"]
        for name in names:
            btn = QPushButton(name)
            btn.setFixedSize(150, 40)
            if name == "오프라인 수집":
                btn.clicked.connect(open_test_setting_callback)
            else:
                btn.clicked.connect(lambda _, n=name: switch_page_callback(n))
            layout.addWidget(btn, alignment=Qt.AlignCenter)
            self.buttons.append(btn)

        layout.addStretch()
    
    def closeEvent(self, event):
        if hasattr(self.prep_page, "graph_timer") and self.prep_page.graph_timer.isActive():
            self.prep_page.graph_timer.stop()
        if self.prep_page.Board is not None and self.prep_page.Board.is_prepared():
            try:
                self.prep_page.stop_stream()
            except Exception:
                pass
            self.prep_page.Board.release_session()
        super().closeEvent(event)


# ----- Webcam Page -----
class WebcamThread(QThread):
    frame_updated = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.frame_updated.emit(img)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class WebcamPage(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        layout = QVBoxLayout()

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(self.handle_back)
        layout.addWidget(self.back_btn, alignment=Qt.AlignLeft)

        self.label = QLabel("웹캠을 켜는 중...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)

        self.back_callback = back_callback

        self.thread = WebcamThread()
        self.thread.frame_updated.connect(self.update_image)
        self.thread.start()

    def update_image(self, img):
        self.label.setPixmap(QPixmap.fromImage(img))

    def handle_back(self):
        self.thread.stop()
        self.back_callback()


# ----- tmp simple page -----
class SimplePage(QWidget):
    def __init__(self, title, back_callback):
        super().__init__()
        layout = QVBoxLayout()

        back_btn = QPushButton("←")
        back_btn.setFixedSize(100, 40)
        back_btn.clicked.connect(back_callback)
        layout.addWidget(back_btn, alignment=Qt.AlignLeft)

        label = QLabel(f"여기는 {title} 페이지 입니다.")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self.setLayout(layout)


# ----- Test Page -----
class TestSettingDialog(QDialog):
    start_test_signal = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Settings")
        self.setFixedSize(400, 300)

        layout = QVBoxLayout()

        grid = QGridLayout()

        self.grasp_time = QLineEdit()
        self.holding_time = QLineEdit()
        self.release_time = QLineEdit()
        self.rest_time = QLineEdit()
        self.trials = QLineEdit()
        self.prosthetic_control = QCheckBox("Prosthetic Control")

        grid.addWidget(QLabel("Grasp Time (s):"), 0, 0)
        grid.addWidget(self.grasp_time, 0, 1)
        grid.addWidget(QLabel("Holding Time (s):"), 1, 0)
        grid.addWidget(self.holding_time, 1, 1)
        grid.addWidget(QLabel("Release Time (s):"), 2, 0)
        grid.addWidget(self.release_time, 2, 1)
        grid.addWidget(QLabel("Rest Time (s):"), 3, 0)
        grid.addWidget(self.rest_time, 3, 1)
        grid.addWidget(QLabel("Total Trials:"), 4, 0)
        grid.addWidget(self.trials, 4, 1)

        layout.addLayout(grid)
        layout.addWidget(self.prosthetic_control)

        button_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        
        self.start_btn.clicked.connect(self.send_settings)

        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.disconnect_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        
    def send_settings(self):
        settings = {
            'grasp': int(self.grasp_time.text()),
            'holding': int(self.holding_time.text()),
            'release': int(self.release_time.text()),
            'rest': int(self.rest_time.text()),
            'trials': int(self.trials.text())
        }
        self.start_test_signal.emit(settings)
        self.accept()

class TestPage(QWidget):
    def __init__(self, back_callback, settings):
        super().__init__()
        self.settings = settings
        self.current_trial = 0
        self.sequence = [
            ("image/grasp.png", self.settings['grasp']),
            ("image/holding.png", self.settings['holding']),
            ("image/release.png", self.settings['release']),
            ("image/rest.png", self.settings['rest'])
        ]
        self.seq_index = 0

        layout = QVBoxLayout()

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn, alignment=Qt.AlignLeft)

        self.label = QLabel("Test Starting...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_step)
        self.next_step()

    def next_step(self):
        if self.current_trial >= self.settings['trials']:
            self.label.setText("Test Completed!")
            self.timer.stop()
            return
    
        img_path, duration = self.sequence[self.seq_index]
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio))
        else:
            self.label.setText(f"Image not found: {img_path}")
    
        self.seq_index += 1
    
        if self.seq_index >= len(self.sequence):
            self.seq_index = 0
            self.current_trial += 1
    
        self.timer.start(duration * 1000)


# ----- Video -----
class VideoThread(QThread):
    frame_updated = pyqtSignal(QImage)
    finished = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_updated.emit(img)
            time.sleep(1/30)  # 프레임 속도 (대충 30fps 가정)
        cap.release()
        self.finished.emit()

    def stop(self):
        self.running = False
        self.wait()

class VideoPage(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        layout = QVBoxLayout()

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(self.handle_back)
        layout.addWidget(self.back_btn, alignment=Qt.AlignLeft)

        self.label = QLabel("Loading video...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)

        self.back_callback = back_callback

        self.thread = VideoThread("video/test.mp4")
        self.thread.frame_updated.connect(self.update_image)
        self.thread.finished.connect(self.handle_back)
        self.thread.start()

    def update_image(self, img):
        self.label.setPixmap(QPixmap.fromImage(img))

    def handle_back(self):
        if self.thread.isRunning():
            self.thread.stop()
        self.back_callback()


# ----- Main -----
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Collector")
        self.setFixedSize(1500, 1300)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_history = []

        self.prep_page = PreparationPage(self.show_main)
        self.main_page = MainPage(self.show_named_page, self.open_test_setting, self.go_back)

        self.stack.addWidget(self.prep_page)
        self.stack.addWidget(self.main_page)

        self.stack.setCurrentWidget(self.prep_page)

    def show_main(self):
        self.page_history.append(self.stack.currentWidget())
        self.stack.setCurrentWidget(self.main_page)

    def show_named_page(self, name):
        if name == "웹캠 연결":
            page = WebcamPage(self.go_back)
        elif name == "영상 보기":
            page = VideoPage(self.go_back)
        else:
            page = SimplePage(name, self.go_back)

        self.stack.addWidget(page)
        self.page_history.append(self.stack.currentWidget())
        self.stack.setCurrentWidget(page)

    def open_test_setting(self):
        dialog = TestSettingDialog()
        dialog.start_test_signal.connect(self.start_test)
        dialog.exec_()

    def start_test(self, settings):
        self.page_history.append(self.stack.currentWidget())
        self.test_page = TestPage(self.go_back, settings)
        self.stack.addWidget(self.test_page)
        self.stack.setCurrentWidget(self.test_page)

    def go_back(self):
        if self.page_history:
            prev_page = self.page_history.pop()
            self.stack.setCurrentWidget(prev_page)

# ----- Main Function -----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
