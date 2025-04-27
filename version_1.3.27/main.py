import sys, os, datetime, time, logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QLabel, QStackedWidget, QDesktopWidget, QFileDialog, QMessageBox, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5 import uic
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial.tools import list_ports
import numpy as np
import pyqtgraph as pq

class PreparationPage(QWidget):
    def __init__(self, switch_to_main_callback):
        super().__init__()

        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(switch_to_main_callback)
        top_bar.addWidget(self.back_btn, alignment=Qt.AlignLeft)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # UI 불러오기 (중앙 내용)
        uic.loadUi("ui/CJ_brainflow_ui_v0_widget.ui", self)
        central = self.findChild(QWidget, "centralwidget")
        if central:
            central.setFixedHeight(970)
        layout.addWidget(self.findChild(QWidget, "centralwidget")) if self.findChild(QWidget, "centralwidget") else None

        # 하단 건너뛰기 버튼
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()

        self.skip_btn = QPushButton("건너뛰기")
        self.skip_btn.setFixedSize(150, 40)
        self.skip_btn.clicked.connect(switch_to_main_callback)
        bottom_bar.addWidget(self.skip_btn, alignment=Qt.AlignCenter)

        bottom_bar.addStretch()
        layout.addLayout(bottom_bar)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(switch_to_main_callback)
        self.timer.setSingleShot(True)
        self.timer.start(10000)

        self.setup_board()
        self.InitGraph()
        self.RefreshSerialPort()

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
        if self.Board:
            self.Board.release_session()
        ports = [p.device for p in list_ports.comports()]
        self.port_COMB.clear()
        for port in ports:
            self.port_COMB.addItem(port)

    def btnConnect(self):
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level=logging.DEBUG)

        params = BrainFlowInputParams()
        params.serial_port = self.Port

        self.Board = BoardShim(self.board_type, params)
        self.Board.prepare_session()

        self.exg_channels = BoardShim.get_exg_channels(self.board_type)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_type)

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

class MainPage(QWidget):
    def __init__(self, switch_page_callback, back_callback):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.back_btn = QPushButton("←")
        self.back_btn.setFixedSize(100, 40)
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn, alignment=Qt.AlignLeft)

        layout.addStretch()
        for name in ["연습", "오프라인 수집", "온라인 수집", "영상 보기"]:
            btn = QPushButton(name)
            btn.setFixedSize(150, 40)
            btn.clicked.connect(lambda _, n=name: switch_page_callback(n))
            layout.addWidget(btn, alignment=Qt.AlignCenter)
        layout.addStretch()

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Collector")
        self.setFixedSize(1400, 1100)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_history = []

        self.prep_page = PreparationPage(self.show_main)
        self.main_page = MainPage(self.show_named_page, self.go_back)

        self.stack.addWidget(self.prep_page)
        self.stack.addWidget(self.main_page)

        self.stack.setCurrentWidget(self.prep_page)

    def show_main(self):
        self.page_history.append(self.stack.currentWidget())
        self.stack.setCurrentWidget(self.main_page)

    def show_named_page(self, name):
        page = SimplePage(name, self.go_back)
        self.stack.addWidget(page)
        self.page_history.append(self.stack.currentWidget())
        self.stack.setCurrentWidget(page)

    def go_back(self):
        if self.page_history:
            prev_page = self.page_history.pop()
            self.stack.setCurrentWidget(prev_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())