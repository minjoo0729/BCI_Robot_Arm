#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:04:14 2024

@author: alfred
"""

#%% Import Modules:
# OS
import os, sys, time, datetime, logging

from serial.tools import list_ports
from threading import Timer

# PyQt5
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QMainWindow,
                             QMessageBox, QFileDialog,)
from PyQt5.QtCore import QTimer
from PyQt5 import uic

# BrainFlow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
# from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# PyQtGraph
from pyqtgraph.Qt import QtCore
import pyqtgraph as pq

# Rest modules
import numpy as np
from playsound import playsound

# TODO: IMPORT FFT
# TODO: IMPORT neural network module
from os import environ

print("Import Complete!")
#%% Variables:
    
main_form_class = uic.loadUiType("ui/CJ_brainflow_ui_v0.ui")[0] # load UI data
FPS_DRAWING = 15
_INIT_MAX_FREQ = 70 # FFT max Frequency
_DISPLAY_TIME = 5 # Window Time
_LOAD_NN_TIME = 1

_INIT_BOARD = 0 # CYTON

_NAMES = ["F3","Fz","F4",
          "T3","C3","Cz","C4","T4",
          "T5","P3","Pz","P4","T6",
          "O1","O2","GND"]

#%% Resizing function for high magnifition

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
    
#%% Main Window class

class MainWindow(QMainWindow, main_form_class):
    # Constructor:
        
    def __init__(self):
        self.__version__ = "v0"
        
        super().__init__() # Inherited from QMainWindow
        
        # UI initializing
        self.setupUi(self)
        self.setUIcenter() # Place window into center
        
        #####################################
        # member variable initializing
        #####################################
        self.StartTime = str(datetime.datetime.now())
        
        self.DrawingTimer = None # Graph Timer
        self.BufferingTimer = None
        self.DeepLearningTimer = None # Neural Network Inference Timer - Not used
        
        self.Port = None # Serial Port for OpenBCI Board
        self.board = None # Board Type
        self.storeData = None # Buffer
        self.overflowCount = 0 # Overflow of EEG - Not used
        
        # Default Board: CYTON
        self.cyton_CB.toggle()
        self.BOARD = _INIT_BOARD
        
        ################################
        # EVENT Binding - Buttons
        ################################
        self.start_PB.clicked.connect(self.btnStartStream) # Start Stream
        self.stop_PB.clicked.connect(self.btnStopStream) # Stop Stream
        
        self.port_COMB.activated[str].connect(self.SelectSerialPort) # Select Serial Port
        self.connect_PB.clicked.connect(self.btnConnect) # Connect to Device
        self.refresh_PB.clicked.connect(self.RefreshSerialPort) # Refresh the Serial Port
        
        self.cyton_CB.clicked.connect(self.ChangeCyton) # Change the Board Type - Cyton
        self.daisy_CB.clicked.connect(self.ChangeDaisy) # Change the Board Type - Daisy
        
        self.Imp_start_PB.clicked.connect(self.startImp) # Calculate the Impedance of each channels
        self.Imp_stop_PB.clicked.connect(self.stopImp) # Stop calculate the Impedance of siggnals
        
        self.Sound_PB.clicked.connect(self.playSound)
        
        # TODO: Saving - Save Specific Event From Buffer
        # self.rec_start_PB.clicked.connect(self.testButton) # For now, Test the 10-20 System
        # self.rec_stop_PB.clicked.connect(None) # Recording Stop
        
        ################################
        # Line Edit
        ################################
        self.time_LE.setText(str(_DISPLAY_TIME)) # Window Display Time
        self.freq_LE.setText(str(_INIT_MAX_FREQ)) # FFT Max Frequency
        self.buffer_LE.setText("450000") # Ring Buffer -> Default: 30min
        
        ################################
        # Action Binding: Save
        ################################
        self.actionSave_As.triggered.connect(self.SaveFileFunc)
        
        ################################
        # Graph setting
        ################################
        self.InitGraph() # Graph Initializing
        
        ################################
        # Update
        ################################
        self.RefreshSerialPort() # Refresh the Serial Port
        self.show() # Show Window
        self.activateWindow()
        
        pass
    
#%% Member function of MainWindow Class: Graph
        
    def InitGraph(self): # Initialize Graph
        self.plots = [] # Axis Settings
        self.curves = [] # Graph(Plot) Settings
        
        # for i in range(16):
        #     # Labeling the plots
        #     self.plots.append(pq.PlotWidget(labels={"left": _NAMES[i]}))
            
        # for j in range(16):
        #     # Settings of Axis
        #     self.plots[j].setYRange(-2200, 2200)
        #     self.plots[j].showAxis('bottom', False)
        #     self.plots[j].setMenuEnabled('bottom', False)
        #     self.plots[j].setMenuEnabled('left', False)
        #     self.plots[j].setMouseEnabled(x=False, y=False)
        #     self.graph_layout.addWidget(self.plots[j])
        #     self.curves.append(self.plots[j].plot())
        
        for i in range(16):
            p = pq.PlotWidget(labels={"left": _NAMES[i]})
            
            p.setYRange(-2200, 2200)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            
            p.setMenuEnabled('left', False)
            p.setMouseEnabled(x=False, y=False)
            
            self.plots.append(p)
            self.graph_layout.addWidget(p)
            self.curves.append(p.plot())

        pass
    
    def ClearGraph(self): # Erase Graph / Not Used yet
        
        for i in range(16):
            self.curves[i].clear()
        
        pass
    
    def Buffering(self):
        
        data = self.board.get_board_data()
        self.saveInterval = data.shape[1]        
        self.TotalBuffer[:, self.saveIdx : self.saveInterval] = data
        
        self.saveIdx += self.saveInterval
        
    
    def DrawGraph(self): # Draw Graph
        
        # Get the Data from board
        # num_points: window size * sampling rate
        data = self.board.get_board_data()
        # data = self.board.get_current_board_data(self.num_points)
        # self.TotalBuffer[self.saveIdx] = data
        # self
        
        for count, channel in enumerate(self.exg_channels):
            # TODO: Filtering here?
            
            self.curves[count].setData(data[channel].tolist())
        
        print(data.shape)
        
        pass
    
#%% Member function of MainWindow Class: Graph
    
    def setUIcenter(self): # Move the Window to the center
        qr = self.frameGeometry()  # Get the position and size of current widget
        cp = QDesktopWidget().availableGeometry().center()  # Get the center position
        qr.moveCenter(cp)  # Move the center of widget to the center of screen
        self.move(qr.topLeft())  # Move the MainWindow
    
    def closeEvent(self, event): # Close Event      
        self.deleteLater()
        event.accept()
    
#%% Member function of MainWindow Class: Buttons
  
    def btnConnect(self): # Connect to the Board
        
        # CHECK THE BOARD TYPE AGIAN
        if self.cyton_CB.isChecked() and not self.daisy_CB.isChecked():
            self.BOARD = 0
        if self.daisy_CB.isChecked() and not self.cyton_CB.isChecked():
            self.BOARD = 2
            
        # Enable logger
        BoardShim.enable_dev_board_logger()
        logging.basicConfig(level = logging.DEBUG)
        
        # Get Default Parameter and Set serial port
        params = BrainFlowInputParams()
        params.serial_port = self.Port
        
        # Prepare session
        print("Connecting...")
        self.board = BoardShim(self.BOARD, params)
        self.board.prepare_session()
        
        # Get channel & sampling rate information
        self.exg_channels = BoardShim.get_exg_channels(self.BOARD)
        self.sampling_rate = BoardShim.get_sampling_rate(self.BOARD)
        
        print("Ready for streaming!")
        
        pass
        
    def btnDisconnect(self): # Disconnect to the Board
        self.board = None # Relase the object
        
        pass
    
    def btnStartStream(self): # Start Streaming
        # TODO: Make Total Buffer   
        self.TotalBuffer = np.zeros((32, int(self.buffer_LE.text())), dtype=np.float64)
        self.saveIdx = 0 ; self.saveInterval = 0
        
        self.window_size = int(self.time_LE.text()) # window size
        self.num_points = self.window_size * self.sampling_rate # Set the size
        self.storeData = None # Delete Buffer
        
        # Start stream; Default size of ringbuffer: 450000
        self.board.start_stream(int(self.buffer_LE.text()))
        
        # Activate Buffering Timer
        self.Buffering = QtCore.QTimer()
        self.Buffering.timeout.connect(self.Buffering) # Draw Graph
        self.Buffering.start(5) # 200Hz
        
        # Activate Drawing Timer
        self.DrawingTimer = QtCore.QTimer()
        self.DrawingTimer.timeout.connect(self.DrawGraph) # Draw Graph
        self.DrawingTimer.start(50) # 20Hz
        
        
        pass
    
    def btnStopStream(self): # Stop streaming
        self.storeData = self.board.get_board_data() # Get Every Stored Data from Board
        self.board.stop_stream() # Stop Streaming
        
        
        self.BufferingTimer.stop()
        self.BufferingTimer = None
        
        # Stop and Delete drawing timer
        self.DrawingTimer.stop()
        self.DrawingTimer = None
            
        pass
    
#%% TEST BTN
    
    def ChangeCyton(self): # Select Board Type: Cyton
        self.BOARD = 0
        
        # Ignore before selection: Dasiy
        if self.daisy_CB.isChecked():
            self.daisy_CB.toggle()
            
        pass
    
    def ChangeDaisy(self): # Select Board Type: Daisy
        self.BOARD = 2
        
        # Ignore before selection: Cyton
        if self.cyton_CB.isChecked():
            self.cyton_CB.toggle()
        pass
    
    def testButton(self): # Test Fn
        self.Fz_LB.setStyleSheet("background-color: yellow;")
        
        pass
    
    def startImp(self): # Collect the Impedance Data
        
        IMP = [] # Impedance Data per ch
        
        # Command for Cyton channels
        name_ch = [1,2,3,4,5,6,7,8,'Q','W','E','R','T','Y','U','I']
        
        # QLabels
        labels = [self.F3_LB, self.Fz_LB, self.F4_LB,
                 self.T3_LB, self.C3_LB, self.Cz_LB, self.C4_LB, self.T4_LB,
                 self.T5_LB, self.P3_LB, self.Pz_LB, self.P4_LB, self.T6_LB,
                 self.O1_LB, self.O2_LB, self.GND_LB]
        try:
            
            for i in self.exg_channels: # for channels
                # Change configuration: LeadOff Detection
                self.board.config_board('dz{}01Z'.format(name_ch[i-1]))
                print("Change Config: PIN {}".format(i))
                
                # Get data
                self.board.start_stream()
                time.sleep(2.5)
                self.board.stop_stream()
                
                # TODO: Do filtering about DATA
                data = self.board.get_board_data()
            
                imp = np.sqrt(2) * np.std(data[i]) * 1.0e-6/6.0e-9 - 2200
                if imp < 0:
                    imp = 0
                    
                IMP.append(imp)
            
                # Change background color
                if imp >= 0:
                    labels[i-1].setStyleSheet("background-color: green;")
                if imp > 1.5e+4:
                    labels[i-1].setStyleSheet("background-color: yellow;")
                if imp > 1.0e+5:
                    labels[i-1].setStyleSheet("background-color: red;")
                    
            print(IMP)
                    
            pass

        except BaseException:
            logging.warning('Exception', exc_info=True)
            
        finally:
            logging.info('End')
        
        pass
    
    def stopImp(self):
        self.board.config_board('d')
        pass
    
    def playSound(self):
        
        playsound('./Sound/grasp.mp3')
        pass
#%% Serial Port Functions
    
    def SelectSerialPort(self, text): # Select Serial From Combo box
        self.Port = text
        print("You Choosed: ", text)
        pass
    
    def RefreshSerialPort(self): # Load possible ports
        if self.board != None:
            self.board.release_session() # Release session
        serial_objs = list_ports.comports() # Load possible ports
        ports = ["None"]
        for serial_obj in serial_objs: # Bring the Lists
            ports.append(serial_obj.device)
        self.port_COMB.clear()
        for port in ports:
            self.port_COMB.addItem(port)
            
#%% Save File Functions

    def SaveFileFunc(self):
        
        # Validating of Data
        if self.storeData.all() != None or self.storeData == None:
            
            # Set the file name and path
            name = datetime.datetime.now().strftime('%y%m%d%%%H%M')
            saveDIR = os.path.join("./Data", name)
            
            filepath, _ = QFileDialog.getSaveFileName(self, "Save File", saveDIR, "*.txt",)
            
            if filepath=="":
                QMessageBox.information(self, "Warning!", "File Save Path is not Valid")
            else:
                np.save(os.path.splitext(filepath)[0], self.storeData)
                QMessageBox.information(
                    self, "Information", "np.data is saved Successfully!"
                )
        else:
            QMessageBox.information(self, "Warning!", "Data is not Valid")

        # name = datetime.datetime.now().strftime('%y%m%d%%%H%M')
        # saveDIR = os.path.join("./Data", name)
        
        # filepath, _ = QFileDialog.getSaveFileName(self, "Save File", saveDIR, "*.txt",)
        
        # if filepath=="":
        #     QMessageBox.information(self, "Warning!", "File Save Path is not Valid")
        # else:
        #     np.save(os.path.splitext(filepath)[0], self.storeData)
        #     QMessageBox.information(
        #         self, "Information", "np.data is saved Successfully!"
        #     )
        
        
        pass            
#%% main() function

if __name__ == "__main__":
    app = QApplication.instance() # Get the Instance
    
    # suppress_qt_warnings() # Resizing the screen
    
    if app is None:
        app = QApplication(sys.argv) # Object Intialize
        
    else:
        print("QApplication instance is already exist: %s" % str(app))
    print("Current Process:", os.getpid())
    
    mainwnd = MainWindow()
    mainwnd.show() # Show the window
    sys.exit(app.exec_()) # Excute
        
        
       
        
