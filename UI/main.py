""" Usage
Ctrl+O: Open file
Ctrl+Shift+O: Open directory
Ctrl+S: Change Save directory (not done)
N: Next Image (not done)
P: Previous Image (not done)
Ctrl+A: Set API (not done)
Ctrl+Return: Count (not done)
Ctrl+W: Clear file list
Ctrl+Q: Quit
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QShortcut
from PyQt5.QtGui import QPixmap, QKeySequence

from mainPage_ui import Ui_MainWindow
from utils import *

class Main(QMainWindow, Ui_MainWindow):
  def __init__(self):
    super().__init__()
    self.setupUi(self)
    self.setEvent()
    self.currentFileList = []
    self.currentFileIdx = -1
    self.savePath = None

  def setEvent(self):
    self.Open.clicked.connect(self.openFile)
    self.OpenDir.clicked.connect(self.openDir)
    self.SaveDir.clicked.connect(lambda: self.showPhoto("SaveDir was clicked"))
    self.NextImg.clicked.connect(lambda: self.showPhoto("NextImg was clicked"))
    self.PrevImg.clicked.connect(lambda: self.showPhoto("PrevImg was clicked"))
    self.SetAPI.clicked.connect(lambda: self.showPhoto("SetAPI was clicked"))
    self.Count.clicked.connect(lambda: self.showPhoto("Count was clicked"))
    self.fileList.itemDoubleClicked.connect(self.openFileListItem)
    ## --- TODO --- ##
    # ctrl+w clear file list
    # ctrl+q quit the app
    self.shortcutClear = QShortcut(QKeySequence("Ctrl+W"), self)
    self.shortcutClear.activated.connect(self.clearFileList)
    self.shortcutQuit = QShortcut(QKeySequence("Ctrl+Q"), self)
    self.shortcutQuit.activated.connect(self.close)

  def showPhoto(self):
    pixmap = QPixmap(self.currentFileList[self.currentFileIdx])
    # self.photo.setPixmap(pixmap)
    if not pixmap.isNull():
      label_size = self.photo.size()
      scaled_pixmap = pixmap.scaled(label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
      self.photo.setPixmap(scaled_pixmap)
      self.photo.setAlignment(QtCore.Qt.AlignCenter)
      ## --- TODO --- ##
      # Show Result if found json

  def showFileList(self, imgList):
    self.currentFileIdx = self.currentFileList.index(imgList[0]) \
      if imgList[0] in self.currentFileList else len(self.currentFileList)
    for imgPath in imgList:
      if imgPath not in self.currentFileList:
        item = QListWidgetItem(imgPath)
        self.fileList.addItem(item)
        self.currentFileList.append(imgPath)

    item = self.fileList.item(self.currentFileIdx)
    if item:
      self.fileList.setCurrentItem(item)

  def clearFileList(self):
    self.currentFileIdx = -1
    self.currentFileList.clear()
    self.fileList.clear()

  def openFile(self):
    self.fname, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "All Files (*);;PNG Files (*.png);;JPG Files (*.jpg *.jpeg)")
    if self.fname:
      self.showFileList([self.fname])
      self.showPhoto()

  def openDir(self):
    self.pname = QFileDialog.getExistingDirectory(self, "Open Directory", ".", QFileDialog.ShowDirsOnly)
    if self.pname:
      # self.fileList.clear()
      imgList = scan_all_images(self.pname)
      self.showFileList(imgList)
      self.showPhoto()

  def openFileListItem(self, item=None):
    self.currentFileIdx = self.currentFileList.index(item.text())
    self.showPhoto()


if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.setWindowTitle("Cell Counting")
  window.show()
  sys.exit(app.exec_())