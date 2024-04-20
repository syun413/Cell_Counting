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
from PyQt5.QtCore import QSize

from mainPage_ui import Ui_MainWindow
from utils import *

class Main(QMainWindow, Ui_MainWindow):
  def __init__(self):
    super().__init__()
    self.setupUi(self)
    self.setEvent()
    self.currentFileList = []
    self.currentFileIdx = -1
    self.savePath = '.'
    self.photoSize = QSize(600, 600)

  def setEvent(self):
    self.Open.setToolTip("Open a single photo.")
    self.Open.clicked.connect(self.openFile)
    self.OpenDir.setToolTip("Open every photo under a directory and update the save directory for counting result.")
    self.OpenDir.clicked.connect(self.openDir)
    self.SaveDir.setToolTip("Change the save directory for counting result.")
    self.SaveDir.clicked.connect(self.changeSaveDir)
    self.NextImg.setToolTip("Show next photo.")
    self.NextImg.clicked.connect(lambda: self.switchCurrentImg(1))
    self.PrevImg.setToolTip("Show previous photo.")
    self.PrevImg.clicked.connect(lambda: self.switchCurrentImg(-1))
    self.SetAPI.setToolTip("Connect to counting model.")
    self.SetAPI.clicked.connect(lambda: print("Not Done"))
    self.Count.setToolTip("Count the current photo. May replace old results.")
    self.Count.clicked.connect(lambda: print("Not Done"))
    self.CountAll.setToolTip("Count the every photo in file list. May replace old results.")
    self.CountAll.clicked.connect(lambda: print("Not Done"))
    self.fileList.itemDoubleClicked.connect(self.openFileListItem)
    self.shortcutClear = QShortcut(QKeySequence("Ctrl+W"), self)
    self.shortcutClear.activated.connect(self.clearFileList)
    self.shortcutQuit = QShortcut(QKeySequence("Ctrl+Q"), self)
    self.shortcutQuit.activated.connect(self.close)
    self.shortcutZoomIn = QShortcut(QKeySequence("Ctrl++"), self)
    self.shortcutZoomIn.activated.connect(self.zoomIn)
    self.shortcutZoomEq = QShortcut(QKeySequence("Ctrl+="), self)
    self.shortcutZoomEq.activated.connect(self.zoomIn)
    self.shortcutZoomOut = QShortcut(QKeySequence("Ctrl+-"), self)
    self.shortcutZoomOut.activated.connect(self.zoomOut)

  def showPhoto(self):
    """Show the current photo at the middle of the window."""

    if self.currentFileIdx < 0 or self.currentFileIdx >= len(self.currentFileList):
      return
    
    pixmap = QPixmap(self.currentFileList[self.currentFileIdx])
    if not pixmap.isNull():
      scaled_pixmap = pixmap.scaled(self.photoSize, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
      self.photo.setPixmap(scaled_pixmap)
      self.photo.setAlignment(QtCore.Qt.AlignCenter)
      # highlight current item
      item = self.fileList.item(self.currentFileIdx)
      if item:
        self.fileList.setCurrentItem(item)
      ## --- TODO --- ##
      # Show Result if found json

  def showFileList(self, imgList):
    """Update the file list on left bottom of the window."""

    # Update current photo
    self.currentFileIdx = self.currentFileList.index(imgList[0]) \
      if imgList[0] in self.currentFileList else len(self.currentFileList)
    # Add newly opened photos
    for imgPath in imgList:
      if imgPath not in self.currentFileList:
        item = QListWidgetItem(imgPath)
        self.fileList.addItem(item)
        self.currentFileList.append(imgPath)

  def clearFileList(self):
    """Clear the current file list and close photos. Triggered by 'Ctrl+W'."""

    self.currentFileIdx = -1
    self.currentFileList.clear()
    self.fileList.clear()
    self.photo.clear()

  def openFile(self):
    """Open a single photo."""

    self.fname, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "All Files (*);;PNG Files (*.png);;JPG Files (*.jpg *.jpeg)")
    if self.fname:
      self.showFileList([self.fname])
      self.showPhoto()

  def openDir(self):
    """Open every photo under a directory and update the save directory for counting result."""

    self.pname = QFileDialog.getExistingDirectory(self, "Open Directory", ".", QFileDialog.ShowDirsOnly)
    if self.pname:
      self.savePath = self.pname
      imgList = scan_all_images(self.pname)
      self.showFileList(imgList)
      self.showPhoto()

  def openFileListItem(self, item=None):
    self.currentFileIdx = self.currentFileList.index(item.text())
    self.showPhoto()

  def zoomIn(self):
    """Enlarge the photo. Triggered by 'Ctrl+=' or 'Ctrl++'"""

    self.photoSize.setWidth(min(2000, self.photoSize.width()+100))
    self.photoSize.setHeight(min(2000, self.photoSize.height()+100))
    self.showPhoto()

  def zoomOut(self):
    """Minify the photo. Triggered by 'Ctrl+-'"""

    self.photoSize.setWidth(max(100, self.photoSize.width()-100))
    self.photoSize.setHeight(max(100, self.photoSize.height()-100))
    self.showPhoto()

  def changeSaveDir(self):
    """Change save directory for counting result."""

    self.pname = QFileDialog.getExistingDirectory(self, "Open save Directory", ".", QFileDialog.ShowDirsOnly)
    if self.pname:
      self.savePath = self.pname

  def switchCurrentImg(self, shift):
    """Switch showing photo to next one or previous one."""

    if self.currentFileIdx < 0:
      return
    self.currentFileIdx += shift
    if self.currentFileIdx >= len(self.currentFileList):
      self.currentFileIdx = len(self.currentFileList)-1
    if self.currentFileIdx < 0:
      self.currentFileIdx = 0
    self.showPhoto()


if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.setWindowTitle("Cell Counting")
  window.show()
  sys.exit(app.exec_())