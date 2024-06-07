""" Usage
Ctrl+O: Open file
Ctrl+Shift+O: Open directory
Ctrl+S: Change Save directory
Right arrow: Next Image 
Left arrow: Previous Image 
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
from api import API

class Main(QMainWindow, Ui_MainWindow):
  ## ----- init ----- ##
  def __init__(self):
    super().__init__()
    self.setupUi(self)
    self.setEvent()
    self.currentFileList = []
    self.currentFileIdx = -1
    self.api = API()
    self.showSaveDir(os.path.join(os.path.abspath("."), "predict"))
    self.photoSize = QSize(600, 600)

  def setEvent(self):
    self.Open.setToolTip("Open a single photo.")
    self.Open.clicked.connect(self.openFile)
    self.OpenDir.setToolTip("Open every photo under a directory and update the save directory for counting result.")
    self.OpenDir.clicked.connect(self.openDir)
    self.SaveDir.setToolTip("Change the save directory for counting result. It will be reset after opening directory.")
    self.SaveDir.clicked.connect(self.changeSaveDir)
    self.NextImg.setToolTip("Show next photo.")
    self.NextImg.clicked.connect(lambda: self.switchCurrentImg(1))
    self.PrevImg.setToolTip("Show previous photo.")
    self.PrevImg.clicked.connect(lambda: self.switchCurrentImg(-1))
    self.SetAPI.setToolTip("Connect to counting model.")
    self.SetAPI.clicked.connect(self.connectToModel)
    self.Count.setToolTip("Count the current photo and save result to Save Dir. May replace old results.")
    self.Count.clicked.connect(self.countImage)
    self.CountAll.setToolTip("Count the every photo in file list and save result to Save Dir. May replace old results.")
    self.CountAll.clicked.connect(self.countAllImage)
    self.showResult.stateChanged.connect(self.showPhoto)
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

  ## ----- show ----- ##

  def showPhoto(self):
    """Show the current photo at the middle of the window."""

    if self.currentFileIdx < 0 or self.currentFileIdx >= len(self.currentFileList):
      return
    
    file_path = self.currentFileList[self.currentFileIdx]
    result_path = find_output_path(file_path, self.savePath)
    if self.showResult.isChecked() and os.path.exists(result_path):
      pixmap = QPixmap(result_path)
    else:
      pixmap = QPixmap(file_path)
    if not pixmap.isNull():
      scaled_pixmap = pixmap.scaled(self.photoSize, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
      self.photo.setPixmap(scaled_pixmap)
      self.photo.setAlignment(QtCore.Qt.AlignCenter)
      # highlight current item
      item = self.fileList.item(self.currentFileIdx)
      if item:
        self.fileList.setCurrentItem(item)
      
      self.showPredictResult()

  def showFileList(self, imgList):
    """Update the file list on left bottom of the window."""

    if not imgList:
      return

    # Update current photo
    self.currentFileIdx = self.currentFileList.index(imgList[0]) \
      if imgList[0] in self.currentFileList else len(self.currentFileList)
    # Add newly opened photos
    for imgPath in imgList:
      if imgPath not in self.currentFileList:
        item = QListWidgetItem(imgPath)
        self.fileList.addItem(item)
        self.currentFileList.append(imgPath)

  def showSaveDir(self, path):
    self.savePath = path
    self.saveDirLabel.setText(f"Current Save Dir: {self.savePath}")
    self.api.Set_Parameter(save_path=path)
    self.showPhoto()

  def showPredictResult(self):
    if self.currentFileIdx < 0 or self.currentFileIdx >= len(self.currentFileList):
      return
    
    file_path = self.currentFileList[self.currentFileIdx]
    result_path = find_output_path(file_path, self.savePath, ".txt")
    if os.path.exists(result_path):
      with open(result_path, 'r') as f:
        # Modify here if change file type
        text = f.read()
      self.result.setText(text)
    else:
      self.result.setText("")

  ## ----- event function ----- ##

  def openFile(self):
    """Open a single photo."""

    fname, _ = QFileDialog.getOpenFileName(self, "Open File", ".", "All Files (*);;PNG Files (*.png);;JPG Files (*.jpg *.jpeg)")
    if fname:
      self.showFileList([fname])
      self.showPhoto()

  def openDir(self):
    """Open every photo under a directory and update the save directory for counting result."""

    pname = QFileDialog.getExistingDirectory(self, "Open Directory", ".", QFileDialog.ShowDirsOnly)
    if pname:
      self.showSaveDir(os.path.join(pname, "predict"))
      imgList = scan_all_images(pname)
      self.showFileList(imgList)
      self.showPhoto()

  def openFileListItem(self, item=None):
    self.currentFileIdx = self.currentFileList.index(item.text())
    self.showPhoto()

  def clearFileList(self):
    """Clear the current file list and close photos. Triggered by 'Ctrl+W'."""

    self.currentFileIdx = -1
    self.currentFileList.clear()
    self.fileList.clear()
    self.photo.clear()

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

    pname = QFileDialog.getExistingDirectory(self, "Open save Directory", ".", QFileDialog.ShowDirsOnly)
    if pname:
      # self.savePath = pname
      self.showSaveDir(pname)

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

  ## ----- API ----- ##

  def connectToModel(self):
    fname, _ = QFileDialog.getOpenFileName(self, "Connect to Model", ".", "All Files (*);;YOLO Models (*.pt)")
    self.api.Set_Parameter(model_path=os.path.abspath(fname))

  def countImage(self):
    if self.currentFileIdx < 0 or self.currentFileIdx >= len(self.currentFileList):
      return
    if self.api.model is None:
      print("Please connect to model before predicting") ## TODO: Show pop-up box

    file_path = self.currentFileList[self.currentFileIdx]
    self.api.predict_image(file_path)
    self.showPhoto()

  def countAllImage(self):
    if self.currentFileIdx < 0 or self.currentFileIdx >= len(self.currentFileList):
      return
    if self.api.model is None:
      print("Please connect to model before predicting") ## TODO: Show pop-up box

    self.api.predict_list(self.currentFileList)
    self.showPhoto()

  # def count_image(self):
  #   if self.currentFileIdx < 0:
  #       return

  #   file_path = self.currentFileList[self.currentFileIdx]
  #   with open(file_path, 'rb') as f:
  #       response = requests.post(
  #           'http://127.0.0.1:5000/upload', files={'file': f})

  #   if response.status_code == 200:
  #       result = response.json()
  #       self.result.setText(f"Count: {result['count']}")
  #   else:
  #       self.result.setText("Error in counting")


if __name__ == '__main__':
  import sys
  app = QtWidgets.QApplication(sys.argv)
  window = Main()
  window.setWindowTitle("Cell Counting")
  window.show()
  sys.exit(app.exec_())