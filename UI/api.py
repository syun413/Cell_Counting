import os

from ultralytics import YOLO
import cv2
import numpy as np
from utils import *

class API:
  def __init__(self):
    self.model = None
    self.save_path = None

  def Set_Parameter(self, model_path=None, save_path=None):
    """All paths must be ABSOLUTE PATH"""
    if model_path is not None:
      self.model = YOLO(model_path)
    if save_path is not None:
      self.save_path = save_path

  def predict_image(self, input_path, conf_threshold = 0.5):
    """All paths must be ABSOLUTE PATH"""
    if self.model is None:
      return "Please set model path before predicting"
      
    image = cv2.imread(input_path)
    image_size = image.shape[:2]

    results = self.model.predict(image, conf = conf_threshold, imgsz = image_size)[0]
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    draw_and_save(results, find_output_path(input_path, self.save_path), image)
    generate_text(results, find_output_path(input_path, self.save_path, ".txt"))

  def predict_list(self, input_path_list, conf_threshold = 0.5):
    """All paths must be ABSOLUTE PATH"""
    if self.model is None:
      return "Please set model path before predicting"
    
    for input_path in input_path_list:
      image = cv2.imread(input_path)
      image_size = image.shape[:2]

      results = self.model.predict(image, conf = conf_threshold, imgsz = image_size)[0]
      if not os.path.exists(self.save_path):
        os.makedirs(self.save_path)

      draw_and_save(results, find_output_path(input_path, self.save_path), image)
      generate_text(results, find_output_path(input_path, self.save_path, ".txt"))


id2color = {0: (0,0,255),
  1: (0,255,0),
  2: (255,0,0),
  3: (0,255,255),
  4: (255,255,0),
  5: (255,0,255)
} # (b, g, r)


def draw_and_save(results, output_path, image):
  for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
    color = id2color[class_id]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, str(round(score,2)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=2, lineType=2)
  cv2.imwrite(output_path, image)
      
def generate_text(results, output_path):
  cell_count = [0] * len(results.names)
  for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    cell_count[int(class_id)] += 1
  text = ""
  for i in range(len(results.names)):
    text += f"{results.names[i]}: {cell_count[i]}\n"
  with open(output_path, 'w') as f:
    f.write(text)