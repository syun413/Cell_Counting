import os

from ultralytics import YOLO
import cv2
import numpy as np


## ----- API ----- ##

def Set_Parameter(model_path):
  """All paths must be ABSOLUTE PATH"""
  global model
  model = YOLO(model_path)

def predict_image(input_path, conf_threshold = 0.5):
  """All paths must be ABSOLUTE PATH"""
  image = cv2.imread(input_path)
  image_size = image.shape[:2]

  results = model.predict(image, conf = conf_threshold, imgsz = image_size)[0]

  draw_and_save(results, find_output_path(input_path), image)
  text = generate_text(results)
  return text

def predict_list(input_path_list, conf_threshold = 0.5):
  """All paths must be ABSOLUTE PATH"""
  for image_name in input_path_list:
    image = cv2.imread(image_name)


## ----- Backend ----- ##

model = None

id2color = {0: (0,0,255),
  1: (0,255,0),
  2: (255,0,0),
  3: (0,255,255),
  4: (255,255,0),
  5: (255,0,255)
} # (b, g, r)

def find_output_path(input_path):
  name, ext = os.path.splitext(input_path)
  return f"{name}_predict{ext}"

def draw_and_save(results, output_path, image):
  for x1, y1, x2, y2, score, class_id in results.boxes.data.tolist():
    color = id2color[class_id]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, str(round(score,2)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=2, lineType=2)
  cv2.imwrite(output_path, image)
      
def generate_text(results):
  cell_count = [0] * len(results.names)
  for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    cell_count[int(class_id)] += 1
  text = ""
  for i in range(len(results.names)):
    text += f"{results.names[i]}: {cell_count[i]}\n"
  return text
