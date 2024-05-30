import os

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("./runs/detect/train4/weights/best.pt")
input_image_dir = "./data/images/test/"
label_dir = "./data/labels/test/"
output_dir = "./result/"
threshold = 0.5
iou_threshold = 0.3
image_size = 256

# id2name = {0: "red",
#   1: "green",
#   2: "blue",
#   3: "co_rg",
#   4: "co_gb",
#   5: "co_rb"
# }

id2color = {0: (0,0,255),
  1: (0,255,0),
  2: (255,0,0),
  3: (0,255,255),
  4: (255,255,0),
  5: (255,0,255)
} # (b, g, r)

## ----- Helper Function ----- ##

def read_yolo_labels(label_path):
  with open(label_path, 'r') as file:
    lines = file.readlines()
  boxes = []
  for line in lines:
    class_id, x_center, y_center, width, height = map(float, line.strip().split())
    x_min = int((x_center - width/2) * image_size)
    y_min = int((y_center - height/2) * image_size)
    x_max = int((x_center + width/2) * image_size)
    y_max = int((y_center + height/2) * image_size)
    boxes.append([class_id, [x_min, y_min, x_max, y_max]])
  return boxes

def draw_bbox(image, boxes, texts):
  for box, text in zip(boxes, texts):
    x1, y1, x2, y2 = map(int, box[1])
    color = id2color[box[0]]

    cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
    cv2.putText(image, str(text), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=2, lineType=2)
      
def cal_iou(box1, box2):
  cls1, [x11, y11, x12, y12] = box1
  cls2, [x21, y21, x22, y22] = box2
  if cls1 != cls2:
    # print("class doesn't match")
    return 0.0
  x_left = max(x11, x21)
  y_top = max(y11, y21)
  x_right = min(x12, x22)
  y_bottom = min(y12, y22)
  if x_right < x_left or y_bottom < y_top:
      return 0.0
  intersection_area = (x_right - x_left) * (y_bottom - y_top)
  box1_area = (x12-x11) * (y12-y11)
  box2_area = (x22-x21) * (y22-y21)
  iou = intersection_area / float(box1_area + box2_area - intersection_area)
  # if iou <= iou_threshold:
  #   print("iou too small")
  # else:
  #   print("Find 1")
  return iou

def add_caption(image, text):
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
  text_x = int((2*image_size+60 - text_size[0]) / 2)
  text_y = image_size + 50
  cv2.putText(image, text, (text_x, text_y), font, font_scale, (0,0,0), 2)


## ----- Main ----- ##

overall_acc = 0
all_img_count = 0
for image_name in os.listdir(input_image_dir):
  if image_name.endswith(".tif"):
    image_path = os.path.join(input_image_dir, image_name)
    image = cv2.imread(image_path)
    true_image = image.copy()

    results = model.predict(image, conf=threshold, imgsz=image_size)[0]

    label_path = os.path.join(label_dir, image_name.replace(".tif", ".txt"))
    true_boxes = read_yolo_labels(label_path)
    classes = [results.names[box[0]] for box in true_boxes]
    draw_bbox(true_image, true_boxes, classes)

    boxes = []
    scores = []
    for result in results.boxes.data.tolist():
      x1, y1, x2, y2, score, class_id = result
      boxes.append([class_id, [x1, y1, x2, y2]])
      scores.append(round(score,2))
    draw_bbox(image, boxes, scores)

    true_count = 0
    false_count = 0
    ground_true_count = len(true_boxes)
    for pred_box in boxes:
      find = False
      for true_box in true_boxes:
        if cal_iou(true_box, pred_box) > iou_threshold:
          true_count += 1
          true_boxes.remove(true_box)
          find = True
          break
      if not find:
        false_count += 1
  
    # print(f"true count = {true_count}, false count = {false_count}, # true_boxes = {ground_true_count}")
    if ground_true_count == 0:
      accuracy = 1 if (true_count+false_count) == 0 else 0
    else:
      accuracy = true_count / float(max(true_count+false_count, ground_true_count))

    merged_image = 255 * np.ones((image_size+70, 2*image_size+60, 3), dtype=np.uint8)
    merged_image[20:20+image_size, 20:20+image_size] = true_image
    merged_image[20:20+image_size, -20-image_size:-20] = image

    # Add caption to the merged image
    add_caption(merged_image, f"Accuracy = {accuracy}")

    # Save the merged image
    cv2.imwrite(os.path.join(output_dir, image_name.replace(".tif", "_result.png")), merged_image)

    overall_acc += accuracy
    all_img_count += 1

if all_img_count > 0:
  overall_acc = overall_acc / all_img_count
else:
  overall_acc = 0
print(f"Accuracy = {overall_acc}")