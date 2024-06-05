from yolo_test_api import *

model_path = "./runs/detect/train6/weights/best.pt"
test_path = "./data/images/test/114_2.tif"

Set_Parameter(model_path)
print(predict_image(test_path))