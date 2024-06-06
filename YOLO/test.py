from yolo_test_api import *

model_path = "./best.pt"
test_path = "./test/114_2.tif"

Set_Parameter(model_path)
print(predict_image(test_path))