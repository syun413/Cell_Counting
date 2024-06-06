# Flask Application 後端
from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# 載入模型
# model = ...
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# # 定義圖片轉換
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# 處理圖片並返回計數結果

def count_cells(image):
    return 42 # 測試用，暫時回傳一個固定數字
    # image = transform(image).unsqueeze(0)

    # with torch.no_grad():
    #     outputs = model(image)

    # count = ...  # 根據模型輸出計算細胞的數量
    # return count

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    count = count_cells(image)

    return jsonify({"count": count})


if __name__ == '__main__':
    app.run(debug=True)