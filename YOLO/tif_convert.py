import os
from PIL import Image


def convert_tif_to_jpg(input_folder, output_folder):
    # 確認輸入資料夾存在
    if not os.path.exists(input_folder):
        print(f"輸入資料夾不存在: {input_folder}")
        return

    # 如果輸出資料夾不存在，則建立該資料夾
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 取得所有符合格式的 .tif 檔案
    for filename in os.listdir(input_folder):
        if filename.endswith("_predict.tif") or filename.endswith("_predict.tiff"):
            # 建立完整的檔案路徑
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, os.path.splitext(filename)[0] + ".jpg")

            # 讀取並轉換圖片
            with Image.open(input_path) as img:
                img = img.convert("RGB")  # 確保圖片為 RGB 模式
                img.save(output_path, "JPEG")
            print(f"轉換完成: {input_path} -> {output_path}")

input_folder = "./test"
output_folder = "./test/predict"
convert_tif_to_jpg(input_folder, output_folder)
