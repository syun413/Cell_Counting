# pip install pillow
# usage: python slice.py <input_dir> <output_dir>

import sys
from PIL import Image
import os

def split_image(image_path, output_dir):
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Error in opening the image file: {image_path}. Make sure the image path is correct and the file format is supported.")
        return

    image_width, image_height = image.size
    
    # 根據原始圖片大小決定分割後每邊的圖片數量
    tile_width = tile_height = 256
    if image_width != image_height:
      print("Unsupported image size for: ", image_path)
      return
    else:
      tiles_per_side = image_width / tile_width

    for x in range(0, tiles_per_side):
        for y in range(0, tiles_per_side):
            left = x * tile_width
            upper = y * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # 根據當前的 x, y 位置創建一個新的圖片
            tile = image.crop((left, upper, right, lower))
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{x}_{y}.png")
            tile.save(output_path)

    print(f"Image {os.path.basename(image_path)} successfully split into {tiles_per_side * tiles_per_side} tiles.")

def process_directory(input_dir, output_dir):
  for root, dirs, files in os.walk(input_dir):
    for file in files:
      if file.lower().endswith('.tif'):
        split_image(os.path.join(root, file), output_dir)

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python slice.py <input_dir> <output_dir>")
  else:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_directory(input_dir, output_dir)
