# pip install pillow
# usage: python slice.py <input_dir> <output_dir>

import sys
from PIL import Image
import os

def split_image(image_path, output_dir, tile_size=256):
  try:
    image = Image.open(image_path)
  except IOError:
    print(f"Error in opening the image file: {image_path}. Make sure the image path is correct and the file format is supported.")
    return

  image_width, image_height = image.size
  new_width = ((image_width - 1) // tile_size + 1) * tile_size
  new_height = ((image_height - 1) // tile_size + 1) * tile_size

  new_image = Image.new("RGB", (new_width, new_height), "black")
  new_image.paste(image, (0, 0))

  for x in range(0, new_width, tile_size):
    for y in range(0, new_height, tile_size):
      box = (x, y, x + tile_size, y + tile_size)
      tile = new_image.crop(box)
      
      output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{x}{y}.tif")
      tile.save(output_path)

  print(f"Image {os.path.basename(image_path)} successfully split")

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
