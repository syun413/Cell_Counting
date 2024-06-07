"""
Pre-request: pip install pillow, pip install opencv-python
Show usage: python augmentation.py -h
"""


import os
import sys
import argparse
from PIL import Image, ImageFilter
import cv2
import numpy as np
import xml.etree.ElementTree as ET


color_flags={1: "red", 2: "blue", 3: "green", 4: "co_rb", 5: "co_gb", 6: "co_rg"}
flags_color={"red": 1, "blue": 2, "green": 3, "co_rb": 4, "co_gb": 5, "co_rg": 6}
def mod_color(tif_path, xml_path, output_tif, output_xml):
  # modify tif
  with Image.open(tif_path) as image:
    r, g, b = image.split()
    output_image = Image.merge("RGB", (g, b, r))
    output_image.save(output_tif)
  # modify xml
  xml = ET.parse(xml_path)
  root = xml.getroot()
  for obj in root.findall('object'):
    name = obj.find('name').text
    if name in flags_color:
      current_num = flags_color[name]
      if current_num > 3:
        new_num = (current_num - 3) % 3 + 4
      else:
        new_num = current_num % 3 + 1
      obj.find('name').text = color_flags[new_num]
  xml.write(output_xml)

def mod_rotate(tif_path, xml_path, output_tif, output_xml):
  # modify tif
  with Image.open(tif_path) as image:
    output_image = image.rotate(-90, expand=True)
    output_image.save(output_tif)
  # modify xml
  xml = ET.parse(xml_path)
  root = xml.getroot()
  height = int(root.find('size').find('height').text)
  for obj in root.findall('object'):
    xmin = int(obj.find('bndbox').find('xmin').text)
    ymin = int(obj.find('bndbox').find('ymin').text)
    xmax = int(obj.find('bndbox').find('xmax').text)
    ymax = int(obj.find('bndbox').find('ymax').text)
    obj.find('bndbox').find('xmin').text = str(height - ymax)
    obj.find('bndbox').find('ymin').text = str(xmin)
    obj.find('bndbox').find('xmax').text = str(height - ymin)
    obj.find('bndbox').find('ymax').text = str(xmax)
  xml.write(output_xml)

# def mod_sharpen(tif_path, xml_path, output_tif, output_xml, mask):
#   # modify tif
#   with Image.open(tif_path) as image:
#     kernel_values = [
#             -1, -1, -1,
#             -1,  9, -1,
#             -1, -1, -1
#         ]
#     kernel = ImageFilter.Kernel((3,3), kernel_values, scale=1.0)
#     output_image = image.filter(kernel)
#     output_image.save(output_tif)
#   # modify xml
#   xml = ET.parse(xml_path)
#   xml.write(output_xml)

def mod_sharp(tif_path, xml_path, output_tif, output_xml, output_tif2, output_xml2):
  image = cv2.imread(tif_path, -1)
  # blur
  output_image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
  cv2.imwrite(output_tif, output_image)
  # modify xml
  xml = ET.parse(xml_path)
  xml.write(output_xml)
  # sharpen
  # laplacian will enlarge noises, use unsharp mask instead
  mask = np.subtract(image, output_image)
  output_image2 = cv2.addWeighted(image, 1.1, mask, 0.1, 0)
  cv2.imwrite(output_tif2, output_image2)
  # modify xml
  xml.write(output_xml2)

def arg_color(tif_path, xml_path, output_path, file, idx):
  output_tif = os.path.join(output_path, file).replace('.tif', f"_{idx}.tif")
  output_xml = output_tif.replace('.tif', '.xml')
  mod_color(tif_path, xml_path, output_tif, output_xml)
  output_tif2 = output_tif.replace(f"{idx}.tif", f"{idx+1}.tif")
  output_xml2 = output_tif2.replace('.tif', '.xml')
  mod_color(output_tif, output_xml, output_tif2, output_xml2)
  return 2

def arg_rot(tif_path, xml_path, output_path, file, idx):
  output_tif = os.path.join(output_path, file).replace('.tif', f"_{idx}.tif")
  output_xml = output_tif.replace('.tif', '.xml')
  mod_rotate(tif_path, xml_path, output_tif, output_xml)
  output_tif2 = output_tif.replace(f"{idx}.tif", f"{idx+1}.tif")
  output_xml2 = output_tif2.replace('.tif', '.xml')
  mod_rotate(output_tif, output_xml, output_tif2, output_xml2)
  output_tif3 = output_tif.replace(f"{idx}.tif", f"{idx+2}.tif")
  output_xml3 = output_tif3.replace('.tif', '.xml')
  mod_rotate(output_tif2, output_xml2, output_tif3, output_xml3)
  return 3

def arg_sharp(tif_path, xml_path, output_path, file, idx):
  output_tif = os.path.join(output_path, file).replace('.tif', f"_{idx}.tif")
  output_xml = output_tif.replace('.tif', '.xml')
  # mask = mod_blur(tif_path, xml_path, output_tif, output_xml)
  output_tif2 = output_tif.replace(f"{idx}.tif", f"{idx+1}.tif")
  output_xml2 = output_tif2.replace('.tif', '.xml')
  mod_sharp(tif_path, xml_path, output_tif, output_xml, output_tif2, output_xml2)
  return 2

def process_directory(input_path, output_path, color=False, rot=False, sharp=False, all=False, idx=1):
  """
  Process all .tif under the input directory
  """
  files = os.listdir(input_path)
  for file in files:
    if file.endswith(".tif"):
      tif_path = os.path.join(input_path, file)
      xml_path = tif_path.replace('.tif', '.xml')
      index = idx
      if os.path.exists(xml_path):
        if(color or all):
          index += arg_color(tif_path, xml_path, output_path, file, index)
        if(rot or all):
          index += arg_rot(tif_path, xml_path, output_path, file, index)
        if(sharp or all):
          index += arg_sharp(tif_path, xml_path, output_path, file, index)
      else:
        print(f"No XML file found for {tif_path}")

## -------- main -------- ##

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Image data augmentation.")
  # Must-be argument
  parser.add_argument("input_path", type=str, help="Input directory of the image file.")
  parser.add_argument("output_path", type=str, help="Output directory for the processed image.")
  # Optional argument
  parser.add_argument("--color", action="store_true", help="Exchange color channels in the image.")
  parser.add_argument("--rotate", action="store_true", help="Apply rotate operations on the image.")
  parser.add_argument("--sharpen", action="store_true", help="Adjust the image's sharpness.")
  parser.add_argument("--all", action="store_true", help="Apply color, rotate, and sharpen.")
  parser.add_argument("--index", type=int, help="The index that output filenames start from.")

  args = parser.parse_args()
  if not args.index:
    args.index = 1
  process_directory(args.input_path, args.output_path, args.color, args.rotate, args.sharpen, args.all, args.index)