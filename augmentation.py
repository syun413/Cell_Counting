"""
Pre-request: pip install pillow
Show usage: python augmentation.py -h
"""


import os
import sys
import argparse
from PIL import Image
import xml.etree.ElementTree as ET


color_flags={1: "red", 2: "blue", 3: "green", 4: "co_rb", 5: "co_gb", 6: "co_rg"}
flags_color={"red": 1, "blue": 2, "green": 3, "co_rb": 4, "co_gb": 5, "co_rg": 6}
def mod_color(tif_path, xml_path, output_tif, output_xml):
  # change color
  image = Image.open(tif_path)
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
  print()

def mod_sharpen(tif_path, xml_path, output_tif, output_xml):
  print()

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
  return 1

def arg_clar(tif_path, xml_path, output_path, file, idx):
  output_tif = os.path.join(output_path, file).replace('.tif', f"_{idx}.tif")
  output_xml = output_tif.replace('.tif', '.xml')
  mod_sharpen(tif_path, xml_path, output_tif, output_xml)
  return 1

def process_directory(input_path, output_path, color=False, rot=False, clar=False):
  """
  處理指定目錄下的所有 TIF 和 XML 檔案。
  """
  for file in os.listdir(input_path):
    if file.endswith(".tif"):
      tif_path = os.path.join(input_path, file)
      xml_path = tif_path.replace('.tif', '.xml')
      if os.path.exists(xml_path):
        idx = 1
        if(color):
          idx += arg_color(tif_path, xml_path, output_path, file, idx)
        if(rot):
          idx += arg_rot(tif_path, xml_path, output_path, file, idx)
        if(clar):
          idx += arg_clar(tif_path, xml_path, output_path, file, idx)
      else:
        print(f"No XML file found for {tif_path}")

# 主函數，調用 process_directory 並傳入目錄路徑
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Image data augmentation.")
  # Must-be argument
  parser.add_argument("input_path", type=str, help="Input path of the image file.")
  parser.add_argument("output_path", type=str, help="Output path for the processed image.")
  # Optional argument
  parser.add_argument("--color", action="store_true", help="Exchange color channels in the image.")
  parser.add_argument("--rotate", action="store_true", help="Apply rotate operations on the image.")
  parser.add_argument("--clarity", action="store_true", help="Adjust the image's sharpness.")

  args = parser.parse_args()
  process_directory(args.input_path, args.output_path, args.color, args.rotate, args.clarity)