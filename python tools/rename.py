# usage: python rename.py <dir_path>

import sys
import os

def rename_tif_files(directory_path):
  tif_files = [file for file in os.listdir(directory_path) if file.lower().endswith('.tif')]
  tif_files.sort()  # 對文件名進行排序，以確保重新命名是按照原始的字母順序

  for i, filename in enumerate(tif_files, start=1):
    old_path = os.path.join(directory_path, filename)
    new_path = os.path.join(directory_path, f"{i}.tif")
    os.rename(old_path, new_path)
    print(f"Renamed {filename} to {i}.tif")

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: rename.py <directory_path>")
  else:
    directory_path = sys.argv[1]
    if not os.path.exists(directory_path):
        print("The directory does not exist.")
    elif not os.path.isdir(directory_path):
        print("The specified path is not a directory.")
    else:
        rename_tif_files(directory_path)
