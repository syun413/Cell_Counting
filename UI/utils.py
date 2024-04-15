import re, os
from PyQt5.QtGui import QImageReader

def natural_sort(list, key=lambda s:s):
  """
  Sort the list into natural alphanumeric order.
  """
  def get_alphanum_key_func(key):
      convert = lambda text: int(text) if text.isdigit() else text
      return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
  sort_key = get_alphanum_key_func(key)
  list.sort(key=sort_key)

def scan_all_images(folder_path):
  extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
  images = []

  for root, dirs, files in os.walk(folder_path):
    for file in files:
      if file.lower().endswith(tuple(extensions)):
        relative_path = os.path.join(root, file)
        path = os.path.abspath(relative_path)
        images.append(path)
  natural_sort(images, key=lambda x: x.lower())
  return images