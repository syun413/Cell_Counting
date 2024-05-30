import os
import xml.etree.ElementTree as ET

def convert_to_yolo(size, box):
    x = (box[0] + box[1]) / 2.0 / size[0]
    y = (box[2] + box[3]) / 2.0 / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)

def xml_to_yolo(xml_file, output_dir, class_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    filename = os.path.basename(xml_file).replace('.xml', '.txt')
    output_file = os.path.join(output_dir, filename)
    
    with open(output_file, 'w') as out_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = class_mapping[class_name]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            box = (xmin, xmax, ymin, ymax)
            yolo_box = convert_to_yolo((width, height), box)
            out_file.write(f"{class_id} {' '.join(map(str, yolo_box))}\n")

def convert_all_xml_to_yolo(xml_dir, output_dir, class_mapping):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_to_yolo(xml_file, output_dir, class_mapping)

# 使用範例
xml_directory = '/Users/liang/Documents/Medical/DataSet/training_data_20240412/non-empty'  # XML 文件所在目錄
output_directory = '/Users/liang/Documents/Medical/DataSet/training_data_20240412/yolo'  # YOLO 格式標註文件輸出目錄
class_mapping = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'co_rg': 3,
    'co_gb': 4,
    'co_rb': 5,
}

convert_all_xml_to_yolo(xml_directory, output_directory, class_mapping)
