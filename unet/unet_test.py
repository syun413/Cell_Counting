import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tensorflow.keras.models import load_model
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
import xml.etree.ElementTree as ET

## ----- Helper Functions ----- ##

def predict_image(model, image):
    """Predict the class probabilities for each pixel in the image using the model."""
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    prediction = model.predict(image)
    return prediction[0]  # Remove batch dimension

def count_cell_types(predicted):
    cell_counts = {}
    classes =  ['background', 'red', 'blue', 'green', 'co_rb', 'co_gb', 'co_rg']
    for class_id in range(1, 7):
        mask = (predicted == class_id)
        labeled_array, num_features = label(mask) # labeled_array 會把他偵測出的同一個物件變成同一個編號，如果是 num_features 的話大概是有幾個連通塊
        props = regionprops(labeled_array)
        
        count = 0
        for prop in props:
            area = prop.area
            perimeter = prop.perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 判斷是否應該被分成多個細胞的邏輯（這裡假設圓度大於一定值才算一個細胞）
            if circularity > 0.5:
                count += 1
            else:
                count += 2  # 簡單假設形狀不規則的算兩個細胞

        cell_counts[classes[class_id]] = count

    return cell_counts

def plot_image(predict):
    color_map = [
        [0,0,0], #background
        [200,0,0], #red
        [0,0,200], #blue
        [0,200,0], #green
        [200,0,200], #co_rb
        [0,200,200], #co_gb
        [200,200,0]  #co_rg
    ]
    colored_img = np.zeros((predict.shape[0], predict.shape[1], 3))
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            colored_img[i, j, :] = color_map[predict[i, j]]
    return colored_img

def calc_accuracy(predict, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    true_counts = {}
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in true_counts:
            true_counts[class_name] += 1
        else:
            true_counts[class_name] = 1

    classes = set(true_counts.keys()).union(set(predict.keys()))

    true_count = sum(min(true_counts.get(cls, 0), predict.get(cls, 0)) for cls in classes)
    false_count = sum(abs(predict.get(cls, 0) - true_counts.get(cls, 0)) for cls in classes)

    # Calculate accuracy
    accuracy = true_count / (true_count + false_count) if (true_count + false_count) > 0 else 0

    return accuracy

## ----- Load ----- ##

model = load_model("./model/0502_0025.h5")

input_path = "./data/test/"
output_path = "./results/"

## ----- Predict ----- ##

total_accuracy = 0.0
total_count = 0
for file in os.listdir(input_path):
  if file.endswith(".tif"):
    print(file)

    tif_path = os.path.join(input_path, file)
    image = tiff.imread(tif_path)
    normal_image = image / 255.0

    prediction = predict_image(model, normal_image)
    predicted_classes = np.argmax(prediction, axis=-1) # Convert predictions to class indices (argmax over the last dimension)
    colored_image = plot_image(predicted_classes)

    # Plot the predicted class image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide the axes ticks

    axes[1].imshow(colored_image)
    axes[1].set_title('Predicted Classes')
    axes[1].axis('off')  # Hide the axes ticks

    ## ----- Count Cells ----- ##
    cell_counts = count_cell_types(predicted_classes)
    print("Cell counts:", cell_counts)
    xml_path = tif_path.replace('.tif', '.xml')
    accuracy = calc_accuracy(cell_counts, xml_path)
    print("accuracy =", accuracy)

    counts_text = ", ".join([f"{cell_type}: {count}" for cell_type, count in cell_counts.items()])
    counts_text += f"\naccuracy = {accuracy}"
    plt.text(0.5, -0.1, counts_text, ha='center', va='top', transform=fig.transFigure, fontsize=12)

    result_path = os.path.join(output_path, file).replace('.tif', '.png')
    plt.savefig(result_path, bbox_inches='tight')  # Save the figure as a PNG file
    # plt.show()

    total_accuracy += accuracy
    total_count += 1

total_accuracy /= total_count
print("Overall accuracy =", total_accuracy)

