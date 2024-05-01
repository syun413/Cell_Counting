import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

## ----- Helper Functions ----- ##

def predict_image(model, image):
    """Predict the class probabilities for each pixel in the image using the model."""
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match model input
    prediction = model.predict(image)
    return prediction[0]  # Remove batch dimension

def count_cell_types(prediction, threshold=0.5):
    """Count the number of times each cell type is predicted in the image."""
    counts = {}
    for i, label in enumerate(['background', 'red', 'blue', 'green', 'co_rb', 'co_gb', 'co_rg'], start=0):
        # Count pixels where the probability of the class is above the threshold
        counts[label] = np.sum(prediction[:,:,i] > threshold)
    return counts

## ----- Load ----- ##

model = load_model("./model/model.h5")

input_path = "./data/test/"
output_path = "./results/"

for file in os.listdir(input_path):
  if file.endswith(".tif"):
    tif_path = os.path.join(input_path, file)
    image = tiff.imread(tif_path)
    normal_image = image / 255.0

    ## ----- Predict ----- ##

    prediction = predict_image(model, normal_image)

    # Convert predictions to class indices (argmax over the last dimension)
    predicted_classes = np.argmax(prediction, axis=-1)

    # Plot the predicted class image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide the axes ticks

    axes[1].imshow(predicted_classes, cmap='jet')
    axes[1].set_title('Predicted Classes')
    axes[1].axis('off')  # Hide the axes ticks

    plt.colorbar(axes[1].imshow(predicted_classes, cmap='jet'), ax=axes[1])
    result_path = os.path.join(output_path, file).replace('.tif', '.png')
    plt.savefig(result_path)  # Save the figure as a PNG file
    plt.show()

    ## ----- Count Cells ----- ##
    cell_counts = count_cell_types(prediction)
    print("Cell counts:", cell_counts)
