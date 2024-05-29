import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances, display_top_masks
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from matplotlib import pyplot as plt
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn import model as modellib, utils
from PIL import Image, ImageDraw
from datetime import datetime
import random
random.seed(42)

# test GPU
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(f"Device: {c}")

####### hyperparameter #######

learning_rate = 0.001
epochs = 30
steps = 10  # steps per epoch
num_classes = 6
eval_ratio = 0.3

####### file path ######

now = datetime.now().strftime("%m%d_%H%M")
model_save_path = f"./model/{now}.h5"
log_save_path = f"./model/{now}_log"
# model_temp_path = "./model/temp.h5"
history_path = f"./model/{now}.png"
annotation_path = "./data/annotation.json"
train_data_path = "./data/train"


####### load data #######

class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation, images):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        coco_json = annotation
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            if image['file_name'] not in images:
                continue
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(train_data_path, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids

# split train and validation
image_files = [f for f in os.listdir(train_data_path) if f.endswith('.tif')]
total_images = len(image_files)

random.shuffle(image_files)
split_point = int(total_images * eval_ratio)
train_files = image_files[:split_point]
val_files = image_files[split_point:]

with open(annotation_path, 'r') as file:
    annotations = json.load(file)

dataset_train = CocoLikeDataset()
dataset_train.load_data(annotations, train_files)
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data(annotations, val_files)
dataset_val.prepare()


def visualize(save_path):
    dataset = dataset_train
    image_ids = dataset.image_ids
    #image_ids = np.random.choice(dataset.image_ids, 3)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        display_top_masks(image, mask, class_ids, dataset.class_names, limit=2)   

    # define image id
    image_id = 0
    # load the image
    image = dataset_train.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = dataset_train.load_mask(image_id)

    # display_instances(image, r1['rois'], r1['masks'], r1['class_ids'],
    # dataset.class_names, r1['scores'], ax=ax, title="Predictions1")

    # extract bounding boxes from the masks
    bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, dataset_train.class_names, save_fig_path=save_path)

# visualize(f"./example.png")



####### Training #######


# define a configuration for the model
class CellConfig(Config):
    NAME = "Cell_Cfg"
    NUM_CLASSES = 1 + num_classes # classes + background
    STEPS_PER_EPOCH = steps
    # DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence
    # GPU_COUNT = 2
    IMAGES_PER_GPU = 4
    LEARNING_RATE = learning_rate

# prepare config
config = CellConfig()
config.display() 


# ROOT_DIR = os.path.abspath("./")
# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
# # Directory to save logs and trained model
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, log_save_path)
# # Path to trained weights file
# DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, model_save_path)
LOG_PATH = os.path.abspath(log_save_path)

model = MaskRCNN(mode='training', model_dir=LOG_PATH, config=config, log_dir=LOG_PATH)
# load weights (mscoco) and exclude the output layers
# model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=epochs, 
            layers='heads',)


####### Draw history #######

history = model.keras_model.history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(history_path)