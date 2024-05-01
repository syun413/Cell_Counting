## ----- import ----- ##

import os
import numpy as np
import tifffile as tiff
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback
from datetime import datetime

## ----- hyperparameter ----- ##

learning_rate = 0.001
batch_size = 4
epochs = 50
image_height = 256
image_width = 256
channels = 3
num_classes = 6

## ----- file path ----- #

now = datetime.now().strftime("%m%d_%H%M")
model_save_path = f"./model/{now}.h5"
model_temp_path = "./model/temp.h5"
history_path = f"./model/{now}.png"
train_data_path = "./data/train"
# valid_data_path = "./data/valid"

## ----- load data ----- ##

def load_image(image_path):
    """Read tif image and turn into numpy array."""
    image = tiff.imread(image_path)
    image = image / 255.0  # 正規化到 [0, 1]
    return image

def parse_xml(xml_path):
    """Parse xml to create masks."""
    color_to_label = {'red': 1, 'blue': 2, 'green': 3, 'co_rb': 4, 'co_gb': 5, 'co_rg': 6}
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        color = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        if color in color_to_label:
            mask[ymin:ymax, xmin:xmax] = color_to_label[color]
    one_hot_mask = to_categorical(mask, num_classes+1)
    return one_hot_mask

def load_data(directory):
    """從指定目錄加載所有圖片和標籤."""
    images = []
    masks = []
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            image_path = os.path.join(directory, filename)
            xml_path = os.path.join(directory, filename.replace('.tif', '.xml'))
            images.append(load_image(image_path))
            masks.append(parse_xml(xml_path))
    return np.array(images), np.array(masks)

train_images, train_masks = load_data(train_data_path)
# valid_images, valid_labels = load_data(valid_data_path)


## ----- Unet ----- ##

def conv_block(input_tensor, num_filters):
    """卷積塊：包含兩個卷積層和激活函數."""
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """編碼器塊，包含卷積塊和最大池化層."""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    """解碼器塊，包含上採樣、串接和卷積塊."""
    x = UpSampling2D((2, 2))(input_tensor)
    x = concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_classes):
    """構建 Unet 模型."""
    inputs = Input(input_shape)

    # 編碼器
    c1, p1 = encoder_block(inputs, 64)
    c2, p2 = encoder_block(p1, 128)
    c3, p3 = encoder_block(p2, 256)
    c4, p4 = encoder_block(p3, 512)

    # 橋接
    b = conv_block(p4, 1024)

    # 解碼器
    d1 = decoder_block(b, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    # 輸出
    outputs = Conv2D(num_classes+1, (1, 1), activation='softmax')(d4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# 模型實例化
input_shape = (image_height, image_width, channels)
unet_model = build_unet(input_shape, num_classes)

# 編譯模型
unet_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# 模型結構總結
unet_model.summary()

## ----- train ----- ##

# check GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(f"Device: {c}")

# 分割數據集
train_images, valid_images, train_masks, valid_masks = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=42
)

# 使用 `ImageDataGenerator` 來進行訓練和驗證
train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_masks, batch_size=batch_size)
valid_generator = valid_datagen.flow(valid_images, valid_masks, batch_size=batch_size)

# 設置模型儲存回調
checkpoint_callback = ModelCheckpoint(
    model_temp_path, save_best_only=True, monitor='val_loss', mode='min'
)
# 設置提前停止回調
early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, mode='min'
)

# 訓練模型
history = unet_model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=len(valid_images) // batch_size,
    callbacks=[checkpoint_callback, early_stopping_callback, TqdmCallback(verbose=1)],
    verbose=0  # 關閉內置的進度條，使用 TqdmCallback 代替
)

# # 繪製訓練過程中的損失和精度變化
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
plt.show()

# 模型評估
best_model = load_model(model_temp_path)
eval_result = best_model.evaluate(valid_generator, steps=len(valid_images) // batch_size)
print(f"Final validation loss: {eval_result[0]:.4f}")
print(f"Final validation accuracy: {eval_result[1]:.4f}")

## ----- save model ----- ##
best_model.save(model_save_path)
print(f"Model saved to {model_save_path}")
