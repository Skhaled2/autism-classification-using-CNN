# autism-classification-using-CNN
This project aims to detect autism in children using Convolutional Neural Networks (CNNs) based on image data. The dataset consists of images categorized into autistic and non-autistic children.

#Table of Contents
Installation
Dataset Preparation
Model Architecture
Training the Model
Evaluating the Model
Results

#Installation
-To run this project, you need to install the following libraries:
pip install tensorflow
pip install matplotlib

#Dataset Preparation
-Dataset Structure:
  The dataset should be organized into the following structure:
autism/
    train/
        autistic/
        non_autistic/
    test/
        autistic/
        non_autistic/
    valid/
        autistic/
        non_autistic/

-Folder Creation:
  The following script ensures the required folders are created:
import os

autism_folder = 'C:/autism'
subfolders = ['train', 'test', 'valid']
categories = ['non_autistic', 'autistic']

for folder in [autism_folder] + [os.path.join(autism_folder, sub) for sub in subfolders]:
    os.makedirs(folder, exist_ok=True)

for subfolder in subfolders:
    for category in categories:
        os.makedirs(os.path.join(autism_folder, subfolder, category), exist_ok=True)
        
-Data Augmentation:
  ImageDataGenerator is used for data augmentation:
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(autism_folder, 'train'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(autism_folder, 'valid'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(autism_folder, 'test'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

#Model Architecture
-The model is built using the VGG16 architecture with additional layers for binary classification:

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

#Training the Model
-The model is trained with the following parameters:

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping]
)

#Evaluating the Model
-The model is evaluated on the test set:

test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

#Results
the final accuracy is 81% and the curves is provided

#References
VGG16 Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition (https://arxiv.org/abs/1409.1556)
TensorFlow Documentation: https://www.tensorflow.org/api_docs






