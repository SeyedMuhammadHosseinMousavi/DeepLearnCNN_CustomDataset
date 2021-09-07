# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 13:48:22 2021

@author: Seyed Muhammad Hossein Mousavi

Comments: 
TensorFlow (library)+ Keras (library) + Python (language)+
 Spyder (cross-platform integrated development environment)
 are used to classify 3 facial expressions classes of IKFDB
 (RGB-D) face dataset [1] using Convolutional Neural Network
 (CNN) deep learning. Please cite [1] paper if you used the code. 
Number of samples is 1500 which each class contains 500 ‘jpg’
 images. Main folder of ‘deeplr’ contains three subfolders of
 ‘Joy’, ‘Neutral’ and ‘Sadness’. 
Custom dataset: You can easily replace your desire dataset
 here in the main folder and put each class in a separated 
 subfolder. Also, you can increase or decrease number of 
 classes. Of course, you have to change ‘num_classes = 3’
 argument according to your number of classes.
Please feel free to contact me for any issues or guide:
Seyed Muhammad Hossein Mousavi 
mosavi.a.i.buali@gmail.com
[1] Mousavi, S.M.H., Mirinezhad, S.Y. Iranian kinect face database (IKFDB): a color-depth based face database collected by kinect v.2 sensor. SN Appl. Sci. 3, 19 (2021). https://doi.org/10.1007/s42452-020-03999-y

"""
# Clearing all vars
#%reset

#Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Loading dataset
import pathlib
data_dir = pathlib.Path('deeplr')


# Define parameters
batch_size = 32
img_height = 180
img_width = 180

# Deviding training and validation to 80-20 %
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Show class names
class_names = train_ds.class_names
print(class_names)

# Visualize some data
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Creating the model
num_classes = 3
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

#Model Training and validation
epochs=6
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualize training plots
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Pridicting a new image 
from keras.preprocessing.image import load_img
# load the image
img = load_img('Neutral.jpg')
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)




