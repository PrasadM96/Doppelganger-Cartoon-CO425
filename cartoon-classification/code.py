import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import os
import zipfile

PATH = r'G:\Education\Semester 8\425-FYP\As  a classsification matching\Classification'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

batch_size = 5
epochs = 50
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          class_mode='categorical')
#Found 145 images belonging to 14 classes.


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=validation_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')
#Found 14 images belonging to 14 classes.
labels = (train_data_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
print(labels)



IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

base_model.summary()



'''
final_img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

final_img_tfl = np.expand_dims(final_img, axis=0)
print(final_img_tfl.shape)

feature_batch = base_model(final_img_tfl)
print(feature_batch.shape)
'''
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#feature_batch_average = global_average_layer(feature_batch)
#print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(units = 58, input_shape = (520,), activation='softmax')
#prediction_batch = prediction_layer(feature_batch_average)
#print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=5,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=5
)



img = tf.io.read_file(r"G:\Education\Semester 8\425-FYP\As  a classsification matching\jackfrost.jpg")
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
final_img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


plt.subplot(121), plt.imshow(final_img)


#Expand Tensor for Model (Input shape)
y = np.expand_dims(final_img, axis=0)

#Predict Image Tensor with model
prediction = model.predict(y)
prediction_squeeze = np.squeeze(prediction, axis=0)

label_array = np.array(labels)

#print(type(label))
max_val=0
max_label=''

for key, value in labels.items():
    real_label = prediction_squeeze[key]

    print ("{0:.0%}".format(real_label), value)
    
    if(max_val<real_label):
        max_val = real_label
        max_label = value
print(max_label,max_val) 

def loadImages(path):
    '''Put files into lists and return them as one list with all images
     in the folder'''
    image_file = sorted([os.path.join(path, file)
                          for file in os.listdir(path )
                          if file.endswith('.jpg')])
    return image_file

path = r"G:\Education\Semester 8\425-FYP\As  a classsification matching\Classification\train"

image_list = loadImages(path+ "\\" + max_label)
print(image_list)

path = np.array(image_list)
path_string = (path[0])

img = tf.io.read_file(path_string)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)


plt.subplot(122), plt.imshow(img)

plt.savefig(r"G:\Education\Semester 8\425-FYP\As  a classsification matching\jackfrost_match.png", dpi=150)

   