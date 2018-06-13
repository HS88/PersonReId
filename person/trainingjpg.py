from __future__ import division, print_function, absolute_import

import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from modelmlfn import MLFN

DATASET = '../dataset/Duke'
##LIST = os.path.join(DATASET, 'train.list')
LIST = "train.list"
##TRAIN = os.path.join(DATASET, 'bounding_box_train')
TRAIN ="detected/"

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8

sess = tf.Session(config=config)
set_session(sess)

model = MLFN((128, 64, 3), depth=[4,4,4,4], cardinality=32, width=32, classes=767, T_dimension=1024)

for layer in model.layers:
   layer.trainable = True

# load data
images, labels = [], []
num = 0
limit = int((1*7360))
with open(LIST, 'r') as f:
  for line in f:
    if num > imit:
      break
    num = num + 1
    line = line.strip()
    img, lbl = line.split()
    img = image.load_img(TRAIN+img, target_size=[128, 64])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    images.append(img[0])
    labels.append(int(lbl))

images = np.array(images)
labels = to_categorical(labels)

# train
batch_size = 64
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=False)

print("Starting to compile")
model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy')
print("Starting to fit")
model.fit_generator(datagen.flow(images, labels, batch_size=batch_size), steps_per_epoch=int((len(images)/batch_size+1)*1.3), epochs=10)
model.save('model_depth16_card32_width32_epoch100_.ckpt')
