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
from keras.utils import multi_gpu_model

###
#	After running this script, you will get a trained model.
#	This script can train the model on multiple GPUs in parallel
# 	and hence is faster
#   In the end, we will save the trained model that can be evaluated
###


DATASET = '../dataset/Duke'
LIST = "train.list"
TRAIN ="detected/"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
set_session(sess)

# This is where the model with required dimensions is created
# Here, model has input of size (128,64,3) and will consist of total 4+4+4+4=16 layers of resnext blocks
# where each layer will have 32 convolution-blocks in 2nd layer of a resnext block and 32 channels
# NOTE:- Resnext block is a three layered block, with 2nd layer consisting of multiple convolution blocks

with tf.device('/cpu:0'):
  model = MLFN((128, 64, 3), depth=[4,4,4,4], cardinality=32, width=32,classes=767, T_dimension=1024)

#Command to train model in parallel
parallel_model = multi_gpu_model(model, gpus=8)

for layer in model.layers:
   layer.trainable = True

# load data
images, labels = [], []
num = 0

with open(LIST, 'r') as f:
  for line in f:
    line = line.strip()
    img, lbl = line.split()
    img = image.load_img(TRAIN+img, target_size=[128,64])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    images.append(img[0])
    labels.append(int(lbl))

images = np.array(images)
labels = to_categorical(labels)

#get input data in batches
batch_size = 16
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=False)

print("Starting to compile")
parallel_model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy')
print("Starting to fit")

#Command to train the model
parallel_model.fit_generator(datagen.flow(images, labels, batch_size=batch_size), steps_per_epoch=len(images)/batch_size, epochs=100)
#saving the trained model
model.save('model_depth16_card32_width32_epoch100_.ckpt')
