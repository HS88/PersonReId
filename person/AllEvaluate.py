from __future__ import division, print_function, absolute_import

import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image

TEST = "test.list"
TEST_NUM = 1360 
QUERY = "query.list"
QUERY_NUM = 350


def extract_feature(dir_path, net, limit):
  features = []
  infos = []
  num = 1
  with open(dir_path, 'r') as f:
    for image_name in f:
      if (num>limit):
          break
      arr = image_name.split('_')
      person = int(arr[0])
      camera = int(arr[1][1])
      num = num + 1
      image_path = "detected/" + image_name[:-1]
      #image_path = "detected/" + image_name
      #num+=1  
      img = image.load_img(image_path, target_size=(128, 64))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      feature = net.predict(x)
      features.append(np.squeeze(feature))
      infos.append((person, camera))

  return features, infos


file = open("oldevaluation_10epochs.txt","w")
# use GPU to calculate the similarity matrix
query_t = tf.placeholder(tf.float32, (None, None))
test_t = tf.placeholder(tf.float32, (None, None))
query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# load model

net = load_model('model_depth16_card16_width16_epoch125_.ckpt')
file.write("Model starting")
print("Model starting")
net = Model(input=net.input, output=net.get_layer('avg_pool').output)
print("After model")
file.write("Test feature extraction started")
print("Test festure extraction")
test_f, test_info = extract_feature(TEST, net,TEST_NUM)
file.write("Query feature extraction started")
print("Query feature extraction")
query_f, query_info = extract_feature(QUERY, net,QUERY_NUM)

match = []
junk = []

for q_index, (qp, qc) in enumerate(query_info):
  tmp_match = []
  tmp_junk = []
  for t_index, (tp, tc) in enumerate(test_info):
    if tp == qp and qc != tc:
      tmp_match.append(t_index)
    elif tp == qp or tp == -1:
      tmp_junk.append(t_index)
  match.append(tmp_match)
  junk.append(tmp_junk)

result = sess.run(tensor, {query_t: query_f, test_t: test_f})
result_argsort = np.argsort(result, axis=1)

file.write("mAP calculation is starting")

rank_1 = 0.0
mAP = 0.0

for idx in range(len(query_info)):
  recall = 0.0
  precision = 1.0
  hit = 0.0
  cnt = 0.0
  ap = 0.0
  YES = match[idx]
  IGNORE = junk[idx]
  rank_flag = True
  file.write("Inside evaluation loop")
  print("Inside evalation loop")
  for i in list(reversed(range(0, TEST_NUM))):
    k = result_argsort[idx][i]
    #print("k")
    #print(k)
#    file.write(i)
    if k in IGNORE:
      continue
    else:
      cnt += 1
      if k in YES:
        hit += 1
        if rank_flag:
          rank_1 += 1
      tmp_recall = hit/len(YES)
      tmp_precision = hit/cnt
      ap = ap + (tmp_recall - recall)*((precision + tmp_precision)/2)
      recall = tmp_recall
      precision = tmp_precision
      rank_flag = False
    if hit == len(YES):
      break
  mAP += ap

file.write('Rank 1:\t%f'%(rank_1 / QUERY_NUM))
print ('Rank 1:\t%f'%(rank_1 / QUERY_NUM))
print ('mAP:\t%f'%(mAP / QUERY_NUM))
file.write('mAP:\t%f'%(mAP / QUERY_NUM))
