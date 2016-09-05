# Extract feature vectors of images.
from constant import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import time
import cPickle
import argparse
import math
import copy

#### Parameters ####
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, required=False)
parser.add_argument('--model_type_str', required=True) # nocrop
parser.add_argument('--model_name', required=True) # 0, all
parser.add_argument('--test_type_str', required=True) # nocrop
parser.add_argument('--test_name', required=True) # 0, 20, 40, 60, 80, 100

args = parser.parse_args()
gpu = int(args.gpu)
model_type_str = args.model_type_str
model_name = args.model_name
test_type_str = args.test_type_str
test_name = args.test_name
'''

gpu = 0
model_type_str = 'nocrop'
model_name = 'all'
test_type_str = 'nocrop'
test_name = '40'
'''

####################

import caffe

caffe.set_device(gpu)
caffe.set_mode_gpu()

print 'Processing: finetune_alexnet_{}_{} on GPU {}, test on {}'.format(model_type_str, model_name, gpu, test_type_str)

test_net = caffe.Net(result_root + 'model/finetune_alexnet_{}_{}/img2vec_{}_{}.prototxt'\
                .format(model_type_str, model_name, test_type_str, test_name),
                result_root + 'model/finetune_alexnet_{}_{}/finetune_alexnet_{}_{}.caffemodel'\
                .format(model_type_str, model_name, model_type_str, model_name),
                caffe.TEST)

image_sum = 0
test_file = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, test_type_str, test_name), 'r')
image_sum = len(test_file.readlines())
test_file.close()

vectors = [[] for i in range(len(synset_names))]
# vectors[label] = (vector, predict_id, index_in_lmdb)

start_time = time.time()
for i in range(image_sum):
    test_net.forward()
    label = int(test_net.blobs['label'].data[0])
    vector = copy.copy(test_net.blobs['fc8_occlusion'].data[0])
    index_in_lmdb = i
    predict_id = test_net.blobs['prob'].data[0].argmax()
    
    vectors[label].append((vector, predict_id, index_in_lmdb))
    
    if (i + 1) % 1000 == 0:
        second = int(time.time() - start_time)
        now_time = time.strftime("%H:%M:%S", time.gmtime(second))
        estimated = int(float(image_sum) / (i + 1) * second)
        estimated_time = time.strftime("%H:%M:%S", time.gmtime(estimated))
        print '[{}/{}] {:5}/{:5}: {} {}'.format(now_time, estimated_time, i + 1, image_sum, predict_id, label)
    
with open(result_root + 'img2vec/vectors_{}_{}_{}_{}.pickle'.format(model_type_str, model_name, test_type_str, test_name), 'wb') as f:
    cPickle.dump(vectors, f)
    
print 'done'