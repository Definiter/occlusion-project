# Test.
from constant import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import time
import cPickle
import argparse
import math

#test_dataset = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
#test_dataset = ['0', '20', '40', '60', '80', '100']
#### Parameters ####
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, required=False)
parser.add_argument('--model_type_str', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--test_type_str', required=True)
parser.add_argument('--test_name', required=True)

args = parser.parse_args()
gpu = int(args.gpu)
model_type_str = args.model_type_str
model_name = args.model_name
test_type_str = args.test_type_str
test_name = args.test_name
'''

gpu = 0
model_type_str = 'nocrop'
model_name = '0'
test_type_str = 'nocrop'
test_name = '20'
'''
####################

import caffe

caffe.set_device(gpu)
caffe.set_mode_gpu()

print 'Processing: finetune_alexnet_{}_{} on GPU {}, test on {}'.format(model_type_str, model_name, gpu, test_type_str)

test_net = caffe.Net(result_root + 'model/finetune_alexnet_{}_{}/test_{}_{}.prototxt'\
                .format(model_type_str, model_name, test_type_str, test_name),
                result_root + 'model/finetune_alexnet_{}_{}/finetune_alexnet_{}_{}.caffemodel'\
                .format(model_type_str, model_name, model_type_str, model_name),
                caffe.TEST)

image_sum = 0
test_file = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, test_type_str, test_name), 'r')
image_sum = len(test_file.readlines())
test_file.close()

test_iters = int(math.ceil(image_sum / 256.0)) # batch_size
print test_iters, image_sum

accuracy = 0.0
for i in range(test_iters):
    test_net.forward()
    accuracy += test_net.blobs['accuracy_test'].data
    print '[{}/{}] {}'.format(i, test_iters, accuracy / (i + 1))
accuracy /= test_iters
print 'accuracy: ', accuracy
    
with open(result_root + 'test/accuracy_{}_{}_{}_{}.pickle'.format(model_type_str, model_name, test_type_str, test_name), 'wb') as f:
    cPickle.dump(accuracy, f)