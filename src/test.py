# Test.
from constant import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import time
import cPickle
import argparse

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
model_type_str = 'prototype'
model_name = '0'
test_type_str = '1k_nocrop_obj'
test_name = '0'
'''
####################

import caffe

caffe.set_device(gpu)
caffe.set_mode_gpu()

print 'Processing: finetune_alexnet_{}_{} on GPU {}, test on {}'.format(model_type_str, model_name, gpu, test_type_str)

'''
net = caffe.Net(result_root + 'model/finetune_alexnet_{}_{}/deploy.prototxt'.format(model_type_str, model_name),
                result_root + 'model/finetune_alexnet_{}_{}/finetune_alexnet_{}_{}.caffemodel'.format(model_type_str, model_name, model_type_str, model_name),
                caffe.TEST)
'''
net = caffe.Net(result_root + 'model/bvlc_alexnet/deploy.prototxt',
                result_root + 'model/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

test_file = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, test_type_str, test_name), 'r')

positive_true = 0
image_sum = 0
accuracy = 0.0

lines = test_file.readlines()
image_sum = len(lines)
now_sum = 0

start_time = time.time()
for i, line in enumerate(lines):
    now_sum += 1
    image_path, class_id = line.split(' ')
    image_name = image_path.split('/')[-1]
    class_id = int(class_id)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
    out = net.forward()
    predict_id = out['prob'][0].argmax()
    if not '1k' in model_type_str:
        true_id = class_id
    else:
        true_id = new_to_original_class_id[class_id]
    if true_id == predict_id:
        positive_true += 1        
    if i % 1000 == 0:
        second = int(time.time() - start_time)
        now_time = time.strftime("%H:%M:%S", time.gmtime(second))
        estimated = int(float(image_sum) / now_sum * second)
        estimated_time = time.strftime("%H:%M:%S", time.gmtime(estimated))
        print '[{}/{}] {:5}/{:5}: {} {} {} {}'.format(now_time, estimated_time, i, \
                image_sum, image_name, predict_id, true_id, positive_true / float(now_sum))
accuracy = positive_true / float(image_sum)

print 'positive_true: ', positive_true
print 'image_sum: ', image_sum
print 'accuracy: ', accuracy
    
with open(result_root + 'test/accuracy_{}_{}_{}_{}.pickle'.format(model_type_str, model_name, test_type_str, test_name), 'wb') as f:
    cPickle.dump(accuracy, f)