# Visualize aperture.
# 20 classes, 11 test datasets, 20 images.
from constant import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import time
import cPickle
import argparse

#### Parameters ####
gpu = 0
net_name = 'all'
model_type_str = 'aperture'
test_type_str = 'aperture'
test_dataset = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
####################

import caffe

caffe.set_device(gpu)
caffe.set_mode_gpu()

print 'Processing: finetune_alexnet_{}_{} on GPU {}, test on {}'.format(model_type_str, net_name, gpu, test_type_str)

net = caffe.Net(result_root + 'model/finetune_alexnet_{}_{}/deploy.prototxt'.format(model_type_str, net_name),
                result_root + 'model/finetune_alexnet_{}_{}/finetune_alexnet_{}_{}.caffemodel'.format(model_type_str, net_name, model_type_str, net_name),
                caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

top_images = [[[] for i in test_dataset] for i in range(20)]
# images[class_id][dataset_size][..] = (confidence, img_path)

positive_true = [[0 for i in test_dataset] for i in range(20)]
image_sum = [[0 for i in test_dataset] for i in range(20)]
class_accuracy = [[0.0 for i in test_dataset] for i in range(20)]

start_time = time.time()
for index, test_name in enumerate(test_dataset):
    now_time = time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start_time)))
    print '[{}] Processing dataset: {}'.format(now_time, test_name)
    test_file = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, test_type_str, test_name), 'r')
    lines = test_file.readlines()
    i_sum = len(lines)
    for i, line in enumerate(lines):
        image_path, class_id = line.split(' ')
        class_id = int(class_id)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
        out = net.forward()
        
        predict_id = out['prob'][0].argmax()
        if class_id == predict_id:
            positive_true[class_id][index] += 1
        image_sum[class_id][index] += 1
        
        top_images[class_id][index].append((out['prob'][0][class_id], image_path))
        
        now_time = time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start_time)))
        if i % 1000 == 0:
            print '[{}] {:5}/{:5}: {} {}'.format(now_time, i, i_sum, image_path, out['prob'][0][class_id])

for class_id in range(20):
    for test_index in range(len(test_dataset)):
        class_accuracy[class_id][test_index] = float(positive_true[class_id][test_index]) / image_sum[class_id][test_index]
            
for class_id in range(20):
    for test_id in range(len(test_dataset)):
        top_images[class_id][test_id] = sorted(top_images[class_id][test_id], reverse = True)

with open(result_root + 'test/top_images.pickle', 'wb') as f:
    cPickle.dump(top_images, f)
with open(result_root + 'test/class_accuracy.pickle', 'wb') as f:
    cPickle.dump(class_accuracy, f)
    
print 'done'