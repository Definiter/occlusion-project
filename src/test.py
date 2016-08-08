# Test.
from constant import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import time
import cPickle
import argparse

#### Parameters ####
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, required=False)
parser.add_argument('--net_name', required=True)
parser.add_argument('--model_type_str', required=True)
parser.add_argument('--test_type_str', required=True)
args = parser.parse_args()
gpu = int(args.gpu)
net_name = args.net_name
model_type_str = args.model_type_str
test_type_str = args.test_type_str
#test_dataset = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
test_dataset = ['0', '20', '40', '60', '80', '100']
####################

import caffe

is_finetuned = True

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

positive_true = [0 for i in test_dataset]
image_sum = [0 for i in test_dataset]
accuracy = [0.0 for i in test_dataset]

start_time = time.time()
for index, test_name in enumerate(test_dataset):
    test_file = open('{}dataset/test_{}_{}.txt'.format(imagenet_root, test_type_str, test_name), 'r')

    lines = test_file.readlines()
    image_sum[index] = len(lines)
    for i, line in enumerate(lines):
        image_path, class_id = line.split(' ')
        class_id = int(class_id)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image_path))
        out = net.forward()
        predict_id = out['prob'][0].argmax()
        if is_finetuned:
            true_id = class_id
        else:
            true_id = new_to_original_class_id[class_id]
        now_time = time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start_time)))
        if i % 1000 == 0:
            print '[{}] {:5}/{:5}: {} {} {}'.format(now_time, i, image_sum[index], image_path, predict_id, true_id)
        if true_id == predict_id:
            positive_true[index] += 1        
    accuracy[index] = positive_true[index] / float(image_sum[index])
    print 'positive_true: ', positive_true
    print 'image_sum: ', image_sum
    print 'accuracy: ', accuracy
    
with open(result_root + 'test/positive_true_{}_{}_{}.pickle'.format(model_type_str, test_type_str, net_name), 'wb') as f:
    cPickle.dump(positive_true, f)
with open(result_root + 'test/image_sum_{}_{}_{}.pickle'.format(model_type_str, test_type_str,net_name), 'wb') as f:
    cPickle.dump(image_sum, f)
with open(result_root + 'test/accuracy_{}_{}_{}.pickle'.format(model_type_str, test_type_str,net_name), 'wb') as f:
    cPickle.dump(accuracy, f)