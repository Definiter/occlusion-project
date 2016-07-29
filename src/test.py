# Test.
from constant import *
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import time
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, required=False)
parser.add_argument('--net_name', required=True)
args = parser.parse_args()

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

is_finetuned = True
#net = caffe.Net(imagenet_root + 'model/bvlc_alexnet/deploy.prototxt',
#                imagenet_root + 'model/bvlc_alexnet/bvlc_alexnet.caffemodel',
#                caffe.TEST)

#net_names = ['0', '25', '33', '50', '66', '80', 'all']

gpu = int(args.gpu)
caffe.set_device(gpu)
caffe.set_mode_gpu()

net_name = args.net_name
print 'Processing: finetune_alexnet_crop_' + net_name, 'on GPU', gpu

net = caffe.Net(imagenet_root + 'model/finetune_alexnet_crop_{}/deploy.prototxt'.format(net_name),
                imagenet_root + 'model/finetune_alexnet_crop_{}/finetune_alexnet_crop_{}.caffemodel'.format(net_name, net_name),
                caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(50,3,227,227)

test_dataset = [(0.0, 0), (1.0/4, 4), (1.0/3, 3), (1.0/2, 3), (2.0/3, 3), (4.0/5, 3), (1.0, 1)]
positive_true = [0 for i in test_dataset]
image_sum = [0 for i in test_dataset]
accuracy = [0.0 for i in test_dataset]

start_time = time.time()
for index, (occlu_size, occlu_num) in enumerate(test_dataset):
    percent = str(int(100 * occlu_size))
    test_file = open('{}dataset/test_crop_{}.txt'.format(imagenet_root, percent), 'r')

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
    
with open(result_root + 'test/positive_true_{}.pickle'.format(net_name), 'wb') as f:
    cPickle.dump(positive_true, f)
with open(result_root + 'test/image_sum_{}.pickle'.format(net_name), 'wb') as f:
    cPickle.dump(image_sum, f)
with open(result_root + 'test/accuracy_{}.pickle'.format(net_name), 'wb') as f:
    cPickle.dump(accuracy, f)