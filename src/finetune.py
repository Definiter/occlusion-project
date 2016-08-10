# Fine tune.
from constant import *
import os
import numpy as np
import time
import cPickle
from pylab import *
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, required=False)
parser.add_argument('--model_name', required=True)
parser.add_argument('--model_type_str', required=True) # crop / nocrop / aperture
args = parser.parse_args()
gpu = int(args.gpu)
model_name = args.model_name
model_type_str = args.model_type_str
'''

gpu = 0
model_name = '60'
model_type_str = '1k_nocrop_obj'

import caffe

niter = 10000
# Losses will also be stored in the log.
train_loss = np.zeros(niter)
train_accuracy = np.zeros(niter)
test_accuracy = {}

caffe.set_device(gpu)
caffe.set_mode_gpu()

print 'Processing: finetune_alexnet_{}_{}'.format(model_type_str, model_name), 'on GPU', gpu, ',', model_type_str

solver = caffe.SGDSolver(result_root + 'model/finetune_alexnet_{}_{}/solver.prototxt'.format(model_type_str, model_name))
solver.net.copy_from(imagenet_root + 'model/bvlc_alexnet/bvlc_alexnet.caffemodel')

start_time = time.time()
# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    train_accuracy[it] = solver.net.blobs['accuracy_train'].data
    second = int(time.time() - start_time)
    estimated = int(float(niter) / (it + 1) * second)
    estimated_day = estimated / 3600 / 24
    now_time = time.strftime("%H:%M:%S", time.gmtime(second))
    estimated_time = time.strftime("%H:%M:%S", time.gmtime(estimated))
    if it % 100 == 0 or (it <= 1000 and it % 10 == 0):
        test_iters = 10
        accuracy = 0
        for i in arange(test_iters):
            solver.test_nets[0].forward()
            accuracy += solver.test_nets[0].blobs['accuracy_test'].data
        accuracy /= test_iters
        test_accuracy[it] = accuracy
        print '[{} / {}d {}] iter{:6} | train_loss={:10.6f}, train_accuracy={:10.6f}, test_accuracy={:10.6f}'.format(now_time, estimated_day, estimated_time, it, float(train_loss[it]), float(train_accuracy[it]), accuracy)
    elif it % 10 == 0:
        print '[{} / {}d {}] iter{:6} | train_loss={:10.6f}, train_accuracy={:10.6f}'.format(now_time, estimated_day, estimated_time, it, float(train_loss[it]), float(train_accuracy[it]))
        
solver.net.save(result_root + 'model/finetune_alexnet_{}_{}/finetune_alexnet_{}_{}.caffemodel'.format(model_type_str, model_name, model_type_str, model_name))

with open(result_root + 'finetune/train_loss_{}_{}.pickle'.format(model_type_str, model_name), 'wb') as f:
    cPickle.dump(train_loss, f)
with open(result_root + 'finetune/train_accuracy_{}_{}.pickle'.format(model_type_str, model_name), 'wb') as f:
    cPickle.dump(train_accuracy, f)
with open(result_root + 'finetune/test_accuracy_{}_{}.pickle'.format(model_type_str, model_name), 'wb') as f:
    cPickle.dump(test_accuracy, f)

print 'done'