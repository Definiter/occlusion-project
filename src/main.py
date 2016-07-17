# Initialize caffe.
import matplotlib
matplotlib.use('Agg')
from constant import *
from vector import *
import numpy as np
import caffe
import cPickle

# 1. Initialize caffe network.
print("Start initializing caffe network.")
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
mode = 'gpu'
if mode == 'gpu':
    caffe.set_device(0)
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
net = caffe.Classifier(caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt',
        caffe_root + 'models/vgg16/VGG_ILSVRC_16_layers.caffemodel',
        mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
        channel_swap=(2, 1, 0),
        raw_scale=255,
        image_dims=(224, 224))
# Dimensions: 224x224, with 3 channels. Batch size 1
# NOTE: maybe can use batching to speed up processing?
net.blobs['data'].reshape(1, 3, 224, 224)
print("Initialized caffe network.")

'''
# 2. Extractor.
print("Start extracting vectors.")
from extractor import Extractor
extractor = Extractor('conv4_1', 0.3, net)
extractor.extract()
extractor.save()
print("Extracted vectors in all classes.")

# 3. Cluster.
print("Start clustering.")
from cluster import *
extractor = None
with open(research_root + 'result/' + dataset_name + 'extractor.pickle', 'rb') as handle:
    extractor = cPickle.load(handle)
    print(str(len(extractor.vectors)) + " vectors loaded.")
print(str(len(extractor.vectors)) + " vectors loaded.")
cluster = Cluster(64, extractor.vectors, extractor.layer)
cluster.clustering()
cluster.save()
print("Clustered all vectors.")
'''

# 4. Visualize cluster.
cluster = None
with open(research_root + 'result/' + dataset_name + 'cluster.pickle', 'rb') as handle:
    cluster = cPickle.load(handle)
    print("Cluster loaded.")
from vis_utility import *
visualize_cluster(net, cluster)

# 5. Assign cluster.
cluster.assign()


# 6. Classification of test dataset.


# 7. Visualize result.