'''
Extract vectors in given layer.
'''

import os
import numpy as np
import cPickle
import caffe
from constant import *
from vector import *
    
class Extractor:
    def __init__(self, layer, sample_fraction, net):
        self.layer = layer
        self.sample_fraction = sample_fraction
        self.net = net
        self.classnames = []
        self.sample_mask = []
        
        self.training_images = [] # [[vector1, vector2, ...], [...], ...]
        self.validation_images = [] # same
        self.test_images = [] # same
        
        self.labels = np.loadtxt(caffe_root + 'data/ilsvrc12/synset_words.txt', str, delimiter='\t')
    
    # Find better way to write it to distribute more evenly
    def sample(self, width, height, number):
        prob_true = number * 1.0 / width / height
        return np.random.rand(height, width) < prob_true
    
    def extract(self):
        self.classnames = os.listdir(research_root + 'data/' + dataset_name)
        for class_id, classname in enumerate(self.classnames):
            print('Processing class: ' + classname)
            filenames = os.listdir(research_root + 'data/' + dataset_name + classname)
            image_sum = len(filenames)
            for file_id, filename in enumerate(filenames):
                path = os.path.abspath(os.path.join(research_root + 'data/' + dataset_name + classname, filename))

                # Feed image into net.
                self.net.predict([caffe.io.load_image(path)], oversample=False)
                if verbose:
                    print('Processing image: '+ path)
                    print("Predicted class is #{}.".format(self.net.blobs['prob'].data[0].argmax()))
                    top_k = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
                    print self.labels[top_k]

                #net.blobs[layer].data[batch][z][y][x]
                response = self.net.blobs[self.layer].data[0] # 0th batch.
                num_response = len(response)
                height_response = len(response[0])
                width_response = len(response[0][0])
                if len(self.sample_mask) == 0:
                    # Sample_mask not initialized yet; sample new
                    print str(num_response) + ' filters of ' + str(height_response) + 'x' + str(width_response)
                    self.sample_mask = self.sample(width_response, height_response,
                            float(self.sample_fraction) * width_response * height_response)
                # TODO: this could be parallelized by multiplication -- then filtering out 0 columns
                image = []
                for y in range(height_response):
                    for x in range(width_response):
                        if self.sample_mask[y][x]:
                            ## NOTE: DOUBLE CHECK IF FIRST IS Y SECOND IS X, corresponding to images
                            v = Vector()
                            v.data = response[:, y, x].copy()
                            v.origin_file = path
                            v.location = (x, y)
                            v.class_id = class_id
                            image.append(v)

                if file_id < image_sum * training_ratio:
                    self.training_images.append(image)
                if image_sum * training_ratio <= file_id and file_id < image_sum * (training_ratio + validation_ratio):
                    self.validation_images.append(image)
                if image_sum * (training_ratio + validation_ratio) <= file_id:
                    self.test_images.append(image)
                    
            print len(self.training_images)
            print len(self.validation_images)
            print len(self.test_images)
                        
                    
    def save(self):
        self.net = None
        if not os.path.exists(research_root + 'result/' + dataset_name):
            os.makedirs(research_root + 'result/' + dataset_name)
        with open(research_root + 'result/' + dataset_name + 'extractor.pickle', 'wb') as handle:
            cPickle.dump(self, handle)