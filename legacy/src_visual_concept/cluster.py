import cPickle
from vector import *
from constant import *
from sklearn.cluster import KMeans
import numpy as np
import os
import copy

class Cluster:
    def __init__(self, n_cluster, extractor):
        self.n_cluster = n_cluster
        self.n_class = len(extractor.classnames)
        self.n_init = 10
        
        self.layer = extractor.layer
        self.kmeans = None
        self.predicted = None
        self.kmeans_scores = None
        
        self.classnames = extractor.classnames
        self.cluster_to_class = None
        self.class_to_cluster = None
        self.cluster_assigned = None
        self.max_accuracy = 0
        self.max_accuracy_assign = None
        
        self.training_images = extractor.training_images # [[vector1, vector2, ...], [...], ...]
        self.validation_images = extractor.validation_images # same
        self.test_images = extractor.test_images # same
        
        self.count = 0
        
    def clustering(self):
        self.kmeans = KMeans(init = 'k-means++', n_clusters = self.n_cluster, n_init = self.n_init, verbose = 1)
        vectors = [v for image in self.training_images for v in image]
        self.predicted = self.kmeans.fit_predict([v.data for v in vectors])
        self.kmeans_scores = []
        for i in range(len(vectors)):
            self.kmeans_scores.append(self.kmeans.score(vectors[i].data.reshape(1, -1)))
         
    def save(self):
        if not os.path.exists(research_root + 'result/' + dataset_name):
            os.makedirs(research_root + 'result/' + dataset_name)
        with open(research_root + 'result/' + dataset_name + 'cluster.pickle', 'wb') as handle:
            cPickle.dump(self, handle)
            
    def dfs(self, class_id):
        if (class_id == self.n_class):
            print "Trying assignment: ", self.count
            self.count = self.count + 1
            self.cluster_to_class = [-1 for i in range(self.n_cluster)]
            for i in range(self.n_class):
                #print i, "->", self.class_to_cluster[i]
                self.cluster_to_class[self.class_to_cluster[i]] = i
            accuracy, _ = self.classify_all(self.validation_images)
            #print "accuracy", accuracy
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.max_accuracy_assign = copy.copy(self.cluster_to_class)
            return
        for i in range(self.n_cluster):
            if not self.cluster_assigned[i]:
                self.cluster_assigned[i] = True
                self.class_to_cluster[class_id] = i
                self.dfs(class_id + 1)
                self.cluster_assigned[i] = False
                self.class_to_cluster[class_id] = -1
            
    def assign(self):
        self.cluster_to_class = [-1 for i in range(self.n_cluster)]
        self.class_to_cluster = [-1 for i in range(self.n_class)]
        
        self.cluster_assigned = [False for i in range(self.n_cluster)]
        self.count = 0
        #self.dfs(0)
        print(self.max_accuracy)
        
        
        self.class_to_cluster = [0, 21, 17]
        for i in range(self.n_class):
            self.cluster_to_class[self.class_to_cluster[i]] = i
        
        accuracy, _ = self.classify_all(self.validation_images)
        print(accuracy)
        
        #self.cluster_to_class = copy.copy(self.max_accuracy_assign)
        accuracy, _ = self.classify_all(self.test_images)
        print(accuracy)
        '''
        for each assignment solution:
            calc classification error in validation dataset
        choose one solution with least error
        calc classification error in test dataset
        '''
        
    # Input: sampled vectors in image, output: class_id
    def classify_by_vote(self, vectors):
        predicted = self.kmeans.predict([v.data for v in vectors])
        class_vote = [0 for i in range(self.n_class)]
        for cluster_id in predicted:
            class_id = self.cluster_to_class[cluster_id]
            if class_id >= 0:
                class_vote[class_id] = class_vote[class_id] + 1
        class_id = np.argmax(class_vote)
        return class_id
        
       
    def classify_all(self, images):
        image_class = [-1 for i in range(len(images))]
        accuracy = 0
        for i, vectors in enumerate(images):
            image_class[i] = self.classify_by_vote(vectors)
            if image_class[i] == vectors[0].class_id:
                accuracy = accuracy + 1
        accuracy = float(accuracy) / len(images)
        return accuracy, image_class
           
            
            