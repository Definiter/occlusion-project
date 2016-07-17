import cPickle
from vector import *
from constant import *
from sklearn.cluster import KMeans
import os

class Cluster:
    def __init__(self, n_cluster, extractor, net):
        self.n_cluster = n_cluster
        self.n_init = 10
        
        self.layer = extractor.layer
        self.kmeans = None
        self.predicted = None
        self.kmeans_scores = None
        
        self.classnames = extractor.classnames
        self.cluster_to_class = None

        self.training_images = extractor.training_images # [[vector1, vector2, ...], [...], ...]
        self.validation_images = extractor.validation_images # same
        self.test_images = extractor.test_images # same
        
    def clustering(self):
        self.kmeans = KMeans(init = 'k-means++', n_clusters = self.n_cluster, n_init = self.n_init, verbose = 1)
        vectors = [v for v in image for image in self.training_images]
        self.predicted = self.kmeans.fit_predict([v.data for v in vectors])
        self.kmeans_scores = []
        for i in range(len(self.vectors)):
            self.kmeans_scores.append(self.kmeans.score(self.vectors[i].data.reshape(1, -1)))
         
    def save(self):
        if not os.path.exists(research_root + 'result/' + dataset_name):
            os.makedirs(research_root + 'result/' + dataset_name)
        with open(research_root + 'result/' + dataset_name + 'cluster.pickle', 'wb') as handle:
            cPickle.dump(self, handle)
            
    def assign(self):
        self.cluster_to_class = [-1 for i in range(self.n_cluster)]
        '''
        for each assignment solution:
            calc classification error in validation dataset
        choose one solution with least error
        calc classification error in test dataset
        
        '''
        
        
    # Input: sampled vectors in image, output: class_id
    def classify_by_vote(self, vectors):
        predicted = self.kmeans.predict([v.data for v in vectors])
        class_vote = [0 for i in range(self.classnames)]
        for cluster_id in image_predicted:
            class_id = self.cluster_to_class[cluster_id]
            if (class_id >= 0)
                class_vote[class_id] = class_vote[class_id] + 1
        class_id = np.argmax(class_vote)
        return class_id
        
       

            