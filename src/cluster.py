import cPickle
from vector import *
from constant import *
from sklearn.cluster import KMeans

class Cluster:
    def __init__(self, n_cluster, vectors, layer):
        self.n_cluster = n_cluster
        self.n_init = 10
        self.vectors = vectors
        self.layer = layer
        self.kmeans = None
        self.predicted = None
        self.kmeans_scores = None
        
    def clustering(self):
        self.kmeans = KMeans(init = 'k-means++', n_clusters = self.n_cluster, n_init = self.n_init, verbose = 1)
        self.predicted = self.kmeans.fit_predict([v.data for v in self.vectors])
        self.kmeans_scores = []
        for i in range(len(self.vectors)):
            self.kmeans_scores.append(self.kmeans.score(self.vectors[i].data.reshape(1, -1)))
         
    def assign(self):
        pass
    
    def save(self):
        with open(research_root + 'result/' + dataset_name + 'cluster.pickle', 'wb') as handle:
            cPickle.dump(self, handle)