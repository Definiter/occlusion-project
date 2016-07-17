'''
Class for storing vector data.
'''
class Vector:
    def __init__(self):
        self.data = []
        self.origin_file = None # File path.
        self.location = None # Arrays of (x, y).
        self.class_id = -1 # Class it belongs to.
        self.cluster_id = -1 # Cluster it belongs to.