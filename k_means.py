
class MyKMeans:
    
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
    
    def fit(self):
        # 1. Select k random cluster seeds
        # 2. Calculate closest seed for each point, assign it to cluster
        # 3. Calculate new seeds as average of all points in cluster
        # 4. Repeat 2. and 3. until cluster sets stop changing
        pass
    
    def predict(self):
        pass