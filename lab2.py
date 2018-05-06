import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN



iris=datasets.load_iris()
filename = 'iris.txt'
datar=[]
with open(filename,'r') as file_to_read:
    while (True):
        datat=[]
        lines = file_to_read.readline()
        if not lines:
            break
        att1, att2, att3, att4,\
        temp = [i for i in lines.split(",")]
        datat.append(float(att1))
        datat.append(float(att2))
        datat.append(float(att3))
        datat.append(float(att4))
        datar.append(datat)
D=np.array(datar)



def get_density(self, x, X, y=None, sample_weight=None):
        superweight = 0.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if sample_weight is None:
            sample_weight = np.ones((n_samples, 1))
        else:
            sample_weight = sample_weight
        for y in range(n_samples):
            kernel = kernelize(x, X[y], h=self.h, degree=n_features)
            kernel = kernel * sample_weight[y] / (self.h ** n_features)
            superweight = superweight + kernel
        density = superweight / np.sum(sample_weight)
        return density
    
def kernelize(x, y, h, degree):
    	kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    	return kernel
    
def DENCLUE(D, eps=0.3, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(D)
    coreSampleMask = np.zeros_like(db.labels_, dtype = bool)
    coreSampleMask[db.core_sample_indices_] = True
    clusterLabels = iris.target
    uniqueClusterLabels = set(clusterLabels)
    colors = ['red', 'green', 'blue', 'grey', 'black']
    markers = ['x', 'o', '+', '*', '8', 'H', '3']
    for i, cluster in enumerate(uniqueClusterLabels):
        clusterIndex = (clusterLabels == cluster)
        coreSamples = D[clusterIndex & coreSampleMask]
        plt.scatter(coreSamples[:, 0] + coreSamples[:, 1], coreSamples[:, 2] + coreSamples[:, 3],c=colors[i],  marker=markers[i], s=80)
        noiseSamples = D[clusterIndex & ~coreSampleMask]
        plt.scatter(noiseSamples[:, 0] + noiseSamples[:, 1],noiseSamples[:, 2] + noiseSamples[:, 3], c=colors[i], marker=markers[i], s=26)
    plt.show()

DENCLUE(D, 10, 10)
