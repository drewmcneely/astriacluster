from cluster import *

import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement

import pickle

from collections import Counter

data = pd.read_pickle('output/clustered_data.pkl')
with open('output/optics.pkl', 'rb') as f:
    clust = pickle.load(f)

def labels_eps(eps):
    out = cluster.cluster_optics_dbscan(reachability=clust.reachability_,
                               core_distances=clust.core_distances_,
                               ordering=clust.ordering_, eps=eps)
    return out

eps = 0.1
hlim = 1e11
labels = labels_eps(eps)
data["labels"] = labels
print(data.shape)
data = data[np.abs(data["SpecificAngularMomentum0"]) < hlim]
data = data[np.abs(data["SpecificAngularMomentum1"]) < hlim]
data = data[np.abs(data["SpecificAngularMomentum2"]) < hlim]
print(data.shape)
print(Counter(labels).most_common(10))

#hs = data[["SpecificAngularMomentum0","SpecificAngularMomentum1", "SpecificAngularMomentum2" ]].to_numpy()
#hs = hs/hlim * 4.0
#hs = list(map(tuple, hs))
#hs = np.array(hs, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#hs = hs[:5000]
## binary little endian
#el = PlyElement.describe(hs, 'angular_momentum')
#PlyData([el]).write('html/ply/head_angular_momentum.ply')
#print("done")
