import spaceobjectcluster as cl
import json

clust = cl.read_optics_object()
data = cl.label_data(cl.read_unlabeled_data(), clust, epsilons=[0.1])

data = data.groupby('labels_epsilon_0.10')

for (l, d) in data:
    position = d[['Position0', 'Position1', 'Position2']]
    position = position.to_numpy() / 6356000
    if l == -1: fname = 'outliers.json'
    else: fname = str(l) + '.json'

    with open(fname, 'w') as f:
        json.dump(position.tolist(), f)
