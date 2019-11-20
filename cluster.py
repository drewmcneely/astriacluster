#!/usr/bin/python3
import time
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice

filename = 'data.pkl'
df = pd.read_pickle(filename)

#Index(['CatalogId', 'NoradId', 'Name', 'BallCoeff', 'Country', 'Epoch', 'Cart',
#       'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom', 'EquEx', 'EquEy',
#       'EquHx', 'EquHy', 'EquLm', 'OrbitType', 'BirthDate', 'DragCoeff',
#       'ReflCoeff', 'AreaToMass', 'Operator', 'Users', 'Purpose',
#       'DetailedPurpose', 'LaunchMass', 'DryMass', 'Power', 'Lifetime',
#       'Contractor', 'LaunchSite', 'LaunchVehicle', 'Position', 'Velocity',
#       'SpecificEnergy', 'SpecificAngularMomentum', 'LRL'],
#      dtype='object')

df = df[np.logical_and(df.Ecc > 0 , df.Ecc < 1)]

df = df[[
    'SMA',
    'Ecc',
    'EquEx',
    'EquEy',
    'EquHx',
    'EquHy',
    'EquLm',
    'SpecificEnergy',
    'SpecificAngularMomentum',
    'LRL',
    ]]

cols = ['LRL','SpecificAngularMomentum']

tmp = pd.concat([pd.DataFrame(df[x].to_numpy().tolist()).add_prefix(x) for x in cols], axis=1)
df = pd.concat([df, tmp], axis=1)
df = df.dropna()

df1 = df[[
    'SpecificEnergy',
    'LRL0',
    'LRL1',
    'LRL2',
    'SpecificAngularMomentum0',
    'SpecificAngularMomentum1',
    'SpecificAngularMomentum2'
    ]]

#print(df1.columns)
#Index(['SpecificEnergy', 'LRL0', 'LRL1', 'LRL2', 'SpecificAngularMomentum0',
#       'SpecificAngularMomentum1', 'SpecificAngularMomentum2'],
#      dtype='object')

df1 = StandardScaler().fit_transform(df1)

# Params
eps = 0.3
min_samples=15

algorithm = cluster.DBSCAN(eps=eps)
print("Fitting df1 to algorithm")
t0 = time.time()

# catch warnings related to kneighbors_graph
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="the number of connected components of the " +
            "connectivity matrix is [0-9]{1,2}" +
            " > 1. Completing it to avoid stopping the tree early.",
        category=UserWarning)
    warnings.filterwarnings(
        "ignore",
        message="Graph is not fully connected, spectral embedding" +
        " may not work as expected.",
        category=UserWarning)
    algorithm.fit(df1)

t1 = time.time()
if hasattr(algorithm, 'labels_'):
    y_pred = algorithm.labels_.astype(np.int)
else:
    y_pred = algorithm.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111)

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00']),
    int(max(y_pred) + 1))))
# add black color for outliers (if any)
colors = np.append(colors, ["#000000"])
ax.scatter(df['SMA'], df['Ecc'], s=10, color=colors[y_pred])
ax.set(
        title='Clustering of Keplerian Elements using DBSCAN',
        xlim=(0,6e7),
        ylim=(0,1),
        xlabel='SMA',
        ylabel='Ecc'
        )

#plt.xlim(-2.5, 2.5)
#plt.ylim(-2.5, 2.5)
#plt.xticks(())
#plt.yticks(())
#plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
#transform=plt.gca().transAxes, size=15,
#horizontalalignment='right')
#plot_num += 1

plt.show()
#space = np.arange(len(df1.index))
#reachability = clust.reachability_[clust.ordering_]
#labels = clust.labels_[clust.ordering_]
#
#plt.figure(figsize=(10, 7))
#G = gridspec.GridSpec(2, 3)
#ax1 = plt.subplot(G[0, :])
#ax2 = plt.subplot(G[1, 0])
#ax3 = plt.subplot(G[1, 1])
#ax4 = plt.subplot(G[1, 2])
#
## Reachability plot
#colors = ['g.', 'r.', 'b.', 'y.', 'c.']
#for klass, color in zip(range(0, 5), colors):
#    Xk = space[labels == klass]
#    Rk = reachability[labels == klass]
#    ax1.plot(Xk, Rk, color, alpha=0.3)
#ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
#ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
#ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
#ax1.set_ylabel('Reachability (epsilon distance)')
#ax1.set_title('Reachability Plot')
#
#plt.tight_layout()
#plt.show()
