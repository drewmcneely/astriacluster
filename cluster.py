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

def main():
    raw_data = read_data()
    all_data, clustering_data = clean_data(raw_data)
    labels = dbscan(clustering_data)
    data_to_plot = all_data[['SMA', 'Ecc']]
    plot(data_to_plot, labels)


def read_data(filename='input/data.pkl'): return pd.read_pickle(filename)

def clean_data(df):
    """ Take a full DataFrame and return cleaned DataFrames

    The column labels of the input are expected to be as follows.
    df.column:
    Index(['CatalogId', 'NoradId', 'Name', 'BallCoeff', 'Country', 'Epoch', 'Cart',
           'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom', 'EquEx', 'EquEy',
           'EquHx', 'EquHy', 'EquLm', 'OrbitType', 'BirthDate', 'DragCoeff',
           'ReflCoeff', 'AreaToMass', 'Operator', 'Users', 'Purpose',
           'DetailedPurpose', 'LaunchMass', 'DryMass', 'Power', 'Lifetime',
           'Contractor', 'LaunchSite', 'LaunchVehicle', 'Position', 'Velocity',
           'SpecificEnergy', 'SpecificAngularMomentum', 'LRL'],
          dtype='object')

    Two DataFrames are in the output:
    all_data: Clean data with all relevant columns
    clustering_data: one with only the data necessary for clustering

    all_data.columns    
    Index(['Position', 'Velocity', 'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom',
           'EquEx', 'EquEy', 'EquHx', 'EquHy', 'EquLm', 'SpecificEnergy', 'LRL',
           'SpecificAngularMomentum', 'LRL0', 'LRL1', 'LRL2',
           'SpecificAngularMomentum0', 'SpecificAngularMomentum1',
           'SpecificAngularMomentum2'],
          dtype='object') 

    clustering_data.columns
    Index(['SpecificEnergy', 'LRL0', 'LRL1', 'LRL2', 'SpecificAngularMomentum0',
           'SpecificAngularMomentum1', 'SpecificAngularMomentum2'],
        dtype='object')
    """

    df = df[np.logical_and(df.Ecc > 0 , df.Ecc < 1)]

    df = df[[
        'Position', 'Velocity',
        'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom',
        'EquEx', 'EquEy', 'EquHx', 'EquHy', 'EquLm',
        'SpecificEnergy', 'LRL', 'SpecificAngularMomentum',
        ]]

    def flatten(df):
        cols = ['LRL','SpecificAngularMomentum']
        tmp = pd.concat([pd.DataFrame(df[x].to_numpy().tolist()).add_prefix(x) for x in cols], axis=1)
        return pd.concat([df, tmp], axis=1)

    df = flatten(df)
    all_data = df.dropna()

    clustering_data = all_data[[
        'SpecificEnergy',
        'LRL0',
        'LRL1',
        'LRL2',
        'SpecificAngularMomentum0',
        'SpecificAngularMomentum1',
        'SpecificAngularMomentum2'
        ]]

    return all_data, clustering_data

def dbscan(clustering_data):
    """ Take clustering data and return a list of labels 

    input: dataframe of data to be clustered
    output: list of ints representing labels for each row in clustering_data
    """

    clustering_data = StandardScaler().fit_transform(clustering_data)

    # Params
    eps = 0.3
    min_samples=15

    algorithm = cluster.DBSCAN(eps=eps)
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

        algorithm.fit(clustering_data)

    t1 = time.time()
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
    else:
        labels = algorithm.predict(clustering_data)

    return labels

def plot(data, labels):
    """ Plot the given DataFrame with colors corresponding to given labels.

    input data must have rows representing points and columns representing axes
    data.shape must be (num_points, 2) or (num_points, 3)
    len(labels) must be num_points
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = np.array(
        list(
            islice(
                cycle(
                    ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
                    ),
                int(max(labels) + 1)
                )
            )
        )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])

    ax.scatter(data['SMA'], data['Ecc'], s=10, color=colors[labels])
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

if __name__=="__main__": main()
