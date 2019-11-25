#!/usr/bin/python3
import warnings

import pandas as pd
import numpy as np

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from plot_cluster import *

import pickle

def main():
    data = pd.read_pickle('output/clustered_data.pkl')
    with open('output/optics.pkl', 'rb') as f:
        clust = pickle.load(f)

    def labels_eps(eps):
        out = cluster.cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=eps)
        return out
    labels = clust.labels_[clust.ordering_]

    eps = [0.1, 0.2, 0.3, 0.5, 2.0]
    db_labels = [labels] + [labels_eps(e) for e in eps]
    figs = [constants_of_motion_plot(data, l) for l in db_labels]
    fig2s = [keplerian_plot(data, l) for l in db_labels]
    fig3s = [equinoctial_plot(data, l) for l in db_labels]

    titleadd = ["Default OPTICS"] + ["epsilon = %f" % e for e in eps]
    for fs in (figs, fig2s, fig3s):
        for k, f in enumerate(fs):
            s = f._suptitle.get_text()
            f._suptitle.set_text(s + "\n" + titleadd[k])

    for i in plt.get_fignums():
        f = plt.figure(i)
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        #plt.constrained_layout()
        plt.savefig('figure%d.png' % i, dpi=200)

def optics_save():
    raw_data = pd.read_pickle('input/data.pkl')
    all_data, clustering_data = clean_data(raw_data)
    clust = optics(clustering_data, eps=0.15)
    all_data.to_pickle('output/clustered_data.pkl')
    with open('output/optics.pkl', 'wb') as f:
        pickle.dump(clust, f)

def cluster_data():
    raw_data = pd.read_pickle('input/data.pkl')
    all_data, clustering_data = clean_data(raw_data)
    all_data['dbscan_labels'] = dbscan(clustering_data, eps=0.15)
    all_data.to_pickle('output/clustered_data.pkl')

def plot_data():
    data = pd.read_pickle('output/clustered_data.pkl')
    labels = data['dbscan_labels']
    fig = equinoctial_plot(data, labels)
    fig2 = keplerian_plot(data, labels)
    fig3 = constants_of_motion_plot(data, labels)
    plt.show()

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

    clustering_data = StandardScaler().fit_transform(clustering_data)

    return all_data, clustering_data

def dbscan(clustering_data, eps=0.3, min_samples=15):
    """ Take clustering data and return a list of labels 

    input: dataframe of data to be clustered
    output: list of ints representing labels for each row in clustering_data
    """

    print("dbscan")
    algorithm = cluster.DBSCAN(eps=eps, min_samples=min_samples)

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

    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
    else:
        labels = algorithm.predict(clustering_data)

    return labels

def optics(clustering_data, eps=0.3, min_samples=15):
    """ Take clustering data and return a list of labels 

    input: dataframe of data to be clustered
    output: list of ints representing labels for each row in clustering_data
    """

    print("optics")
    algorithm = cluster.OPTICS()

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

    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_.astype(np.int)
    else:
        labels = algorithm.predict(clustering_data)

    return algorithm

if __name__=="__main__": main()
