#!/usr/bin/python3
import warnings

import pandas as pd
import numpy as np
from numpy import cross
from numpy.linalg import norm

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from plot_cluster import *

import pickle
import json

from collections import Counter

from math import pi

UnlabeledData = pd.DataFrame
LabeledData = pd.DataFrame
OpticsObject = cluster.OPTICS

def from_spaceobjects_json(filename="../input/SpaceObjects-20191030.json") -> UnlabeledData:
    """ Load the specified json file and return a pandas DataFrame with extra calculated data.

    data.keys()
    ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10']
    
    data['0'].keys()
    ['Objects', 'LastUpdateTime']
    
    type(data['0']['LastUpdateTime'])
    str
    
    type(data['0']['Objects'])
    list
    
    type(data['0']['Objects'][0])
    dict
    
    data['0']['Objects'][0].keys()
    
    ['CatalogId', 'NoradId', 'Name', 'BallCoeff', 'Birthdate', 'Country', 'Epoch', 'Cart', 'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom', 'EquEx', 'EquEy', 'EquHx', 'EquHy', 'EquLm', 'OrbitType']
    
    EquEx = e*cos(omega + Omega)
    EquEy = e*sin(omega + Omega)
    EquHx = tan(i/2)*cos(Omega)
    EquHy = tan(i/2)*sin(Omega)
    l_nu = nu + omega + Omega
    EquLm = MeanAnom + omega + Omega ???

    columns of output are as follows:
    Index(['CatalogId', 'NoradId', 'Name', 'BallCoeff', 'Country', 'Epoch', 'SMA',
       'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom', 'EquEx', 'EquEy', 'EquHx',
       'EquHy', 'EquLm', 'OrbitType', 'BirthDate', 'DragCoeff', 'ReflCoeff',
       'AreaToMass', 'Operator', 'Users', 'Purpose', 'DetailedPurpose',
       'LaunchMass', 'DryMass', 'Power', 'Lifetime', 'Contractor',
       'LaunchSite', 'LaunchVehicle', 'SpecificEnergy', 'LRL0', 'LRL1', 'LRL2',
       'SpecificAngularMomentum0', 'SpecificAngularMomentum1',
       'SpecificAngularMomentum2', 'Position0', 'Position1', 'Position2',
       'Velocity0', 'Velocity1', 'Velocity2'],
      dtype='object')

    """

    with open(filename, 'r') as f:
        jsondata = json.load(f)

    data = []
    for num in jsondata:
        data += jsondata[num]['Objects']

    df = pd.DataFrame(data)
    df = df[pd.notnull(df.Cart)] # Purge rows with a missing 'Cart' value

    mu = 3.986004418e14
    df['Position'] = [x[:3] for x in df.Cart]
    df['Velocity'] = [x[3:] for x in df.Cart]
    df['SpecificEnergy'] = [norm(v)**2 / 2 - mu/norm(r) for (r, v) in zip(df.Position, df.Velocity)]
    df['SpecificAngularMomentum'] = [cross(r, v) for (r, v) in zip(df.Position, df.Velocity)]
    df['LRL'] = [(cross(v, h) / mu) - (r/norm(r)) for (r, v, h) in zip(df.Position, df.Velocity, df.SpecificAngularMomentum)]

    def flatten(df):
        cols = ['LRL','SpecificAngularMomentum', 'Position', 'Velocity']
        tmp = pd.concat([pd.DataFrame(df[x].to_numpy().tolist()).add_prefix(x) for x in cols], axis=1)
        return pd.concat([df, tmp], axis=1)

    df = flatten(df)
    df = df.drop(columns=['Cart', 'Position', 'Velocity', 'SpecificAngularMomentum', 'LRL'])

    return df

def clean_data(data: UnlabeledData) -> UnlabeledData:
    """ Take a DataFrame and return cleaned DataFrames.

    Columns expected in the input and output are as follows:
    Index(['CatalogId', 'NoradId', 'Name', 'BallCoeff', 'Country', 'Epoch', 'SMA',
       'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom', 'EquEx', 'EquEy', 'EquHx',
       'EquHy', 'EquLm', 'OrbitType', 'BirthDate', 'DragCoeff', 'ReflCoeff',
       'AreaToMass', 'Operator', 'Users', 'Purpose', 'DetailedPurpose',
       'LaunchMass', 'DryMass', 'Power', 'Lifetime', 'Contractor',
       'LaunchSite', 'LaunchVehicle', 'SpecificEnergy', 'LRL0', 'LRL1', 'LRL2',
       'SpecificAngularMomentum0', 'SpecificAngularMomentum1',
       'SpecificAngularMomentum2', 'Position0', 'Position1', 'Position2',
       'Velocity0', 'Velocity1', 'Velocity2'],
      dtype='object')
    """

    data = data[np.logical_and(data.Ecc > 0 , data.Ecc < 1)]

    hlim = 1.2e11
    for i in ["0", "1",]:
        col = "SpecificAngularMomentum" + i
        data = data[np.abs(data[col]) < hlim]

    data = data[np.logical_and(data["SpecificAngularMomentum2"] > -5.6e10,
                               data["SpecificAngularMomentum2"] < 1.4e11,)]

    nullable_entries = data[[
        'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom',
        'EquEx', 'EquEy', 'EquHx', 'EquHy', 'EquLm',
        'SpecificEnergy', 'LRL0', 'LRL1', 'LRL2',
        'SpecificAngularMomentum0',
        'SpecificAngularMomentum1',
        'SpecificAngularMomentum2',
        'Position0', 'Position1', 'Position2', 
        'Velocity0', 'Velocity1', 'Velocity2', 
        ]]

    records_to_keep = nullable_entries.notnull().any(axis=1)

    data = data[records_to_keep]

    return data

def write_unlabeled_data(data: UnlabeledData, filename='unlabeled_data.pkl'): data.to_pickle(filename)
def read_unlabeled_data(filename='unlabeled_data.pkl') -> UnlabeledData: return pd.read_pickle(filename)

def write_unlabeled_data_to_aframe_json(data: UnlabeledData, filename='/home/drew/cluster/html/json/unlabeled_data.json'):
    data.to_json(filename, 'records')

def optics(data: UnlabeledData) -> OpticsObject:
    """ Return an OpticsObject fitted to given data.

    input: dataframe of data to be clustered
    output: list of ints representing labels for each row in clustering_data
    """

    clusterable_data = data[[
        'SpecificEnergy',
        'LRL0',
        'LRL1',
        'LRL2',
        'SpecificAngularMomentum0',
        'SpecificAngularMomentum1',
        'SpecificAngularMomentum2'
        ]]

    clusterable_data = StandardScaler().fit_transform(clusterable_data)

    algorithm = cluster.OPTICS()

    print("Fitting data...")
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
        algorithm.fit(clusterable_data)

    print("Done!")

    return algorithm

def write_optics_object(optics_object: OpticsObject, filename='optics_object.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(optics_object, f)

def read_optics_object(filename='optics_object.pkl') -> OpticsObject:
    with open(filename, 'rb') as f:
        optics_object = pickle.load(f)
    return optics_object

def dbscan_labels_from_optics(optics_object, epsilon):
    labels = cluster.cluster_optics_dbscan(
            reachability=optics_object.reachability_,
            core_distances=optics_object.core_distances_,
            ordering=optics_object.ordering_, eps=epsilon)
    return labels

def remap_labels_to_sorted(labels, num_labels=7):
    """ Return labels that sequence by the size of the cluster.

    -1 still corresponds to outliers
    """

    labelcount = Counter(labels)
    sorted_labels = [c[0] for c in labelcount.most_common()]
    sorted_labels.remove(-1)
    sorted_labels = sorted_labels[:num_labels]

    def labelmap(l):
        if l not in sorted_labels: return -1
        else: return sorted_labels.index(l)

    newlabels = list(map(labelmap, labels))
    return newlabels

def label_data(data, optics_object, epsilons=np.arange(0.05, 1.0, 0.05)):
    output = data.copy()
    for epsilon in epsilons:
        column_name = "labels_epsilon_" + "%.2f"%epsilon
        labels = dbscan_labels_from_optics(optics_object, epsilon)
        output[column_name] = remap_labels_to_sorted(labels)

    output["labels_optics"] = optics_object.labels_

    return output

def write_labeled_data_to_aframe_json(data: LabeledData, filename='/home/drew/cluster/html/json/labeled_data.json'):
    data.to_json(filename, 'records')

def organize_labeled_data(data: LabeledData, epsilon=0.1):
    label_column = 'labels_epsilon_' + '%.2f'%epsilon
    labels = data[label_column]
    unique_labels = set(labels)
    organized_data = {}
    for l in unique_labels:
        labels_bool = (labels == l)
        organized_data[l] = data[labels_bool]

    return organized_data

#def main():
#    data = pd.read_pickle('output/clustered_data.pkl')
#    with open('output/optics.pkl', 'rb') as f:
#        clust = pickle.load(f)
#
#    def labels_eps(eps):
#        out = cluster.cluster_optics_dbscan(reachability=clust.reachability_,
#                                   core_distances=clust.core_distances_,
#                                   ordering=clust.ordering_, eps=eps)
#        return out
#    labels = clust.labels_[clust.ordering_]
#
#    eps = [0.1, 0.2, 0.3, 0.5, 2.0]
#    db_labels = [labels] + [labels_eps(e) for e in eps]
#    figs = [constants_of_motion_plot(data, l) for l in db_labels]
#    fig2s = [keplerian_plot(data, l) for l in db_labels]
#    fig3s = [equinoctial_plot(data, l) for l in db_labels]
#
#    titleadd = ["Default OPTICS"] + ["epsilon = %f" % e for e in eps]
#    for fs in (figs, fig2s, fig3s):
#        for k, f in enumerate(fs):
#            s = f._suptitle.get_text()
#            f._suptitle.set_text(s + "\n" + titleadd[k])
#
#    for i in plt.get_fignums():
#        f = plt.figure(i)
#        plt.tight_layout(rect=[0, 0, 1, 0.85])
#        #plt.constrained_layout()
#        plt.savefig('figure%d.png' % i, dpi=200)

#def main2():
#    df = generate_data()
#    df.to_pickle('input/data.pkl')

#def generate_json():
#    data = pd.read_pickle('output/clustered_data.pkl')
#    with open('output/optics.pkl', 'rb') as f:
#        clust = pickle.load(f)
#
#
#    eps = 0.1
#    labels = labels_eps(eps)
#    data["labels"] = labels
#    print(data.shape)
#    print(data.shape)
#    print(Counter(labels).most_common(10))

#def optics_save():
#    raw_data = pd.read_pickle('input/data.pkl')
#    all_data, clustering_data = clean_data(raw_data)
#    clust = optics(clustering_data, eps=0.15)
#    all_data.to_pickle('output/clustered_data.pkl')
#    with open('output/optics.pkl', 'wb') as f:
#        pickle.dump(clust, f)

#def cluster_data():
#    raw_data = pd.read_pickle('input/data.pkl')
#    all_data, clustering_data = clean_data(raw_data)
#    all_data['dbscan_labels'] = dbscan(clustering_data, eps=0.15)
#    all_data.to_pickle('output/clustered_data.pkl')

#def plot_data():
#    data = pd.read_pickle('output/clustered_data.pkl')
#    labels = data['dbscan_labels']
#    fig = equinoctial_plot(data, labels)
#    fig2 = keplerian_plot(data, labels)
#    fig3 = constants_of_motion_plot(data, labels)
#    plt.show()

#def dbscan(clustering_data, eps=0.3, min_samples=15):
#    """ Take clustering data and return a list of labels 
#
#    input: dataframe of data to be clustered
#    output: list of ints representing labels for each row in clustering_data
#    """
#
#    print("dbscan")
#    algorithm = cluster.DBSCAN(eps=eps, min_samples=min_samples)
#
#    # catch warnings related to kneighbors_graph
#    with warnings.catch_warnings():
#        warnings.filterwarnings(
#            "ignore",
#            message="the number of connected components of the " +
#                "connectivity matrix is [0-9]{1,2}" +
#                " > 1. Completing it to avoid stopping the tree early.",
#            category=UserWarning)
#        warnings.filterwarnings(
#            "ignore",
#            message="Graph is not fully connected, spectral embedding" +
#            " may not work as expected.",
#            category=UserWarning)
#        algorithm.fit(clustering_data)
#
#    if hasattr(algorithm, 'labels_'):
#        labels = algorithm.labels_.astype(np.int)
#    else:
#        labels = algorithm.predict(clustering_data)
#
#    return labels

