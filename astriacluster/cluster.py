#!/usr/bin/python3
import warnings

import pandas as pd
import numpy as np
from numpy import cross, pi
from numpy.linalg import norm

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from collections import Counter

def fit_optics(data):
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

default_eps = 0.1
def dbscan_labels_from_optics(optics_object, epsilon=default_eps):
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

def organize_labeled_data(data, epsilon=default_eps):
    label_column = 'labels_epsilon_' + '%.2f'%epsilon
    labels = data[label_column]
    unique_labels = set(labels)
    organized_data = {}
    for l in unique_labels:
        labels_bool = (labels == l)
        organized_data[l] = data[labels_bool]

    return organized_data
