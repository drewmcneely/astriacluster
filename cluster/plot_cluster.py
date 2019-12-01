import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from itertools import cycle, islice

from math import pi

import numpy as np

def label_colors(labels):
    color_list = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    outlier_color = ["#00000020"]

    colors = list(islice(cycle(color_list), int(max(labels)+1))) + outlier_color
    return colors

def plot(ax, data, labels):
    """ Plot the given DataFrame with colors corresponding to given labels.

    input data must have rows representing points and columns representing axes
    data.shape must be (num_points, 2) or (num_points, 3)
    len(labels) must be num_points
    """

    data = np.stack(data.to_numpy()).T

    if data.shape[0] == 2: proj=None
    elif data.shape[0] == 3: proj='3d'

    colors = label_colors(labels)
    colors = [colors[l] for l in labels]
    ax.scatter(*data, s=1, color=colors)

def equinoctial_plot(data, labels):
    fig, (pq, hk) = plt.subplots(1,2)
    fig.suptitle("Clustered Scatter Plot of Equinoctial Elements\nfor all objects known to AstriaGraph as of 10/30/2019")

    plot(pq, data[['EquEx', 'EquEy']], labels)
    pq.set(xlim=(-0.8,0.8), ylim=(-0.8,0.8),
        xlabel="$p=e\ \cos(\omega + \Omega)$",
        ylabel="$q=e\ \sin(\omega + \Omega)$",
        )

    plot(hk, data[['EquHx', 'EquHy']], labels)
    hk.set(xlim=(-1.3,1.3), ylim=(-1.3,1.3),
        xlabel="$h=\\tan(i/2)\ cos(\Omega)$",
        ylabel="$k=\\tan(i/2)\ sin(\Omega)$",
        )

    return fig

def keplerian_plot(data, labels):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Clustered Scatter Plot of Keplerian Elements\nfor all objects known to AstriaGraph as of 10/30/2019")
    plot(ax, data[['SMA', 'Ecc', 'Inc']], labels)
    ax.set(
        xlim=(0,5e7), ylim=(0,1), zlim=(0,2.0),
        xlabel="Semimajor Axis", ylabel="Eccentricity", zlabel="Inclination",
        )

    return fig

def constants_of_motion_plot(data, labels):

    fig = plt.figure()
    fig.suptitle("Clustered Scatter Plot of Constants of Motion\nfor all objects known to AstriaGraph as of 10/30/2019")

    h = fig.add_subplot(122, projection='3d')
    plot(h, data['SpecificAngularMomentum'], labels)
    h_lim = 1e11
    h.set(title="Angular Momentum Vectors",
            xlim=(-h_lim, h_lim), ylim=(-h_lim, h_lim), zlim=(-2.5e10, 1.4e11),
            xlabel="$h_x$", ylabel="$h_y$", zlabel="$h_z$",
            )

    e = fig.add_subplot(121, projection='3d')
    plot(e, data['LRL'], labels)
    e.set(title="Laplace-Runge-Lenz (Eccentricity) Vectors",
            xlim=(-1,1), ylim=(-1,1), zlim=(-1,1),
            xlabel="$e_x$", ylabel="$e_y$", zlabel="$e_z$",
            )

    return fig

def reachability_plot(clust):

    fig, ax = plt.subplots(1)

    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    space = np.arange(len(labels))

    colors = label_colors(labels)
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax.plot(Xk, Rk, color, alpha=0.3)
    ax.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax.set_ylabel('Reachability (epsilon distance)')
    ax.set_title('Reachability Plot')
    
    return fig

