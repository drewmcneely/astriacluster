#!/usr/bin/python3
import json

from math import pi

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from numpy import cross
from numpy.linalg import norm

# data.keys()
# ['0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10']
# 
# data['0'].keys()
# ['Objects', 'LastUpdateTime']
# 
# type(data['0']['LastUpdateTime'])
# str
# 
# type(data['0']['Objects'])
# list
# 
# type(data['0']['Objects'][0])
# dict
# 
# data['0']['Objects'][0].keys()
# 
# ['CatalogId', 'NoradId', 'Name', 'BallCoeff', 'Birthdate', 'Country', 'Epoch', 'Cart', 'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom', 'EquEx', 'EquEy', 'EquHx', 'EquHy', 'EquLm', 'OrbitType']
# 
# EquEx = e*cos(omega + Omega)
# EquEy = e*sin(omega + Omega)
# EquHx = tan(i/2)*cos(Omega)
# EquHy = tan(i/2)*sin(Omega)
# l_nu = nu + omega + Omega
# EquLm = MeanAnom + omega + Omega ???

mu = 3.986004418e14

def generate_data():
    """ Return a dictionary filled with data pulled from AstriaGraph on 10-30-2019.

    Each dictionary entry will be a list of values, entries in lists corresponding to each object
    Dictionary will have the following keys:
    CatalogId
    NoradId
    Name
    BallCoeff
    Birthdate
    Country
    Epoch
    Cart
    SMA
    Ecc
    Inc
    RAAN
    ArgP
    MeanAnom
    EquEx
    EquEy
    EquHx
    EquHy
    EquLm
    OrbitType
    Position (Values in meters)
    Velocity (Values in meters/second)
    AngularMomentum
    LRL
    Energy
    AngularMomentum_x
    AngularMomentum_y
    AngularMomentum_z
    LRL_x
    LRL_y
    LRL_z
    """

    filename = "input/SpaceObjects-20191030.json"
    with open(filename, 'r') as f:
        data = json.load(f)

    output_data = { 
        'CatalogId': [],
        'NoradId': [],
        'Name': [],
        'BallCoeff': [],
        'Birthdate': [],
        'Country': [],
        'Epoch': [],
        'Cart': [],
        'SMA': [],
        'Ecc': [],
        'Inc': [],
        'RAAN': [],
        'ArgP': [],
        'MeanAnom': [],
        'EquEx': [],
        'EquEy': [],
        'EquHx': [],
        'EquHy': [],
        'EquLm': [],
        'OrbitType': [],
        }

    for num in data:
        for o in data[num]['Objects']:
            for k in output_data.keys():
                try:
                    output_data[k].append(o[k])
                except:
                    pass


    rs = [np.array(x[:3]) for x in output_data['Cart']]
    vs = [np.array(x[3:]) for x in output_data['Cart']]
    Es = [norm(v)**2 / 2 - mu/norm(r) for (r, v) in zip(rs, vs)]
    hs = [cross(r, v) for (r, v) in zip(rs, vs)]
    lrls = [(cross(v, h) / mu) - (r/norm(r)) for (r, v, h) in zip(rs, vs, hs)]
    es = [norm(lrl) for lrl in lrls]

    h_xs = [h[0] for h in hs]
    h_ys = [h[1] for h in hs]
    h_zs = [h[2] for h in hs]

    lrl_xs = [lrl[0] for lrl in lrls]
    lrl_ys = [lrl[1] for lrl in lrls]
    lrl_zs = [lrl[2] for lrl in lrls]

    dat = {
        'Position': rs,
        'Velocity': vs,
        'AngularMomentum': hs,
        'LRL': lrls,
        'Energy': Es,
        'AngularMomentum_x': h_xs,
        'AngularMomentum_y': h_ys,
        'AngularMomentum_z': h_zs,
        'LRL_x': lrl_xs,
        'LRL_y': lrl_ys,
        'LRL_z': lrl_zs,
        }

    output_data.update(dat)

    return output_data

data = generate_data()

size = 1
def generate_equinoctial_plot():
    fig, (pq_plot, hk_plot) = plt.subplots(1,2)
    fig.suptitle("Scatter Plot of Equinoctial Elements\nfor all objects known to AstriaGraph as of 10/30/2019")
    pq_plot.scatter(data['EquEx'], data['EquEy'], s=size)
    pq_plot.set_xlabel("$e\ \cos(\omega + \Omega)$")
    pq_plot.set_ylabel("$e\ \sin(\omega + \Omega)$")
    pq_plot.set(xlim=(-0.8,0.8), ylim=(-0.8,0.8))
    hk_plot.scatter(data['EquHx'], data['EquHy'], s=size)
    hk_plot.set_xlabel("$\\tan(i/2)\ cos(\Omega)$")
    hk_plot.set_ylabel("$\\tan(i/2)\ sin(\Omega)$")
    hk_plot.set(xlim=(-1.3,1.3), ylim=(-1.3,1.3))

    return fig

def generate_keplerian_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("Scatter Plot of Keplerian Elements\nfor all objects known to AstriaGraph as of 10/30/2019")
    ax.scatter(data['SMA'], data['Ecc'], data['Inc'], s=size)
    ax.set(
        xlim=(0,6e7),
        ylim=(0,1),
        zlim=(0,pi),
        xlabel="SMA",
        ylabel="ECC",
        zlabel="INC",
        )

    return fig

def generate_constants_of_motion_plot():

    fig = plt.figure()
    fig.suptitle("Scatter Plot of Constants of Motion\nfor all objects known to AstriaGraph as of 10/30/2019")

    hvec_plot = fig.add_subplot(122, projection='3d')
    hvec_plot.plot(data['AngularMomentum_x'], data['AngularMomentum_y'], data['AngularMomentum_z'], 'o', markersize=size)
    #print(hvec_plot.properties().keys())

    h_lim = 1e11
    hvec_plot.set(title="Angular Momentum Vectors",
            xlim=(-h_lim, h_lim),
            ylim=(-h_lim, h_lim),
            zlim=(-h_lim, h_lim),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            )

    evec_plot = fig.add_subplot(121, projection='3d')
    evec_plot.plot(data['LRL_x'], data['LRL_y'], data['LRL_z'], 'o', markersize=size) 
    evec_plot.set(title="Laplace-Runge-Lenz (Eccentricity) Vectors",
            xlim=(-1,1),
            ylim=(-1,1),
            zlim=(-1,1),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            )

    #es = [norm(lrl) for lrl in data['LRL']]
    #eccs = data['Ecc']
    #fig2, (es_plot, eccs_plot) = plt.subplots(2)
    #es_plot.hist(es, bins=1000, range=(0,0.02))
    #eccs_plot.hist(eccs, bins=1000, range=(0,0.02))

    return fig

fig = generate_constants_of_motion_plot()
fig2 = generate_equinoctial_plot()
plt.show()
