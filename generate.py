#!/usr/bin/python3
import json

from math import pi

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from numpy import cross
from numpy.linalg import norm

import pandas as pd

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

def main():
    df = generate_data()
    df.to_pickle('data.pkl')

def generate_data(filename="SpaceObjects-20191030.json"):
    """ Load the specified json file and return a pandas DataFrame with extra calculated data.
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

    return df

if __name__ == '__main__':
    main()
