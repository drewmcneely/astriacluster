#!/usr/bin/python3

import pandas as pd
import numpy as np

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

cols = ['LRL','SpecificAngularMomentum']

df1 = pd.concat([pd.DataFrame(df[x].to_numpy().tolist()).add_prefix(x) for x in cols], axis=1)
df1.insert(0, 'SpecificEnergy', df['SpecificEnergy'])
