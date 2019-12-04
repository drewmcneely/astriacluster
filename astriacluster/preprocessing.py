from . import constants

import numpy as np

def preprocess_data(jsondata):
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

    data = []
    for num in jsondata:
        data += jsondata[num]['Objects']

    df = pd.DataFrame(data)
    df = df[pd.notnull(df.Cart)] # Purge rows with a missing 'Cart' value

    positions = [x[:3] for x in df.Cart]
    velocities = [x[3:] for x in df.Cart]

    energies = [norm(v)**2 / 2 - mu/norm(r) for (r, v) in zip(positions, velocities)]
    h_vectors = [cross(r, v) for (r, v) in zip(positions, velocities)]
    lrl_vectors = [(cross(v, h) / mu) - (r/norm(r)) for (r, v, h) in zip(positions, velocities, h_vectors)]

    def flatten(vector_list, name):

        vector_array = np.array(vector_list)
        df = pd.DataFrame(vector_array)
        df.add_prefix(name + '_')

        return df

    positions = flatten(positions, 'r')
    velocities = flatten(velocities, 'v')
    h_vectors = flatten(h_vectors, 'h')
    lrl_vectors = flatten(lrl_vectors, 'lrl')

    df = df.drop(columns=['Cart'])
    df = pd.concat([df, positions, velocities, h_vectors, lrl_vectors], axis='columns')
    df['energy'] = energies

    df['perigee'] = df['SMA'] * (1 - df['Ecc'])
    df['apogee'] = df['SMA'] * (1 + df['Ecc'])
    df['perigee_altitude'] = df['perigee'] - constants.earth_radius
    df['apogee_altitude'] = df['apogee'] - constants.earth_radius

    df['period'] = 2*constants.pi * np.sqrt(df['SMA']**3 / mu)

    return df

def purge_outliers(data):
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
        col = "h_" + i
        data = data[np.abs(data[col]) < hlim]

    data = data[np.logical_and(data["h_2"] > -5.6e10,
                               data["h_2"] < 1.4e11,)]

    nullable_entries = data[[
        'SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'MeanAnom',
        'EquEx', 'EquEy', 'EquHx', 'EquHy', 'EquLm',
        'energy',
        'lrl_0', 'lrl_1', 'lrl_2',
        'h_0', 'h_1', 'h_2',
        'r_0', 'r_1', 'r_2', 
        'v_0', 'v_1', 'v_2', 
        ]]

    records_to_keep = nullable_entries.notnull().any(axis=1)

    data = data[records_to_keep]

    return data

