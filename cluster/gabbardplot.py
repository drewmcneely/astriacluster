import spaceobjectcluster as cl

import numpy as np

import matplotlib.pyplot as plt

mu = 3.986004418e14

optics_object = cl.read_optics_object()
unlabeled_data = cl.read_unlabeled_data()

data = cl.label_data(unlabeled_data, optics_object, epsilons = [0.1])

data['Perigee'] = data['SMA'] * (1 - data['Ecc']) - 6356000
data['Apogee'] = data['SMA'] * (1 + data['Ecc']) - 6356000

data['Period'] = 2*np.pi * np.sqrt(data['SMA']**3 / mu)

organized_data = data.groupby('labels_epsilon_0.10')
for l in organized_data:
    label = l[0]
    df = l[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df['Period'], df['Perigee'], s=1, label='Perigee')
    ax.scatter(df['Period'], df['Apogee'], s=1, label='Apogee')
    ax.legend()
    ax.set(
            xlabel='Period [s]',
            ylabel = 'Height [m]',
            )
    if label == -1:
        title = fig.suptitle('Gabbard plot for outliers')
    else:
        title = fig.suptitle('Gabbard plot for label %i' % label)

    plt.savefig(title.get_text() + '.png')
