from pathlib import Path

import json
import pickle

import pandas as pd

data_dir = Path('/home/drew/astriacluster/data')
figures_dir = Path('/home/drew/astriacluster/figures')

raw_data_dir = data_dir / 'raw/'
raw_data_path = raw_data_dir / 'SpaceObjects-20191030.json'

processed_data_dir = data_dir / 'processed/'
processed_data_path = processed_data_dir / 'data.pkl'

labeled_data_dir = data_dir / 'labeled/'
labeled_data_path = labeled_data_dir / 'data.pkl'
optics_path = labeled_data_dir / 'optics.pkl'

clustered_data_dir = data_dir / 'clustered/'
clustered_data_path = clustered_data_dir / 'data.pkl'

def read_spaceobjects_json(filename=raw_data_path):
    with open(filename, 'r') as f:
        jsondata = json.load(f)
    return jsondata

def write_processed_data(data, filename=processed_data_path): data.to_pickle(filename)
def read_processed_data(filename=processed_data_path): return pd.read_pickle(filename)
def write_processed_data_csv(data, filename=(processed_data_dir/'data.csv')): data.to_csv(filename, index=False)

def write_optics_object(optics_object, filename=optics_path):
    with open(filename, 'wb') as f:
        pickle.dump(optics_object, f)

def read_optics_object(filename=optics_path):
    with open(filename, 'rb') as f:
        optics_object = pickle.load(f)
    return optics_object

def write_processed_data(data, filename=processed_data_path): data.to_pickle(filename)
def read_processed_data(filename=processed_data_path): return pd.read_pickle(filename)

def csv_clusters(data, labels, data_path=clustered_data_dir):
    relevant_columns = ['CatalogId', 'NoradId', 'Name']
    data = data[relevant_columns]

    for i in data.groupby(labels):
        label = i[0]
        df = i[1]

        filepath = data_path / (label + ".csv")
        df.to_csv(filepath, index=False)

def write_clusters_json(data, labels, filename='clusters.json'):
    """ Take a DataFrame and write as a json object featuring a 2D array for each cluster.
    """

    filepath = clustered_data_dir / filename
    jsondata = {}
    for i in data.groupby(labels):
        label = i[0]
        df = i[1].to_numpy().tolist()
        jsondata[label] = df

    with open(filepath, 'w') as f:
        json.dump(jsondata, f)



