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
def csv_processed_data(data, filename=(processed_data_dir/'data.csv')): data.to_csv(filename, index=False)

def write_optics_object(optics_object, filename=optics_path):
    with open(filename, 'wb') as f:
        pickle.dump(optics_object, f)

def read_optics_object(filename='optics_object.pkl'):
    with open(filename, 'rb') as f:
        optics_object = pickle.load(f)
    return optics_object

def write_labeled_data_to_aframe_json(data, filename='/home/drew/cluster/html/json/labeled_data.json'):
    data.to_json(filename, 'records')

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
