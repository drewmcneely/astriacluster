from pathlib import Path

import json
import pickle

data_dir = Path('/home/drew/astriacluster/data')
figures_dir = Path('/home/drew/astriacluster/figures')

raw_data_path = data_dir / 'raw/SpaceObjects-20191030.json'
processed_data_path = data_dir / 'processed/data.pkl'

optics_path = data_dir / 'clustered/optics.pkl'
clustered_data_path = data_dir / 'clustered/data.pkl'

def from_spaceobjects_json(filename=raw_data_path):
    with open(filename, 'r') as f:
        jsondata = json.load(f)
    return jsondata

def write_processed_data(data, filename=processed_data_path): data.to_pickle(filename)
def read_processed_data(filename=processed_data_path) -> UnlabeledData: return pd.read_pickle(filename)

def write_optics_object(optics_object, filename=optics_path):
    with open(filename, 'wb') as f:
        pickle.dump(optics_object, f)

def read_optics_object(filename='optics_object.pkl'):
    with open(filename, 'rb') as f:
        optics_object = pickle.load(f)
    return optics_object

def write_labeled_data_to_aframe_json(data, filename='/home/drew/cluster/html/json/labeled_data.json'):
    data.to_json(filename, 'records')
