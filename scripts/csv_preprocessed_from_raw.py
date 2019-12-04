import sys, os
sys.path.append(os.path.abspath("../astriacluster/"))
from astriacluster import io, preprocessing, cluster

jsondata = io.read_spaceobjects_json()
processed_data = preprocessing.preprocess_data(jsondata)
io.csv_processed_data(processed_data)
