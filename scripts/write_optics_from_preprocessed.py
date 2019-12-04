import sys, os
sys.path.append(os.path.abspath("../astriacluster/"))
from astriacluster import io, preprocessing, cluster

data = io.read_processed_data()
