Files in this folder should be scripts using the astriacluster package to manipulate data, generate plots, etc.
Feel free to also do the same in a live python shell, but try to document what you did.

All scripts should start with the following:

	import sys, os
	sys.path.append(os.path.abspath("../astriacluster/")
	from astriacluster import [cluster, io, constants, preprocessing, etc.]
