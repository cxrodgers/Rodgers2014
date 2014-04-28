# Load dumped times and bin
# This will be used to plot exemplar traces later

import numpy as np, os.path
import pandas, itertools
from kkpandas import kkrs
import kkpandas
from ns5_process import myutils, LBPB
import my, my.dataload

# Load results from main0
try:
    ulabel2block2folded
except NameError:
    ulabel2block2folded = myutils.pickle_load('ulabel2block2folded')

# Binning params
gs = kkpandas.base.GaussianSmoother(smoothing_window=0.05)
bins = np.linspace(-5, 5, int(round(10/.050)))

# Use these bins throughout this directory
bins = np.linspace(-5, 5, 200, False)
np.savetxt('bins', bins)


# Bin each
ulabel2binned = {}
for ulabel, block2folded in ulabel2block2folded.items():
    binned = kkpandas.Binned.from_dict_of_folded(block2folded, bins=bins,
        meth=gs.smooth)
    ulabel2binned[ulabel] = binned

# Save
myutils.pickle_dump(ulabel2binned, 'ulabel2binned')

