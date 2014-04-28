# Plot pivoted aurocs, find intervals of significance
# This dumps a dataframe called intervals, with columns t1, t2, and valid
# t1: index into `bins` at which significant effect begins
#   Inclusive. So if t1 == 50, then the time between bins[50] and bins[51]
#   is signif, but not any earlier.
# t2: index into `bins` at which significant effect ends
#   Half-open. So if t2 == 190, then the time between bins[190] and bins[191]
#   is not signif, but the previous bin is.
#   This can also be equal to len(bins) - 1, which means that it was significant
#   for the entire duration testable, up to and including the last bin.

import numpy as np, os.path
import pandas, itertools
from kkpandas import kkrs
import kkpandas
from ns5_process import myutils, LBPB
import my, my.dataload
import matplotlib.pyplot as plt

# Helper functions for interval-setting logic
def first_true(ser, which='first', mungval=None, **kwargs):
    """Returns index of first True in series, or one plus the last index"""
    if ser.ndim == 1:
        idx = np.where(ser)[0]
        if len(idx) == 0:
            return mungval if mungval else ser.index[-1] + 1
        else:
            return ser.index[idx[0]]
    else:
        return ser.apply(first_true, mungval=mungval, **kwargs)

def loss_of_signif(auroc, p, direction):
    """Returns boolean array of places where p > .05 or direction changes"""
    if direction:
        return ((p > 0.05) | (auroc < .5))
    else:
        return ((p > 0.05) | (auroc > .5))

# Pivoted aurocs, p values
piv = pandas.load('block_effect')
bins = np.loadtxt('bins')
bincenters = bins[:-1] + np.diff(bins) / 2.
refbin = np.argmin(np.abs(bins - 0.)) - 1

# Adjust p-values, all together (though this hardly changes anything)
piv['p'] = my.stats.r_adj_pval(piv['p'])

# Iterate over units and find interval of significance
outrec_l = []
for ulabel in piv.columns.levels[1]:
    p = piv['p'][ulabel]
    auroc = piv['auroc'][ulabel]
    
    # End of interval
    # Find the first bin after refbin that loses significance
    # If this is refbin itself, then no valid interval of significance
    # If it is 999, then sig the whole time after refbin    
    ls_mask = loss_of_signif(auroc[refbin:], p[refbin:], 
        auroc[refbin] > 0.5)
    end_interval = first_true(ls_mask, mungval=999)
    
    # Beginning of interval
    # Find the first bin before refbin that loses significance
    # If this is refbin itself, then no valid interval of significance
    # If it is 999, then it is significant for the entire interval preceding refbin    
    ls_mask = loss_of_signif(auroc[:refbin][::-1], p[:refbin][::-1], 
        auroc[refbin] > 0.5)
    begin_interval = first_true(ls_mask, mungval=999)

    # Store
    outrec_l.append((ulabel, begin_interval, end_interval))
intervals = pandas.DataFrame(outrec_l, columns=['ulabel', 't1', 't2'])
intervals = intervals.set_index('ulabel')

# Note where intervals are valid (ie, signif at refbin)
intervals['valid'] = True
intervals['valid'][(intervals.t1 == refbin) | (intervals.t2 == refbin)] = False

# Make the start inclusive, and deal with munged
intervals['t1'] = intervals['t1'].replace({999: -1})
intervals['t1'] += 1

# Leave the stop half-open but demung it
intervals['t2'] = intervals['t2'].replace({999: len(bincenters)})

# Save
intervals.save('intervals_of_signif')


