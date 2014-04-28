# Simple question: when do the LB and PB psths diverge?
# U-test each bin, mark the first that loses significance

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
#bins = np.linspace(-5, 5, int(round(10/.050)))
#~ bins = np.arange(-5, 5, .050)
bins = np.loadtxt('bins')


# We know the effect exists from -50ms to 0ms
# This is between refbin - 1 and refbin
refbin = np.argmin(np.abs(bins - 0.))

# Bin each
out_l = []
for ulabel, block2folded in ulabel2block2folded.items():
    myutils.printnow(ulabel)
    
    # Bin trials separately
    LB_binned = kkpandas.Binned.from_folded_by_trial(block2folded['LB'], 
        bins=bins, meth=gs.smooth)
    PB_binned = kkpandas.Binned.from_folded_by_trial(block2folded['PB'], 
        bins=bins, meth=gs.smooth)

    # Test for differences in each bin
    for nbin in range(len(LB_binned.counts)):
        # Do the test
        LB_counts = LB_binned.counts.ix[nbin]
        PB_counts = PB_binned.counts.ix[nbin]
        utest_res = my.stats.r_utest(PB_counts, LB_counts, fix_float=10000)
        
        # Store
        utest_res['ulabel'] = ulabel
        utest_res['bin'] = nbin
        out_l.append(utest_res)

# DataFrame it and pivot on the values we need
outdf = pandas.DataFrame.from_records(out_l)
piv = pandas.pivot_table(outdf, rows=['bin'], cols=['ulabel'], 
    values=['auroc', 'p'])

# save the results
piv.save('block_effect')
