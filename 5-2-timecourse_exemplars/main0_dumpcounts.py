# Estimate time course of effect in both blocks, with exemplars
# Unlike the other timecourse directory, here we only analyze by block,
# and not by go/nogo etc
# So begin by redumping the dfoldeds

import numpy as np, os.path
import pandas, itertools
from kkpandas import kkrs
import kkpandas
from ns5_process import myutils, LBPB
import my, my.dataload


# Which ulabels showed significant effect of hold
boot_results = pandas.load('../3-1-hold/hold_results')
sig_boot_results = boot_results[boot_results.p_adj < .05]

my.misc.no_warn_rs()

# All data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
session_db = gets['session_db']

# Which units to analyze
units_to_analyze = unit_db[unit_db.include].index

# Analysis to do: Pick a ton of time around random hits
folding_kwargs = {'dstart': -5, 'dstop': 5}   

# Initialize data structure to fill
ulabel2block2folded = myutils.nested_defaultdict()

# Iterate over significantly modulated units
outrec_l = []
for ulabel, inrec in sig_boot_results.iterrows():
    if ulabel not in units_to_analyze:
        print "warning: skipping", ulabel
    #~ myutils.printnow(ulabel)
    
    # All the various links
    session_name = kkpandas.kkrs.ulabel2session_name(ulabel)
    unum = kkpandas.kkrs.ulabel2unum(ulabel)
    kks = my.dataload.session2kk_server(session_name)
    rs = my.dataload.session2rs(session_name)
    
    # Load trials info
    trials_info = kkpandas.io.load_trials_info(rs.full_path)

    # pipeline
    dfolded = my.dataload.ulabel2dfolded(ulabel, 
        folding_kwargs=folding_kwargs,
        trial_picker_kwargs='random hits by block')

    # Store
    ulabel2block2folded[ulabel] = dfolded

# Dump
myutils.pickle_dump(ulabel2block2folded, 'ulabel2block2folded')