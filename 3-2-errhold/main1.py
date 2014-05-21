# Dump counts in hold period, now including error trials

import numpy as np, os.path
import pandas, itertools
import kkpandas, kkpandas.kkrs
from ns5_process import myutils, LBPB
import my, my.dataload

# Hooks to data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']

# How long the hold should be, and how long to count over
ratname2hold_time = {'CR12B': 50, 'CR17B': 50, 'YT6A': 50,
    'CR24A': 250, 'CR20B': 250, 'CR21A': 250}

# Which units to analyze
units_to_analyze = unit_db[unit_db.include].index

# Iterate over units
ulabel2snameout2counts = {}
for ulabel in units_to_analyze:
    #~ myutils.printnow(ulabel)
    
    # Behavioral info on the session
    session_name = unit_db['session_name'][ulabel]
    rs = my.dataload.session2rs(session_name)
    trials_info = kkpandas.io.load_trials_info(rs.full_path)
    
    # Check hold durations self-consistent
    ratname = unit_db['ratname'][ulabel]
    hold_time = ratname2hold_time[ratname]
    hold_durations = trials_info.stim_onset - trials_info.cpoke_start
    if hold_time == 250: 
        assert np.all(hold_durations >= .249)
        assert np.all(hold_durations <= .351)
    else:
        assert np.all(hold_durations > 0)
        assert np.all(hold_durations <= .101)
    
    # Use this to set the folding kwargs
    dt = hold_time / 1000.
    folding_kwargs = {'dstart': -dt, 'dstop': 0.}
    
    # Get the folded results
    block2folded = my.dataload.ulabel2dfolded(ulabel, 
        folding_kwargs=folding_kwargs,
        trial_picker_kwargs='by outcome')

    # Slice out trials where hold > 50
    keep_trials = hold_durations[hold_durations > .050].index
    if len(keep_trials) < len(hold_durations):
        assert hold_time == 50
        
        # Some trials dropped, iterate over foldeds in dict and slice
        block2folded_longholds = {}
        for block, folded in block2folded.items():
            block2folded_longholds[block] = folded.get_slice(
                map(lambda tn: tn in keep_trials, folded.labels))
        
        block2folded = block2folded_longholds
    else:
        # No trials dropped
        assert hold_time == 250
  
    # Count in the designated window
    snameout2counts = myutils.map_d(lambda f: f.apply(len), block2folded)
    
    # Store
    ulabel2snameout2counts[ulabel] = snameout2counts

# Dump
myutils.pickle_dump(ulabel2snameout2counts, 'ulabel2snameout2counts')