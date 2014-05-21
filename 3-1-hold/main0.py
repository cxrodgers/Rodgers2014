# Dump spike counts from hold period


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
rec_l = []
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
        trial_picker_kwargs='random hits by block')

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
    block2counts = myutils.map_d(lambda f: f.apply(len), block2folded)
    
    # Get the counts by block
    counts_by_block = my.misc.parse_by_block(
        lb_counts=block2counts['LB'], pb_counts=block2counts['PB'],
        lb_trial_numbers=block2folded['LB'].labels,
        pb_trial_numbers=block2folded['PB'].labels,
        session_name=session_name)

    # Store
    res = {
        'LB_counts': block2counts['LB'], 'PB_counts': block2counts['PB'],
        'counts_by_block': counts_by_block, 'ulabel': ulabel, 'dt': dt,
        'LB_trials': block2folded['LB'].labels, 
        'PB_trials': block2folded['PB'].labels}
    rec_l.append(res)

# Dump
res_df = pandas.DataFrame.from_records(rec_l).set_index('ulabel')
res_df.save('counts_results')
