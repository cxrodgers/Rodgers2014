# Calculate passive only far from trials


import matplotlib.pyplot as plt
import numpy as np, pandas
import my.dataload, my, kkpandas, my.plot
from ns5_process import myutils
import kkpandas.kkrs

my.plot.publication_defaults()
my.plot.font_embed()

# Load hold results
hold_results = pandas.load('../3-1-hold/hold_results')

# Calculate spont rate in epochs far (>2s) from trial onsets
rec_l = []
for ulabel in hold_results.index:
    spikes = my.dataload.ulabel2spikes(ulabel)
    ospikes = spikes.copy()
    
    # Get trials info, all trials even munged
    session_name = kkpandas.kkrs.ulabel2session_name(ulabel)
    rs = my.dataload.session2rs(session_name)
    trials_info = kkpandas.io.load_trials_info(rs.full_path, drop_munged=False)
   
    # Stim onsets in seconds
    stim_onsets = trials_info.stim_onset.values
    
    # in rare casess this contains nan
    if np.any(np.isnan(stim_onsets)):
        print "warning: nan in %s" % ulabel
        stim_onsets = stim_onsets[~np.isnan(stim_onsets)]
    
    # Analyze data only between first and last trial
    t_start = stim_onsets.min()
    t_stop = stim_onsets.max()
    spikes = spikes[(spikes > t_start) & (spikes < t_stop)]
    
    # Keep track of how many bins we are counting spikes over
    # ie, not too far from stim onsets
    # 0.1 is just an arbitrary size limiting our resolution for this calc
    included_bins = np.arange(t_start, t_stop, .1)
    
    # Remove spikes too close to stim onsets
    for stim_onset in stim_onsets:
        spikes = spikes[
            (spikes - stim_onset > 2.0) |
            (spikes - stim_onset < -2.0)]
        included_bins = included_bins[
            (included_bins - stim_onset > 2.0) | 
            (included_bins - stim_onset < -2.0)]
    
    # Calculate included spikes and included time
    incl_spikes = len(spikes)
    incl_time = len(included_bins) * .1
    incl_ratio = incl_time / (t_stop - t_start)
    
    # Firing rate
    try:
        FR2 = incl_spikes / incl_time
    except ZeroDivisionError:
        FR2 = np.nan
    
    rec_l.append({'ulabel': ulabel, 'incl_ratio': incl_ratio,
        'passive_spont': FR2})
sponts_df = pandas.DataFrame.from_records(rec_l).set_index('ulabel')
sponts_df.save('sponts_df')