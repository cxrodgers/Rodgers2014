# Correlate RTs with neural activity
# Extract RT and MT for each trial, separated by block and correct/incorrect
# Correlate firing rate with each measure for each neuron and look at
# distributions of r and p across neurons.

import my, my.dataload, pandas, scipy.stats
import kkpandas, kkpandas.kkrs
import numpy as np

my.misc.no_warn_rs()

gets = my.dataload.getstarted()
unit_db = gets['unit_db']
unit_db['hold_dur'] = .050
unit_db['hold_dur'][unit_db.ratname.isin(['CR24A', 'CR20B', 'CR21A'])] = .25
units_to_analyze = unit_db[unit_db.include]

# How to load all trials without splitting
trial_picker_kwargs = {'labels': ['all'], 'label_kwargs': [{}]}

# What kind of trials
# Each is analyzed separately
include_l = [
    'all hits',
    'go hits', # This is the only set for which MT is valid!
    'nogo hits',
    ]


# Extract TRIALS_INFO for each session
session2ti = {}
for session_name in units_to_analyze.session_name.unique():
    # Get trials info
    rs = my.dataload.session2rs(session_name)
    trials_info = kkpandas.io.load_trials_info(rs.full_path)
    
    # Calculate RT and MT
    trials_info['RT'] = trials_info['cpoke_stop'] - trials_info['stim_onset']
    trials_info['MT'] = trials_info['choice_made'] - trials_info['stim_onset']
    
    # Error check
    assert np.all(trials_info['RT'] > 0)
    assert np.all(trials_info['MT'] > 0)
    
    # Store
    session2ti[session_name] = trials_info

# Correlate with firing rates
# Should skip the ones with no prefblock!
rec_l, datarec_l = [], []
for ulabel in units_to_analyze.index:
    #~ print ulabel
    # Get TI
    session_name = kkpandas.kkrs.ulabel2session_name(ulabel)
    trials_info = session2ti[session_name]
    
    # Get firing rate on each trial
    folded = my.dataload.ulabel2dfolded(ulabel, 
        trial_picker_kwargs=trial_picker_kwargs,
        folding_kwargs={'dstart': -units_to_analyze['hold_dur'][ulabel], 
        'dstop': 0.})['all']
    counts = pandas.Series(folded.apply(len), index=folded.labels)
    
    # Correlate RT with counts, separately by block
    for block in [2, 4]:
        # Analyze each trial set separately
        for include in include_l:
            # Subslice the trials
            if include == 'go hits':
                subti = my.pick_rows(trials_info, block=block, go_or_nogo='go',
                    outcome='hit', nonrandom=0)
            elif include == 'nogo hits':
                subti = my.pick_rows(trials_info, block=block, go_or_nogo='nogo',
                    outcome='hit', nonrandom=0)
            elif include == 'all hits':
                subti = my.pick_rows(trials_info, block=block,
                    outcome='hit', nonrandom=0)
            else:
                1/0
            
            # Get the RT and MT for each included trial
            v1a, v1b = subti['RT'].values, subti['MT'].values
            
            # Squarerooted counts for each included trial
            v2 = np.sqrt(counts[subti.index].values)
            
            # Store data
            datarec = {'ulabel': ulabel, 'block': 'LB' if block == 2 else 'PB', 
                'RT': v1a, 'MT': v1b, 'sqrt_counts': v2,
                'include': include}
            datarec_l.append(datarec)
            
            # Correlate
            ma, ya, ra, pa, sea = scipy.stats.linregress(v2, v1a)
            mb, yb, rb, pb, seb = scipy.stats.linregress(v2, v1b)
            
            # Store 
            rec = {'ulabel': ulabel, 'block': 'LB' if block == 2 else 'PB', 
                'rRT': ra, 'pRT': pa, 'rMT': rb, 'pMT': pb,
                'include': include}
            rec_l.append(rec)

# Dataframe and dump
res = pandas.DataFrame.from_records(rec_l).set_index(
    ['include', 'ulabel', 'block'])
datares = pandas.DataFrame.from_records(datarec_l).set_index(
    ['include', 'ulabel', 'block'])
res.save('res')
datares.save('datares')    
