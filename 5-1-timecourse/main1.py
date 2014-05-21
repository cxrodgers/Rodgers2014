# Dump foldeds locked to:
#   cpoke_start
#   stim_onset
#   cpoke_stop
#   choice_made
# Also gets distributions locked to lots of events in the trials
# Slices trials based on a variety of criteria, eg after_nogo


import my, my.dataload
import numpy as np, pandas, kkpandas
from ns5_process import LBPB, bcontrol
import glob, os.path


# Load cached prev_rew_stop if it exists
if os.path.exists('cached'):
    cached = my.misc.pickle_load('cached')
else:
    cached = {}





def insert_events_and_times(ulabel, trials_info):
    """Inserts after_go, after_nogo, and times of those events"""
    # Get trials_info and define next_start as next_cpoke_start
    trials_info['next_start'] = trials_info['cpoke_start'].shift(-1)
    
    # Also define 'after_nogo_hit'
    trials_info['after_nogo_hit'] = 0
    after_nogo = 1 + np.asarray(
        my.pick(trials_info, outcome='hit', go_or_nogo='nogo'))
    after_nogo = filter(lambda bt: bt in trials_info.index, after_nogo)
    trials_info['after_nogo_hit'][after_nogo] = 1
    
    # Also define 'after_go_hit'
    trials_info['after_go_hit'] = 0
    after_go = 1 + np.asarray(
        my.pick(trials_info, outcome='hit', go_or_nogo='go'))
    after_go = filter(lambda bt: bt in trials_info.index, after_go)
    trials_info['after_go_hit'][after_go] = 1

    # If previous trial was NOGO hit, get cpoke_stop on that trial
    trials_info['prev_nogo_stop'] = trials_info['cpoke_stop'].shift(1)
    trials_info['prev_nogo_stop'][
        trials_info.after_nogo_hit != 1] = np.nan
    
    return trials_info


# Pick all hits
base_tpk = {
    'labels': LBPB.mixed_stimnames,
    'label_kwargs': [{'stim_name':s} for s in LBPB.mixed_stimnames],
    'nonrandom' : 0,
    'outcome' : 'hit',
    }

# Separately get distirbution of latencies to each of the following events,
# split across each category of trial
event_name_list = [
    'cpoke_start', 'stim_onset', 'cpoke_stop', 'choice_made', 'next_start',
    'prev_nogo_stop']

# Various trial splits
after_trial_type_l = ['after_nogo_hit', 'after_go_hit', 'after_all']

# Categories of trials
stim_groups = {
    'LB GO': ['le_lo_lc', 'le_hi_lc'],
    'LB NO': ['ri_lo_lc', 'ri_hi_lc'],
    'PB GO': ['ri_lo_pc', 'le_lo_pc'],
    'PB NO': ['ri_hi_pc', 'le_hi_pc'],
    }

# Folding kwargs
folding_kwargs = {'dstart': -3.0, 'dstop': 3.0}

# Which units to analyze
gets = my.dataload.getstarted()
units_to_analyze = gets['unit_db'][gets['unit_db'].include].index


# Iterate over locking events and after trial type
locking_event_l = ['stim_onset', 'cpoke_stop', 'cpoke_start']
for AFTER_TRIAL_TYPE in after_trial_type_l:
    # Optionally add in a keyword for AFTER_TRIAL_TYPE
    tpk = base_tpk.copy()
    if AFTER_TRIAL_TYPE == 'after_nogo_hit':
        tpk['after_nogo_hit'] = 1
    elif AFTER_TRIAL_TYPE == 'after_go_hit':
        tpk['after_go_hit'] = 1

    # Iterate over locking events
    for locking_event in locking_event_l:
        # Keep separate frames for folded results and the latency distributions
        rec_l = []
        rec_l2 = []

        # Iterate over units
        for ulabel in units_to_analyze:
            #~ my.printnow(ulabel)
            
            # Load trials info, then add some events and times
            trials_info = my.dataload.ulabel2trials_info(ulabel)
            trials_info = insert_events_and_times(ulabel, trials_info)

            # Store results so far
            rec = {'ulabel': ulabel, 'trials_info': trials_info}
    
            # Get folded, locked to locking event
            time_picker = kkpandas.timepickers.TrialsInfoTimePicker(trials_info)
            dfolded = my.dataload.ulabel2dfolded(ulabel, 
                trials_info=trials_info,
                time_picker=time_picker,
                time_picker_kwargs={'event_name': locking_event},
                trial_picker_kwargs=tpk,
                folding_kwargs=folding_kwargs,
                )
            rec['dfolded'] = dfolded
            
            # Store
            rec_l.append(rec)
            
            ## Now latency distributions
            # Iterate over stim groups and get latency distributions
            for stim_group_name, grouped_snames in stim_groups.items():
                # Pick the same trials that went into folded
                distrs_tpk = tpk.copy()
                distrs_tpk.pop('labels')
                distrs_tpk.pop('label_kwargs')
                subti = my.pick_rows(trials_info, stim_name=grouped_snames,
                    **distrs_tpk)
                
                # Iterate over event and get latencies for each
                for event_name in event_name_list:
                    # Store time deltas by trial for this event
                    distr = (subti[event_name] - subti[locking_event]).values
                    
                    rec_l2.append({'ulabel': ulabel, 
                        'event': event_name, 
                        'group': stim_group_name, 
                        'distr': distr})
            

        # DataFrame both
        resdf = pandas.DataFrame.from_records(rec_l).set_index('ulabel')
        times_distr = pandas.DataFrame.from_records(rec_l2).set_index(
            ['ulabel', 'group', 'event'])['distr']

        # Save, using locking event and after_nogo as keys
        suffix = '_lock_%s_%s' % (locking_event, AFTER_TRIAL_TYPE)
        resdf.save('dfoldeds' + suffix)
        times_distr.save('times_distrs' + suffix)
