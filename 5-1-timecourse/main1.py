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


def cached_load_prev_rew_stop(ulabel, trials_info):
    """Caches prev_rew_stop for session; otherwise loads it from bcontrol
    
    ulabel : uses this to figure out which session, and checks the cache
    trials_info : uses this info, especially after_go_hit, to find
        prev_rew_stop
    
    Returns: series prev_rew_stop
        You can insert this into trials_info
    """
    # which session?
    session = kkpandas.kkrs.ulabel2session_name(ulabel)
    
    # return immediately if already in
    if session in cached:
        return cached[session]
    
    # get peh
    rs = my.dataload.session2rs(session)
    bcf_l = glob.glob(os.path.join(rs.full_path, '*.mat'))
    assert len(bcf_l) == 1
    bcl = bcontrol.Bcontrol_Loader(filename=bcf_l[0])
    bcl.load()
    bcld = bcontrol.dictify_mat_struct(bcl.data)
    peh = bcld['peh']
    
    # get sync
    b2n_sync = np.loadtxt(os.path.join(rs.full_path, 'SYNC_B2N'))   

    # get time of previous reward
    prev_rew_ser = get_prev_rew_stop_from_peh(peh, trials_info, b2n_sync)
    
    # put in cache
    cached[session] = prev_rew_ser
    
    # return
    return prev_rew_ser

def get_prev_rew_stop_from_peh(peh, trials_info, b2n_sync):
    """Go through peh and trials_info, return time of previous reward
    
    trials_info : after_go_hit must be set
    """
    # If previous trial was GO hit, get last outpoke on this trial
    rewlast_d = {}
    for ti_num in my.pick(trials_info, after_go_hit=1):
        # ti_num is the btrial number which starts at 1
        # we subtract one and index into peh
        # in certain circumstances I think TI can have one more than peh
        pokes_trial = peh[ti_num - 1]['pokes']
    
        # check the port that was rewarded on the previous trial
        if ti_num - 1 not in trials_info.index:
            # This can happen if previous trial was munged
            continue
        port = trials_info['correct_side'][ti_num - 1][0].upper()
        
        # Get the pokes from the current trial
        if pokes_trial[port].size == 0:
            # no pokes
            continue
        if pokes_trial[port].ndim == 1:
            # overflattened
            outpokes = np.array([pokes_trial[port][1]])
        else:
            outpokes = pokes_trial[port][:, 1]
        outpokes = outpokes[~np.isnan(outpokes)]
        
        # convert to neural time base
        outpokes_nbase = np.polyval(b2n_sync, outpokes)
        
        # get the last outpoke before center center poke
        outpokes_nbase = outpokes_nbase[
            outpokes_nbase < trials_info['cpoke_start'][ti_num]]
        if len(outpokes_nbase) == 0:
            continue
        else:
            rewlast_d[ti_num] = outpokes_nbase.max()
    res = pandas.Series(rewlast_d)
    
    # Check that prev_rew_stop was always after previous choice_made
    assert np.all(res > trials_info['choice_made'].shift(1)[res.index])
    
    return res


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
    
    # If previous trial was GO hit, get previous reward stop
    trials_info['prev_rew_stop'] = cached_load_prev_rew_stop(
        ulabel, trials_info)
    
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
    'prev_rew_stop', 'prev_nogo_stop']

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
            my.printnow(ulabel)
            
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
