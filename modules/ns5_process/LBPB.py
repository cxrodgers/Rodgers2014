"""Various ways of dealing with stimulus names in LBPB task"""
import numpy as np

short2long = {'lelo': 'LEFT+LOW', 'rilo': 'RIGHT+LOW', 'lehi': 'LEFT+HIGH',
    'rihi': 'RIGHT+HIGH'}

sns = list(range(1, 13))
stimnames = [
    'lo_pc_go', 'hi_pc_no', 'le_lc_go', 'ri_lc_no',
    'le_hi_pc', 'ri_hi_pc', 'le_lo_pc', 'ri_lo_pc',
    'le_hi_lc', 'ri_hi_lc', 'le_lo_lc', 'ri_lo_lc']
sn2name = {k: v for k, v in zip(sns, stimnames)}

mixed_stimnames = stimnames[4:]
mixed_sns = sns[4:]
mixed_sn2name = {k: v for k, v in zip(mixed_sns, mixed_stimnames)}

sound_block_tuple = (
    ('lehi', 'PB'), ('rihi', 'PB'), ('lelo', 'PB'), ('rilo', 'PB'),
    ('lehi', 'LB'), ('rihi', 'LB'), ('lelo', 'LB'), ('rilo', 'LB'))
block_sound_tuple = tuple(t[::-1] for t in sound_block_tuple)

stimname2sound_block_tuple = {sn: t 
    for sn, t in zip(mixed_stimnames, sound_block_tuple)}
stimname2block_sound_tuple = {sn: t 
    for sn, t in zip(mixed_stimnames, block_sound_tuple)}


sn2sound_block_tuple = {sn: t 
    for sn, t in zip(mixed_sns, sound_block_tuple)}
sn2block_sound_tuple = {sn: t 
    for sn, t in zip(mixed_sns, block_sound_tuple)}

mixed_stimnames_noblock = ('le_hi', 'ri_hi', 'le_lo', 'ri_lo')


# Functional definitions of various trial events
def start_cpoke(trial_events):
    """Returns the initiation time of the triggering C-poke in trial_events"""
    all_cpoke_starts = trial_events[trial_events.event == 'hold_center_in']
    return all_cpoke_starts.time.values[-1]

def end_cpoke(trial_events):
    """Returns the end time of the trigger C-poke in trial_events
    
    This will return a maximum of 250ms since it doesn't detect longer
    than the stimulus itself
    """
    subevents1 = trial_events[trial_events.event == 'hold_center2_out']
    subevents2 = trial_events[trial_events.event == 'possible_short_cpoke_out']

    if len(subevents1) != 0:
        # Might be multiple hold_center2_out, depending on forgiveness window
        #assert len(subevents1) == 1
        return subevents1.time.values[-1]    
    else:
        # No hold_center2_out
        # Use possible short cpoke out
        assert len(subevents2) == 1
        return subevents2.time.values[-1]    

def stim_onset(trial_events, strict=True):
    """Returns the stimulus onset time in trial_events"""
    subevents = trial_events[trial_events.event == 'play_stimulus_in']
    if len(subevents) != 1:
        print 'multiple stimulus onsets detected'
        if strict:
            return np.nan
    return subevents.time.values[0]

def choice_made(trial_events, strict=True):
    """Return end of choosing_side, which can be hit or miss on GO or NOGO"""
    subevents = trial_events[trial_events.event == 'choosing_side_out']
    if len(subevents) != 1:
        print 'multiple choices detected'
        if strict:
            return np.nan
    return subevents.time.values[0]  


# Canonical block ordering
def block_trace_is_munged(block_trace):
    # These are what the trialnumbers should go to after modding by 160
    blocknum2acceptable_mods = {
        1 : list(range(1, 21)),
        2 : list(range(21, 81)),
        3 : list(range(81, 101)),        
        4 : [0] + list(range(101, 160)),}
    
    # check each block
    munged = False
    for blocknum, acceptable_mods in blocknum2acceptable_mods.items():
        block_trace_slc = block_trace[block_trace == blocknum]
        achieved_mods = np.mod(block_trace_slc.index, 160)
        if not np.all(np.in1d(achieved_mods, acceptable_mods)):
            munged = True
    
    return munged