# This script writes out a dataframe for each unit, with the block,
# spike count, and localization soudn (sL) and pd sound (sP) for every trial
# in evoked_resps from main0
# Next script runs the decoderse on these values

import pandas, numpy as np, my, kkpandas, kkpandas.kkrs, my.dataload
from ns5_process import LBPB, myutils

def parse_ulabel_to_tall_for_target(evoked_resp):
    """Parse each evoked resp into format for decoding target"""
    stimnames = LBPB.mixed_stimnames

    # Iterate over stimnames
    tall_l = []
    for stimname in LBPB.mixed_stimnames:
        counts = evoked_resp[stimname]
        
        # Use the rule for target sounds
        s1, s2, block = stimname.split('_')
        s1, s2, block = consts[s1], consts[s2], consts[block]
        
        # Concatenate results for this stimname
        tall_l += [{'sL': s1, 'sP': s2, 'block': block, 'count': count}
            for count in counts]
    tall_df = pandas.DataFrame.from_records(tall_l)
    return tall_df

def parse_ulabel_to_tall_for_cue(evoked_resp):
    """Parse each evoked resp into format for decoding cue"""
    stimnames = ['lo_pc_go', 'hi_pc_no', 'le_lc_go', 'ri_lc_no']

    # Iterate over stimnames
    tall_l = []
    for stimname in stimnames:
        counts = evoked_resp[stimname]
        
        # Use the rule for cue sounds
        s1, block, action = stimname.split('_')
        s1, block = consts[s1], consts[block]
        
        # Concatenate results for this stimname
        tall_l += [{'sound': s1, 'block': block, 'count': count}
            for count in counts]
    tall_df = pandas.DataFrame.from_records(tall_l)
    return tall_df

# Load data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
evoked_resps = pandas.load('evoked_resps')

# some consts
consts = {'le': 0, 'ri': 1, 'lo': 0, 'hi': 1, 'lc': 0, 'pc': 1}

# Create decodable dataframes for target and cue, from individual units
ulabel2tall_target = {}
ulabel2tall_cue = {}
for ulabel, evoked_resp in evoked_resps.iterrows():
    ulabel2tall_target[ulabel] = parse_ulabel_to_tall_for_target(evoked_resp)
    ulabel2tall_cue[ulabel] = parse_ulabel_to_tall_for_cue(evoked_resp)

# Create decodable dataframes for target and cue, from ensembles
# Separately for A1 ensembles, PFC ensembles, and joint
session2tall_target = {}
for ensem in ['A1', 'PFC']:
    # Iterate over sessions
    session2tall_target[ensem] = {}
    for session_name, ulabels in unit_db.groupby('session_name').groups.items():
        
        # Add each ulabel
        counts_df_l = []
        noncounts_df_l = []
        for ulabel in ulabels:
            # Include only those in evoked_resps and in the right ensemble
            if ulabel not in evoked_resps.index:
                continue 
            if evoked_resps['region'][ulabel] != ensem:
                continue
            
            # Get the tall for this ulabel
            utall = ulabel2tall_target[ulabel]
            
            # Separately parse out the counts for this unit, and the metadata
            ucount = utall['count'].copy()
            ucount.name = ulabel
            counts_df_l.append(ucount)
            noncounts_df_l.append(utall.drop('count', axis=1))
        
        # If no data, then continue to next session
        if len(counts_df_l) == 0:
            continue
        
        # Ensure the metadata is same for all units
        for test_df in noncounts_df_l[1:]:
            assert np.all(test_df == noncounts_df_l[0])
        metadata = noncounts_df_l[0].copy()
        
        # Put the counts together and join with metadata
        ucounts_df = pandas.concat(counts_df_l, axis=1, verify_integrity=True)
        session_tall = pandas.concat([ucounts_df, metadata], axis=1,
            verify_integrity=True)
        
        # Store
        session2tall_target[ensem][session_name] = session_tall



# Save
my.misc.pickle_dump(ulabel2tall_target, 'ulabel2tall_target')
my.misc.pickle_dump(ulabel2tall_cue, 'ulabel2tall_cue')
for ensem in ['A1', 'PFC']:
    my.misc.pickle_dump(session2tall_target[ensem], 
        '%s_session2tall_target' % ensem)
