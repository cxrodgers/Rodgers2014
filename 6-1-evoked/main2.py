# This script runs the decoders on dataframes from main1
# Run the decoders on the following:
# 1) MSUA (ensemble) decoding of target sounds
# 2) Single-unit decoding of cue sounds


import pandas, numpy as np, my, kkpandas, kkpandas.kkrs
from ns5_process import LBPB, myutils
import scipy.stats
import sklearn
import sklearn.linear_model
import decoders



# Load data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
evoked_resps = pandas.load('evoked_resps')
ulabel2tall_cue = myutils.pickle_load('ulabel2tall_cue')

# Load these to decode target from session
session2tall_target = {}
for ensem in ['A1', 'PFC']:
    session2tall_target[ensem] = myutils.pickle_load('%s_session2tall_target' % ensem)

# These are the columns that are not responses in session2tall_target
non_ulabel_cols = ['block', 'sL', 'sP']

# some consts
consts = {'le': 0, 'ri': 1, 'lo': 0, 'hi': 1, 'lc': 0, 'pc': 1}




# Decode cue from individual neurons
# This is for a correlation plot between hold resp and target tuning
rec_l = []
for ulabel, tall in ulabel2tall_cue.items():
    for block in ['lc', 'pc']:
        # Parse out just the trials of interest
        sub_tall = tall[tall.block == consts[block]]        

        # Decode the cue sound
        rec = {'ulabel': ulabel, 'block': block}

        # Fit a linear model and deparse the output
        model_res = decoders.logreg_score2(sub_tall, output='sound',
            class_weight='auto')
        rec['flat'] = model_res['flat_score']
        rec['cateq'] = model_res['equalized_score']
        rec['n_spikes'] = sub_tall['count'].sum()
        rec['n_trials'] = len(sub_tall)
        rec_l.append(rec)

decode_results_sua_cue = pandas.DataFrame.from_records(rec_l)

# save
decode_results_sua_cue.save('decode_results_sua_cue')



# Decode target from ensembles
decode_results_msua_target = {}
for ensem in ['A1', 'PFC']:
    rec_l = []
    for session_name, tall in session2tall_target[ensem].items():
        # Now decode
        for var in ['sL', 'sP']:
            # Separately analyze each block
            for block in ['lc', 'pc']:
                rec = {'session': session_name, 'block': block, 'var': var}
                
                # Parse out just the trials of interest
                sub_tall = tall[tall.block == consts[block]]
                sub_tall_counts_only = sub_tall.drop(non_ulabel_cols, axis=1)
                assert sub_tall_counts_only.shape[1] > 0

                #CV
                # Fit a linear model and deparse the output
                model_res = decoders.logreg_score2(sub_tall,
                    input=sub_tall_counts_only, output=var,
                    class_weight='auto', cross_validate=True,
                    C=1.0)
                rec['cateq_CV'] = model_res['equalized_score']
                

                # Normal
                model_res = decoders.logreg_score2(sub_tall,
                    input=sub_tall_counts_only, output=var,
                    class_weight='auto', cross_validate=False)
                rec['cateq'] = model_res['equalized_score']

                
                rec['n_spikes'] = sub_tall_counts_only.sum().sum()
                rec['n_units'] = sub_tall_counts_only.shape[1]
                rec_l.append(rec)
    decode_results_msua_target[ensem] = pandas.DataFrame.from_records(rec_l)

# Save
for ensem in ['A1', 'PFC']:
    decode_results_msua_target[ensem].save('decode_results_msua_target_%s' % ensem)
