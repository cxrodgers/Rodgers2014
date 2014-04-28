# Compare hold results on posture-equalized and other trials
# Also finds the shuffled distribution of expected rule encoding under
# the hypothesis that no block difference exists in the posture-equalized
# trials

import pandas
import numpy as np
import matplotlib.pyplot as plt
import my
import kkpandas.kkrs
import scipy.stats
import random
from matplotlib import mlab

my.plot.publication_defaults()
my.plot.font_embed()


# This parameter determines how many times to shuffle the labels
# to estimate the shuffled distribution
# This is needed to estimate distribution of population rule encoding
# under permutation test.
# It is not necessary to simply view the difference between posture-equalized
# and other trials.
# I used a value of 10000 for the paper
# Can get reasonable results with 1000
N_SHUFFLES = 10000


# hold results
hold_results = pandas.load('../3-1-hold/hold_results')

# angles and counts
full_data = pandas.load('full_data')

# subsampled indexes
session2subindex = my.misc.pickle_load('session2subindex')

# Iterate over ulabels and compare original and subsampled hold effect    
#res = {}
rec_l = []
for ulabel in full_data.index.levels[0]:
    # Get subsampled indexes
    session = kkpandas.kkrs.ulabel2session_name(ulabel)
    block2subindex = session2subindex[session]

    # Get this ulabel
    fdu = full_data.ix[ulabel]

    # Verify matches hold data
    mLB2 = my.pick_rows(fdu, block='LB')['counts'].mean()
    mPB2 = my.pick_rows(fdu, block='PB')['counts'].mean()
    assert np.allclose(
        hold_results.ix[ulabel][['mLB', 'mPB']].values.astype(np.float), 
        [mLB2, mPB2])
    
    # Index into full data
    sub_LB = fdu.ix[block2subindex['LB']]
    sub_PB = fdu.ix[block2subindex['PB']]
    
    # Check that posture equalization worked: angle should be very close
    # to equal, but slightly greater in PB
    pe_blockdiff = sub_LB.angl.mean() - sub_PB.angl.mean()
    assert pe_blockdiff > 0 and pe_blockdiff < 1 # degrees
    
    # The NON posture-equalized trials
    nsub = fdu.ix[[idx for idx in fdu.index if
        idx not in block2subindex['LB'] and
        idx not in block2subindex['PB']]]
    nsub_LB = my.pick_rows(nsub, block='LB')
    nsub_PB = my.pick_rows(nsub, block='PB')
    
    # Extract hold effect on subsampled
    sub_mLB = sub_LB['counts'].mean()
    sub_mPB = sub_PB['counts'].mean()
    
    # And on nonsubbed
    nsub_mLB = nsub_LB['counts'].mean()
    nsub_mPB = nsub_PB['counts'].mean()
    
    # Check that the subbed and nonsubbed add up to the original
    assert \
        sub_LB['counts'].sum() + nsub_LB['counts'].sum() == \
        my.pick_rows(fdu, block='LB')['counts'].sum()
    assert \
        sub_PB['counts'].sum() + nsub_PB['counts'].sum() == \
        my.pick_rows(fdu, block='PB')['counts'].sum()
    
    # Shuffle
    # Choose N_SHUFFLES random labelling schemes, average each one
    faked_sub_mLB_l = []
    faked_sub_mPB_l = []
    for nshuf in range(N_SHUFFLES):
        # Choose faked indexes
        faked_idxs = np.random.permutation(
            list(block2subindex['LB']) + 
            list(block2subindex['PB']))
        faked_idxs_LB = faked_idxs[:len(sub_LB)]
        faked_idxs_PB = faked_idxs[len(sub_LB):]
            
        # Faked means
        faked_sub_mLB_l.append(fdu.ix[faked_idxs_LB]['counts'].mean())
        faked_sub_mPB_l.append(fdu.ix[faked_idxs_PB]['counts'].mean())
    
    # Store
    rec_l.append({
        'ulabel': ulabel,
        'sub_mLB': sub_mLB, 'sub_mPB': sub_mPB, # subbed 
        'nsub_mLB': nsub_mLB, 'nsub_mPB': nsub_mPB, # non-subbed
        'faked_LB': np.array(faked_sub_mLB_l), # all faked sub LB 
        'faked_PB': np.array(faked_sub_mPB_l), # all faked sub PB
        })
sub_res = pandas.DataFrame.from_records(rec_l).set_index('ulabel')


# Join sub_res on the original
sub_res = sub_res.join(hold_results[['mLB', 'mPB', 'dt', 'region', 'p_adj']])
sub_res['prefblock'] = 'LB'
sub_res['prefblock'][sub_res.mPB > sub_res.mLB] = 'PB'

# Ratios of all, posture-equalized, and non-posture-equalized trials
sub_res['rdiff'] = (sub_res['mPB'] / sub_res['mLB'])
sub_res['sub_rdiff'] = (sub_res['sub_mPB'] / sub_res['sub_mLB'])
sub_res['nsub_rdiff'] = (sub_res['nsub_mPB'] / sub_res['nsub_mLB'])

# Ratio of the rule-encoding on each shuffle
sub_res['faked_rdiff'] = sub_res['faked_PB'] / sub_res['faked_LB']


# Save
sub_res.save('posture_equalization_results')