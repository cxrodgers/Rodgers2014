# Sub-sample counts with overlapping head angle
# All this script does is choose the trials with same mean angle

import pandas
import numpy as np
import matplotlib.pyplot as plt
import my
import kkpandas.kkrs


# angles and counts
full_data = pandas.load('full_data')

# Choose the sub-indexed trials for each session
session2subindex = {}
for ulabel in full_data.index.levels[0]:
    # Skip if we already did this session
    session = kkpandas.kkrs.ulabel2session_name(ulabel)
    if session in session2subindex:
        continue
    
    # each block, sorted by angle
    fdu = full_data.ix[ulabel]
    fdu_LB = my.pick_rows(fdu, block='LB').sort('angl')
    fdu_PB = my.pick_rows(fdu, block='PB').sort('angl')

    # find how many we can include
    # LB is always less than PB
    for n_include in range(1, len(fdu)):
        # Take the highest LB values
        sub_LB = fdu_LB.angl.values[-n_include:]
        
        # And the lowest PB values
        sub_PB = fdu_PB.angl.values[:n_include]
        
        # Break when the mean PB > mean LB
        assert len(sub_LB) > 0
        assert len(sub_PB) > 0
        if sub_LB.mean() < sub_PB.mean():
            break

    # This should never happen
    if n_include >= len(fdu) / 2:
        1/0

    # Take one less so that mean PB is definitely less than mean LB
    n_include = n_include - 1

    # Extract indices
    sub_idxs_LB = np.asarray(fdu_LB.index)[-n_include:]
    sub_idxs_PB = np.asarray(fdu_PB.index)[:n_include]
    
    # We should have inverted the typical order
    assert \
        fdu.angl[sub_idxs_LB].mean() > \
        fdu.angl[sub_idxs_PB].mean()

    # Store
    session2subindex[session] = {
        'LB': sub_idxs_LB,
        'PB': sub_idxs_PB}

# Save results
my.misc.pickle_dump(session2subindex, 'session2subindex')
