# Test the counts from main0

import numpy as np
import my
from ns5_process import myutils, LBPB
import scipy.stats, os.path, pandas, kkpandas, kkpandas.kkrs
import matplotlib.mlab as mlab
import my, my.dataload
import matplotlib.pyplot as plt, my.plot

# Load data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']

# Load the results of the counts dump
counts_results = pandas.load('counts_results')

# How many spikes to include
# But this was already used in the prefilter for drift, so it doesn't
# make a difference here
MIN_TOTAL_SPIKE_COUNT = 20

# Iterate over ulabels and form DataFrame with counts by block
rec_l = []
for ulabel, row in counts_results.iterrows():
    # Test (reference group is second)
    utest_res = my.stats.r_utest(row['PB_counts'], row['LB_counts'])
    utest_res['pmw'] = utest_res.pop('p')
    
    # Bootstrap CIs (reference group is first)
    # This produces a bunch of warning about p-values, but it's no problem
    # because we're not using the bootstrapped p-value
    db = my.bootstrap.DiffBootstrapper(row['LB_counts'], row['PB_counts'],
        n_boots=1000)
    db.execute()
    
    # Parse bootstrap results into record
    dbres = {
        'CI_LB_l': db.CI_1[0], 'CI_LB_h': db.CI_1[1],
        'CI_PB_l': db.CI_2[0], 'CI_PB_h': db.CI_2[1]}

    # Other info
    utest_res.update(dbres)
    utest_res['ulabel'] = ulabel
    utest_res['n_spikes'] = np.sum(row['LB_counts']) + np.sum(row['PB_counts'])
    utest_res['mLB'] = row['LB_counts'].mean()
    utest_res['mPB'] = row['PB_counts'].mean()
    
    # Store
    rec_l.append(utest_res)
test_results = pandas.DataFrame.from_records(rec_l).set_index('ulabel')

assert np.all(test_results.n_spikes > MIN_TOTAL_SPIKE_COUNT)

# Adjust the p-value
test_results['p_adj'] = my.stats.r_adj_pval(test_results.pmw)

# Join counts_results, test_results, unit_db
hold_results = test_results.join(
    counts_results[['LB_counts', 'PB_counts', 'counts_by_block', 'dt']]).join(
    unit_db[['region', 'ratname', 'session_name', 'unum', 'group']])

# Save
hold_results.save('hold_results')

