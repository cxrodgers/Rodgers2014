# First vidtrack script
# Put head angle and spike count data together, run ANOVAs, and dump

import vidtrack
import my, my.dataload, kkpandas, os, pandas, numpy as np
import scipy.stats, my.stats

# Video data location
root_dir = '../data/miscellaneous/vidtrack'
session_names = [
    'CR24A_121022_001_behaving',
    'CR20B_120602_001_behaving',
    'CR17B_110803_001_behaving']

# Check out units from labelled sessions showing hold effect
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')
counts_results = pandas.load('../3-1-hold/counts_results')
unit_db = unit_db.join(hold_results[['p_adj', 'mLB', 'mPB', 'dt']])
units_to_analyze = unit_db[
    unit_db.include & 
    (unit_db.p_adj < .05) &
    unit_db.session_name.isin(session_names)
    ].index


# Iterate over units
# For each one, run the statistical tests and dump the full dataframe
# of position and spike counts
test_res_d = {}
full_dfs_d = {}
for ulabel in units_to_analyze:
    my.printnow(ulabel)

    # Hold period counts
    counts_result = counts_results.ix[ulabel]
    hold_result = hold_results.ix[ulabel]

    # Parse into block and trial number and form spike_df
    labels = np.concatenate([
        counts_result['LB_trials'], counts_result['PB_trials']])
    counts = np.concatenate([
        counts_result['LB_counts'], counts_result['PB_counts']])
    blocks = np.concatenate([
        ['LB'] * len(counts_result['LB_trials']), 
        ['PB'] * len(counts_result['PB_trials'])])
    spike_df = pandas.DataFrame({'btrial': labels, 'counts': counts, 
        'block': blocks}).set_index('btrial')

    # Behavioral info on the session
    session_name = unit_db['session_name'][ulabel]
    rs = my.dataload.session2rs(session_name)
    trials_info = kkpandas.io.load_trials_info(rs.full_path)

    # Vid tracking object
    vts = vidtrack.Session(os.path.join(root_dir, session_name))

    # Convert to dataframe of head position
    location_df = vidtrack.vts_db2head_pos_df(vts, in_degrees=True)
    
    # Join and dropna (errors, short holds, etc)
    full_df = location_df.join(spike_df).dropna()
    
    # Do several tests
    test_res = {}
    
    # Test 2. ANOVA counts ~ block * angl
    aov_res = my.stats.anova(full_df, 'np.sqrt(counts) ~ block * angl', typ=2)
    test_res['aov_pblock'] = aov_res['pvals']['p_block']
    test_res['aov_pangl'] = aov_res['pvals']['p_angl']
    test_res['aov_pcross'] = aov_res['pvals']['p_block:angl']
    test_res['aov_essblock'] = aov_res['ess']['ess_block']
    test_res['aov_essangl'] = aov_res['ess']['ess_angl']
    test_res['aov_esscross'] = aov_res['ess']['ess_block:angl']    
    test_res['aov_pF'] = aov_res['lm'].f_pvalue
    
    # Store
    test_res_d[ulabel] = test_res
    full_dfs_d[ulabel] = full_df

# DataFrame and store results
test_res = pandas.DataFrame.from_records(test_res_d).T.sort()
full_data = pandas.concat(full_dfs_d).sort()
test_res.save('test_res')
full_data.save('full_data')