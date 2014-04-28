# Example distributions of spike counts v head angle
import vidtrack
import my, my.dataload, kkpandas, os, pandas, numpy as np
import scipy.stats, my.stats
import matplotlib.pyplot as plt
from matplotlib import mlab

# Video data location
root_dir = '../data/miscellaneous/vidtrack'
session_names = [
    'CR24A_121022_001_behaving',
    'CR20B_120602_001_behaving',
    'CR17B_110803_001_behaving',]

# Check out units from labelled sessions showing hold effect
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')
counts_results = pandas.load('../3-1-hold/counts_results')
unit_db = unit_db.join(hold_results[['p_adj', 'mLB', 'mPB', 'dt']])

# Test results
test_res = pandas.load('test_res').sort('aov_pangl')

# Exemplars
exemplars = [
    'CR24A_121022_001_behaving-116',
    'CR24A_121022_001_behaving-511',
    ]

plots_to_make = [
    'exemplars',
    ]

def make_plot(ax, ulabel):
    """Convenience function to scatter counts by angle with trend lines"""
    # Hold period counts
    counts_result = counts_results.ix[ulabel]
    hold_result = hold_results.ix[ulabel]
    
    # Test results
    test_result = test_res.ix[ulabel]

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

    # Vid tracking object
    vts = vidtrack.Session(os.path.join(root_dir, session_name))

    # Convert to dataframe of head position
    location_df = vidtrack.vts_db2head_pos_df(vts, in_degrees=True)
    
    # Join and dropna (errors, short holds, etc)
    full_df = location_df.join(spike_df).dropna()
    
    # Subtract off mean head angle
    full_df['angl'] = full_df['angl'] - full_df['angl'].mean()
    
    # Plot
    block2color = {'LB': 'b', 'PB': 'r'}

    # Fit to all data
    m, b, r, p, se = scipy.stats.linregress(full_df['angl'], 
        np.sqrt(full_df['counts']))
    sdump.append("all: %0.3f" % p)
    xvals = mlab.prctile(full_df.angl, (5, 95))
    ax.plot(xvals, np.polyval((m, b), xvals), '-', color='k', lw=3)

    for block in ['LB', 'PB']:
        subdf = my.pick_rows(full_df, block=block)
        
        # Plot the individual points
        ax.plot(subdf.angl, np.sqrt(subdf.counts), ',', mew=.5, mec=block2color[block])
        ax.plot(subdf.angl, np.sqrt(subdf.counts), ls='none', marker='s', 
            ms=1, mew=.5, mec=block2color[block])
        
        # Plot the linfits
        m, b, r, p, se = scipy.stats.linregress(subdf['angl'], 
            np.sqrt(subdf['counts']))
        sdump.append("block %s: %0.3f" % (block, p))
        xvals = mlab.prctile(subdf.angl, (15, 85))
        ax.plot(xvals, np.polyval((m, b), xvals), '-', 
            color=block2color[block], lw=3)
    
    mmm = np.sqrt(full_df.counts.max())
    ax.set_ylim((-.5, mmm + .5))
    ax.set_yticks(list(range(int(mmm) + 1)))
    ax.set_xticks((-45, -30, -15, 0, 15, 30, 45))
    
    #~ ax.set_ylim((-.5, np.sqrt(full_df.counts.max()) + 0.5))    
    ax.set_xlabel('head angle (degrees)')
    ax.set_ylabel('sqrt(trial spike count)')
    my.plot.despine(ax)


sdump = []
if 'exemplars' in plots_to_make:
    # This also prints out p-values on the full fit, as well as within
    # each block
    
    for exemplar in exemplars:
        sdump.append(exemplar)
        f, ax = plt.subplots(figsize=(3, 3))
        make_plot(ax, exemplar)
        #~ ax.set_title(exemplar)
        f.savefig('exemplar-%s.svg' % exemplar)

sdump_s = "\n".join(sdump)
with file('stat__head_angle_linfits_vs_FR', 'w') as fi:
    fi.write(sdump_s)
print sdump_s

plt.show()
