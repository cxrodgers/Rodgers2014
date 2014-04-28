# Plots


import numpy as np, os.path
import pandas, itertools
from kkpandas import kkrs
import kkpandas
from ns5_process import myutils, LBPB
import my, my.dataload, my.plot, scipy.stats
import matplotlib.pyplot as plt
from matplotlib import mlab

plots_to_make = [
    'text summary of intervals',
    'exemplars traces',
    ]


# Hold results
hold_results = pandas.load('../3-1-hold/hold_results')
hold_results['mdiff'] = hold_results['mPB'] - hold_results['mLB']



# Specify the exemplars to use
region2exemplars = {
    'A1': [
        'CR12B_110426_001_behaving-229',
        'CR12B_110503_001_behaving-430',
        'CR17B_110804_001_behaving-329',
        'CR21A_120505_003_behaving-508',
        'CR20B_120602_001_behaving-712',
        ],
    'PFC': [
        'CR24A_121022_001_behaving-417',
        'CR24A_121024_001_behaving-125',
        'CR20B_120602_001_behaving-439',
        'CR24A_121023_001_behaving-237',
        'CR20B_120605_001_behaving-411',
        ]}


my.plot.font_embed()
my.plot.publication_defaults()


# Load effect
piv = pandas.load('block_effect')
bins = np.loadtxt('bins')
bincenters = bins[:-1] + np.diff(bins) / 2.
F_SAMP = np.mean(np.diff(bincenters))
refbin = np.argmin(np.abs(bins - 0.)) - 1
intervals = pandas.load('intervals_of_signif')

# Load rates and parse into data frame rates_df
ulabel2binned = pandas.load('ulabel2binned')
all_ulabels = sorted(ulabel2binned.keys())
lbr_df = pandas.DataFrame([ulabel2binned[ulabel].rate['LB'] 
    for ulabel in all_ulabels], index=all_ulabels)
pbr_df = pandas.DataFrame([ulabel2binned[ulabel].rate['PB'] 
    for ulabel in all_ulabels], index=all_ulabels)
rates_df = pandas.concat([lbr_df, pbr_df], axis=1, verify_integrity=True,
    keys=('LB', 'PB'))
rates_df.index.name = 'ulabel'

# Check binned line up with the testing bins
assert np.max(np.abs(np.array([
    binned.t - bincenters for binned in ulabel2binned.values()]))) < 1e-6

# Yoked z-score the rates
rates_df = rates_df.sub(rates_df.mean(1), axis=0)
rates_df = rates_df.div(rates_df.std(1, ddof=0), axis=0)

# Add metadata to rates_df
regions = pandas.Series(['A1' if kkpandas.kkrs.is_auditory(u) else 'PFC'
    for u in rates_df.index], index=rates_df.index, name='region')
blockpref = pandas.Series(['PB' if hold_results['mdiff'][u] > 0 else 'LB'
    for u in rates_df.index], index=rates_df.index, name='blockpref')
sessions = pandas.Series([kkpandas.kkrs.ulabel2session_name(u)
    for u in rates_df.index], index=rates_df.index, name='session')
assert np.all(hold_results['p_adj'][blockpref.index] < .05)
rates_df.index = pandas.MultiIndex.from_arrays(
    [sessions, blockpref, regions, rates_df.index])
rates_df.index = rates_df.index.droplevel('session')


if 'text summary of intervals' in plots_to_make:
    a1_intervals = intervals.ix[regions[regions=='A1'].index]
    pfc_intervals = intervals.ix[regions[regions=='PFC'].index]
    sdump = []
    sdump.append("A1: (n=%d)" % len(a1_intervals))
    data = a1_intervals.t1.values
    q1, q2, q3 = (mlab.prctile(data, (25, 50, 75)) - refbin) * F_SAMP
    sdump.append("before. q1: %0.3f s, q2: %0.3f s, q3: %0.3f s" % (q1, q2, q3))
    data = a1_intervals.t2.values
    q1, q2, q3 = (mlab.prctile(data, (25, 50, 75)) - refbin) * F_SAMP
    sdump.append("after. q1: %0.3f s, q2: %0.3f s, q3: %0.3f s" % (q1, q2, q3))
    
    sdump.append("\nPFC: (n=%d)" % len(pfc_intervals))
    data = pfc_intervals.t1.values
    q1, q2, q3 = (mlab.prctile(data, (25, 50, 75)) - refbin) * F_SAMP
    sdump.append("before. q1: %0.3f s, q2: %0.3f s, q3: %0.3f s" % (q1, q2, q3))
    data = pfc_intervals.t2.values
    q1, q2, q3 = (mlab.prctile(data, (25, 50, 75)) - refbin) * F_SAMP
    sdump.append("after. q1: %0.3f s, q2: %0.3f s, q3: %0.3f s" % (q1, q2, q3))

    sdump = "\n".join(sdump)
    print sdump
    with file('stat__distribution_rule_encoding_start_and_stop_both_regions', 'w') as fi:
        fi.write(sdump)

# Plot curated selection
# This will do a one-column, combing LB and PB preferring gorups
if 'exemplars traces' in plots_to_make:

    # params to separate traces
    trace_sep = 12

    # Figure objects
    region2handles = {
        'A1': plt.subplots(1, 1, figsize=(3, 4)),
        'PFC': plt.subplots(1, 1, figsize=(3, 4)),}
    for region, (f, ax) in region2handles.items():
        f.subplots_adjust(bottom=.15, top=.9)
        f.suptitle('Example %s neurons' % region, size=14)
        ax.set_ylabel('normalized firing rate')
        ax.set_xlabel('time from stimulus onset (s)')
        #f.text(.5, .01, 'time from stimulus onset (s)', ha='center', va='center', size=14)

    max_units_in_group = max(map(len, region2exemplars.values()))

    # We'll use this to find ulabel
    rates_df_by_ulabel = rates_df.copy()
    rates_df_by_ulabel.index = rates_df.index.reorder_levels((2, 0, 1))

    # Plot each region
    for region, exemplars in region2exemplars.items():
        # Find the figure for this region
        f, ax = region2handles[region]
        
        # Plot each one in specified order
        for nu, ulabel in enumerate(exemplars):
            lb_row = rates_df_by_ulabel.ix[ulabel]['LB'].values[0]
            pb_row = rates_df_by_ulabel.ix[ulabel]['PB'].values[0]
            
            # Get the times and optionally slice
            t1_sig, t2_sig = intervals.ix[ulabel][['t1', 't2']]
            t1_full, t2_full = 0, len(bincenters) - 1
            
            # Set the "offset" of the trace, decreasing order from top
            offset = trace_sep * (max_units_in_group - nu - 1)
            
            # Thin lines outside of signif hold
            ax.plot(bincenters[t1_full:t2_full], 
                offset + lb_row[t1_full:t2_full], 
                color='b', lw=1)
            ax.plot(bincenters[t1_full:t2_full], 
                offset + pb_row[t1_full:t2_full], 
                color='r', lw=1)           
            
            # Thick lines where signif
            ax.plot(bincenters[t1_sig:t2_sig], 
                offset + lb_row[t1_sig:t2_sig], 
                color='b', lw=1.5)
            ax.plot(bincenters[t1_sig:t2_sig], 
                offset + pb_row[t1_sig:t2_sig], 
                color='r', lw=1.5)
            
            # Fill between the sig area
            ax.fill_between(x=bincenters[t1_sig:t2_sig], 
                y1=offset + lb_row[t1_sig:t2_sig],
                y2=offset + pb_row[t1_sig:t2_sig], color='gray', alpha=.35)
            #~ 1/0
            # Zero-line
            ax.plot([0, 0], (-trace_sep/2, trace_sep * max_units_in_group - 5), 'k:')
        
    # Pretty
    for region, (f, ax) in region2handles.items():
        ax.set_xlim((-3, 3))
        ax.set_ylim((-trace_sep/2, trace_sep * max_units_in_group - 5))
        ax.set_yticks([])
        for side in ['left', 'top', 'right']:
            ax.spines[side].set_visible(False)
            ax.tick_params(**{side:False})
    
    
    region2handles['A1'][0].patch.set_visible(False)
    region2handles['A1'][0].savefig('A1_traces_by_neuron.pdf')
    region2handles['PFC'][0].patch.set_visible(False)
    region2handles['PFC'][0].savefig('PFC_traces_by_neuron.pdf')



plt.show()
