# Demonstrates that tuning for each sound (as assessed by cue trials)
# does not correlate with hold effect.

import pandas, numpy as np, my, my.dataload, kkpandas, kkpandas.kkrs
from ns5_process import LBPB, myutils
import scipy.stats
import matplotlib.pyplot as plt, my.plot

my.plot.font_embed()
my.plot.publication_defaults()

# Load data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')
hold_results['diffHz'] = (hold_results['mPB'] - hold_results['mLB']) / \
    hold_results['dt']


# Load results and join on hold effect
# This is only valid for indiv neurons
analysis = 'sua_cue'

# Previously there was a bug here where ulabel was set to index,
# but this is non-unique
res = pandas.load('decode_results_%s' % analysis)

# Add in information about hold period, region, and auditory-responsiveness
res = res.join(hold_results[['diffHz', 'p_adj']], on='ulabel')
res = res.join(unit_db[['region', 'audresp']], on='ulabel')

# Filter: keep only units for which there were 10 spikes total (across blocks)
keep_indexes = res[['block', 'n_spikes', 'ulabel']].pivot_table(
    rows='ulabel', cols='block', values='n_spikes').sum(1) >= 10
keep_ulabels = keep_indexes.index[keep_indexes.values]
res = my.pick_rows(res, ulabel=keep_ulabels)

# Additional filtering: drop units with low trial count (in EITHER block)
# Sometimes erratic results on the decoding analysis from such units
MIN_TRIAL_COUNT = 25
keep_indexes = res.pivot_table(
    rows='ulabel', cols='block', values='n_trials').min(1) >= MIN_TRIAL_COUNT
keep_ulabels = keep_indexes.index[keep_indexes.values]
res = my.pick_rows(res, ulabel=keep_ulabels)

# Assign prefblock
res['prefblock'] = 'ns'
res['prefblock'][res.p_adj.isnull()] = 'NA'
res['prefblock'][(res.p_adj < .05) & (res.diffHz > 0)] = 'PB'
res['prefblock'][(res.p_adj < .05) & (res.diffHz < 0)] = 'LB'

# Rename
decode_res = res.copy()

# Pivot the decodability in each block
repvar = 'ulabel'
metric = 'cateq'
piv = decode_res.pivot_table(rows=repvar, cols=['block'], values=metric)    

# Join hold effect (quantified as log-ratio) and region
tunings = piv.join(hold_results[['mPB', 'mLB', 'region']])
tunings['lograt'] = np.log10(tunings['mPB'] / tunings['mLB'])

# Rename the tuning metrics
tune_metrics = ['lc_tuneQ', 'pc_tuneQ']
tunings = tunings.rename_axis({'lc': 'lc_tuneQ', 'pc': 'pc_tuneQ'})

# Drop any units that weren't tested in both blocks (for instance, because
# failed the trial count criterion in one block but not the other)
# with the current filtering, this shouldn't happen anyway
tunings = tunings.dropna()

for region in ['A1', 'PFC']:
    subtunings = my.pick_rows(tunings, region=region)
    
    # Correlate with hold effect
    f, axa = plt.subplots(1, 2, figsize=(7,3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475)
    for ax, tune_metric in zip(axa.flatten(), tune_metrics):
        # Set x and y
        x = subtunings[tune_metric].values
        y = subtunings.lograt.values
        
        # Calculate trend
        m, b, rval, pval, stderr = \
            scipy.stats.stats.linregress(x.flatten(), y.flatten())
        
        # stats in title
        #~ ax.set_title('p=%0.3f r=%0.3f n=%d' % (pval, rval, len(x)), size='small')
        
        # Plot the points
        ax.plot(x, 10**y, 'ko', mfc='none')
        ax.set_yscale('log')

        # Y-limits
        if region == 'A1':
            ax.set_yticks((.1, .3, 1, 3, 10))
            ax.set_yticklabels((.1, .3, 1, 3, 10))
            ax.tick_params(which='minor', width=0)
            ax.set_ylim((.1, 10))
        if region == 'PFC':
            ax.set_yticks((.1, .33, 1, 3, 10))
            ax.set_yticklabels((.1, .33, 1, 3, 10))
            ax.tick_params(which='minor', width=0)
            ax.set_ylim((.1, 10))

        # X-limits
        if region == 'A1':
            ax.set_xticks((.5, .6, .7, .8, .9))
            ax.set_xlim((.5, 0.9))
            textpos = .8, .2
        if region == 'PFC':
            ax.set_xticks((.5, .55, .6))
            ax.set_xlim((.5, .6))
            textpos = .575, .2

        # Stats
        ax.text(textpos[0], textpos[1], 'p=%0.3f\nr=%0.3f' % (pval, rval), 
            ha='center', va='center')

        # Plot the trend
        tr_x0, tr_x1 = ax.get_xlim()
        tr_y0 = m * tr_x0 + b
        tr_y1 = m * tr_x1 + b
        ax.plot([tr_x0, tr_x1], [10**tr_y0, 10**tr_y1], 'g-')
        
        # Labels
        if tune_metric == 'lc_tuneQ':
            ax.set_xlabel("decodability of noise burst\nfrom neuron's firing")
        elif tune_metric == 'pc_tuneQ':
            ax.set_xlabel("decodability of warble\nfrom neuron's firing")
        else:
            ax.set_xlabel(tune_metric)
        ax.set_ylabel('ratio of pre-stimulus spike rate\npitch disc. / local.')
        


    #~ f.suptitle((region, metric))
    f.savefig('hold_vs_tuneQ_%s_%s.svg' % (region, metric))

plt.show()