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
res = pandas.load(
    'decode_results_%s' % analysis
    ).set_index('ulabel')
res = res.join(hold_results[['diffHz', 'p_adj']])
res = res.join(unit_db[['region', 'audresp']])

# Assign prefblock
res['prefblock'] = 'ns'
res['prefblock'][res.p_adj.isnull()] = 'NA'
res['prefblock'][(res.p_adj < .05) & (res.diffHz > 0)] = 'PB'
res['prefblock'][(res.p_adj < .05) & (res.diffHz < 0)] = 'LB'
decode_res = res.reset_index()


# Drop units with low spike counts?
# Here we count over the entire session
# This was 20 in the hold period analysis, thoguh that was fewer df, etc.
#~ MIN_SPIKE_COUNT = 5
#~ decode_res = decode_res[decode_res.n_spikes >= MIN_SPIKE_COUNT]

#~ MIN_TRIAL_COUNT = 25
#~ decode_res = decode_res[decode_res.n_trials >= MIN_TRIAL_COUNT]


# Pivot the decodability
repvar = 'ulabel'
metrics = ['auroc', 'cateq']
metrics = ['cateq']
piv = decode_res.pivot_table(rows=repvar, cols=['block'], values=metrics)    


YSCALE = 'linear'
YSCALE = 'log'




# Doesn't really matter which metric we use
for metric in metrics:
    # Join size of hold effect and tuning estimates
    tunings = pandas.DataFrame(hold_results['diffHz'], columns=['hold_size'])
    tunings['lograt'] = np.log10(hold_results.mPB / hold_results.mLB)
    tunings['region'] = hold_results['region']
    tunings['lc_tuneQ'] = piv[metric, 'lc']
    tunings['pc_tuneQ'] = piv[metric, 'pc']
    
    # Tune metrics to plot
    tune_metrics = ['lc_tuneQ', 'pc_tuneQ']

    # Not all were tested
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
            ax.set_title('p=%0.3f r=%0.3f n=%d' % (pval, rval, len(x)), size='small')
            
            # Plot the points
            if YSCALE == 'linear':
                ax.plot(x, y, 'ko', mfc='none')

                # Limits
                if region == 'A1':
                    ax.set_yticks((-.8, -.4, 0, .4, .8))
                    ax.set_ylim((-.8, .8))
                if region == 'PFC':
                    ax.set_yticks((-.8, -.4, 0, .4, .8))
                    ax.set_ylim((-.8, .8))

            else:
                ax.plot(x, 10**y, 'ko', mfc='none')
                ax.set_yscale('log')

                # Limits
                if region == 'A1':
                    #ax.set_yticks((-.8, -.4, 0, .4, .8))
                    ax.set_yticks((.1, .3, 1, 3, 10))
                    ax.set_yticklabels((.1, .3, 1, 3, 10))
                    #plt.yticks((.1, .33, 1, 3.3, 10), (.1, .33, 1, 3.3, 10))
                    ax.tick_params(which='minor', width=0)
                    ax.set_ylim((.1, 10))
                if region == 'PFC':
                    #ax.set_yticks((-.8, -.4, 0, .4, .8))
                    ax.set_yticks((.1, .33, 1, 3, 10))
                    ax.set_yticklabels((.1, .33, 1, 3, 10))
                    #plt.yticks((.1, .33, 1, 3.3, 10), (.1, .33, 1, 3.3, 10))
                    ax.tick_params(which='minor', width=0)
                    ax.set_ylim((.1, 10))

            # X-limits
            if region == 'A1':
                ax.set_xticks((.5, .6, .7, .8, .9))
                #~ ax.set_xlim((.5, 0.9))
                #~ tr_x0, tr_x1 = .5, .85
            if region == 'PFC':
                ax.set_xticks((.5, .55, .6))
                #~ ax.set_xlim((.5, .6))
                #~ tr_x0, tr_x1 = .5, .58


            
            # Plot the trend
            tr_x0, tr_x1 = ax.get_xlim()
            tr_y0 = m * tr_x0 + b
            tr_y1 = m * tr_x1 + b
            if YSCALE == 'linear':
                ax.plot([tr_x0, tr_x1], [tr_y0, tr_y1], 'g-')
            else:
                ax.plot([tr_x0, tr_x1], [10**tr_y0, 10**tr_y1], 'g-')
            
            # Labels
            if tune_metric == 'lc_tuneQ':
                ax.set_xlabel("decodability of noise burst\nfrom neuron's firing")
            elif tune_metric == 'pc_tuneQ':
                ax.set_xlabel("decodability of warble\nfrom neuron's firing")
            else:
                ax.set_xlabel(tune_metric)
            ax.set_ylabel('ratio of spike rate\npitch disc. / local.')
            


        #~ f.suptitle((region, metric))
        f.savefig('hold_vs_tuneQ_%s_%s.svg' % (region, metric))

plt.show()