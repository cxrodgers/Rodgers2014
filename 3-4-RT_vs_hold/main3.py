# Summary figure
# Combines across prefblock and uses MT (go hits) for simplicity

import my, my.dataload, my.plot
import pandas, scipy.stats
import kkpandas, kkpandas.kkrs
import numpy as np
import matplotlib.pyplot as plt

my.plot.font_embed()
my.plot.publication_defaults()

# Load data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')
hold_results['prefblock'] = 'ns'
hold_results['prefblock'][hold_results.mPB > hold_results.mLB] = 'PB'
hold_results['prefblock'][hold_results.mLB > hold_results.mPB] = 'LB'
hold_results['prefblock'][hold_results.p_adj >= .05] = 'ns'

# Load results from main
res = pandas.load('res')
datares = pandas.load('datares')

# Tall
res = res.reset_index()

# Add prefblock
res['prefblock'] = [hold_results['prefblock'][ulabel]
    for ulabel in res.ulabel.values]
res['region'] = [hold_results['region'][ulabel]
    for ulabel in res.ulabel.values]

# Pivot
res = res.pivot_table(
    rows=['region', 'block', 'ulabel'],
    cols='include')

# Include first
res.columns = res.columns.swaplevel(0, 1)

# Dump the analyses that aren't valid
res = res.drop([
    ('all hits', 'pMT'),
    ('all hits', 'rMT'),
    ('nogo hits', 'pMT'),
    ('nogo hits', 'rMT'),
    ], axis=1)

# Dump the ones with no prefblock
#~ res = res.ix[hold_results[hold_results.prefblock != 'ns'].index]

# adjust p-value
for colname, vals in res.iterkv():
    if colname[1].startswith('p'):
        newkey = (colname[0], colname[1] + '_adj')
        res[newkey] = my.stats.r_adj_pval(vals)
        print "%r: %d/%d signif corr" % (newkey,
            np.sum(res[newkey] < .05), len(res))

plots_to_make = [
    'pop hists',
    'pop hists RT',
    ]


if 'pop hists' in plots_to_make:
    sdump = ['correlating motion time and hold period effect across neurons']
    
    # bins for histogramming r
    bins = np.linspace(-.4, .4, 15)
    
    # One figure per region
    for region in ['A1', 'PFC']:
        f, axa = plt.subplots(1, 2, figsize=(7, 3))
        f.subplots_adjust(left=.125, right=.95, wspace=.475)
        f.suptitle((region, 'MT'))
        
        # One col per block
        for nb, block in enumerate(['LB', 'PB']):
            ax = axa[block == 'PB']
            
            # Get r and pvals for correlation
            rvals = res['go hits']['rMT'][region][block]
            pvals = res['go hits']['pMT_adj'][region][block]
            
            # Test pop
            t, p = scipy.stats.ttest_1samp(rvals, 0.)

            sdump.append(
                'MT %s %s. rvals: mean %0.3f p=%0.3f. %d/%d indiv sig' % (
                region, block, rvals.mean(), p, np.sum(pvals<.05), len(pvals)))

            # Plot hist
            my.plot.hist_p(data=rvals.values, p=pvals.values, bins=bins, ax=ax)
            ax.plot([0, 0], ax.get_ylim(), 'g', lw=2)

            #ax.set_title('%s p=%0.3f' % (block, p))
            ax.set_title('Localization' if block == 'LB' else 'Pitch Discrimination')
            ax.set_xticks((-.4, -.2, 0, .2, .4))
            ax.set_xlim((-.4, .4))
            my.plot.despine(ax)
            ax.set_ylabel('number of neurons')
        f.savefig('FR_vs_MT_%s.svg' % region)
    
    sdump_s = "\n".join(sdump)
    print sdump_s
    with file('stat__motion_time_vs_hold_period_FR', 'w') as fi:
        fi.write(sdump_s)

if 'pop hists RT' in plots_to_make:
    sdump = ['correlating reaction time and hold period effect across neurons']
    
    # bins for histogramming r
    bins = np.linspace(-.4, .4, 15)
    
    # One figure per region
    for region in ['A1', 'PFC']:
        f, axa = plt.subplots(1, 2, figsize=(7, 3))
        f.subplots_adjust(left=.125, right=.95, wspace=.475)
        f.suptitle((region, 'RT'))
        
        # One col per block
        for nb, block in enumerate(['LB', 'PB']):
            ax = axa[block == 'PB']
            
            # Get r and pvals for correlation
            rvals = res['all hits']['rRT'][region][block]
            pvals = res['all hits']['pRT_adj'][region][block]
            
            # Test pop
            t, p = scipy.stats.ttest_1samp(rvals, 0.)
            
            sdump.append(
                'RT %s %s. rvals: mean %0.3f p=%0.3f. %d/%d indiv sig' % (
                region, block, rvals.mean(), p, np.sum(pvals<.05), len(pvals)))


            # Plot hist
            my.plot.hist_p(data=rvals.values, p=pvals.values, bins=bins, ax=ax)
            ax.plot([0, 0], ax.get_ylim(), 'g', lw=2)

            #ax.set_title('%s p=%0.3f' % (block, p))
            ax.set_title('Localization' if block == 'LB' else 'Pitch Discrimination')
            ax.set_xticks((-.4, -.2, 0, .2, .4))
            ax.set_xlim((-.4, .4))
            ax.set_ylabel('number of neurons')
            my.plot.despine(ax)
        f.savefig('FR_vs_RT_%s.svg' % region)

    sdump_s = "\n".join(sdump)
    print sdump_s
    with file('stat__reaction_time_vs_hold_period_FR', 'w') as fi:
        fi.write(sdump_s)

plt.show()
