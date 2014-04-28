# Plot spont vs LB vs PB 


import matplotlib.pyplot as plt
import numpy as np, pandas
import my.dataload, my, kkpandas, my.plot
from ns5_process import myutils
import kkpandas.kkrs

my.plot.publication_defaults()
my.plot.font_embed()


# Load hold results
hold_results = pandas.load('../3-1-hold/hold_results')

# load spont
sponts_df = pandas.load('sponts_df')
hold_results['spont'] = sponts_df['passive_spont']


# Hzify
hold_results['mLBhz'] = hold_results['mLB'] / hold_results['dt']
hold_results['mPBhz'] = hold_results['mPB'] / hold_results['dt']

# Normalize
hold_results['nmLBhz'] = hold_results['mLBhz'] - hold_results['spont']
hold_results['nmPBhz'] = hold_results['mPBhz'] - hold_results['spont']

# Group
hold_results['dir'] = 'ns'
hold_results['dir'][(hold_results.p_adj < .05) & (
    hold_results.mLB > hold_results.mPB)] = 'LB'
hold_results['dir'][(hold_results.p_adj < .05) & (
    hold_results.mPB > hold_results.mLB)] = 'PB'

# To hold stats
sdump = ['Comparing hold period rate in each block vs spont rate']

# Plot
f, axa = plt.subplots(2, 3, figsize=(8, 6))
f.subplots_adjust(hspace=.4, wspace=.45)
dir_list = ['LB', 'PB', 'ns']
colnames = ['mLBhz', 'spont', 'mPBhz']
hold_results[colnames] = hold_results[colnames].sub(hold_results.spont, axis=0)
for (dir, region), subdf in hold_results.groupby(['dir', 'region']):
    ax = axa[region == 'PFC', dir_list.index(dir)]
    

    # Test vs spont
    uL = my.stats.r_utest(subdf.mLBhz, subdf.spont, paired='TRUE')
    uP = my.stats.r_utest(subdf.mPBhz, subdf.spont, paired='TRUE')
    pL = uL['p']
    pP = uP['p']
    
    sdump.append("%s %s pP: %f pL: %f" % (region, dir, pP, pL))
    
    if region == 'A1':
        ax.set_yticks((-6, -4, -2, 0, 2, 4, 6))
        ax.set_ylim((-6, 6))
        pheight = 5.5
    else:
        ax.set_yticks((-4, -2, 0, 2, 4))
        ax.set_ylim((-4, 4))
        pheight = 3.6
    #~ if dir == 'LB':
        #~ ax.set_ylabel('normalized population response (Hz)')
        
    # p-comps
    ax.plot([0.1, .9], [pheight, pheight], 'k-')
    ax.plot([1.1, 1.9], [pheight, pheight], 'k-')
    if pL < .05:
        ax.text(0.5, pheight+.1, '*', color='k', ha='center', va='bottom', size=18)
    else:
        ax.text(0.5, pheight+.1, 'ns', color='k', ha='center', va='bottom', size=12)
    if pP < .05:
        ax.text(1.5, pheight+.1, '*', color='k', ha='center', va='bottom', size=18)
    else:
        ax.text(1.5, pheight+.1, 'ns', color='k', ha='center', va='bottom', size=12)
        
    # Title
    #~ ax.set_title('%s %s %0.3f %0.3f' % (region, dir, pL, pP))
    sss = 'no preference'
    if dir == 'LB':
        sss = 'local.-preferring'
    if dir == 'PB':
        sss = 'pitch disc.-preferring'
    ax.set_title("%s %s" % (region, sss))

    my.plot.vert_bar(ax=ax,
        bar_lengths=np.mean(subdf[colnames].values, axis=0),
        bar_errs=my.misc.sem(subdf[colnames].values, axis=0),
        bar_labels=['local.', 'spont', 'pitch\ndisc.'],
        bar_colors=['b', 'gray', 'r'],
        tick_labels_rotation=0,
        plot_bar_ends=''
        )
    my.plot.despine(ax)

f.savefig('blocks_vs_true_spont.svg')

# Dump stats
sdump_s = "\n".join(sdump)
print sdump_s
with file('stat__spont_rate_vs_hold_period_rate', 'w') as fi:
    fi.write(sdump_s)

plt.show()