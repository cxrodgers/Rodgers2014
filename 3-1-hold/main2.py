# Hold period exemplar plots
# Also write out mean FRs and p_adj for the exemplars as a stat__

import numpy as np
import matplotlib.pyplot as plt
from ns5_process import myutils
import kkpandas, pandas, os.path, kkpandas.kkrs
import my, my.plot, my.dataload


# Load hold results
hold_results = pandas.load('hold_results')

# Plot defaults
my.plot.font_embed()
my.plot.publication_defaults()

# Which to plot
exemplars = [
    'CR20B_120602_001_behaving-712',
    'CR20B_120602_001_behaving-612',
    'CR24A_121021_001_behaving-257',
    'CR24A_121021_001_behaving-259',
    ]

# yaxis params
barplot_ymax = (12, 16, 20, 12); barplot_ynum = 5
psth_ymax = (50, 25, 25, 10); psth_ynum = 6
y_star = (12, 6, 9, 3.9)


# Text summary stats
for exemplar in exemplars:
    print exemplar
    row = hold_results.ix[exemplar]
    print "%d LB trials, mean %0.3f; %d PB trials, mean %0.3f." % (
        len(row['LB_counts']), row['mLB']/.25, 
        len(row['PB_counts']), row['mPB']/.25,)
    print "pmw = %r , p_adj = %r \n" % (row['pmw'], row['p_adj'])


for nu, (ulabel, rec) in enumerate(hold_results.ix[exemplars].iterrows()):
    # Bars
    cub = rec['counts_by_block']
    block_means = map(np.mean, cub)
    block_stderrs = map(myutils.std_error, cub)
    
    # Normalize to count window
    block_means = np.asarray(block_means) / rec['dt']
    block_stderrs = np.asarray(block_stderrs) / rec['dt']

    # Plot bars
    f, axa = plt.subplots(1, 2, figsize=(7,3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475)

    ax = axa[0]
    bar_centers = np.arange(len(block_means))[::2] + 1
    ax.bar(left=bar_centers, align='center',
        height=block_means[::2], yerr=block_stderrs[::2], color='b',
        ecolor='k', label='Localization', capsize=0, width=1) # LB
    ax.bar(left=bar_centers+1, align='center',
        height=block_means[1::2], yerr=block_stderrs[1::2], color='r',
        ecolor='k', label='Pitch Discrimination', capsize=0, width=1) # PB
    ax.set_xlim((0.5, bar_centers.max()+1.75))
    ax.set_xticks(range(4, bar_centers.max() + 2, 4))

    #ax.set_title(ulabel)
    ax.set_ylabel('Spike rate (Hz)')
    ax.set_xlabel('Block number')
    my.plot.despine(ax)
    ax.set_ylim((0, barplot_ymax[nu]))
    ax.set_yticks(np.linspace(0, barplot_ymax[nu], barplot_ynum))
    #ax.set_ylim((0, 5*np.ceil(ax.get_ylim()[1] / 5.)))
    #ax.legend(loc='upper right')
    
    # Get PSTH
    ratname = kkpandas.kkrs.ulabel2ratname(ulabel)
    sname2folded = my.dataload.ulabel2dfolded(ulabel, 
        folding_kwargs={'dstart': -.3, 'dstop': .3},
        trial_picker_kwargs='random hits by block')
    
    # Choose bins such that zero is NOT included (plotting nicety)
    bins = kkpandas.define_bin_edges2(t_start=-.3, t_stop=.3, bins=18)
    binned_LB = kkpandas.Binned.from_folded_by_trial(sname2folded['LB'], bins=bins)
    binned_PB = kkpandas.Binned.from_folded_by_trial(sname2folded['PB'], bins=bins)
    
    # now plot
    ax = axa[1]
    x = binned_LB.t
    data = binned_LB.rate_in('Hz')
    my.plot.errorbar_data(x=x, data=data, ax=ax, axis=1, color='b', label='LB',
        eb_kwargs={'color': 'b'})
    data = binned_PB.rate_in('Hz')
    my.plot.errorbar_data(x=x, data=data, ax=ax, axis=1, color='r', label='PB',
        eb_kwargs={'color': 'r'})
    
    # labels and ranges
    ax.set_ylabel('Spike rate (Hz)')
    ax.set_xlabel('Time rel. to stimulus onset (s)')
    ax.set_xlim((-.3, .3))
    ax.set_xticks((-.25, 0, .25))
    ax.set_ylim((0, psth_ymax[nu]))
    ax.set_yticks(np.linspace(0, psth_ymax[nu], psth_ynum))
    
    # epochs labeled at top of plot
    barpos = ax.get_ylim()[1] * .9
    ax.plot([0, .25], [barpos, barpos], lw=2, color='lime', solid_capstyle='butt')
    ax.plot([-.25, 0], [barpos, barpos], lw=2, color='purple', solid_capstyle='butt')
    ax.text(.125, barpos*1.03, 'stimulus', color='lime', ha='center', va='bottom')
    ax.text(-.125, barpos*1.03, 'hold', color='purple', ha='center', va='bottom')    
    
    # Fill between to show shaded area
    subslc = np.where((binned_LB.t > -.25) & (binned_LB.t < 0))[0]
    lb_vals = binned_LB.rate_in('Hz').mean(1).values
    pb_vals = binned_PB.rate_in('Hz').mean(1).values
    pre_y_LB = np.interp(-.25, 
        [binned_LB.t[subslc[0]-1], binned_LB.t[subslc[0]]],
        [lb_vals[subslc[0]-1], lb_vals[subslc[0]]])
    post_y_LB = np.interp(0, 
        [binned_LB.t[subslc[-1]], binned_LB.t[subslc[-1]+1]],
        [lb_vals[subslc[-1]], lb_vals[subslc[-1]+1]])
    pre_y_PB = np.interp(-.25, 
        [binned_LB.t[subslc[0]-1], binned_LB.t[subslc[0]]],
        [pb_vals[subslc[0]-1], pb_vals[subslc[0]]])
    post_y_PB = np.interp(0, 
        [binned_LB.t[subslc[-1]], binned_LB.t[subslc[-1]+1]],
        [pb_vals[subslc[-1]], pb_vals[subslc[-1]+1]])
    ax.fill_between(
        np.concatenate([[-.25], binned_LB.t[subslc], [0]]),
        y1=np.concatenate([[pre_y_LB], lb_vals[subslc], [post_y_LB]]),
        y2=np.concatenate([[pre_y_PB], pb_vals[subslc], [post_y_PB]]),
        color='gray', alpha=0.5, lw=0)
    
    # A star
    ax.text(-.125, y_star[nu], '*', size=18, va='center', ha='center')
    
    # pretty
    my.plot.despine(ax)

    # save
    #f.patch.set_visible(False)
    plt.savefig('exemplar-%s.pdf' % (ulabel))
    

plt.show()