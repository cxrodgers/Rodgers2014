# Like main2, but locked to different events, and in some cases GO and NOGO
# are averaged together, and we use a 50ms Gaussian to smooth, and also
# we re-load the data for each plot (due to different locking etc)

import my, my.dataload
import numpy as np, pandas, kkpandas, os.path
from ns5_process import LBPB
import matplotlib.pyplot as plt, my.plot
from shared import *
import scipy.stats

my.plot.font_embed()
my.plot.publication_defaults()

plots_to_make = [
    #~ 'Timecourse in prefblock, split by region, locked to stimulus onset',
    'Timecourse in prefblock, split by region, locked to entering center poke',
    #~ 'Timecourse in prefblock, separately by region, following succesful nogo',
    ]

# Bin params (t_start and t_stop should match folded params)
# The number of time bins tested affects the p-value correction later
# Cannot go all the way to -3 because there is a smoothing artefact
# (unless we dump more spike times beyond -3)
bins = kkpandas.define_bin_edges2(t_start=-2.9, t_stop=2.9, binwidth=.05)
bincenters = 0.5 * (bins[:-1] + bins[1:])

# These are only used in one of the plots, the after_nogo plot
distr_bins = kkpandas.define_bin_edges2(t_start=-3., t_stop=3, binwidth=.1)

# Smoothing object
gs = kkpandas.base.GaussianSmoother(smoothing_window=.05)
meth = gs.smooth

# Assign prefblock
hold_results = pandas.load(os.path.expanduser(
    '~/Dropbox/figures/20130621_figfinal/02_hold/hold_results'))
hold_results['prefblock'] = 'xx'
hold_results['prefblock'][hold_results.mPB > hold_results.mLB]  = 'PB'
hold_results['prefblock'][hold_results.mLB > hold_results.mPB]  = 'LB'
hold_results['prefblock'][hold_results.p_adj > .05]  = 'ns'




def prefblock_timecourse_split_by_region(binneds, all_event_latencies):
    """Helper function for two very similar plots"""
    ## Zscored rates
    rates = mean_and_zscored_binneds2(binneds)
    
    # Slice out rates and latencies just during prefblock
    prefblock_rates, nprefblock_rates, prefblock_latencies = \
        rearrange_by_prefblock(rates, all_event_latencies, hold_results)    

    ## Create figure
    f, ax = plt.subplots(1, 1, figsize=(6.5,3.5)) # Used to be 7x4
    f.subplots_adjust(left=.125, right=.95, wspace=.475)
    region2color = {
        'PFC': 'purple', 
        'A1': 'orange',
        }

    ## TRACES
    # Iterate over regions
    sdump = []
    for region, color in region2color.items():
        # Select region, mean over GO/NOGO
        subdf = prefblock_rates.ix[region].mean(axis=1, level=1)
        subarr = np.array(subdf)
        
        # Test vs 0
        pvals = scipy.stats.ttest_1samp(subarr, 0)[1]
        pvals = my.stats.r_adj_pval(pvals)
        
        # This will be the thick line
        # Rates which are sig greater than 0
        sigrates = subarr.mean(0)
        sigrates[pvals >= .05] = np.nan
        sigrates[sigrates < 0] = np.nan
        
        # Dump text summary
        sig_msk = pvals < .05
        first_sig = np.where(sig_msk & (sigrates > 0))[0][0]
        lose_sig = np.where(~sig_msk[first_sig:])[0][0] + first_sig
        sdump.append("%s: n=%d first sig at %0.3f then loses sig at %0.3f" % (region,
            subarr.shape[0], bincenters[first_sig], bincenters[lose_sig]))

        ## Plot
        plot_data = subdf.values[:, bincenters <= .926]
        plot_x = bincenters[bincenters <= .926]
        plot_sigrates = sigrates[bincenters <= .926]
        my.plot.errorbar_data(data=plot_data, x=plot_x, ax=ax,
            fill_between=True, color=color, lw=0)
        
        # Overplot thick line where sig
        ax.plot(plot_x, plot_sigrates, color=color, lw=4)
    
    return f, ax, sdump


## Plots
if 'Timecourse in prefblock, split by region, locked to stimulus onset' in plots_to_make:
    # Prefblock rates only. Purple: PFC. Orange: A1. Thicker where sig.
    # Locked to stimulus onset, after all trials
    ## Load data
    locking_event = 'stim_onset'
    after_trial_type = 'after_all'
    try:
        binneds
    except NameError:
        suffix, all_event_latencies, binneds = load_data(
            locking_event=locking_event, after_trial_type=after_trial_type,
            bins=bins, meth=meth)

    ## Make the figure with the helper function
    f, ax, sdump = prefblock_timecourse_split_by_region(binneds, all_event_latencies)
    
    sdump = "\n".join(sdump)
    print sdump
    with file('stat__distribution_of_hold_effect_duration_at_the_population level', 'w') as fi:
        fi.write(sdump)
    
    ## Pretty
    set_xlabel_by_locking_event(ax, locking_event=locking_event)   

    ## UNIQUE TO THIS FIGURE
    ax.set_xlim((-3, 1))
    ax.set_ylim((-.5, 1.5))
    ax.set_ylabel('normalized population response')
    ax.set_xticks((-3, -2, -1, 0, 1))

    ax.plot(ax.get_xlim(), [0, 0], 'k:')
    ax.plot([0, 0], ax.get_ylim(), 'k:')
    
    f.savefig('Timecourse in prefblock, split by region, '\
        'locked to stimulus onset%s.pdf' % suffix)


if 'Timecourse in prefblock, split by region, locked to entering center poke' in plots_to_make:
    # Just like above, but locked to cpoke_start
    ## Load data
    locking_event = 'cpoke_start'
    after_trial_type = 'after_all'
    try:
        binneds
    except NameError:
        suffix, all_event_latencies, binneds = load_data(
            locking_event=locking_event, after_trial_type=after_trial_type,
            bins=bins, meth=meth)

    ## Make the figure with the helper function
    f, ax, sdump = prefblock_timecourse_split_by_region(binneds, all_event_latencies)
    
    ## Pretty
    set_xlabel_by_locking_event(ax, locking_event=locking_event)   

    ## UNIQUE TO THIS FIGURE
    ax.set_xlim((-3, 1))
    ax.set_ylim((-.5, 1.5))
    ax.set_ylabel('normalized population response')
    ax.set_xticks((-3, -2, -1, 0, 1))

    ax.plot(ax.get_xlim(), [0, 0], 'k:')
    ax.plot([0, 0], ax.get_ylim(), 'k:')

    f.savefig('Timecourse in prefblock, split by region, '\
        'locked to entering cpoke%s.pdf' % suffix)
    

if 'Timecourse in prefblock, separately by region, following succesful nogo' in plots_to_make:
    # After nogo hits only. Average GO and NOGO trials. Prefblock only. 
    # Separate axes by region.
    # Lock to cpoke start.
    # Show distrs from previous NOGO stop
    
    ## Load data
    locking_event = 'cpoke_start'
    after_trial_type = 'after_nogo_hit'
    try:
        binneds
    except NameError:
        suffix, all_event_latencies, binneds = load_data(
            locking_event=locking_event, after_trial_type=after_trial_type,
            bins=bins, meth=meth)

    ## Zscored rates
    rates = mean_and_zscored_binneds2(binneds)
    
    # Slice out rates and latencies just during prefblock
    prefblock_rates, nprefblock_rates, prefblock_latencies = \
        rearrange_by_prefblock(rates, all_event_latencies, hold_results)    

    ## Create figure
    region_l = ['PFC', 'A1']
    f, axa = plt.subplots(1, 2, figsize=(7,3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475)
    bottom_edge = -2

    ## TRACES
    for region, ax in zip(region_l, axa):
        # Data for this region
        subdf = prefblock_rates.ix[region].mean(axis=1, level=1)

        # Plot the data
        my.plot.errorbar_data(data=subdf.values, x=bincenters, ax=ax,
            fill_between=True, color='gray')
        
        ## DISTRS
        # Get the distrs from the same ulabels
        ax_distrs = index_by_last_level(prefblock_latencies, subdf.index)

        # Unlike all other plots, here we just want to plot the end of the
        # previous NOGO
        concatted = concat_series(ax_distrs.swaplevel(0, 1)['prev_nogo_stop'])
        add_time_distrs(concatted, ax=ax, event='prev_nogo_stop', 
            yval=bottom_edge, bins=distr_bins, maxheight=1.0)
    
        ## Pretty
        pretty(ax, bottom_edge)
        set_xlabel_by_locking_event(ax, locking_event=locking_event)   

        ## UNIQUE TO THIS FIGURE
        ax.set_title(region)
        ax.set_xlim((-3, 1))
        if ax is axa[0]:
            ax.set_ylabel('normalized population response')
        f.suptitle('Following a successful nogo trial')
        ax.set_xticks((-3, -2, -1, 0, 1))
        ax.text(-1.65, -0.85, 'previous nogo trial', color='r', ha='center')
        ax.set_yticks((-2, -1, 0, 1, 2))
        f.savefig('Timecourse in prefblock, separately by region, '\
            'following succesful nogo%s.pdf' % suffix)

plt.show()
