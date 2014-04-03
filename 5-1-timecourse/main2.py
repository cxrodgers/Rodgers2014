# PSTHs, split by prefblock and region and go/nogo

import my, my.dataload
import numpy as np, pandas, kkpandas, os.path, itertools
from ns5_process import LBPB
import matplotlib.pyplot as plt, my.plot
from shared import *

my.plot.font_embed()
my.plot.publication_defaults()

## What figures to make
plots_to_make = [
    'Region-averaged timecourse in prefblock, split by go-nogo',
    'Region-averaged timecourse, split by block, separately by go/nogo and prefblock',
    'Region-averaged timecourse, split by prefblock, separately by go/nogo',
    ]

# Bin params (t_start and t_stop should match folded params)
bins = kkpandas.define_bin_edges2(t_start=-3., t_stop=3, binwidth=.05)
bincenters = 0.5 * (bins[:-1] + bins[1:])
distr_bins = kkpandas.define_bin_edges2(t_start=-3., t_stop=3, binwidth=.1)
meth = np.histogram

# What type of data to get: locked to cpoke stop, after all trials
locking_event = 'cpoke_stop'
after_trial_type = 'after_all'

# Load data
try:
    binneds
except NameError:
    suffix, all_event_latencies, binneds = load_data(
        locking_event=locking_event, after_trial_type=after_trial_type,
        bins=bins, meth=meth)

# Assign prefblock
hold_results = pandas.load(os.path.expanduser(
    '~/Dropbox/figures/20130621_figfinal/02_hold/hold_results'))
hold_results['prefblock'] = 'xx'
hold_results['prefblock'][hold_results.mPB > hold_results.mLB]  = 'PB'
hold_results['prefblock'][hold_results.mLB > hold_results.mPB]  = 'LB'
hold_results['prefblock'][hold_results.p_adj > .05]  = 'ns'

## Zscored rates
# Mean and zscore
rates = mean_and_zscored_binneds2(binneds)

# Slice out rates and latencies just during prefblock
# Useful for some analyses below
prefblock_rates, nprefblock_rates, prefblock_latencies = \
    rearrange_by_prefblock(rates, all_event_latencies, hold_results)
   
## PLOTS
if 'Region-averaged timecourse in prefblock, split by go-nogo' in plots_to_make:
    # Region-averaged timecourse in prefblock, split by go/nogo
    # * Left ax: PFC. Right ax: A1
    # * During prefblock only
    # * One trace for GO, one trace for NOGO
    # * Distrs: stim_onset, choice_made (GO only), next_trial (NO only)    
    
    f, axa = plt.subplots(1, 2, figsize=(7,3))
    region_l = ['PFC', 'A1']
    f.subplots_adjust(left=.125, right=.95, wspace=.475)
    label2color = {'GO': 'g', 'NO': 'gray'}
    bottom_edge = -1.5


    for region, ax in zip(region_l, axa):
        ## TRACES
        subdf = prefblock_rates.ix[region]

        for gng in ['GO', 'NO']:
            # Plot the data
            my.plot.errorbar_data(data=subdf[gng].values, x=bincenters, ax=ax,
                fill_between=True, 
                color=label2color[gng], label=gng)
        
        ## DISTRS
        # Get the distrs from the same ulabels
        ax_distrs = index_by_last_level(prefblock_latencies, subdf.index)
        
        # Plot them
        plot_distrs(ax, ax_distrs, bottom_edge, locking_event, distr_bins)


        ## UNIQUE TO THIS FIGURE
        ax.set_xlim((-3, 3))
        ax.set_title(region)
        
        # Legend
        ax.text(2, 1.25, 'NOGO', color='gray', ha='center', weight='bold')
        ax.text(2, 1.0, 'GO', color='green', ha='center', weight='bold')


        ## PRETTY
        pretty(ax, bottom_edge)
        set_xlabel_by_locking_event(ax, locking_event=locking_event)    

        
        f.savefig('Region-averaged timecourse in prefblock, '\
            'split by go-nogo%s.pdf' % suffix)


if 'Region-averaged timecourse, split by block, separately by go/nogo and prefblock' in plots_to_make:
    # 4 x 2 subplots
    # Left col: LB-preferring neurons. Right col: PB-preferring neurons
    # Rows: A1 NOGO trials, PFC NOGO trials, A1 GO trials, PFC GO trials
    # Each axis contains a trace for each block
    # Distrs: stim_onset, and choice_made XOR next_trials, as appropriate

    # Set up the figure
    prefblock_l = ['LB', 'PB']
    region_gng_l = [
        ('A1', 'NO'),
        ('PFC', 'NO'),
        ('A1', 'GO'),
        ('PFC', 'GO'),
        ]
    label2color = {'LB': 'b', 'PB': 'red'}
    f, axa = plt.subplots(len(region_gng_l), len(prefblock_l), figsize=(7, 14))
    f.subplots_adjust(left=.1, right=.95, hspace=.6, wspace=.5, top=.95, bottom=.05)
    bottom_edge = -2
    
    # Iterate over axes
    for (region, gng), prefblock in itertools.product(region_gng_l, prefblock_l):
        ## PLOT TRACES
        # Get axis object for this set of parameters
        ax = axa[region_gng_l.index((region, gng)), 
            prefblock_l.index(prefblock)]

        # Which ulabels have this prefblock
        ax_ulabels = my.pick(hold_results, 
            prefblock=prefblock, region=region)

        # Get rates for this set of parameters
        subdf = rates.ix[region].xs(gng, level='gng', axis=1).ix[
            ax_ulabels]

        # Iterate over blocks and plot them
        for label, color in label2color.items():
            my.plot.errorbar_data(
                data=subdf[label].values, x=bincenters, ax=ax,
                fill_between=True, color=color, label=label)
        
        ## DISTRS
        # Take same ulabels, combine across blocks
        ax_distrs = take_distrs_from_ulabels_and_combine_blocks(
            all_event_latencies, ulabels=subdf.index, gng=gng)
        
        # Plot them
        plot_distrs(ax, ax_distrs, bottom_edge, locking_event, distr_bins, maxheight=1)

        ## UNIQUE TO THIS FIGURE
        ax.set_title("%s, %s trials only" % (region, 
            'go' if gng == 'GO' else 'nogo'))
        
        ax.set_yticks((-2, -1, 0, 1, 2))

        ## PRETTY
        pretty(ax, bottom_edge)
        set_xlabel_by_locking_event(ax, locking_event=locking_event)    

    
    f.savefig('Region-averaged timecourse, split by block, separately by '\
        'go-nogo and prefblock%s.pdf' % suffix)


if 'Region-averaged timecourse, split by prefblock, separately by go/nogo' in plots_to_make:
    # 2 x 2 plot
    # Columns: left=PFC, right=A1
    # Rows: Top = nogo trials, bottom = go trials
    # Traces: preferred block (purple); non-preferred block (green)
    # Distrs: stim_onset, and choice_made XOR next_trials, as appropriate
    
    # Make figure
    region_gng_l = [
        ('A1', 'NO'),
        ('PFC', 'NO'),
        ('A1', 'GO'),
        ('PFC', 'GO'),
        ]    
    label2color = {'P': 'purple', 'NP': 'green'}
    bottom_edge = -2
    f, axa = plt.subplots(2, 2, figsize=(7, 7))
    f.subplots_adjust(left=.125, right=.95, wspace=.475, hspace=.4)
    
    # Iterate over axes
    for region, gng in region_gng_l:
        ## PLOT TRACES
        # Which axis
        ax = axa[gng=='GO', region=='A1']
        
        # Concatenate pref and nonpref rates in this region and for this gng
        subdf = pandas.concat([
            prefblock_rates.ix[region][gng],
            nprefblock_rates.ix[region][gng]],
            keys=['P', 'NP'], verify_integrity=True, axis=1)
        
        # Iterate over pref/nonpref and plot them
        for label, color in label2color.items():
            my.plot.errorbar_data(
                data=subdf[label].values, x=bincenters, ax=ax,
                fill_between=True, color=color, label=label)
        
        ## DISTRS
        # Take same ulabels, combine across blocks
        ax_distrs = take_distrs_from_ulabels_and_combine_blocks(
            all_event_latencies, ulabels=subdf.index, gng=gng)
        
        # Plot them
        plot_distrs(ax, ax_distrs, bottom_edge, locking_event, distr_bins, maxheight=.75)        
        
        ## UNIQUE TO THIS FIGURE
        ax.set_yticks((-2, -1, 0, 1, 2))
        ax.set_title("%s, %s trials only" % (region, 
            'go' if gng == 'GO' else 'nogo'))
        ax.text(2.8, 1.75, 'preferred', color='purple', ha='right')
        ax.text(2.8, 1.5, 'non-preferred', color='green', ha='right')
        
        if ax in axa[:, 0]:
            ax.set_ylabel('normalized pop. response')      

        ## PRETTY
        pretty(ax, bottom_edge)
        set_xlabel_by_locking_event(ax, locking_event=locking_event)            
            
    # Save
    f.savefig('Region-averaged timecourse, split by prefblock, '\
        'separately by go-nogo%s.pdf' % suffix)

plt.show()