# Helper functions
import numpy as np, pandas, kkpandas, os.path


## Data loading
def load_data(locking_event, after_trial_type, bins, meth):
    """Load the appropriate dataset given by locking_event and after_trial_type
    
    Returns: suffix, all_event_latencies, binneds
        all_event_latencies is a series with multi-index (
            block, gng, event, ulabel)
    """
    suffix = '_lock_%s_%s' % (locking_event, after_trial_type)
    dfoldeds = pandas.load('dfoldeds' + suffix)
    times_distr = pandas.load('times_distrs' + suffix)

    # Bin each individual sname from each folded
    binneds = dfoldeds['dfolded'].apply(
        lambda dfolded: kkpandas.Binned.from_dict_of_folded(
            dfolded, bins=bins, meth=meth))

    # Convert times_distr to a series keyed by (block, gng, event, ulabel)
    # instead of by (ulabel, group, event)
    all_event_latencies = times_distr.copy()
    all_event_latencies.index = pandas.MultiIndex.from_tuples([
        (idx[1][:2], idx[1][-2:], idx[2], idx[0])
        for idx in all_event_latencies.index],
        names=['block', 'gng', 'event', 'ulabel'],)

    return suffix, all_event_latencies, binneds


## Zscoring and grouping functions
# Group by the following
default_stim_groups = {
    'LB GO': ['le_lo_lc', 'le_hi_lc'],
    'LB NO': ['ri_lo_lc', 'ri_hi_lc'],
    'PB GO': ['ri_lo_pc', 'le_lo_pc'],
    'PB NO': ['ri_hi_pc', 'le_hi_pc'],
    }

def mean_and_zscored_binneds(binneds, stim_groups=default_stim_groups):
    """For each ulabel, zscores binned rates across all snames, 
    then groups by snames."""
    series_l, index_l = [], []
    for ulabel in binneds.index:
        # Get the binned for this ulabel
        rates = binneds[ulabel].rate
        
        # Zscore all snames together
        rates2 = rates - rates.values.flatten().mean()
        rates2 = rates2 / rates2.values.flatten().std()
        
        # Iterate over stim groups
        for stim_group_name, grouped_snames in stim_groups.items():
            # Select only grouped snames and mean
            meaned = rates2[grouped_snames].mean(axis=1).values

            # Store
            series_l.append(pandas.Series(meaned))
            index_l.append((ulabel, stim_group_name))

    # Dataframe the zscored, selected rates
    rates = pandas.concat(series_l, axis=1)
    rates.columns = pandas.MultiIndex.from_tuples(
        index_l, names=['ulabel', 'group'])

    # Put ulabels on the rows and stim group on the columns
    rates = rates.T.unstack()
    rates.columns = rates.columns.swaplevel(0, 1)

    return rates

def mean_and_zscored_binneds2(binneds, stim_groups=default_stim_groups):
    """Same as mean_and_zscored_binneds, but puts block and typ as diff levels
    
    Returns: rates, DataFrame
        rows: region, ulabel
        columns: block, gng, bin
    """
    series_l, index_l = [], []
    for ulabel in binneds.index:
        # Get the binned for this ulabel
        rates = binneds[ulabel].rate
        
        # Zscore all snames together
        rates2 = rates - rates.values.flatten().mean()
        rates2 = rates2 / rates2.values.flatten().std()
        
        # Iterate over stim groups
        for stim_group_name, grouped_snames in stim_groups.items():
            # Select only grouped snames and mean
            meaned = rates2[grouped_snames].mean(axis=1).values

            # Store
            series_l.append(pandas.Series(meaned))
            
            # These will become row and column indices
            # ulabel, block, gng
            index_l.append((
                ulabel, 
                stim_group_name[:2], # block
                stim_group_name[-2:] # gng
                ))

    # Dataframe the zscored, selected rates
    rates = pandas.concat(series_l, axis=1)
    rates.columns = pandas.MultiIndex.from_tuples(
        index_l, names=['ulabel', 'block', 'gng'])
    rates.index.name = 'bin'

    # Put ulabels on the rows and bins on the columns
    rates = rates.stack('ulabel').unstack('bin')

    # Add region as a level to index
    rates.index = pandas.MultiIndex.from_arrays([
        ['A1' if kkpandas.kkrs.is_auditory(ulabel) else 'PFC' 
            for ulabel in rates.index],
        rates.index, 
        ], names=['region', 'ulabel'])    

    return rates

def rearrange_by_prefblock(rates, all_event_latencies, hold_results):
    """Given binned rates, slice out just during each ulabel's prefblock.
    
    Returns: prefblock_rates, nprefblock_rates, prefblock_latencies
    """
    # Extract the rate just during the prefblock for each unit
    LB_pref_ulabels = hold_results[hold_results.prefblock == 'LB'].index    
    PB_pref_ulabels = hold_results[hold_results.prefblock == 'PB'].index
    rates_indexswap = rates.swaplevel(0, 1)
    prefblock_rates = pandas.concat([
        rates_indexswap.ix[LB_pref_ulabels]['LB'],
        rates_indexswap.ix[PB_pref_ulabels]['PB'],
        ], verify_integrity=True).swaplevel(0, 1)

    # Also extract during non-prefblock and concat these
    nprefblock_rates = pandas.concat([
        rates_indexswap.ix[LB_pref_ulabels]['PB'],
        rates_indexswap.ix[PB_pref_ulabels]['LB'],
        ], verify_integrity=True).swaplevel(0, 1)

    # Similarly, extract event latencies just during preferred block
    prefblock_latencies = pandas.concat([
        index_by_last_level(all_event_latencies['LB'], LB_pref_ulabels),
        index_by_last_level(all_event_latencies['PB'], PB_pref_ulabels),
        ], verify_integrity=True)

    return prefblock_rates, nprefblock_rates, prefblock_latencies


## Slicer functions
def index_by_last_level(ser, keys):
    return ser.swaplevel(0, -1).ix[keys].swaplevel(0, -1)

def concat_series(ser):
    return np.concatenate(list(ser))

## Helps with difficult concatenation across blocks
def take_distrs_from_ulabels_and_combine_blocks(all_event_latencies, ulabels, 
    gng):
    """Take latencies just from specified ulabels and combine across blocks
    
    Returns in a format suitable for plot_distrs
    """
    # Get the distrs corresponding to these params, for both blocks
    ax_distrs = index_by_last_level(
        all_event_latencies.unstack('block').ix[gng],
        ulabels)
    
    # We need to concatenate across blocks, and also put GO/NOGO on 
    # the index again for compatibility with plot_distrs
    res_l, idx_l = [], []
    for (event, ulabel), ser in ax_distrs.iterrows():
        res_l.append(concat_series(ser))
        idx_l.append((gng, event, ulabel))
    ax_distrs = pandas.Series(res_l, 
        index=pandas.MultiIndex.from_tuples(idx_l, 
            names=('gng', 'event', 'ulabel')))

    return ax_distrs


## Plotting functions
def add_time_distrs(concatted, ax, event, yval, bins, maxheight=.5):
    """
    Histogram and display the latencies along `yval` at the bottom of the ax
    """
    event_name2color = {'stim_onset': 'k', 'choice_made': 'g', 
        'next_start': 'gray', 'prev_rew_stop': 'g', 'prev_nogo_stop': 'gray'}
    
    # Histogram
    counts, edges = np.histogram(concatted, bins=bins)
    
    # Normalize to max height of 0.5
    counts = counts / float(counts.max()) * maxheight
    
    # Center bins, add zero at end
    distr_bins = 0.5 * (edges[:-1] + edges[1:])
    distr_bins = np.concatenate([distr_bins, 
        [distr_bins.max() + np.mean(np.diff(distr_bins))]])
    counts = np.concatenate([counts, [0]])
    
    # Plot against bottom edge
    ax.plot(distr_bins, counts + yval, '-', 
        color=event_name2color[event])

def plot_distrs(ax, ax_distrs, bottom_edge, locking_event, distr_bins, 
    maxheight=0.5):
    """Plot event latencies into axis
    
    ax_distrs should be keyed by gng, event, ulabel
    """
    # Add stim_onset event from all distrs (unless we're doing cpoke_start)
    if locking_event != 'cpoke_start':
        concatted = concat_series(ax_distrs.swaplevel(0, 1)['stim_onset'])
        add_time_distrs(concatted, ax=ax, event='stim_onset', 
            yval=bottom_edge, bins=distr_bins, maxheight=maxheight)
    
    # Add choice_made event from go_distrs
    if 'GO' in ax_distrs:
        concatted = concat_series(ax_distrs['GO']['choice_made'])
        add_time_distrs(concatted, ax=ax, event='choice_made', 
            yval=bottom_edge, bins=distr_bins, maxheight=maxheight)

    # Add next_start event from nogo_distrs
    # This one can be NaN in rare cases
    if 'NO' in ax_distrs:
        concatted = concat_series(ax_distrs['NO']['next_start'])
        concatted = concatted[~np.isnan(concatted)]
        add_time_distrs(concatted, ax=ax, event='next_start', yval=bottom_edge,
            bins=distr_bins, maxheight=maxheight)    

def set_xlabel_by_locking_event(ax, locking_event):
    if locking_event == 'cpoke_start':
        ax.set_xlabel('time from entering center poke (s)')
    if locking_event == 'cpoke_stop':
        ax.set_xlabel('time from exiting center poke (s)')
    elif locking_event == 'stim_onset':
        ax.set_xlabel('time from stimulus onset (s)')

def pretty(ax, bottom_edge):
    # Zero lines
    ax.plot(ax.get_xlim(), [0, 0], 'k:')
    #~ ax.plot([0, 0], ax.get_ylim(), 'k:')
    
    # Set ylim, with room for time distrs at bottom
    ax.set_ylim((bottom_edge, -bottom_edge))
    ax.plot([0, 0], ax.get_ylim(), 'k:')

    # Pretty
    ax.set_xlim((-3, 3))
