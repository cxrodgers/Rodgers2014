# Simpler version of main2
# Measure average firing rate in each outcome -- (hit, error, valid wrong port)
# in both blocks.
# Then test for population level effects on the normalized means.
import numpy as np, os.path
import pandas, itertools
from kkpandas import kkrs
import kkpandas
from ns5_process import myutils, LBPB
import my, my.plot, my.dataload
import matplotlib.pyplot as plt


my.plot.font_embed()
my.plot.publication_defaults()

# load data - the preferred block, and the results of main.py
hold_results = pandas.load('../3-1-hold/hold_results')
hold_results['diffHz'] = (hold_results['mPB'] - hold_results['mLB']) / hold_results['dt']
ulabel2snameout2counts = myutils.pickle_load('ulabel2snameout2counts')

# unit info
gets = my.dataload.getstarted()
session_db = gets['session_db']
unit_db = gets['unit_db']

# Which units to analyze: all well sorted units that passed feature check
# and are in analyzeable sessions
units_to_analyze = unit_db[unit_db.include].index

# We will skip any that do not have a boot result, or that have too few
# spikes in some category(?)
MIN_TRIALS_PER_CATEGORY = 3

NORMALIZATION = 'subtract'

# categories of outcomes
# name this by targ meaning / action / distractor meaning
cat2snameouts = {
    'go_go_**_LB' : ['le_hi_lc-hit', 'le_lo_lc-hit'], # targ go, went
    'no_no_**_LB' : ['ri_hi_lc-hit', 'ri_lo_lc-hit'], # targ nogo, didn't go
    'no_go_**_LB' : ['ri_hi_lc-error', 'ri_lo_lc-error'], # targ nogo / went anyway
    #'go_no_**_LB' : ['le_hi_lc-error', 'le_lo_lc-error'], # targ go / but didn't go
    # This is the only one that the identity of the distractor matters, and not target
    '**_wp_go_LB' : ['le_lo_lc-wrong_port', 'ri_lo_lc-wrong_port'],
    
    'go_go_**_PB' : ['le_lo_pc-hit', 'ri_lo_pc-hit'], # targ go, went
    'no_no_**_PB' : ['le_hi_pc-hit', 'ri_hi_pc-hit'], # targ nogo, didn't go
    'no_go_**_PB' : ['le_hi_pc-error', 'ri_hi_pc-error'], # targ nogo / went anyway
    #'go_no_**_PB' : ['le_lo_pc-error', 'ri_lo_pc-error'], # targ go / but didn't go
    # This is the only one that the identity of the distractor matters, and not target
    '**_wp_go_PB' : ['le_hi_pc-wrong_port', 'le_lo_pc-wrong_port'],
    }


# Check no double counting
all_snameouts = np.concatenate(cat2snameouts.values())
assert len(all_snameouts) == len(np.unique(all_snameouts))


# Iterate over units and concatenate the counts for all categories
rec_l = []
for ulabel in units_to_analyze:
    # Check for hold period preference
    boot_row = hold_results.ix[ulabel]
    if boot_row['p_adj'] < .05:
        prefblock = 'PB' if boot_row['diffHz'] > 0 else 'LB'
    else:
        prefblock = 'ns'

    # Get counts
    snameout2counts = ulabel2snameout2counts[ulabel]
    
    # Concatenate over categories
    cat2counts = {}
    for cat, snameouts in cat2snameouts.items():
        cat2counts[cat] = np.concatenate([
            snameout2counts[snameout].astype(np.int) for snameout in snameouts])
    
    # Store
    cat2counts['prefblock'] = prefblock
    cat2counts['ulabel'] = ulabel
    cat2counts['region'] = 'A1' if kkpandas.kkrs.is_auditory(ulabel) else 'PFC'
    rec_l.append(cat2counts)
cat_counts = pandas.DataFrame.from_records(rec_l).set_index([
    'region', 'prefblock', 'ulabel'])

# Dump ones with too few spikes per category
to_drop = (cat_counts.applymap(len) < MIN_TRIALS_PER_CATEGORY).any(1)
print "dropping %d ulabels for too few spikes in a category" % to_drop.sum()
cat_counts = cat_counts[~to_drop]

# Convert counts to rates
cat_rates = cat_counts.applymap(np.mean)
cat_durs = map(lambda s: .25 if s in ['CR24A', 'CR20B', 'CR21A'] else .05, 
    map(kkpandas.kkrs.ulabel2ratname, cat_counts.index.get_level_values(2)))
cat_rates = cat_rates.divide(cat_durs, axis='index')

# Combine across prefblocks
cat_rates_combined = pandas.concat([
    cat_rates.reset_index(['region', 'ulabel']).ix['LB'].rename_axis(
        lambda s: s.replace('LB', 'P').replace('PB', 'NP')),
    cat_rates.reset_index(['region', 'ulabel']).ix['PB'].rename_axis(
        lambda s: s.replace('LB', 'NP').replace('PB', 'P')),
    ]).set_index(['region', 'ulabel']).sort()

# Normalize to hits in non-preferred block
cat_nrates_combined = cat_rates_combined.sub(
    cat_rates_combined[['go_go_**_NP', 'no_no_**_NP']].mean(1), axis='index')

# Reindex by P and NP on columns
cat_nrates_combined = cat_nrates_combined.rename_axis(
    lambda s: (s.split('_')[-1], s[:8]))
cat_nrates_combined.columns = pandas.MultiIndex.from_tuples(
    cat_nrates_combined.columns, names=('block_type', 'trial_type'))


# Bar plot
bar_order = ['go_go_**', 'no_no_**', 'no_go_**', '**_wp_go']

# For stats test
rec_l = []

# Plot each region
for region in ['A1', 'PFC']:
    f, ax = plt.subplots(1, 1, figsize=(6.65,4))
    f.subplots_adjust(bottom=.3, right=.95, left=.3)
    
    # Iterate over blocks and add to bar_lengths, etc
    bar_lengths, bar_errs, bar_positions, bar_names = [], [], [], []
    for block in ['P', 'NP']:
        # Subdf it
        subdf = cat_nrates_combined[block][bar_order].ix[region]
        bar_lengths += list(subdf.mean())
        bar_errs += list(my.misc.sem(subdf, axis=0))
        bar_names += map(lambda s: block + '_' + s, bar_order)
        
        # Check every pair of bars
        for i1 in range(len(bar_order)):
            for i2 in range(i1 + 1, len(bar_order)):
                if i1 == i2:
                    continue
                c1, c2 = bar_order[i1], bar_order[i2]
                utest_res = my.stats.r_utest(subdf[c1], subdf[c2], paired='TRUE')
                rec_l.append({'c1': c1, 'c2': c2, 'p': utest_res['p'], 
                    'block': block, 'region': region})
        
        # Add positions, with a blank space between blocks
        if len(bar_positions) == 0:
            bar_positions += list(range(subdf.shape[1]))
        else:
            bar_positions += list(range(1 + len(bar_positions), 
                1 + len(bar_positions) + subdf.shape[1]))

    # Now make the plot
    my.plot.vert_bar(bar_lengths=bar_lengths, bar_labels=['']*len(bar_lengths), #bar_names,
        bar_positions=bar_positions, tick_labels_rotation=90,
        ax=ax, bar_colors=['w', 'w', 'gray', 'orange']*2, 
        bar_errs=bar_errs, plot_bar_ends=None)
    
    # Pretty
    #f.suptitle(region)
    ax.set_ylabel('normalized pop. response')
    if region == 'A1':
        ax.set_ylim((-2, 10))
        ax.set_yticks((-2, 0, 2, 4, 6, 8, 10))
        comp_yloc1, comp_yloc2 = 6.5, 6.5
    else:
        ax.set_ylim((-2, 10))
        ax.set_yticks((-2, 0, 2, 4, 6, 8, 10))
        comp_yloc1, comp_yloc2 = 5, 5
    
    my.plot.despine(ax)
    f.patch.set_visible(False)
    f.savefig('%s_omnibus_error.svg' % region)
    

# Adjust p-values
barcomps = pandas.DataFrame.from_records(rec_l).set_index(['region', 'block'])
barcomps['p_adj'] = 1.0
barcomps['p_adj'].ix['A1'] = my.stats.r_adj_pval(barcomps['p'].ix['A1'], meth='bonf')
barcomps['p_adj'].ix['PFC'] = my.stats.r_adj_pval(barcomps['p'].ix['PFC'], meth='bonf')
barcomps['is_sig'] = barcomps['p_adj'] < .05
plt.show()        


# Text summary
sdump = []
sdump.append('Comparison of all trial types by region, in preferred and non-preferred blocks')
sdump.append('Note that all comparisons with **_wp_go are sig, and all others are nonsig')
sdump.append(str(barcomps))
sdump_s = "\n".join(sdump)
print sdump_s
with file('stat__anticipatory_effect_by_all_trial_types_in_both_blocks', 'w') as fi:
    fi.write(sdump_s)
