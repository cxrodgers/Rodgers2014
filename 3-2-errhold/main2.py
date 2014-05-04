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

plots_to_make = [
    'A1 pretty; normalized, combined across prefblock',
    'PFC pretty; normalized, combined across prefblock',
    ]
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

# We will skip any that do not have a boot result
MIN_TRIALS_PER_CATEGORY = 3

# categories of outcomes
cat2snameouts = {
    'hit_LB': ['le_hi_lc-hit', 'le_lo_lc-hit', 'ri_hi_lc-hit', 'ri_lo_lc-hit'],
    'hit_PB': ['le_hi_pc-hit', 'le_lo_pc-hit', 'ri_hi_pc-hit', 'ri_lo_pc-hit'],
    'interference_LB': ['ri_lo_lc-wrong_port', 'le_lo_lc-wrong_port'],
    'interference_PB': ['le_hi_pc-wrong_port', 'le_lo_pc-wrong_port'],
    }

# Which error trials to include
# There are not enough nogo-on-go errors to include, so discard them
cat2snameouts['error_LB'] = ['ri_hi_lc-error', 'ri_lo_lc-error']
cat2snameouts['error_PB'] = ['le_hi_pc-error', 'ri_hi_pc-error']

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

# Error check that preferred block is correct
temp = cat_counts.reset_index(['region'])
assert np.all(np.diff(
    temp.ix['PB'][['hit_LB', 'hit_PB']].applymap(np.mean)) > 0)
assert np.all(np.diff(
    temp.ix['LB'][['hit_LB', 'hit_PB']].applymap(np.mean)) < 0)


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

# Normalize
cat_nrates_combined = cat_rates_combined.sub(
    cat_rates_combined['hit_NP'], axis='index')



cols = ['hit_NP', 'interference_NP', 'hit_P', 'interference_P']
for region in ['PFC', 'A1']:
    f, ax = plt.subplots(figsize=(3, 3))
    region_df = cat_nrates_combined.ix[region][cols]
    my.plot.vert_bar(
        ax=ax,
        bar_lengths=region_df.mean(),
        bar_errs=my.misc.sem(region_df, axis=0),
        bar_colors=['w', 'orange', 'w', 'orange'],
        plot_bar_ends=None,
        bar_positions=[0, 1, 3, 4]
        )
    
    comparisons = [
        ('hit_NP', 'interference_NP', 0, 1),
        ('hit_P', 'interference_P', 3, 4),
        ('interference_P', 'interference_NP', 1, 4)
        ]
    
    if region == 'A1':
        sigline = 7.5
        ylim = ((-2, 10))
    else:
        sigline = 4.5
        ylim = ((-1, 5))
    ax.set_ylim(ylim)
    
    # Hits vs inter, npref block
    sdump = []
    sdump.append(region)
    for col1, col2, xpos1, xpos2 in comparisons:
        p1 = my.stats.r_utest(region_df[col1], region_df[col2],
            paired='TRUE', fix_float=1000)['p']
        sdump.append("* %s vs %s: p=%0.4f" % (col1, col2, p1))
    
    sdump_s = "\n".join(sdump)
    print sdump_s
    with file('stat__anticipatory_effect_on_error_trials_%s' % region, 'w') as fi:
        fi.write(sdump_s)

    # save
    f.savefig('%s_rule_encoding_interference_trials.svg' % region)

plt.show()
