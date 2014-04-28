# The final figs
# Decodability of target in both blocks and regions


import pandas, numpy as np, my, my.dataload, kkpandas, kkpandas.kkrs
from ns5_process import LBPB, myutils
import scipy.stats
import matplotlib.pyplot as plt, my.plot

# Load data
gets = my.dataload.getstarted()
unit_db = gets['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')
hold_results['diffHz'] = (hold_results['mPB'] - hold_results['mLB']) / \
    hold_results['dt']


my.plot.font_embed()
my.plot.publication_defaults()

# Load ensemble results
decode_res = {}
import os.path
dirname = '.'
for ensem in ['A1', 'PFC']:
    decode_res[ensem + '_msua_target'] = pandas.load(os.path.join(dirname,
        'decode_results_msua_target_%s' % ensem))


## PIVOT DATA FOR ANOVA
rec_l = []
for ensem in ['A1', 'PFC']:
    # Load from this region
    thisreg = decode_res[ensem + '_msua_target']
    thisreg['ensem'] = ensem

    # Keep only certain columns
    rec_l.append(thisreg[['ensem', 'session', 'block', 'var', 'cateq', 'cateq_CV']])
df2 = pandas.concat(rec_l, axis=0, verify_integrity=True, ignore_index=True)
df2 = df2.rename(columns={'cateq': 'score'})

# And for bar plot
scores_df2 = df2.pivot_table(
    cols=['ensem', 'var', 'block'], rows=['session'], values='score')

## RUN ANOVA
# To keep our sanity, let's separately analyze each block
# ie for both blocks: score ~ ensem * var + Error(session\var)
anova_res2 = my.stats.anova(df2, fmla='score ~ ensem * var * block', typ=2)
sdump = ''
sdump += "Ensemble tuning by block, variable, and region\n"
sdump += str(anova_res2['aov'])
sdump += '\n'
sdump += "p_F: %0.4f, p_region: %0.4f, p_var: %0.4f" % (
    anova_res2['lm'].f_pvalue, 
    anova_res2['pvals']['p_ensem'],
    anova_res2['pvals']['p_var'])
sdump += '\n'

# We assume this below in plotting asterisks
# If not, change the number of asterisks plotted
assert anova_res2['pvals']['p_ensem'] < 0.001

# Count number of units and number of ensembles included
sdump += "Dataset size for anova:\n"
A1_df_for_N = decode_res['A1_msua_target'].pivot_table(
    rows='session', values='n_units')
PFC_df_for_N = decode_res['PFC_msua_target'].pivot_table(
    rows='session', values='n_units')
sdump += "A1: n=%d ensembles; PFC: n=%d ensembles" % (
    len(A1_df_for_N), len(PFC_df_for_N))
sdump += '\n'
sdump += "A1: n=%d neurons; PFC: n=%d neurons" % (
    A1_df_for_N.sum(), PFC_df_for_N.sum())
sdump += '\n'

def df_sem(df):
    """Ignores null values in calculation of N"""
    return df.std() / np.sqrt((~pandas.isnull(scores_df2)).sum())

## POST-HOCS
pval_recs = []
sdump += 'post-hocs:\n'
for ensem in ['A1', 'PFC']:
    for var in ['sL', 'sP']:
        LB_values = scores_df2[ensem][var]['lc'].dropna()
        PB_values = scores_df2[ensem][var]['pc'].dropna()
        assert np.all(LB_values.index == PB_values.index)
        #~ pval = my.stats.r_utest(LB_values, PB_values, paired='TRUE', 
            #~ fix_float=True)['p']
        pval = scipy.stats.ttest_rel(LB_values, PB_values)[1]
        pval_recs.append({'ensem': ensem, 'var': var, 'p': pval})
        sdump += '%s %s unadj p=%0.4f\n' % (ensem, var, pval)
pval_df = pandas.DataFrame.from_records(pval_recs)
pval_df['p_adj'] = my.stats.r_adj_pval(pval_df['p'])

# Should all be nonsig; below we assume this in plotting 'ns'
assert np.all(pval_df['p_adj'] > 0.05)

with file('stat__anova_evoke_decode', 'w') as fi:
    fi.write(sdump)
print sdump



# Bar plot of decodability
f, ax = plt.subplots(figsize=(7, 4))
f.subplots_adjust(bottom=.2, right=.85)
my.plot.vert_bar(
    bar_positions=[0, 1, 3, 4, 6, 7, 9, 10],
    bar_lengths=scores_df2.mean(),
    bar_errs=df_sem(scores_df2),
    bar_colors=['b', 'r'] * 4,
    tick_labels_rotation=90, plot_bar_ends=None,
    #~ bar_labels=[' '.join(tuple) for tuple in scores_df2.columns],
    bar_labels = ['' for tuple in scores_df2.columns],
    ax=ax)
ax.set_ylim((.45, .75))


# Labels for x-ax
#~ f.text(.05, .165, 'region', ha='center', va='center')
#~ f.text(.05, .125, 'sound', ha='center', va='center')
#~ f.text(.05, .085, 'block', ha='center', va='center')

# Chance level
ax.plot(ax.get_xlim(), [.5, .5], 'k:')

# Significance bars
ax.plot([0, 1], [.625, .625], 'k')
ax.text(.5, .63, 'ns', color='k', ha='center', va='bottom')
ax.plot([3, 4], [.6, .6], 'k')
ax.text(3.5, .605, 'ns', color='k', ha='center', va='bottom')
ax.plot([6, 7], [.55, .55], 'k')
ax.text(6.5, .555, 'ns', color='k', ha='center', va='bottom')
ax.plot([9, 10], [.55, .55], 'k')
ax.text(9.5, .555, 'ns', color='k', ha='center', va='bottom')


ax.plot([0.5, 3.5], [.675, .675], 'k') # first group
ax.plot([6.5, 9.5], [.6, .6], 'k') # second group
ax.plot([2, 8], [.7, .7], 'k') # link groups
ax.plot([2, 2], [.675, .7], 'k') # vert1
ax.plot([8, 8], [.6, .7], 'k') # vert2
ax.text(5, .71, '***', ha='center', va='bottom', size=18)



f.patch.set_visible(False)
f.savefig('MSUA_tuning_by_block_and_sound_both_regions.svg')






plt.show()



