# Catch trial analysis

import numpy as np
import os.path
import glob, pandas
import my, my.plot, my.dataload
import matplotlib.pyplot as plt
import scipy.stats

dir = '../data/miscellaneous/catch_trial_task'

my.plot.publication_defaults()
my.plot.font_embed()

# for ratnames
gets = my.dataload.getstarted()


def my_binom_test(x0, n0, x1, n1):
    return scipy.stats.fisher_exact([[x0, n0-x0], [x1, n1-x1]])[1]

catch_task_filenames = {
    'CR21A': 'CR21A_trials_info',
    'CR20B': 'CR20B_trials_info',
    }

ratname2TI = {}

for ratname, fn in catch_task_filenames.items():
    # Load trials info
    subTI = pandas.load(os.path.join(dir, fn))
    
    # Form "sperf" which has nhits and ntrials for each stim name
    gobj = subTI.groupby('stim_name')['outcome']
    nhits = gobj.apply(lambda ser: (ser == 'hit').sum())
    ntrials = gobj.apply(len)
    sperf = pandas.concat([nhits, ntrials], axis=1, keys=['nhits', 'ntrials'])

    # We want to compare:
    # ri_ca_lc to ri_hi_lc and ri_lo_lc together
    # le_ca_lc to le_hi_lc and le_lo_lc together
    # 
    # ca_hi_pc to le_hi_pc and ri_hi_pc together
    # ca_lo_pc to le_lo_pc and ri_lo_pc together
    # Always compare the first stim (catch) to the combination of the other two
    compgroups = [
        ['ri_ca_lc', 'ri_lo_lc', 'ri_hi_lc'],
        ['le_ca_lc', 'le_lo_lc', 'le_hi_lc'],
        ['ca_hi_pc', 'le_hi_pc', 'ri_hi_pc'],
        ['ca_lo_pc', 'le_lo_pc', 'ri_lo_pc']]
    
    compgroupnames = [
        'localization\nnogo', 'localization\ngo',
        'pitch disc.\nnogo', 'pitch disc.\ngo',
        ]
    
    # Iterate over compgroups and test
    testrec_l = []
    for compgroup in compgroups:
        # Get vals for each stimulus within compgroup
        s0, n0 = sperf['nhits'][compgroup[0]], sperf['ntrials'][compgroup[0]]
        s1, n1 = sperf['nhits'][compgroup[1]], sperf['ntrials'][compgroup[1]]
        s2, n2 = sperf['nhits'][compgroup[2]], sperf['ntrials'][compgroup[2]]
        
        # mean perf -- m0 is catch, m12 is the combination of the other
        m0 = s0 / float(n0)
        m12 = (s1 + s2) / float(n1 + n2)

        # CIs on the mean perf
        CI0 = tuple(my.stats.binom_confint(x=s0, n=n0))
        CI12 = tuple(my.stats.binom_confint(x=s1+s2, n=n1+n2))
        
        # Test
        pJ = my_binom_test(s0, n0, s1 + s2, n1 + n2)
        
        # Store
        testrec_l.append({'sname': compgroup[0], 
            'pJ': pJ,
            'm0': m0, 'm12': m12, 'n0': n0, 'n12': n1+n2,
            'CI0': CI0, 'CI12': CI12})
    testres = pandas.DataFrame.from_records(testrec_l).set_index('sname')

    # Form the bar lengths, errorbars, pvalues
    bar_lengths, bar_errs, bar_names, pvals = [], [], [], []
    for sname in testres.index:
        bar_lengths += [testres['m0'][sname], testres['m12'][sname]]
        bar_errs.append(testres['CI0'][sname])
        bar_errs.append(testres['CI12'][sname])
        pvals.append(testres['pJ'][sname])
    
    # Plot results
    f, ax = plt.subplots(1, 1, figsize=(7, 3))
    f.subplots_adjust(bottom=.1, right=.9)
    
    # Plot them
    my.plot.vert_bar(ax=ax,
        bar_lengths=bar_lengths, 
        bar_errs=np.abs(np.transpose(bar_errs)),
        bar_positions=[0, 1, 3, 4, 6, 7, 9, 10], bar_colors=['r', 'w'] * 4,
        plot_bar_ends=None,
        mpl_ebar=False)
    
    # P-value comparison
    SIGPOS = 1.1
    for nump, p in enumerate(pvals):
        ax.plot([nump*3, nump*3+1], [SIGPOS, SIGPOS], 'k-')
        ax.text(nump*3 + 0.5, SIGPOS+.05, 'ns' if p > 0.05 else '*', 
            ha='center', va='bottom')
    
    # cut off those pvalues for pretty
    ax.set_ylim((0, 1))
    
    # categorize
    ax.set_xticks((0.5, 3.5, 6.5, 9.5))
    ax.set_xticklabels(compgroupnames)
    
    # Pretty
    my.plot.despine(ax)
    f.suptitle(ratname)
    f.suptitle(gets['ratname2num'][ratname])
    f.savefig('catch_modest_%s.svg' % ratname)
    
    # Text summary
    print ratname
    print testres[['pJ', 'n0', 'n12']]

plt.show()