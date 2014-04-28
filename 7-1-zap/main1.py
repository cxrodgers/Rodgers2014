# Summary figures


import zap
import numpy as np
import matplotlib.pyplot as plt
import my, pandas
import scipy.stats

my.plot.font_embed()
my.plot.publication_defaults()

TI_d = my.misc.pickle_load('../data/miscellaneous/zap/TI_d')
session_db = pandas.load('../data/miscellaneous/zap/zap_session_db')

# Which to include
incl_session_db = session_db[session_db.include]

# Apply the same filter to TI_d
TI_d = dict([(k, v) for k, v in TI_d.items() if k in incl_session_db.index ])

# Compute stats
sstats = zap.compute_summary_stats(TI_d)
sstats2 = sstats.copy()
sstats2.columns = sstats2.columns.swaplevel(0, 1)


# Which sessions to include
task_name = ['cued', 'uncued']
ratname2col = {'CR25A': 'red', 'CR28A': 'orange', 'CR28B': 'green'}


plots_to_make = [
    'all trials, by block',
    'split by rat and trial type, conn pairs',
    'bar of impairment', # ordered first by rat, then type
    'indiv sname bar of impairment',
    'example sessions',
    ]

ratname_l = ['CR25A', 'CR28A', 'CR28B']
short_ratname = {'CR25A': 'Z1', 'CR28A': 'Z2', 'CR28B': 'Z3'}



## TEXT SUMMARY
def normdiff_ser(x, y):
    #return ((x-y) / x).mean()
    return (x-y).mean()

sdump = []
keylist = ['block2_zap1', 'block4_zap1', 'block2_nogo_zap1', 'block4_nogo_zap1']
for key in keylist:
    # Subslice sstats on key
    key_stats = sstats2[key].copy()
    
    # Ratname
    key_stats['ratname'] = map(lambda s: s[:5], sstats.index)
    
    # Plot each rat
    for ratname, rat_stats in key_stats.groupby('ratname'):
        # P-value on rat
        n_lt = np.sum(rat_stats['ctl_m'] > rat_stats['zap_m'])
        nnn = len(rat_stats)
        pval = scipy.stats.binom_test(n_lt, nnn)
        
        # Get shrot name
        shorname = short_ratname[ratname]
        ratnum = int(shorname[1])
    
        # Subrat stats
        sdump.append(" ".join((key, shorname, ratname, '%0.4f' % normdiff_ser(
            rat_stats['ctl_m'], rat_stats['zap_m']), \
            'p=%0.3f' % pval, \
            '%d/%d sessions' % (n_lt, nnn))))
sdump = "\n".join(sdump)
print sdump
with file('stat__zap_by_block_and_gonogo', 'w') as fi:
    fi.write(sdump)









if 'all trials, by block' in plots_to_make:
    # Which trial groups
    keylist_l = ['block2_zap1', 'block4_zap1']
    
    # Label the trial groups
    labels_l = ['localization', 'pitch discrimination']

    # Plot each key in keylist
    sdump = []
    f, axa = plt.subplots(1, len(keylist_l), figsize=(7, 3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475)
    for key, ax, label in zip(keylist_l, axa.flatten(), labels_l):
        # Subslice sstats on key
        key_stats = sstats2[key].copy()
        
        # Significance, assessed with mann-whitney and with Fisher
        key_stats['signif_mw'] = key_stats['pmw'] < .05
        key_stats['signif_fish'] = key_stats['pfish'] < .05
        
        # Ratname
        key_stats['ratname'] = map(lambda s: s[:5], sstats.index)
        
        # Plot each rat
        for ratname, rat_stats in key_stats.groupby('ratname'):
            color = ratname2col[ratname]
            
            # Separately plot indiv sig sessions
            for is_sig, subrat_stats in rat_stats.groupby('signif_fish'):
                if is_sig:
                    ax.plot(subrat_stats['ctl_m'], subrat_stats['zap_m'], '+',
                        color=color, mew=2, mfc='none', mec=color, ms=8)
                else:
                    ax.plot(subrat_stats['ctl_m'], subrat_stats['zap_m'], 'o',
                        color=color, mew=1, mfc='none', mec=color)#, ms=4)
            
            # P-value on rat (sign test on which sessions are impaired)
            n_lt = np.sum(rat_stats['ctl_m'] > rat_stats['zap_m'])
            nnn = len(rat_stats)
            pval = scipy.stats.binom_test(n_lt, nnn)
            
            # Get shrot name
            shorname = short_ratname[ratname]
            ratnum = int(shorname[1])
        
            # Legend
            ax.text(.05, 1.0 - ratnum * .1,
                '%s n=%d' % (short_ratname[ratname], nnn),
                color=ratname2col[ratname])
            
            # Subrat stats
            sdump.append(" ".join((key, ratname, '%0.4f' % normdiff_ser(
                rat_stats['ctl_m'], rat_stats['zap_m']), 'p=%0.3f' % pval)))
        
        # Unity line
        ax.plot([0, 1], [0, 1], 'k-')
        ax.plot([0, 1], [.5, .5], 'k:')
        ax.axis('scaled')
        ax.set_ylim([0, 1]); ax.set_xlim([0, 1])
        ax.set_ylabel('zap performance')
        ax.set_xlabel('control performance')
        ax.set_xticks((0, .25, .5, .75, 1.0))
        ax.set_yticks((0, .25, .5, .75, 1.0))
        ax.set_title(label)

        
    # These are already handled in text summary
    #~ sdump = "\n".join(sdump)
    #~ print sdump
    #~ with file('stat__zap_by_block_v2', 'w') as fi:
        #~ fi.write(sdump)

    f.savefig('all trials by block.svg')


## SPLIT BY RAT

if 'split by rat and trial type, conn pairs' in plots_to_make:
    # What plots to make
    keylist_l = ['block2_go_zap1', 'block2_nogo_zap1',  
        'block4_go_zap1', 'block4_nogo_zap1']
    ratname_l = ['CR25A', 'CR28A', 'CR28B']

    # Plot each key in keylist
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    f.subplots_adjust(left=.125, right=.95, wspace=.475, bottom=.2)
    for nrat, ratname in enumerate(ratname_l):
        # Subslice on this rat
        rat_stats = sstats.ix[map(lambda s: s[:5] == ratname, sstats.index)]
        color = ratname2col[ratname]        
        
        # Plot each key group
        for nkey, key in enumerate(keylist_l):
            # For each session
            cdata = rat_stats['ctl_m'][key]
            zdata = rat_stats['zap_m'][key]
            pvals = rat_stats['pfish'][key]

            # We plot grouped by key first, then rat
            xvals = [nkey*8 + nrat*2, nkey*8 + nrat*2 + 1]
            
            # Connected pairs
            for cc, zz, p in zip(cdata.values, zdata.values, pvals.values):
                if p < .05:
                    ax.plot(xvals, [cc, zz], '-', 
                        color=color, marker='', mfc='none')
                    ax.plot(xvals[1], [zz], marker='+', color='k')
                else:
                    ax.plot(xvals, [cc, zz], '-', 
                        color=color, marker='', mfc='none')
                    ax.plot(xvals[1], [zz], marker='o', mfc='none', mec='k')
            
            # Sign test
            pval = scipy.stats.binom_test(x=np.sum(cdata>zdata), n=len(cdata))
            
            if pval < .05:
                ax.text(np.mean(xvals), 1.0, '*', size=18, color='k', 
                    ha='center', va='center')
            
    my.plot.despine(ax)
    ax.set_ylim([0, 1.02])
    ax.set_xlim((-1, 10 * len(ratname_l)))
    ax.set_xticks(0.5 + 2*np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]))
    ax.set_xticklabels([
        'Z1', 'Z2\n\nLocalization\nGO', 'Z3',
        'Z1', 'Z2\n\nLocalization\nNOGO', 'Z3',
        'Z1', 'Z2\n\nPitch Disc.\nGO', 'Z3',
        'Z1', 'Z2\n\nPitch Disc.\nNOGO', 'Z3',
        ])
    ax.set_ylabel('performance')
    f.savefig('split by rat and type, connected pairs.svg')



if 'bar of impairment' in plots_to_make:
    # What plots to make
    keylist_l = ['block2_go_zap1', 'block2_nogo_zap1',  
        'block4_go_zap1', 'block4_nogo_zap1']
    ratname_l = ['CR25A', 'CR28A', 'CR28B']

    # Plot each key in keylist
    f, ax = plt.subplots(1, 1, figsize=(7, 3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475, bottom=.25)
    xts = []
    for nrat, ratname in enumerate(ratname_l):
        # Subslice on this rat
        rat_stats = sstats.ix[map(lambda s: s[:5] == ratname, sstats.index)]
        color = ratname2col[ratname]        
        
        # Plot each key group
        for nkey, key in enumerate(keylist_l):
            # For each session
            cdata = rat_stats['ctl_m'][key]
            zdata = rat_stats['zap_m'][key]

            # We plot grouped by key first, then rat
            xval = nrat*5 + nkey +.15
            xts.append(xval)
            
            # Connected pairs
            ax.bar([xval], height=[(cdata-zdata).mean()],
                yerr=[my.misc.sem(cdata - zdata)],
                color=color, edgecolor='k', lw=1,
                capsize=0, align='center', ecolor='k')
            
            # Sign test the difference
            pval = scipy.stats.binom_test(x=np.sum(cdata>zdata), n=len(cdata))
            
            if pval < .05:
                ax.text(xval, .8, '*', size=18, color='k', 
                    ha='center', va='center')
            
    my.plot.despine(ax, which=('right', 'top', 'bottom'))
    ax.set_ylim([-.1, 1])
    ax.set_xlim((-.5, 14))
    ax.plot(ax.get_xlim(), [0, 0], 'k-')
    ax.set_xticks(xts)

    # First L, then PD, but no room on xlabels
    ax.set_xticklabels(3 * ['go', 'nogo', 'go', 'nogo'])
    ax.set_ylabel('impairment')
    f.savefig('split by rat and type, bar of impairment.svg')

if 'example sessions' in plots_to_make:
    # What plots to make
    keylist_l = ['block2_go_zap1', 'block2_nogo_zap1',  
        'block4_go_zap1', 'block4_nogo_zap1']

    example_sessions = [
        'CR25A_131018_001_behaving',
        'CR28A_131020_001_behaving',
        'CR28B_131018_001_behaving',
        ]
    

    # Plot each key in keylist
    f, ax = plt.subplots(1, 1, figsize=(7, 3.75))
    f.subplots_adjust(left=.125, right=.95, wspace=.475, bottom=.5)
    xts = []
    for nsess, session in enumerate(example_sessions):
        # Subslice on this session
        sess_stats = sstats.ix[session]
        ratname = session[:5]
        color = ratname2col[ratname]        
        
        # Plot each key group
        for nkey, key in enumerate(keylist_l):
            # We plot grouped by session first, then key
            xvals = [nsess * 10 + nkey * 2, nsess*10 + nkey * 2 + .7]
            xts.append(np.mean(xvals))

            # Heights
            heights = np.asarray([
                sstats['ctl_m'][key][session],
                sstats['zap_m'][key][session]])
            
            # E
            yerrs = np.asarray([[
                sstats['ctl_l'][key][session],
                sstats['ctl_h'][key][session],],
                [sstats['zap_l'][key][session],
                sstats['zap_h'][key][session],]]).T
            yerr2 = np.array([heights - yerrs[0], yerrs[1] - heights])

            ax.bar(xvals, height=heights, yerr=yerr2,
                color=[color, 'w'], edgecolor=['none', color], lw=2,
                capsize=0, align='center', ecolor='k', width=.5)
            
            pval = sstats['pfish'][key][session]
            
            if pval < .001:
                ax.text(np.mean(xvals), 1.05, '***', size=18, color='k', 
                    ha='center', va='center')     
            elif pval < .01:
                ax.text(np.mean(xvals), 1.05, '**', size=18, color='k', 
                    ha='center', va='center')     
            elif pval < .05:
                ax.text(np.mean(xvals), 1.05, '*', size=18, color='k', 
                    ha='center', va='center')
            
    my.plot.despine(ax, which=('right', 'top'))#, 'bottom'))
    ax.set_ylim([0, 1.0])
    ax.set_xlim((-1, 28))
    #~ ax.plot(ax.get_xlim(), [0, 0], 'k-')
    ax.set_xticks(xts)
    
    # L, then PD, but no room on xlabels
    ax.set_xticklabels(3 * ['go', 'nogo', 'go', 'nogo'])
    ax.set_ylabel('performance')
    f.savefig('example sessions.svg')


if 'indiv sname bar of impairment' in plots_to_make:
    # What plots to make
    keylist_l = [
        'stim5_zap1', 'stim6_zap1', 'stim7_zap1', 'stim8_zap1',
        'stim9_zap1', 'stim10_zap1', 'stim11_zap1', 'stim12_zap1',]
    ratname_l = ['CR25A', 'CR28A', 'CR28B']

    # Plot each key in keylist
    f, ax = plt.subplots(1, 1, figsize=(7, 3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475, bottom=.25)
    for nrat, ratname in enumerate(ratname_l):
        # Subslice on this rat
        rat_stats = sstats.ix[map(lambda s: s[:5] == ratname, sstats.index)]
        color = ratname2col[ratname]        
        
        # Plot each key group
        for nkey, key in enumerate(keylist_l):
            # For each session
            cdata = rat_stats['ctl_m'][key]
            zdata = rat_stats['zap_m'][key]

            # We plot grouped by key first, then rat
            xval = nkey*(len(ratname_l) + 1) + nrat
            
            # Offset the localization snames
            if nkey >= 4:
                xval += 5
            
            ax.bar([xval], height=[(cdata-zdata).mean()],
                yerr=[my.misc.sem(cdata - zdata)],
                color=color, edgecolor='k', lw=1,
                capsize=0, align='center', ecolor='k')
            
            
            # Sign test the difference
            pval = scipy.stats.binom_test(x=np.sum(cdata>zdata), n=len(cdata))
            
            if pval < .05:
                ax.text(xval, .8, '*', size=18, color='k', 
                    ha='center', va='center')
            
    my.plot.despine(ax)
    ax.set_ylim([-.1, 1])
    ax.set_xlim((-1, len(keylist_l) * (len(ratname_l) + 1) + 5))
    ax.plot(ax.get_xlim(), [0, 0], 'k-')
    
    ax.set_xticks([n for n in range(32) if not np.mod(n, 4) == 3])
    ax.set_xticks([1, 5, 9, 13, 22, 26, 30, 34])
    ax.set_xticklabels([
        'LEFT\nHIGH\npitch\ndisc',
        'RIGHT\nHIGH\npitch\ndisc',
        'LEFT\nLOW\npitch\ndisc',
        'RIGHT\nLOW\npitch\ndisc',
        'LEFT\nHIGH\nlocal',
        'RIGHT\nHIGH\nlocal',
        'LEFT\nLOW\nlocal',
        'RIGHT\nLOW\nlocal',        
        ])
    ax.set_ylabel('impairment')
    f.savefig('indiv sname impairment.svg')





plt.show()

