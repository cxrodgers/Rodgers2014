# Boxplot indiv sname differences

from ns5_process import myutils, LBPB
import kkpandas
import pandas, os.path
import numpy as np
import my, my.dataload, scipy.stats, my.plot
import matplotlib.pyplot as plt

my.plot.publication_defaults()
my.plot.font_embed()

# Load data
unit_db = my.dataload.getstarted()['unit_db']

# Analyze only included units with audresps
units_to_analyze = my.pick_rows(unit_db, audresp=['good', 'weak', 'sustained'],
    include=True)

# Load evoked responses from previous analysis
evoked_resps = pandas.load('evoked_resps')

# Load hold results
hold_results = pandas.load('../3-1-hold/hold_results')

# Mean the evoked responses
mer = evoked_resps[LBPB.mixed_stimnames].applymap(np.mean)

# Convert to hertz
mer = mer.divide(evoked_resps['evok_dt'], axis=0)

# Join the hold for subtraction
hr = hold_results[['mLB', 'mPB']].divide(hold_results['dt'], axis=0)
mer['hold_LB_hz'] = hr['mLB'].ix[mer.index]
mer['hold_PB_hz'] = hr['mPB'].ix[mer.index]

# Square-root to normalize variance
mer = np.sqrt(mer)

# Split by block
snames = LBPB.mixed_stimnames_noblock






# Top row -- do not take abs
# Bottom row -- take abs
# Within each ax: 
#  without subhold: first A1 4 snames; then PFC 4 snames
#  with subhold: first A1 4 snames; then PFC 4 snames
f, axa = plt.subplots(2, 1, figsize=(12, 10))
f.subplots_adjust(hspace=.6, left=.125, right=.95)

for subhold in [False, True]:
    # Each row is abs or not
    for take_abs in [False, True]:
        # Get the ax
        ax = axa[int(take_abs)]
        
        # First plot A1, then PFC
        for region in ['A1', 'PFC']:
            # Units for this region
            sub_mer = mer.ix[my.pick(units_to_analyze, region=region)]
            
            # Grab the meaned response by sname
            sub_mer_LB = sub_mer[[sname + '_lc' for sname in snames]]
            sub_mer_LB.columns = snames
            sub_mer_PB = sub_mer[[sname + '_pc' for sname in snames]]
            sub_mer_PB.columns = snames

            # Subtract the hold if necessary
            if subhold:
                sub_mer_LB = sub_mer_LB.sub(sub_mer['hold_LB_hz'], axis=0)
                sub_mer_PB = sub_mer_PB.sub(sub_mer['hold_PB_hz'], axis=0)

            # Now actually take diff
            diff = sub_mer_PB - sub_mer_LB

            # Take the absdiff
            if take_abs:
                diff = np.abs(diff)

            # Plot the diff
            if region == 'A1':
                positions = [0, 1, 2, 3]
            else:
                positions = [5, 6, 7, 8]
            
            # Offset the subhold
            if subhold:
                positions = [p+10 for p in positions]
            
            # Do the bloxplot
            bp = ax.boxplot(diff.values, notch=1, whis=2.5, bootstrap=5000,
                positions=positions)
            
            # Pretty it
            for typ, l_l in bp.items():
                for l in l_l:
                    l.set_color('k')
            for l in bp['whiskers']:
                l.set_linestyle('-')
            for l in bp['caps']:
                l.set_visible(False)
            
            # Kruskal
            F, p = scipy.stats.kruskal(*diff.values.T)
            
            # Plot results of kruskal
            SIGHEIGHT = 5.7 if take_abs else 5.4
            if p < .05:
                ax.text(np.mean(positions), SIGHEIGHT, '*')
            else:
                ax.text(np.mean(positions), SIGHEIGHT + .1, 'n.s.', 
                    ha='center', va='bottom')
            ax.plot([np.min(positions), np.max(positions)], [SIGHEIGHT, SIGHEIGHT], 'k-')
            
            # P-value on each group vs 0
            # This only makes sense if not take_abs
            if not take_abs:
                # Save the p-values so that we can correct them
                p_l = [scipy.stats.wilcoxon(diff[sname].values)[1]
                    for sname in snames]
                p_l = my.stats.r_adj_pval(p_l)
                
                # iterate over corrected so we can plot *s
                for nsname, sname in enumerate(snames):
                    p = p_l[nsname]
                    print sname, p
                    if p < .05:
                        ax.plot([positions[nsname]], [0], 'k*')
            
        
        # Zero line and labels
        if take_abs:
            ax.set_ylim((0, 6))
            ax.set_ylabel('absolute value of above')
        else:
            ax.plot([-1,19], [0, 0], 'k:')
            ax.set_ylim((-6, 6))
            ax.set_ylabel('sqrt(pitch disc.) - sqrt(local.)')
        my.plot.despine(ax)
        


    for ax in axa.flatten():
        ax.set_xlim((-1, 19))
        xts = ((0, 1, 2, 3,  5, 6, 7, 8,
            10, 11, 12, 13,  15, 16, 17, 18))
        short_snames = [sname.replace('_', '') for sname in list(snames) * 4]
        long_snames = [LBPB.short2long[sname] for sname in short_snames]
        long_snames = [sname.replace('+', '\n').lower() for sname in long_snames]
        ax.set_xticks(xts)
        ax.set_xticklabels(long_snames)#, rotation=0, size='small')

f.savefig('boxplots_all_indiv_sname.svg')
    
plt.show()
