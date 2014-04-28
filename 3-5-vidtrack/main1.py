# Head angle histograms
import vidtrack
import my, my.dataload, kkpandas, os, pandas, numpy as np
import scipy.stats, my.stats
import matplotlib.pyplot as plt
import my.plot

my.plot.publication_defaults()
my.plot.font_embed()

plots_to_make = [
    'angl by session',
    ]

session_names = [
    'CR24A_121022_001_behaving',
    'CR20B_120602_001_behaving',
    'CR17B_110803_001_behaving']

test_res = pandas.load('test_res')
full_data = pandas.load('full_data')

# full_data is indexed by ulabel but we only need one per session
idxs = [[s for s in full_data.index.levels[0] if s.startswith(session_name)][0]
    for session_name in session_names]


if 'angl by session' in plots_to_make:
    #~ f, axa = plt.subplots(1, 3, figsize=(9.5, 3))
    metric = 'angl'
    sdump = ['head angle by block']
    # One row per session
    for idx in idxs:
        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        # Get data for these session
        df = full_data.ix[idx]
        bins = np.linspace(df[metric].min(), df[metric].max(), 30)
        
        # Slice out values for each block
        to_hist = [
            my.pick_rows(df, block=block)[metric].values 
            for block in ['LB', 'PB']]
        
        # Subtract off session mean
        sess_mean = np.mean(np.concatenate(to_hist))
        to_hist = [a - sess_mean for a in to_hist]
        
        sdump.append("Head angle diffs, %s" % idx)
        sdump.append('means by block: ' + str(map(np.mean, to_hist)))
        sdump.append('diff: ' + str(np.diff(map(np.mean, to_hist))))
        
        heights, bbins, patches = ax.hist(
            to_hist, color=['b', 'r'], histtype='step', 
            label=['LB', 'PB'])
        ax.set_ylim((0, np.max(map(np.max, heights)) * 1.1))
    
        ax.set_xticks((-45, -30, -15, 0, 15, 30, 45))
        ax.set_xlabel('head angle (degrees)')
        ax.set_ylabel('number of trials')
        
        my.plot.despine(ax)
        
        f.savefig('angl_by_session_%s.svg' % idx[6:12])

sdump_s = "\n".join(sdump)
with file('stat__head_angle_diffs_by_block', 'w') as fi:
    fi.write(sdump_s)
print sdump_s

plt.show()
