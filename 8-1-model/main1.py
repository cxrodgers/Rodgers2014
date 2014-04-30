# Performance versus attn, for all params
#
# Conclude --
# 1) The larger N is, the less attentional amplification is required in
#    order to achieve criterion performance, which is generally limited
#    by switching ability. This is largely independent of the SNR in the 
#    sensory neurons.
#
# 2) Exception to above: for a fixed noise level, if the SNR is so low 
#    that the attentional amplification is more than 10x the signal strength,
#    then we move into the overdrive region. The nonspecific attn signal
#    washes out the sensory signal.

import numpy as np
import my, my.plot
import matplotlib.pyplot as plt
from matplotlib import mlab
import pandas, itertools
import model
import glob
import os.path

my.plot.publication_defaults()
my.plot.font_embed()


# Results of simulation
# These are the links to the large simulation used in the paper results
# Replace the filename here with the newer simulation file if you rerun it
data_dir = '../data/miscellaneous/model'
filename = 'simulation'

# Load results
resdf = pandas.load(os.path.join(data_dir, filename + '_resdf'))
d = my.misc.pickle_load(os.path.join(data_dir, filename + '_params.pickle'))

# Parameter arrays that are indexed into
N_l = np.asarray(d['N_l'])
noise_level_l = np.asarray(d['noise_level_l'])
gains_l = np.asarray(d['gains_l'])

# What to divide performance by
NT_TEST = d['NT_TEST']

# Replace nN with N, which is always integer anyway
resdf['N'] = N_l[resdf.nN]


## BUG CHECKING GAINS AND ASSIGNING grange IN resdf
# This is what the gains should have been
ggains = np.concatenate([
    -10**(np.linspace(-3, 3, 40))[::-1], # -1000 to -.001
    10**(np.linspace(-3, 3, 40)), # .001 to 1000
    np.linspace(-20, 20, 80), # -20 to 20
    ])
sorted_ggains = np.sort(ggains)

# Check that we have accurately reconstituted it
assert np.allclose(sorted_ggains, gains_l)

# Slice out the two gain ranges, using argsort
# These are the indexes into gains_l; also, can choose these values in n_gains_l
argsort_ggains = np.argsort(ggains)
undo_sort = np.argsort(argsort_ggains) # magic
range1 = undo_sort[:80] # check: gains_l[range1] is exponentially spaced
range2 = undo_sort[80:]
resdf['grange'] = 'narrow'
resdf['grange'][resdf['ngain'].isin(range1)] = 'broad'
assert resdf[resdf['grange'] == 'narrow']['ngain'].isin(range2).all()
## END OF BUG CHECKING

# We only want the BROAD and the N=320 runs
N = 320
grange = 'broad'
resdf = my.pick_rows(resdf, grange=grange, N=N)

# Pivot and prepare to average over nreps
pivdf2 = resdf.pivot_table(
    rows=['n_noise_level', 'ngain'],
    cols=['nrep'], values=['score1b', 'score2b']) / 4. / NT_TEST

# Median performance over nreps
medians = pivdf2.median(axis=1, level=0)


# To plot the data, we want noise level on the columns and gain on the rows
data = medians['score1b'].unstack('n_noise_level')
data2 = medians['score2b'].unstack('n_noise_level')



# Create figure
# Rows for N
# 4 columns: narrow and broad, score1 and score2
f, axa = plt.subplots(1, 2, figsize=(9, 3))
f.subplots_adjust(left=.125, right=.95, wspace=.75)

# Noise levels to plot
n_noise_levels_to_plot = list(range(len(noise_level_l)))
n_noise_levels_to_plot = np.where(
    (noise_level_l > 3) & (noise_level_l < 65))[0]

# Which n_noise_levels to plot
idxs = my.rint(100 * np.linspace(.4, 1.0, len(n_noise_levels_to_plot)))[::-1]
colors = plt.cm.get_cmap('Blues', 100)(idxs)
colors2= plt.cm.get_cmap('Reds', 100)(idxs)

# Get the gains (relative to signal) using the indexes of data (ngain)
gain_vs_sig = gains_l[data.index]

# Iterate over noise_levels
for nnnl, n_noise_level in enumerate(n_noise_levels_to_plot): #data['m'].columns:
    # Get the noise level, and gain_vs_noise
    noise_level = noise_level_l[n_noise_level]
    x_to_plot = gain_vs_sig / noise_level
    
    # Plot this trace
    y_to_plot = data[n_noise_level].copy()
    axa[0].plot(x_to_plot, y_to_plot,
        color=colors[nnnl])
    
    # Plot this trace
    y_to_plot = data2[n_noise_level].copy()
    axa[1].plot(-x_to_plot, y_to_plot,
        color=colors2[nnnl])
    
    # Legend the SNR
    lognoise = my.rint(np.log2(noise_level_l[n_noise_level]))
    axa[0].text(-5, 1.0 - nnnl * .1, '2^-%d' % lognoise, color=colors[nnnl],
        ha='center', va='center')
    axa[1].text(5, 1.0 - nnnl * .1, '2^-%d' % lognoise, color=colors2[nnnl],
        ha='center', va='center')
    
axa[0].set_ylabel('prob(correct for task 1)')    
axa[1].set_ylabel('prob(correct for task 2)')
axa[0].set_xlabel('strength of task 1 signal')    
axa[1].set_xlabel('strength of task 2 signal')    


# Axis params
for ax in axa:
    ax.set_xlim((-1, 10))
    ax.set_ylim((0, 1.01))
    ax.set_xscale('symlog')
    my.plot.despine(ax)

my.plot.rescue_tick(x=5, y=None, f=f)

f.savefig('perf vs task switch.svg')
plt.show()

