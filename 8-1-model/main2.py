# Plot gain needed to reach criterion performance
import numpy as np
import my, my.plot
import matplotlib.pyplot as plt
from matplotlib import mlab
import pandas, itertools
import model
import os.path
import glob

my.plot.publication_defaults()
my.plot.font_embed()

# Results of simulation
# These are the links to the large simulation used in the paper results
# Replace the filename here with the newer simulation file if you rerun it
data_dir = '../data/miscellaneous/model'
filename = '20131121173602'

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
## END BUG CHECKING GAINS


# Pivot and prepare to average over nreps (by putting them on the columns)
# Divide by 4*NT_TEST since NT_TEST examples of each stimulus
# This computation takes a while!
# Note that we COMBINE the broad and narrow granges here
# Later, we will index into gains_l in order to recover the actual gain
pivdf3 = resdf.pivot_table(
    rows=['N', 'n_noise_level', 'ngain'],
    cols=['nrep'], values=['score1b']) / 4. / NT_TEST

# Take median over nreps (columns)
# Ignore the low and high CIs
meaned = pivdf3.median(axis=1)

# Find attn gain necessary to reach criterion
criterion = .8


# First we unstack the gain, which, since we've combined broad and narrow,
# will have 160 columns (gain levels).
# Then, for each row, find the gain (column index) at which performance 
# crosses criterion.
# Finally unstack noise_level to make it shaped like the plot we want
cross_point_m = meaned.unstack('ngain').apply(
    lambda arr: np.where(arr > criterion)[0][0] 
    if np.any(arr > criterion) else None, 
    axis=1).unstack('n_noise_level')

# Also test where it falls below criterion again
cross_point_m2 = meaned.unstack('ngain').apply(
    lambda arr: np.where(arr > criterion)[0][-1] 
    if np.any(arr > criterion) else None, 
    axis=1).unstack('n_noise_level')

# Concatenate all the cross points
cross_point = pandas.concat(
    [cross_point_m, cross_point_m2],#, cross_point_l, cross_point_h], 
    axis=1, keys=['m', 'm2'])#, 'l', 'h'])

# Use cross_point to index back into gains_l
cross_point_val = cross_point.applymap(
    lambda idx: gains_l[int(idx)] if not np.isnan(idx) else np.nan)
cross_point_val.index = cross_point.index
cross_point_val.columns = cross_point.columns



# Make a heatmap version of the above
f, ax = plt.subplots(figsize=(5, 5))
f.subplots_adjust(right=.8)
im = my.plot.imshow(np.log10(cross_point_val['m'].divide(noise_level_l, axis=1).T), 
    cmap=plt.cm.hot, ax=ax)
my.plot.harmonize_clim_in_subplots(fig=f, clim=(-1.0, 3))
ax.set_yticks(range(len(noise_level_l)))
ax.set_yticklabels(['2^-%d' % my.rint(np.log2(val)) for val in noise_level_l])
ax.set_xticks(range(len(N_l)))
ax.set_xticklabels([str(int(val)) for val in N_l[::-1]])
ax.set_xlabel('network size')
ax.set_ylabel('sensory signal-to-noise ratio')

# Colorbar
cbar_ax = f.add_axes([.85, .15, .05, .7])
cb = f.colorbar(im, cax=cbar_ax)
cb.set_ticks((-1, 0, 1, 2, 3))
cb.set_ticklabels((.1, 1, 10, 100, 1000))

f.savefig('parameter space 1.svg')
plt.show()




# Make a heatmap version of the above
f, ax = plt.subplots(figsize=(5, 5))
f.subplots_adjust(right=.8)
im = my.plot.imshow(np.log10(cross_point_val['m2'].divide(noise_level_l, axis=1).T), 
    cmap=plt.cm.hot, ax=ax)
my.plot.harmonize_clim_in_subplots(fig=f, clim=(-2.0, 3))
ax.set_yticks(range(len(noise_level_l)))
ax.set_yticklabels(['2^-%d' % my.rint(np.log2(val)) for val in noise_level_l])
ax.set_xticks(range(len(N_l)))
ax.set_xticklabels([str(int(val)) for val in N_l[::-1]])
ax.set_xlabel('network size')
ax.set_ylabel('sensory signal-to-noise ratio')

# Colorbar
cbar_ax = f.add_axes([.85, .15, .05, .7])
cb = f.colorbar(im, cax=cbar_ax)
cb.set_ticks((-1, 0, 1, 2, 3))
cb.set_ticklabels((.1, 1, 10, 100, 1000))


f.savefig('parameter space 2.svg')
plt.show()

