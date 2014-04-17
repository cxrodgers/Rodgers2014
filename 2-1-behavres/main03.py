# Example trace for one rat, averaged performance over sessions,
# showing distinct behavioral responses to same stimulus

import numpy as np
from kkpandas.kkrs import unit2unum, session2rs, session2kk_server
import kkpandas
from kkpandas.utility import panda_pick, panda_pick_data
from ns5_process import LBPB
import matplotlib.pyplot as plt
import pandas, os.path
import my.plot, my, my.dataload

my.plot.font_embed()
my.plot.publication_defaults()

# Load results from main
trials_info_dd = my.misc.pickle_load('trials_info_dd')

# only plot trials up to now
DROP_X_AFTER = 700


# Plot kwargs
BLACK_SQUARE = {'marker': 's', 'mfc': 'none', 'ms': 5, 'mec': 'k', 'mew': 1}
LEGEND_LOC = 'above plot'

# Consistent figure sizing
figsize = (7, 3)
half_figsize = (3.5, 3)
bbox_to_anchor = None
subplots_adjust_kwargs = {'bottom':.2, 'right':.95}
half_subplots_adjust_kwargs = {'left':.2, 'bottom':.25, 'right':.95, 'top':.95}

def block_trace_is_munged(block_trace):
    """Given a Series of block number, indexed by trial number, check if munged"""
    # These are what the trialnumbers should go to after modding by 160
    blocknum2acceptable_mods = {
        1 : list(range(1, 21)),
        2 : list(range(21, 81)),
        3 : list(range(81, 101)),        
        4 : [0] + list(range(101, 160)),}
    
    # check each block
    munged = False
    for blocknum, acceptable_mods in blocknum2acceptable_mods.items():
        block_trace_slc = block_trace[block_trace == blocknum]
        achieved_mods = np.mod(block_trace_slc.index, 160)
        if not np.all(np.in1d(achieved_mods, acceptable_mods)):
            munged = True
    
    return munged

# extract sessions to analyze
gets = my.dataload.getstarted()
session_name_l = gets['session_list']

# optionally filter by ratname
session_name_l = filter(lambda s: 'CR21A' in s, session_name_l)

# extract trials info from each
trials_info_d = {}
for session_name in session_name_l:
    #~ rs = my.dataload.session2rs(session_name)#, kk_servers, data_dirs)
    #~ trials_info = kkpandas.io.load_trials_info(rs.full_path)
    
    # Load trials info
    ratname = kkpandas.kkrs.ulabel2ratname(session_name)
    trials_info = trials_info_dd[ratname][session_name]
    
    # Store
    trials_info_d[session_name] = trials_info
    
    # Check if munged
    if block_trace_is_munged(trials_info.block):
        1/0


# Use stimulus RIGHT+LOW and average performance over sessions
stimuli = ['ri_lo_lc', 'ri_lo_pc']
binwidth = 10
bins = np.arange(1, 2000, binwidth, dtype=np.int)
outcomes = ['hit', 'error', 'wrong_port', 'future_trial']
perf_hist_d = {}
for stimulus in stimuli:
    perf_hist_d[stimulus] = pandas.DataFrame(index=bins[:-1], columns=outcomes, 
        data=np.zeros((len(bins) - 1, len(outcomes)), dtype=np.int))

# Iterate over sessions * stimuli and count outcomes in each session
for ti in trials_info_d.values():
    for stimulus, perf_hist in perf_hist_d.items():
        # Pick the trials with this stimulus
        picked = panda_pick_data(ti, stim_name=stimulus, nonrandom=0)
        assert picked.index.max() <= bins.max()
        
        # Now histogram the results by trial
        # We also make sure that the numbers add up
        counts_check = 0
        for outcome in outcomes:
            trial_numbers = panda_pick(picked, outcome=outcome)
            counts, edges = np.histogram(trial_numbers, bins=bins)
            perf_hist[outcome] += counts
            counts_check += counts.sum()
        assert counts_check == len(picked)

# Norm
nperf_hist_d = {}
for stimulus in perf_hist_d.keys():
    non_future = perf_hist_d[stimulus].drop(['future_trial'], axis=1)
    col = non_future.sum(axis=1).astype(np.float)
    nperf_hist_d[stimulus] = perf_hist_d[stimulus].divide(col, axis=0)


# Extract traces to plot
LB_GO_L = nperf_hist_d['ri_lo_lc']['error'] * 100
LB_GO_R = nperf_hist_d['ri_lo_lc']['wrong_port'] * 100
LB_NOGO = nperf_hist_d['ri_lo_lc']['hit'] * 100
PB_GO_L = nperf_hist_d['ri_lo_pc']['wrong_port'] * 100
PB_GO_R = nperf_hist_d['ri_lo_pc']['hit'] * 100
PB_NOGO = nperf_hist_d['ri_lo_pc']['error'] * 100

# Plot
f = plt.figure(figsize=figsize); ax = f.add_subplot(111)
f.subplots_adjust(**subplots_adjust_kwargs)
x = np.asarray(LB_GO_L.index) + binwidth/2. - 1
if DROP_X_AFTER:
    x[x>DROP_X_AFTER] = np.nan

# Plot traces in LB, coloring by action and marking by outcome (NOGO=hit)
ax.plot(x, LB_GO_L, color='b', lw=1, label='GO LEFT')
ax.plot(x, LB_NOGO, color='gray', lw=1, label='NOGO', **BLACK_SQUARE)
ax.plot(x, LB_GO_R, color='r', lw=1, label='GO RIGHT')

# Plot traces in PB, coloring by action and marking by outcome (GO R = hit)
ax.plot(x, PB_GO_L, color='b', lw=1, marker=None, markerfacecolor='k', 
    markeredgecolor='k', mew=1)
ax.plot(x, PB_GO_R, color='r', lw=1, **BLACK_SQUARE)
ax.plot(x, PB_NOGO, color='gray', lw=1, marker=None, markerfacecolor='k', 
    markeredgecolor='k', mew=1)


# Vertical divisions between blocks
for tnum in range(0, 641, 80):
    l1, = ax.plot([tnum, tnum], [0, 100], 'k-', lw=1)
    #l2, = ax.plot([tnum+20, tnum+20], [0, 1], 'k-')
    
    # Shading in cue
    if np.mod(tnum, 160) == 0:
        color = 'SkyBlue'
    else:
        color = 'pink'
    ax.fill_betweenx(y=[0, 100], x1=tnum, x2=tnum+19,
        color=color, edgecolor=color)    


# Xticks on the block boundaries
ax.set_xticks(range(0, 700, 80))
ax.tick_params(bottom=False, top=False)
ax.set_xlim((0, 700))
ax.set_ylabel('Probability')

# Make room for double x-axis
ax.set_yticks((0, 20, 40, 60, 80, 100))
ax.set_yticklabels((0, 20, 40, 60, 80, 100))
ax.set_yticklabels((0, .20, .40, .60, .80, 1))

# block labels
for xt in range(40, 700, 80):
    if np.mod(xt, 160) == 40:
        ax.text(xt, -18, 'local.', ha='center', va='center')
    else:
        ax.text(xt, -18, 'pitch\ndisc.', ha='center', va='center')

# double-label x-axis
ax.text(-20, -5.5, 'trial #', ha='right', va='center')
ax.text(-20, -18, 'block', ha='right', va='center')


# Legendary
if LEGEND_LOC == 'in plot':
    ax.text(600, 60, 'go right', color='r', size='medium', ha='center', va='center')
    ax.text(600, 50, 'nogo', color='gray', size='medium', ha='center', va='center')
    ax.text(600, 40, 'go left', color='b', size='medium', ha='center', va='center')
    ax.text(600, 70, 'hit', color='k', size='medium', ha='center', va='center')
    ax.plot([579], [70], **BLACK_SQUARE)
elif LEGEND_LOC == 'above plot':
    lkwargs = {'ha': 'center', 'va': 'bottom', 'size': 12}
    #~ ax.text(25, 101, 'hit', color='k', **lkwargs)
    ax.text(110, 100, 'go right', color='r', **lkwargs)
    ax.text(200, 100, 'nogo', color='gray', **lkwargs)
    ax.text(280, 100, 'go left', color='b', **lkwargs)
    
    # Extra one to be dragged to legend in AI
    #ax.plot([120], [50], **BLACK_SQUARE)
    #ax.set_ylim((0, 1))


# Save file
plt.show()
#f.patch.set_visible(False)
plt.savefig('rilo_over_session.svg')


