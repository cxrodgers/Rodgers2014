# Generate plot of performance in both blocks of every session
# This script can also make the plot of GO v NOGO performance

import my.plot
import numpy as np
import my, my.plot, my.dataload
import matplotlib.pyplot as plt
from perf import perfcount, perfratio

# Load results from main
trials_info_dd = my.misc.pickle_load('trials_info_dd')

# Defaults for paper figures
my.plot.font_embed()
my.plot.publication_defaults()

# Ratnames / numbers
gets = my.dataload.getstarted()
ratname2num = gets['ratname2num']
ratnamelist = list(ratname2num.index)
ratnumlist = list(ratname2num.values)

# Consistent panel sizes throughout this figure
figsize = (7, 3)
half_figsize = (3.5, 3)
bbox_to_anchor = None
subplots_adjust_kwargs = {'bottom':.2, 'right':.95}
half_subplots_adjust_kwargs = {'left':.2, 'bottom':.25, 'right':.95, 'top':.95}

# First plot: Performance overall is above chance for both LB and PB in all rats
f, ax = plt.subplots(figsize=figsize)
for nrat, ratname in enumerate(ratnamelist):
    trials_info_d = trials_info_dd[ratname]
    # Get the performances
    LB_perf = np.array([perfratio(ti, block=2)[0] 
        for ti in trials_info_d.values()])
    PB_perf = np.array([perfratio(ti, block=4)[0] 
        for ti in trials_info_d.values()])
    
    # Individual session points
    for lbp, pbp in zip(LB_perf, PB_perf):
        # Each point
        ax.plot([nrat*3], [lbp], marker='_', mec='b', mfc='k', ms=12)
        ax.plot([nrat*3 + 1], [pbp], marker='_', mec='r', mfc='k', ms=12)

f.subplots_adjust(**subplots_adjust_kwargs)

# Chance line
chance_line, = ax.plot([-.75, (len(ratnamelist)-1)*3+1.75], [50, 50], 'k:')
ax.text(8, 48, 'chance', va='top', ha='center')

# Labels, legends
ax.text(4, 95, 'localization', color='#0000FF', va='center', ha='center')
ax.text(4, 88, 'pitch discrimination', color='r', va='center', ha='center')
ax.set_ylim((40, 100))
ax.set_ylabel('Performance')
ax.set_xlim((-1, (len(ratnamelist) - 1)*3 + 2 ))
ax.set_xticks(np.array([nrat * 3 + 0.5 for nrat in range(len(ratnamelist))]))
ax.set_xticklabels(ratnumlist, rotation=0)
ax.set_yticklabels((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


plt.show()

#f.patch.set_visible(False)
plt.savefig('perf_by_session_and_block.svg')


# Panel B. Average performance on LB and PB, GO and NOGO, is above chance
kwarg_list = [
    {'block': 2, 'go_or_nogo': 'go'},
    {'block': 2, 'go_or_nogo': 'nogo'},
    {'block': 4, 'go_or_nogo': 'go'},
    {'block': 4, 'go_or_nogo': 'nogo'}]
kwarg_labels = [
    'Local.\nGO', 'Local.\nNOGO', 
    'Pitch\nDisc.\nGO', 'Pitch\nDisc.\nNOGO']
meanperf, stdperf, minperf, maxperf, stderrperf = [], [], [], [], []
#~ ratnamelist = ['CR12B', 'YT6A', 'CR17B', 'CR20B', 'CR21A', 'CR24A'][::-1]

for ratname in ratnamelist:
    # Get the list of trials_info over sessions
    trials_info_d = trials_info_dd[ratname]
    
    # Store mean and stdev of perf on each category
    for kwargs in kwarg_list:
        data = [perfratio(ti, **kwargs)[0] for ti in trials_info_d.values()]
        meanperf.append(np.mean(data))
        stdperf.append(np.std(data))
        minperf.append(np.min(data))
        maxperf.append(np.max(data))
        stderrperf.append(np.std(data) / np.sqrt(len(data)))
meanperf = np.reshape(meanperf, (len(ratnamelist), 4)) # rat = row
stdperf = np.reshape(stdperf, (len(ratnamelist), 4)) # rat = row
minperf = np.reshape(minperf, (len(ratnamelist), 4))
maxperf = np.reshape(maxperf, (len(ratnamelist), 4))
stderrperf = np.reshape(stderrperf, (len(ratnamelist), 4))


# Line plot version
f, ax = plt.subplots(figsize=half_figsize)
f.subplots_adjust(**subplots_adjust_kwargs)
x = np.arange(len(kwarg_labels), dtype=np.int)
shrinkage = 1/15.
for nrat, ratname in enumerate(ratnamelist):
    yerr = stderrperf[nrat]
    ax.errorbar(x + nrat * shrinkage - len(ratnamelist)*shrinkage/2, meanperf[nrat], 
        yerr=yerr, label=ratname, capsize=0)
#ax.plot([-1, len(kwarg_labels)], [.5, .5], 'k--', label='Chance')
ax.set_xticks(x)
ax.set_xticklabels(kwarg_labels)
ax.set_xlim((x[0]-.5, x[-1]+.5))
ax.set_ylim((40, 100))
ax.set_ylabel('Performance')
ax.set_xlabel('Stimulus Type')
f.subplots_adjust(**half_subplots_adjust_kwargs)
#plt.legend(title='Ratname', bbox_to_anchor=bbox_to_anchor, prop={'size': 'medium'})

ratcolorlist = [l.get_color() for l in ax.lines][:len(ratnamelist)]
for nrat, (ratname, ratcolor) in enumerate(zip(ratnumlist, ratcolorlist)):
    ax.text(-.4, 70-nrat*5, ratname, color=ratcolor, ha='left', va='center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)
plt.show()
f.patch.set_visible(False)
f.savefig('perf_by_stimtype_line.svg')