# Plot results from main7

import numpy as np
import my, my.plot
import pandas
import itertools
import matplotlib.pyplot as plt
import kkpandas.kkrs
import os.path

my.plot.publication_defaults()
my.plot.font_embed()

aov_df = pandas.load('aov_df')

# Drop the unit that failed the F-test (see notes in previous)
aov_df = aov_df[aov_df.real_pF < .05]
#~ aov_df = aov_df.drop('CR24A_121022_001_behaving-610')


def stacked_bar_ess(data, axa=None,
    metrics=('aov_essangl', 'aov_esscross', 'aov_essblock'),
    col_list=('black', 'purple', 'orange')):
    """Stacked bar of ESS"""
    # Which metrics and in which order (bottom to top)
    metrics = list(metrics)

    # Iterate over regions
    for region in ['A1', 'PFC']:
        if axa is None:
            f, ax = plt.subplots(figsize=(3, 3))
        else:
            ax = axa[region == 'PFC']
        
        # Break by prefblock
        subdf = my.pick_rows(data, region=region)
        
        # Plot ESS of block and angl
        esss = subdf[metrics].values.T
        
        # To make a stacked bar, do a cumsum
        esss_cumsum = np.vstack([np.zeros_like(esss[0]),
            np.cumsum(esss, axis=0)])

        # And plot each on top of the previous
        left = np.arange(len(subdf)) + 1
        for nrow in range(esss.shape[0]):
            ax.bar(left=left, height=esss[nrow], bottom=esss_cumsum[nrow],
                color=col_list[nrow], align='center')

        # Pretty
        ax.set_xlim((.5, left.max() + .5))
        ax.set_ylim((0, 1))
        ax.set_ylabel('fraction of explainable variance')
        ax.set_xlabel('neuron')
        
        # Line at 50%
        ax.plot(ax.get_xlim(), [.5, .5], '-', lw=2, color='magenta')


def scatter_ess(data, axa=None,
    metrics=('aov_essblock', 'aov_essangl', 'aov_esscross'),
    ):
    metrics = list(metrics)
    
    # Plot each region
    for region in ['A1', 'PFC']:
        # Choose axis
        if axa is None:
            f, ax = plt.subplots(figsize=(3, 3))
        else:
            ax = axa[region == 'PFC']
        
        subdf = my.pick_rows(data, region=region)
        
        # Plot ESS of block and angl
        esss = subdf[metrics].values.T

        # boxplot
        box_shit = ax.boxplot(esss.T, sym='', positions=range(len(esss)),
            widths=.5)
        for key in ['caps', 'fliers', 'whiskers']:
            for handle in box_shit[key]:
                handle.set_visible(False)
                handle.set_color('gray')
        for line in box_shit['medians']:
            line.set_lw(4)
        for line in box_shit['boxes']:
            line.set_color('gray')

        # ESS
        ax.plot(esss, 'x', ms=4, color='gray')

        # Pretty
        ax.set_xticks([0, 1, 2])
        ax.set_xlim((-.5, 2.5))
        ax.set_ylim((0, 1))
        ax.set_yticks((0, .25, .5, .75, 1.0))
        ax.set_ylabel('fraction of explainable variance')
        #~ ax.set_title('region=' + region)

        my.plot.despine(ax)


metrics = ['ess_angl', 'ess_block']
col_list = ['k', 'orange']
scatter_labels = ['Head\nAngle', 'Block']

# Text
sdump = ['FEV of head angle in real and simulated data']
for region in ['A1', 'PFC']:
    subdf = my.pick_rows(aov_df, region=region)
    sdump.append("%s n=%d: median ess_angl: %0.3f, median fake_mean_ess_angl: %0.3f" % (
        region, len(subdf), subdf['real_ess_angl'].median(),
        subdf['fake_mean_ess_angl'].median()))
sdump_s = "\n".join(sdump)
with file('stat__FEV_of_head_angle_in_real_and_sim_data', 'w') as fi:
    fi.write(sdump_s)
print sdump_s

# Stacked bar
f, axa = plt.subplots(2, 2, figsize=(7, 6.5))
f.subplots_adjust(left=.125, right=.95, wspace=.475, bottom=.05, top=.95, hspace=.4)
stacked_bar_ess(aov_df, axa=axa[0], metrics=['real_' + m for m in metrics],
    col_list=col_list)
stacked_bar_ess(aov_df, axa=axa[1], metrics=['fake_mean_' + m for m in metrics],
    col_list=col_list)

# Pretty
for ax in axa.flatten():
    if ax.get_xlim()[1] > 12:
        ax.set_xticks(range(2, 18, 2))
    else:
        ax.set_xticks(range(1, 8, 1))
f.savefig('stacked_bar.svg')
plt.show()


# Scatter
f, axa = plt.subplots(2, 2, figsize=(5, 6.5))
f.subplots_adjust(left=.125, right=.95, wspace=.475, bottom=.05, top=.95, hspace=.4)
scatter_ess(aov_df, axa=axa[0], metrics=['real_' + m for m in metrics])
scatter_ess(aov_df, axa=axa[1], metrics=['fake_mean_' + m for m in metrics])

# Pretty
for ax in axa.flatten():
    ax.set_xticklabels(scatter_labels)
    ax.set_xlim((-.5, 1.5))
f.savefig('scatter.svg')


plt.show()

