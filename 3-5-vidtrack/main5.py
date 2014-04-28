# Plots on posture-equalization
import pandas
import numpy as np
import matplotlib.pyplot as plt
import my
import kkpandas.kkrs
import scipy.stats
import random
from matplotlib import mlab
import statsmodels

my.plot.publication_defaults()
my.plot.font_embed()

plots_to_make = [
    'lograt rule-encoding distrs',
    'magnitude comparison bar',
    'indiv connected pairs',
    ]

# Results from main4
sub_res = pandas.load('posture_equalization_results')



def mad_1d(arr1d):
    """My implementation of MAD.
    
    I think there is a bug in statsmodels.robust.mad because the
    median is not subtracted from the data
    """
    deviations = arr1d - np.median(arr1d)
    return np.median(np.abs(deviations)) / .6745

def mad_2d(arr):
    """My implmentation of MAD"""
    return np.array([mad_1d(row) for row in arr])

## Plots
if 'lograt rule-encoding distrs' in plots_to_make:
    # Distributions of block difference in PE, non-PE, and shuffled
    # How to histogram each dataset
    bins = np.linspace(-2, 2, 22)
    shuffled_bins = np.linspace(-2, 2, 40)

    # Make figure
    f, axa = plt.subplots(3, 2, figsize=(7, 6))
    f.subplots_adjust(left=.125, right=.95, wspace=.475, hspace=.6)
    
    sdump = []
    
    # One column per region
    for region, subsubres in sub_res.groupby('region'):
        # Histogram the subbed rule encoding
        ax = axa[0, region == 'PFC']
        ax.hist(np.log10(subsubres.sub_rdiff), histtype='stepfilled', bins=bins,
            color='k')
        #~ ax.set_title('%s: posture-equalized trials' % region)
        ax.set_title('posture-equalized trials')

        # Histogram NON-subbed rule-encoding from this region
        ax = axa[1, region == 'PFC']
        ax.hist(np.log10(subsubres.nsub_rdiff), histtype='stepfilled', bins=bins,
            color='k')
        #~ ax.set_title('%s: all other trials' % region)
        ax.set_title('all other trials')

        # Get the shuffled rule encodings: each row a neuron, each col a shuffle
        faked_rdiffs = np.array(list(subsubres['faked_rdiff'].values))

        # Prepare the faked rule encodings for histogramming
        # Concatenate across shuffles and neurons to produce population estimate
        # Drop infs from the histogram display
        to_hist = np.log10(np.concatenate(faked_rdiffs))
        inf_shuffle_mask = np.isinf(to_hist)
        to_hist = to_hist[~inf_shuffle_mask]
        
        # Histogram the FAKED rule encoding
        counts, edges = np.histogram(to_hist, bins=shuffled_bins,
            weights=np.ones_like(to_hist) / to_hist.size)
        
        # Plot the histogram
        ax = axa[2, region == 'PFC']
        ax.plot(0.5 * (edges[:-1] + edges[1:]), counts, color='k')
        ax.set_title('shuffled')
        
        # Calculate MAD and STD, ie PRE, of the actual rule encoding
        real_mad = statsmodels.robust.stand_mad(np.log10(subsubres.sub_rdiff))
        real_npe_mad = statsmodels.robust.stand_mad(np.log10(subsubres.nsub_rdiff))
        real_std = np.log10(subsubres.sub_rdiff).std()
        real_npe_std = np.log10(subsubres.nsub_rdiff).std()
        
        # Distr of MAD and STD, ie PRE, over shuffles
        mad_over_shuffles = statsmodels.robust.stand_mad(np.log10(faked_rdiffs))
        std_over_shuffles = np.log10(faked_rdiffs).std(axis=0)
        
        # floored 1-tailed p-value of actual std vs faked distr
        std_n_more_extreme = np.sum(std_over_shuffles > real_std) + \
            np.sum(~np.isfinite(std_over_shuffles))
        mad_n_more_extreme = np.sum(mad_over_shuffles > real_mad)
        assert np.sum(~np.isfinite(mad_over_shuffles)) == 0
        std_pval = (std_n_more_extreme + 1) / float(faked_rdiffs.shape[1])
        mad_pval = (mad_n_more_extreme + 1) / float(faked_rdiffs.shape[1])
        
        # Text summary
        sdump.append(region)
        sdump.append("MAD, mean %0.3f, distr over shuffles: %s" % \
            (mad_over_shuffles.mean(), 
            ' ' .join(['%0.3f' % v for v in mlab.prctile(
            mad_over_shuffles, (25, 50, 95, 97.5))])))
        sdump.append("MAD, nP.E. actual=%0.3f, P.E. actual=%0.3f, pval=%0.6f" % (
            real_npe_mad, real_mad, mad_pval))
        sdump.append("STDEV, nanmean %0.3f, distr over shuffles: %s" % \
            (np.nanmean(std_over_shuffles), 
            ' '.join(['%0.3f' % v for v in mlab.prctile(
            std_over_shuffles, (25, 50, 95, 97.5))])))
        sdump.append("STDEV, nP.E. actual=%0.3f, P.E. actual=%0.3f, p=%0.6f" % (
            real_npe_std, real_std, std_pval))
        sdump.append('')
    
    # Pretty
    axa[0, 0].set_yticks((0, 1, 2, 3, 4))
    axa[0, 1].set_yticks((0, 2, 4, 6))
    axa[1, 0].set_yticks((0, 1, 2, 3, 4))
    axa[1, 1].set_yticks((0, 2, 4, 6))
    axa[2, 0].set_yticks((0, .25, 0.5))
    axa[2, 1].set_yticks((0, .25, 0.5))
    
    for ax in f.axes:
        my.plot.despine(ax)
        ax.set_xticks((-2, -1, 0, 1, 2))
        ax.plot([0, 0], ax.get_ylim(), 'r-', lw=2)

    f.savefig('hists.svg')
    
    sdump_s = "\n".join(sdump)
    with file('stat__head_angle_PRE_permutation_test', 'w') as fi:
        fi.write(sdump_s)
    print sdump_s


if 'magnitude comparison bar' in plots_to_make:
    # Iterate over regions
    for region, subsubres in sub_res.groupby('region'):
        # Plot magnitude of rule-encoding as bars, split by prefblock
        f, ax = plt.subplots(1, 1, figsize=(3, 5))
        
        # Store heights, errs, pvals here
        mu_l, sem_l, p_l, labels_l = [], [], [], []
        
        # Iterate over prefblock
        for prefblock, sssres in subsubres.groupby('prefblock'):
            # Get mean and error for this subset
            mu = np.log10(sssres[['sub_rdiff', 'nsub_rdiff']]).mean()
            sem = my.misc.sem(
                np.log10(sssres[['sub_rdiff', 'nsub_rdiff']]), axis=0)
            
            # Test whether they differ
            p = my.stats.r_utest(
                np.log10(sssres.nsub_rdiff.values), 
                np.log10(sssres.sub_rdiff.values),
                paired='TRUE')['p']
            
            # Append to lists
            mu_l += list(mu)
            sem_l += list(sem)
            p_l.append(p)
            labels_l += [
                '%s\n%s\nP.E.' % (region, prefblock), 
                '%s\n%s\nrest' % (region, prefblock), 
                ]
            
        # Plot bars for full and subsampled
        my.plot.vert_bar(
            bar_positions=[0, 1, 3, 4],
            bar_lengths=mu_l, bar_errs=sem_l, 
            bar_colors=['blue', 'orange'] * 2, 
            ax=ax,
            tick_labels_rotation=0,
            plot_bar_ends='')
        ax.set_xticks((0.5, 3.5, ))
        ax.set_xticklabels(['localization\npreferring', 
            'pitch disc.\npreferring',] * 2)
        
        # zero line
        ax.plot(ax.get_xlim(), [0, 0], 'k-')
        
        # sig lines
        for n_p, p in enumerate(p_l):
            sigpos = .75
            ax.plot([n_p * 3, n_p * 3 + 1], [sigpos, sigpos], 'k-')
            if p < 0.05:
                ax.text(n_p * 3 + 0.5, sigpos + .1, '*', ha='center', va='center',
                    fontsize='18', color='k')
            else:
                ax.text(n_p * 3 + 0.5, sigpos + .1, 'ns', ha='center', va='center',
                    color='k')
        
        # legend
        ax.text(4.5, -.85, 'posture-equalized trials', color='blue', ha='right')
        ax.text(4.5, -.95, 'rest of trials', color='orange', ha='right')
        
        # pretty
        ax.set_ylabel('rule encoding: log10(pitch disc. / local)')
        ax.set_ylim((-1, 1))
        ax.plot(ax.get_xlim(), [0, 0], 'r-', lw=2)
        
        f.savefig('%s_magnitude_comparison_bars.svg' % region)


    

if 'indiv connected pairs' in plots_to_make:
    # Plot as connected pairs
    for region, subsubres in sub_res.groupby('region'):
        f, ax = plt.subplots(1, 1, figsize=(3, 5))        
        for ulabel, row in subsubres.iterrows():
            # Note it's backwards
            ax.plot([1, 0], 
                np.log10([row['nsub_rdiff'], row['sub_rdiff']]), 'ko-')

        ax.set_ylabel('rule encoding: log10(pitch disc. / local)')
        ax.set_xlim((-.5, 1.5))
        ax.set_xticks((0, 1))
        ax.set_xticklabels(('posture\nequalized\ntrials', 'rest of\ntrials', ))
        
        if region == 'PFC':
            ax.set_ylim((-1.5, 1.5))
        else:
            ax.set_ylim((-1.5, 1.5))
        
        ax.plot(ax.get_xlim(), [0, 0], 'r-', lw=2)
    
        f.savefig('%s_conn_pairs.svg' % region)


plt.show()