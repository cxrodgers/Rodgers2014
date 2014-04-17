# Hold period population plots
import numpy as np
import matplotlib.pyplot as plt
from ns5_process import myutils
import kkpandas, pandas, os.path
from kkpandas import kkrs
import my, my.plot, my.dataload
import scipy.stats

my.plot.font_embed()
my.plot.publication_defaults()

# Params
suppress_crucifix_arms = False
savefigs = True

plt.close('all')
plots_to_make = [
    'text summary',
    'linear crucifix',
    'log crucifix',
    'ratio histogram',
    'difference histogram',
    ]

regions_to_plot = ['A1', 'PFC']


# Load data
res = pandas.load('hold_results')
gets = my.dataload.getstarted()
udb = gets['unit_db']

# Insert audresp
res['audresp'] = udb['audresp']

# Insert ratio and diference metrics
res['diffHz'] = (res['mPB'] - res['mLB']) / res['dt']
res['ratio'] = res['mPB'] / res['mLB']
res['lograt'] = np.log10(res['ratio'])

# Split on brain region
a1_units = res[res.region == 'A1']
pfc_units = res[res.region == 'PFC']
region2subdf = {'A1': a1_units, 'PFC': pfc_units}

if 'text summary' in plots_to_make:
    sdump = []
    # Print nsig
    for region, subdf in region2subdf.items():
        sdump.append("%s: n=%d neurons from %d sessions and %d rats" % (
            region, len(subdf), len(subdf.session_name.unique()), 
            len(subdf.ratname.unique())))
        nLB_pref = np.sum((subdf.p_adj<.05)&(subdf.mLB > subdf.mPB))
        nPB_pref = np.sum((subdf.p_adj<.05)&(subdf.mLB < subdf.mPB))
        sdump.append(" %d/%d units show effect, %d LBpref, %dPBpref" % (
            np.sum(subdf.p_adj < .05), len(subdf), nLB_pref, nPB_pref))
        sdump.append(" binomial test on L vs PD: p=%0.3f" % scipy.stats.binom_test(
            [nLB_pref, nPB_pref]))
        
        # Compare audresp and naudresp cells
        subdf2 = subdf[subdf.audresp.isin(['good', 'weak', 'sustained'])]
        subdf3 = subdf[~subdf.audresp.isin(['good', 'weak', 'sustained'])]
        audresp_sig, audresp_total = np.sum(subdf2.p_adj<.05), len(subdf2)
        naudresp_sig, naudresp_total = np.sum(subdf3.p_adj<.05), len(subdf3)
        sdump.append(" audresp - %d/%d ; non-audresp - %d/%d" % (
            audresp_sig, audresp_total, naudresp_sig, naudresp_total))
        sdump.append(" Fisher's Exact on rule-encoding in audresp and naudresp: "\
            "p = %0.4f" % scipy.stats.fisher_exact([
                [audresp_sig, audresp_total - audresp_sig],
                [naudresp_sig, naudresp_total - naudresp_sig]])[1])
    
    # Print effect size
    sigA1 = a1_units[a1_units.p_adj < .05]
    sdump.append("A1: median sig diff %0.3f , median sig ratio %0.3f" % (
        np.median(np.abs(sigA1.mPB - sigA1.mLB) / sigA1.dt),
        np.median(np.max([
            sigA1.mPB / sigA1.mLB, sigA1.mLB / sigA1.mPB], axis=0))))
    sigPFC = pfc_units[pfc_units.p_adj < .05]
    sdump.append("PFC: median sig diff %0.3f , median sig ratio %0.3f" % (
        np.median(np.abs(sigPFC.mPB - sigPFC.mLB) / sigPFC.dt),
        np.median(np.max([
            sigPFC.mPB / sigPFC.mLB, sigPFC.mLB / sigPFC.mPB], axis=0))))

    sigA1 = a1_units
    sdump.append("A1: median diff %0.3f , median ratio %0.3f" % (
        np.median(np.abs(sigA1.mPB - sigA1.mLB) / sigA1.dt),
        np.median(np.max([
            sigA1.mPB / sigA1.mLB, sigA1.mLB / sigA1.mPB], axis=0))))
    sigPFC = pfc_units
    sdump.append("PFC: median diff %0.3f , median ratio %0.3f" % (
        np.median(np.abs(sigPFC.mPB - sigPFC.mLB) / sigPFC.dt),
        np.median(np.max([
            sigPFC.mPB / sigPFC.mLB, sigPFC.mLB / sigPFC.mPB], axis=0))))
    
    sdump2 = '\n'.join(sdump)
    with file('stat__rule_encoding_strength_and_prevalence', 'w') as fi:
        fi.write(sdump2)
        print sdump2



 
# Plots
if 'linear crucifix' in plots_to_make:
    for region in regions_to_plot:
        # Grab points, CIs, pvals
        subdf = region2subdf[region]
        lb_CIs = np.asarray([subdf.CI_LB_l, subdf.CI_LB_h]).T
        pb_CIs = np.asarray([subdf.CI_PB_l, subdf.CI_PB_h]).T
        pvals = subdf.p_adj.values

        # Put into Hz
        x = subdf.mLB / subdf.dt
        xerr = lb_CIs / subdf.dt.values[:, None]
        y = subdf.mPB / subdf.dt
        yerr = pb_CIs / subdf.dt.values[:, None]

        if suppress_crucifix_arms:
            xerr, yerr = None, None

        # Create the figure
        f, ax = plt.subplots(figsize=(3,3))
        my.plot.crucifix(x=x, y=y, xerr=xerr, yerr=yerr, p=pvals, 
            ax=ax, axtype='linear', zero_substitute=.3e-1,
            suppress_null_error_bars=True)
        
        # Labels
        ax.set_xlabel('Localization F.R. (Hz)')
        ax.set_ylabel('Pitch Discrimination F.R. (Hz)')
        ax.set_xlim((0, 50)); ax.set_ylim((0, 50))
        #ax.set_title(region)
        
        if savefigs:
            f.patch.set_visible(False)
            f.savefig('%s_linear_crucifix.svg' % region)

if 'log crucifix' in plots_to_make:
    for region in regions_to_plot:
        # Grab points, CIs, pvals
        subdf = region2subdf[region]
        lb_CIs = np.asarray([subdf.CI_LB_l, subdf.CI_LB_h]).T
        pb_CIs = np.asarray([subdf.CI_PB_l, subdf.CI_PB_h]).T
        pvals = subdf.p_adj.values

        # Put into Hz
        x = subdf.mLB / subdf.dt
        xerr = lb_CIs / subdf.dt.values[:, None]
        y = subdf.mPB / subdf.dt
        yerr = pb_CIs / subdf.dt.values[:, None]

        if suppress_crucifix_arms:
            xerr, yerr = None, None

        # Create the figure
        f, ax = plt.subplots(figsize=(3,3))
        my.plot.crucifix(x=x, y=y, xerr=xerr, yerr=yerr, p=pvals, 
            ax=ax, axtype='log', zero_substitute=1e-4,
            suppress_null_error_bars=True)
        
        # Labels
        ax.set_xlabel('Localization F.R. (Hz)')
        ax.set_ylabel('Pitch Discrimination F.R. (Hz)')
        if region == 'A1':
            ax.set_xlim((1e-1, 100)); ax.set_ylim((1e-1, 100))
        else:
            ax.set_xlim((1e-2, 100)); ax.set_ylim((1e-2, 100))
        if savefigs:
            f.patch.set_visible(False)
            f.savefig('%s_log_crucifix.svg' % region)

if 'ratio histogram' in plots_to_make:
    for region in regions_to_plot:
        # Data from this region
        subdf = region2subdf[region]
        PBpref = subdf['lograt']
        pvals = subdf.p_adj
        
        # Set up histogram
        bins_extent = np.max([PBpref.max(), -PBpref.min()])
        bins = np.linspace(-bins_extent, bins_extent, 25)
        
        # Plot
        f, ax = plt.subplots(figsize=(3,3))
        ax.hist(PBpref, color='w', bins=bins)
        pb_handles = ax.hist(
            PBpref[(PBpref > 0) & (pvals < .05)], color='r', bins=bins)
        lb_handles = ax.hist(
            PBpref[(PBpref < 0) & (pvals < .05)], color='b', bins=bins)
        ax.set_xlim((bins[0], bins[-1]))
        
        # Yaxis
        ax.set_ylim((0, ax.get_ylim()[1] * 1.25))
        ax.plot([0, 0], ax.get_ylim(), 'k-', lw=3)
        ax.set_ylabel('number of neurons')
        
        # Xticks
        if 10 ** bins_extent < 20:
            ax.set_xticks(np.log10([.1, 1/3., 1, 3, 10]))
            ax.set_xticklabels(('1/10x', '1/3x', '1x', '3x', '10x'))
            ax.set_xticklabels(('0.1', '0.33', '1.0', '3.0', '10'))
            #ax.set_xticklabels((r'$\frac{1}{10}$x', r'$\frac{1}{3}$x', r'1x', r'3x', r'10x')) 
        else: # more zoomed out
            ax.set_xticks(np.log10([.1, 1, 10]))
            ax.set_xticklabels(('1/10x', '1x', '10x'))
            ax.set_xticks(np.log10([.03, 0.1, 0.3, 1, 3,10, 30]))
            ax.set_xticks(np.log10([.03, .1, .3, 1, 3,10, 30]))
            ax.set_xticklabels(('.03', '.1', '.3', '1', '3', '10', '30'))
            #ax.set_xticklabels((r'$\frac{1}{10}$x', r'$\frac{1}{3}$x', r'1x', r'3x', r'10x')) 
            #ax.set_xticklabels(['%0.2f' % v for v in 10**ax.get_xticks()])
    
        ax.set_xlabel('ratio of spike rate (fold)')
        
        # Labels
        v1 = lb_handles[0].sum() 
        v2 = pb_handles[0].sum()
        v3 = len(PBpref)
        ax.text(.2, .75, '%d/%d\nprefer\nlocalization' % (v1, v3), 
            color='b', ha='center', va='center', transform = ax.transAxes)
        ax.text(.8, .75, '%d/%d\nprefer\npitch disc.' % (v2, v3), 
            color='r', ha='center', va='center', transform = ax.transAxes)
        
        my.plot.despine(ax)
        #~ plt.show()
        #~ continue
        
        if savefigs:
            f.patch.set_visible(False)
            f.savefig('%s_ratio_histogram.svg' % region)

if 'difference histogram' in plots_to_make:
    for region in regions_to_plot:
        # Grab points, CIs, pvals
        subdf = region2subdf[region]
        PBpref = subdf['diffHz']
        pvals = subdf.p_adj
        
        # Set up histogram
        bins_extent = np.max([PBpref.max(), -PBpref.min()])
        bins = np.linspace(-bins_extent, bins_extent, 25)
        
        # Plot
        f, ax = plt.subplots(figsize=(3,3))
        ax.hist(PBpref, color='w', bins=bins)
        pb_handles = ax.hist(
            PBpref[(PBpref > 0) & (pvals < .05)], color='r', bins=bins)
        lb_handles = ax.hist(
            PBpref[(PBpref < 0) & (pvals < .05)], color='b', bins=bins)
        ax.set_xlim((bins[0], bins[-1]))
        
        # Yaxis
        ax.set_ylim((0, ax.get_ylim()[1] * 1.25))
        ax.plot([0, 0], ax.get_ylim(), 'k-', lw=3)
        ax.set_ylabel('number of neurons')
        
        # Xticks
        ax.set_xlabel('difference in spike rate (Hz)')
        
        # Labels
        v1 = lb_handles[0].sum() 
        v2 = pb_handles[0].sum()
        v3 = len(PBpref)
        ax.text(.2, .75, '%d/%d\nprefer\nlocalization' % (v1, v3), 
            color='b', ha='center', va='center', transform = ax.transAxes)
        ax.text(.8, .75, '%d/%d\nprefer\npitch disc.' % (v2, v3), 
            color='r', ha='center', va='center', transform = ax.transAxes)
        
        
        my.plot.despine(ax)

        if savefigs:
            #f.patch.set_visible(False)
            f.savefig('%s_diff_histogram.svg' % region)



plt.show()
