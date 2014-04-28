# Plot results of main0
# correlation between hold and stim-equalized change
# Also print out the summary stats on stim-equalized subhold
# These are used to point out the CR12B thing
# Maybe some example units too


from ns5_process import myutils, LBPB
import kkpandas
import pandas, os.path
import numpy as np
import my, my.dataload, my.plot, scipy.stats
import matplotlib.pyplot as plt




# Load stim-equalized with hold
suffix = 'whold'
A1_stim_equalized_res_whold = pandas.load('A1_stim_equalized_res' + '_' + suffix)
PFC_stim_equalized_res_whold = pandas.load('PFC_stim_equalized_res' + '_' + suffix)

# And without (only used in subtraction)
suffix = 'subhold'
A1_stim_equalized_res_subhold = pandas.load('A1_stim_equalized_res' + '_' + suffix)
PFC_stim_equalized_res_subhold = pandas.load('PFC_stim_equalized_res' + '_' + suffix)

# Load other
unit_db = my.dataload.getstarted()['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')
hold_results['mdiffHz'] = (hold_results.mPB - hold_results.mLB) / hold_results.dt
evoked_resps = pandas.load('../6-1-evoked/evoked_resps')



plots_to_make = [
    'text summary',
    'evoked change vs hold change',
    'exemplar 1x4',
    ]
my.plot.font_embed()
my.plot.publication_defaults()


# Apply min spike thresh
# This doesn't change much -- these points are many clustered near the origin
MIN_SPIKES = 0
A1_stim_equalized_res_whold = A1_stim_equalized_res_whold[
    A1_stim_equalized_res_whold.n_spikes >= MIN_SPIKES]
PFC_stim_equalized_res_whold = PFC_stim_equalized_res_whold[
    PFC_stim_equalized_res_whold.n_spikes >= MIN_SPIKES]
PFC_stim_equalized_res_subhold = PFC_stim_equalized_res_subhold[
    PFC_stim_equalized_res_subhold.n_spikes >= MIN_SPIKES]


# Text summary of stim-equalized
if 'text summary' in plots_to_make:
    sdump = []
    sdump.append("A1 stim equalized:")
    sig_rows = A1_stim_equalized_res_whold[
        A1_stim_equalized_res_whold.p_adj < .05]
    sdump.append("%d/%d sig neurons, including hold" % (
        len(sig_rows), len(A1_stim_equalized_res_whold)))
    sig_rows = A1_stim_equalized_res_subhold[
        A1_stim_equalized_res_subhold.p_adj < .05]
    sdump.append("%d/%d sig neurons, subtracting hold" % (
        len(sig_rows), len(A1_stim_equalized_res_subhold)))

    sdump.append("\nPFC stim equalized:")
    sig_rows = PFC_stim_equalized_res_whold[
        PFC_stim_equalized_res_whold.p_adj < .05]
    sdump.append("%d/%d sig neurons, including hold" % (
        len(sig_rows), len(PFC_stim_equalized_res_whold)))
    sig_rows = PFC_stim_equalized_res_subhold[
        PFC_stim_equalized_res_subhold.p_adj < .05]
    sdump.append("%d/%d sig neurons, subtracting hold" % (
        len(sig_rows), len(PFC_stim_equalized_res_subhold)))
    
    sdump = "\n".join(sdump)
    print sdump
    with file('stat__prevalence_of_significant_change_in_evoked', 'w') as fi:
        fi.write(sdump)

if 'evoked change vs hold change' in plots_to_make:
    sdump = []
    for region in ['A1', 'PFC']:
        # Get the change in stim-equalized. This is in Hz
        if region == 'A1':
            y = A1_stim_equalized_res_whold.mdiff
            p_evok = A1_stim_equalized_res_whold.p_adj[y.index]
            line_x = np.array([-18, 27])
            lim = (-20, 30)
            ticks = (-20, -10, 0, 10, 20, 30)
            legend_x = 20
            legend_y = [-5, -10, -15]
        else:
            y = PFC_stim_equalized_res_whold.mdiff
            p_evok = PFC_stim_equalized_res_whold.p_adj[y.index]
            line_x = np.array([-14, 11])
            lim = (-20, 20)
            ticks = (-20, -10, 0, 10, 20)
            legend_x = 13.3
            legend_y = [-5, -9, -13]
        
        # Get the change in hold period
        x = hold_results['mdiffHz'][y.index]
        
        # p-values
        p_hold = hold_results['p_adj'][y.index]
        

        # masks
        sigboth = (p_hold < .05) & (p_evok < .05)
        sighold = (p_hold < .05) & ~sigboth
        sigevok = (p_evok < .05) & ~sigboth
        signeith = ~(sighold | sigevok | sigboth)

        # scatter plot with trend
        f, ax = plt.subplots(figsize=(3,3))
        m, b, rval, pval, stderr = scipy.stats.linregress(x, y)
        
        # Plot each mask
        ax.plot(x, y, marker='o', ls='', mew=1, mfc='none', color='gray')
        
        line_y = m * line_x + b
        trend_line, = ax.plot(line_x, line_y, 'k-', lw=2)
        #~ ax.legend([trend_line], ['trend' % m], loc='lower right', frameon=False, prop={'size':'medium'})
        
        sdump.append("\n%s: delta hold vs delta evoked" % region)
        sdump.append("m=%0.3f, b=%0.3f, p=%0.3e, r=%0.3f, n=%d" % (
            m, b, pval, rval, len(x)))
        
        
        ax.text(legend_x, legend_y[0], 'p = %0.2e' % pval, ha='center', va='center')
        ax.text(legend_x, legend_y[1], 'r = %0.3f' % rval, ha='center', va='center')
        ax.text(legend_x, legend_y[2], 'n = %d' % len(x), ha='center', va='center')
        
        
        ax.set_xlabel('baseline change (PD vs L, Hz)')
        ax.set_ylabel('evoked change (PD vs L, Hz)')
        ax.plot([0, 0], lim, 'k-')
        ax.plot(lim, [0,0],'k-')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        my.plot.despine(ax)
        f.patch.set_visible(False)
        f.savefig('%s_evoked_v_baseline.svg' % region)

    sdump = "\n".join(sdump)
    print sdump
    with file('stat__corr_evoked_vs_baseline_change', 'w') as fi:
        fi.write(sdump)




ulabel2params = {
    'CR21A_120503_004_behaving-604': {
        'ymax': 125, 'bins': np.arange(-.125, .2, 2/300.),},
    'CR20B_120603_001_behaving-406': {
        'ymax': 50, 'bins': np.arange(-.125, .2, .01),},
    }


if 'exemplar 1x4' in plots_to_make:
    for ulabel, params in ulabel2params.items():
        # Plotting params for this unit
        ymax = params['ymax']
        bins = params['bins']
    
        # Get spikes
        dfolded = my.dataload.ulabel2dfolded(ulabel)
        
        # Make folded
        f, axa = plt.subplots(1, 4, figsize=(6.65,2))
        f.subplots_adjust(wspace=.2, left=.05, right=.95)
        
        # Set up the stim labels
        stim_order = ['le_hi', 'le_lo', 'ri_hi', 'ri_lo']
        nice_titles = ['LEFT+HIGH', 'LEFT+LOW', 'RIGHT+HIGH', 'RIGHT+LOW']
        
        # Bin and plot each
        for stim, ax in zip(stim_order, axa.flatten()):
            # bin it
            lbf, pbf = dfolded[stim + '_lc'], dfolded[stim + '_pc']
            lbb = kkpandas.Binned.from_folded_by_trial(lbf, bins=bins)
            pbb = kkpandas.Binned.from_folded_by_trial(pbf, bins=bins)
            
            my.plot.errorbar_data(lbb.rate_in('Hz').values, x=lbb.t*1000, ax=ax, 
                color='b', axis=1, fill_between=True)
            my.plot.errorbar_data(pbb.rate_in('Hz').values, x=pbb.t*1000, ax=ax, 
                color='r', axis=1, fill_between=True)
            ax.set_title(stim)
        
        # scale bar
        if ulabel == 'CR21A_120503_004_behaving-604':
            axa[0].plot([100, 150], [100, 100], 'k-')
            axa[0].plot([100, 100], [100, 125], 'k-')
            axa[0].text(145, 95, '50 ms', ha='center', va='top', size=12)
            axa[0].text(95, 112.5, '25 Hz', ha='right', va='center', size=12)
        else:
            axa[0].plot([120, 170], [35, 35], 'k-')
            axa[0].plot([120, 120], [35, 45], 'k-')
            axa[0].text(145, 34, '50 ms', ha='center', va='top', size=12)
            axa[0].text(115, 40, '10 Hz', ha='right', va='center', size=12)

        # Each ax pretty and shade onset window
        for ax, nice_titl in zip(axa.flatten(), nice_titles):
            # Y axis needs room for a little triangle
            ax.plot([0], [0], 'k^')
            ax.set_ylim((-ymax/30., ymax))
            
            # Tight x
            ax.set_xlim((bins[0]*1000, bins[-1]*1000))
            
            # onset window
            t1, t2 = unit_db.ix[ulabel][['audresp_t1', 'audresp_t2']]
            t1, t2 = t1 * 1000, t2 * 1000
            ax.fill_betweenx(x1=t1, x2=t2, y=[0, ymax], alpha=.4, color='gray', lw=0)

            my.plot.despine(ax, which=('left', 'right', 'top', 'bottom'))
            ax.set_title(nice_titl)
            ax.set_xticks([]); ax.set_yticks([])
            
        
        # Save fig
        f.patch.set_visible(False)
        f.savefig('exemplar_1x4_%s.pdf' % ulabel)
    

plt.show()