# Exemplar neurons that encode rule during reward consumption
import my, my.dataload
import numpy as np, pandas, kkpandas, os.path
from ns5_process import LBPB
import matplotlib.pyplot as plt, my.plot
from shared import *


my.plot.font_embed()
my.plot.publication_defaults()

# Bin params (t_start and t_stop should match folded params)
bins = kkpandas.define_bin_edges2(t_start=-3., t_stop=3, binwidth=.05)
distr_bins = np.linspace(-3, 3, 101)

# Smoothing
gs = kkpandas.base.GaussianSmoother(smoothing_window=.05)
meth = gs.smooth
meth = np.histogram # just for the data loading; we resmooth with gs.smooth later

# Flag controlling which dataset we load
after_trial_type = 'after_all'
locking_event = 'cpoke_stop'

# Load data -- we really just need all_event_latencies for this script
try:
    binneds
except NameError:
    suffix, all_event_latencies, binneds = load_data(
        locking_event=locking_event, after_trial_type=after_trial_type,
        bins=bins, meth=meth)

# Exemplars
exemplars = {}
exemplars['A1'] = [
    u'CR12B_110429_001_behaving-338', u'CR17B_110803_001_behaving-340'] 
exemplars['PFC'] = [
    u'CR20B_120612_001_behaving-208', u'CR24A_121027_001_behaving-214'][::-1]


# Plot the exemplars binneds
for region, ulabels in exemplars.items():
    # Create figure for this region
    region_l = ['PFC', 'A1']
    f, axa = plt.subplots(1, 2, figsize=(7,3))
    f.subplots_adjust(left=.125, right=.95, wspace=.475)
    f.suptitle('example %s neurons that encode rule during reward' % region)

    # One axis for each ulabel
    for nu, ulabel in enumerate(ulabels):
        # Re-grab foldeds so we can plot error bars
        trial_picker_kwargs = {
            'labels': ['LB GO', 'PB GO'],
            'label_kwargs': [
                {'stim_name':['le_hi_lc', 'le_lo_lc']},
                {'stim_name':['le_lo_pc', 'ri_lo_pc']},
                ],
            }
        dfolded = my.dataload.ulabel2dfolded(ulabel, 
            folding_kwargs={'dstart': -3.2, 'dstop': 3.2}, # avoid artefact
            time_picker_kwargs={'event_name': 'cpoke_stop'},
            trial_picker_kwargs=trial_picker_kwargs)
        
        # Bin by trial
        binned_LB = kkpandas.Binned.from_folded_by_trial(
            dfolded['LB GO'], bins=bins, meth=gs.smooth)
        binned_PB = kkpandas.Binned.from_folded_by_trial(
            dfolded['PB GO'], bins=bins, meth=gs.smooth)

        # Plot on GO trials in both blocks
        ax = axa[nu]
        my.plot.errorbar_data(x=binned_LB.t, data=binned_LB.rate_in('Hz').values,
            fill_between=True, color='b', ax=ax, axis=1)
        my.plot.errorbar_data(x=binned_PB.t, data=binned_PB.rate_in('Hz').values,
            fill_between=True, color='r', ax=ax, axis=1)
        
        ## DISTRS
        # Take same ulabels, combine across blocks
        ax_distrs = take_distrs_from_ulabels_and_combine_blocks(
            all_event_latencies, ulabels=[ulabel], gng='GO')
        
        # Plot them
        bottom_edge = 0
        max_height = .15 * ax.get_ylim()[1]
        plot_distrs(ax, ax_distrs, bottom_edge, locking_event, distr_bins, 
            maxheight=max_height)        

        ## PRETTY
        # Zero lines
        ax.plot(ax.get_xlim(), [0, 0], 'k:')
        ax.plot([0, 0], ax.get_ylim(), 'k:')
        
        set_xlabel_by_locking_event(ax, locking_event=locking_event)            
        ax.set_xlim((-3, 3))
        ax.set_xticks((-3, -2, -1, 0, 1, 2, 3))
        
        ax.set_ylabel('firing rate (Hz)')           

    f.savefig('exemplars_%s_encoding_rule_during_reward.pdf' % region)

plt.show()
