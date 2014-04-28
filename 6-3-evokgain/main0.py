# Estimate change in evoked rate across blocks
# With and without spont rate

from ns5_process import myutils, LBPB
import kkpandas
import pandas, os.path
import numpy as np
import my, my.dataload, scipy.stats


# Load evoked responses from previous analysis
evoked_resps = pandas.load('../6-1-evoked/evoked_resps')
unit_db = my.dataload.getstarted()['unit_db']
hold_results = pandas.load('../3-1-hold/hold_results')


# Discard units that don't pass the test of evoked > spont
DISCARD_TOO_WEAK = True

# Bootstrapping parameters
N_BOOTS = 10000 # takes a while, but there are some units right at sigthresh
MIN_BUCKET = 5

# Helper function to run on each neuron
def test_stimulus_equalized_strength(rec, ulabel):
    """Test stimulus-equalized strength in 'rec' using bootstrap"""
    # Put into bootstrap format (list of lists)
    bootstrap_format = []
    for sname in LBPB.mixed_stimnames_noblock:
        # Extract counts for this stimulus
        lb_counts = rec[sname + '_lc']
        pb_counts = rec[sname + '_pc']

        # Convert to Hz
        lb_rates = lb_counts / rec['evok_dt']
        pb_rates = pb_counts / rec['evok_dt']

        # Subtract off spont rate
        if SUBTRACT_OFF_HOLD:
            lb_rates = lb_rates - \
                hold_results['mLB'][ulabel] / hold_results['dt'][ulabel]
            pb_rates = pb_rates - \
                hold_results['mPB'][ulabel] / hold_results['dt'][ulabel]
    
        bootstrap_format.append((lb_rates, pb_rates))

    # Run the bootstrap
    bs_res1 = my.bootstrap.difference_CI_bootstrap_wrapper(
        bootstrap_format, n_boots=N_BOOTS, min_bucket=MIN_BUCKET)

    # Also just do simple averaging, first within, then across stimuli
    # To ensure the means are basically the same
    # Though we need the fancy method to get CIs and p-values
    mLB_simple = np.mean(map(np.mean, [rec[sname + '_lc'] 
        for sname in LBPB.mixed_stimnames_noblock])) / rec['evok_dt']
    mPB_simple = np.mean(map(np.mean, [rec[sname + '_pc'] 
        for sname in LBPB.mixed_stimnames_noblock])) / rec['evok_dt']
    if SUBTRACT_OFF_HOLD:
        mLB_simple -= rec['hold_mLB'] / rec['hold_dt']
        mPB_simple -= rec['hold_mPB'] / rec['hold_dt']

    # Deparse the info
    bs_res2 = {'ulabel': ulabel, 'n_spikes': n_spikes, 'dt': rec['evok_dt'],
        'mLB' : bs_res1['means'][0], 'mPB' : bs_res1['means'][1],
        'mLB_simple' : mLB_simple, 'mPB_simple' : mPB_simple, 
        'mdiff' : bs_res1['mean_difference'],
        'p_boot' : bs_res1['p'],
        }    
    return bs_res2




## DISCARD TOO_WEAK
# Simple test of whether evoked strength tends to be greater than spont
# Originally included cue, error, etc, so not guaranteed for this dataset
too_weak_ulabels = []
for ulabel, rec in evoked_resps.iterrows():
    if DISCARD_TOO_WEAK:
        # Set up the check
        all_counts = np.concatenate(list(rec[LBPB.mixed_stimnames]))
        all_counts_hz = all_counts / rec['evok_dt']
        
        # Hold rate estimate, weighted by block prevalence
        n_lb_trials = np.sum(map(len, [
            rec[sname + '_lc'] for sname in LBPB.mixed_stimnames_noblock]))
        n_pb_trials = np.sum(map(len, [
            rec[sname + '_pc'] for sname in LBPB.mixed_stimnames_noblock]))
        weight_lb = n_lb_trials / float(n_lb_trials + n_pb_trials)
        weight_pb = n_pb_trials / float(n_lb_trials + n_pb_trials)
        hold_rate_hz = (
            weight_lb * rec['hold_mLB'] + weight_pb * rec['hold_mPB']) / \
            rec['hold_dt']

        # Test
        t_ev, p_ev = scipy.stats.ttest_1samp(all_counts_hz, hold_rate_hz)
        if p_ev > .05:
            print "discarding %s too weak" % ulabel
            too_weak_ulabels.append(ulabel)
np.savetxt('too_weak_ulabels', too_weak_ulabels, fmt='%s')


# Whether to subtract off baseline for that block before comparing
# Do both ways and store both
for SUBTRACT_OFF_HOLD in [False, True]:
    # Where to put results
    stim_equalized_res_l = []

    # Iterate over ulabels and test indiv stimuli, and stimulus-equalized
    for ulabel, rec in evoked_resps.iterrows():
        # Skip too weak
        if ulabel in too_weak_ulabels:
            continue
        
        # Count spikes overall
        n_spikes = np.sum(np.concatenate(rec[LBPB.mixed_stimnames]))

        # Test stimulus-equalized strength
        stim_equalized_res = test_stimulus_equalized_strength(rec, ulabel)
        stim_equalized_res_l.append(stim_equalized_res)

    # dataframe the equalized results
    stim_equalized_res = pandas.DataFrame.from_records(
        stim_equalized_res_l).set_index('ulabel')
    stim_equalized_res['p_boot'][
        stim_equalized_res['p_boot'] < (1./N_BOOTS)] = 1. / N_BOOTS

    # Split by brain region
    stim_equalized_res['is_auditory'] = map(kkpandas.kkrs.is_auditory, 
        stim_equalized_res.index)
    A1_stim_equalized_res = stim_equalized_res[stim_equalized_res.is_auditory]
    PFC_stim_equalized_res = stim_equalized_res[~stim_equalized_res.is_auditory]

    # Adjust
    A1_stim_equalized_res['p_adj'] = my.stats.r_adj_pval(
        A1_stim_equalized_res.p_boot)
    PFC_stim_equalized_res['p_adj'] = my.stats.r_adj_pval(
    PFC_stim_equalized_res.p_boot)

    # Diagnostics
    if SUBTRACT_OFF_HOLD:
        print "HOLD SUBRACTED:"
        suffix = 'subhold'
    else:
        print "WITH HOLD:"
        suffix = 'whold'

    print "skipped %d too weak ulabels" % len(too_weak_ulabels)
    print "A1 stim equalized:"
    sig_rows = A1_stim_equalized_res[A1_stim_equalized_res.p_adj < .05]
    print "%d sig ulabels from %d" % (len(sig_rows), len(A1_stim_equalized_res))
    print "PFC stim equalized:"
    sig_rows = PFC_stim_equalized_res[PFC_stim_equalized_res.p_adj < .05]
    print "%d sig ulabels from %d" % (len(sig_rows), len(PFC_stim_equalized_res))

    # Save
    A1_stim_equalized_res.save('A1_stim_equalized_res' + '_' + suffix)
    PFC_stim_equalized_res.save('PFC_stim_equalized_res' + '_' + suffix)

