# Evoked analysis
# Dump spike counts from random hits in onset window
# Include both A1 and PFC cells


from ns5_process import myutils, LBPB
import kkpandas
import pandas, os.path
import numpy as np
import my, my.dataload

my.misc.no_warn_rs()

# Unit db
gets = my.dataload.getstarted()
unit_db = gets['unit_db']

# Hold results
hold_results = pandas.load('../3-1-hold/hold_results')

# Cells to analyze
to_analyze = unit_db[
    unit_db.include &
    unit_db.audresp.isin(['weak', 'good', 'sustained'])]


# Initialize return variables
outrec_l = []

# Which trials to get -- include cue this time
trial_picker_kwargs = {
    'labels': LBPB.stimnames,
    'label_kwargs': [{'stim_name': s} for s in LBPB.stimnames],
    'nonrandom': 0, 'outcome': 'hit'}

# Iterate over units
for ulabel, rec in to_analyze.iterrows():
    myutils.printnow(ulabel)

    # Get folded and store for later PSTHing
    dfolded = my.dataload.ulabel2dfolded(ulabel, 
        trial_picker_kwargs=trial_picker_kwargs,
        folding_kwargs={'dstart': 0., 'dstop': .05})

    # count in window
    t1, t2 = rec['audresp_t1'], rec['audresp_t2']
    evoked_spikes = myutils.map_d(
        lambda f: f.count_in_window(t1, t2), dfolded)
    
    # Insert more info
    evoked_spikes['evok_nspk'] = np.sum(map(np.sum, evoked_spikes.values()))
    evoked_spikes['evok_dt'] = t2 - t1
    evoked_spikes['ulabel'] = ulabel
    
    # Store
    outrec_l.append(evoked_spikes)
   
# DataFrame the spike counts
evoked_resps = pandas.DataFrame.from_records(outrec_l).set_index('ulabel')

# Join on some additional info
evoked_resps = evoked_resps.join(
    to_analyze[['region', 'audresp']])
for holdcol in ['mLB', 'mPB', 'dt', 'p_adj', 'auroc']:
    evoked_resps['hold_' + holdcol] = hold_results[holdcol]

# Save the evoked counts and folded
evoked_resps.save('evoked_resps')