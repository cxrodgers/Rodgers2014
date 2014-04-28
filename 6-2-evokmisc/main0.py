# Dump evoked counts from all audresp units, so that we can 
# then calculate some simple statistics on audresp latencies and strengths
from ns5_process import myutils, LBPB
import kkpandas, kkpandas.kkrs
import pandas, os.path
import numpy as np
import my, my.dataload, my.plot

# Include only units that passed feature check and were audresp
unit_db = my.dataload.getstarted()['unit_db']
audresp_units = unit_db.ix[unit_db.include &
    unit_db.audresp.isin(['good', 'weak', 'sustained'])]

# Iterate over units and regrab spikes from all stimuli, all trials
trial_picker_kwargs = 'all'
folding_kwargs = {'dstart': -.050, 'dtstop': .050}
rec_l = []
for ulabel in audresp_units.index:
    myutils.printnow(ulabel)
    # Get folded and concatenate all trials
    dfolded = my.dataload.ulabel2dfolded(ulabel, 
        trial_picker_kwargs=trial_picker_kwargs)
    folded = np.sum(dfolded.values())
    
    # Count in pre and evok win
    t1, t2 = unit_db[['audresp_t1', 'audresp_t2']].ix[ulabel]
    pre_counts = folded.count_in_window(-.050, 0.)
    evok_counts = folded.count_in_window(t1, t2)
    
    # Store
    rec_l.append({'pre': pre_counts, 'evok': evok_counts, 'dt': t2 - t1,
        'ulabel': ulabel})

# Save
df = pandas.DataFrame.from_records(rec_l).set_index('ulabel')
df.save('counts')

