# Collate all behavioral data from included sessions

import numpy as np
from my.dataload import session2rs, session2kk_server
import kkpandas
from ns5_process import LBPB, myutils
import my
from perf import perfcount, perfratio


def session2ratname(session):
    return session.split('_')[0]

gets = my.dataload.getstarted()
session_db = gets['session_db']
session_db['ratname'] = map(session2ratname, session_db.index)

# Return variable
trials_info_dd = dict([(ratname, {}) for ratname in session_db.ratname.unique()])

# Iterate over sessions
for session_name, row in session_db[session_db.include].iterrows():
    # link back to RS
    rs = session2rs(session_name)#, kk_servers, data_dirs)
    
    # Get behavioral data
    trials_info = kkpandas.io.load_trials_info(rs.full_path)
    trials_info_dd[row['ratname']][session_name] = trials_info

#~ # Get everything in one list as well
#~ allti = np.sum([d.values() for d in trials_info_dd.values()])

myutils.pickle_dump(trials_info_dd, 'trials_info_dd')
