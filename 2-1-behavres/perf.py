import numpy as np
import my

RETURN_PCT = True

class Perf:
    def __init__(self, trials_info):
        self.trials_info = trials_info
    
    def count(self, **kwargs):
        return perfcount(self.trials_info, **kwargs)

def perfcount(trials_info, **kwargs):
    """Filter trials_info by kwargs and return performance"""
    if 'nonrandom' not in kwargs:
        kwargs = kwargs.copy()
        kwargs['nonrandom'] = 0
    
    data = my.pick_rows(trials_info, **kwargs)
    
    nhits = np.sum(data.outcome == 'hit')
    nerrs = np.sum(data.outcome == 'error')
    nspos = np.sum(data.outcome == 'wrong_port')
    nfut = np.sum(data.outcome == 'future_trial')
    
    assert nhits + nerrs + nspos + nfut == len(data)
    
    return nhits, nerrs, nspos
    
    
def perfratio(trials_info, **kwargs):
    nhits, nerrs, nspos = perfcount(trials_info, **kwargs)
    tot = float(nhits + nerrs + nspos)
    
    if RETURN_PCT:
        tot = tot / 100.
    return nhits / tot, nerrs / tot, nspos / tot
