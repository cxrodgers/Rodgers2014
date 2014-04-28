"""Objects and functions for analyzing effect of zap on behavior"""


from ns5_process import bcontrol
import numpy as np, pandas
import my, my.stats, my.plot, glob, os.path
import matplotlib.pyplot as plt
import datetime, time
import scipy.stats

def compare(zdata, cdata):
    """Return mean, binom CIs, U-test, and Fisher's test on two unpaired arrays.
    
    This the inner function called by each control/disruption comparison.
    They differ mainly in how they choose zdata and cdata.
    zdata and cdata should be 1 for hit and 0 for non-hit
    
    Fisher's Exact Test (pfish) is the preferred p-value to use here.
    """
    subres = {}
    subres['ctl_m'] = np.mean(cdata)
    subres['ctl_l'], subres['ctl_h'] = my.stats.binom_confint(data=cdata)
    subres['zap_m'] = np.mean(zdata)
    subres['zap_l'], subres['zap_h'] = my.stats.binom_confint(data=zdata)
    utest_res = my.stats.r_utest(zdata, cdata)
    subres['auroc'] = utest_res['auroc']
    subres['pmw'] = utest_res['p']
    
    table = [
        [(cdata.astype(np.int) == 0).sum(), (cdata.astype(np.int) == 1).sum()],
        [(zdata.astype(np.int) == 0).sum(), (zdata.astype(np.int) == 1).sum()]]
    subres['pfish'] = scipy.stats.fisher_exact(table)[1]
    return subres

class ZapAnalyzer:
    """Object to perform control/disruption comparisons across multiple types"""
    def __init__(self, TI):
        self.original_TI = TI.copy()
        self.prep()
    
    def prep(self):
        """Sets up TI. Also gets zap_types and block list"""
        # Start with original
        TI = self.original_TI.copy()

        # Drop munged
        TI = TI[~TI.is_munged]
        
        # Convenience columns
        TI['is_hit'] = (TI['outcome'] == 'hit').astype(np.int)
        TI['short_sname'] = TI.stim_name.apply(lambda s: s[:5])

        # Store
        self.TI = TI
        
        # All zaps should be of type 1
        assert TI['zap'].isin([0,1]).all()
        self.zap_types = [1]

    def compute_by_stimulus(self):
        """Compare each stim*block individually"""
        # Test zap types separately
        res = {}
        for zt in self.zap_types:
            for stim_number in [5, 6, 7, 8, 9, 10, 11, 12]:
                subTI = self.TI[self.TI.stim_number == stim_number]
                zdata = subTI[subTI.zap == zt]['is_hit']
                cdata = subTI[subTI.zap == 0]['is_hit']
                if len(zdata) == 0 or len(cdata) == 0:
                    1/0
                subres = compare(zdata, cdata)
                res['stim%d_zap%d' % (stim_number, zt)] = subres
        
        return res

    def compute_simple_zap(self):
        """Simple U-test on zap across all non-cue trials"""
        # Test zap types separately
        res = {}
        for zt in self.zap_types:
            # We first remove cue trials
            TImixed = self.TI[self.TI.stim_number.isin([5,6,7,8,9,10,11,12])]
            zdata = TImixed[TImixed.zap == zt]['is_hit']
            cdata = TImixed[TImixed.zap == 0]['is_hit']
            subres = compare(zdata, cdata)
            res['simple_zap%d' % zt] = subres
        
        return res

    def compute_by_block(self):
        """By block"""
        res = {}
        for zt in self.zap_types:
            for block in [1, 2, 3, 4]:
                zdata = my.pick_rows(self.TI, zap=zt, block=block)['is_hit']
                cdata = my.pick_rows(self.TI, zap=0, block=block)['is_hit']
                
                # Skip if nothing
                if len(zdata) < 5 or len(cdata) < 5:
                    continue
                
                subres = compare(zdata, cdata)
                res['block%d_zap%d' % (block, zt)] = subres
        return res

    def compute_by_block_and_gonogo(self):
        """By block and go/nogo"""
        res = {}
        for zt in self.zap_types:
            for block in [1, 2, 3, 4]:
                for gng in ['go', 'nogo']:
                    subres = {}
                    
                    zdata = my.pick_rows(self.TI, zap=zt, block=block, 
                        go_or_nogo=gng)['is_hit']
                    cdata = my.pick_rows(self.TI, zap=0, block=block, 
                        go_or_nogo=gng)['is_hit']
                    
                    # Skip if nothing
                    if len(zdata) < 5 or len(cdata) < 5:
                        continue
                    
                    subres = compare(zdata, cdata)

                    res['block%d_%s_zap%d' % (block, gng, zt)] = subres
        return res


def compute_summary_stats(TI_d):
    """Compute summary stats on each session in TI_d"""
    sstats_l = []
    session_names = []

    # These are the results that are stored from each computation
    keepcols = ['pmw', 'pfish',
        'zap_l', 'zap_m', 'zap_h',
        'ctl_l', 'ctl_m', 'ctl_h']    

    for session_name, TI in TI_d.items():
        session_names.append(session_name)
        #rec = session_db.ix[session_name]

        # Run all computes
        za = ZapAnalyzer(TI)
        res = {}

        res.update(za.compute_by_block())
        res.update(za.compute_by_block_and_gonogo())
        res.update(za.compute_simple_zap())
        res.update(za.compute_by_stimulus())

        # Store summary stats
        resdf = pandas.DataFrame.from_records(res).T
        #resdf['diff'] = resdf['zap_m'] - resdf['ctl_m']
        sstats_l.append(resdf[keepcols])

    # Summary stats
    bigdf = pandas.concat(sstats_l, keys=session_names, verify_integrity=True)
    bigdf.index.names = ['session', 'analysis']
    bigdf = bigdf.unstack('analysis')
    return bigdf

