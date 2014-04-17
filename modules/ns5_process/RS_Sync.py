"""Module containing all synchronization code for neural and behavioral

Move this to kkpandas.kkrs as it does not use any ns5 or OE 
"""

import numpy as np
from ns5_process import bcontrol
import kkpandas
import os.path
import collections

# For sync_b_n_v
import pandas
import vidtrack, my, my.dataload

class RS_Syncer:
    """Defines preferred interface for accessing trial numbers and times
    
    This defines the "right" way to access the onset times and trial numbers
    
    Can return either neural or behavioral times
    
    Attributes
    btrial_numbers : Behavioral trial numbers that occurred during the
        recording of this session. This is defined by the index column of
        trials_info, which matches the displayed values in Matlab.
        The trial_%d_out states inserted into the events structure were named
        using this convention.
    
    ntrial_numbers : range(0, len(TIMESTAMPS)), that is, a 0-based counting
        of detected trials in neural data

    btrial_start_times : Start times of behavioral trial, length same as
        btrial_numbers, in behavioral time base
    
    Not implemented till I decide more:
    ntrial_start_times : Start times of OE Segments, length same as
        ntrial_numbers. Note not triggered on the same trial event as
        btrial_start_times. Not implemented yet till I decide whether it
        should actually refer to the digital trial pulse times
    
    btrial_onset_times : Onset of stimulus in behavioral trials
        Of length equal to btrial_numbers
    
    ntrial_onset_times : Onset of stimulus in neural trials
        Of length equal to ntrial_numbers. This one should actually be
        referencing the same physical event as btrial_onset_times
    
    Preferred accessor methods
    
    bnum2trialstart_nbase, bnum2trialstart_bbase : dicts from behavioral 
        trial number to behavioral trial start time, in either time base. 
        Returns None if the behavioral trial did not actually occur during 
        the recording
        Would like to make these accept lists of trial numbers ...
    
    n2b, b2n : convert time base
    
    Preferred writer methods (to be used by syncing functions)
    write_btrial_numbers : etc.
    """
    def __init__(self, rs):
        self.rs = rs
        self._set_sync()
        self._set_btrial_numbers()
        self._set_btrial_start_times()
        self._set_bnum2trialstart_d()
    
    def _set_btrial_numbers(self):
        """Defines behavioral trial numbers that actually occurred
        
        1) Those listed in TRIAL_NUMBERS
        2) If TRIAL_NUMBERS doesn't exist, load from events structure
        3) If events structure doesn't exist, load from bcontrol
        """
        try:
            self.btrial_numbers = np.loadtxt(
                os.path.join(self.rs.full_path, 'TRIAL_NUMBERS'), dtype=np.int)
        except IOError:
            # Load from bcontrol in case syncing hasn't happened yet
            1/0
    
    def _set_sync(self):
        """Defines syncing function between timebases
        
        1) Load from SYNC_B2N in RS
        """
        try:
            self._sync_poly = np.loadtxt(
                os.path.join(self.rs.full_path, 'SYNC_B2N'), dtype=np.float)
        except IOError:
            # Auto sync? Leave None?
            1/0
    
        self._sync_poly_inv = np.array([1., -self._sync_poly[1]]) \
            / self._sync_poly[0]
        
    def b2n(self, bt):
        return np.polyval(self._sync_poly, bt)
    
    def n2b(self, nt):
        return np.polyval(self._sync_poly_inv, nt)
    
    def _set_btrial_start_times(self):
        """Defines btrial onset times
        
        Requires btrial_numbers to exist, because that's how I know which
        trials actually occurred during the recording.
        
        1) As listed in events structure - which is defined to be in the
            neural time base.
        2) If doesn't exist, load from bcontrol
        
        Needs sync info, because
        Stores trialstart_nbase and trialstart_bbase
        """
        try:
            trials_info = kkpandas.io.load_trials_info(self.rs.full_path)
            events = kkpandas.io.load_events(self.rs.full_path)
        except IOError:
            # These haven't been dumped yet
            # Should load them from bcontrol file
            # Then the syncing functions can use this same accessor method
            # Code not written yet
            1/0
        
        # Find the time of each trial change
        # For speed, first find the events matching the trial change string
        matching_events = events.event.str.match('trial_(\d+)_out')
        mask = (matching_events.apply(len) > 0)
        subevents = events[mask]
        res = []
        for bn in self.btrial_numbers:
            # Will error here if non-unique results
            res.append(
                subevents[subevents.event == 'trial_%d_out' % bn].time.item())
        
        # Convert to arrays and store        
        self.trialstart_nbase = np.array(res)
        self.trialstart_bbase = self.n2b(self.trialstart_nbase)
    
    def _set_bnum2trialstart_d(self):
        self.bnum2trialstart_nbase = collections.defaultdict(lambda: None)
        self.bnum2trialstart_bbase = collections.defaultdict(lambda: None)
        
        assert len(self.btrial_numbers) == len(self.trialstart_nbase)
        self.bnum2trialstart_nbase.update(zip(
            self.btrial_numbers, self.trialstart_nbase))
        
        assert len(self.btrial_numbers) == len(self.trialstart_bbase)
        self.bnum2trialstart_bbase.update(zip(
            self.btrial_numbers, self.trialstart_bbase))








def sync_b_n_v(session_name, 
    vidtrack_root='/media/hippocampus/chris/20130916_vidtrack_sessions',
    hold_filename='../../figures/20130621_figfinal/02_hold/counts_results',
    MAX_BN_SYNC_ERR=.002):
    """Sanity check syncing function.
    
    Loads data from video, behavioral, and neural. Produces one dataframe
    with info from everything. Useful for checking sure all trial numbers
    are lined up. Still have to watch the video manually for some things.
    
    Would be worth breaking this into some smaller components
    """

    ## LOADING DATA
    # Video tracking
    vts = vidtrack.Session(full_path=os.path.join(vidtrack_root, session_name))
    locdf = vidtrack.vts_db2head_pos_df(vts)
    srv = vidtrack.interact.Server(db_filename=vts.file_schema.db_filename,
        image_filenames=vts.file_schema.image_filenames,
        image_full_filenames=vts.file_schema.image_full_filenames,
        image_priorities=None,
        hide_title=False)

    # RS, events, trials_info
    rs = my.dataload.session2rs(session_name)
    rss = RS_Syncer(rs)
    trials_info = kkpandas.io.load_trials_info(rs.full_path, drop_munged=False)
    events = kkpandas.io.load_events(rs.full_path)

    # Hold period analysis
    # Use a representative ulabel from the hold analysis
    # Should probably prioritize sigmod ulabels
    hold_counts_results = pandas.load(hold_filename)
    repres_ulabel = filter(lambda s: s.startswith(session_name), 
        hold_counts_results.index)[0]
    hold_counts_result = hold_counts_results.ix[repres_ulabel]

    # Raw spikes
    spks = my.dataload.ulabel2spikes(repres_ulabel)


    ## FORMING DATAFRAME
    # Construct dataframe, starting with TRIALS_INFO
    # All of this is in the neural timebase, converted from behavioral at some point.
    syncdf = trials_info[['block', 'stim_name', 'outcome', 'is_munged',
        'cpoke_start', 'stim_onset', 'choice_made']].copy()
    syncdf.index.name = 'btrial'

    # Duration of cpoke
    syncdf['pdur_act'] = trials_info['cpoke_stop'] - syncdf['cpoke_start']
    syncdf['pdur_req'] = syncdf['stim_onset'] - syncdf['cpoke_start']

    # Now add the ntrial numbers and TIMESTAMPS
    # Some of this logic could go into RS_Sync
    assert len(rss.btrial_numbers) == len(rs.read_timestamps())
    syncdf['ts'] = np.nan
    syncdf['ts'][rss.btrial_numbers] = rs.read_timestamps() / 30e3
    syncdf['ntrial'] = -1
    syncdf['ntrial'][rss.btrial_numbers] = list(range(len(rss.btrial_numbers)))

    # Now add the spike counts and hold analysis
    # Trials that were not analyzed will have a count of -1
    hold_counts = np.concatenate([
        hold_counts_result['LB_counts'], hold_counts_result['PB_counts']])
    hold_trials = np.concatenate([
        hold_counts_result['LB_trials'], hold_counts_result['PB_trials']])
    syncdf['hcnt'] = -1
    syncdf['hcnt'][hold_trials] = hold_counts

    # Head angle on each trial, where it exists
    syncdf = syncdf.join(locdf[['Mx', 'My', 'angl']])

    # Convert time base
    syncdf['stim_v'] = np.polyval(vts.n2v_sync, syncdf.stim_onset)


    ## SANITY CHECKS
    # Hold period analysis was done correctly
    # Refold the spikes directly from scratch here
    # May be a floating point or slight sync issue here
    incl_mask = syncdf.hcnt != -1
    delta = hold_counts_result['dt']
    refolded = kkpandas.Folded.from_flat(spks, 
        starts=syncdf['stim_onset'][incl_mask].values - delta,
        stops=syncdf['stim_onset'][incl_mask].values,
        subtract_off_center=False, labels=syncdf.index[incl_mask])
    assert np.all(syncdf['hcnt'][incl_mask] == refolded.apply(len))

    # All trials in hold analysis had sufficiently long actual hold
    assert np.all(syncdf['pdur_req'][incl_mask] > delta)
    assert np.all(syncdf['pdur_act'] > syncdf['pdur_req'])

    # Check that neural and behavioral are lined up
    # TIMESTAMPS should match stim_onset almost perfectly (with slight error)
    # This ensures that the neural trials and behavioral trials are lined up
    bn_err = syncdf.stim_onset - syncdf.ts
    assert np.all(bn_err.dropna() < MAX_BN_SYNC_ERR)
    assert bn_err.abs().max() > 1e-4 # this could be lower, probably


    ## MANUAL SANITY CHECKS
    # Check that video matches syncdf
    # The video trials should match the syncdf trials in outcome and timing
    # Because inter-trial interval is variable, this can only happen if
    # the btrial numbering is correct.



    # Check that the head angle measurements match up
    # The images shoulod match the frames
    # And the angles should match the images/frames

    return syncdf
