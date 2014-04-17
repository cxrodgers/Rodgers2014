"""Helper functions and variables for all data in a recording session.

This allows you to create a well-formed directory for ns5_process and fill
it with data.

Try to keep logic relating to any specific experiment out of this module.

RecordingSession spec:
* A directory
* File containing neural channels to put into database
  16 17 18 20 22 24 26 28
* File containing channel groupings
  16 17 18 20
  22 24 26 28
* File containing analog channels to put into database (if any)
  7 8
* TIMESTAMPS with times in samples to extract
* Time limits filename with soft limits on first line, then hard
"""

import shutil
import glob
import os.path
#~ import ns5
import time
import numpy as np
#~ import TrialSlicer
#~ try:
    #~ import OpenElectrophy as OE
#~ except:
    #~ print "cannot import OE"
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#~ import KlustaKwikIO
import scipy.signal
#~ import SpikeTrainContainers
from myutils import printnow
import myutils
import datetime
import warnings

# Because no ns5 files
warnings.filterwarnings('ignore', module='ns5_process.RecordingSession$')    

# Globals
ALL_CHANNELS_FILENAME = 'NEURAL_CHANNELS_TO_GET'
GROUPED_CHANNELS_FILENAME = 'NEURAL_CHANNEL_GROUPINGS'
ANALOG_CHANNELS_FILENAME = 'ANALOG_CHANNELS'
TIMESTAMPS_FILENAME = 'TIMESTAMPS'
TIME_LIMITS_FILENAME = 'TIME_LIMITS'
FULL_RANGE_UV = 8192. # maximum input range of signal

# Fs is read from ns5 file, but after the db exists, it will just assume
# 30K rather than try to re-read from ns5 file (which may no longer exist).
# a better solution is to read from the actual signals of interest.
FIXED_SAMPLING_RATE = 30000. 


RAW_BLOCK_NAME = 'Raw Data'
SPIKE_BLOCK_NAME = 'Spike-filtered Data'

def write_channel_numbers(filename, list_of_lists):
    """Writes a list of lists of channel numbers to a file.
    
    Each list in list_of_lists in written to a line of the file.
    Each line contains channel numbers with a single space following each.
    
    Actually I started using this for all kinds of writes, it works
    like np.save and np.load except there is no requirement that each
    entry be the same length.
    """
    fi = file(filename, 'w')
    for chlist in list_of_lists:
        for ch in chlist: 
            fi.write(str(ch) + ' ')
        fi.write('\n')
    fi.close()

def read_channel_numbers(filename, dtype=np.int):
    """ Load TETRODE_CHANNELS control file
    This is funny-looking because not all tetrodes have same number
    of channels!
    """
    f = file(filename)
    x = f.readlines()
    f.close()
    return [[dtype(c) for c in r] for r in [str.split(rr) for rr in x]]

class FiltererForSpikes:
    """Object that handles filtering raw data for spikes.
    
    This is intended to be created and used by RecordingSession.
    Initialized with all of its parameters, and then call
    do_filter to use it on raw data.
    """
    def __init__(self, fixed_sampling_rate, filter_ord1=3, low_cut=300., 
        smooth_spikes=False, 
        filter_ord2=3, high_cut=3000., filter_type=scipy.signal.butter):
        """Initializes spike filterer for later data processing.
        
        fixed_sampling_rate : sampling rate of your data
        filter_ord1 : Order of high-pass filter
        low_cut : Low-frequency cutoff of high-pass filter, in Hz
        smooth_spikes : boolean, specifies whether to additionally low-pass
            spikes to smooth them. If False, then filter_ord2 and high_cut
            are ignored.
        filter_ord2 : Order of optional low-pass filter
        high_cut : High-frequency cutoff of optional low-pass filter
        filter_type : something that returns filter parameters,
            like scipy.signal.butter
        
        It's intended that these options are set once at initialization
        and then not changed, since presumably you would want to 
        filter all of your data in the same way.
        """    
        self.fixed_sampling_rate = fixed_sampling_rate
        self.filter_ord1 = filter_ord1
        self.low_cut = low_cut
        self.smooth_spikes = smooth_spikes
        self.filter_type = filter_type
        
        # Design filter
        self._define_spike_filter_1()
        
        # Optionally design second filter
        if self.smooth_spikes:
            self.filter_ord2 = filter_ord2
            self.high_cut = high_cut
            self._define_spike_filter_2()
    
    def _define_spike_filter_1(self):
        """Define a high-pass filter"""
        normed_low_cut = self.low_cut / (self.fixed_sampling_rate / 2.)
        self._filter1 = self.filter_type(self.filter_ord1, 
            normed_low_cut, btype='high')

    def _define_spike_filter_2(self):
        """Define an optional low-pass filter for smoothing spikes"""
        normed_high_cut = self.high_cut / (self.fixed_sampling_rate / 2.)
        self._filter2 = self.filter_type(self.filter_ord2,
            normed_high_cut, btype='low')

    def filter(self, input_signal):
        """Filters input signal for spike detection.
        
        input_signal : numpy array, should have sampling rate of
            self.sampling_rate
        
        Returns output_signal, of same length as input_signal, which has
        been filtered.
        """        
        # High pass filter
        output_signal = scipy.signal.filtfilt(self._filter1[0],
            self._filter1[1], input_signal)
        
        # If a second lowpass filter, apply
        if self.smooth_spikes:            
            output_signal = scipy.signal.filtfilt(self._filter2[0],
                self._filter2[1], output_signal)
        
        return output_signal    

class RecordingSession:
    """Object linked to a directory containing data for processing.
    
    Provides methods to read and write data from that directory.
    
    Some methods look for stereotype filenames which are defined by
    globals above. Others find a target by globbing. The latter provides
    getter methods and return None if the file doesn't exist. Perhaps
    the former should provide getter methods that always work, even 
    if file doesn't exist. Or is that too confusing?
    """
    def __init__(self, dirname, stifle_warning=False):
        """Create object linked to a data directory
        
        If directory does not exist, will create it.
        """
        self.full_path = os.path.normpath(dirname)
        self.session_name = os.path.basename(self.full_path)
        
        if not os.path.exists(self.full_path):
            os.mkdir(self.full_path)
        elif self.get_ns5_filename() is None and not stifle_warning:
            warnings.warn("warning: no ns5 file in %s" % self.full_path)
        
        self.session = None
        self.group_multiplier = None
    
    def get_db_filename(self):
        return os.path.join(self.full_path, self.session_name + '.db')
    
    def open_db(self):
        print "deprecated, go through get_OE_session"
        OE.open_db('sqlite:///' + self.get_db_filename())
        self.session = OE.Session()
    
    def get_OE_session(self):
        """Returns SQL session, opening database only if needed.
        
        Let's try to keep database opens to a minimum.
        """
        if self.session is not None:
            return self.session
        else:
            OE.open_db('sqlite:///' + self.get_db_filename())
            self.session = OE.Session()
            return self.session
    
    def get_ns5_loader(self):
        """Returns Loader object for ns5 file"""
        l = ns5.Loader(filename=self.get_ns5_filename())
        l.load_file()
        return l
    
    def read_channel_groups(self):
        """Returns a list of channel groups"""
        return read_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME))
    
    def read_neural_channel_ids(self):
        """Returns a list of all channel numbers"""
        return read_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME))[0]
    
    def read_analog_channel_ids(self):
        """Returns a list of analog channels, or None if no file"""
        try:
            return read_channel_numbers(\
                os.path.join(self.full_path, ANALOG_CHANNELS_FILENAME))[0]
        except IOError:
            return None
    
    def read_time_limits(self):
        """Returns tuple of soft time limits and then hard, or None if missing"""
        try:
            data = read_channel_numbers(os.path.join(self.full_path,
                TIME_LIMITS_FILENAME), dtype=np.float)
        except IOError:
            return None
        
        return data[0], data[1]
    
    def write_time_limits(self, soft_time_limits, hard_time_limits):
        """Writes time limits in seconds to directory"""
        write_channel_numbers(os.path.join(self.full_path, 
            TIME_LIMITS_FILENAME), [soft_time_limits, hard_time_limits])

    def write_channel_groups(self, list_of_lists):
        """Writes metadata for channel groupings."""
        write_channel_numbers(\
            os.path.join(self.full_path, GROUPED_CHANNELS_FILENAME),
            list_of_lists)
    
    def write_neural_channel_ids(self, list_of_channels):
        """Writes list of channels to be processed."""
        write_channel_numbers(\
            os.path.join(self.full_path, ALL_CHANNELS_FILENAME),
            [list_of_channels])
    
    def write_analog_channel_ids(self, list_of_channels):
        """Writes analog channel numbers."""
        write_channel_numbers(\
            os.path.join(self.full_path, ANALOG_CHANNELS_FILENAME),
            [list_of_channels])
    
    def get_ns5_filename(self):
        """Returns ns5 filename in recording session"""
        filename_list = glob.glob(os.path.join(self.full_path, '*.ns5'))
        if len(filename_list) == 0:
            return None        
        
        if len(filename_list) > 1:
            print "warning: multiple ns5 files exist in %s" % self.full_path
        
        return filename_list[0]

    def get_raw_data_block(self):
        """Returns Block with raw data name, or None"""
        # Open database, get session
        #self.open_db()
        session = self.get_OE_session()
        
        # Check to see whether data has already been added
        q = session.query(OE.Block).filter(OE.Block.name == RAW_BLOCK_NAME)
        if q.count() > 0:
            return q.one()   
        else:
            return None
    
    def get_spike_block(self):
        """Returns Block with spike block name, or None"""
        # Open database, get session
        #self.open_db()
        session = self.get_OE_session()
        
        # Check to see whether data has already been added
        q = session.query(OE.Block).filter(OE.Block.name == SPIKE_BLOCK_NAME)
        if q.count() > 0:
            return q.one()   
        else:
            return None
    
    def add_file(self, filename):
        """Copy file into RecordingSession."""
        shutil.copy(filename, self.full_path)
    
    def add_timestamps(self, list_of_values):
        """Adds timestamps by writing values to file in directory"""
        # different format, one value per line
        list_to_write = [[v] for v in list_of_values]        
        write_channel_numbers(\
            os.path.join(self.full_path, TIMESTAMPS_FILENAME),
            list_to_write)
    
    def read_timestamps(self):
        """Returns numpy array of timestamps"""
        t = read_channel_numbers(os.path.join(self.full_path, 
            TIMESTAMPS_FILENAME))
        return np.array([tt[0] for tt in t])
    
    def calculate_trial_boundaries(self, soft_limits_sec=None,
        hard_limits_sec=None, meth='end_of_previous', final_sample=None):
        """Reads timestamps and returns trial boundaries.
        
        For implementation, see TrialSlicer. This is just a thin wrapper
        around that.
        
        final_sample : Last sample in the file. If None, then uses
            l.header.n_samples (but note this requires ns5 file to exist).
        
        Returns: (t_starts, t_stops)
        """
        if final_sample is None:
            l = self.get_ns5_loader()
            final_sample = l.header.n_samples
        
        fs = self.get_sampling_rate()
        t = self.read_timestamps()
        
        # Get time limits for slicing
        if hard_limits_sec is None:
            hard_limits_sec = self.read_time_limits()[1]
        if soft_limits_sec is None:
            soft_limits_sec = self.read_time_limits()[0]
        hard_limits = np.array(\
            np.asarray(hard_limits_sec) * fs, dtype=np.int)
        soft_limits = np.array(\
            np.asarray(soft_limits_sec) * fs, dtype=np.int)
        
        # Slice Trials around timestamps
        t_starts, t_stops = TrialSlicer.slice_trials(\
            timestamps=t,
            soft_limits=soft_limits, 
            hard_limits=hard_limits, 
            meth=meth, 
            data_range=(0, final_sample))
        
        return t_starts, t_stops
    
    
    def put_neural_data_into_db(self, soft_limits_sec=None, 
        hard_limits_sec=None, verbose=False, force=False,
        commit_chunk_size=1):
        """Loads neural data from ns5 file and puts into OE database.
        
        Slices around times provided in TIMESTAMPS. Also puts events in
        with label 'Timestamp' at each TIMESTAMP.
       
        Time limits will be loaded from disk unless provided.
        
        force : if True and database exists, destroys database and then runs
            as usual
        
        Returns OE block.
        """
        if verbose:
            count_time = datetime.datetime.now()
            printnow("putting raw data into database")
        
        # force run?
        if force and os.path.exists(self.get_db_filename()):
            if verbose:
                printnow("deleting file %s" % self.get_db_filename())
            os.remove(self.get_db_filename())
        
        # Open database, get session
        #self.open_db()
        session = self.get_OE_session()
        
        # See if operation already occurred
        block = self.get_raw_data_block()
        if block is not None:
            if verbose:
                printnow("raw block already exists, returning")
            return block      
        
        # Read time stamps and set limits in samples
        t_starts, t_stops = self.calculate_trial_boundaries()
        
        # Determine what channels to grab
        chlist = self.read_neural_channel_ids() + \
            [ch+128 for ch in self.read_analog_channel_ids()]
        
        # Create an OE block to store the data
        block = OE.Block(fileOrigin = self.get_ns5_filename())
        block.name = RAW_BLOCK_NAME
        session.add(block)
        session.commit()
        
        # Now create a reader for the filename
        blr = OE.neo.io.BlackrockIO(self.get_ns5_filename())     
        
        # Load one segment at a time to conserve memory
        for n, (t_start, t_stop) in enumerate(zip(t_starts, t_stops)):
            if verbose:
                vmsg = "reading segment %d from %d to %d" % (n, t_start, t_stop)
                printnow(vmsg)
            # Reads time slice, converts to microvolts
            seg = blr.read_segment(t_start=t_start, t_stop=t_stop, 
                full_range=FULL_RANGE_UV, chlist=chlist)
            seg.name = 'Segment %d' % n
            seg.fileOrigin = self.get_ns5_filename()        
            
            # Convert to OE object and append to OE block
            seg2 = OE.io.io.hierachicalNeoToOe(seg)        
            block._segments.append(seg2)
            
            # Add to database
            session.add(seg2)
            
            # Commit
            # This is the time bottleneck
            if np.mod(n, commit_chunk_size) == (commit_chunk_size - 1):
                session.commit()
        session.commit()

        # Add events at TIMESTAMPS
        if verbose:
            printnow("adding event timestamps")
        t = self.read_timestamps()
        if len(t) == len(block._segments):
            for tt, seg in zip(t, block._segments):
                e = OE.Event(name='Timestamp', label='Timestamp')
                e.time = (tt/self.get_sampling_rate())
                #e.save()
                seg._events.append(e)
        else:
            print "warning: timestamps were dropped so I can't add events"

        # Save to database
        if verbose:
            printnow("final commit and expunge")
        session.commit()
        session.expunge_all()
        
        if verbose:
            time_taken = datetime.datetime.now() - count_time
            printnow("operation finished in %0.2f s" % time_taken.total_seconds())
    
    def generate_spike_block(self, CAR=True, smooth_spikes=False, 
        filterer=None, force=False, verbose=False):
        """Filters the data for spike extraction.
        
        CAR: If True, subtract the common-average of every channel.
        smooth_spikes: If True, add an additional low-pass filtering step to
            the spike filter.
        filterer : provide your own spike filterer which acts like class
            FiltererForSpikes. If you specify, then smooth_spikes is ignored,
            because filterer will take care of that.
        
        If spike block already exists, will return without doing anything,
        unless you specify force=True, in which case the existing spike block
        will be overwritten.
        
        """
        # Open connection to the database
        #self.open_db()
        session = self.get_OE_session()
        
        # Check that I haven't already run
        sb = self.get_spike_block()
        if sb is not None:
            if force:
                # delete existing block
                if verbose:
                    printnow("deleting existing spike block")
                session.delete(sb)
                session.commit()
                session.expunge_all()
            else:
                if verbose:
                    printnow("spike filtering already done!")
                return
        
        # Find the raw data block
        raw_block = self.get_raw_data_block()
        
        # Create a new block for referenced data, and save to db.
        if verbose:
            printnow("Creating spike block")
        spike_block = OE.Block(\
            name=SPIKE_BLOCK_NAME,
            info='Referenced and filtered neural data suitable for spike detection',
            fileOrigin=self.get_db_filename())
        spike_block.save(session=session)
        
        # Create a spike_filterer to use
        self.filterer = filterer
        if self.filterer is None:
            self.filterer = FiltererForSpikes(
                fixed_sampling_rate=self.get_ns5_loader().header.f_samp,
                smooth_spikes=smooth_spikes)        

        # Make RecordingPoint for each channel, linked to tetrode number with `group`
        # Also keep track of link between channel and RP with ch2rpid dict
        # TODO: check that this works with int channel, not float
        if verbose:
            printnow("Creating recording points")
        ch2rpid = dict()
        for tn, ch_list in enumerate(self.read_channel_groups()):
            for ch in ch_list:
                rp = OE.RecordingPoint(name=('RP%d' % ch), 
                    id_block=spike_block.id, trodness=len(ch_list),
                    channel=ch, group=tn)
                rp_id = rp.save(session=session)
                ch2rpid[ch] = rp_id

        # Copy old segments over to new block
        if verbose:
            printnow("populating segments ... this may take a while")
        for old_seg in raw_block._segments:
            # Create a new segment in the new block with the same name
            new_seg = OE.Segment(name=old_seg.name, info=old_seg.info,
                id_block=spike_block.id)
            new_seg.save(session=session)

            # Populate with filtered signals
            new_seg = self._create_spike_seg(old_seg, new_seg, CAR, session, 
                ch2rpid)


    def _create_spike_seg(self, old_seg, new_seg, CAR, session, ch2rpid):
        """Create segment with filtered data in new spike block.
        
        old_seg : OE.Segment with AnalogSignal to copy
        new_seg : OE.Segment to populate with spike-filtered copies of
            AnalogSignal from old_seg
        CAR : if True, then use common-average reference
        
        If CAR, then the commmon-average will be subtracted from each
        signal. It will then be filtered, using self.filterer. The result
        is stored in the new segment, with all of the signal metadata
        copied over.
        """
        # Get the channel groupings
        channel_groups = self.read_channel_groups()
        flat_channel_groups = [item for sublist in channel_groups 
            for item in sublist]
        
        # Get signals (including only ones we want to proces)
        siglist = filter(lambda x: x.channel in flat_channel_groups,
            old_seg._analogsignals)
        
        # Error check length and sampling rate
        assert len(np.unique([len(sig.signal) for sig in siglist])) == 1
        all_sampling_rates = np.array([sig.sampling_rate for sig in siglist])
        assert np.abs(all_sampling_rates.max() - all_sampling_rates.min()) < 1.
        assert np.abs(all_sampling_rates.max() - 
            self.get_ns5_loader().header.f_samp) < 1.
        fixed_sampling_rate = all_sampling_rates.mean()
        fixed_signal_length = len(siglist[0].signal)
        
        # Compute average of each
        if CAR: ref_sig = np.mean([sig.signal for sig in siglist], axis=0)
        else: ref_sig = np.zeros((fixed_signal_length,))
        
        # Put the reference signal into the new segment
        car_sig = OE.AnalogSignal(name='Reference Signal',
            signal=ref_sig,
            info='CAR calculated from good channels for this segment',
            sampling_rate=fixed_sampling_rate,
            t_start=siglist[0].t_start,
            id_segment=new_seg.id)
        session.add(car_sig)

        # Put all the referenced signals in id_car_seg
        for sig in siglist:
            # Subtract the CAR and filter, then error check
            referenced_signal = sig.signal - car_sig.signal
            filtered_signal = self.filterer.filter(referenced_signal)
            if np.isnan(filtered_signal).any():
                print "ERROR: Filtered signal contains NaN!"
            if np.isinf(filtered_signal).any():
                print "ERROR: Filtered signal contains Inf!"
            
            # Store in db
            new_sig = OE.AnalogSignal(\
                name=sig.name,
                signal=filtered_signal,
                id_segment=new_seg.id,
                channel=sig.channel,
                t_start=sig.t_start,
                sampling_rate=sig.sampling_rate,
                id_recordingpoint=ch2rpid[sig.channel])
            session.add(new_sig)
        
        # Save
        session.add(new_seg)
        session.commit()

    
    def avg_over_list_of_events(self, event_list, chn, meth='avg', t_start=None, t_stop=None):
        """Given a list of Event and a channel number, returns average signal.
        
        event_list : list of OE Event, ordered by id_segment
        chn : channel number
        session : the OE session from which you acquired the list of events
        
        Currently it's required that the list of Event be sorted by
        id_segment. For some reason if I sort it here, it causes an error.
        It's also required that only no two Event come from the same Segment.
        
        All AnalogSignal from channel `chn` containing an Event in event_list
        will be averaged together, triggered on the time of the Event.
        
        This function grabs the signals, error-checks, and then
        the actual work is done by a lower level function that averages
        AnalogSignal (but does not error check).
        """
        signal_list = self.get_signal_list_from_event_list(event_list, chn)
        
        # Call signal averaging function
        return self.avg_over_signals_with_triggers(
            signal_list, [e.time for e in event_list], meth=meth, 
            t_start=t_start, t_stop=t_stop)
    
    def avg_over_signals_with_triggers(self, signal_list, trigger_times, 
        t_start=None, t_stop=None, meth='avg'):
        """Averages list of AnalogSignal triggered on times.
        
        signal_list : list of AnalogSignal to be averaged
        trigger_times : trigger times, one per AnalogSignal
        t_start, t_stop : the returned average will go from t_start to
            t_stop, relative to trigger time. If None, then the maximum
            amount of time will be returned, which is set by the overlap
            of the time series contained in each signal.
        
        You must ensure that the trigger times are lined up correctly
        with AnalogSignal before calling this function, and that all
        AnalogSignal contain sufficient data for the t_start and t_stop
        you specify.
        
        Currently this function contains some paranoid error checking
        of the OE AnalogSignal.time_slice functionality.
        
        Returns tuple (return_t, avgsig) where return_t is an array
        of times (relative to trigger) and avgsig is the averaged value
        at those times.
        """
        #f = plt.figure()
        #ax = f.add_subplot(111)
        # If no times are provided, figure out maximum overlap
        if t_start is None:
            # Furthest back we can go
            t_start = np.max([sig.t_start - trigger \
                for sig, trigger in zip(signal_list, trigger_times)])
        if t_stop is None:
            # Furthest forward we can go
            t_stop = np.min([sig.t()[-1] - trigger \
                for sig, trigger in zip(signal_list, trigger_times)])        
        
        # Get all the slices in a list, get the returned times
        slices = [ ]
        return_t = None
        
        # Iterate through signals
        for sig, trigger in zip(signal_list, trigger_times):
            #assert(sig.segment._events[0].time == trigger)
            # Get time slice around trigger time and append to list
            #ax.plot(sig.t() - trigger, sig.signal)
            slc = sig.time_slice(trigger + t_start, trigger + t_stop)
            slices.append(slc.signal)
            
            # Error check return_t for consistency (paranoid)
            if return_t is None:
                return_t = slc.t() - trigger
            else:
                # The return_t must not differ by more than one sampling period
                assert len(return_t) == len(slc.t())
                assert np.all(\
                    (slc.t() - trigger - return_t) < (1./slc.sampling_rate))
        
        # Average and return
        avgsig = np.mean(slices, axis=0)
        #ax.plot(return_t, avgsig, 'k', lw=4)
        if meth == 'avg':
            return (return_t, avgsig)
        elif meth =='all':
            return (return_t, np.array(slices))
    
    def get_signal_list_from_event_list(self, event_list, chn):
        """Returns list of signals from specified channel and event.
        
        Returns list of AnalogSignal in the same order as event list
        from the correct channel.
        """        
        # Get signals from this channel
        signal_list = self.get_OE_session().query(OE.AnalogSignal).\
            filter(OE.AnalogSignal.channel == chn).all()
        
        # Order them in the same way that the events were ordered
        sig_idseg_list = [sig.id_segment for sig in signal_list]        
        ev_idseg_list = [e.id_segment for e in event_list]
        if len(np.unique(ev_idseg_list)) != len(ev_idseg_list):
            raise(ValueError("More than one of the specified event per segment"))        
        signal_list = [signal_list[sig_idseg_list.index(id_seg)] \
            for id_seg in ev_idseg_list]

        # Check alignment
        assert np.all(
            np.array([sig.id_segment for sig in signal_list]) ==
            np.array(ev_idseg_list)), "Mismatched segments and events, somehow"     
        
        return signal_list
    
    def spectrum(self, signal_list, NFFT=2**10, meth='avg_db',
        normalization=0.0, truncatepow2=False):
        """Returns the spectrum of a list of signals.
        
        The mean of each signal is subtracted before computing. The first
        component (DC) is dropped.
        
        signal_list : list of AnalogSignal to compute spectrum of
        meth :
            'avg_first' : average signals, compute spectrum, convert to dB
                not implemented
            'avg_db' : compute spectrum, convert to dB, average spectra
            'all' : return all spectra in dB without averaging
        normalization :
            multiply power in each bin by freq**normalization
            So a 1/f spectrum (in amplitude) is normalized by 2.0
        truncatepow2 : if True, truncate to length greatest power of two
            not implemented
        
        Returns:
        (Pxx, freqs)
        If meth is 'all', Pxx has shape (N_signals, NFFT/2)
        If meth is 'avg_db', Pxx has shape (NFFT/2,)
        """
        Pxx_list = []
        Fs = self.get_sampling_rate()
        for sig in signal_list:
            Pxx, freqs = mlab.psd(sig.signal - sig.signal.mean(), 
                Fs=Fs, NFFT=NFFT)
            # Drop DC
            Pxx = Pxx[1:]
            freqs = freqs[1:]
            
            # Normalize and add to list
            Pxx_list.append(Pxx.flatten() * freqs**normalization)
        
        if meth == 'all':
            return (10*np.log10(np.array(Pxx_list)), freqs)
        elif meth == 'avg_db':
            # Create list of spectra in db, discarding infs
            Pxx_list_db = []
            for Pxx in Pxx_list:
                Pxx_db = 10 * np.log10(Pxx)
                if not np.any(np.isinf(Pxx_db)): 
                    Pxx_list_db.append(Pxx_db)
            
            # Warn
            if len(Pxx_list_db) != len(Pxx_list):
                print "warning: discarding spectra with infs"
            
            # Average and return
            return (np.mean(np.array(Pxx_list_db), axis=0), freqs)

    def run_spikesorter(self, save_to_db=True, detection_kwargs=None,
        feature_kwargs=None, cluster_kwargs=None, save_to_klusters=None):
        """Sorts all groups in the database.
        
        Figures out which groups exist. Then calls another method to
        sort each group.
        
        If save_to_db is True, the sorted info will be written to the database.
        
        feature_kwargs : dict of arguments to be passed to
            spikesorter.computeFeatures, or None for RecordingSession defaults. 
            Default for unspecified keywords is the OE default, not the 
            RecordingSession default.
        cluster_kwargs : ditto for clustering
        
        detection_kwargs : similar, but defaults are different, see
            run_spikesorter_on_group
        
        save_to_klusters: if None, do nothing
            if True, will store Klusters output in a subdirectory with
            a name from self.next_klusters_dir()
            otherwise, whatever you pass will be used as :basename:
            in spikesorter.save_to_klusters()
        """
        session = self.get_OE_session()
        
        # Find groups
        group_list = list(np.unique([rp.group for rp in 
            session.query(OE.RecordingPoint).all()]))
        if None in group_list: group_list.remove(None)

        # Optionally, choose directory to save klusters output
        if save_to_klusters is True:
            newdir = self.next_klusters_dir()
            os.mkdir(newdir)
            save_to_klusters = os.path.join(newdir, self.session_name)

        # spike sort each group
        for group in group_list:
            self.run_spikesorter_on_group(group, save_to_db,
                detection_kwargs=detection_kwargs,
                feature_kwargs=feature_kwargs, cluster_kwargs=cluster_kwargs,
                save_to_klusters=save_to_klusters)
    
    def next_klusters_dir(self):
        """Returns full path to next available subdirectory like 'klusters%d'"""
        n = 0
        subdir = os.path.join(self.full_path, 'klusters%d' % n)    
        while os.path.exists(subdir) and os.path.isdir(subdir):
            n += 1
            subdir = os.path.join(self.full_path, 'klusters%d' % n)    
        return os.path.join(self.full_path, 'klusters%d' % n)

    def last_klusters_dir(self):
        """Returns full path to last existing subdirectory like 'klusters%d'
        
        If no such subdirectory exists, returns None.
        
        Note
        ----
        Start checking at klusters0 and gives up whenever the first missing
        directory is found. Thus if you have gaps in your subdirectories, it
        will give up at the first gap.
        """
        n = 0
        subdir = os.path.join(self.full_path, 'klusters%d' % n)    
        while os.path.exists(subdir) and os.path.isdir(subdir):
            n += 1
            subdir = os.path.join(self.full_path, 'klusters%d' % n)    
        
        if n == 0:
            # No such subdirectory
            return None
        else:
            return os.path.join(self.full_path, 'klusters%d' % (n-1))
    
    def run_spikesorter_on_group(self, group, save_to_db=True, 
        detection_kwargs=None, feature_kwargs=None, cluster_kwargs=None,
        save_to_klusters=None):
        """Run spike sorting on one group and return spikesorter object.
        
        Useful for playing around with the returned data.
        
        group : integer, number of group to sort
        save_to_db : if True, writes sorted info to database
        feature_kwargs : dict of keyword arguments to pass to OE
            computeFeatures. If None, then use the defaults as defined
            here, or in the feature method.
            Default for unspecified keywords is the OE default, not the 
            RecordingSession default.            
        cluster_kwargs : ditto for clustering.
        
        A slightly different syntax for detection that I think is more clear.
        A default dict for detection kwargs is defined in this function.
        This dict will be updated with the dict that you pass as
        `detection_kwargs`. That means that anything you don't specify will
        be set to the default in this method, rather than the OE default.
        Here are the defaults for `detection_kwargs`:
            'sign' : '-', 
            'median_thresh' : 4.5,
            'left_sweep' : 6/30000., 
            'right_sweep' : 17/30000.,
            'consistent_across_segments' : True,
            'consistent_across_channels' : False,
            'correct_times' : True       
        
        save_to_klusters : if None, do not save klusters output.
            Otherwise, should be a fully specified basename for
            spikesorter.save_to_klusters
        """
        printnow('running spikesorter on group %d' % group)
        # Default keyword arguments dict
        # Override OE algorithms defaults here, if desired.
        if feature_kwargs is None:
            feature_kwargs = {'output_dim': 8, 'start_sample': 0, 
                'num_samples': 0}
        if cluster_kwargs is None:
            cluster_kwargs = {'n': 8}
                
        # New syntax for default detection kwargs
        detection_kwargs_to_use = {
            'sign' : '-', 
            'median_thresh' : 4.5,
            'left_sweep' : 6/30000., 
            'right_sweep' : 17/30000.,
            'consistent_across_segments' : True,
            'consistent_across_channels' : False,
            'correct_times' : True }
        if detection_kwargs is not None:
            detection_kwargs_to_use.update(detection_kwargs)
        
        session = self.get_OE_session()
        
        # Get RecordingPoint on this group
        rp_list = session.query(OE.RecordingPoint).filter((
            OE.RecordingPoint.group == int(group)) and 
            (OE.RecordingPoint.id_block == rs.get_spike_block().id)).all()
        
        # Create spikesorter for this group
        spikesorter = OE.SpikeSorter(mode='from_filtered_signal', 
            session=session, recordingPointList=rp_list)

        # Call detection algorithm with the specified kwargs
        printnow('detection')
        spikesorter.computeDetectionEnhanced(\
            OE.detection.EnhancedMedianThreshold, 
            **detection_kwargs_to_use)     
        
        printnow('extraction')
        spikesorter.computeExtraction(OE.extraction.WaveformExtractor)        
        printnow('featuring')
        spikesorter.computeFeatures(OE.feature.PCA, **feature_kwargs)
        printnow('clustering')
        spikesorter.computeClustering(OE.clustering.KMean, **cluster_kwargs)
        
        if save_to_db:
            printnow('saving to db')
            spikesorter.save_to_db()        
        
        if save_to_klusters:
            spikesorter.save_to_klusters(basename=save_to_klusters)
    
        return spikesorter

    def prep_for_klusters(self, override_path=None, verbose=False, force=False,
        force_renumber=False):
        """Prepare fet and spk files for processing with Klusters.
        
        override_path : Full path to where klusters file exists.
            If None, use self.last_klusters_dir()
            
        force_renumber : Always increment group number.
            If False, guesses whether to do this based on the existence
            of a group 0.
        
        Renames to be 1-based instead of 0-based
        Creates xml file
        """
        
        if override_path is None:
            override_path = self.last_klusters_dir()
        kdir = override_path
        
        # Determine whether files are 1-based or 0-based
        fet0 = glob.glob(os.path.join(kdir, '*.fet.0'))
        clu0 = glob.glob(os.path.join(kdir, '*.clu.0'))
        
        if len(fet0) == 1 and len(clu0) == 1:
            renumber = True
        elif len(fet0) == 0 and len(clu0) == 0:
            renumber = False
        else:
            print "error: cannot renumber fet and clu files"
            return
        
        if force_renumber:
            renumber = True
        
        if renumber:
            # get file names and numbers
            fns = sorted(os.listdir(kdir), reverse=True)
            if 'backup' in fns: fns.remove('backup')
            fnums = map(lambda fn: int(fn.split('.')[-1]), fns)

            # Rename each one
            for fn, fnum in zip(fns, fnums):
                newfn = os.path.splitext(fn)[0] + '.%d' % (fnum + 1)
                assert not os.path.exists(os.path.join(kdir, newfn))
                
                os.rename(os.path.join(kdir, fn), os.path.join(kdir, newfn))
        
        newxmlname = os.path.join(kdir, self.session_name + '.xml')
        if not force and os.path.exists(newxmlname):
            if verbose:
                print "%s already exists" % newxmlname
        else:
            groups = self.read_channel_groups()
            self.write_klusters_xml_file(newxmlname, groups)

    def write_klusters_xml_file(self, filename, groups=None, nbits=16, 
        voltage_range=20, amplification=1000, offset=2048, n_samples=24, 
        peak_sample_index=11, n_features=8):
        """Writes the xml file with sorting parameters that Klusters needs.
        
        This includes information about waveforms and features. These will be
        associated with the fet and clu files in the same directory:
        the first channel group is associated with *.fet.1 and *.clu.1, etc.
        
        groups : a list of lists. Each sublist is the channel numbers in
            that group. eg [[1,2,3], [4,5,6,7]]
        
        Here I write the actual channel numbers but I'm not sure Klusters
        actually uses them. It probably just needs to know how many channels
        per group.
        """
        from lxml import etree

        # Acquisition system parameters, including the binary specification
        # for the spk files
        p = etree.Element("parameters")
        acs = etree.Element("acquisitionSystem")
        p.append(acs)
        el = etree.Element("nBits"); el.text = '16'
        acs.append(el)
        el = etree.Element("samplingRate")
        el.text = str(int(self.get_sampling_rate()))
        acs.append(el)
        el = etree.Element("voltageRange"); el.text = str(voltage_range)
        acs.append(el)
        el = etree.Element("amplification"); el.text = str(amplification)
        acs.append(el)
        el = etree.Element("offset"); el.text = str(offset)
        acs.append(el)
        
        # Channel groups
        sd = etree.Element("spikeDetection")
        cg = etree.Element("channelGroups")
        for group in groups:
            g = etree.Element("group")
            c = etree.Element("channels")
            for cnum in group:
                cc = etree.Element("channel")
                cc.text = str(cnum)
                c.append(cc)
            g.append(c)
            cg.append(g)
            
            el = etree.Element("nSamples"); el.text = str(n_samples)
            g.append(el)
            el = etree.Element("peakSampleIndex")
            el.text = str(peak_sample_index)
            g.append(el)
            el = etree.Element("nFeatures"); el.text = str(n_features)
            g.append(el)
        sd.append(cg)
        p.append(sd)        
        
        # Output xml
        fi = file(filename, 'w')
        fi.write(etree.tostring(p, pretty_print=True))
        fi.close()


    def spiketime_dump(self):
        # Dump in klustakwik format
        w = KlustaKwikIO.KlustaKwikIO(filename=os.path.join(
            self.full_path, self.session_name))    
        w.write_block(self.get_spike_block())        
    
    def get_sampling_rate(self):
        """Reads and returns sampling rate from ns5 file. Always 30K"""
        return FIXED_SAMPLING_RATE
        #return self.get_ns5_loader().header.f_samp
    
    def get_spiketrains_raw(self):
        """Reads neural data from directory and returns as spiketrains"""
        kkl = KlustaKwikIO.KK_loader(self.full_path)
        kkl.execute(group_multiplier=self.group_multiplier)
        return kkl.spiketrains        
    
    def get_spiketrains_centered(self, window='only hard'):
        """Returns spiketrains centered around timestamps
        
        if window is 'original':
            Spiketrain will assign spikes to original trials
        if window is 'only_hard':
            Spiketrain will assign spikes with hard limits of timestamps,
            and therefore trials will be of the same duration.
        
        Because this operates on the KlustaKwik files and the timestamps
        file, it can't be used to trigger on other events. Probably should
        be rewritten like avg_over_list_of_events to operate on event list,
        and take its data from the db rather than the flat files.

        That will make it easier to use original segment boundaries too.
        
        On the other hand this is currently an OE-free method.
        """
        spiketrain_dict = self.get_spiketrains_raw()
        
        # Add trial times. Currently requires positive window sizes,
        # meaning can't handle windows that don't include onset.
        if window == 'only hard':
            timestamps = self.read_timestamps()
            hard_limits_samples = [int(np.rint(x * self.get_sampling_rate())) 
                for x in self.read_time_limits()[1]]
            for spiketrain in spiketrain_dict.values():
                spiketrain.add_trial_info(onsets=timestamps, 
                    onset_trial_numbers=np.array(range(len(timestamps))),
                    pre_win=-hard_limits_samples[0],
                    post_win=hard_limits_samples[1])        
        elif window == 'original':            
            t_starts, t_stops = self.calculate_trial_boundaries()
            
            # this is going to require rewriting the add_trial_info method
            1/0
        else:
            print "warning: unrecognized window"
        
        return spiketrain_dict
    
    def get_psths(self, combine_units=False):
        """Read KlustaKwik format, calculates psths
        
        If unsorted, returns MUA PSTHs.
        If sorted, returns single unit PSTHs.
        If also_plot_all_spikes and sorted, will also return sum of all units.
        """
        sts = self.get_spiketrains_centered(window='only hard')
        psths = {}
        for groupnum, st in sts.items():
            if combine_units:
                psths[groupnum] = st.get_psth()
            else:
                psths[groupnum] = {}
                uids = st.get_unique_unit_IDs()
                if np.all(uids == np.array(None)) or len(uids) == 0:
                    print "no units, help"
                    1/0
                elif len(uids) == 1:
                    print "this is where code for MUA goes"
                    1/0
                else:                
                    for uid in uids:
                        psths[groupnum][uid] = st.get_psth(pick_units=[uid])
        
        return psths
    
    def get_trial_timing_from_db(self):
        """Returns number, start, stop, and center of each trial in samples.
        
        Returns:
            recarray with labels (trial, t_start, t_stop, t_center)
        
        Python indexing (half-open) is used.
        Does this by querying db. Could also run TrialSlicer on ns5 file.
        """
        fs = self.get_sampling_rate()
        t_nums, t_starts, t_stops = [], [], []
        warnt = False
        
        # Find all segments from spike block, ordered by id
        session = self.get_OE_session()

        # Get segment_id, t_start, and signal length from EVERY SIGNAL
        q2 = session.query(OE.AnalogSignal.id_segment,
            OE.AnalogSignal.t_start,
            OE.AnalogSignal.signal_shape)
        
        # Put results in array, converting to float (or else it foolishly
        # truncates the t_start to 3 decimals)
        sig_info = np.array(q2.all(), dtype=np.float)
        
        # Get segment_id and info field from EVERY SEGMENT (in spike block)
        # This also truncates integers to 3 decimal places!
        seg_info = np.array(session.query(OE.Segment.id, OE.Segment.info).
            filter(OE.Segment.block == self.get_spike_block()).all(), 
            dtype=np.int)
        
        # For each segment id, find the corresponding signals and extract
        # the trial start and trial duration
        trial_info = []
        for seg_id_int, seg_info_field in seg_info:
            matching_sigs = sig_info[np.rint(sig_info[:, 0]).astype(np.int) == seg_id_int]
            
            t_start = np.unique(np.rint(fs * matching_sigs[:, 1]).astype(np.int))
            assert len(t_start) == 1
            
            sig_shape = np.unique(np.rint(matching_sigs[:, 2]).astype(np.int))
            assert len(sig_shape) == 1            
            
            # Now append info field (trial number), trial start, and trial len            
            trial_info.append((seg_info_field, t_start[0], sig_shape[0]))
        
        # Now we have an array with trial number, start, and length
        trial_info_a = np.array(trial_info, dtype=np.int)
        
        # check for trials not in mat-file, often indicated by -99
        to_remove = (trial_info_a[:, 0] < 0)
        if np.sum(to_remove) > 0:
            print ("warning: %d trials missing in behavior" % np.sum(to_remove))
            trial_info_a = trial_info_a[~to_remove, :]
        
        # sort by trial numbers
        i = np.argsort(trial_info_a[:, 0])
        
        # assert that trials are ordered in time
        assert np.all(i == np.argsort(trial_info_a[:, 1]))
        
        # get onset times
        t_centers = self.read_timestamps()
        if len(to_remove) > 0:
            t_centers = t_centers[~to_remove]
        
        
        # assert stimulus is actually within the trial
        assert np.all(t_centers > trial_info_a[i, 1])
        
        return np.rec.fromarrays([
            trial_info_a[i, 0], # trial numbers
            trial_info_a[i, 1], # trial starts
            trial_info_a[i, 1] + trial_info_a[i, 2], # trial stops
            t_centers], # stimulus onsets
            names='trial,t_start,t_stop,t_center')


    
    def get_spike_picker(self, skip_trial_numbering=False,
        check_against_trial_slicer=True, override_path=None):
        """Returns a SpikePicker object for spike time analysis.
        
        skip_trial_numbering : if True, attempt to assign trial numbers
            to each spike, based on trial times. if False, spikes will not
            be assigned to trials.
        
        check_against_trial_slicer : if True, reslice trials and confirm
            that this matches the times of each segment. The original
            ns5 file must be available.     

        override_path : if True, load spiketrains from directory specified.
            This is useful if you want to recluster, but keep everything
            else the same. First checks for a subdirectory within RS;
            if this does not exist, loads full directory.
        """
        # Load spiketrains
        if override_path:
            # Check for subdirectory first, then full path
            subdir = os.path.join(self.full_path, override_path)
            if not (os.path.exists(subdir) and os.path.isdir(subdir)):
                subdir = override_path
            if not (os.path.exists(subdir) and os.path.isdir(subdir)):
                raise "specified path %s does not exist" % override_path

            kkl = KlustaKwikIO.KK_loader(subdir)
            kkl.execute(group_multiplier=self.group_multiplier)
            sts = kkl.spiketrains    
        else:
            # Use defaults
            sts = self.get_spiketrains_raw()
        
        # Convert to spike picker
        fs = self.get_sampling_rate()
        sp = SpikeTrainContainers.SpikePicker(sts, f_samp=fs)
        
        # Return now if no need to number trials
        if skip_trial_numbering:
            return sp
        
        # Iterate through segments and get trial numbers and times
        t_nums, t_starts, t_stops = [], [], []        
        warnt = False
        for n, seg in enumerate(self.get_spike_block()._segments):                        
            # Get trial label from info field (or raw number)
            if seg.info is not None:
                t_nums.append(int(seg.info))
            else:
                t_nums.append(n)
                warnt = True
            
            # Get times
            sig = seg._analogsignals[0]
            t_starts.append(int(np.rint(sig.t_start * fs)))
            t_stops.append(t_starts[-1] + len(sig.signal))
    
        # Warn if couldn't parse the info field
        if warnt:
            print "warning: auto assigned trial numbers"
        
        # Error check trial labels and times
        if len(t_nums) != len(np.unique(t_nums)):
            raise ValueError("Inconsistent trial labels")
        if not np.all(np.argsort(t_starts) == range(len(t_starts))):
            raise ValueError("unsorted trial starts")
        if not np.all(np.argsort(t_stops) == range(len(t_stops))):
            raise ValueError("unsorted trial stops")        
        
        # reslice and error check
        if check_against_trial_slicer:
            skip_assert = False
            try:
                t_starts2, t_stops2 = self.calculate_trial_boundaries()
            except IOError:
                print "warning: you requested trial checking but can't load ns5"
                skip_assert = True
            if not skip_assert:
                assert np.all(np.asarray(t_starts) == np.asarray(t_starts2))
                assert np.all(np.asarray(t_stops) == np.asarray(t_stops2))
        
        # assign trial number to each spike
        t_centers = self.read_timestamps()
        sp.assign_trial_numbers(t_nums, t_starts, t_stops, t_centers)
        
        return sp

    def run_klustakwik(self, subdir=None, processes=4, n_features=8, force=False):
        """Run KlustaKwik on spike subdirectory.
        
        subdir : if None, then the last subdirectory like 'klusters%d' will
        be used. Otherwise, specify the subdirectory name manually. Whatever
        you pass will be joined to self.full_path
        
        processes : how many independent KlustaKwik processes to spawn
        n_features : how many features from the fetfiles to use
            Right now this is not auto-detected from the feature file,
            because sometimes the feature file includes time as a feature,
            and so it's easier to specify exactly how many to use.
        force : if False and if *.klg.* files exist in the subdirectory,
            then returns without doing anything.
        """
        import multiprocessing, errno
        # Choose directory to run klustakwik
        if not subdir:
            subdir = self.last_klusters_dir()
        else:
            subdir = os.path.join(self.full_path, subdir)
        
        # Check if klustakwik has already been run
        klg_list = glob.glob(os.path.join(subdir, '*.klg.*'))
        if len(klg_list) > 0 and not force:
            print "klustakwik already ran in %s, continuing" % subdir
            return
        
        p = multiprocessing.Pool(processes)
        
        # Find groups to run on
        fetfilenames = sorted(glob.glob(os.path.join(subdir, '*.fet.*')))
        #fetfilenames = [os.path.split(fn)[1] for fn in fetfilenames]
        fetfilenumbers = [int(fn.split('.')[-1]) for fn in fetfilenames]
        basename = myutils.unique_or_error(
            [os.path.splitext(os.path.splitext(fn)[0])[0] 
                for fn in fetfilenames])
        
        for groupnumber in fetfilenumbers:
            p.apply_async(run_klustakwik_on_group, 
                (n_features, basename, groupnumber))
        p.close()
        
        # some funky code to make it work with ipython ctrl+c
        notintr = False
        while not notintr:
            try:
                p.join()
                notintr = True
            except OSError, ose:
                if ose.errno != errno.EINTR:
                    raise ose
        

    # Not sure the next 3 will ever be used for anything
    #~ def convert_neuron_name_to_neuron_id(self, neuron_name_list):
        #~ """Given a list of neuron names, returns id of each neuron"""
        #~ nid_list = []
        #~ q = self.get_OE_session().query(OE.Neuron)
        #~ for nname in neuron_name_list:
            #~ id = q.filter(OE.Neuron.name == nname).one().id
            #~ nid_list.append(id)
        
        #~ return nid_list

    #~ def get_neuron_name_list(self):
        #~ """Return a list of neuron names"""
        #~ session = self.get_OE_session()
        #~ neuron_list = session.query(OE.Neuron).all()
        #~ neuron_name_list = [neuron.name for neuron in neuron_list]
        #~ if None in neuron_name_list:
            #~ raise(ValueError("Some neurons named None!"))
        #~ if len(neuron_name_list) != len(np.unique(np.array(neuron_name_list))):
            #~ raise(ValueError("Some neurons have duplicate names"))
        
        #~ return neuron_name_list
    
    #~ def get_neuron_number_list(self):
        #~ """Returns a list of neuron numbers as integers.
        
        #~ Assumes neurons name match "Neuron %d *"
        #~ If not, raises error for malformed name.
        #~ Not necessarily the same as the neuron id!
        #~ """
        #~ nn_list = []
        #~ for nname in self.get_neuron_name_list():
            #~ m = glob.re.match('Neuron (\d+) ', nname)
            #~ if m is None:
                #~ raise(ValueError("Malformed neuron name: %s" % nname))
            #~ else:
                #~ nn_list.append(int(m.group(1)))
        #~ return nn_list


def run_klustakwik_on_group(n_features, basename, groupnumber,
    min_clusters=3, max_clusters=14):
    """Calls KlustaKwik on parameters"""
    feature_string = '1'*n_features + '0'
    #time.sleep(5)
    syscall = ' '.join([
        "KlustaKwik %s %d" % (basename, groupnumber),
        "-UseFeatures %s" % feature_string,
        "-MinClusters %d" % min_clusters,
        "-MaxClusters %d" % max_clusters,
        "-Screen 0"])
    printnow(syscall)
    os.system(syscall)
    printnow("group %d done" % groupnumber)


class RS_CR12B(RecordingSession):
    def __init__(self, *args, **kwargs):
        # Session name is corrupted with these guys because they
        # are in directories like 001, 002, etc.
        # Also super() only works with new-style classes
        #super(RS_CR12B, self).__init__(*args, **kwargs)
        RecordingSession.__init__(self, *args, **kwargs)
        self.session_name = self.full_path
    
    def calculate_trial_boundaries(self):
        t = self.read_timestamps()
        return (t-7500, t+15000)

    def get_spike_block(self):
        """Returns Block with spike block name, or None"""
        # Open database, get session
        #self.open_db()
        session = self.get_OE_session()
        
        # Check to see whether data has already been added
        q = session.query(OE.Block).filter(OE.Block.name == 'CAR Tetrode Data')
        if q.count() > 0:
            return q.one()   
        else:
            return None
    
    def get_db_filename(self):
        return glob.glob(os.path.join(self.full_path, '*.db'))[0]
