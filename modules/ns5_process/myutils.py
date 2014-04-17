import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import mlab
import matplotlib.pylab
import scipy.stats
import matplotlib
import wave
import struct
import os.path
import datetime
import scipy.io
import scipy.signal
from lxml import etree
Element = etree.Element
import kkpandas

longname = {'lelo': 'LEFT+LOW', 'rilo': 'RIGHT+LOW', 'lehi': 'LEFT+HIGH',
    'rihi': 'RIGHT+HIGH'}
LBPB_short2long = longname
LBPB_sns = list(range(1, 13))
LBPB_stimnames = [
    'lo_pc_go', 'hi_pc_no', 'le_lc_go', 'ri_lc_no',
    'le_hi_pc', 'ri_hi_pc', 'le_lo_pc', 'ri_lo_pc',
    'le_hi_lc', 'ri_hi_lc', 'le_lo_lc', 'ri_lo_lc']
LBPB_sn2name = {k: v for k, v in zip(LBPB_sns, LBPB_stimnames)}


def pickle_dump(obj, filename):
    import cPickle
    with file(filename, 'w') as fi:
        cPickle.dump(obj, fi)

def pickle_load(filename):
    import cPickle
    with file(filename) as fi:
        res = cPickle.load(fi)
    return res

def invert_linear_poly(p):
    return np.array([1, -p[1]]).astype(np.float) / p[0]

def soundsc(waveform, fs=44100, normalize=True):
    import scikits.audiolab as skab
    waveform = np.asarray(waveform)
    n = waveform.astype(np.float) / np.abs(waveform).max()
    skab.play(n, fs)


class GaussianSmoother:
    def __init__(self, filter_std=5, width=10, gain=1):
        self.filter_std = float(filter_std)
        self.width = float(width)
        self.gain = float(gain)
        
        self.n = np.arange(filter_std * width, dtype=np.int)
        self.b = np.exp( -(self.n.astype(np.float) ** 2) / (2 * filter_std**2) )
        self.b = self.gain * self.b / np.sqrt((self.b ** 2).sum())
        self.a = np.array([1])
    
    def execute(self, input_data, **filter_kwargs):
        input_data = np.asarray(input_data)
        res = scipy.signal.filtfilt(self.b, self.a, input_data, **filter_kwargs)
        return res
    



class ToneLoader:
    
    def __init__(self, filename=None):
        self.filename = filename
        self.data_dict = None
        
        if self.filename is not None:
            self._load()
    
    def _load(self):
        self._parse_date()
        self.data_dict = scipy.io.loadmat(self.filename)
    
    def _parse_date(self):
        """Split filename on _ and set date field"""
        split_fields = os.path.split(self.filename)[1].split('_')[1:]
        split_fields[-1] = os.path.splitext(split_fields[-1])[0]
        
        # convert strings to integers, will crash here if parsing wrong
        self.date_fields = map(int, split_fields)
        
        self.datetime = datetime.datetime(*self.date_fields)
    
    def aliased_tones(self, Fs=30e3, take_abs=True):
        """Returns the tone frequencies as they would appear aliased"""
        start = self.tones        
        aliased = np.mod(start + Fs/2., Fs) - Fs/2.
        
        if take_abs:
            return np.abs(aliased)
        else:
            return aliased
    
    @property
    def tones(self):
        if self.data_dict is None:
            self._load()
        return self.data_dict['tones'].flatten()
    
    @property
    def attens(self):
        if self.data_dict is None:
            self._load()
        return self.data_dict['attens'].flatten()
    
    def __repr__(self):
        return "ToneLoader('%s')" % self.filename


def parse_bitstream(bitstream, onethresh=.7 * 2**15, zerothresh=.3 * 2**15,
    min_interword=190, bitlen=10, certainty_thresh=7, nbits=16, 
    debug_mode=False):
    """Parse digital words from an analog trace and return times + values.
    
    This is for asynchronous digital communication, meaning a digital word
    is sent at unknown times. It is assumed that each word begins with a high
    bit ("1") to indicate when parsing should begin. Thereafter bits are read
    off in chunks of length `bitlen` and decoded to 1 or 0.
    
    Finally the `nbits` sequential bits are converted to an integer value
    by assuming LSB last, and subtracting the start bit. Example:
        1000000000000100 => 4
    
    Arguments
    ---------
    bitstream : analog trace
    onethresh : minimum value to decode a one
    zerothresh : maximum value to decode a zero
    min_interword : reject threshold crossings that occur more closely spaced 
        than this. Default is slightly longer than anticipated word duration 
        to avoid edge cases. Minimal error checking is done so this will not
        work for noisy signals -- spurious voltage transients could be decoded
        as zeros and potentially mask subsequent words within `min_interword`.
    bitlen : duration of each decoded bit, in samples
    certainty_thresh : number of samples per decoded bit necessary to decode
        it. That is, at least this many samples out of `bitlen` samples need
        to be above onethresh XOR below zerothresh. An error occurs if this
        threshold is not met.
    nbits : number of decoded bits per word
    debug_mode : plot traces of each word
    
    Returns times, numbers:
        times : times in samples of ONSET of each word
        numbers : value of each word
    """
    # Dot product the decoded bits with this to convert to integer
    wordconv = 2**np.arange(nbits, dtype=np.int)[::-1]

    # Threshold the signal
    ones = np.where(bitstream > onethresh)[0]
    #zeros = np.where(bitstream < zerothresh)[0] # never actually used?

    # Return empties if no ones found
    if len(ones) == 0:
        return np.array([]), np.array([])

    # Find when start bits occur, rejecting those within refractory period
    trigger_l = [ones[0]]
    for idx in ones[1:]:
        if idx > trigger_l[-1] + min_interword:
            trigger_l.append(idx)
    trial_start_times = np.asarray(trigger_l, dtype=np.int)

    # Plot if necessary
    if debug_mode:
        plt.figure()
        for trial_start in trial_start_times:
            plt.plot(bitstream[trial_start:trial_start+min_interword])
        plt.show()

    # Decode bits from each word
    trial_numbers = []
    for trial_start in trial_start_times:
        word = []
        # Extract one bit at a time
        for nbit in range(nbits):
            bitstr = bitstream[trial_start + nbit*bitlen + range(bitlen)]
            
            # Decode as 1, 0, or unknown
            if np.sum(bitstr > onethresh) > certainty_thresh:
                bitchoice = 1
            elif np.sum(bitstr < zerothresh) > certainty_thresh:
                bitchoice = 0
            else:
                bitchoice = -1
            word.append(bitchoice)
        
        # Fail if unknown bits occurred
        if -1 in word:
            1/0
        
        # Convert to integer
        val = np.sum(wordconv * np.array(word))
        trial_numbers.append(val)
    trial_numbers = np.asarray(trial_numbers, dtype=np.int)
    
    # Drop the high bit signal
    trial_numbers = trial_numbers - (2**(nbits-1))
    
    return trial_start_times, trial_numbers


def load_waveform_from_wave_file(filename, dtype=np.float, rescale=False,
    also_return_fs=False, never_flatten=False, mean_channel=False):
    """Opens wave file and reads, assuming signed shorts.
    
    if rescale, returns floats between -1 and +1
    if also_return_fs, returns (sig, f_samp); else returns sig
    if never_flatten, then returns each channel its own row
    if not never_flatten and only 1 channel, returns 1d array
    if mean_channel, return channel mean (always 1d)
    """
    wr = wave.Wave_read(filename)
    nch = wr.getnchannels()
    nfr = wr.getnframes()
    sig = np.array(struct.unpack('%dh' % (nfr*nch), wr.readframes(nfr)), 
        dtype=dtype)
    wr.close()
    
    # reshape into channels
    sig = sig.reshape((nfr, nch)).transpose()
    
    if mean_channel:
        sig = sig.mean(axis=0)
    
    if not never_flatten and nch == 1:
        sig = sig.flatten()
    
    if rescale:
        sig = sig / 2**15
    
    if also_return_fs:
        return sig, wr.getframerate()
    else:
        return sig   

def wav_write(filename, signal, dtype=np.int16, rescale=True, fs=8000):
    """Write wave file to filename.
    
    If rescale:
        Assume signal is between -1.0 and +1.0, and will multiply by maximum
        of the requested datatype. Otherwise `signal` will be converted
        to dtype as-is.
    
    If signal.ndim == 1:
        writes mono
    Elif signal.ndim == 2:
        better be stereo with each row a channel
    """
    if signal.ndim == 1:
        nchannels = 1
    elif signal.ndim == 2:
        nchannels = 2
        signal = signal.transpose().flatten()
    
    assert signal.ndim == 1, "only mono supported for now"
    assert dtype == np.int16, "only 16-bit supported for now"
    
    # rescale and convert signal
    if rescale:
        factor = np.iinfo(dtype).max
        sig = np.rint(signal * factor).astype(dtype)
    else:
        sig = sig.astype(dtype)
    
    # pack (don't know how to make this work for general dtype
    sig = struct.pack('%dh' % len(sig), *list(sig))
    
    # write to file
    ww = wave.Wave_write(filename)
    ww.setnchannels(nchannels)
    ww.setsampwidth(np.iinfo(dtype).bits / 8)
    ww.setframerate(fs)
    ww.writeframes(sig)
    ww.close()

def auroc(data1, data2, return_p=False):
    """Return auROC and two-sided p-value (if requested)"""
    try:
        U, p = scipy.stats.mannwhitneyu(data1, data2)
        p = p * 2
    except ValueError:
        print "some sort of error in MW"
        print data1
        print data2
        if return_p:
            return 0.5, 1.0
        else:
            return 0.5
    AUC  = 1 - (U / (len(data1) * len(data2)))

    if return_p:
        return AUC, p
    else:
        return AUC

def utest(x, y, return_auroc=False, print_mannwhitneyu_warnings=True,
    print_empty_data_warnings=True):
    """Drop-in replacement for scipy.stats.mannwhitneyu with different defaults
    
    If an error occurs in mannwhitneyu, this prints the message but returns
    a reasonable value: p=1.0, U=.5*len(x)*len(y), AUC=0
    
    print_mannwhitneyu_warnings : print warnings caught from underlying utest
    print_empty_data_warnings : print a warning of one of the data is empty
    
    This also calculates two-sided p-value and AUROC. The latter is only
    returned if return_auroc is True, so that the default is compatibility
    with mannwhitneyu.
    """
    badflag = False
    try:
        U, p = scipy.stats.mannwhitneyu(x, y)
        p = p * 2
    except ValueError as v:
        if print_mannwhitneyu_warnings:
            print "Caught exception:", v
        badflag = True

    # Calculate AUC
    if badflag:
        # Make up reasonable return values
        p = 1.0
        U = .5 * len(x) * len(y)
        AUC = .5
    elif len(x) == 0 or len(y) == 0:
        if print_empty_data_warnings:
            print "warning: one argument to mannwhitneyu is empty"
        AUC = .5
    else:
        AUC  = 1 - (U / (len(x) * len(y)))
    
    # Now return
    if return_auroc:
        return U, p, AUC
    else:
        return U, p

class UniqueError(Exception):
    pass

def unique_or_error(a):
    """Asserts that `a` contains only one unique value and returns it"""    
    u = np.unique(np.asarray(a))
    if len(u) == 0:
        raise UniqueError("no values found")
    if len(u) > 1:
        raise UniqueError("%d values found, should be one" % len(u))
    else:
        return u[0]

class OnlyOneError(Exception):
    pass

def only_one(l):
    """Returns the only value in l, or l itself if non-iterable."""
    # listify
    if not hasattr(l, '__len__'):
        l = [l]
    
    # check length
    if len(l) != 1:
        raise OnlyOneError("must contain exactly one value")
    
    # return entry
    return l[0]

def plot_with_trend_line(x, y, xname='X', yname='Y', ax=None):
    dropna = np.isnan(x) | np.isnan(y)
    x = x[~dropna]
    y = y[~dropna]
    
    if ax is None:    
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(x, y, '.')
    #p = scipy.polyfit(x.flatten(), y.flatten(), deg=1)    
    m, b, rval, pval, stderr = \
        scipy.stats.stats.linregress(x.flatten(), y.flatten())
    ax.plot([x.min(), x.max()], m * np.array([x.min(), x.max()]) + b, 'k:')
    plt.legend(['Trend r=%0.3f p=%0.3f' % (rval, pval)], loc='best')
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    plt.show()


def polar_plot_by_sound(Y, take_sqrt=False, normalize=False, ax=None, 
    yerr=None, **kwargs):
    """Y should have 4 columns in it, one for each sound."""
    if hasattr(Y, 'index'):
        YY = Y[['rihi', 'lehi', 'lelo', 'rilo']].values.transpose()
    else:
        YY = Y.transpose()
    
    if YY.ndim == 1:
        YY = YY[:, np.newaxis]

    if normalize:
        YY = np.transpose([row / row.mean() for row in YY.transpose()])

    YY[YY < 0.0] = 0.0
    if take_sqrt:
        YY = np.sqrt(YY)
    
    

    YYY = np.concatenate([YY, YY[0:1, :]])

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, polar=True)

    if yerr is None:
        ax.plot(np.array([45, 135, 225, 315, 405])*np.pi/180.0, YYY, **kwargs)
    else:
        ax.errorbar(x=np.array([45, 135, 225, 315, 405])*np.pi/180.0, y=YYY, 
            yerr=yerr)
    return ax


def prefidx(A, B):
    return (A - B) / (A + B)

class Spectrogrammer:
    """Turns a waveform into a spectrogram"""
    def __init__(self, NFFT=256, downsample_ratio=1, new_bin_width_sec=None,
        max_freq=40e3, min_freq=5e3, Fs=200e3, noverlap=None, normalization=0,
        detrend=matplotlib.pylab.detrend_mean, **kwargs):
        """Object to turn waveforms into spectrograms.
        
        This is a wrapper around mlab.specgram. What this object provides
        is slightly more intelligent parameter choice, and a nicer way
        to trade off resolution in frequency and time. It also remembers
        parameter choices, so that the same object can be used to batch
        analyze a bunch of waveforms using the `transform` method.
        
        Arguments passed to mlab.specgram
        ----------------------------------
        NFFT - number of points used in each segment
            Determines the number of frequency bins, which will be
            NFFT / 2 before stripping out those outside min_freq and max_freq
        
        noverlap - int, number of samples of overlap between segments
            Default is NFFT / 2
        
        Fs - sampling rate
        
        detrend - detrend each segment before FFT
            Default is to remove the mean (DC component)
        
        **kwargs - anything else you want to pass to mlab.specgram
        
        
        Other arguments
        ---------------
        downsample_ratio - int, amount to downsample in time
            After all other calculations are done, the temporal resolution
        
        new_bin_width_sec - float, target temporal resolution
            The returned spectrogram will have a temporal resolution as
            close to this as possible.
            If this is specified, then the downsample_ratio is adjusted
            as necessary to achieve it. If noverlap is left as default,
            it will try 50% first and then 0, to achieve the desired resolution.
            If it is not possible to achieve within a factor of 2 of this
            resolution, a warning is issued.
        
        normalization - the power in each frequency bin is multiplied by
            the frequency raised to this power.
            0 means do nothing.
            1 means that 1/f noise becomes white.
        
        min_freq, max_freq - discard frequencies outside of this range
        
        
        Returns
        -------
        Pxx - 2d array of power in dB. Shape (n_freq_bins, n_time_bins)
            May contain -np.inf where the power was exactly zero.
        
        freqs - 1d array of frequency bins
        
        t - 1d array of times
        
        
        Theory
        ------
        The fundamental tradeoff is between time and frequency resolution and
        is set by NFFT.
        
        For instance, consider a 2-second signal, sampled at 1024Hz, chosen
        such that the number of samples is 2048 = 2**11.
        *   If NFFT is 2048, you will have 1024 frequency bins (spaced 
            between 0KHz and 0.512KHz) and 1 time bin. 
            This is a simple windowed FFT**2, with the redundant negative
            frequencies discarded since the waveform is real.
            Note that the phase information is lost.
        *   If NFFT is 256, you will have 128 frequency bins and 8 time bins.
        *   If NFFT is 16, you will have 8 freqency bins and 128 time bins.
        
        In each case the FFT-induced trade-off is:
            n_freq_bins * n_time_bins_per_s = Fs / 2
            n_freq_bins = NFFT / 2
        
        So far, using only NFFT, we have traded off time resolution for
        frequency resolution. We can achieve greater noise reduction with
        appropriate choice of noverlap and downsample_ratio. The PSD
        function achieves this by using overlapping segments, then averaging
        the FFT of each segment. The use of noverlap in mlab.specgram is 
        a bit of a misnomer, since no temporal averaging occurs there!
        But this object can reinstate this temporal averaging.
        
        For our signal above, if our desired temporal resolution is 64Hz,
        that is, 128 samples total, and NFFT is 16, we have a choice.
        *   noverlap = 0. Non-overlapping segments. As above, 8 frequency
            bins and 128 time bins. No averaging
        *   noverlap = 64. 50% overlap. Now we will get 256 time bins.
            We can then average together each pair of adjacent bins
            by downsampling, theoretically reducing the noise. Note that
            this will be a biased estimate since the overlapping windows
            are not redundant.
        *   noverlap = 127. Maximal overlap. Now we will get about 2048 bins,
            which we can then downsample by 128 times to get our desired
            time resolution.
        
        The trade-off is now:
            overlap_factor = (NFFT - overlap) / NFFT
            n_freq_bins * n_time_bins_per_s * overlap_factor = Fs / downsample_ratio / 2
        
        Since we always do the smoothing in the time domain, n_freq bins = NFFT / 2
        and the tradeoff becomes
            n_time_bins_per_s = Fs / downsample_ratio / (NFFT - overlap)
        
        That is, to increase the time resolution, we can:
            * Decrease the frequency resolution (NFFT)
            * Increase the overlap, up to a maximum of NFFT - 1
              This is a sort of spurious improvement because adjacent windows
              are highly correlated.
            * Decrease the downsample_ratio (less averaging)
        
        To decrease noise, we can:
            * Decrease the frequency resolution (NFFT)
            * Increase the downsample_ratio (more averaging, fewer timepoints)
        
        How to choose the overlap, or the downsample ratio? In general,
        50% overlap seems good, since we'd like to use some averaging, but
        we get limited benefit from averaging many redundant samples.        
        
        This object tries for 50% overlap and adjusts the downsample_ratio
        (averaging) to achieve the requested temporal resolution. If this is
        not possible, then no temporal averaging is done (just like mlab.specgram)
        and the overlap is increased as necessary to achieve the requested
        temporal resolution.
        """
        self.downsample_ratio = downsample_ratio # until set otherwise
        
        # figure out downsample_ratio
        if new_bin_width_sec is not None:
            # Set noverlap to default
            if noverlap is None:
                # Try to do it with 50% overlap
                noverlap = NFFT / 2
            
            # Calculate downsample_ratio to achieve this
            self.downsample_ratio = \
                Fs * new_bin_width_sec / float(NFFT - noverlap)
            
            # If this is not achievable, then try again with minimal downsampling
            if np.rint(self.downsample_ratio).astype(np.int) < 1:
                self.downsample_ratio = 1
                noverlap = np.rint(NFFT - Fs * new_bin_width_sec).astype(np.int)
            
        # Convert to nearest int and test if possible
        self.downsample_ratio = np.rint(self.downsample_ratio).astype(np.int)        
        if self.downsample_ratio == 0:
            print "requested temporal resolution too high, using maximum"
            self.downsample_ratio = 1
    
        # Default value for noverlap if still None
        if noverlap is None:
            noverlap = NFFT / 2
        self.noverlap = noverlap
        
        # store other defaults
        self.NFFT = NFFT
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.Fs = Fs
        self.normalization = normalization
        self.detrend = detrend
        self.specgram_kwargs = kwargs

    
    def transform(self, waveform):
        """Converts a waveform to a suitable spectrogram.
        
        Removes high and low frequencies, rebins in time (via median)
        to reduce data size. Returned times are the midpoints of the new bins.
        
        Returns:  Pxx, freqs, t    
        Pxx is an array of dB power of the shape (len(freqs), len(t)).
        It will be real but may contain -infs due to log10
        """
        # For now use NFFT of 256 to get appropriately wide freq bands, then
        # downsample in time
        Pxx, freqs, t = mlab.specgram(waveform, NFFT=self.NFFT, 
            noverlap=self.noverlap, Fs=self.Fs, detrend=self.detrend, 
            **self.specgram_kwargs)
        
        # Apply the normalization
        Pxx = Pxx * np.tile(freqs[:, np.newaxis] ** self.normalization, 
            (1, Pxx.shape[1]))

        # strip out unused frequencies
        Pxx = Pxx[(freqs < self.max_freq) & (freqs > self.min_freq), :]
        freqs = freqs[(freqs < self.max_freq) & (freqs > self.min_freq)]

        # Rebin in size "downsample_ratio". If last bin is not full, discard.
        Pxx_rebinned = []
        t_rebinned = []
        for n in range(0, len(t) - self.downsample_ratio + 1, 
            self.downsample_ratio):
            Pxx_rebinned.append(
                np.median(Pxx[:, n:n+self.downsample_ratio], axis=1).flatten())
            t_rebinned.append(
                np.mean(t[n:n+self.downsample_ratio]))

        # Convert to arrays
        Pxx_rebinned_a = np.transpose(np.array(Pxx_rebinned))
        t_rebinned_a = np.array(t_rebinned)

        # log it and deal with infs
        Pxx_rebinned_a_log = -np.inf * np.ones_like(Pxx_rebinned_a)
        Pxx_rebinned_a_log[np.nonzero(Pxx_rebinned_a)] = \
            10 * np.log10(Pxx_rebinned_a[np.nonzero(Pxx_rebinned_a)])


        self.freqs = freqs
        self.t = t_rebinned_a
        return Pxx_rebinned_a_log, freqs, t_rebinned_a

def set_fonts_big(undo=False):
    if not undo:
        matplotlib.rcParams['font.size'] = 16.0
        matplotlib.rcParams['xtick.labelsize'] = 'medium'
        matplotlib.rcParams['ytick.labelsize'] = 'medium'
    else:
        matplotlib.rcParams['font.size'] = 10.0
        matplotlib.rcParams['xtick.labelsize'] = 'small'
        matplotlib.rcParams['ytick.labelsize'] = 'small'

def my_imshow(C, x=None, y=None, ax=None, cmap=plt.cm.RdBu_r, clim=None,
    center_clim=False):
    """Wrapper around imshow with better defaults.
    
    Plots "right-side up" with the first pixel (0, 0) in the upper left,
    not the lower left. So it's like an image or a matrix, not like
    a graph.
    
    Other changes to the imshow defaults:
        extent : the limits should match the data, not the number of pixels
        interpolation : 'nearest' instead of smoothed
        axis : 'auto' so that it fits the available space instead of
            constraining the pixels to be square
    
    x : numerical labels for the columns
    y : numerical labels for the rows
        if None, the indexes are used, accounting for the y-axis flip
    center_clim : 
    
    Return the image
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    if x is None:
        x = np.array(range(C.shape[1]))
    if y is None:
        y = np.array(range(C.shape[0]))
    extent = x[0], x[-1], y[0], y[-1]
    #plt.imshow(np.flipud(C), interpolation='nearest', extent=extent, cmap=cmap)
    im = ax.imshow(np.flipud(C), interpolation='nearest', extent=extent, cmap=cmap)
    ax.axis('auto')
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    
    # Account for y-axis flip
    ax.set_yticklabels([int(v) for v in ax.get_yticks()[::-1]])
    
    if clim is not None:
        im.set_clim(clim)
    
    if center_clim:
        cl = np.asarray(im.get_clim())
        im.set_clim(-np.abs(cl).max(), np.abs(cl).max())
    
    return im
    
    #plt.show()

def harmonize_clim_in_subplots(fig=None, axa=None, center_clim=False, trim=1):
    """Set clim to be the same in all subplots in figur
    
    f : Figure to grab all axes from, or None
    axa : the list of subplots (if f is None)
    center_clim : if True, the mean of the new clim is always zero
    trim : does nothing if 1 or None
        otherwise, sets the clim to truncate extreme values
        for example, if .99, uses the 1% and 99% values of the data
    """
    # Which axes to operate on
    if axa is None:
        axa = fig.get_axes()
    axa = np.asarray(axa)

    # Two ways of getting new clim
    if trim is None or trim == 1:
        # Get all the clim
        all_clim = []        
        for ax in axa.flatten():
            for im in ax.get_images():
                all_clim.append(np.asarray(im.get_clim()))
        
        # Find covering clim and optionally center
        all_clim_a = np.array(all_clim)
        new_clim = (np.min(all_clim_a[:, 0]), np.max(all_clim_a[:, 1]))
    else:
        # Trim to specified prctile of the image data
        data_l = []
        for ax in axa.flatten():
            for im in ax.get_images():
                data_l.append(np.asarray(im.get_array()).flatten())
        data_a = np.concatenate(data_l)
        
        # New clim
        new_clim = mlab.prctile(data_a, (100.*(1-trim), 100.*trim))
    
    # Optionally center
    if center_clim:
        new_clim = np.max(np.abs(new_clim)) * np.array([-1, 1])
    
    # Set to new value
    for ax in axa.flatten():
        for im in ax.get_images():
            im.set_clim(new_clim)
    
    return new_clim

def iziprows(df):
   series = [df[col] for col in df.columns]
   series.insert(0, df.index)
   return itertools.izip(*series)

def list_intersection(l1, l2):
    return list(set.intersection(set(l1), set(l2)))

def list_union(l1, l2):
    return list(set.union(set(l1), set(l2)))


def parse_space_sep(s, dtype=np.int):
    """Returns a list of integers from a space-separated string"""
    s2 = s.strip()
    if s2 == '':
        return []    
    else:
        return [dtype(ss) for ss in s2.split()]

def r_adj_pval(a, meth='BH'):
    import rpy2.robjects as robjects
    r = robjects.r
    robjects.globalenv['unadj_p'] = robjects.FloatVector(
        np.asarray(a).flatten())
    return np.array(r("p.adjust(unadj_p, '%s')" % meth)).reshape(a.shape)


def std_error(data, axis=None):
    if axis is None:
        N = len(data)
    else:
        N = np.asarray(data).shape[axis]
    
    return np.std(np.asarray(data), axis) / np.sqrt(N)

def printnow(s):
    """Write string to stdout and flush immediately"""
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()

def plot_mean_trace(ax=None, data=None, x=None, errorbar=True, axis=0, **kwargs):
    
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    
    data = np.asarray(data)
    
    if np.min(data.shape) == 1:
        data = data.flatten()
    if data.ndim == 1:
        single_trace = True
        errorbar = False
        
        if x is None:
            x = range(len(data))
    else:
        single_trace = False
        
        if x is None:
            x = range(len(np.mean(data, axis=axis)))
    
    if single_trace:
        ax.plot(x, data, **kwargs)
    else:
        if errorbar:
            ax.errorbar(x=x, y=np.mean(data, axis=axis),
                yerr=std_error(data, axis=axis), **kwargs)
        else:
            ax.plot(np.mean(data, axis=axis), **kwargs)

def plot_asterisks(pvals, ax, x=None, y=0, yd=1, levels=None):
    pvals = np.asarray(pvals)
    if levels is None:
        levels = [.05, .01, .001, .0001]
    if x is None:
        x = range(len(pvals))
    x = np.asarray(x)

    already_marked = np.zeros(len(pvals), dtype=np.bool)
    for n, l in enumerate(levels):
        msk = (pvals < l)# & ~already_marked
        if np.any(msk):
            ax.plot(x[np.where(msk)[0]], n*yd + y * np.ones_like(np.where(msk)[0]),
                marker='*', color='k', ls='None')
        #already_marked = already_marked | msk
    plt.show()

def times2bins_int(times, f_samp=1.0, t_start=None, t_stop=None):
    """Returns a 1-0 type spiketrain from list of times.
    
    Note the interface is different from times2bins, which is for
    general histogramming. This function expects times in seconds, and uses
    f_samp to convert to bins. The other function expects times in samples,
    and uses f_samp to convert to seconds.
    
    If multiple spikes occur in same bin, you still get 1 ... not sure
    this is right .... Essentially you're getting a boolean
    
    'times' : seconds
    'f_samp' : sampling rate of returned spike train
    """
    f_samp = float(f_samp)
    if t_stop is None:
        t_stop = times.max() + 1/f_samp
    if t_start is None:
        t_start = times.min()
    
    # set up return variable
    len_samples = np.rint(f_samp * (t_stop - t_start)).astype(np.int)
    res = np.zeros(len_samples, dtype=np.int)
    
    # set up times as indexes
    times = times - t_start
    times_samples = np.rint(times * f_samp).astype(np.int)
    times_samples = times_samples[~(
        (times_samples < 0) | (times_samples >= len(res)))]
    
    # set res
    res[times_samples] = 1
    return res
    
    

def times2bins(times, f_samp=None, t_start=None, t_stop=None, bins=10,
    return_t=False):
    """Given event times and sampling rate, convert to histogram representation.
    
    If times is list-of-list-like, will return list-of-list-like result.
    
    f_samp : This is for the case where `times` is in samples and you want
        a result in seconds. That is, times is divided by this value.
    
    Returns: res[, t_vals]
    Will begin at t_start and continue to t_stop
    """
    
    # dimensionality
    is2d = True
    try:
        len(times[0])
    except (TypeError, IndexError):
        is2d = False

    # optionally convert units    
    if f_samp is not None:
        if is2d:
            times = np.array([t / f_samp for t in times])
        else:
            times = np.asarray(times) / f_samp
    
    # defaults for time
    if is2d:
        if t_start is None:
            t_start = min([x.min() for x in times])
        if t_stop is None:
            t_stop = max([x.max() for x in times])
    else:
        if t_start is None:
            t_start = times.min()
        if t_stop is None:
            t_stop = times.max()

    # determine spacing of time bins
    t_vals = np.linspace(t_start, t_stop, bins + 1)

    # histogram
    if not is2d:
        res = np.histogram(times, bins=t_vals)[0]
    else:
        res = np.array([np.histogram(x, bins=t_vals)[0] for x in times])

    if return_t:
        return res, t_vals
    else:
        return res

def plot_rasters(obj, ax=None, full_range=1.0, y_offset=0.0, plot_kwargs=None):
    """Plots raster of spike times or psth object.
    
    obj : PSTH object, or array of spike times (in seconds)
    ax : axis object to plot into
    plot_kwargs : any additional plot specs. Defaults:
        if 'color' not in plot_kwargs: plot_kwargs['color'] = 'k'
        if 'ms' not in plot_kwargs: plot_kwargs['ms'] = 4
        if 'marker' not in plot_kwargs: plot_kwargs['marker'] = '|'
        if 'ls' not in plot_kwargs: plot_kwargs['ls'] = 'None'    
    full_range: y-value of top row (last trial), default 1.0
    
    Assumes that spike times are aligned to each trial start, and uses
    this to guess where trial boundaries are: any decrement in spike
    time is assumed to be a trial boundary. This will sometimes lump
    trials together if there are very few spikes.
    """
    # get spike times
    try:
        ast = obj.adjusted_spike_times / float(obj.F_SAMP)
    except AttributeError:
        ast = obj    
    
    try:
        # is it already folded?
        len(ast[0])
        folded_spike_times = ast
    except TypeError:
        # need to fold
        # convert into list representation
        folded_spike_times = fold_spike_times(ast)

    # build axis
    if ax is None:
        f = plt.figure(); ax = f.add_subplot(111)
    
    # plotting defaults
    if plot_kwargs is None:
        plot_kwargs = {}
    if 'color' not in plot_kwargs: plot_kwargs['color'] = 'k'
    if 'ms' not in plot_kwargs: plot_kwargs['ms'] = 4
    if 'marker' not in plot_kwargs: plot_kwargs['marker'] = '|'
    if 'ls' not in plot_kwargs: plot_kwargs['ls'] = 'None'
    
    if full_range is None:
        full_range = float(len(folded_spike_times))
    
    for n, trial_spikes in enumerate(folded_spike_times):
        ax.plot(trial_spikes, 
            y_offset + np.ones(trial_spikes.shape, dtype=np.float) * 
            n / float(len(folded_spike_times)) * full_range,
            **plot_kwargs)

def histogram_pvals(eff, p, bins=20, thresh=.05, ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    if np.sum(p > thresh) == 0:
        ax.hist(eff[p<=thresh], bins=bins, histtype='barstacked', color='r')    
    elif np.sum(p < thresh) == 0:
        ax.hist(eff[p>thresh], bins=bins, histtype='barstacked', color='k')            
    else:
        ax.hist([eff[p>thresh], eff[p<=thresh]], bins=bins, 
            histtype='barstacked', color=['k', 'r'], rwidth=1.0)    
    plt.show()

def sort_df_by_col(a, col):
    return a.ix[a.index[np.argsort(np.asarray(a[col]))]]

def pick_mask(df, **kwargs):
    """Returns mask of df, s.t df[mask][key] == val for key, val in kwargs
    """
    mask = np.ones(len(df), dtype=np.bool)
    for key, val in kwargs.items():
        mask = mask & (df[key] == val)
    
    return mask

def pick_count(df, **kwargs):
    return np.sum(pick_mask(df, **kwargs))


def polar_plot_by_sound(Y, take_sqrt=False, normalize=False, ax=None, **kwargs):
    """Y should have 4 columns in it, one for each sound."""
    if hasattr(Y, 'index'):
        YY = Y[['rihi', 'lehi', 'lelo', 'rilo']].values.transpose()
    else:
        YY = Y.transpose()
    
    if YY.ndim == 1:
        YY = YY[:, np.newaxis]

    if normalize:
        YY = np.transpose([row / row.mean() for row in YY.transpose()])

    YY[YY < 0.0] = 0.0
    if take_sqrt:
        YY = np.sqrt(YY)
    
    

    YYY = np.concatenate([YY, YY[0:1, :]])

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, polar=True)

    ax.plot(np.array([45, 135, 225, 315, 405])*np.pi/180.0, YYY, **kwargs)
    
    
    plt.xticks(np.array([45, 135, 225, 315, 405])*np.pi/180.0,
        ['RIGHT+HIGH', 'LEFT+HIGH', 'LEFT+LOW', 'RIGHT+LOW'])

def map_d(func, dic):
    """Map like func(val) for items in dic and maintain keys"""
    return dict([(key, func(val)) for key, val in dic.items()])

def filter_d(cond, dic):
    """Filter by cond(val) for items in dic and maintain keys"""
    return dict([(key, val) for key, val in dic.items() if cond(val)])
    

def getstarted():
    """Load all my data into kkpandas and RS objects
    
    Returns:
    xmlfiles, kksfiles, data_dirs, xml_roots, well_sorted_units, kk_servers, \
        dd_onset_windows, non_audresp_units = myutils.getstarted()
    
    dd_onset_windows : dict {ratname : dict { ulabel : array of onset window}}
        This one might change ... not sure this is the best way to handle this.
        First of all, seems like it should be loaded from disk, not hard-coded.
        Second, is the choice of ulabel as session_name-unum temporary or
        permanent?
    
    Each is a dict keyed by ratname
    """
    import warnings
    import my.dataload
    warnings.warn('deprecated, see my.dataload')
    gets = my.dataload.getstarted()
    res = [gets[k] for k in ['xmlfiles', 'kksfiles', 'data_dirs', 'xml_roots',
        'manual_units', 'kk_servers']]
    res += [None, None]
    return res

def load_channel_mat(filename, return_dataframe=True, dump_filler=True):
    """Load data from bao-lab format for single channel
    
    filename : path to file that has been broken out into each channel
        separately
    
    Each mat-file contains structured array like this:
    ans = 

                LFP: [1x763 single]
               PDec: [1x763 int16]
                 CH: [1x1 struct]
        Epoch_Value: [1 20 55]
    --- data.trial(1).CH
    ---- latency, spikewaveform    
    
    Returns: trials_info, spike_times
        trials_info : array of shape (n_trials, 3)
            if return_dataframe, this is a pandas.DataFrame instead
        spike_times : list of length n_trials, each containing array of
            trial-locked spike times.
    """
    # Load the structured array 'trial', with length n_trials
    data = scipy.io.loadmat(filename, squeeze_me=True)    
    trial = data['data']['trial'].item()
    
    # Here is the spikes from each trial
    # The purpose of the extra `flatten` is to ensure that arrays containing
    # 0 or 1 spike time are always 1-dimensional instead of 0-dimensional
    spike_times = map(lambda t: t.item()[2]['latency'].item().flatten(), trial) 

    # Here is the information about each trial
    trials_info = np.array(map(lambda t: t.item()[3], trial))
    
    # Remove trials containing nothing useful, ie delay 55ms or atten 70
    if dump_filler:
        bad_mask = (
            (trials_info[:, 2] == 55) |
            (trials_info[:, 1] == 70))
        trials_info = trials_info[~bad_mask]
        spike_times = list(np.asarray(spike_times)[~bad_mask])
    
    # Make a DataFrame
    if return_dataframe:
        trials_info = pandas.DataFrame(trials_info, 
            columns=['freq', 'atten', 'light'])    
    
    return trials_info, spike_times


def test_vs_baseline(data, baseline_idxs, test_idxs, debug_figure=False, 
    bins=10):
    raise DeprecationError("moved to my.peakpick")


def means_tester(d0, d1):
    return np.mean(d1) - np.mean(d0)

def keep(d0, d1):
    return (d0, d1)

class BootstrapError(BaseException):
    pass

def CI_compare(CI1, CI2):
    """Return +1 if CI1 > CI2, -1 if CI1 < CI2, 0 if overlapping"""
    if CI1[1] < CI2[0]:
        return -1
    elif CI2[1] < CI1[0]:
        return +1
    else:
        return 0

def simple_bootstrap(data, n_boots=1000, min_bucket=20):
    if len(data) < min_bucket:
        raise BootstrapError("too few samples")
    
    res = []
    data = np.asarray(data)
    for boot in range(n_boots):
        idxs = np.random.randint(0, len(data), len(data))
        draw = data[idxs]
        res.append(np.mean(draw))
    res = np.asarray(res)
    CI = mlab.prctile(res, (2.5, 97.5))
    
    return res, res.mean(), CI

def difference_CI_bootstrap_wrapper(data, **boot_kwargs):
    """Given parsed data from single ulabel, return difference CIs.
    
    data : same format as bootstrap_main_effect expects
    
    Will calculate the following statistics:
        means : mean of each condition, across draws
        CIs : confidence intervals on each condition
        mean_difference : mean difference between conditions
        difference_CI : confidence interval on difference between conditions
        p : two-tailed p-value of 'no difference'
    
    Returns:
        dict of those statistics
    """
    # Yields a 1000 x 2 x N_trials matrix:
    # 1000 draws from the original data, under both conditions.
    bh = bootstrap_main_effect(data, meth=keep, **boot_kwargs)

    # Find the distribution of means of each draw, across trials
    # This is 1000 x 2, one for each condition
    # hist(means_of_all_draws) shows the comparison across conditions
    means_of_all_draws = bh.mean(axis=2)

    # Confidence intervals across the draw means for each condition
    condition_CIs = np.array([
        mlab.prctile(dist, (2.5, 97.5)) for dist in means_of_all_draws.T])

    # Means of each ulabel (centers of the CIs, basically)
    condition_means = means_of_all_draws.mean(axis=0)

    # Now the CI on the *difference between conditions*
    difference_of_conditions = np.diff(means_of_all_draws).flatten()
    difference_CI = mlab.prctile(difference_of_conditions, (2.5, 97.5)) 

    # p-value of 0. in the difference distribution
    cdf_at_value = np.sum(difference_of_conditions < 0.) / \
        float(len(difference_of_conditions))
    p_at_value = 2 * np.min([cdf_at_value, 1 - cdf_at_value])
    
    # Should probably floor the p-value at 1/n_boots

    return {'p' : p_at_value, 
        'means' : condition_means, 'CIs': condition_CIs,
        'mean_difference': difference_of_conditions.mean(), 
        'difference_CI' : difference_CI}

def bootstrap_main_effect(data, n_boots=1000, draw_meth='equal', meth=None,
    min_bucket=5):
    """Given 2xN set of data of unequal sample sizes, bootstrap main effect.

    We will generate a bunch of fake datasets by resampling from data.
    Then we combine across categories. The total number of data points
    will be the same as in the original dataset; however, the resampling
    is such that each category is equally represented.    
    
    data : list of length N, each entry a list of length 2
        Each entry in `data` is a "category".
        Each category consists of two groups.
        The question is: what is the difference between the groups, without
        contaminating by the different size of each category?
    
    n_boots : number of times to randomly draw, should be as high as you
        can stand
    
    meth : what to apply to the drawn samples from each group
        If None, use means_tester
        It can be any function that takes (group0, group1)
        Results of every call are returned

    Returns:
        np.asarray([meth(group0, group1) for group0, group1 in each boot])    
    """    
    if meth is None:
        meth = means_tester
    
    # Convert to standard format
    data = [[np.asarray(d) for d in dd] for dd in data]
    
    # Test
    alld = np.concatenate([np.concatenate([dd for dd in d]) for d in data])
    if len(np.unique(alld)) == 0:
        raise BootstrapError("no data")
    elif len(np.unique(alld)) == 1:
        raise BootstrapError("all data points are identical")
    
    # How many to generate from each group, total
    N_group0 = np.sum([len(category[0]) for category in data])
    N_group1 = np.sum([len(category[1]) for category in data])
    N_categories = len(data)
    
    # Which categories to draw from
    res_l = []
    for n_boot in range(n_boots):
        # Determine the representation of each category
        # Randomly generating so in the limit each category is equally
        # represented. Alternatively we could use fixed, equal representation,
        # but then we need to worry about rounding error when it doesn't
        # divide exactly evenly.
        fakedata_category_label_group0 = np.random.randint(  
            0, N_categories, N_group0)
        fakedata_category_label_group1 = np.random.randint(
            0, N_categories, N_group1)
        
        # Draw the data, separately by each category
        fakedata_by_group = [[], []]
        for category_num in range(N_categories):
            # Group 0
            n_draw = np.sum(fakedata_category_label_group0 == category_num)
            if len(data[category_num][0]) < min_bucket:
                raise BootstrapError("insufficient data in a category")
            idxs = np.random.randint(0, len(data[category_num][0]),
                n_draw)
            fakedata_by_group[0].append(data[category_num][0][idxs])
            
            # Group 1
            n_draw = np.sum(fakedata_category_label_group0 == category_num)
            if len(data[category_num][1]) < min_bucket:
                raise BootstrapError("insufficient data in a category")
            idxs = np.random.randint(0, len(data[category_num][1]),
                n_draw)
            fakedata_by_group[1].append(data[category_num][1][idxs])
        
        # Concatenate across categories
        fakedata_by_group[0] = np.concatenate(fakedata_by_group[0])
        fakedata_by_group[1] = np.concatenate(fakedata_by_group[1])
        
        # Test difference in means
        #res = np.mean(fakedata_by_group[1]) - np.mean(fakedata_by_group[0])
        res = meth(fakedata_by_group[0], fakedata_by_group[1])
        res_l.append(res)
    
    return np.asarray(res_l)


def bootstrap_main_effect2(data, n_boots=1000, combo_meth='subtract_separately', meth=None):
    """Given 2xN set of data of unequal sample sizes, bootstrap main effect.

    We will generate a bunch of fake datasets by resampling from data.
    Then we combine across categories. The total number of data points
    will be the same as in the original dataset; however, the resampling
    is such that each category is equally represented.    
    
    data : list of length N, each entry a list of length 2
        Each entry in `data` is a "category".
        Each category consists of two groups.
        The question is: what is the difference between the groups, without
        contaminating by the different size of each category?
    
    n_boots : number of times to randomly draw, should be as high as you
        can stand
    
    meth : what to apply to the drawn samples from each group
        If None, use means_tester
        It can be any function that takes (group0, group1)
        Results of every call are returned

    Returns:
        np.asarray([meth(group0, group1) for group0, group1 in each boot])    
    """    
    if meth is None:
        meth = means_tester
    
    data = [[np.asarray(d) for d in dd] for dd in data]
    
    # How many to generate from each group, total
    N_group0 = np.sum([len(category[0]) for category in data])
    N_group1 = np.sum([len(category[1]) for category in data])
    N_group = np.rint(np.mean([N_group0, N_group1])).astype(np.int)
    N_categories = len(data)
    N_draws_per_group = np.ceil(N_group / N_categories).astype(np.int)
    
    # Which categories to draw from
    res_l = []
    for n_boot in range(n_boots):
        # Iterate over groups, processing each separately
        # Draw the data, separately by each category
        fakedata_by_group = [[], []]
        for category_num in range(N_categories):
            # Group 0
            if len(data[category_num][0]) == 0:
                raise BootstrapError("no data in a category")
            idxs = np.random.randint(0, len(data[category_num][0]),
                N_draws_per_group)
            fakedata_by_group[0].append(data[category_num][0][idxs])
            
            # Group 1
            if len(data[category_num][1]) == 0:
                raise BootstrapError("no data in a category")            
            idxs = np.random.randint(0, len(data[category_num][1]),
                N_draws_per_group)
            fakedata_by_group[1].append(data[category_num][1][idxs])
        

        if combo_meth == 'subtract_together':
            # Concatenate across categories
            fakedata_by_group[0] = np.concatenate(fakedata_by_group[0])
            fakedata_by_group[1] = np.concatenate(fakedata_by_group[1])
            
            # Test difference in means
            res = np.mean(fakedata_by_group[1] - fakedata_by_group[0])
        elif combo_meth == 'subtract_separately':
            res = np.mean([
                np.mean(fakedata_by_group[1][n_category]) - 
                np.mean(fakedata_by_group[0][n_category]) 
                for n_category in range(N_categories)])
            #res = meth(fakedata_by_group[0], fakedata_by_group[1])

            
        else:
            1/0
        res_l.append(res)
    
    return np.asarray(res_l)


from collections import defaultdict
def nested_defaultdict():
    """Returns a defaultdict with generator equal to this method.
    In [30]: nd = myutils.nested_defaultdict()

    In [31]: nd[3]
    Out[31]: defaultdict(<function nested_defaultdict at 0x48ca500>, {})

    In [32]: nd[3][2] = 'rat'

    In [33]: nd[3]
    Out[33]: defaultdict(<function nested_defaultdict at 0x48ca500>, {2: 'rat'})

    In [34]: nd[4]
    Out[34]: defaultdict(<function nested_defaultdict at 0x48ca500>, {})

    In [35]: nd[5][6][7] = 'cat'

    In [36]: nd
    Out[36]: defaultdict(<function nested_defaultdict at 0x48ca500>, 
        {3: defaultdict(<function nested_defaultdict at 0x48ca500>, 
        {2: 'rat'}), 4: defaultdict(<function nested_defaultdict at 0x48ca500>, 
        {}), 5: defaultdict(<function nested_defaultdict at 0x48ca500>, 
        {6: defaultdict(<function nested_defaultdict at 0x48ca500>, 
        {7: 'cat'})})})
    """
    return defaultdict(nested_defaultdict)


def plot_LBPB_by_block_from_ulabel(ulabel, folding_kwargs=None, **binning_kwargs):
    """Convenience function for plotting PSTHs by block"""
    res = LBPB_get_dfolded_by_block_from_ulabel(ulabel, folding_kwargs=folding_kwargs)
    binned = kkpandas.Binned.from_dict_of_folded(res, **binning_kwargs)    
    #return kkpandas.plotting.plot_binned(binned)
    return kkpandas.chris.plot_all_stimuli_by_block(binned)


def crucifix_plot(x, y, xerr, yerr, p=None, ax=None, factor=None,
    below_color='b', above_color='r', nonsig_color='gray', maxval=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    x, y, xerr, yerr = np.asarray(x), np.asarray(y), np.asarray(xerr), np.asarray(yerr)
    if p is not None:
        p = np.asarray(p)
    if factor is not None:
        x, y, xerr, yerr = factor*x, factor*y, factor*xerr, factor*yerr
    
    max_l = []
    for n, (xval, yval, xerrval, yerrval) in enumerate(zip(x, y, xerr, yerr)):
        if p is not None:
            pval = p[n]
        else:
            pval = 1.0
        
        # What color
        if pval < .05:
            if yval < xval:
                color = below_color
                pointspec = '.'
                linespec = '-'
            else:
                color = above_color
                pointspec = '.'
                linespec = '-'
            alpha = 1
        else:
            color = nonsig_color
            pointspec = '.'
            linespec = '-'
            alpha = .5
        
        # Now actually plot
        ax.plot([xval], [yval], pointspec, color=color, alpha=alpha)
        
        # plot error bars
        if hasattr(xerrval, '__len__'):
            ax.plot(xval + np.asarray(xerrval), [yval, yval], linespec, 
                alpha=alpha, color=color, 
                markerfacecolor=color, markeredgecolor=color)
            ax.plot([xval, xval], yval + np.asarray(yerrval), linespec, 
                alpha=alpha, color=color, 
                markerfacecolor=color, markeredgecolor=color)
            max_l += list(xval + np.asarray(xerrval))
            max_l += list(yval + np.asarray(yerrval))
        else:
            ax.plot([xval-xerrval, xval+xerrval], [yval, yval], linespec, 
                alpha=alpha, color=color, 
                markerfacecolor=color, markeredgecolor=color)
            ax.plot([xval, xval], [yval-yerrval, yval+yerrval], linespec, 
                alpha=alpha, color=color, 
                markerfacecolor=color, markeredgecolor=color)
            max_l += [xval+xerrval, yval+yerrval]

    # Plot the unity line
    if maxval is None:
        maxval = np.max(max_l)
    ax.plot([0, maxval], [0, maxval], 'k:')
    ax.set_xlim([0, maxval])
    ax.set_ylim([0, maxval])
    
    ax.axis('scaled')
    
    
    return ax


def print_xml(et, filename, prettify_text_nodes=True, **kwargs):
    """Print an ElementTree to a file.
    
    If prettify_text_nodes, fixes some indentation issues with text nodes.
    """
    from lxml import etree
    str_to_write = etree.tostring(et, pretty_print=True)
    if prettify_text_nodes:
        str_to_write = prettify_text_nodes_in_pretty_xml(str_to_write, **kwargs)
    
    fi = file(filename, 'w')
    fi.write(str_to_write)    
    fi.close()

def prettify_text_nodes_in_pretty_xml(string_of_xml, n_spaces=2, 
    special_case_end_element=True):
    """Fix the spacing from etree pretty print.
    
    Problems to fix:
    1)  Text nodes are not indented at all.
    2)  The closing tag after a text node is not indented.
    
    string_of_xml : output of etree.tostring(et, pretty_print=True)
    n_spaces : Amount to reindent a text node, vs its previous open element
    special_case_end_element : if True, also fix the closing tag
    
    Returns: prettified
        A new string with indentation fixed.
    """    
    # Input and output lists
    in_list = string_of_xml.split('\n')
    out_list = []

    # Initialize some persistent loop variables
    indent_level, just_did_text_line = 0, False

    # Iterate over lines
    for nline, orig_line in enumerate(in_list):
        # Strip the line
        stripped_line = orig_line.lstrip()
        
        # Deal with elements and text nodes separately
        if len(stripped_line) > 0 and stripped_line[0] == '<':
            # Not a text node, determine it's indent level
            new_indent_level = len(orig_line) - len(stripped_line)
            
            # Deal with a special case    
            if just_did_text_line and new_indent_level == 0:
                # This is the end tag after a text node, inappropriately indented
                # Fix it
                out_list.append(' ' * indent_level + stripped_line)
            else:
                # Just any old element line, leave it
                out_list.append(orig_line)
            
            # Update the loop variables
            indent_level, just_did_text_line = new_indent_level, False
        else:
            # This is a text node. Indent to current level + 2
            out_list.append(' ' * (indent_level + n_spaces) + stripped_line)
            just_did_text_line = True
    prettified = '\n'.join(out_list)
    return prettified



def insert_notes(el, notes):
    el.append(Element("notes"))
    el[-1].text = notes
inn = insert_notes

def add_child(el, child_name, notes=None, insert_pos=None, **kwargs):
    child = Element(child_name, **kwargs)
    if notes is not None:
        inn(child, notes)
    if insert_pos is None:
        el.append(child)
    else:
        el.insert(insert_pos, child)
    return child

def load_xml_file(filename, unprettyprint=True):
    if unprettyprint:
        parser = etree.XMLParser(remove_blank_text=True)
    else:
        parser = None
    return etree.parse(filename, parser=parser).getroot()

def allclose_2d(a):
    """Returns True if entries in `a` are close to equal.
    
    a : iterable, convertable to array
    
    If the entries are arrays of different size, returns False.
    If len(a) == 1, returns True.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("input to allclose_2d cannot be 0d")
    return np.all([np.allclose(a[0], aa) for aa in a[1:]])

def apply_and_filter_by_regex(pattern, list_of_strings, sort=True, 
    squeeze=True, warn_on_regex_miss=False):
    """Apply regex pattern to each string, return hits from each match.
    
    The pattern is applied to each string in the list using re.match.
    If there is no match, the string is skipped.
    If there is a match, the groups will be appended to a list of hits.
    Optionally the list of hits is sorted. Then it is returned.
    
    pattern - regex pattern, passed to re.match
    
    list_of_strings - duh
    
    sort - if True, sort the list of hits
    
    squeeze - if True and there is only one group, simply save this group,
        rather than a 1-tuple.
    """
    import re
    res = []
    for s in list_of_strings:
        m = re.match(pattern, s)
        if m is None:
            if warn_on_regex_miss:
                print "warning: regex miss %s" % s
            continue
        else:
            if len(m.groups()) == 1 and squeeze:
                # Only 1 group
                res.append(m.groups()[0])
            else:
                res.append(m.groups())
    
    if sort:
        return sorted(res)
    else:
        return res

def dict_sum(dicts, warn_on_overlap=True):
    """Combines list of dict into one, warning if some of the keys overlap"""
    # Check for overlap
    if warn_on_overlap:
        overlaps = set.intersection(*[set(dd.keys()) for dd in dicts])
        if len(overlaps) > 0:
            print "warning: overlapping keys in dict_sum: %r" % overlaps

    # Combine
    d = {}
    for dd in dicts:
        d.update(dd)
    return d