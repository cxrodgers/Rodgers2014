"""Methods to analyze/visualize point processes"""

import numpy as np
import matplotlib.pyplot as plt

def intscale(x, scale):
    return np.rint(x / scale).astype(np.int)

def smooth_at_scale(timestamps, scale, oversample_ratio=4, t_min=None, 
    t_max=None):
    """Wrapper around smoothing function for specifying scale in sec
    
    timestamps : values in seconds
    scale : smoothing parameter in seconds (half-width of Gaussian)
    oversample_ratio : how densely to discretize the smoothed function,
        as a multiple of the smoothing parameter
    
    Returns: n_ssig, x_ssig
        n_ssig : discretized time of the smoothed signal (fs=OSR/scale))
        x_ssig : smoothed values at times in n_ssig
    """
    if t_min is None:
        t_min = timestamps.min()
    if t_max is None:
        t_max = timestamps.max()
    
    # Scale the signal in time
    factor = float(scale) / oversample_ratio
    sig = intscale(timestamps, factor)
    n_min = intscale(t_min, factor)
    n_max = intscale(t_max, factor)
    
    # in this scaling, the smoothing parameter is now just the oversample_ratio
    n_ssig, x_ssig = smooth_event_train(sig, 
        filter_std=oversample_ratio, n_min=n_min, n_max=n_max)
    
    t_ssig = n_ssig * factor
    
    return t_ssig, n_ssig, x_ssig
    

def smooth_event_train(timestamps, filter_std=10, 
    filter_truncation_width=None, n_min=None, n_max=None): 
    """Returns a filtered time series representation.
    
    For a specified list of event times, returns a filtered time series
    representation as a tuple: (n_op, x_op).  
    
    The time points are given in n_op (in samples) and the values
    are given in x_op. A gaussian will be added to x_op centered at each
    timestamp.
    
    timestamps: array of time values in samples when events occurred
    filter_std: Standard deviation (width) of the Gaussian, in samples.
        Can be a non-integer number of samples.
    filter_truncation_width: Where to truncate the Gaussian, in samples.
        This must be an integer number of samples. Default is the closest
        integer to 5 * `filter_std`.
    n_min : Extend or truncate returned values to start at this.
        Default: first timestamp
    n_max : Extend or truncate returned values to stop at this.
        Inclusive, unlike Python indexing.
        Default: last timestamp
    
    Returns: n_op, x_op
    n_op: the sample number of each point. 
    x_op: the value of the smoothed function at each sample.        
    """
    # Finalize the default values of parameters    
    if filter_truncation_width is None: 
        filter_truncation_width = np.rint(5 * filter_std).astype(np.int)
    
    # Convert timestamps to integers for use in indexing
    timestamps = np.asarray(timestamps, dtype=np.int)
    
    # Deal with edge case
    if len(timestamps) == 0:
        if n_min is None or n_max is None:
            raise ValueError("empty timestamps and no specified time frame")
        n_op = np.arange(n_min, n_max + 1, dtype=np.int)
        x_op = np.zeros_like(n_op)
        return n_op, x_op
    
    # Determine the range of the output
    # n_min, n_max are the requested bounds for result
    # start_sample, stop_sample are the bounds used for internal calculation
    start_sample = np.min(timestamps) - filter_truncation_width
    stop_sample = np.max(timestamps) + filter_truncation_width
    
    # Make it bigger if necessary, for instance because more was requested
    # than necessary
    if n_min is not None and n_min < start_sample:
        # More requested on left than necessary
        start_sample = n_min
    if n_max is not None and n_max > stop_sample:
        # More requested on right than necessary
        stop_sample = n_max

    # generate normalized gaussian on n_gauss and x_gauss
    # here is why filter_truncation_width must be an integer, but filter_std
    # doesn't.
    n_gauss = np.arange(-filter_truncation_width,
        filter_truncation_width + 1)
    x_gauss = np.exp( -(np.float64(n_gauss) ** 2) / (2 * filter_std**2) )
    x_gauss = x_gauss / np.sum(x_gauss)
    
    # initialize return variables 
    # We do calculation on full range from start_sample to stop_sample
    # And will truncate to n_min, n_max later
    n_op = np.arange(start_sample, stop_sample + 1)
    x_op = np.zeros(n_op.shape) # value, float64
    
    # for each timestamp, add a gaussian to x_op
    for timestamp in timestamps:
        x_op[timestamp - start_sample + n_gauss] += x_gauss
    
    # default behavior: truncate to timestamps
    if n_min is None:
        n_min = timestamps.min()
    if n_max is None:
        n_max = timestamps.max()
    
    # Now truncate if necessary
    if n_min > n_op[0] and n_min < n_op[-1]:
        truncate_start = np.where(n_op == n_min)[0]
        x_op = x_op[truncate_start:]
        n_op = n_op[truncate_start:]
    if n_max < n_op[-1] and n_max > n_op[0]:
        truncate_stop = np.where(n_op == n_max)[0]
        x_op = x_op[:truncate_stop + 1]
        n_op = n_op[:truncate_stop + 1]
    
    return (n_op, x_op)


class MultiscaleSmoother:
    def __init__(self, timestamps=None, scales=None, oversample_ratio=4,
        n_levels=4):
        # store defaults
        self.timestamps = timestamps
        self.scales = scales
        self.oversample_ratio = oversample_ratio
        self.n_levels = n_levels
        
        # where results go
        self.n_l, self.x_l, self.t_l = [], [], []
    
    def _define_scales(self):
        if self.scales is None:
            coarsest_scale = np.median(np.diff(self.timestamps)) / 2.0
            self.scales = [coarsest_scale / (2 ** n) 
                for n in range(self.n_levels)]    
        
        if np.any(self.scales == 0):
            raise ValueError("scale cannot be zero")
    
    def calculate(self):
        """Calculate and store"""
        if self.timestamps is None:
            raise ValueError("no data provided")
        
        # Define levels of processing
        self._define_scales()
        self.n_l, self.x_l, self.t_l = [], [], []
        
        # Where to start
        global_t_min = self.timestamps.min()
        global_t_max = self.timestamps.max()
        
        # Calculate at each scale
        for scale in self.scales:
            n_ssig, x_ssig = smooth_at_scale(self.timestamps, scale, 
                self.oversample_ratio)
            self.n_l.append(n_ssig)
            self.x_l.append(x_ssig)
            self.t_l.append(n_ssig * float(scale) / self.oversample_ratio)

class MultiscalePlotter:
    def __init__(self, data=None):
        """Initialize a plotter on a smoother"""
        self.data = data
        self.ax = None
    
    def plot(self, ax=None):
        # where to plot
        if not ax:
            f = plt.figure()
            ax = f.add_subplot(111)

        for nn, (n, x, t) in enumerate(
            zip(self.data.n_l, self.data.x_l, self.data.t_l)):
            ax.imshow(x.reshape(1, -1), extent=(t[0], t[-1], nn, nn+1), 
                interpolation='nearest', aspect='auto', cmap=plt.cm.hot)
        ax.set_ylim((0, nn + 1))
        plt.show()
