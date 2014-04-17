import numpy as np
from SpikeTrainContainers import PSTH
import unittest


"""Test cases for PSTHs.

Developed for rigid and elastic PSTHs. The behavior is not entirely
consistent between the two. The biggest problem is how to deal with the
end of the range.

For rigid PSTHs, the range is specified as a closed interval, 
eg [-2,+2] in the below examples. In this case:
* Spikes at time +3 are discarded, which is good
* Spikes at time +1 and +2 are binned together, since the last bin is closed
  in np.histogram, which doesn't feel right
  
In contrast, for elastic PSTHs it's most natural to specify the range as
a half-open interval eg [-2, 3)
* Spikes at time +3 are included in that last bin, which doesn't feel right
* Spikes at time +1 and +2 are binned separately, which is good.
"""

class testRigid(unittest.TestCase):
    def testHistValues(self):
        # Simple example
        # Two trials, extending from -2 to +2 (centered, inclusive)
        # Two spikes, at +1 and +2 (centered)
        
        # Create a PSTH 
        psth = PSTH(F_SAMP=1000., 
            adjusted_spike_times=[1, 2],
            n_trials=2, range=(-2, 2),
            nbins=4)

        t, counts = psth.hist_values(style='rigid')
        
        # Bin edges stretch across range, inclusively
        self.assertTrue(np.all(psth._bin_edges == np.arange(-2, 3)))
        self.assertTrue(np.all(counts == np.array([0, 0, 0, 1.0])))

    def testWithSpuriousSpike(self):
        # Same as testHistValues, but with an extra spike that is outside
        # of any trial, so the results should be the same.
        
        # Create a PSTH 
        psth = PSTH(F_SAMP=1000., 
            adjusted_spike_times=[1, 2, 3],
            n_trials=2, range=(-2, 2),
            nbins=4)

        t, counts = psth.hist_values(style='rigid')
        
        # Bin edges stretch across range, inclusively
        self.assertTrue(np.all(psth._bin_edges == np.arange(-2, 3)))
        self.assertTrue(np.all(counts == np.array([0, 0, 0, 1.0])))



class testElastic(unittest.TestCase):
    def testHistValues(self):
        # Simple example
        # Two triggers: n=10, 20
        # Trial 1 extends from -2 to +1 (centered at first trigger)
        # Trial 2 extends from -1 to +2 (centered at second trigger)
        # Spikes occurred at centered times +1 and +2
        
        # Initialize PSTH with this info
        psth = PSTH(F_SAMP=1000.,
            adjusted_spike_times=[1, 2],
            t_starts=[8, 19], t_stops=[12, 23], t_centers=[10, 20],
            nbins=5)
        
        # Check that range was set based on trial extents
        self.assertTrue(np.all(psth.range == [-2, 3]))
        
        # Calculate histogram values
        t, counts = psth.hist_values(style='elastic')
        
        # Bin edges were auto set to include all trials
        self.assertTrue(np.all(psth._bin_edges == np.arange(-2, 4)))
        
        # The first and last bins contain only one trial
        self.assertTrue(np.all(psth._trials == np.array([1, 2, 2, 2, 1])))

        # Time +1 includes two trials, so that count is divided by 2.
        # Time +2 includes only one trial, so that count is divided by 1.
        self.assertTrue(np.all(counts == np.array([0, 0, 0, 0.5, 1.0])))

    def testWithSpuriousSpike(self):
        # Same as testHistValues, but with an extra spike that is outside
        # of any trial, so the results should be the same.
        # Note that a spike at time 3 would be included in the last bin,
        # which is not strictly correct because the t_stop is not inclusive.
        psth = PSTH(F_SAMP=1000.,
            adjusted_spike_times=[1, 2, 4],
            t_starts=[8, 19], t_stops=[12, 23], t_centers=[10, 20],
            nbins=5)        
        self.assertTrue(np.all(psth.range == [-2, 3]))
        t, counts = psth.hist_values(style='elastic')
        self.assertTrue(np.all(psth._bin_edges == np.arange(-2, 4)))
        self.assertTrue(np.all(psth._trials == np.array([1, 2, 2, 2, 1])))
        self.assertTrue(np.all(counts == np.array([0, 0, 0, 0.5, 1.0])))

    def testWiderBins(self):
        # Same as testHistValues, but now the bins are wider so that
        # a trial might cover only half of it.
        psth = PSTH(F_SAMP=1000.,
            adjusted_spike_times=[1, 2, 4],
            t_starts=[8, 19], t_stops=[12, 23], t_centers=[10, 20],
            nbins=2)        
        self.assertTrue(np.all(psth.range == [-2, 3]))
        t, counts = psth.hist_values(style='elastic')

        # Wider bins covering the same range
        self.assertTrue(np.all(psth._bin_edges == np.array([-2, .5, 3])))

        # Each trial is only partly contained by one of the bins
        self.assertTrue(np.all(psth._trials == np.array([5, 3])))            
        self.assertTrue(np.all(counts == np.array([0., 5/3.])))
    


class testBinWidth(unittest.TestCase):
    def testElastic(self):
        psth = PSTH(F_SAMP=1000.,
            adjusted_spike_times=[1, 2, 4],
            t_starts=[8, 19], t_stops=[12, 23], t_centers=[10, 20],
            binwidth=.001)        
        
        self.assertTrue(np.all(psth.range == [-2, 3]))
        t, counts = psth.hist_values(style='elastic')
        
        self.assertTrue(np.all(psth._bin_edges == np.arange(-2, 4)))
        self.assertTrue(np.all(psth._trials == np.array([1, 2, 2, 2, 1])))
        self.assertTrue(np.all(counts == np.array([0, 0, 0, 0.5, 1.0])))    

    def testElasticWiderBins(self):
        # Now bins are specified to be 2ms wide.
        # Trial 1: -2 to 1 (closed)
        # Trial 2: -1 to 2 (closed)
        # Spikes: +1, +2
        # Thus the bins are [-2,0), [0, 2), [2, 4]
        # The middle bin includes one spike and two trials, so
        # 0.5 spike/trial/bin.
        # The last bin includes one spike and only half a trial, because 
        # one trial extends through half of it, so
        # 2 spike/trial/bin
        
        psth = PSTH(F_SAMP=1000.,
            adjusted_spike_times=[1, 2],
            t_starts=[8, 19], t_stops=[12, 23], t_centers=[10, 20],
            binwidth=.002)        
        
        self.assertTrue(np.all(psth.range == [-2, 4]))
        t, counts = psth.hist_values(style='elastic')
        self.assertTrue(np.all(psth.range == [-2, 4]))
        self.assertTrue(np.all(psth._bin_edges == np.array([-2, 0, 2, 4])))
        self.assertTrue(np.all(psth._trials == np.array([3, 4, 1])))            
        self.assertTrue(np.all(counts == np.array([0, 0.5, 2.])))



if __name__ == '__main__':
    unittest.main()