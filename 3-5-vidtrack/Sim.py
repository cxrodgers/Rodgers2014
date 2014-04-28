import numpy as np
import my
import pandas

class Sim:
    def __init__(self, target_r=.9, Nt=100, b0=.3, b1=.9, noise_level=.1, 
        aov_typ=2, poissonify=False):
        """Initialize a new simulation and set parameters"""
        self.target_r = target_r
        self.Nt = int(my.rint(Nt)) # 0d array sometimes
        self.b0 = b0
        self.b1 = b1
        self.noise_level = noise_level
        self.aov_typ = aov_typ
        self.poissonify = poissonify
    
    def generate_inputs(self):
        """Set x0, x1, and noise"""
        # Draw the factors with specified correlation
        data = np.random.multivariate_normal([0, 0], 
            [[1, self.target_r], [self.target_r, 1]], self.Nt)
        self.x0 = data[:, 0]
        self.x1 = data[:, 1]
        
        self.noise = self.noise_level * np.random.standard_normal((self.Nt,))
    
    def generate_response(self):
        """Calculate response (includes noise)"""
        # The response is a linear summation of both factors
        self.response = \
            self.b0 * self.x0 + \
            self.b1 * self.x1 + \
            self.noise
        
        # Testing whether saturation matters
        # self.response[self.response < 0] = 0
        
        if self.poissonify:
            # Poissonify it
            # If you do this, makes sure PB==1 (a worse case scenario for xcorr)
            # and remember to fit to the sqrt of the response
            self.response = np.array([
                np.random.poisson(2**sr, 1)[0] for sr in self.response])

    def test(self):
        """Attempt to reverse-engineer coefficients"""
        # Form the dataframe
        self.df = pandas.DataFrame.from_dict({
            'x0': self.x0, 'x1': self.x1, 'response': self.response})

        # Run anova
        if self.poissonify:
            self.aovres = my.stats.anova(self.df, 'np.sqrt(response) ~ x0 + x1', 
                typ=self.aov_typ)
        else:
            self.aovres = my.stats.anova(self.df, 'response ~ x0 + x1', 
                typ=self.aov_typ)


class SimDiscrete(Sim):
    def __init__(self, p_flip=.9, Nt=100, b0=.3, b1=.9, noise_level=.1, 
        aov_typ=2, poissonify=False):
        """Initialize a new simulation and set parameters"""
        self.p_flip = p_flip
        self.Nt = int(my.rint(Nt)) # 0d array sometimes
        self.b0 = b0
        self.b1 = b1
        self.noise_level = noise_level
        self.aov_typ = aov_typ 
        self.poissonify = poissonify
    
    def generate_inputs(self):
        """Set x0 (normal), x1 (binary), and noise"""
        # Random normals
        self.x0 = np.random.standard_normal((self.Nt,))

        # Discretify x1, starting with max possible corr
        self.x1 = (self.x0 > 0).astype(np.int)

        # Flip a random number of bits, to decrease correlation
        nflip = my.rint(self.p_flip * self.Nt)
        self.x1[:nflip] = np.random.binomial(1, .5, nflip)
        
        # Scale
        self.x1 = self.x1 - self.x1.mean()
        self.x1 = self.x1 / self.x1.std()
        
        # Generate noise
        self.noise = self.noise_level * np.random.standard_normal((self.Nt,))        

class SimData(Sim):
    """This uses x0 and x1 from real behavioral data"""
    def __init__(self, real_data_l, n_session=0, b0=.3, b1=.9, noise_level=.1, 
        aov_typ=2, poissonify=False):
        """Initialize a new simulation and set parameters"""
        self.n_session = int(n_session)
        self.real_data_l = real_data_l
        self.Nt = len(self.real_data_l[self.n_session])
        self.b0 = b0
        self.b1 = b1
        self.noise_level = noise_level
        self.aov_typ = aov_typ    
        self.poissonify = poissonify
    
    def generate_inputs(self):
        """Set x0 (angl) and x1 (block) from real data, as well as noise"""
        # x0 is the head angle
        real_data = self.real_data_l[self.n_session]
        self.x0 = real_data['x0'].values

        # x1 is the block ID
        self.x1 = real_data['x1'].values
        
        # Generate noise
        self.noise = self.noise_level * np.random.standard_normal((self.Nt,))       


class SimData2(Sim):
    """This uses x0, x1, and mean output from real behavioral data"""
    def __init___(self, n_neuron=0, b0=.3, b1=.9, aov_typ=2):
        pass
    
    def generate_inputs(self):
        # Set all these values s.t. var(response) == var(real response)
        pass
