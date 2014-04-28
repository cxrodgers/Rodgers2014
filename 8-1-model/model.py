import numpy as np, scipy.optimize

## THESE DEFINE THE RULES
# Matrix to convert stim_ids (0, 1, 2, or 3) into a 4-tuple representing
# presence of (left, right, low, high)
id2mat = np.array([
    [1, -1, -1, 1], # left high
    [-1, 1, -1, 1], # right high
    [1, -1, 1, -1], # left low
    [-1, 1, 1, -1]])# right low

def stim2resp1(stim_mat):
    """Correct output activations for task 1"""
    return stim_mat[:, [0, 1]]

def stim2resp2(stim_mat):
    """Correct output activations for task 2"""
    return stim_mat[:, [2, 3]]

def stim2choice1(stim_mat):
    """Returns 0 for go-left, 1 for nogo"""
    return (stim_mat[:, 1] == 1).astype(np.int)

def stim2choice2(stim_mat):
    """Returns 2 for go-right, 3 for nogo"""
    return 2 + (stim_mat[:, 3] == 1).astype(np.int)
    
    
## GENERATING DATA
def generate_training_set(NT_per_id=30):
    """Generate training stimuli and responses.
    
    Sets the following attributes of self:
        stim_ids : array, length approximately self.NT_train
            Each entry is 0, 1, 2, or 3; representing which stimulus pair
        stim_mat : array, (NT_train, 4)
            Each entry is a 4-tuple representing which stimulus pair
        resp_mat1, resp_mat2 : array, (NT_train, 2)
            The correct responses in each block, ie presence of left and
            right in block 1. Just slices into stim_mat
            These are demeaned
        Y_train : array, (N, NT_train)
            Response of each neuron on each trial
            Gaussian noise of stdev=noise_level is added to each response
    """
    # Training stimuli
    #stim_ids = np.floor(np.random.random(self.NT_train) * 4).astype(np.int)
    stim_ids = np.concatenate([[ns] * NT_per_id for ns in range(4)])

    # Convert stim_ids into stim_mat
    # each row has 4 columns: left, right, low, high
    # for the presence of each of those sounds in that sound pair
    stim_mat = np.array([id2mat[stim_id] for stim_id in stim_ids])
    
    return stim_ids, stim_mat



## MODEL
class Model:
    """Object encapsulating separate circuits model"""
    def __init__(self, N1=15, N2=15, noise_level=.3, spooler=False):
        """Initialize new Model object.
        
        N1  : int, number of neurons in circuit 1 (eg localization)
        N2  : int, number of neurons in circuit 2 (eg pitch disc)
        
        It is assumed that there are 4 stimulus pairs, eg
        LEFT+HIGH, RIGHT+HIGH, LEFT+LOW, RIGHT+LOW. These are indicated with
        a binary 4-tuple: (left, right, low, high).
        
        Each neuron will be initialized with a random RF. This RF is also a
        4-tuple and the response is a linear combination with the stimulus.
        
        Finally, a training set of stimuli and responses is generated.
        """
        # Save params
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.N = self.N1 + self.N2
        self.noise_level = noise_level
        
        if spooler is True:
            import randspool
            self.spooler = randspool.randspool(order=8)
        elif spooler is False:
            self.spooler = None
        else:
            self.spooler = spooler
        
        # Construct random tuning curves
        # This is the tuning of each neuron (col) for each stimulus (row)
        self.RF = np.random.standard_normal((4, self.N))

    def compute_activations(self, stim_mat):
        activations = np.dot(stim_mat, self.RF)

        # Add noise to responses
        # Most computationally intensive step!
        if self.spooler is not None:
            activations += self.noise_level * self.spooler.get(activations.shape)  
        else:
            activations += self.noise_level * np.random.standard_normal(
                activations.shape)  

        return activations

    def compute_output(self, activations):
        # Decode from WW1 and WW2
        b_ww1 = np.dot(activations[:, :self.N1], self.WW1)
        b_ww2 = np.dot(activations[:, self.N1:], self.WW2)

        # Combine them
        output_units = np.hstack([b_ww1, b_ww2])
        
        # Choose the behavioral response by WTA
        choice = np.argmax(output_units, axis=1)
        
        return output_units, choice

    def train(self, stim_mat, resp_mat1, resp_mat2):
        """Trains the mapping from neural activations to behavioral response
        
        Finds the best linear mapping from the noisy activations in Y_train
        to the correct behavioral response. N1 neurons are trained on response
        for block 1; N2 neurons are trained on response for block 2.
        
        Sets the following:
            WW1 : array 
                Mapping from (N1, NT_train) to resp_mat1
            WW2 : array
                Mapping from (N2, NT_train) to resp_mat2
        """
        self.train_stim_mat = stim_mat.copy()
        self.train_resp1 = resp_mat1.copy()
        self.train_resp2 = resp_mat2.copy()
        
        # Training set activations -- the activation of each neuron to each input
        # NxNT_train matrix, activation on each trial
        # These will be noisy
        self.train_activations = self.compute_activations(self.train_stim_mat)
        
        # WW1 is the mapping from 1/2 of training set to resp_mat1
        lstsq_res_WW1 = np.transpose([
            scipy.optimize.nnls(    
                self.train_activations[:, :self.N1], 
                self.train_resp1[:, ncol])[0]
            for ncol in range(self.train_resp1.shape[1])]
            )
        self.WW1 = lstsq_res_WW1


        # WW2 is the mapping from other 1/2 of trainign set to resp_mat2
        lstsq_res_WW2 = np.transpose([
            scipy.optimize.nnls(    
                self.train_activations[:, self.N1:], 
                self.train_resp2[:, ncol])[0]
            for ncol in range(self.train_resp2.shape[1])]
            )
        self.WW2 = lstsq_res_WW2


    def test(self, stim_mat, gain=3, apply_to=1, additive=True):
        """Test the decoding of the noisy responses
        
        Currently we reuse the training set, but should generate a new test
        set.
        
        First, the responses of population 1 are multiplied by `gain`,
        symbolizing activation from PFC that is specific to subpopulation
        but not to stimulus.
        
        Second, the fixed output mappings WW1 and WW2 are used to map
        subpops 1 and 2 onto behavioral responses.
        
        Finally, the final behavioral response is chosen as the winner-take-all
        maximum over output units.
        """
        self.test_stim_mat = stim_mat.copy()
        
        # Test activations
        # NxNT_train matrix, activation on each trial
        # These will be noisy
        self.test_activations = self.compute_activations(self.test_stim_mat)
        
        # Add in the attn signal
        # Additional activations for one subpop
        # This implements an decrease to 33% in pop1
        self.test_activations_wattn = self.test_activations.copy()
        if apply_to == 1:
            if additive:
                self.test_activations_wattn[:, :self.N1] = \
                    self.test_activations_wattn[:, :self.N1] + gain
            else:
                self.test_activations_wattn[:, :self.N1] = \
                    self.test_activations_wattn[:, :self.N1] * gain
        elif apply_to == 2:
            if additive:
                self.test_activations_wattn[:, self.N1:] = \
                    self.test_activations_wattn[:, self.N1:] + gain
            else:
                self.test_activations_wattn[:, self.N1:] = \
                    self.test_activations_wattn[:, self.N1:] * gain
        else:
            raise ValueError('must be 1 or 2')

        # Compute output
        self.test_output, self.test_choice = self.compute_output(
            self.test_activations_wattn)
