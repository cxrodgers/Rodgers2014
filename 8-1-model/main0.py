# For the paper, the following params were used:
# NT_TRAIN = 1000
# NT_TEST = 100
# N_REPS = 1000
# But that will take a very long time to run and generate a lot of data
# The results of that long simulation are provided in the data directory,
# so you can skip rerunning the simulation if you want.
#
# Here, the values are smaller, so that the file size and running time
# is more manageable. The plots will be a bit noisier than those in the
# paper.
#
# This is still the longest-running script in the entire repository,
# even with the lower values.
import numpy as np
import my, my.plot
import pandas, itertools
import model
    
"""Parameters

To iterate over:
gain
noise_level
N

Simulation params:
NT_TRAIN -- How many noisy stimuli to train the network on.
    Just needs to be enough that it asymptotes to optimal performance.
NT_TEST -- How many noisy stimuli to test the network on.
    The score is based on the fraction of correct responses, so this should
    be high enough that the score is not unduly quantized.
N_REPS -- How many times to generate a new training and new testing set.
    We need enough estimate the variability over train/test combinations.
    For instance, especially for small networks, we might get unlucky with 
    the RFs that we get.
"""
import datetime
import os.path
output_dir = '../data/miscellaneous/model'

spooler = None

NT_TRAIN = 100
NT_TEST = 50
N_REPS = 15

# List of params to check
N_l = [10, 20, 30, 40, 60, 80, 120, 160, 320, 640][::-1]
noise_level_l = [.5, 1, 2, 4, 8, 16, 32, 64, 128]

# Logspace and linspace to check various plots
# When plotting in gain-vs-noise, we need exponentially spaced
# huge values ~1000x the signal or ~10x the noise because that effect 
# looks good on an xlog pot
# When plotting in gain-vs-signal, we need good linear coverage ~20x the signal
log_max_noise = np.log10(np.max(noise_level_l)) # 10x this
gains_l = sorted(np.concatenate([
    -10**(np.linspace(-3, 3, 40))[::-1], # -1000 to -.001
    10**(np.linspace(-3, 3, 40)), # .001 to 1000
    np.linspace(-20, 20, 80), # -20 to 20
    ]))


# We'll iterate lastly over gain, so set others to rows
params_df = pandas.DataFrame(
    list(itertools.product(range(len(N_l)), range(len(noise_level_l)))),
    columns=['nN', 'n_noise_level'])

# Pre-generate all test datasets
test_stim_mats = [
    model.generate_training_set(NT_TEST)[1] for nset in range(len(gains_l))]
choice1_l = map(model.stim2choice1, test_stim_mats)
choice2_l = map(model.stim2choice2, test_stim_mats)

# Run the simulations
res_l = []
for nsim, (nN, n_noise_level) in params_df.iterrows():
    # Get the params from the lists
    N = N_l[nN]
    noise_level = noise_level_l[n_noise_level]
    print nsim, N, noise_level
    
    # Iterate over training sets
    for nrep in range(N_REPS):
        # Generate training stimuli and correct responses
        train_stim_ids, train_stim_mat = model.generate_training_set(NT_TRAIN)
        train_resp1 = model.stim2resp1(train_stim_mat)
        train_resp2 = model.stim2resp2(train_stim_mat)

        # Train model
        m = model.Model(N1=N, N2=N, noise_level=noise_level, spooler=spooler)
        m.train(train_stim_mat, train_resp1, train_resp2)

        # Test various gains
        for ngain, gain in enumerate(gains_l):
            # Get the test set
            test_stim_mat = test_stim_mats[ngain]
            choice1 = choice1_l[ngain]
            choice2 = choice2_l[ngain]

            # Force gain positive
            if gain >= 0:
                # Test
                m.test(test_stim_mat, gain=gain, apply_to=1)
            else:
                m.test(test_stim_mat, gain=-gain, apply_to=2)

            # Score
            score1 = np.sum(m.test_choice == choice1)
            score2 = np.sum(m.test_choice == choice2)

            # Also score and count congruent NOGO as correct
            # That is, count choices 1 and 3 as equivalent
            mtc2 = m.test_choice.copy()
            mtc2[mtc2 == 1] = 3
            c1 = choice1.copy()
            c1[c1 == 1] = 3
            c2 = choice2.copy()
            c2[c2 == 1] = 3
            score1b = np.sum(mtc2 == c1)
            score2b = np.sum(mtc2 == c2)

            res = {'ngain': ngain, 'nsim': nsim, 'nrep': nrep,
                'score1': score1, 'score2': score2,
                'score1b': score1b, 'score2b': score2b}
            res_l.append(res)

# DataFrame it
resdf = pandas.DataFrame.from_records(res_l)

# Join on simulation parameters
resdf = resdf.join(params_df, on='nsim')

prefix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


resdf.save(os.path.join(output_dir, '%s_resdf' % prefix))
my.misc.pickle_dump({'N_l': N_l, 'noise_level_l': noise_level_l,
    'gains_l': gains_l, 'NT_TEST': NT_TEST}, 
    os.path.join(output_dir, '%s_params.pickle' % prefix))


