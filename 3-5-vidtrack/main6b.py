# Same as 6a but using the simulation regressors instead of real data
import numpy as np
import my
import pandas
import itertools
import matplotlib.pyplot as plt
import Sim
import kkpandas.kkrs


my.plot.font_embed()
my.plot.publication_defaults()

POISSONIFY = False

# Get the actual angles and blocks from analyzed sessions
full_data = pandas.load('full_data')

# Sim params
nreps = 150
params_l = {
    'noise_level': [.5],
    'b0': np.array([0, .2, .4, .6, .8, 1.0]) ** .5,
    'p_flip': [.1],
    'Nt': [565], # taken from real_data_l[1], the intermediate trial #
    }

# These are the indexes of the parameters
param_idxs_df = pandas.DataFrame(
    data=list(itertools.product(*[
        range(len(l)) for l in params_l.values()])),
    columns=params_l.keys())

# These are the actual values of the parameters
param_vals_df = param_idxs_df.apply(lambda ser: 
        [params_l[ser.name][val] for val in ser.values]) 


res_l = []
for nsim, sim_params in param_vals_df.iterrows():
    for nrep in range(nreps):
        # Set b1 
        b1 = np.sqrt(1 - sim_params['b0'] ** 2)
        
        # Run the simulation
        s = Sim.SimDiscrete(aov_typ=2, b1=b1, poissonify=POISSONIFY, 
            **sim_params)
        s.generate_inputs()
        s.generate_response()
        s.test()
        
        # Get the results
        res = s.aovres['ess'].to_dict()
        res['nsim'] = nsim
        res['nrep'] = nrep

        # Measure the achieved correlation
        res['achieved_r'] = np.corrcoef(s.x0, s.x1)[0, 1]

        res_l.append(res)
res_df = pandas.DataFrame.from_records(res_l)

# Mean over reps
meaned = res_df.set_index(['nsim', 'nrep']).mean(axis=0, level=0)
stddd = res_df.set_index(['nsim', 'nrep']).std(axis=0, level=0)

# Join
joined_df = param_idxs_df.join(meaned)
joined_df = joined_df.join(stddd, rsuffix='_std')
joined_df.index.name = 'nsim'

# This calculates the empirical relationship between n_session and achieved_r
p_flip2achieved_r = joined_df.reset_index().set_index(['p_flip', 'nsim']).mean(
    axis=0, level=0)['achieved_r']



# Mean and std of ESS, sorted by b0
ess_x0_m = joined_df.pivot_table(rows='b0', cols='p_flip', values='ess_x0').sort()
ess_x0_std = joined_df.pivot_table(rows='b0', cols='p_flip', values='ess_x0_std').sort()

for np_flip in ess_x0_m.columns:
    f, ax = plt.subplots(1, 1, figsize=(3,3))

    # Get means and stds for this session
    sess_ess_m = ess_x0_m[np_flip]
    sess_ess_std = ess_x0_std[np_flip]
    xvals = (params_l['b0'][sess_ess_m.index]) ** 2
    
    # Plot error bars
    ax.plot([0, 1], [0, 1], 'k-')
    ax.errorbar(x=xvals, y=sess_ess_m, yerr=sess_ess_std, color='r', 
        capsize=0, lw=0, elinewidth=1)
    ax.plot(xvals, sess_ess_m, 'ro', mec='r', mfc='none', lw=0)

    # Pretty
    ax.set_title('simulated regressors')
    ax.axis('scaled')
    ax.set_xlim((-.1, 1.1))
    ax.set_ylim((-.1, 1.1))
    ax.set_xticks((0, .25, .5, .75, 1))
    ax.set_yticks((0, .25, .5, .75, 1))

    #~ ax.set_title('session %d r=%0.3f' % (
        #~ n_session, n_session2achieved_r[n_session]))
    ax.set_xlabel('true FEV of head angle')
    ax.set_ylabel('estimated FEV of head angle')

    f.savefig('simulation_simbinom_pflip%0.3f_r%0.2f_%s.svg' % (
        params_l['p_flip'][np_flip],
        p_flip2achieved_r[np_flip],
        'poissonified' if POISSONIFY else 'continous'))

plt.show()

