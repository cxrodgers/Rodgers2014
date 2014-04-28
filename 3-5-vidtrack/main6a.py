# Simulate how ANOVA untangles two correlated regressors
# This is with the regressors from real data
# Run this once with and without POISSONIFY

import numpy as np
import my, my.plot
import pandas
import itertools
import matplotlib.pyplot as plt
import Sim
import kkpandas.kkrs

my.plot.font_embed()
my.plot.publication_defaults()

# Get the actual angles and blocks from analyzed sessions
full_data = pandas.load('full_data')

# Choose the sub-indexed trials for each session
session2data = {}
for ulabel in full_data.index.levels[0]:
    # Skip if we already did this session
    session = kkpandas.kkrs.ulabel2session_name(ulabel)
    if session in session2data:
        continue
    
    # each block, sorted by angle
    fdu = full_data.ix[ulabel]
    
    # Scale results
    data = fdu[['angl']]
    data['angl'] = data['angl'] - data['angl'].mean()
    data['angl'] = data['angl'] / data['angl'].std()
    
    # Add in block
    # In some cases we obtain slightly worse results when PB is 1
    # So let's do that for a worse-case estimate
    # This is because the Poisson goes to zero in one block, and gets
    # high in the other block
    data['block'] = 0
    data['block'][fdu['block'] == 'PB'] = 1
    data['block'] = data['block'] - data['block'].mean()
    data['block'] = data['block'] / data['block'].std()
    
    # Rename
    data = data.rename(columns={'angl': 'x0', 'block': 'x1'})
    session2data[session] = data

# Listify and dump session name
real_data_l = [session2data[key] for key in sorted(session2data.keys())]


# Sim params
nreps = 150
params_l = {
    'n_session': [2], # Just do 2, the intermediate correlation session
    'noise_level': [.5],
    'b0': np.array([0, .2, .4, .6, .8, 1.0]) ** .5,
    }

# These are the indexes of the parameters
param_idxs_df = pandas.DataFrame(
    data=list(itertools.product(*[
        range(len(l)) for l in params_l.values()])),
    columns=params_l.keys())

# These are the actual values of the parameters
param_vals_df = param_idxs_df.apply(lambda ser: 
        [params_l[ser.name][val] for val in ser.values]) 


# Run with and without POISSONIFY
for POISSONIFY in [False, True]:
    res_l = []
    for nsim, sim_params in param_vals_df.iterrows():
        for nrep in range(nreps):
            # Set b1 
            b1 = np.sqrt(1 - sim_params['b0'] ** 2)
            
            # Run the simulation
            s = Sim.SimData(real_data_l=real_data_l, aov_typ=2, b1=b1, 
                poissonify=POISSONIFY, **sim_params)
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
    n_session2achieved_r = joined_df.reset_index().set_index(['n_session', 'nsim']).mean(
        axis=0, level=0)['achieved_r']

    # Mean and std of ESS, sorted by b0
    ess_x0_m = joined_df.pivot_table(rows='b0', cols='n_session', values='ess_x0').sort()
    ess_x0_std = joined_df.pivot_table(rows='b0', cols='n_session', values='ess_x0_std').sort()

    for n_session in ess_x0_m.columns:
        f, ax = plt.subplots(1, 1, figsize=(3,3))

        # Get means and stds for this session
        sess_ess_m = ess_x0_m[n_session]
        sess_ess_std = ess_x0_std[n_session]
        xvals = (params_l['b0'][sess_ess_m.index]) ** 2
        
        # Plot error bars
        ax.plot([0, 1], [0, 1], 'k-')
        ax.errorbar(x=xvals, y=sess_ess_m, yerr=sess_ess_std, color='r', 
            capsize=0, lw=0, elinewidth=1)
        ax.plot(xvals, sess_ess_m, 'ro', mec='r', mfc='none', lw=0)

        # Pretty
        ax.axis('scaled')
        ax.set_xlim((-.1, 1.1))
        ax.set_ylim((-.1, 1.1))
        ax.set_xticks((0, .25, .5, .75, 1))
        ax.set_yticks((0, .25, .5, .75, 1))
        #~ ax.set_title('session %d r=%0.3f' % (
            #~ n_session, n_session2achieved_r[n_session]))
        ax.set_xlabel('true FEV of head angle')
        ax.set_ylabel('estimated FEV of head angle')

        f.savefig('simulation_real_angles_sess%d_%s.svg' % (
            n_session, 'poissonified' if POISSONIFY else 'continous'))

plt.show()