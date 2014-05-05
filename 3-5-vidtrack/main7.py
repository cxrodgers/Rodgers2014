# For each real neuron, fit a model based entirely on head angle
# Then run ANOVA ~ HeadAngle * Block
# Do results look like actual results?
#
# There's one that should probably be dumped from the analysis:
# CR24A_121022_001_behaving-610
# This one fails the F-test in the real data, because such a small dependence
# on head angle and a marginally signif effect of Block
# Then, in the fake analysis, it has a low F-value of 1.6 (real value 2.0)
# and basically gives random results every time.


import numpy as np
import my, my.plot
import pandas
import itertools
import matplotlib.pyplot as plt
import kkpandas.kkrs
import os.path

my.plot.font_embed()
my.plot.publication_defaults()


# Number of times to draw responses from model, to estimate true mean
# 1000 times takes a long time to process, but is necessary for convergence
N_FAKES = 1000

# Get the actual angles and blocks from analyzed sessions
full_data = pandas.load('full_data')

# Fit the model for each unit
ulabels = full_data.index.levels[0]
#ulabels = ['CR24A_121022_001_behaving-610']
rec_l = []
for ulabel in ulabels:
    # Full data -- angl, block, counts
    fdu = full_data.ix[ulabel]

    # First insert the sqrt_counts to fit
    fdu['sqrt_counts'] = np.sqrt(fdu['counts'])

    # Fit a new resp
    # sqrt(counts) ~ B * angl + noise
    B = np.polyfit(fdu.angl.values, fdu.sqrt_counts.values, deg=1)[0]
    
    # Now fit noise to the residuals
    resids = fdu.sqrt_counts.values - B * fdu.angl.values
    noise_m = np.mean(resids)
    noise_std = np.std(resids)
    
    # Fit a bunch of fake anovas
    fake_l = []
    for n_fake in range(N_FAKES):
        # Finally draw fictitious responses from this model
        fdu['faked_sqrt_counts'] = B * fdu.angl.values + \
            np.random.standard_normal(fdu.angl.values.shape) * noise_std \
            + noise_m        
        
        # Run it
        fake_aovres = my.stats.anova(fdu, 'faked_sqrt_counts ~ angl + block',
            typ=2)
        
        # Store results
        fakerec = fake_aovres['ess'].to_dict().copy()
        fake_l.append(fakerec)
    fake_df = pandas.DataFrame.from_records(fake_l)

    # Fit the real anova
    real_aovres = my.stats.anova(fdu, 'sqrt_counts ~ angl + block', typ=2)

    # Check that EV (as returned by ANOVA) matches EV (from factors)
    lm = real_aovres['lm']
    pred_angl = fdu.angl * lm.params['angl']
    pred_block = (fdu.block == 'PB').values.astype(np.int) * \
        lm.params['block[T.PB]']
    assert np.allclose(
        real_aovres['ess']['ess_block'] / real_aovres['ess']['ess_angl'],
        np.var(pred_block) / np.var(pred_angl))

    # Deparse and store
    rec = {'ulabel': ulabel, 
        'region': 'A1' if kkpandas.kkrs.is_auditory(ulabel) else 'PFC',
        'real_ess_angl': real_aovres['ess']['ess_angl'],
        'real_ess_block': real_aovres['ess']['ess_block'],
        'real_pF': real_aovres['lm'].f_pvalue,
        }
    
    # Update with faked results
    for colname in fake_df:
        rec['fake_mean_%s' % colname] = fake_df[colname].mean()
        rec['fake_std_%s' % colname] = fake_df[colname].std()            

    # Store
    rec_l.append(rec)

# DataFrame results
aov_df = pandas.DataFrame.from_records(rec_l).set_index('ulabel')

# Sort by real_aov_angl
aov_df = aov_df.sort('real_ess_angl')
aov_df.save('aov_df')
