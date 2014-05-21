# Plot the block-folded hit rate averaged over all sessions

import numpy as np, os.path
import pandas
import kkpandas
from ns5_process import myutils, LBPB
import my, my.dataload, my.plot
import scipy.signal
import matplotlib.pyplot as plt

my.misc.no_warn_rs()
my.plot.font_embed()
my.plot.publication_defaults()


trials_info_dd = my.misc.pickle_load('trials_info_dd')

# Load hooks to sessions
gets = my.dataload.getstarted()



# Consistent figure sizing
figsize = (7, 3)
half_figsize = (3.5, 3)
bbox_to_anchor = None
subplots_adjust_kwargs = {'bottom':.2, 'right':.95}
half_subplots_adjust_kwargs = {'left':.2, 'bottom':.25, 'right':.95, 'top':.95}

# Plotting kwargs
BLACK_SQUARE = {'marker': 's', 'mfc': 'none', 'ms': 5, 'mec': 'k', 'mew': 1}

# Plotting kwargs by outcome
outcome_kw = {
    'hit': {'color':'k'},
    'interference': {'color': 'orange', 'marker': None},
    }

# How to more finely parse the errors
outcome_column = 'outcome2'
outcome_list = ['hit', 'interference']

plots_to_make = [
    'blolded average',
    'non-blolded average'
    ]


# Plotting subroutine
def plot_outcomes(outcomes_df, cols=None, level=0, ax=None, double_up=True,
    black_squares_on_hits=False):
    if ax is None:
        f, ax = plt.subplots()
    
    # Optionally grab subset of columns
    if cols:
        outcomes_df = outcomes_df.reindex_axis(cols, axis=1, level=level)
    
    # Plot each outcome over trial
    for outcome in outcome_list:
        ax.plot(outcomes_df.index, outcomes_df[outcome], label=outcome,
            **outcome_kw[outcome])
    
    # hits
    if black_squares_on_hits:
        kw = outcome_kw['hit'].copy()
        kw.update(BLACK_SQUARE)
        ax.plot(outcomes_df.index, outcomes_df['hit'], **kw)
    
    # Optionally double up
    if double_up:
        for outcome in outcome_list:
            # Plot shifted and the connection
            ax.plot(outcomes_df.index+160, outcomes_df[outcome], **outcome_kw[outcome])
            ax.plot([160, 161], outcomes_df[outcome][[160, 1]], **outcome_kw[outcome])

        if black_squares_on_hits:
            kw = outcome_kw['hit'].copy()
            kw.update(BLACK_SQUARE)
            ax.plot(outcomes_df.index+160, outcomes_df['hit'], **kw)
            ax.plot([160, 161], outcomes_df['hit'][[160, 1]], **kw)
        
        # X ax
        ax.set_xlim((1, 200))
        ax.set_xticks((80, 160))#, 240, 320))
    else:
        ax.set_xlim((1, 160))
        ax.set_xticks((80, 160))

    # Vertical divisions between blocks
    for tnum in range(0, 161 if double_up else 641, 80):
        # Plot the line halfway between the border trials
        l1, = ax.plot([tnum+.5, tnum+.5], [0, 100], 'k-', lw=1)
        
        # Shading in cue
        color = 'SkyBlue' if np.mod(tnum, 160) == 0 else 'pink'
        ax.fill_betweenx(y=[0, 100], x1=tnum+.5, x2=tnum+20.5,
            color=color, edgecolor=color)    

    # Make room for double x-axis
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.set_yticklabels((0, 20, 40, 60, 80, 100))
    ax.set_yticklabels((0, .20, .40, .60, .80, '1.0'))

    # Ylabel
    ax.set_ylabel('Probability')
    
    return ax

# Params
drop_first_block = True
average_by = 'block'

# Get TRIALS_INFO from each and block-fold it
blolded_outcomes_l, trials_info_l = [], []

# Figure out which sessions to run on
sessions_to_include = gets['session_list']

## Run each session
# Run each session
for session_name in sessions_to_include:
    # skip catch trial session
    if session_name == 'CR20B_120613_001_behaving':
        continue

    # Load
    ratname = kkpandas.kkrs.ulabel2ratname(session_name)
    trials_info = trials_info_dd[ratname][session_name]
    
    # Drop the forced trials
    trials_info = trials_info.ix[trials_info.nonrandom == 0]

    # Identify trials with a GO distracter
    trials_info['distracter_means_go'] = 0
    trials_info['distracter_means_go'][my.pick(trials_info,
        block=2, stim_name=('le_lo_lc', 'ri_lo_lc'))] = 1
    trials_info['distracter_means_go'][my.pick(trials_info,
        block=4, stim_name=['le_lo_pc', 'le_hi_pc'])] = 1            
    
    # Categorize errors more finely than usual
    trials_info['outcome2'] = trials_info['outcome'].copy()
    trials_info['outcome2'][my.pick(trials_info, 
        outcome='wrong_port', distracter_means_go=1)] = 'interference'
    
    # Store the loaded trials info (for later calculation of non-blolded)
    trials_info_l.append(trials_info.copy())
    
    # Optionally drop the first block for blolding purpose
    if drop_first_block:
        trials_info = trials_info.ix[trials_info.index > 20]
    
    # Mod
    trials_info['modtrial'] = np.mod(np.asarray(trials_info.index) - 1, 160) + 1

    # Group by modtrial (going from 1 - 160)
    gobj = trials_info.groupby('modtrial')
    
    # This gets the block for each modtrial (blocks 1-4), which should
    # be unique if the modtrial was done correctly
    blolded_outcomes = gobj[['block']].agg(myutils.unique_or_error)
    
    # This counts the number of included trials in each modtrial
    blolded_outcomes['included_trials'] = gobj[outcome_column].apply(len)
    
    # Count each outcome
    for outcome in outcome_list:
        blolded_outcomes[outcome] = gobj[outcome_column].apply(
            lambda ser: (ser == outcome).mean())

    # Store
    blolded_outcomes_l.append(blolded_outcomes)

# Concatenate into a big df
blolded_outcomes = pandas.concat(
    blolded_outcomes_l, axis=1,
    keys=sessions_to_include, verify_integrity=True)
nonblolded_outcomes = pandas.concat(
    [trials_info[outcome_column] for trials_info in trials_info_l], axis=1,
    keys=sessions_to_include, verify_integrity=True)


## Do the blolding
# Average everything together
if average_by == 'block':
    # Average all blocks together
    boswp = blolded_outcomes.swaplevel(0, 1, axis=1)
    bit = boswp['included_trials']
    bo_avg = pandas.concat(
        [(boswp[outcome] * bit).sum(axis=1) / bit.sum(axis=1)
            for outcome in outcome_list],
        keys=outcome_list, axis=1, verify_integrity=True)
elif average_by == 'session':
    # Average first by block within session, then across sessions
    bo_avg = blolded_outcomes.mean(axis=1, level=1)

# Calculate average across all sessions WITHOUT block-folding
# First calculate the number of included trials
divide_by = (~pandas.isnull(nonblolded_outcomes)).sum(1).astype(np.float)
# Now count each outcome and divide by number of included trials
nbo_avg = pandas.concat(
    [(nonblolded_outcomes == outcome).sum(1) / divide_by 
    for outcome in outcome_list],
    keys=outcome_list, axis=1, verify_integrity=True)

# Convert to percent
bo_avg = bo_avg * 100
nbo_avg = nbo_avg * 100


if 'blolded average' in plots_to_make:
    # Plot blolded average
    f, ax = plt.subplots(figsize=figsize)
    f.subplots_adjust(**subplots_adjust_kwargs)
    plot_outcomes(bo_avg, ax=ax, black_squares_on_hits=True)

    # trial labels
    ax.set_xticks((20, 80, 100, 160, 180))
    ax.tick_params(bottom=False, top=False)

    # block labels
    ax.text(50, -18, 'localization', ha='center', va='center')
    ax.text(130, -18, 'pitch discrimination', ha='center', va='center')
    ax.text(10, -18, 'cue', ha='center', va='center')
    ax.text(90, -18, 'cue', ha='center', va='center')
    ax.text(170, -18, 'cue', ha='center', va='center')

    # double-label x-axis
    ax.text(-4, -5.5, 'trial #', ha='right', va='center')
    ax.text(-4, -18, 'block', ha='right', va='center')

    # Legendary
    lkwargs = {'ha': 'center', 'va': 'bottom', 'size': 12}
    ax.text(30, 101, 'hit', color='k', **lkwargs)
    ax.text(70, 101, 'interference', color='orange', **lkwargs)

    f.savefig('blolded_average.svg')





if 'non-blolded average' in plots_to_make:
    # Plot non-blolded average
    f, ax = plt.subplots(figsize=figsize)
    f.subplots_adjust(**subplots_adjust_kwargs)
    for outcome in outcome_list:
        ax.plot(nbo_avg.index, nbo_avg[outcome], label=outcome, **outcome_kw[outcome])
    ax.set_xticks(range(0, 700, 80))
    ax.tick_params(bottom=False, top=False)
    ax.set_xlim((0, 700))


    # Vertical divisions between blocks
    for tnum in range(0, 641, 80):
        # Plot the line halfway between the border trials
        l1, = ax.plot([tnum+.5, tnum+.5], [0, 100], 'k-', lw=1)#.5)
        
        # Shading in cue
        color = 'SkyBlue' if np.mod(tnum, 160) == 0 else 'pink'
        ax.fill_betweenx(y=[0, 100], x1=tnum+.5, x2=tnum+20.5,
            color=color, edgecolor=color)   


    # Make room for double x-axis
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ax.set_yticklabels((0, .20, .40, .60, .80, '1.0'))
    ax.set_ylabel('Probability')


    # block labels
    for xt in range(40, 700, 80):
        if np.mod(xt, 160) == 40:
            ax.text(xt, -18, 'local.', ha='center', va='center')
        else:
            ax.text(xt, -18, 'pitch\ndisc.', ha='center', va='center')

    # double-label x-axis
    ax.text(-20, -5.5, 'trial #', ha='right', va='center')
    ax.text(-20, -18, 'block', ha='right', va='center')


    # Legendary
    lkwargs = {'ha': 'center', 'va': 'bottom', 'size': 12}
    ax.text(25, 101, 'hit', color='k', **lkwargs)
    ax.text(150, 101, 'interference', color='orange', **lkwargs)

    #f.patch.set_visible(False)
    f.savefig('nonblolded_average.svg')

    plt.show()
