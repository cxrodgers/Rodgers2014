# Typical / strong exemplars
from ns5_process import myutils, LBPB
import kkpandas, kkpandas.kkrs
import pandas, os.path
import numpy as np
import my, my.dataload, my.plot
import matplotlib.pyplot as plt

my.plot.font_embed()
my.plot.publication_defaults()

unit_db = my.dataload.getstarted()['unit_db']

# Binning params
bins = np.linspace(-.25, .5, 151)
bincenters = bins[:-1] + np.diff(bins) / 2.
binwidth = np.diff(bins).mean()

# Exemplars
exemplar1 = 'CR17B_110731_001_behaving-406' # evoked response near median
exemplar2 = 'CR21A_120503_004_behaving-604' # median sustained responder

# Plot exemplars
for nax, ulabel in enumerate([exemplar1, exemplar2]):
    dfolded = my.dataload.ulabel2dfolded(ulabel,
        trial_picker_kwargs='all', 
        folding_kwargs={'dstart': -.25, 'dstop': .5})
    all_folded = np.sum(dfolded.values())
    binned = kkpandas.Binned.from_folded(all_folded, bins=bins)
    
    t1, t2 = unit_db.ix[ulabel][['audresp_t1', 'audresp_t2']]
    f, ax = plt.subplots(figsize=(3,3))
    bincenters = binned.t
    ax.plot(bincenters, binned.rate_in('Hz'), 'k')
    
    # Fill between the onset window
    n1 = np.argmin(np.abs(bincenters - t1))
    n2 = np.argmin(np.abs(bincenters - t2)) + 1    
    # Prepend and postpend a halfbin because the onset times are
    # halfway between the bins
    x = np.concatenate([
        [bincenters[n1-1:n1+1].mean()],
        bincenters[n1:n2],
        [bincenters[n2-1:n2+1].mean()],])
    y22 = binned.rate_in('Hz')[0].values
    y2 = np.concatenate([
        [y22[n1-1:n1+1].mean()],
        y22[n1:n2],
        [y22[n2-1:n2+1].mean()],])    
    ax.fill_between(x=x, y1=0, y2=y2, color='gray', alpha=0.5)
    
    if ulabel == 'CR17B_110731_001_behaving-406':
        ax.set_ylim((0, 25))
    
    # epochs
    barpos = ax.get_ylim()[1] * .92
    ax.plot([0, .25], [barpos, barpos], lw=2, color='lime', solid_capstyle='butt')
    ax.plot([-.15, 0], [barpos, barpos], lw=2, color='purple', solid_capstyle='butt')
    #ax.plot([-.25, -.05], [barpos, barpos], lw=2, color='purple', ls='steps--', solid_capstyle='butt')
    ax.text(.125, barpos*1.03, 'stimulus', color='lime', ha='center', va='bottom')
    ax.text(-.075, barpos*1.03, 'hold', color='purple', ha='center', va='bottom')
    
    # pretty
    ax.set_xlim(-.15, .35)
    ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_xticks((-.1, 0, .1, .2, .3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(right=False, top=False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spike rate (Hz)')

    #f.tight_layout()
    f.patch.set_visible(False)
    plt.show()
    
    f.savefig('typical_psths_%s.pdf' % ulabel)
plt.show()
