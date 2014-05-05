# Histograms of audresp latencies and strength
from ns5_process import myutils, LBPB
import kkpandas, kkpandas.kkrs
import pandas, os.path
import numpy as np, scipy.stats
import my, my.dataload, my.plot
import matplotlib.pyplot as plt

my.plot.font_embed()
my.plot.publication_defaults()


# Load results just measured
unit_db = my.dataload.getstarted()['unit_db']
counts = pandas.load('counts')

# Put into summary
summary = counts[['dt']]
summary['mEv'] = counts['evok'].apply(np.mean)
summary['mSp'] = counts['pre'].apply(np.mean)
summary['mEvHz'] = summary['mEv'] / summary['dt']
summary['mSpHz'] = summary['mSp'] / .050
summary['diffHz'] = summary['mEvHz'] - summary['mSpHz']
summary['diffspks'] = summary['diffHz'] * summary['dt']
summary['ratio'] = summary['mEvHz'] / summary['mSpHz']
summary['region'] = unit_db['region'][summary.index]
summary['latency'] = 1000*unit_db[['audresp_t1', 'audresp_t2']].ix[
    summary.index].mean(1)
assert unit_db['include'][summary.index].all()

# Comparison of prevalence of audresp cells across regions
sdump = ''
A1_cells = my.pick_rows(unit_db, region='A1', include=True)
PFC_cells = my.pick_rows(unit_db, region='PFC', include=True)
n_A1_cells, n_audresp_A1_cells = map(len,
    [A1_cells, my.pick(A1_cells, audresp=['good', 'weak', 'sustained'])])
n_PFC_cells, n_audresp_PFC_cells = map(len,
    [PFC_cells, my.pick(PFC_cells, audresp=['good', 'weak', 'sustained'])])
sdump +=  "* A1: %d/%d\n" % (n_audresp_A1_cells, n_A1_cells)
sdump +=  "* PFC: %d/%d\n" % (n_audresp_PFC_cells, n_PFC_cells)
sdump +=  "* p=%0.4f, Fisher's Exact Test\n" % scipy.stats.fisher_exact([
    [n_audresp_A1_cells, n_A1_cells - n_audresp_A1_cells],
    [n_audresp_PFC_cells, n_PFC_cells - n_audresp_PFC_cells],])[1]
with file('stat__comparison_of_prevalence_of_audresp_cells_across_regions', 'w') as fi:
    fi.write(sdump)
print sdump

# Histogram of strengths
f, axa = plt.subplots(1, 3, figsize=(9, 3))
f.subplots_adjust(wspace=.5, bottom=.25, left=.05, right=.95)
ax = axa[0]
v1 = my.pick_rows(summary, region='A1')['diffspks']
v2 = my.pick_rows(summary, region='PFC')['diffspks']
with file('stat__audresp_response_parameters_across_regions', 'w') as fi:
    fi.write("Response comparison. A1: n=%d; PFC: n=%d\n" % (len(v1), len(v2)))
    utest_res = my.stats.r_utest(v1, v2, fix_float=True)
    fi.write("strength (spikes)\n")
    fi.write("* A1 median: %f; PFC median: %f; p-values: %0.4f\n" % (
        v1.median(), v2.median(), utest_res['p']))
    ax.hist([v1, v2],
        color=['blue', 'orange'], rwidth=0.8,
        bins=np.arange(0, 1.5, .15),
        )
    ax.set_xlabel('evoked increase\n(# spikes)')
    ax.set_xticks((0, .5, 1.0, 1.5))

    # Strengths as % of spont
    ax = axa[1]
    v1 = my.pick_rows(summary, region='A1')['ratio']
    v2 = my.pick_rows(summary, region='PFC')['ratio']
    utest_res = my.stats.r_utest(v1, v2, fix_float=True)
    fi.write("strength (ratio)\n")
    fi.write("* A1 median: %f; PFC median: %f; p-values: %0.4f\n" % (
        v1.median(), v2.median(), utest_res['p']))
    ax.hist([100*v1, 100*v2],
        color=['blue', 'orange'], rwidth=0.8,
        bins=np.arange(100, 1000, 100),
        )
    ax.set_xlabel('evoked\n(% of spont)')
    ax.set_xticks((100, 300, 500, 700, 900))


    # Plot the latencies
    ax = axa[2]
    v1 = my.pick_rows(summary, region='A1')['latency']
    v2 = my.pick_rows(summary, region='PFC')['latency']
    utest_res = my.stats.r_utest(v1, v2, fix_float=True)
    fi.write('latency\n')
    fi.write("* A1 median: %f; PFC median: %f; p-values: %0.4f\n" % (
        v1.median(), v2.median(), utest_res['p']))
    ax.hist([v1, v2], bins=1000*np.arange(0.001, 0.035, 0.004),
        color=['blue', 'orange'])
    ax.set_xlabel('latency (ms)')

    ax.set_xticks((0, 10, 20, 30))


# Pretty
for ax in f.axes:
    my.plot.despine(ax)
    ax.set_ylabel('# of neurons')
    ax.text(.85, .9, 'A1', color='blue', size=14, ha='center', transform=ax.transAxes)
    ax.text(.85, .8, 'PFC', color='orange', size=14, ha='center', transform=ax.transAxes)
#~ f.patch.set_visible(False)
f.savefig('misc_evoked_stats.svg')

plt.show()