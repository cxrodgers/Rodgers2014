import pandas
import my, my.dataload, my.plot, numpy as np
import matplotlib.pyplot as plt
import os.path

my.plot.font_embed()
my.plot.publication_defaults()

hold_results = pandas.load('hold_results')

# Rat names
gets = my.dataload.getstarted()


rec_l = []
for ratname, subdf1 in hold_results.groupby('ratname'):
    for region in ['A1', 'PFC']:
        subdf = my.pick_rows(subdf1, region=region)

        if len(region) > 0:
            nL = len(subdf[(subdf.p_adj < .05) & (subdf.mLB > subdf.mPB)])
            nP = len(subdf[(subdf.p_adj < .05) & (subdf.mLB < subdf.mPB)])
            nN = len(subdf[(subdf.p_adj >= .05)])
        else:
            nL, nP, nN = 0, 0, 0
        rec_l.append((ratname, region, nL, nP, nN))

res = pandas.DataFrame.from_records(rec_l, 
    columns=('ratname', 'region', 'nL', 'nP', 'nN')).set_index(
    ['region', 'ratname'])

divby = res.sum(1).astype(np.float)
divby[divby < 8] = 0.
res2 = 100 * res.divide(divby, 'index')
res2 = res2.applymap(lambda x: 0 if np.isnan(x) else x)
res2 = res2.applymap(lambda x: 0 if np.isinf(x) else x)

#~ ratnames = sorted(res.index.levels[1])
ratnames = gets['ratname2num'].index
ratnums = gets['ratname2num'].values

f1, axa1 = plt.subplots(2, 1, figsize=(6.5, 4)) # PFC
f2, axa2 = plt.subplots(2, 1, figsize=(6.5, 4)) # A1


for axa in [axa1, axa2]:
    axa[0].set_ylabel('# neurons')
    axa[1].set_ylabel('% of neurons')
    axa[1].set_ylim((0, 100))

region = 'PFC'
for df, ax in zip([res, res2], axa1):
    x = np.array(list(range(6)))
    ax.bar(left=x, height=df.ix[region]['nL'][ratnames], color='b', width=.3)
    ax.bar(left=x+.3, height=df.ix[region]['nP'][ratnames], color='r', width=.3)
    ax.bar(left=x+.6, height=df.ix[region]['nN'][ratnames], color='k', width=.3)
    
    ax.set_xticks(x+.45)
    ax.set_xlim((0, 6))
    ax.set_xticklabels(ratnums)
#f1.suptitle('PFC')
#f1.patch.set_visible(False)


region = 'A1'
for df, ax in zip([res, res2], axa2):
    x = np.array(list(range(6)))
    ax.bar(left=x, height=df.ix[region]['nL'][ratnames], color='b', width=.3)
    ax.bar(left=x+.3, height=df.ix[region]['nP'][ratnames], color='r', width=.3)
    ax.bar(left=x+.6, height=df.ix[region]['nN'][ratnames], color='k', width=.3)
    
    ax.set_xticks(x+.45)
    ax.set_xlim((0, 6))
    ax.set_xticklabels(ratnums)
#f2.suptitle('A1')
#f2.patch.set_visible(False)

f1.savefig('PFC_across_rats.svg')
f2.savefig('A1_across_rats.svg')

plt.show()
