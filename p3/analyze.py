import matplotlib.pyplot as plt

N = 20
X = [2048, 16384, 65536]
D = [16, 24, 32]
TYPES = ['seq', 'thrust', 'cuda', 'shmem']
FMTS = ['-ro', '-g^', '-bs', '-mD']

times = {typ: {} for typ in TYPES}

for typ in TYPES:
    for (x, d) in zip(X, D):
        with open(f"data/{typ}/n{x}-d{d}-{typ}.data", "r") as fin:
            times[typ][x] = sum(int(t) for t in fin) / N

plt.xlabel('points')
plt.ylabel('speedup')
for (typ, fmt) in zip(TYPES, FMTS):
    plt.plot(X, [times['seq'][x]/times[typ][x] for x in X], fmt, label=typ)
plt.legend(loc='center right')
plt.show()
#plt.savefig('data/plot.png', format='png', dpi=255)
#plt.clf()

